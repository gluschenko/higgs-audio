"""Export HiggsAudio forward pass to ONNX.

This script exports the model's multimodal ``forward()`` graph rather than the
custom autoregressive ``generate()`` loop. The exported graph is therefore most
useful as a building block for custom runtimes that own the decode loop.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import click
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise click.BadParameter(f"Unsupported dtype '{value}'. Choose from: {', '.join(sorted(mapping))}.")
    return mapping[value]


def _pick_text_token_id(model) -> int:
    reserved_ids = {
        model.config.pad_token_id,
        model.config.audio_in_token_idx,
        model.config.audio_out_token_idx,
    }
    vocab_size = model.config.text_config.vocab_size
    for candidate in (1, 42, 128, 256, 512, 1024):
        if 0 <= candidate < vocab_size and candidate not in reserved_ids:
            return candidate
    return 0


def _pick_higgs_audio_v2_text_token_id(model) -> int:
    reserved_ids = {
        model.config.pad_token_id,
        model.config.audio_token_id,
        model.config.audio_delay_token_id,
    }
    for candidate in (1, 42, 128, 256, 512, 1024):
        if 0 <= candidate < model.config.vocab_size and candidate not in reserved_ids:
            return candidate
    return 0


class HiggsAudioOnnxWrapper(torch.nn.Module):
    """Small wrapper that exposes only tensor inputs / outputs for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        audio_feature_attention_mask: torch.Tensor,
        audio_in_ids: torch.Tensor,
        audio_in_ids_start: torch.Tensor,
        audio_out_ids: torch.Tensor,
        audio_out_ids_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            output_audio_hidden_states=False,
            return_dict=True,
        )

        audio_logits = outputs.audio_logits
        if audio_logits is None:
            audio_logits = outputs.logits.new_zeros(
                (0, self.model.config.audio_num_codebooks, self.model.config.audio_codebook_size)
            )

        return (
            outputs.logits,
            audio_logits,
            outputs.expanded_input_ids,
            outputs.attention_mask.to(torch.int64),
            outputs.audio_out_mask.to(torch.int64),
        )


class HiggsAudioV2OnnxWrapper(torch.nn.Module):
    """Wrapper for Transformers' native HiggsAudioV2ForConditionalGeneration."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_input_ids: torch.Tensor,
        audio_input_ids_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask.bool(),
            use_cache=False,
            logits_to_keep=0,
        )
        logits = outputs.logits
        return logits.reshape(
            logits.shape[0],
            logits.shape[1],
            self.model.config.num_codebooks,
            self.model.config.codebook_size,
        )


def _build_dummy_inputs(
    model,
    batch_size: int,
    text_seq_len: int,
    num_audio_inputs: int,
    audio_feature_seq_len: int,
    audio_in_tokens_per_placeholder: int,
    num_audio_outputs: int,
    audio_out_tokens_per_placeholder: int,
    device: torch.device,
):
    if batch_size <= 0:
        raise click.BadParameter("dummy_batch_size must be greater than 0.")
    if text_seq_len <= 0:
        raise click.BadParameter("dummy_text_seq_len must be greater than 0.")
    if num_audio_inputs < 0 or num_audio_outputs < 0:
        raise click.BadParameter("dummy audio placeholder counts cannot be negative.")
    if audio_feature_seq_len <= 0:
        raise click.BadParameter("dummy_audio_feature_seq_len must be greater than 0.")
    if audio_in_tokens_per_placeholder < 0 or audio_out_tokens_per_placeholder < 0:
        raise click.BadParameter("dummy audio token counts cannot be negative.")
    if num_audio_inputs > 0 and audio_in_tokens_per_placeholder <= 0:
        raise click.BadParameter("dummy_audio_in_tokens must be greater than 0 when audio inputs are enabled.")
    if num_audio_outputs > 0 and audio_out_tokens_per_placeholder <= 0:
        raise click.BadParameter("dummy_audio_out_tokens must be greater than 0 when audio outputs are enabled.")

    min_required_text_tokens = num_audio_inputs + num_audio_outputs + 1
    if text_seq_len < min_required_text_tokens:
        raise click.BadParameter(
            "dummy_text_seq_len must be at least "
            f"{min_required_text_tokens} to host all placeholder tokens and one text token."
        )

    text_token_id = _pick_text_token_id(model)
    input_ids = torch.full((batch_size, text_seq_len), text_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, text_seq_len), dtype=torch.long, device=device)

    next_slot = 0
    for idx in range(num_audio_inputs):
        batch_idx = idx % batch_size
        input_ids[batch_idx, next_slot] = model.config.audio_in_token_idx
        next_slot += 1

    for idx in range(num_audio_outputs):
        batch_idx = idx % batch_size
        input_ids[batch_idx, next_slot] = model.config.audio_out_token_idx
        next_slot += 1

    num_mel_bins = model.config.audio_encoder_config.num_mel_bins
    model_dtype = next(model.parameters()).dtype
    audio_features = torch.randn(
        num_audio_inputs,
        num_mel_bins,
        audio_feature_seq_len,
        dtype=model_dtype,
        device=device,
    )
    audio_feature_attention_mask = torch.ones(
        (num_audio_inputs, audio_feature_seq_len),
        dtype=torch.long,
        device=device,
    )

    audio_in_total_tokens = num_audio_inputs * audio_in_tokens_per_placeholder
    audio_in_ids = torch.randint(
        low=0,
        high=model.config.audio_codebook_size,
        size=(model.config.audio_num_codebooks, audio_in_total_tokens),
        dtype=torch.long,
        device=device,
    )
    audio_in_ids_start = torch.arange(
        0,
        audio_in_total_tokens,
        audio_in_tokens_per_placeholder,
        dtype=torch.long,
        device=device,
    )

    audio_out_total_tokens = num_audio_outputs * audio_out_tokens_per_placeholder
    audio_out_ids = torch.randint(
        low=0,
        high=model.config.audio_codebook_size,
        size=(model.config.audio_num_codebooks, audio_out_total_tokens),
        dtype=torch.long,
        device=device,
    )
    audio_out_ids_start = torch.arange(
        0,
        audio_out_total_tokens,
        audio_out_tokens_per_placeholder,
        dtype=torch.long,
        device=device,
    )

    return (
        input_ids,
        attention_mask,
        audio_features,
        audio_feature_attention_mask,
        audio_in_ids,
        audio_in_ids_start,
        audio_out_ids,
        audio_out_ids_start,
    )


def _build_higgs_audio_v2_dummy_inputs(
    model,
    batch_size: int,
    text_seq_len: int,
    audio_frames: int,
    device: torch.device,
):
    if batch_size <= 0:
        raise click.BadParameter("dummy_batch_size must be greater than 0.")
    if text_seq_len <= 0:
        raise click.BadParameter("dummy_text_seq_len must be greater than 0.")
    if audio_frames <= 0:
        raise click.BadParameter("dummy_audio_out_tokens must be greater than 0 for HiggsAudioV2 export.")
    if text_seq_len < audio_frames + 1:
        raise click.BadParameter("dummy_text_seq_len must be at least dummy_audio_out_tokens + 1.")

    text_token_id = _pick_higgs_audio_v2_text_token_id(model)
    input_ids = torch.full((batch_size, text_seq_len), text_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, text_seq_len), dtype=torch.bool, device=device)
    input_ids[:, :audio_frames] = model.config.audio_token_id

    audio_input_ids = torch.randint(
        low=0,
        high=model.config.codebook_size,
        size=(batch_size, audio_frames, model.config.num_codebooks),
        dtype=torch.long,
        device=device,
    )
    audio_input_ids_mask = torch.ones((batch_size, audio_frames), dtype=torch.bool, device=device)

    return input_ids, attention_mask, audio_input_ids, audio_input_ids_mask


def _export_onnx(
    wrapper: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    opset: int,
    use_external_data: bool,
    use_dynamo: bool,
) -> None:
    input_names = [
        "input_ids",
        "attention_mask",
        "audio_features",
        "audio_feature_attention_mask",
        "audio_in_ids",
        "audio_in_ids_start",
        "audio_out_ids",
        "audio_out_ids_start",
    ]
    output_names = [
        "logits",
        "audio_logits",
        "expanded_input_ids",
        "expanded_attention_mask",
        "expanded_audio_out_mask",
    ]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "text_seq"},
        "attention_mask": {0: "batch", 1: "text_seq"},
        "audio_features": {0: "num_audio_inputs", 2: "audio_feature_seq"},
        "audio_feature_attention_mask": {0: "num_audio_inputs", 1: "audio_feature_seq"},
        "audio_in_ids": {1: "audio_in_tokens"},
        "audio_in_ids_start": {0: "num_audio_inputs"},
        "audio_out_ids": {1: "audio_out_tokens"},
        "audio_out_ids_start": {0: "num_audio_outputs"},
        "logits": {0: "batch", 1: "merged_seq"},
        "audio_logits": {0: "audio_out_tokens"},
        "expanded_input_ids": {0: "batch", 1: "merged_seq"},
        "expanded_attention_mask": {0: "batch", 1: "merged_seq"},
        "expanded_audio_out_mask": {0: "batch", 1: "merged_seq"},
    }

    export_kwargs = {
        "args": args,
        "f": output_path.as_posix(),
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": opset,
        "export_params": True,
        "do_constant_folding": True,
    }

    signature = inspect.signature(torch.onnx.export)
    if "external_data" in signature.parameters:
        export_kwargs["external_data"] = use_external_data
    elif "use_external_data_format" in signature.parameters:
        export_kwargs["use_external_data_format"] = use_external_data

    if "dynamo" in signature.parameters:
        export_kwargs["dynamo"] = use_dynamo

    torch.onnx.export(wrapper, **export_kwargs)


def _export_higgs_audio_v2_onnx(
    wrapper: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    opset: int,
    use_external_data: bool,
    use_dynamo: bool,
) -> None:
    export_kwargs = {
        "args": args,
        "f": output_path.as_posix(),
        "input_names": [
            "input_ids",
            "attention_mask",
            "audio_input_ids",
            "audio_input_ids_mask",
        ],
        "output_names": ["audio_logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "text_seq"},
            "attention_mask": {0: "batch", 1: "text_seq"},
            "audio_input_ids": {0: "batch", 1: "audio_frames"},
            "audio_input_ids_mask": {0: "batch", 1: "audio_frames"},
            "audio_logits": {0: "batch", 1: "text_seq"},
        },
        "opset_version": opset,
        "export_params": True,
        "do_constant_folding": True,
    }

    signature = inspect.signature(torch.onnx.export)
    if "external_data" in signature.parameters:
        export_kwargs["external_data"] = use_external_data
    elif "use_external_data_format" in signature.parameters:
        export_kwargs["use_external_data_format"] = use_external_data

    if "dynamo" in signature.parameters:
        export_kwargs["dynamo"] = use_dynamo

    torch.onnx.export(wrapper, **export_kwargs)


def _detect_model_type(model_path: str) -> str | None:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, local_files_only=Path(model_path).exists())
    except Exception:
        return None
    return getattr(config, "model_type", None)


@click.command()
@click.option(
    "--model-path",
    required=True,
    type=str,
    help="Local checkpoint directory or Hugging Face repo id.",
)
@click.option("--output-path", required=True, type=click.Path(path_type=Path))
@click.option("--device", default="cpu", show_default=True, help="Torch device used during export, e.g. cpu or cuda:0.")
@click.option(
    "--dtype",
    default="float32",
    show_default=True,
    help="Checkpoint dtype to load for export: float32, float16 or bfloat16.",
)
@click.option(
    "--attn-implementation",
    default="eager",
    show_default=True,
    type=click.Choice(["eager", "sdpa"], case_sensitive=False),
    help="Use an ONNX-friendly attention implementation during export.",
)
@click.option("--opset", default=17, show_default=True, type=int)
@click.option("--dummy-batch-size", default=1, show_default=True, type=int)
@click.option("--dummy-text-seq-len", default=16, show_default=True, type=int)
@click.option("--dummy-num-audio-inputs", default=1, show_default=True, type=int)
@click.option("--dummy-audio-feature-seq-len", default=300, show_default=True, type=int)
@click.option("--dummy-audio-in-tokens", default=8, show_default=True, type=int)
@click.option("--dummy-num-audio-outputs", default=1, show_default=True, type=int)
@click.option("--dummy-audio-out-tokens", default=8, show_default=True, type=int)
@click.option(
    "--external-data/--single-file",
    default=True,
    show_default=True,
    help="Store large weights in external data files instead of a single monolithic ONNX file.",
)
@click.option(
    "--dynamo/--no-dynamo",
    default=False,
    show_default=True,
    help="Use the newer torch.export-based ONNX exporter when supported by the installed PyTorch.",
)
def main(
    model_path: str,
    output_path: Path,
    device: str,
    dtype: str,
    attn_implementation: str,
    opset: int,
    dummy_batch_size: int,
    dummy_text_seq_len: int,
    dummy_num_audio_inputs: int,
    dummy_audio_feature_seq_len: int,
    dummy_audio_in_tokens: int,
    dummy_num_audio_outputs: int,
    dummy_audio_out_tokens: int,
    external_data: bool,
    dynamo: bool,
) -> None:
    """Export a HiggsAudio checkpoint to ONNX."""
    torch_dtype = _parse_dtype(dtype)
    if device == "cpu" and torch_dtype == torch.float16:
        raise click.BadParameter("float16 export on CPU is not supported reliably. Use float32/bfloat16 or export on CUDA.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    model_type = _detect_model_type(model_path)

    if model_type == "higgs_audio_v2":
        try:
            from transformers import HiggsAudioV2ForConditionalGeneration
        except ImportError as exc:
            raise click.ClickException(
                "The installed transformers package does not provide HiggsAudioV2ForConditionalGeneration. "
                "Use a newer Transformers build for model_type='higgs_audio_v2'."
            ) from exc

        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation.lower(),
            local_files_only=Path(model_path).exists(),
        )
        model = model.to(torch_device)
        model.eval()

        wrapper = HiggsAudioV2OnnxWrapper(model).to(torch_device).eval()
        dummy_inputs = _build_higgs_audio_v2_dummy_inputs(
            model=model,
            batch_size=dummy_batch_size,
            text_seq_len=dummy_text_seq_len,
            audio_frames=dummy_audio_out_tokens,
            device=torch_device,
        )

        with torch.inference_mode():
            _ = wrapper(*dummy_inputs)
            _export_higgs_audio_v2_onnx(
                wrapper=wrapper,
                args=dummy_inputs,
                output_path=output_path,
                opset=opset,
                use_external_data=external_data,
                use_dynamo=dynamo,
            )

        click.echo(f"Exported native HiggsAudioV2 ONNX graph to {output_path}")
        click.echo(
            "Note: this exports the forward pass only. The custom autoregressive generate loop is intentionally excluded."
        )
        return

    from boson_multimodal.model.higgs_audio import HiggsAudioModel

    model = HiggsAudioModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation.lower(),
        device_map=None,
    )
    model = model.to(torch_device)
    model.eval()

    wrapper = HiggsAudioOnnxWrapper(model).to(torch_device).eval()
    dummy_inputs = _build_dummy_inputs(
        model=model,
        batch_size=dummy_batch_size,
        text_seq_len=dummy_text_seq_len,
        num_audio_inputs=dummy_num_audio_inputs,
        audio_feature_seq_len=dummy_audio_feature_seq_len,
        audio_in_tokens_per_placeholder=dummy_audio_in_tokens,
        num_audio_outputs=dummy_num_audio_outputs,
        audio_out_tokens_per_placeholder=dummy_audio_out_tokens,
        device=torch_device,
    )

    with torch.inference_mode():
        _ = wrapper(*dummy_inputs)
        _export_onnx(
            wrapper=wrapper,
            args=dummy_inputs,
            output_path=output_path,
            opset=opset,
            use_external_data=external_data,
            use_dynamo=dynamo,
        )

    click.echo(f"Exported ONNX graph to {output_path}")
    click.echo(
        "Note: this exports HiggsAudio forward() only. The custom generate() loop, HF cache objects, "
        "and CUDA graph capture are intentionally excluded."
    )
    click.echo(
        "Tracing note: include every modality you need in the dummy export shapes, because ONNX tracing only "
        "preserves paths exercised during export."
    )


if __name__ == "__main__":
    main()
