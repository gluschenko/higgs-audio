"""Export the Higgs Audio V2 tokenizer to ONNX.

The Hugging Face tokenizer checkpoint uses ``HiggsAudioV2TokenizerModel`` from
Transformers. This script exports its encoder, decoder, or full reconstruction
forward graph.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import click
import torch
from huggingface_hub import hf_hub_download, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_tokenizer_model(model_id: str, device: torch.device, dtype: torch.dtype):
    try:
        from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel
    except ImportError as exc:
        raise click.ClickException(
            "The installed transformers package does not provide HiggsAudioV2TokenizerModel. "
            "Install a recent version, for example `pip install -U transformers`, or install "
            "Transformers from source if the class is not in your stable release yet."
        ) from exc

    if Path(model_id).exists():
        model_path = Path(model_id)
    else:
        model_path = Path(
            snapshot_download(
                model_id,
                allow_patterns=["config.json", "model.pth", "preprocessor_config.json"],
            )
        )

    config = HiggsAudioV2TokenizerConfig.from_pretrained(model_path)
    model = HiggsAudioV2TokenizerModel(config)

    state_path = model_path / "model.pth"
    if not state_path.exists():
        state_path = Path(hf_hub_download(model_id, filename="model.pth"))

    state_dict = torch.load(state_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        click.echo(
            "Loaded model.pth with non-strict state_dict. "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}",
            err=True,
        )

    model.to(dtype=dtype)
    model.to(device)
    model.eval()
    return model


class _UnusedSemanticModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("Semantic model is not available in decoder-only export.")


def _remove_weight_norms(module: torch.nn.Module) -> None:
    for child in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except (ValueError, AttributeError):
            pass


def _load_legacy_tokenizer(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    skip_semantic_model: bool,
):
    import boson_multimodal.audio_processing.higgs_audio_tokenizer as tokenizer_module

    if Path(model_id).exists():
        model_path = Path(model_id)
    else:
        model_path = Path(
            snapshot_download(
                model_id,
                allow_patterns=["model.pth"],
            )
        )

    original_from_pretrained = tokenizer_module.AutoModel.from_pretrained
    if skip_semantic_model:
        tokenizer_module.AutoModel.from_pretrained = lambda *args, **kwargs: _UnusedSemanticModel()
    try:
        model = tokenizer_module.HiggsAudioTokenizer(
            D=256,
            ratios=[8, 5, 4, 2, 3],
            sample_rate=24000,
            bins=1024,
            n_q=8,
            codebook_dim=64,
            semantic_sample_rate=16000,
            device=str(device),
        )
    finally:
        tokenizer_module.AutoModel.from_pretrained = original_from_pretrained

    state_path = model_path / "model.pth"
    if not state_path.exists():
        state_path = Path(hf_hub_download(model_id, filename="model.pth"))
    state_dict = torch.load(state_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        click.echo(
            "Loaded legacy model.pth with non-strict state_dict. "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}",
            err=True,
        )
    _remove_weight_norms(model)
    model.to(dtype=dtype)
    model.to(device)
    model.eval()
    return model


def _parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = value.lower()
    if key not in mapping:
        raise click.BadParameter(f"Unsupported dtype '{value}'. Choose from: {', '.join(sorted(mapping))}.")
    return mapping[key]


def _export(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
    external_data: bool,
    dynamo: bool,
) -> None:
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
        export_kwargs["external_data"] = external_data
    elif "use_external_data_format" in signature.parameters:
        export_kwargs["use_external_data_format"] = external_data
    if "dynamo" in signature.parameters:
        export_kwargs["dynamo"] = dynamo

    torch.onnx.export(module, **export_kwargs)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, bandwidth: float | None):
        super().__init__()
        self.model = model
        self.bandwidth = bandwidth

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model.encode(input_values, bandwidth=self.bandwidth, return_dict=True)
        return outputs.audio_codes


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        outputs = self.model.decode(audio_codes, return_dict=True)
        return outputs.audio_values


class LegacyDecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        vq_code = audio_codes.permute(1, 0, 2)
        quantized = self.model.quantizer.decode(vq_code).transpose(1, 2)
        quantized_acoustic = self.model.fc_post2(quantized).transpose(1, 2)
        return self.model.decoder_2(quantized_acoustic)


class ForwardWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, bandwidth: float | None):
        super().__init__()
        self.model = model
        self.bandwidth = bandwidth

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_values, bandwidth=self.bandwidth, return_dict=True)
        return outputs.audio_codes, outputs.audio_values


def _dummy_audio(model: torch.nn.Module, batch_size: int, num_samples: int, device: torch.device) -> torch.Tensor:
    if batch_size <= 0:
        raise click.BadParameter("dummy_batch_size must be greater than 0.")
    if num_samples <= 0:
        raise click.BadParameter("dummy_num_samples must be greater than 0.")
    if num_samples % model.config.downsample_factor != 0:
        raise click.BadParameter(
            "dummy_num_samples should be divisible by model.config.downsample_factor "
            f"({model.config.downsample_factor}) for clean encoder/decoder shapes."
        )
    return torch.randn(batch_size, 1, num_samples, dtype=next(model.parameters()).dtype, device=device)


def _dummy_codes(model: torch.nn.Module, batch_size: int, code_length: int, device: torch.device) -> torch.Tensor:
    if batch_size <= 0:
        raise click.BadParameter("dummy_batch_size must be greater than 0.")
    if code_length <= 0:
        raise click.BadParameter("dummy_code_length must be greater than 0.")

    # HF documents audio_codes as (batch, num_quantizers, codes_length).
    if hasattr(model, "quantizer") and hasattr(model.quantizer, "vq"):
        num_quantizers = len(model.quantizer.vq.layers)
    elif hasattr(model, "quantizer") and hasattr(model.quantizer, "quantizers"):
        num_quantizers = len(model.quantizer.quantizers)
    else:
        acoustic_config = model.config.acoustic_model_config
        num_quantizers = (
            acoustic_config["n_codebooks"] if isinstance(acoustic_config, dict) else acoustic_config.n_codebooks
        )
    codebook_size = model.quantizer.bins if hasattr(model, "quantizer") and hasattr(model.quantizer, "bins") else model.config.codebook_size
    return torch.randint(
        low=0,
        high=codebook_size,
        size=(batch_size, num_quantizers, code_length),
        dtype=torch.long,
        device=device,
    )


@click.command()
@click.option("--model-id", default="bosonai/higgs-audio-v2-tokenizer", show_default=True)
@click.option("--output-dir", default="artifacts/tokenizer_onnx", show_default=True, type=click.Path(path_type=Path))
@click.option(
    "--loader",
    default="legacy",
    show_default=True,
    type=click.Choice(["legacy", "transformers"], case_sensitive=False),
    help="Use legacy loader for model.pth or Transformers loader for HiggsAudioV2TokenizerModel.",
)
@click.option(
    "--component",
    default="both",
    show_default=True,
    type=click.Choice(["encoder", "decoder", "forward", "both"], case_sensitive=False),
)
@click.option("--device", default="cpu", show_default=True, help="Torch device used during export, e.g. cpu or cuda:0.")
@click.option("--dtype", default="float32", show_default=True, help="Export dtype: float32, float16, or bfloat16.")
@click.option("--bandwidth", default=None, type=float, help="Tokenizer bandwidth in kbps. Defaults to model default.")
@click.option("--dummy-batch-size", default=1, show_default=True, type=int)
@click.option("--dummy-num-samples", default=24000, show_default=True, type=int)
@click.option("--dummy-code-length", default=25, show_default=True, type=int)
@click.option("--opset", default=17, show_default=True, type=int)
@click.option("--external-data/--single-file", default=True, show_default=True)
@click.option("--dynamo/--no-dynamo", default=False, show_default=True)
def main(
    model_id: str,
    output_dir: Path,
    loader: str,
    component: str,
    device: str,
    dtype: str,
    bandwidth: float | None,
    dummy_batch_size: int,
    dummy_num_samples: int,
    dummy_code_length: int,
    opset: int,
    external_data: bool,
    dynamo: bool,
) -> None:
    """Export bosonai/higgs-audio-v2-tokenizer components to ONNX."""
    torch_dtype = _parse_dtype(dtype)
    if device == "cpu" and torch_dtype == torch.float16:
        raise click.BadParameter("float16 export on CPU is unreliable. Use float32/bfloat16 or export on CUDA.")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    if loader.lower() == "legacy":
        if component.lower() not in {"decoder", "both"}:
            raise click.BadParameter("legacy loader currently supports decoder export. Use --component decoder or both.")
        model = _load_legacy_tokenizer(
            model_id,
            torch_device,
            torch_dtype,
            skip_semantic_model=component.lower() in {"decoder", "both"},
        )
        component = "decoder" if component.lower() == "both" else component
    else:
        model = _load_tokenizer_model(model_id, torch_device, torch_dtype)

    selected = component.lower()
    export_encoder = selected in {"encoder", "both"}
    export_decoder = selected in {"decoder", "both"}
    export_forward = selected == "forward"

    with torch.inference_mode():
        if export_encoder:
            wrapper = EncoderWrapper(model, bandwidth).to(torch_device).eval()
            dummy = (_dummy_audio(model, dummy_batch_size, dummy_num_samples, torch_device),)
            _ = wrapper(*dummy)
            _export(
                module=wrapper,
                args=dummy,
                output_path=output_dir / "higgs_audio_v2_tokenizer_encoder.onnx",
                input_names=["input_values"],
                output_names=["audio_codes"],
                dynamic_axes={
                    "input_values": {0: "batch", 2: "num_samples"},
                    "audio_codes": {0: "batch", 2: "codes_length"},
                },
                opset=opset,
                external_data=external_data,
                dynamo=dynamo,
            )

        if export_decoder:
            wrapper_cls = LegacyDecoderWrapper if loader.lower() == "legacy" else DecoderWrapper
            wrapper = wrapper_cls(model).to(torch_device).eval()
            dummy = (_dummy_codes(model, dummy_batch_size, dummy_code_length, torch_device),)
            _ = wrapper(*dummy)
            _export(
                module=wrapper,
                args=dummy,
                output_path=output_dir / "higgs_audio_v2_tokenizer_decoder.onnx",
                input_names=["audio_codes"],
                output_names=["audio_values"],
                dynamic_axes={
                    "audio_codes": {0: "batch", 2: "codes_length"},
                    "audio_values": {0: "batch", 2: "num_samples"},
                },
                opset=opset,
                external_data=external_data,
                dynamo=dynamo,
            )

        if export_forward:
            wrapper = ForwardWrapper(model, bandwidth).to(torch_device).eval()
            dummy = (_dummy_audio(model, dummy_batch_size, dummy_num_samples, torch_device),)
            _ = wrapper(*dummy)
            _export(
                module=wrapper,
                args=dummy,
                output_path=output_dir / "higgs_audio_v2_tokenizer_forward.onnx",
                input_names=["input_values"],
                output_names=["audio_codes", "audio_values"],
                dynamic_axes={
                    "input_values": {0: "batch", 2: "num_samples"},
                    "audio_codes": {0: "batch", 2: "codes_length"},
                    "audio_values": {0: "batch", 2: "num_samples"},
                },
                opset=opset,
                external_data=external_data,
                dynamo=dynamo,
            )

    click.echo(f"Exported {selected} ONNX graph(s) to {output_dir}")


if __name__ == "__main__":
    main()
