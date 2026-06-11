from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

try:
    from transformers import Gemma4TextConfig
except ImportError:  # pragma: no cover - only for older local transformers builds.
    Gemma4TextConfig = PretrainedConfig


def _normalize_text_config(text_config: dict[str, Any]) -> dict[str, Any]:
    """Map DiffusionGemma local/global names onto SGLang's Gemma 4 names."""

    text_config = dict(text_config)
    text_config.pop("model_type", None)

    global_head_dim = text_config.get("global_head_dim")
    if global_head_dim is not None:
        text_config.setdefault("swa_head_dim", text_config.get("head_dim"))
        text_config["head_dim"] = global_head_dim

    global_kv_heads = text_config.get("num_global_key_value_heads")
    if global_kv_heads is not None:
        text_config.setdefault(
            "swa_num_key_value_heads", text_config.get("num_key_value_heads")
        )
        text_config["num_key_value_heads"] = global_kv_heads

    return text_config


class DiffusionGemmaConfig(PretrainedConfig):
    """Minimal DiffusionGemma config alias for SGLang day-zero bringup.

    Transformers nightlies already define the official config, but Spark images can
    lag the model release. This alias preserves the top-level diffusion fields and
    converts ``text_config`` into a Gemma 4 text config so existing SGLang shape
    derivation and model code can reason about the language backbone.
    """

    model_type = "diffusion_gemma"
    sub_configs = {"text_config": Gemma4TextConfig}
    _auto_class = "AutoConfig"

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        canvas_length: int = 256,
        max_denoising_steps: int = 48,
        confidence_threshold: float = 0.005,
        stability_threshold: int = 1,
        sampler_config: dict[str, Any] | None = None,
        self_conditioning_size: int | None = None,
        **kwargs,
    ):
        if text_config is None:
            self.text_config = Gemma4TextConfig()
        elif isinstance(text_config, dict):
            self.text_config = Gemma4TextConfig(**_normalize_text_config(text_config))
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = PretrainedConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.sampler_config = sampler_config or {}
        self.self_conditioning_size = self_conditioning_size

        kwargs.setdefault("architectures", ["DiffusionGemmaForBlockDiffusion"])
        super().__init__(**kwargs)
