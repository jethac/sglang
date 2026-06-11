# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""DiffusionGemma SGLang bringup shell.

This file intentionally implements only the DG-S0/DG-S2 foundation:
configuration/registration, BF16 weight remapping into one Gemma 4 backbone,
self-conditioning weight ownership, and the encoder/commit causal forward path.
The bidirectional denoising canvas scheduler is a later rung.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.diffusion_gemma import DiffusionGemmaConfig
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM

logger = logging.getLogger(__name__)


class DiffusionGemmaSelfConditioning(nn.Module):
    """Gated self-conditioning MLP from the vLLM DiffusionGemma implementation."""

    def __init__(
        self,
        hidden_size: int,
        self_conditioning_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.post_norm = RMSNorm(hidden_size, eps=eps, has_weight=False)
        self.gate_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.down_proj = nn.Linear(self_conditioning_size, hidden_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        soft_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pre_norm(soft_embeds)
        signal = self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
        return self.post_norm(inputs_embeds + signal)


def _diffusion_text_config(config: PretrainedConfig) -> PretrainedConfig:
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def _remap_backbone_name(name: str) -> Optional[str]:
    """Map DiffusionGemma checkpoint names to Gemma4ForCausalLM names.

    Returns None for non-language weights that are deliberately quarantined in
    text-only DG-S0/DG-S2.
    """

    if name.startswith("model.encoder.vision_tower.") or name.startswith(
        "model.encoder.embed_vision."
    ):
        return None

    encoder_prefix = "model.encoder.language_model."
    decoder_prefix = "model.decoder."

    if name.startswith(encoder_prefix):
        suffix = name[len(encoder_prefix) :]
        return suffix if suffix.startswith(("model.", "lm_head.")) else "model." + suffix

    if name.startswith(decoder_prefix):
        suffix = name[len(decoder_prefix) :]
        if suffix.startswith("self_conditioning."):
            return None
        return suffix if suffix.startswith(("model.", "lm_head.")) else "model." + suffix

    if name.startswith("vision_tower.") or name.startswith("embed_vision."):
        return None

    return name


class DiffusionGemmaForBlockDiffusion(nn.Module):
    """DiffusionGemma foundation model shell for SGLang.

    The model owns one Gemma 4 causal backbone. Encoder/commit mode delegates to
    that backbone and writes ordinary causal KV. Decoder denoise mode is
    intentionally not implemented here; serving it requires the block-AR
    scheduler and bidirectional canvas attention planned in
    docs/SGLANG_DIFFUSIONGEMMA_FEASIBILITY.md.
    """

    config_class = DiffusionGemmaConfig
    packed_modules_mapping = Gemma4ForCausalLM.packed_modules_mapping

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.text_config = _diffusion_text_config(config)
        self.quant_config = quant_config

        # Match vLLM's DiffusionGemma assumptions for the shared Gemma 4 backbone.
        self.text_config.router_uses_prenormed_input = False
        self.text_config.attention_k_eq_v = True

        self.language_model = Gemma4ForCausalLM(
            self.text_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        sc_size = (
            getattr(config, "self_conditioning_size", None)
            or getattr(self.text_config, "intermediate_size", None)
            or getattr(self.text_config, "hidden_size")
        )
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            hidden_size=self.text_config.hidden_size,
            self_conditioning_size=sc_size,
            eps=getattr(self.text_config, "rms_norm_eps", 1e-6),
        )

        self.dg_weight_load_manifest = {}

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def get_attention_sliding_window_size(self):
        return self.language_model.get_attention_sliding_window_size()

    def dtype(self) -> torch.dtype:
        return self.language_model.dtype()

    def get_diffusion_gemma_geometry_manifest(self) -> dict:
        cfg = self.text_config
        layer_types = list(getattr(cfg, "layer_types", []))
        num_layers = getattr(cfg, "num_hidden_layers", len(layer_types))
        if not layer_types:
            layer_types = ["sliding_attention"] * num_layers

        local_head_dim = getattr(cfg, "swa_head_dim", getattr(cfg, "head_dim", None))
        global_head_dim = getattr(cfg, "head_dim", local_head_dim)
        local_kv_heads = getattr(
            cfg,
            "swa_num_key_value_heads",
            getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", None)),
        )
        global_kv_heads = getattr(
            cfg,
            "num_key_value_heads",
            getattr(cfg, "num_attention_heads", None),
        )
        dtype_bytes = 2

        layers = []
        for layer_id, layer_type in enumerate(layer_types):
            is_global = layer_type == "full_attention"
            head_dim = global_head_dim if is_global else local_head_dim
            kv_heads = global_kv_heads if is_global else local_kv_heads
            layers.append(
                {
                    "layer": layer_id,
                    "type": layer_type,
                    "head_dim": head_dim,
                    "num_attention_heads": getattr(cfg, "num_attention_heads", None),
                    "num_key_value_heads": kv_heads,
                    "sliding_window": None
                    if is_global
                    else getattr(cfg, "sliding_window", None),
                    "bf16_kv_bytes_per_token": (
                        2 * int(kv_heads) * int(head_dim) * dtype_bytes
                        if kv_heads is not None and head_dim is not None
                        else None
                    ),
                }
            )

        return {
            "architecture": "DiffusionGemmaForBlockDiffusion",
            "canvas_length": getattr(self.config, "canvas_length", None),
            "max_denoising_steps": getattr(self.config, "max_denoising_steps", None),
            "confidence_threshold": getattr(self.config, "confidence_threshold", None),
            "stability_threshold": getattr(self.config, "stability_threshold", None),
            "sampler_config": getattr(self.config, "sampler_config", None),
            "num_hidden_layers": num_layers,
            "local_head_dim": local_head_dim,
            "global_head_dim": global_head_dim,
            "layers": layers,
        }

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        diffusion_mode: str = "encoder",
        **kwargs,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if diffusion_mode not in ("encoder", "commit"):
            raise NotImplementedError(
                "DiffusionGemma decoder/canvas mode requires the DG-S3+ "
                "block-AR scheduler and bidirectional canvas attention."
            )
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        sc_params = dict(self.self_conditioning.named_parameters())
        seen_backbone_names = set()
        manifest = {
            "self_conditioning_loaded": [],
            "self_conditioning_missing": [],
            "vision_quarantined": 0,
            "decoder_duplicate_skipped": 0,
            "backbone_seen": 0,
        }

        def remapped_backbone_weights():
            for name, weight in weights:
                if "self_conditioning." in name:
                    sc_name = name.split("self_conditioning.", 1)[1]
                    param = sc_params.get(sc_name)
                    if param is None:
                        manifest["self_conditioning_missing"].append(sc_name)
                    elif tuple(param.shape) != tuple(weight.shape):
                        manifest["self_conditioning_missing"].append(
                            f"{sc_name}: shape {tuple(weight.shape)} != {tuple(param.shape)}"
                        )
                    else:
                        param.data.copy_(weight)
                        manifest["self_conditioning_loaded"].append(sc_name)
                    continue

                mapped = _remap_backbone_name(name)
                if mapped is None:
                    manifest["vision_quarantined"] += 1
                    continue

                if mapped in seen_backbone_names:
                    manifest["decoder_duplicate_skipped"] += 1
                    continue
                seen_backbone_names.add(mapped)
                manifest["backbone_seen"] += 1
                yield mapped, weight

        loaded = self.language_model.load_weights(remapped_backbone_weights())
        manifest["backbone_loaded"] = sorted(loaded)
        manifest["backbone_loaded_count"] = len(loaded)
        self.dg_weight_load_manifest = manifest
        return loaded


EntryClass = DiffusionGemmaForBlockDiffusion

