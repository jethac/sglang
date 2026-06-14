"""NVFP4 KV calibrated global-scale loader.

Calibration files are keyed by architecture signature rather than model name so
fine-tunes and re-uploads with the same KV geometry share one calibration.  The
scale values here use SGLang's dequant convention:

    x_bf16 ~= x_fp4 * block_scale * global_scale

Do not feed vLLM `_k_scale` files here unless they explicitly declare the
SGLang/dequant-global convention.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENV_NVFP4_KV_CALIB = "SGLANG_NVFP4_KV_CALIB"


@dataclass(frozen=True)
class NVFP4KVCalibration:
    arch_signature: str
    k_global_scale: float
    v_global_scale: float
    source: str


@functools.lru_cache(maxsize=32)
def _load_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load NVFP4 KV calibration %s: %r", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Ignoring NVFP4 KV calibration %s: expected JSON object", path)
        return None
    return data


def _cfg_get(hf_config: Any, key: str) -> Any:
    value = getattr(hf_config, key, None)
    if value is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            value = getattr(text_config, key, None)
    return value


def _head_dim(hf_config: Any) -> Any:
    head_dim = _cfg_get(hf_config, "head_dim")
    if head_dim is not None:
        return head_dim
    hidden_size = _cfg_get(hf_config, "hidden_size")
    num_heads = _cfg_get(hf_config, "num_attention_heads")
    try:
        if hidden_size and num_heads:
            return int(hidden_size) // int(num_heads)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def arch_signature(hf_config: Any) -> Optional[str]:
    """Return a stable key from architecture and KV geometry."""

    if hf_config is None:
        return None
    archs = getattr(hf_config, "architectures", None) or [None]
    arch = archs[0] or _cfg_get(hf_config, "model_type") or "unknown"
    parts = [
        str(arch),
        f"L{_cfg_get(hf_config, 'num_hidden_layers')}",
        f"H{_cfg_get(hf_config, 'hidden_size')}",
        f"D{_head_dim(hf_config)}",
        f"KV{_cfg_get(hf_config, 'num_key_value_heads')}",
    ]
    return "-".join(parts)


def _positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _extract_scales(data: dict[str, Any]) -> Optional[tuple[float, float]]:
    sglang = data.get("sglang")
    if isinstance(sglang, dict):
        k = _positive_float(sglang.get("k_global_scale"))
        v = _positive_float(sglang.get("v_global_scale"))
        if k is not None and v is not None:
            return k, v

    k = _positive_float(data.get("k_global_scale"))
    v = _positive_float(data.get("v_global_scale"))
    if k is not None and v is not None:
        return k, v

    shared = _positive_float(data.get("dequant_global_scale"))
    if shared is not None:
        return shared, shared

    convention = str(data.get("scale_convention", "")).lower()
    if convention in ("sglang_dequant_global", "dequant_global_scale"):
        k = _positive_float(data.get("k_scale"))
        v = _positive_float(data.get("v_scale"))
        if k is not None and v is not None:
            return k, v

    return None


def calibrated_kv_global_scales(hf_config: Any) -> Optional[NVFP4KVCalibration]:
    """Load SGLang dequant-global scales for hf_config, if configured."""

    src = os.environ.get(ENV_NVFP4_KV_CALIB)
    if not src:
        return None
    sig = arch_signature(hf_config)
    if sig is None:
        return None

    if os.path.isdir(src):
        path = os.path.join(src, sig + ".json")
        data = _load_json(path)
        source = path
    else:
        data = _load_json(src)
        source = src
        if data and data.get("arch_signature") and data["arch_signature"] != sig:
            logger.info(
                "NVFP4 KV calibration %s skipped: arch_signature %s != %s",
                src,
                data["arch_signature"],
                sig,
            )
            return None

    if not data:
        return None
    scales = _extract_scales(data)
    if scales is None:
        logger.warning(
            "Ignoring NVFP4 KV calibration %s: missing positive SGLang "
            "k_global_scale/v_global_scale",
            source,
        )
        return None
    k_global_scale, v_global_scale = scales
    return NVFP4KVCalibration(
        arch_signature=sig,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        source=source,
    )
