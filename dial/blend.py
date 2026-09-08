"""Runtime scale interpolation for the two active PoliticalGPT LoRA adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeAlias

from peft.tuners.lora import LoraLayer


AdapterName: TypeAlias = str
BaseScaling: TypeAlias = Mapping[AdapterName, Mapping[LoraLayer, float]]


def weights(lambda_value: float) -> dict[str, float]:
    """Return the PEFT scale factors for the active CNN and Fox adapters."""
    return {"cnn": lambda_value, "fox": 1.0 - lambda_value}


def _configured_base_scale(layer: LoraLayer, adapter: AdapterName) -> float:
    rank = layer.r[adapter]
    denominator = math.sqrt(rank) if layer.use_rslora[adapter] else rank
    return layer.lora_alpha[adapter] / denominator


def set_lambda(model: Any, lambda_value: float, base_scaling: BaseScaling) -> None:
    """Rebase both active adapter scales through PEFT for ``lambda_value``.

    ``base_scaling`` is intentionally assertion-only: PEFT's ``set_scale``
    computes each factor from the immutable LoRA configuration, so multiplying
    a previous live scale would introduce drift between requests.
    """
    requested_weights = weights(lambda_value)
    for layer in model.modules():
        if not isinstance(layer, LoraLayer):
            continue
        for adapter in ("cnn", "fox"):
            captured_base_scale = base_scaling[adapter][layer]
            if not math.isclose(captured_base_scale, _configured_base_scale(layer, adapter)):
                raise AssertionError(f"captured base scale for {adapter!r} does not match PEFT configuration")
        layer.set_scale("cnn", requested_weights["cnn"])
        layer.set_scale("fox", requested_weights["fox"])
