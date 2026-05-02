from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ElementwiseFeatureBias(nn.Module):
    """
    Add a trainable per-element bias (and optional scale) to a per-atom feature.

    Typical use: increase element separation of electronegativity-like features before Qeq:

        chi_biased = scale * chi_raw + bias[element]

    Notes:
    - This module is intentionally lightweight and generic.
    - `atomic_numbers` is used to map atoms -> element index in `elements`.
    """

    def __init__(
        self,
        elements: List[int],
        feature_key: str,
        output_key: str,
        atomic_numbers_key: str = "atomic_numbers",
        init_bias: Optional[Dict[int, float]] = None,
        use_scale: bool = True,
        init_scale: float = 1.0,
        zero_mean_bias: bool = True,
    ):
        super().__init__()
        self.elements = [int(z) for z in elements]
        self.feature_key = feature_key
        self.output_key = output_key
        self.atomic_numbers_key = atomic_numbers_key
        self.zero_mean_bias = bool(zero_mean_bias)

        # element -> index (buffer for torchscript friendliness)
        z_to_idx = {int(z): i for i, z in enumerate(self.elements)}
        # store mapping as a simple dict; used only in eager mode
        self._z_to_idx = z_to_idx

        bias0 = torch.zeros(len(self.elements), dtype=torch.get_default_dtype())
        if init_bias is not None:
            for z, v in init_bias.items():
                z_int = int(z)
                if z_int in z_to_idx:
                    bias0[z_to_idx[z_int]] = float(v)
        self.bias = nn.Parameter(bias0)

        self.use_scale = bool(use_scale)
        if self.use_scale:
            self.scale = nn.Parameter(
                torch.tensor(float(init_scale), dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "scale", torch.tensor(1.0, dtype=torch.get_default_dtype())
            )

        self.model_outputs = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        if self.feature_key not in data:
            raise KeyError(f"ElementwiseFeatureBias: missing `{self.feature_key}` in data")
        if self.atomic_numbers_key not in data:
            raise KeyError(
                f"ElementwiseFeatureBias: missing `{self.atomic_numbers_key}` in data"
            )

        feat = data[self.feature_key]
        if feat.dim() == 1:
            feat = feat.unsqueeze(1)

        z = data[self.atomic_numbers_key].view(-1)
        idx = torch.empty_like(z, dtype=torch.long)
        for z_val, i in self._z_to_idx.items():
            idx[z == z_val] = int(i)

        b = self.bias
        if self.zero_mean_bias:
            b = b - b.mean()

        feat_out = self.scale * feat + b[idx].unsqueeze(1)
        data[self.output_key] = feat_out
        return data

