"""
_jax_stubs.py
-------------
Inject minimal stub modules for the JAX/Flax-importing openpi files that
PI0Pytorch transitively depends on, so it can be imported in a PyTorch-only
environment.

Three openpi files import JAX/Flax at module level even though PI0Pytorch
itself is pure PyTorch:
  - openpi/models/gemma.py       → imports flax.linen, jax
  - openpi/models/lora.py        → imports flax, jax
  - openpi/shared/array_typing.py → imports jax, jaxtyping, beartype
  - openpi/shared/image_tools.py  → imports jax (the torch fn doesn't use it)

We pre-populate sys.modules with lightweight pure-Python replacements that
expose exactly the API PI0Pytorch and preprocessing_pytorch.py need.

Call inject() ONCE, before any openpi import.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import torch
import torch.nn.functional as F


def inject() -> None:
    """Inject JAX stubs into sys.modules.  Idempotent."""
    if "openpi.models.gemma" in sys.modules:
        return

    # Python requires parent packages to be in sys.modules before child modules.
    # Pre-populate any intermediate package that doesn't already exist.
    for pkg in ("openpi.models", "openpi.shared"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            sys.modules[pkg] = mod

    # ── openpi.models.lora ────────────────────────────────────────────────

    @dataclasses.dataclass
    class LoRAConfig:
        rank: int = 0
        alpha: float = 1.0

    lora_mod = types.ModuleType("openpi.models.lora")
    lora_mod.LoRAConfig = LoRAConfig
    sys.modules.setdefault("openpi.models.lora", lora_mod)

    # ── openpi.models.gemma ───────────────────────────────────────────────
    # Only used by PI0Pytorch to read architectural hyperparameters.

    @dataclasses.dataclass
    class GemmaConfig:
        width: int
        depth: int
        mlp_dim: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        lora_configs: dict = dataclasses.field(default_factory=dict)

    _GEMMA_CONFIGS = {
        "dummy":           GemmaConfig(width=64,   depth=4,  mlp_dim=128,    num_heads=8, num_kv_heads=1, head_dim=16),
        "gemma_300m":      GemmaConfig(width=1024, depth=18, mlp_dim=4096,   num_heads=8, num_kv_heads=1, head_dim=256),
        "gemma_300m_lora": GemmaConfig(width=1024, depth=18, mlp_dim=4096,   num_heads=8, num_kv_heads=1, head_dim=256),
        "gemma_2b":        GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
        "gemma_2b_lora":   GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
    }

    def get_config(variant: str) -> GemmaConfig:
        if variant not in _GEMMA_CONFIGS:
            raise ValueError(f"Unknown gemma variant: {variant!r}. Known: {list(_GEMMA_CONFIGS)}")
        return _GEMMA_CONFIGS[variant]

    gemma_mod = types.ModuleType("openpi.models.gemma")
    gemma_mod.Config = GemmaConfig
    gemma_mod.GemmaConfig = GemmaConfig
    gemma_mod.get_config = get_config
    gemma_mod.Variant = str
    gemma_mod.PALIGEMMA_VOCAB_SIZE = 257_152
    sys.modules["openpi.models.gemma"] = gemma_mod

    # ── openpi.shared.array_typing ────────────────────────────────────────
    # Used for runtime type-checking decorators.  We replace with no-ops.

    def _identity(fn):
        return fn

    class _AnyType:
        """Absorbs any subscript (e.g., at.UInt8[at.Array, '...'])."""
        def __class_getitem__(cls, _):
            return cls
        def __getitem__(self, _):
            return self

    _any = _AnyType()

    at_mod = types.ModuleType("openpi.shared.array_typing")
    at_mod.typecheck = _identity
    at_mod.Array = _any
    at_mod.UInt8 = _any
    at_mod.Float = _any
    at_mod.Int = _any
    at_mod.Bool = _any
    at_mod.Num = _any
    at_mod.Real = _any
    at_mod.Key = _any
    at_mod.PyTree = _any
    sys.modules.setdefault("openpi.shared.array_typing", at_mod)

    # ── openpi.shared.image_tools ─────────────────────────────────────────
    # resize_with_pad_torch is pure PyTorch; we lift it out of the file that
    # imports JAX at the top.

    def resize_with_pad_torch(
        images: torch.Tensor,
        height: int,
        width: int,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        if images.shape[-1] <= 4:
            channels_last = True
            if images.dim() == 3:
                images = images.unsqueeze(0)
            images = images.permute(0, 3, 1, 2)
        else:
            channels_last = False
            if images.dim() == 3:
                images = images.unsqueeze(0)

        batch_size, _channels, cur_height, cur_width = images.shape
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width  = int(cur_width  / ratio)

        resized = F.interpolate(
            images.float(),
            size=(resized_height, resized_width),
            mode=mode,
            align_corners=(False if mode == "bilinear" else None),
        )

        if images.dtype == torch.uint8:
            resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        elif images.dtype == torch.float32:
            resized = resized.clamp(-1.0, 1.0)

        pad_h0, rem_h = divmod(height - resized_height, 2)
        pad_h1 = pad_h0 + rem_h
        pad_w0, rem_w = divmod(width  - resized_width,  2)
        pad_w1 = pad_w0 + rem_w
        fill = 0 if images.dtype == torch.uint8 else -1.0
        padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=fill)

        if channels_last:
            padded = padded.permute(0, 2, 3, 1)
            if batch_size == 1:
                padded = padded.squeeze(0)
        return padded

    it_mod = types.ModuleType("openpi.shared.image_tools")
    it_mod.resize_with_pad_torch = resize_with_pad_torch
    sys.modules.setdefault("openpi.shared.image_tools", it_mod)
