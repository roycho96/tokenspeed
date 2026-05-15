"""Dense FP8 MLA decode wrappers (sm_90a only).

Loads ``flashmla_dense_fp8.so`` and exposes two entry points:

* :func:`flashmla_dense_fp8_metadata` — build tile-scheduler metadata.
* :func:`flashmla_dense_fp8_fwd`      — dense FP8 MLA decode forward.

For the MLA case (``num_heads_k == 1``) the wrapper folds the per-K-head Q
axis into the ``q_seq`` axis via a stride-only ``.view()`` so there is no
copy; output is allocated in its final ``(batch, seqlen_q_ori, num_heads_q,
head_dim_v)`` shape and viewed for the kernel. Only ``softmax_lse`` still
needs the transpose+reshape (~2 KB tensor, ~1 μs).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import torch
import tvm_ffi

_TILE_SCHEDULER_META_SIZE = 8
_META_BLOCK_SIZE_M = 64
_HEAD_DIM_K = 576
_HEAD_DIM_V = 512


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_flashmla_dense_fp8_module():
    so_path = _objs_dir() / "flashmla_dense_fp8" / "flashmla_dense_fp8.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel FlashMLA dense FP8 library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` on a Hopper host to "
            "build (the kernel is sm_90a only)."
        )
    return tvm_ffi.load_module(str(so_path))


def has_flashmla_dense_fp8() -> bool:
    """Return True if the sm_90a dense FP8 .so was built."""
    try:
        _load_flashmla_dense_fp8_module()
    except Exception:
        return False
    return True


def flashmla_dense_fp8_metadata(
    seqlens_k: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(tile_scheduler_metadata, num_splits)`` for dense FP8 decode.

    Args:
        seqlens_k: (batch,) int32 CUDA tensor of K cache seqlens.
        num_heads_per_head_k: ``num_heads_q // num_heads_k``.
        num_heads_k: number of K/V heads.

    Returns:
        ``(tile_scheduler_metadata, num_splits)`` as int32 CUDA tensors with
        shapes ``(num_sm_parts, 8)`` and ``(batch + 1,)``.
    """
    if not seqlens_k.is_cuda:
        raise RuntimeError("seqlens_k must be a CUDA tensor")
    if seqlens_k.dtype != torch.int32:
        seqlens_k = seqlens_k.to(torch.int32)
    seqlens_k = seqlens_k.contiguous()

    device = seqlens_k.device
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_sm_parts = max(
        1,
        sm_count
        // max(1, num_heads_k)
        // max(1, math.ceil(num_heads_per_head_k / _META_BLOCK_SIZE_M)),
    )
    batch_size = seqlens_k.size(0)

    tile_scheduler_metadata = torch.empty(
        (num_sm_parts, _TILE_SCHEDULER_META_SIZE),
        dtype=torch.int32,
        device=device,
    )
    num_splits = torch.empty((batch_size + 1,), dtype=torch.int32, device=device)

    _load_flashmla_dense_fp8_module().flashmla_dense_fp8_metadata(
        seqlens_k, tile_scheduler_metadata, num_splits
    )
    return tile_scheduler_metadata, num_splits


def flashmla_dense_fp8_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense FP8 MLA decode forward.

    Args:
        q: ``(batch, seqlen_q_ori, num_heads_q, 576)`` float8_e4m3fn.
        k_cache: ``(num_blocks, 64, num_heads_k, 576)`` float8_e4m3fn.
        block_table: ``(batch, max_num_blocks_per_seq)`` int32.
        cache_seqlens: ``(batch,)`` int32.
        tile_scheduler_metadata: ``(num_sm_parts, 8)`` int32.
        num_splits: ``(batch + 1,)`` int32.
        descale_q: ``(1,)`` float32.
        descale_k: ``(1,)`` float32.
        softmax_scale: softmax scale (typically ``1.0 / sqrt(head_dim_k)``).
        causal: causal mask. Forced to False when ``seqlen_q_ori == 1``.

    Returns:
        ``(out, softmax_lse)`` with shapes
        ``(batch, seqlen_q_ori, num_heads_q, 512)`` bf16 and
        ``(batch, num_heads_q, seqlen_q_ori)`` float32.
    """
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError(f"q must be float8_e4m3fn, got {q.dtype}")
    if k_cache.dtype != torch.float8_e4m3fn:
        raise TypeError(f"k_cache must be float8_e4m3fn, got {k_cache.dtype}")
    if q.dim() != 4 or q.size(-1) != _HEAD_DIM_K:
        raise ValueError(
            f"q must have shape (batch, seqlen_q, num_heads_q, {_HEAD_DIM_K}), "
            f"got {tuple(q.shape)}"
        )
    if k_cache.dim() != 4 or k_cache.size(-1) != _HEAD_DIM_K:
        raise ValueError(
            f"k_cache must have shape (num_blocks, page_block_size, "
            f"num_heads_k, {_HEAD_DIM_K}), got {tuple(k_cache.shape)}"
        )

    batch_size, seqlen_q_ori, num_heads_q, _ = q.shape
    num_heads_k = k_cache.size(2)
    if num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"num_heads_q ({num_heads_q}) must be divisible by num_heads_k "
            f"({num_heads_k})"
        )
    num_q_heads_per_hk = num_heads_q // num_heads_k
    q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk

    if seqlen_q_ori == 1:
        causal = False

    if num_heads_k == 1:
        # MLA fast path: original q is already laid out as the kernel wants.
        # `q.view(batch, seqlen_q_ori * num_heads_q, 1, head_dim_k)` merges
        # the (seqlen_q_ori, num_heads_q) dims into q_seq_per_hk without
        # touching memory — see PyTorch's view contiguity rule. Same for the
        # output: allocate in the final caller-facing shape and view for the
        # kernel.
        q_kernel = q.view(batch_size, q_seq_per_hk, 1, _HEAD_DIM_K)
        out = torch.empty(
            (batch_size, seqlen_q_ori, num_heads_q, _HEAD_DIM_V),
            dtype=torch.bfloat16,
            device=q.device,
        )
        out_kernel = out.view(batch_size, q_seq_per_hk, 1, _HEAD_DIM_V)
    else:
        # General path: fold via transpose+reshape (single copy; reshape after
        # transpose forces contiguous, so the previously-trailing
        # `.contiguous()` was redundant).
        q_kernel = (
            q.view(batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk, _HEAD_DIM_K)
            .transpose(2, 3)
            .reshape(batch_size, q_seq_per_hk, num_heads_k, _HEAD_DIM_K)
        )
        out_kernel = torch.empty(
            (batch_size, q_seq_per_hk, num_heads_k, _HEAD_DIM_V),
            dtype=torch.bfloat16,
            device=q.device,
        )

    softmax_lse_packed = torch.empty(
        (batch_size, num_heads_k, q_seq_per_hk),
        dtype=torch.float32,
        device=q.device,
    )

    _load_flashmla_dense_fp8_module().flashmla_dense_fp8_fwd(
        q_kernel,
        k_cache,
        cache_seqlens,
        block_table,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
        out_kernel,
        softmax_lse_packed,
        int(num_heads_q),
        float(softmax_scale),
        bool(causal),
    )

    if num_heads_k == 1:
        # out is already (batch, seqlen_q_ori, num_heads_q, head_dim_v).
        pass
    else:
        out = (
            out_kernel.view(
                batch_size, seqlen_q_ori, num_q_heads_per_hk, num_heads_k, _HEAD_DIM_V
            )
            .transpose(2, 3)
            .reshape(batch_size, seqlen_q_ori, num_heads_q, _HEAD_DIM_V)
        )

    # softmax_lse always needs the transpose: the kernel writes its
    # q_seq_per_hk axis in (sq, h_per_k)-major order but downstream expects
    # (h_full, sq) order. The tensor is small (~batch*num_heads_q*seqlen_q
    # floats); the copy is in the single-μs range.
    softmax_lse = (
        softmax_lse_packed.view(batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk)
        .transpose(2, 3)
        .reshape(batch_size, num_heads_q, seqlen_q_ori)
    )
    return out, softmax_lse
