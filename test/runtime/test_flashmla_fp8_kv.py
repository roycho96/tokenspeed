# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Numerical regression test for the dense FP8 FlashMLA KV cache path.

Compares ``flashmla_dense_fp8_fwd`` against the BF16 reference on a
DSv3-shaped decode/target-verify workload. The dense FP8 entry points live
at ``tokenspeed_kernel.ops.attention.flash_mla``:

* ``flashmla_dense_fp8_metadata(seqlens_k, num_q_heads_per_head_k, num_heads_k)``
* ``flashmla_dense_fp8_fwd(q, k_cache, block_table, cache_seqlens,
  tile_scheduler_metadata, num_splits, descale_q, descale_k, softmax_scale,
  causal)``

``s_q in {1, 2}`` covers plain decode and the spec-decode target-verify path
that ``backends/flashmla.py:forward_extend`` routes through the same kernel
as ``forward_decode`` after this change.

Skip gates: no CUDA, not Hopper-plus (sm_90a), BF16 ``flash_mla`` reference
not importable, or the sm_90a-only ``flashmla_dense_fp8.so`` was not built
(non-Hopper CI host).
"""

from __future__ import annotations

import math
import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# DSv3 shape.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_HEADS = 128
PAGE_SIZE = 64
BATCH = 4
MAX_SEQ_LEN = 1024
KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM


def _is_hopper_plus() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _bf16_flash_mla_unavailable() -> bool:
    try:
        import flash_mla  # noqa: F401

        return False
    except ImportError:
        return True


def _dense_fp8_unavailable() -> bool:
    try:
        from tokenspeed_kernel.thirdparty.cuda.flashmla_dense_fp8 import (
            has_flashmla_dense_fp8,
        )
    except ImportError:
        return True
    return not has_flashmla_dense_fp8()


_SKIP_REASONS: list[str] = []
if not torch.cuda.is_available():
    _SKIP_REASONS.append("CUDA not available")
if torch.cuda.is_available() and not _is_hopper_plus():
    _SKIP_REASONS.append("Requires Hopper-plus (SM_90+)")
if _bf16_flash_mla_unavailable():
    _SKIP_REASONS.append("upstream flash_mla not importable (BF16 reference)")
if _dense_fp8_unavailable():
    _SKIP_REASONS.append("flashmla_dense_fp8.so not built (sm_90a only)")

_SKIP_REASON = "; ".join(_SKIP_REASONS) if _SKIP_REASONS else ""


def _build_synthetic_inputs(s_q: int, device: str = "cuda"):
    torch.manual_seed(0)

    cache_seqlens = torch.tensor([424, 531, 851, 987], device=device, dtype=torch.int32)
    assert cache_seqlens.numel() == BATCH

    num_blocks_per_seq = (cache_seqlens + PAGE_SIZE - 1) // PAGE_SIZE
    max_num_blocks = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    q_bf16 = torch.randn(
        BATCH,
        s_q,
        NUM_HEADS,
        KV_CACHE_DIM,
        device=device,
        dtype=torch.bfloat16,
    )

    block_table = torch.zeros(
        BATCH,
        max_num_blocks,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        block_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache_bf16 = torch.zeros(
        total_num_blocks,
        PAGE_SIZE,
        1,
        KV_CACHE_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(block_table[batch_idx, block_idx].item())
            block_start = block_idx * PAGE_SIZE
            tokens_in_block = min(PAGE_SIZE, total_kv_len - block_start)
            k_cache_bf16[physical_block, :tokens_in_block] = torch.randn(
                tokens_in_block,
                1,
                KV_CACHE_DIM,
                device=device,
                dtype=torch.bfloat16,
            )

    return q_bf16, k_cache_bf16, block_table, cache_seqlens


def _run_bf16_reference(q, k_cache, block_table, cache_seqlens):
    """BF16 reference via upstream ``flash_mla_with_kvcache``."""
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    softmax_scale = 1.0 / math.sqrt(KV_CACHE_DIM)
    s_q = q.shape[1]
    tile_metadata, num_splits = get_mla_metadata(
        cache_seqlens,
        s_q * NUM_HEADS,
        1,
    )
    out, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=KV_LORA_RANK,
        tile_scheduler_metadata=tile_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=True,
    )
    return out, lse


def _run_dense_fp8(q, k_cache, block_table, cache_seqlens):
    """Dense FP8 MLA decode path."""
    from tokenspeed_kernel.ops.attention.flash_mla import (
        flashmla_dense_fp8_fwd,
        flashmla_dense_fp8_metadata,
    )

    softmax_scale = 1.0 / math.sqrt(KV_CACHE_DIM)
    s_q = q.shape[1]

    descale_q = torch.ones(1, device=q.device, dtype=torch.float32)
    descale_k = torch.ones(1, device=q.device, dtype=torch.float32)
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_cache_fp8 = k_cache.to(torch.float8_e4m3fn)

    tile_metadata, num_splits = flashmla_dense_fp8_metadata(
        cache_seqlens,
        s_q * NUM_HEADS,
        1,
    )
    out, lse = flashmla_dense_fp8_fwd(
        q_fp8,
        k_cache_fp8,
        block_table,
        cache_seqlens,
        tile_metadata,
        num_splits,
        descale_q,
        descale_k,
        softmax_scale,
        causal=True,
    )
    return out, lse


@pytest.mark.skipif(bool(_SKIP_REASONS), reason=_SKIP_REASON or "skipped")
@pytest.mark.parametrize("s_q", [1, 2])
def test_flashmla_dense_fp8_matches_bf16_reference(s_q: int) -> None:
    """Dense FP8 output should match the BF16 reference (cosine_similarity > 0.99).

    ``s_q=1`` covers plain decode; ``s_q=2`` covers the target-verify /
    draft-extend spec-decode path that ``backends/flashmla.py:forward_extend``
    routes through the same kernel as ``forward_decode``.
    """
    device = "cuda"
    q_bf16, k_cache_bf16, block_table, cache_seqlens = _build_synthetic_inputs(
        s_q, device=device
    )

    out_bf16, lse_bf16 = _run_bf16_reference(
        q_bf16, k_cache_bf16, block_table, cache_seqlens
    )
    out_fp8, lse_fp8 = _run_dense_fp8(q_bf16, k_cache_bf16, block_table, cache_seqlens)

    assert (
        out_bf16.shape == out_fp8.shape
    ), f"out shape mismatch: bf16={out_bf16.shape} fp8={out_fp8.shape}"
    assert (
        lse_bf16.shape == lse_fp8.shape
    ), f"lse shape mismatch: bf16={lse_bf16.shape} fp8={lse_fp8.shape}"

    cos = F.cosine_similarity(
        out_bf16.flatten().float(),
        out_fp8.flatten().float(),
        dim=0,
    ).item()
    assert cos > 0.99, (
        f"FP8 output diverged from BF16 reference "
        f"(cosine_similarity={cos:.6f}); s_q={s_q}. Expected > 0.99."
    )

    torch.testing.assert_close(
        lse_bf16.float(),
        lse_fp8.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"FP8 lse diverged from BF16 reference at s_q={s_q}",
    )
