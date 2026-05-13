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

"""Numerical regression test for the FlashMLA FP8 KV cache path.

Asserts that ``flash_mla_with_kvcache(..., is_fp8_kvcache=True, ...)`` produces
an output close to the BF16 reference on a DSv3-shaped decode/target-verify
workload. Parametrized over ``s_q in {1, 2}`` so the spec-decode target-verify
path (which feeds ``s_q > 1`` queries to the FP8 kernel via the same routing
as ``forward_decode``) is covered alongside plain decode.

Skip gates: no CUDA, not Hopper-plus, or upstream ``flash_mla`` not importable.
The FP8 path is implemented by the upstream kernel as a runtime dispatch on
``q.element_size() == 1`` plus the ``is_fp8_kvcache=True`` flag -- the same
contract used by ``backends/deepseek_v4.py:329-337`` and
``backends/flashmla.py`` after this change.

Numerical tolerance: cosine similarity > 0.999 on output, and
``torch.allclose(lse_bf16, lse_fp8, atol=1e-2, rtol=1e-2)`` on the log-sum-exp.
``5e-2`` would mask real bugs (off-by-one descale, wrong head_dim slicing);
``> 0.999`` matches vLLM's ``cal_diff`` convention.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch
import torch.nn.functional as F

# DSv3 shape.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_HEADS = 128
PAGE_SIZE = 64
BATCH = 4
MAX_SEQ_LEN = 1024
KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM


def _is_hopper_plus() -> bool:
    """Return True iff the current device is Hopper (SM90) or newer.

    FlashMLA's dense FP8 kernel is Hopper-only in upstream; SM_100 (Blackwell)
    is supported on the sparse path but the dense entry-point used here is
    gated to ``compute_cap >= (9, 0)``.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _flash_mla_unavailable() -> bool:
    try:
        import flash_mla  # noqa: F401

        return False
    except ImportError:
        return True


_SKIP_REASONS: list[str] = []
if not torch.cuda.is_available():
    _SKIP_REASONS.append("CUDA not available")
if torch.cuda.is_available() and not _is_hopper_plus():
    _SKIP_REASONS.append("Requires Hopper-plus (SM_90+)")
if _flash_mla_unavailable():
    _SKIP_REASONS.append("flash_mla not importable")

_SKIP_REASON = "; ".join(_SKIP_REASONS) if _SKIP_REASONS else ""


def _build_synthetic_inputs(s_q: int, device: str = "cuda"):
    """Build a DSv3-shaped ``(q, k_cache, block_table, cache_seqlens)`` quad.

    ``s_q`` is the per-request query length: ``1`` for plain decode, ``> 1``
    for the target-verify / draft-extend spec-decode path that
    ``backends/flashmla.py:forward_extend`` routes through the same FP8
    kernel as ``forward_decode``.
    """
    torch.manual_seed(0)

    # Realistic per-request seqlens; pad to <= MAX_SEQ_LEN.
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


def _run_kernel(q, k_cache, block_table, cache_seqlens, *, is_fp8: bool):
    """Invoke ``flash_mla_with_kvcache`` with BF16 or FP8 routing.

    BF16: bare call -- no FP8 kwargs (matches the original
    ``backends/flashmla.py:forward_decode`` call shape on ``main``).

    FP8: cast both q and k_cache to ``float8_e4m3fn``, pass 1-element
    ``descale_q`` / ``descale_k`` ones tensors, and set
    ``is_fp8_kvcache=True``.
    """
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    softmax_scale = 1.0 / math.sqrt(KV_CACHE_DIM)
    s_q = q.shape[1]

    if is_fp8:
        # NOTE: dynamic q_scale plumbing is a tracked follow-up; the backend
        # pins q_scale=1.0 today, so we mirror that here.
        descale_q = torch.ones(1, device=q.device, dtype=torch.float32)
        descale_k = torch.ones(1, device=q.device, dtype=torch.float32)
        q_kernel = q.to(torch.float8_e4m3fn)
        k_cache_kernel = k_cache.to(torch.float8_e4m3fn)
        tile_metadata, num_splits = get_mla_metadata(
            cache_seqlens,
            s_q * NUM_HEADS,
            1,
            is_fp8_kvcache=True,
        )
        out, lse = flash_mla_with_kvcache(
            q=q_kernel,
            k_cache=k_cache_kernel,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=KV_LORA_RANK,
            tile_scheduler_metadata=tile_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=True,
            is_fp8_kvcache=True,
            descale_q=descale_q,
            descale_k=descale_k,
        )
    else:
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


@pytest.mark.skipif(bool(_SKIP_REASONS), reason=_SKIP_REASON or "skipped")
@pytest.mark.parametrize("s_q", [1, 2])
def test_flashmla_fp8_matches_bf16_reference(s_q: int) -> None:
    """FP8 KV path should match BF16 reference within cosine_similarity > 0.999.

    ``s_q=1`` covers plain decode; ``s_q=2`` covers the target-verify /
    draft-extend spec-decode path that ``backends/flashmla.py:forward_extend``
    routes through the same FP8 kernel as ``forward_decode``.
    """
    device = "cuda"
    q_bf16, k_cache_bf16, block_table, cache_seqlens = _build_synthetic_inputs(
        s_q, device=device
    )

    out_bf16, lse_bf16 = _run_kernel(
        q_bf16, k_cache_bf16, block_table, cache_seqlens, is_fp8=False
    )
    out_fp8, lse_fp8 = _run_kernel(
        q_bf16, k_cache_bf16, block_table, cache_seqlens, is_fp8=True
    )

    # Shape parity sanity check.
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
    assert cos > 0.999, (
        f"FP8 output diverged from BF16 reference (cosine_similarity={cos:.6f}); "
        f"s_q={s_q}. Expected > 0.999."
    )

    torch.testing.assert_close(
        lse_bf16.float(),
        lse_fp8.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"FP8 lse diverged from BF16 reference at s_q={s_q}",
    )
