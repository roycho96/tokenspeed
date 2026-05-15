// TVM FFI binding for the dense FP8 MLA decode kernel (sm_90a).
// Caller (Python) handles any q/out view, transpose, or reshape; this
// binding consumes raw pointers and strides already shaped for the kernel.

#include <cstdint>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/fast_math.h>

#include "tvm_ffi_utils.h"
#include "flashmla_dense_fp8/flash_mla.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kHeadSizeK = 576;
constexpr int kHeadSizeV = 512;
constexpr int kPageBlockSize = 64;

// Tile-scheduler metadata constants.
constexpr int kMetaBlockSizeM = 64;
constexpr int kMetaBlockSizeN = 64;
constexpr int kMetaFixedOverheadNumBlocks = 5;

}  // namespace

// Forward pass for dense FP8 MLA decode on SM90a.
//
// Shapes (caller pre-arranged):
//   q:                       (batch, q_seq_per_hk, num_heads_k, kHeadSizeK), fp8_e4m3
//   k_cache:                 (num_blocks, kPageBlockSize, num_heads_k, kHeadSizeK), fp8_e4m3
//   seqlens_k:               (batch,), int32
//   block_table:             (batch, max_num_blocks_per_seq), int32
//   tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), int32
//   num_splits:              (batch + 1,), int32
//   descale_q:               (1,), float32
//   descale_k:               (1,), float32
//   out:                     (batch, q_seq_per_hk, num_heads_k, kHeadSizeV), bf16
//   softmax_lse:             (batch, num_heads_k, q_seq_per_hk), float32
//
// Caller must reshape its (batch, seqlen_q_ori, num_heads_q, kHeadSizeK)
// query tensor to (batch, q_seq_per_hk, num_heads_k, kHeadSizeK) where
// q_seq_per_hk = seqlen_q_ori * (num_heads_q / num_heads_k); the kernel
// expects num_heads_k as the head axis after that reshape. The same
// inversion is applied to `out` on the way back.
void flashmla_dense_fp8_fwd(
    TensorView q,
    TensorView k_cache,
    TensorView seqlens_k,
    TensorView block_table,
    TensorView tile_scheduler_metadata,
    TensorView num_splits,
    TensorView descale_q,
    TensorView descale_k,
    TensorView out,
    TensorView softmax_lse,
    int64_t num_heads_q,
    double softmax_scale,
    bool is_causal) {
  CHECK_CUDA(q);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(seqlens_k);
  CHECK_CUDA(block_table);
  CHECK_CUDA(tile_scheduler_metadata);
  CHECK_CUDA(num_splits);
  CHECK_CUDA(descale_q);
  CHECK_CUDA(descale_k);
  CHECK_CUDA(out);
  CHECK_CUDA(softmax_lse);

  TVM_FFI_ICHECK(q.dtype() == dl_float8_e4m3fn) << "q must be float8_e4m3fn";
  TVM_FFI_ICHECK(k_cache.dtype() == dl_float8_e4m3fn) << "k_cache must be float8_e4m3fn";
  TVM_FFI_ICHECK(seqlens_k.dtype() == dl_int32) << "seqlens_k must be int32";
  TVM_FFI_ICHECK(block_table.dtype() == dl_int32) << "block_table must be int32";
  TVM_FFI_ICHECK(tile_scheduler_metadata.dtype() == dl_int32)
      << "tile_scheduler_metadata must be int32";
  TVM_FFI_ICHECK(num_splits.dtype() == dl_int32) << "num_splits must be int32";
  TVM_FFI_ICHECK(descale_q.dtype() == dl_float32) << "descale_q must be float32";
  TVM_FFI_ICHECK(descale_k.dtype() == dl_float32) << "descale_k must be float32";
  TVM_FFI_ICHECK(out.dtype() == dl_bfloat16) << "out must be bfloat16";
  TVM_FFI_ICHECK(softmax_lse.dtype() == dl_float32) << "softmax_lse must be float32";

  CHECK_DIM(4, q);
  CHECK_DIM(4, k_cache);
  CHECK_DIM(1, seqlens_k);
  CHECK_DIM(2, block_table);
  CHECK_DIM(2, tile_scheduler_metadata);
  CHECK_DIM(1, num_splits);
  CHECK_DIM(1, descale_q);
  CHECK_DIM(1, descale_k);
  CHECK_DIM(4, out);
  CHECK_DIM(3, softmax_lse);

  TVM_FFI_ICHECK(q.stride(-1) == 1) << "q must have contiguous last dim";
  TVM_FFI_ICHECK(k_cache.stride(-1) == 1) << "k_cache must have contiguous last dim";
  TVM_FFI_ICHECK(seqlens_k.IsContiguous()) << "seqlens_k must be contiguous";
  TVM_FFI_ICHECK(block_table.stride(-1) == 1) << "block_table must have contiguous last dim";
  TVM_FFI_ICHECK(tile_scheduler_metadata.IsContiguous())
      << "tile_scheduler_metadata must be contiguous";
  TVM_FFI_ICHECK(num_splits.IsContiguous()) << "num_splits must be contiguous";
  TVM_FFI_ICHECK(descale_q.IsContiguous()) << "descale_q must be contiguous";
  TVM_FFI_ICHECK(descale_k.IsContiguous()) << "descale_k must be contiguous";

  const int64_t batch_size = q.size(0);
  const int64_t q_seq_per_hk = q.size(1);
  const int64_t num_heads_k = q.size(2);
  TVM_FFI_ICHECK(q.size(3) == kHeadSizeK)
      << "q last dim must equal kHeadSizeK=" << kHeadSizeK;

  TVM_FFI_ICHECK(k_cache.size(1) == kPageBlockSize)
      << "page_block_size must be " << kPageBlockSize;
  TVM_FFI_ICHECK(k_cache.size(2) == num_heads_k)
      << "k_cache num_heads_k mismatch with q";
  TVM_FFI_ICHECK(k_cache.size(3) == kHeadSizeK)
      << "k_cache last dim must equal kHeadSizeK=" << kHeadSizeK;

  const int64_t num_blocks = k_cache.size(0);
  const int64_t max_num_blocks_per_seq = block_table.size(1);
  TVM_FFI_ICHECK(block_table.size(0) == batch_size)
      << "block_table batch mismatch";
  TVM_FFI_ICHECK(seqlens_k.size(0) == batch_size) << "seqlens_k batch mismatch";
  TVM_FFI_ICHECK(num_splits.size(0) == batch_size + 1)
      << "num_splits length must be batch_size + 1";
  TVM_FFI_ICHECK(descale_q.size(0) == 1) << "descale_q must be shape (1,)";
  TVM_FFI_ICHECK(descale_k.size(0) == 1) << "descale_k must be shape (1,)";
  TVM_FFI_ICHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize)
      << "tile_scheduler_metadata inner dim must be " << TileSchedulerMetaDataSize;

  TVM_FFI_ICHECK(out.size(0) == batch_size) << "out batch mismatch";
  TVM_FFI_ICHECK(out.size(1) == q_seq_per_hk) << "out q_seq_per_hk mismatch";
  TVM_FFI_ICHECK(out.size(2) == num_heads_k) << "out num_heads_k mismatch";
  TVM_FFI_ICHECK(out.size(3) == kHeadSizeV)
      << "out last dim must equal kHeadSizeV=" << kHeadSizeV;

  TVM_FFI_ICHECK(softmax_lse.size(0) == batch_size) << "softmax_lse batch mismatch";
  TVM_FFI_ICHECK(softmax_lse.size(1) == num_heads_k) << "softmax_lse num_heads_k mismatch";
  TVM_FFI_ICHECK(softmax_lse.size(2) == q_seq_per_hk) << "softmax_lse q_seq mismatch";

  TVM_FFI_ICHECK(num_heads_q > 0) << "num_heads_q must be positive";
  TVM_FFI_ICHECK(num_heads_q % num_heads_k == 0)
      << "num_heads_q must be divisible by num_heads_k";

  if (q_seq_per_hk == num_heads_q / num_heads_k) {
    // seqlen_q_ori == 1 path: causal mask is a no-op, force false.
    is_causal = false;
  }

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  const int num_sm_parts = static_cast<int>(tile_scheduler_metadata.size(0));
  const int total_num_splits = static_cast<int>(batch_size) + num_sm_parts;

  // Internal accumulators: caller does not need to manage these.
  auto softmax_lse_accum = alloc_tensor(
      {total_num_splits, static_cast<int64_t>(num_heads_k), q_seq_per_hk},
      dl_float32,
      q.device());
  auto out_accum = alloc_tensor(
      {total_num_splits, static_cast<int64_t>(num_heads_k), q_seq_per_hk, kHeadSizeV},
      dl_float32,
      q.device());

  DecodingParams_fp8 params = {};
  params.b = static_cast<int>(batch_size);
  // s_q in the kernel is seqlen_q_ori (pre-fold seqlen). Recover it from
  // q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk.
  const int num_q_heads_per_hk = static_cast<int>(num_heads_q / num_heads_k);
  TVM_FFI_ICHECK(q_seq_per_hk % num_q_heads_per_hk == 0)
      << "q_seq_per_hk must be a multiple of num_heads_q/num_heads_k";
  params.s_q = static_cast<int>(q_seq_per_hk / num_q_heads_per_hk);
  params.q_seq_per_hk = static_cast<int>(q_seq_per_hk);
  params.d = kHeadSizeK;
  params.d_v = kHeadSizeV;
  params.h_q = static_cast<int>(num_heads_q);
  params.h_k = static_cast<int>(num_heads_k);
  params.num_blocks = static_cast<int>(num_blocks);
  params.q_head_per_hk = num_q_heads_per_hk;
  params.is_causal = is_causal;
  params.scale_softmax = static_cast<float>(softmax_scale);
  params.scale_softmax_log2 = static_cast<float>(softmax_scale) * float(M_LOG2E);
  params.topk = -1;  // Dense attention.

  params.h_h_k_ratio = 1;
  params.descale_q_ptr = static_cast<float*>(descale_q.data_ptr());
  params.descale_k_ptr = static_cast<float*>(descale_k.data_ptr());

  params.q_ptr = q.data_ptr();
  params.k_ptr = k_cache.data_ptr();
  params.o_ptr = out.data_ptr();
  params.indices_ptr = nullptr;
  params.softmax_lse_ptr = softmax_lse.data_ptr();

  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k_cache.stride(0);
  params.o_batch_stride = out.stride(0);
  params.q_row_stride = q.stride(1);
  params.k_row_stride = k_cache.stride(1);
  params.o_row_stride = out.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_head_stride = k_cache.stride(2);
  params.o_head_stride = out.stride(2);
  params.indices_batch_stride = 0;
  params.indices_row_stride = 0;

  params.block_table = static_cast<int*>(block_table.data_ptr());
  params.block_table_batch_stride = block_table.stride(0);
  params.page_block_size = kPageBlockSize;
  params.seqlens_k_ptr = static_cast<int*>(seqlens_k.data_ptr());
  params.tile_scheduler_metadata_ptr =
      static_cast<int*>(tile_scheduler_metadata.data_ptr());
  params.num_sm_parts = num_sm_parts;
  params.num_splits_ptr = static_cast<int*>(num_splits.data_ptr());
  params.total_num_splits = total_num_splits;
  params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
  params.oaccum_ptr = out_accum.data_ptr();

  run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, kHeadSizeK>(
      params, stream);

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "flashmla_dense_fp8_fwd kernel launch failed: "
      << cudaGetErrorString(status);
}

// Build tile-scheduler metadata + num_splits for dense FP8 MLA decode.
//
// Shapes (caller pre-arranged):
//   seqlens_k:               (batch,), int32
//   tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), int32 [OUT]
//   num_splits:              (batch + 1,), int32                              [OUT]
//
// Caller computes num_sm_parts as:
//   sm_count / num_heads_k / ceil_div(num_heads_per_head_k, kMetaBlockSizeM)
void flashmla_dense_fp8_metadata(
    TensorView seqlens_k,
    TensorView tile_scheduler_metadata,
    TensorView num_splits) {
  CHECK_CUDA(seqlens_k);
  CHECK_CUDA(tile_scheduler_metadata);
  CHECK_CUDA(num_splits);

  TVM_FFI_ICHECK(seqlens_k.dtype() == dl_int32) << "seqlens_k must be int32";
  TVM_FFI_ICHECK(tile_scheduler_metadata.dtype() == dl_int32)
      << "tile_scheduler_metadata must be int32";
  TVM_FFI_ICHECK(num_splits.dtype() == dl_int32) << "num_splits must be int32";

  CHECK_DIM(1, seqlens_k);
  CHECK_DIM(2, tile_scheduler_metadata);
  CHECK_DIM(1, num_splits);

  TVM_FFI_ICHECK(seqlens_k.IsContiguous()) << "seqlens_k must be contiguous";
  TVM_FFI_ICHECK(tile_scheduler_metadata.IsContiguous())
      << "tile_scheduler_metadata must be contiguous";
  TVM_FFI_ICHECK(num_splits.IsContiguous()) << "num_splits must be contiguous";

  TVM_FFI_ICHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize)
      << "tile_scheduler_metadata inner dim must be " << TileSchedulerMetaDataSize;

  const int batch_size = static_cast<int>(seqlens_k.size(0));
  TVM_FFI_ICHECK(num_splits.size(0) == batch_size + 1)
      << "num_splits length must be batch_size + 1";

  cudaSetDevice(seqlens_k.device().device_id);
  const cudaStream_t stream = get_stream(seqlens_k.device());

  Mla_metadata_params params = {};
  params.seqlens_k_ptr = static_cast<int*>(seqlens_k.data_ptr());
  params.tile_scheduler_metadata_ptr =
      static_cast<int*>(tile_scheduler_metadata.data_ptr());
  params.num_splits_ptr = static_cast<int*>(num_splits.data_ptr());
  params.batch_size = batch_size;
  params.block_size_n = kMetaBlockSizeN;
  params.fixed_overhead_num_blocks = kMetaFixedOverheadNumBlocks;
  params.num_sm_parts = static_cast<int>(tile_scheduler_metadata.size(0));

  get_mla_metadata_func(params, stream);

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "flashmla_dense_fp8_metadata kernel launch failed: "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashmla_dense_fp8_fwd, flashmla_dense_fp8_fwd);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashmla_dense_fp8_metadata, flashmla_dense_fp8_metadata);
