/***************************************************************************************************
 * Declarations of the explicit (de-abstracted) Blackwell FP16 GEMM of example 70
 * (Part E.2.2 of examples/70_blackwell_gemm/Semantics-preserving-de-abstraction.md).
 *
 * This header is included by both 70_blackwell_fp16_gemm.cu (the harness) and
 * 70_blackwell_fp16_gemm_explicit.cu (the kernel).  It holds
 *   - ExplicitGemmParams: the plain-old-data parameter block of the explicit kernel (the ABI between the
 *     two translation units), passed __grid_constant__;
 *   - the declaration of explicit_gemm::run(ExplicitGemmParams const&), defined in the .cu;
 *   - the template bridge make_params(gemm) / run(gemm) that copies the driver-encoded tensor maps, the
 *     problem shape, alpha and beta out of gemm.params() after the unchanged gemm.initialize().
 * Rules (E.2.2): no __device__/__global__ code, no CUTLASS/CuTe include other than cutlass/cutlass.h and
 * cute/container/tuple.hpp, no non-inline non-template definition (no -rdc: both TUs include this file).
 *
 * Legend (as in the .cu): [PROD] production, [HOST] host-side, [CHECK] compile-time verification, [TRACE] toggle-on only.
 * Everything in this header is [HOST] [PROD] except the two [TRACE] declarations under CUTLASS_DEABSTRACTION_TRACE.
 *
 * Byte map of ExplicitGemmParams (CUDA 13.3: alignof(CUtensorMap) = 128, sizeof = 640; CUDA 12.x: 64 and 576; the [CHECK] asserts below
 * pin every offset the device reads, so both are valid):
 *     0  tma_a   128 B   CUtensorMap of A      (device: prefetch + TMA loads, warp 2)
 *   128  tma_b   128 B   CUtensorMap of B      (device: prefetch + TMA loads, warp 2)
 *   256  tma_c   128 B   CUtensorMap of C      (device: prefetch only; beta = 0 means C is never read)
 *   384  tma_d   128 B   CUtensorMap of D      (device: TMA stores, warp 4)
 *   512  m, n    516 n   520  k   524  l       (device reads only k: ld.param.b32 [+520]; k_tiles = (k + 63) >> 6 = 128)
 *   528  alpha   532  beta                     (device: ld.param.v2.b32 [+528]; the epilogue's FMUL / FFMA operands)
 *   536  mode  540  raster_order  544  swizzle_divisor  548  tiles_m  552  tiles_n  556  tiles_l   (host guards only)
 *   560  alpha_ptr  568  beta_ptr              (host guards only: must be null)
 *   576  end of the fields; padded to 640 (or 576) by the CUtensorMap alignment. In SASS the block starts at c[0x0][0x380],
 *        so k is c[0x0][0x588], alpha/beta c[0x0][0x590]/[0x594] and CONSTANT[0] = 0x380 + 640 = 1536 (G.3).
 **************************************************************************************************/
#pragma once

#include <cuda.h>                        // CUtensorMap
#include <cstddef>                       // offsetof
#include <cstdint>
#include <utility>                       // std::declval

#include "cutlass/cutlass.h"             // cutlass::Status
#include "cute/container/tuple.hpp"      // cute::get<i>(p.problem_shape)

#include "70_blackwell_fp16_gemm_explicit_util.hpp"

namespace explicit_gemm {

// [PROD]  The kernel parameter block.  Passed by value as `const __grid_constant__`, so the device reads
// tma_a/tma_b/tma_d (and prefetches tma_c) from parameter space (constant bank 0), k, alpha and beta once at
// entry (E.6.3 step 2).  The remaining fields are host-only validation data for the guards of E.5.1 and are
// never read on the device.  Field order: maps first (64/128-byte aligned), then the scalars.
struct ExplicitGemmParams {
  CUtensorMap tma_a;        // A, encoded by gemm.initialize(): FLOAT16, (K, M, L), box (64, 64, 1), SWIZZLE_128B (Section 5.3)
  CUtensorMap tma_b;        // B, likewise (K, N, L)
  CUtensorMap tma_c;        // C, FLOAT32, (M, N, L), box (32, 16, 1), SWIZZLE_128B_ATOM_32B; prefetched only (beta = 0)
  CUtensorMap tma_d;        // D, likewise; the store descriptor
  int   m, n, k, l;         // problem shape from p.problem_shape; the device reads only k (k_tiles = (k + 63) >> 6)
  float alpha, beta;        // run-time scalars (Section 0): p.epilogue.thread.op_2.op_0.scalars[0], p.epilogue.thread.op_0.scalars[0]
  // ---- host-only validation fields (E.5.1); never read on the device ----
  int   mode;               // int(p.mode), must be kModeGemm (0)
  int   raster_order;       // int(p.scheduler.raster_order_), must be kRasterOrderAlongN (1)
  int   swizzle_divisor;    // p.scheduler.divmod_swizzle_size_.divisor, must be 0 (swizzle disabled)
  unsigned tiles_m, tiles_n, tiles_l;   // p.scheduler.problem_tiles_m_/n_/l_, must be 32, 32, 1
  float const* alpha_ptr;   // p.epilogue.thread.op_2.op_0.scalar_ptrs[0], must be nullptr (no pointer path in the explicit kernel)
  float const* beta_ptr;    // p.epilogue.thread.op_0.scalar_ptrs[0], must be nullptr
};

// [CHECK] the ABI between the two translation units: offsets the device reads, and the two admissible sizes
static_assert(alignof(CUtensorMap) >= 64, "CUtensorMap must be at least 64-byte aligned (E.9 item 15: 64 in CUDA 12.x headers, 128 in CUDA 13 headers)");
static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap is 128 bytes (copy_sm90_desc.hpp:292)");
static_assert(offsetof(ExplicitGemmParams, tma_a) == 0 && offsetof(ExplicitGemmParams, tma_b) == 1 * sizeof(CUtensorMap) &&
              offsetof(ExplicitGemmParams, tma_c) == 2 * sizeof(CUtensorMap) && offsetof(ExplicitGemmParams, tma_d) == 3 * sizeof(CUtensorMap),
              "the four maps are contiguous at the front of the parameter block");
static_assert(offsetof(ExplicitGemmParams, m) == 4 * sizeof(CUtensorMap) && offsetof(ExplicitGemmParams, k) == 4 * sizeof(CUtensorMap) + 8 &&
              offsetof(ExplicitGemmParams, alpha) == 4 * sizeof(CUtensorMap) + 16 && offsetof(ExplicitGemmParams, beta) == 4 * sizeof(CUtensorMap) + 20,
              "scalar field offsets");
static_assert(sizeof(ExplicitGemmParams) == 576 || sizeof(ExplicitGemmParams) == 640,
              "expected 4 x 128 + 12 x 4 + 2 x 8 = 576 bytes (640 if alignof(CUtensorMap) is 128); E.2.2 [confirm at first build]");
static_assert(sizeof(ExplicitGemmParams) <= 4096, "kernel parameter block limit");

// [HOST] [PROD]  The host function (E.5): mirrors GemmUniversalAdapter::run() for the explicit kernel.  Defined in
// 70_blackwell_fp16_gemm_explicit.cu.  Declared before the template so that the unqualified call below finds it.
cutlass::Status run(ExplicitGemmParams const& params);

#if defined(CUTLASS_DEABSTRACTION_TRACE)
// [HOST] [TRACE]  Toggle-on host entry points (E.7.3), defined in 70_blackwell_fp16_gemm_explicit.cu and called only by the harness's
// guarded blocks: trace_begin resets/arms the kernel TU's record buffer, runs the K0 probe kernel and sentinel-fills D
// (E.7.2 B3); trace_end runs H3 on the explicit kernel, dumps the records, disables recording before the timed launches
// and prints the B4 mismatch dump if the compare failed.  Declarations only (no definition in this shared header, E.2.2).
void trace_begin(void* d_ptr, size_t d_bytes);
void trace_end(char const* csv_path, bool passed, float const* d_ptr, float const* ref_d_ptr, int m, int n);
#endif

// [HOST] [PROD]  Bridge: read the CUTLASS Params object that gemm.initialize() built (public accessor
// GemmUniversalAdapter::params(), gemm_universal_adapter.h:226-228) into the explicit parameter block.
// Instantiated only in the harness TU, where Gemm and every CUTLASS header are visible.  Member paths: E.1.1.
// This is the only place where CUTLASS types are touched on the explicit path: the six tensor maps were encoded by
// gemm.initialize() (cuTensorMapEncodeTiled, Section 5.3) and are copied here as 128-byte opaque blobs; the four
// integers of the problem shape and the two scalars are copied out of the epilogue's fusion tree. Expected values for the
// fixed run: m = n = k = 8192, l = 1, alpha = 1.0f, beta = 0.0f, mode 0 (kGemm), raster_order 1 (AlongN), swizzle_divisor 0,
// tiles (32, 32, 1), both pointers null (E.5.1 rejects anything else).
template <class Gemm, class = decltype(std::declval<Gemm const&>().params())>
ExplicitGemmParams make_params(Gemm const& gemm) {
  auto const& p = gemm.params();
  ExplicitGemmParams e{};
  e.tma_a = p.mainloop.tma_load_a.tma_desc_;            // Copy_Traits<SM100_TMA_2SM_LOAD_MULTICAST,...>::tma_desc_ (public)
  e.tma_b = p.mainloop.tma_load_b.tma_desc_;            // Copy_Traits<SM100_TMA_2SM_LOAD,...>::tma_desc_
  e.tma_c = p.epilogue.tma_load_c.tma_desc_;            // Copy_Traits<SM90_TMA_LOAD,...>::tma_desc_
  e.tma_d = p.epilogue.tma_store_d.tma_desc_;           // Copy_Traits<SM90_TMA_STORE,...>::tma_desc_
  e.m = cute::get<0>(p.problem_shape);
  e.n = cute::get<1>(p.problem_shape);
  e.k = cute::get<2>(p.problem_shape);
  e.l = cute::get<3>(p.problem_shape);
  // Sm90LinearCombination tree: op_0 = beta broadcast, op_1 = C fetch, op_2 = (alpha * acc) tree whose op_0 is the
  // alpha broadcast, op_3 = multiply_add (sm90_callbacks_tma_warpspecialized.hpp:182-213,
  // sm90_visitor_tma_warpspecialized.hpp:711-714, 1192-1197; Sm90ScalarBroadcast::Params = Arguments
  // {scalars[1], scalar_ptrs[1], dScalar[1]}, sm90_visitor_load_tma_warpspecialized.hpp:1017-1023).
  e.beta      = p.epilogue.thread.op_0.scalars[0];
  e.beta_ptr  = p.epilogue.thread.op_0.scalar_ptrs[0];
  e.alpha     = p.epilogue.thread.op_2.op_0.scalars[0];
  e.alpha_ptr = p.epilogue.thread.op_2.op_0.scalar_ptrs[0];
  e.mode            = static_cast<int>(p.mode);
  e.raster_order    = static_cast<int>(p.scheduler.raster_order_);
  e.swizzle_divisor = static_cast<int>(p.scheduler.divmod_swizzle_size_.divisor);
  e.tiles_m = p.scheduler.problem_tiles_m_;
  e.tiles_n = p.scheduler.problem_tiles_n_;
  e.tiles_l = p.scheduler.problem_tiles_l_;
  return e;
}

// [HOST] [PROD]  What the harness calls in place of gemm.run() (E.4): the same params_ snapshot that run() would have used.
template <class Gemm, class = decltype(std::declval<Gemm const&>().params())>
cutlass::Status run(Gemm const& gemm) {
  return run(make_params(gemm));
}

} // namespace explicit_gemm
