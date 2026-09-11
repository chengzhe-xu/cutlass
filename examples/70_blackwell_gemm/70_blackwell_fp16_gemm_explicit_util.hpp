// #define CUTLASS_DEABSTRACTION_TRACE 1  // De-abstraction trace toggle (Part C / E.2.3 of Semantics-preserving-de-abstraction.md): uncomment, rebuild with the unchanged cmake command, run the fixed command once; re-comment for the performance binary. Both .cu files include this header first, so the toggle is seen identically by both translation units.
/***************************************************************************************************
 * Host-side helpers for the explicit (de-abstracted) Blackwell FP16 GEMM of example 70
 * (Part E.2.3 of examples/70_blackwell_gemm/Semantics-preserving-de-abstraction.md).
 *
 * Rules (E.2.3): line 1 is the trace toggle; this header includes only <cuda_runtime.h>, <cstdio>, <cstdint> and
 * cutlass/cutlass.h (for cutlass::Status); it holds no device code, no CuTe/CUTLASS kernel types, and it
 * must not redefine CUDA_CHECK / CUTLASS_CHECK (examples/common/helper.h defines them for the harness).
 * Every helper is `inline` or a macro because both translation units include this header.
 *
 * Legend (as in the .cu): [PROD] production, [HOST] host-side, [CHECK] compile-time verification, [TRACE] toggle-on only.
 * The toggle on line 1 selects between the two binaries the note distinguishes: commented out = the performance binary
 * (Part H.1); uncommented = the diagnostics binary that records TRACE_* lines and launch0.csv (Part H.2.1). Both .cu files
 * include this header first, so one edit switches both translation units; deabstraction_trace_run.sh flips it with sed.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#include "cutlass/cutlass.h"

namespace explicit_gemm {

// ---- [PROD] [CHECK]  Section 0 constants that need no CUDA type (checked here so a typo is a build error in both TUs).
//      The fixed run in numbers: 8192^3 problem, 128 x 128 CTA tiles, 64-wide k-tiles (128 per tile), 2 x 2 clusters, grid 64 x 64
//      (x indexes N, y indexes M after the AlongN transpose), 256 threads, 230400 B of dynamic shared memory, 32 x 32 cluster tiles.
constexpr int      kFixedM = 8192, kFixedN = 8192, kFixedK = 8192, kFixedL = 1;   // Section 0 shape
constexpr int      kCtaTileM = 128, kCtaTileN = 128, kTileK = 64;                  // Section 4 CTA tile / k-tile
constexpr int      kClusterX = 2, kClusterY = 2, kClusterZ = 1;                    // Section 2 cluster (2,2,1)
constexpr unsigned kGridX = kFixedN / kCtaTileN, kGridY = kFixedM / kCtaTileM, kGridZ = 1;   // (64,64,1) after the AlongN transpose (Section 7.4, B.1.2 step 2)
constexpr unsigned kBlockThreads = 256;                                             // Section 4
constexpr unsigned kSmemBytes = 230400;                                             // Section 6.2 SharedStorageSize
constexpr unsigned kProblemTilesM = kFixedM / 256, kProblemTilesN = kFixedN / 256, kProblemTilesL = 1;   // 32 x 32 cluster tiles (B.1.1 step 11)
constexpr int      kRasterOrderAlongN = 1;   // enum class RasterOrder { AlongM = 0, AlongN = 1 } (tile_scheduler_detail.hpp:38-41; measured AlongN 1, D.5 H4)
constexpr int      kModeGemm = 0;            // GemmUniversalMode::kGemm (gemm_enumerated_types.h:57-64)
constexpr uint64_t kSmemDescConst = 0x4000404000010000ull;   // D.7 UMMA smem descriptor constant part
constexpr uint32_t kInstrDescConst = 0x10200010u;            // D.7 instruction descriptor

static_assert(kGridX == 64 && kGridY == 64 && kGridZ == 1, "Section 0 launch grid");
static_assert(kGridX % kClusterX == 0 && kGridY % kClusterY == 0 && kGridZ % kClusterZ == 0, "grid divisible by the cluster (check_cluster_dims)");
static_assert(kClusterX * kClusterY * kClusterZ <= 32, "cluster size <= MaxClusterSize (cluster_launch.hpp:82)");
static_assert(kFixedK % kTileK == 0 && kFixedK / kTileK == 128, "128 k-tiles per tile (Section 7.4)");
static_assert(kProblemTilesM == 32 && kProblemTilesN == 32, "32 x 32 cluster tiles (Section 7.4)");
static_assert(kSmemBytes <= 232448, "opt-in dynamic shared memory of the B200 (Section 0)");

// ---- [HOST] [PROD]  Status mapping used by the host function (E.5.2): mirrors ClusterLauncher's Return_Status (cluster_launch.hpp),
//      cudaSuccess -> kSuccess, any CUDA error -> kInvalid.  run() in the .cu then collapses kInvalid (and a non-success
//      cudaGetLastError) to kErrorInternal, as GemmUniversalAdapter::run() does (gemm_universal_adapter.h:564-574; F.2 item 11).
inline cutlass::Status status_from_cuda(cudaError_t e) {
  return e == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kInvalid;
}

// ---- [TRACE]  Trace-only host printing (compiles to nothing with the toggle off).  Currently unused: the toggle-on prints of the .cu
//      sit inside #if defined(CUTLASS_DEABSTRACTION_TRACE) blocks and call std::printf directly (F.2 item 12). ----
#if defined(CUTLASS_DEABSTRACTION_TRACE)
#define EXPLICIT_GEMM_TRACE_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define EXPLICIT_GEMM_TRACE_PRINTF(...) do { } while (0)
#endif

} // namespace explicit_gemm
