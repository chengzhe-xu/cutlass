#include "70_blackwell_fp16_gemm_explicit_util.hpp"   // line 1 of that header is the trace toggle (E.2.3); must be the first include
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
#include "deabstraction_trace.hpp"                     // record buffer + TRACE_RECORD (toggle-on only; before any CUTLASS header with a probe block)
#endif
/***************************************************************************************************
 * Explicit (de-abstracted) Blackwell FP16 GEMM of example 70.
 *
 * One __global__ kernel that re-implements, without any CUTLASS/CuTe abstraction, the GEMM that
 * examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu builds through CollectiveBuilder / GemmUniversal for the
 * fixed run  --m=8192 --n=8192 --k=8192  (alpha = 1, beta = 0, cluster (2,2,1)), plus the host function
 * explicit_gemm::run() that launches it exactly as GemmUniversalAdapter::run() launches the CUTLASS kernel.
 *
 * The knowledge base is examples/70_blackwell_gemm/Semantics-preserving-de-abstraction.md ("the note"):
 *   Section 6.2  shared-memory layout        Section 7   protocol (barriers, roles, CLC, TMA, UMMA, epilogue)
 *   Part B       PTX and coordinate formulas Part D      measured values (descriptors, masks, SASS inventory)
 *   Part E       this implementation's plan  (E.5 host function, E.6 kernel architecture, E.7 bring-up)
 * Every asm string below is copied from the CUTLASS source named next to it (commit cb424739); CuTe is used
 * only at compile time (cute::Swizzle cross-check of the epilogue smem formula); inline PTX covers tcgen05.*,
 * cp.async.bulk.tensor.*, mbarrier.*, fence.*, clusterlaunchcontrol.*, griddepcontrol.*, mapa, elect.sync,
 * bar.sync/arrive, barrier.cluster.* and prefetch.tensormap (Section 1 items 3-4).
 *
 * Address-space rule (E.6.1): every asm operand that names shared memory is the 32-bit shared::cta address
 * `smem_base + OFF_*`; every C++ dereference of shared memory uses the generic pointer `smem + OFF_*`.
 **************************************************************************************************/
/***************************************************************************************************
 * HOW TO READ THIS FILE (legend used in every section banner and in trailing comments)
 *
 *   [PROD]  production path: compiled into the performance binary (line 1 of the util header commented out).
 *   [TRACE] diagnostics only: every such block sits inside #if defined(CUTLASS_DEABSTRACTION_TRACE), is compiled only
 *           when the trace toggle is on (note, Part H.2.1) and emits no instruction otherwise; the SASS of two toggle-off
 *           builds is identical (E.8 D6, G.3). Nothing in a [TRACE] block changes what the kernel computes.
 *   [CHECK] compile-time verification: static_assert and constexpr cross-checks (incl. the CuTe swizzle check); no code.
 *   [HOST]  host-side code: the launch mirror and the trace entry points.
 *   Hardware families named in the banners and wrapper tags: tcgen05 (TMEM alloc/dealloc, MMA, commit, ld), TMA
 *   (cp.async.bulk.tensor loads/stores, prefetch.tensormap), mbarrier (init/arrive/expect_tx/try_wait), CLC
 *   (clusterlaunchcontrol), GDC (griddepcontrol), LAYOUT (CuTe layout, coordinate and byte-offset arithmetic written out).
 *
 * FILE MAP (section numbers appear in the banners below)
 *    1. configuration constants, Section 6.2 shared-memory map, descriptor and protocol constants   [PROD] [CHECK] LAYOUT
 *    2. trace-only types: hang records, heartbeats, K_TAIL, EXPLICIT_WAIT                            [TRACE]
 *    3. PTX wrappers W1-W37, one asm string each, copied from CUTLASS                                [PROD] tcgen05 TMA mbarrier CLC GDC
 *    4. pipeline-state and descriptor helpers; trace-only bounded wait and heartbeat                 [PROD] [TRACE]
 *    5. clc_consume: the CLC response consumer shared by warps 0, 1, 2, 4-7                           [PROD] CLC LAYOUT
 *    6. producer_role (warp 2): TMA loads of A and B into the 8-stage smem ring                       [PROD] TMA mbarrier
 *    7. scheduler_role (warp 1 of rank 0): CLC queries and the throttle                               [PROD] CLC mbarrier
 *    8. mma_role (warp 0): TMEM allocation, 4 UMMAs per k-tile, commits, deallocation                 [PROD] tcgen05
 *    9. epilogue_role (warps 4-7): tcgen05.ld, alpha/beta, swizzled smem stores, TMA stores of D      [PROD] tcgen05 TMA LAYOUT
 *   10. the __global__ kernel: prologue (barrier inits, cluster rendezvous, initial tile), dispatch    [PROD]
 *   11. run(): the host launch mirror of GemmUniversalAdapter::run()                                  [HOST] [PROD]
 *   12. trace_begin / trace_end / mismatch dump                                                       [HOST] [TRACE]
 *
 * NUMBERS AT A GLANCE (fixed run; Section 0, D.7, and TRACE_H1/TRACE_ENCODE of trace_out_v2/host.txt)
 *   problem      M = N = K = 8192, fp16 A (M x K, K-major) and B (N x K, K-major), fp32 D (M x N, M-major), alpha = 1, beta = 0, L = 1
 *   cluster      (2,2,1): rank = x + 2y with x = %cluster_ctaid.x, y = %cluster_ctaid.y; the MMA pair is {2y, 2y+1} (same y);
 *                the even rank is the pair leader that issues the MMAs for both CTAs; CTAs x and x+2 share the A rows (multicast)
 *   grid         (64,64,1) CTAs; 32 x 32 = 1024 cluster tiles of 256 x 256 handed out dynamically by CLC to 33 resident clusters
 *   tile         CTA tile 128 (M) x 128 (N) x 64 (K) per k-tile; 128 k-tiles per tile; one tcgen05.mma.cta_group::2 covers
 *                M 256 (both CTAs' A) x N 128 (both CTAs' B halves) x K 16, so 4 MMAs per k-tile
 *   smem         230400 B dynamic, base 0x400 + (rank << 24) (measured, D.1); barriers 0..391, C/D 512 + 4 x 8192,
 *                A 33792 + 8 x 16384 (Swizzle<3,4,3>), B 164864 + 8 x 8192 (Swizzle<3,4,3>)
 *   TMA boxes    A and B: (64 k, 64 rows, 1) = 8192 B, SWIZZLE_128B; D: (32 m, 16 n, 1) = 2048 B, SWIZZLE_128B_ATOM_32B
 *   per stage    each CTA receives the whole 128 x 64 A stage (its own 64-row half + the other cluster row's half by multicast)
 *                and its own 64 x 64 B half; the pair leader's full[s] barrier expects 2 x (16384 + 8192) = 49152 bytes
 *   TMEM         512 columns allocated, base 0 (measured); accumulator stage st = columns [128 st, 128 st + 128) of 128 lanes (rows)
 *   descriptors  A 0x4000404000010880 + 1024 s + 2 kb, B 0x4000404000012880 + 512 s + 2 kb (low word in 16-byte units); idesc 0x10200010
 *   barriers     mainloop full 1 (+49152 tx) / empty 2; CLC full 1 (+16 tx) / empty 800; acc full 1 / empty 256; throttle 32 / 32; tmem_dealloc 32
 *   masks        A multicast 0x5 << x (ranks x, x+2); mainloop release 0xF (all four); accumulator commit 0x3 << 2y (the pair);
 *                the peer bit (bit 24) cleared on a shared::cluster address names the pair leader's barrier
 *   epilogue     8 subtiles of 128 x 16 per tile; tcgen05.ld.32x32b.x16 gives lane l the 16 columns 16e..16e+15 of TMEM row 32w + l;
 *                Swizzle<2,5,2> on the 8192-byte C/D stage; four (32, 16) TMA store boxes per subtile
 **************************************************************************************************/
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"                 // cutlass::Status only
#include "cute/swizzle.hpp"                  // compile-time cross-check of the epilogue smem swizzle only (Section 1 item 4)

#include "70_blackwell_fp16_gemm_explicit.hpp"

#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
#include "deabstraction_trace_probes.hpp"    // K0 probe kernel and trace_host_post_run_fn (toggle-on only; pulls CUTLASS headers, E.7.3)
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <vector>
#endif

namespace explicit_gemm {

namespace {

///////////////////////////////////////////////////////////////////////////////////////////////////
// 1. [PROD] [CHECK] LAYOUT   E.6.1 configuration constants, Section 6.2 offsets, descriptor and protocol constants
//    (kFixedM/N/K/L, kCtaTile*, kTileK, kCluster*, kGrid*, kBlockThreads, kSmemBytes and the D.7 descriptor words come from
//     70_blackwell_fp16_gemm_explicit_util.hpp; everything here is a compile-time constant, the kernel reads only k, alpha, beta
//     and the tensor maps at run time)
///////////////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t kThreads       = kBlockThreads;   // 8 warps: 0 MMA, 1 Sched (rank 0), 2 MainloopLoad, 3 EpilogueLoad (exits), 4-7 Epilogue
constexpr int      kMmaM          = 256, kMmaN = 128, kMmaK = 16;   // one tcgen05.mma.cta_group::2 per k-block
constexpr int      kKBlocks       = kTileK / kMmaK;                 // 4 MMAs per k-tile
constexpr int      kStages        = 8;               // A/B smem pipeline (Section 4)
constexpr int      kAccStages     = 4;               // TMEM accumulator pipeline, 128 columns each
constexpr int      kClcStages     = 2;               // CLC response slots
constexpr int      kThrottleStages= 2;
constexpr int      kEpiStages     = 4;               // 8192-byte C/D smem stages (reused, ReuseSmemC)
constexpr int      kEpiSubtiles   = 8;               // 128x16 subtiles per 128x128 tile
constexpr int      kEpiTileN      = 16;
constexpr uint32_t kTmemColumns   = 512;             // tcgen05.alloc / dealloc column count
constexpr int      kPrologueKTiles= kStages;         // first load() call covers min(8, k_tiles) k-tiles (kernel :629)
constexpr uint32_t kStageBytesA   = 128 * 64 * 2;    // 16384
constexpr uint32_t kStageBytesAHalf = kStageBytesA / 2;   // 8192: one CTA's 64-row half of the 128x64 A stage (B.4); equal to kStageBytesB by coincidence of the tile shape
constexpr uint32_t kStageBytesB   = 64 * 64 * 2;     // 8192
constexpr uint32_t kEpiStageBytes = 128 * 16 * 4;    // 8192
constexpr uint32_t kStoreBoxBytes = 32 * 16 * 4;     // 2048: the D map's box is (32 M, 16 N, 1) (Section 5.3)

// LAYOUT  What the CUTLASS kernel's CuTe types say about the same buffers (TRACE_H1 lines of trace_out_v2/host.txt), for reference;
// this kernel never computes these layouts: the TMA unit writes them (SWIZZLE_128B) and the UMMA unit reads them through the descriptor.
//   SmemLayoutA  = Sw<3,4,3> o ((_128,_16),_1,_4,(_1,_8)) : ((_64,_1),_0,_16,(_0,_8192))   [fp16 elements]
//                  row m of a k-block holds 16 k values at m * 64 + k; the 4 k-blocks of a k-tile are 16 elements (32 B) apart;
//                  stage s is 8192 elements (16384 B) apart; Swizzle<3,4,3>: byte-address bits [4,7) ^= bits [7,10), i.e. the
//                  16-byte chunk inside a 128-byte row is XOR-ed with (row % 8). A row is 64 fp16 = 128 B (K-major, SW128 atom 8 x 64).
//   SmemLayoutB  = Sw<3,4,3> o ((_64,_16),_1,_4,(_1,_8)) : ((_64,_1),_0,_16,(_0,_4096))    [64 N rows per CTA, 8192 B per stage]
//   SmemLayoutAtomC/D = Sw<2,5,2> o (_32,_4) : (_1,_32)                                  [fp32; 32 M x 4 N atom = 512 B, see epilogue_role]
//   TiledMMA     = MMA_Atom SM100_MMA_F16BF16_2x1SM_SS 256x128x16, ThrLayoutVMNK (2,1,1,1); LayoutC_TV (_2,(_128,_128)):(_128,(_1,_256)):
//                  CTA v of the pair owns accumulator rows [128 v, 128 v + 128) of the 256 x 128 MMA tile, i.e. its own 128 x 128 CTA tile.
//   EpilogueTile = (128, 16); TileShape (256, 128, 64) per pair; CtaShape_MNK (128, 128, 64); ClusterShape (2, 2, 1).
//   Tensor maps (TRACE_ENCODE): A, B FLOAT16 dim 3 shape (8192, 8192, 1) stride bytes (16384, 0) box (64, 64, 1) SWIZZLE_128B L2 promotion 128 B;
//                C, D FLOAT32 shape (8192, 8192, 1) stride bytes (32768, 0) box (32, 16, 1) SWIZZLE_128B_ATOM_32B. Coordinates are
//                (innermost, outer, batch): (k, row, 0) for A/B and (m, n, 0) for C/D, in elements.

// LAYOUT  Section 6.2 byte offsets from the dynamic shared-memory base (== the CUTLASS SharedStorage layout, TRACE_HOST off_* lines).
// Shared::cta address of anything = smem_base + OFF_* with smem_base = 0x400 + (rank << 24); e.g. rank 2's acc_full[1] is 0x02000400 + 240 + 8.
constexpr uint32_t OFF_MAIN_FULL      =      0;   // 8 x 8 B ClusterTransactionBarrier, count 1
constexpr uint32_t OFF_MAIN_EMPTY     =     64;   // 8 x 8 B ClusterBarrier, count 2
constexpr uint32_t OFF_EPI_LOAD_FULL  =    128;   // 4 x 8 B (beta != 0 only; reserved, not initialised: E.10 item 6)
constexpr uint32_t OFF_EPI_LOAD_EMPTY =    160;   // 4 x 8 B (idem)
constexpr uint32_t OFF_LOAD_ORDER     =    192;   // 2 x 8 B (idem)
constexpr uint32_t OFF_CLC_FULL       =    208;   // 2 x 8 B, count 1 (+16 tx bytes)
constexpr uint32_t OFF_CLC_EMPTY      =    224;   // 2 x 8 B, count 800 (only CTA 0's are used)
constexpr uint32_t OFF_ACC_FULL       =    240;   // 4 x 8 B, count 1
constexpr uint32_t OFF_ACC_EMPTY      =    272;   // 4 x 8 B, count 256
constexpr uint32_t OFF_THROTTLE_FULL  =    304;   // 2 x 8 B, count 32
constexpr uint32_t OFF_THROTTLE_EMPTY =    320;   // 2 x 8 B, count 32
constexpr uint32_t OFF_TMEM_DEALLOC   =    336;   // 1 x 8 B, count 32
constexpr uint32_t OFF_CLC_RESPONSE   =    352;   // 2 x 16 B, alignas(16)
constexpr uint32_t OFF_TMEM_BASE_PTR  =    384;   // 4 B, written by tcgen05.alloc
constexpr uint32_t OFF_SMEM_CD        =    512;   // 4 x 8192 B, Swizzle<2,5,2>, 512-byte aligned
constexpr uint32_t OFF_SMEM_A         =  33792;   // 8 x 16384 B, Swizzle<3,4,3>, 1024-byte aligned
constexpr uint32_t OFF_SMEM_B         = 164864;   // 8 x  8192 B, idem
constexpr uint32_t kSmemBaseLow24     =  0x400;   // measured base (D.1 D5): 0x400 + (rank << 24); computed at run time, never assumed

static_assert(OFF_SMEM_A == OFF_SMEM_CD + kEpiStages * kEpiStageBytes + 512, "Section 6.2 (FusionStorage pad to 512)");
static_assert(OFF_SMEM_B == OFF_SMEM_A + kStages * kStageBytesA, "Section 6.2");
static_assert(kSmemBytes == OFF_SMEM_B + kStages * kStageBytesB, "Section 6.2: SharedStorageSize 230400");
static_assert(kStageBytesAHalf == 64u * 64u * 2u, "one CTA's 64x64 fp16 half of the 128x64 A stage (B.4): TMA box (64, 64, 1)");
static_assert(OFF_SMEM_A % 1024 == 0 && OFF_SMEM_B % 1024 == 0, "SW128 stage bases need 1 KiB alignment (Section 7.5)");
static_assert(OFF_SMEM_CD % 512 == 0, "SW128_32B epilogue stages need 512 B alignment (Section 7.7)");
static_assert(OFF_CLC_RESPONSE % 16 == 0 && OFF_TMEM_BASE_PTR % 4 == 0, "b128 response slots");
static_assert(kSmemBaseLow24 + kSmemBytes < (1u << 18), "14-bit UMMA start-address field spans 256 KiB (B.3.2 step 4)");
static_assert(kThreads == 256 && kMmaM == 256 && kMmaN == 128 && kKBlocks == 4, "Section 4");

// tcgen05 LAYOUT  UMMA shared-memory descriptors (Section 7.6, D.7): 64-bit = {hi word constant, lo word = LBO (1 = 16 B) | (start_address >> 4)
// in a 14-bit field}. Per stage s and k-block kb the low word is lo_a = 0x10880 + 1024 s + 2 kb, lo_b = 0x12880 + 512 s + 2 kb (units of 16 B:
// 1024 = 16384 B, 512 = 8192 B, 2 = 32 B = 16 fp16 k values). Example, rank 0, s = 3, kb = 2: desc_a = 0x4000404000011484, desc_b = 0x4000404000012E84.
constexpr uint32_t kSmemDescHi  = 0x40004040u;   // SBO = 64 (1024 B) at [32,46), version = 1 at [46,48), layout_type = 2 (SWIZZLE_128B) at [61,64)
constexpr uint32_t kSmemDescLo0 = 0x00010000u;   // LBO = 1 (16 B) at [16,30); start address (addr >> 4) & 0x3FFF is OR'ed in at run time
constexpr uint32_t kInstrDesc   = kInstrDescConst;   // 0x10200010: c_format F32 (bit 4), n_dim 128 >> 3 = 16 at [17,23), m_dim 256 >> 4 = 16 at [24,29)
static_assert(((uint64_t(kSmemDescHi) << 32) | kSmemDescLo0) == kSmemDescConst, "D.7 descriptor constant 0x4000404000010000");
static_assert((64u | (1u << 14) | (2u << 29)) == kSmemDescHi, "SmemDescriptor high word (mma_sm100_desc.hpp:101-127)");
static_assert(((1u << 4) | (16u << 17) | (16u << 24)) == kInstrDesc, "InstrDescriptor bit map (mma_sm100_desc.hpp:416-443)");
static_assert(((kSmemBaseLow24 + OFF_SMEM_A) >> 4) == 0x880 && ((kSmemBaseLow24 + OFF_SMEM_B) >> 4) == 0x2880, "D.7: desc_a low word 0x10880, desc_b 0x12880 for the measured base");

// mbarrier  Protocol constants (Section 7.2): arrive counts and transaction bytes of every barrier, and the multicast masks
constexpr uint32_t kMainloopTxBytes  = 2 * (kStageBytesA + kStageBytesB);   // 49152: six 8192-byte boxes per stage credit the leader
constexpr uint32_t kClcTxBytes       = 16;
constexpr uint32_t kMainFullCount    = 1, kMainEmptyCount = 2;      // 2/2 + 2/1 - 1
constexpr uint32_t kClcFullCount     = 1, kClcEmptyCount  = 800;    // 32 + 4 * (32 + 128 + 32); beta = 0
constexpr uint32_t kAccFullCount     = 1, kAccEmptyCount  = 256;    // 2 x 128 epilogue threads
constexpr uint32_t kThrottleCount    = 32, kTmemDeallocCount = 32;
constexpr uint16_t kMainloopReleaseMask = 0xF;                       // calculate_multicast_mask<kRowCol> (sm100_pipeline.hpp:57-91)
constexpr uint32_t kPeerBitMask      = 0xFEFFFFFFu;                  // cute::Sm100MmaPeerBitMask (copy_sm100_tma.hpp:45); nvcc folds it to 0xFEFFFFF8
constexpr uint64_t kCacheHintEvictNormal = 0x1000000000000000ull;    // TMA::CacheHintSm100::EVICT_NORMAL = Sm100MemDescDefault
constexpr uint32_t kWaitTicks        = 0x989680u;                    // ClusterBarrier::wait suspend-time hint (barrier.h:415)
constexpr uint32_t kBarEpilogue      = 1, kBarTmemAlloc = 6;         // ReservedNamedBarriers (barrier.h:169-178)
constexpr uint32_t kEpilogueThreads  = 128, kTmemAllocBarThreads = 160;
static_assert(kMainloopTxBytes == 49152 && kClcEmptyCount == 32 + 4 * (32 + 128 + 32), "Section 7.2 counts");

// [CHECK] LAYOUT  CuTe cross-check (compile time only, the only use of CuTe in this TU): the epilogue smem byte formula uses
// Swizzle<2,5,2> on byte offsets, bits [5,7) ^= bits [7,9) (Layout_MN_SW128_32B_Atom<float>, Section 4; B.3.3 step 17).
// Example: byte offset 0x1E4 (row 121 of a 32 x 4 fp32 atom column block) -> 0x1E4 ^ ((0x1E4 & 0x180) >> 2) = 0x1E4 ^ 0x60 = 0x184.
__host__ __device__ constexpr uint32_t epi_swz(uint32_t o) { return o ^ ((o & 0x180u) >> 2); }
using EpiSwizzle = cute::Swizzle<2, 5, 2>;
static_assert(EpiSwizzle::num_bits == 2 && EpiSwizzle::num_base == 5 && EpiSwizzle::num_shft == 2, "Swizzle<2,5,2>");
static_assert(EpiSwizzle::yyy_msk::value == 0x180 && EpiSwizzle::zzz_msk::value == 0x60, "Swizzle<2,5,2> masks: bits [7,9) fold into bits [5,7)");
static_assert(epi_swz(0x080u) == EpiSwizzle{}(0x080u) && epi_swz(0x1E4u) == EpiSwizzle{}(0x1E4u) && epi_swz(0x7FFCu) == EpiSwizzle{}(0x7FFCu),
              "epi_swz matches cute::Swizzle<2,5,2> (E.6.1) [confirm at first build: apply() is constexpr]");
// Swizzle<3,4,3> (A/B stages) is applied by the TMA unit on write and by the UMMA unit on read (SWIZZLE_128B); the kernel never computes it.
static_assert(cute::Swizzle<3, 4, 3>::yyy_msk::value == 0x380 && cute::Swizzle<3, 4, 3>::num_shft == 3, "Swizzle<3,4,3>: 16-byte chunk ^= row mod 8");

///////////////////////////////////////////////////////////////////////////////////////////////////
// 2. [TRACE]  Trace-only diagnostics (E.7.2 B1/B2, E.7.3): hang records in a host-mapped pinned buffer, heartbeats, K_TAIL.
//    With the toggle off this section reduces to the two #else lines: EXPLICIT_WAIT is the plain blocking wait and the CLC
//    record kinds are 0 (clc_consume then emits nothing).
///////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
enum WaitSite : uint32_t {
  WAIT_NONE = 0, WAIT_MAINLOOP_FULL, WAIT_MAINLOOP_EMPTY, WAIT_ACC_FULL, WAIT_ACC_EMPTY, WAIT_CLC_FULL, WAIT_CLC_EMPTY,
  WAIT_THROTTLE_FULL, WAIT_THROTTLE_EMPTY, WAIT_TMEM_DEALLOC, TAIL_MAINLOOP, TAIL_ACC, TAIL_CLC, WAIT_SITE_COUNT
};
struct HangRec {                       // 64 bytes; slot ((by * 64 + bx) * 8 + warp)
  uint32_t site, bx, by, rank;         // site != 0 once a wait exceeded kHangNs
  uint32_t warp, lane, addr, parity;   // barrier shared::cta address and parity of the stuck wait
  uint32_t tile, ktile;                // tile counter and k-tile of the stuck wait
  uint32_t hb_tile, hb_phase;          // heartbeat (lane 0 of each role warp): tile counter and phase marker
  uint64_t t;                          // %globaltimer at the hang record
  uint32_t hb_valid, pad_[1];
};
static_assert(sizeof(HangRec) == 64, "HangRec");
constexpr uint32_t kHangSlots = 64 * 64 * 8;
constexpr uint64_t kHangNs = 4000000000ull;        // 4 s
constexpr uint64_t kHangSpinNs = 1000000000ull;    // 1 s of extra spinning so that every stuck warp records
enum HbPhase : uint32_t { HB_TILE_START = 1, HB_TAIL = 2, HB_ALLOC = 3, HB_EXIT = 4 };
static __device__ HangRec* g_explicit_hang_buf = nullptr;
// K_TAIL (kind 13) is declared in deabstraction_trace.hpp's TraceKind enum (additive kind; kinds 12-15 were free, K_COUNT = 16)
// [TRACE] every blocking wait of the roles goes through this macro: with the toggle on it is the bounded, hang-recording wait (4 s)
#define EXPLICIT_WAIT(addr, parity, site, tile, q) mbar_wait_traced((addr), (parity), (site), (tile), (q))
constexpr uint32_t K_CLC_SCHED_KIND = K_CLC_SCHED;   // clc_consume<> record kinds (K5b, K5c)
constexpr uint32_t K_CLC_MMA_KIND   = K_CLC_MMA;
#else
// [PROD] with the toggle off: the baseline's mbarrier.try_wait.parity loop with the 0x989680 suspend hint (W6); site/tile/q vanish
#define EXPLICIT_WAIT(addr, parity, site, tile, q) mbar_wait((addr), (parity))
constexpr uint32_t K_CLC_SCHED_KIND = 0u;             // no records with the toggle off
constexpr uint32_t K_CLC_MMA_KIND   = 0u;
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// 3. [PROD]  E.6.2 PTX wrappers (W1-W37); each asm string is copied verbatim from the CUTLASS source named in its comment,
//    so the emitted PTX is identical to the CUTLASS kernel's (verified row by row in G.3). Grouped by hardware family:
//      warp/cluster identity  W1 (shfl), W2 (elect.sync), W3/W4 (%cluster_ctarank, %cluster_ctaid)
//      mbarrier               W5 init, W6 blocking wait, W7 peek, W8 test_wait, W9 local arrive, W10 remote arrive (mapa),
//                             W11 pair-leader arrive (peer bit), W12 local arrive.expect_tx, W13 remote arrive.expect_tx
//      fences                 W14 fence.mbarrier_init, W15 fence.proxy.async (generic-proxy writes -> async proxy)
//      cluster / named bars   W16/W17 barrier.cluster arrive/wait, W18/W19 bar.sync / bar.arrive
//      TMA                    W20 prefetch.tensormap, W21 2SM multicast load, W22 2SM load, W23 store, W24 commit_group, W25 wait_group.read
//      tcgen05                W26-W28 alloc/dealloc/relinquish, W29 mma, W30 commit (multicast), W31 ld 32x32b.x16, W32 wait::ld
//      CLC                    W33 try_cancel (the query), W34 decode (is_canceled, get_first_ctaid)
//      GDC                    W35 griddepcontrol.wait, W36 launch_dependents; W37 = __syncwarp()
//    A wrapper does exactly one thing; the protocol (who calls it, on which barrier, with which count/parity) lives in sections 5-10.
///////////////////////////////////////////////////////////////////////////////////////////////////

// cute::cast_smem_ptr_to_uint (cute/arch/util.hpp:93-108): the 32-bit shared::cta address of a generic pointer
__device__ __forceinline__ uint32_t smem_u32(void const* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// W1 [warp]  cutlass::canonical_warp_idx_sync (cutlass.h:127-133)
__device__ __forceinline__ int warp_idx_sync() { return __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x / 32), 0); }

// W2 [warp]  cute::elect_one_sync (cluster_sm90.hpp:180-197): pred is 1 in exactly one lane of the converged warp, 0 elsewhere
__device__ __forceinline__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
}

// W3/W4 [cluster]  cute::block_rank_in_cluster, cute::block_id_in_cluster (cluster_sm90.hpp:154-158, 126-132)
__device__ __forceinline__ uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r) :); return r; }
__device__ __forceinline__ uint32_t cluster_ctaid_x() { uint32_t x; asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) :); return x; }
__device__ __forceinline__ uint32_t cluster_ctaid_y() { uint32_t y; asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) :); return y; }

// W5 [mbarrier]  ClusterBarrier::init (barrier.h:391-403)
__device__ __forceinline__ void mbar_init(uint32_t smem_addr, uint32_t arrive_count) {
  asm volatile(
    "{\n\t"
    "mbarrier.init.shared::cta.b64 [%1], %0; \n"
    "}"
    :
    : "r"(arrive_count), "r"(smem_addr)
    : "memory");
}

// W6 [mbarrier]  ClusterBarrier::wait (barrier.h:410-430): blocking try_wait loop with the suspend-time hint in a register
__device__ __forceinline__ void mbar_wait(uint32_t smem_addr, uint32_t phase) {
  uint32_t ticks = kWaitTicks;
  asm volatile(
    "{\n\t"
    ".reg .pred       P1; \n\t"
    "LAB_WAIT: \n\t"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
    "@P1 bra DONE; \n\t"
    "bra     LAB_WAIT; \n\t"
    "DONE: \n\t"
    "}"
    :
    : "r"(smem_addr), "r"(phase), "r"(ticks)
    : "memory");
}

// W7 [mbarrier]  ClusterBarrier::try_wait (barrier.h:461-478): one peek, no suspend hint; 1 = WaitDone, 0 = WaitAgain
__device__ __forceinline__ uint32_t mbar_try_wait(uint32_t smem_addr, uint32_t phase) {
  uint32_t waitComplete;
  asm volatile(
    "{\n\t"
    ".reg .pred P1; \n\t"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
    "selp.b32 %0, 1, 0, P1; \n\t"
    "}"
    : "=r"(waitComplete)
    : "r"(smem_addr), "r"(phase)
    : "memory");
  return waitComplete;
}

// W8 [mbarrier]  ClusterBarrier::test_wait (barrier.h:435-457): CLC producer_tail only
__device__ __forceinline__ uint32_t mbar_test_wait(uint32_t smem_addr, uint32_t phase, uint32_t pred) {
  uint32_t waitComplete;
  asm volatile(
    "{\n\t"
    ".reg .pred P1; \n\t"
    ".reg .pred P2; \n\t"
    "setp.eq.u32 P2, %3, 1;\n\t"
    "@P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
    "selp.b32 %0, 1, 0, P1; \n\t"
    "}"
    : "=r"(waitComplete)
    : "r"(smem_addr), "r"(phase), "r"(pred)
    : "memory");
  return waitComplete;
}

// W9 [mbarrier]  ClusterBarrier::arrive (local, barrier.h:509-518)
__device__ __forceinline__ void mbar_arrive_local(uint32_t smem_addr) {
  asm volatile(
    "{\n\t"
    "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
    "}"
    :
    : "r"(smem_addr)
    : "memory");
}

// W10 [mbarrier]  ClusterBarrier::arrive(cta_id, pred) (barrier.h:486-499): remote arrive through mapa; the `if (pred)` stays in C++
__device__ __forceinline__ void mbar_arrive_remote(uint32_t smem_addr, uint32_t cta_id) {
  asm volatile(
    "{\n\t"
    ".reg .b32 remAddr32;\n\t"
    "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
    "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
    "}"
    :
    : "r"(smem_addr), "r"(cta_id)
    : "memory");
}

// W11 [mbarrier]  cutlass::arch::umma_arrive_2x1SM_sm0 (barrier.h:905-921): arrive on the pair leader's barrier through the peer-bit-masked address
__device__ __forceinline__ void mbar_arrive_cluster_masked(uint32_t smem_addr) {
  uint32_t bar_intptr = smem_addr & kPeerBitMask;
  asm volatile (
    "{\n\t"
    "mbarrier.arrive.shared::cluster.b64 _, [%0];\n\t"
    "}"
    :
    : "r"(bar_intptr)
    : "memory");
}

// W12 [mbarrier]  ClusterTransactionBarrier::arrive_and_expect_tx (local, barrier.h:588-598)
__device__ __forceinline__ void mbar_arrive_expect_tx_local(uint32_t smem_addr, uint32_t transaction_bytes) {
  asm volatile(
    "{\n\t"
    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
    "}"
    :
    : "r"(transaction_bytes), "r"(smem_addr)
    : "memory");
}

// W13 [mbarrier]  ClusterTransactionBarrier::arrive_and_expect_tx(bytes, cta_id, pred) (barrier.h:606-620): every lane executes, lanes with pred == 1 act
__device__ __forceinline__ void mbar_arrive_expect_tx_remote_pred(uint32_t smem_addr, uint32_t cta_id, uint32_t pred, uint32_t transaction_bytes) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    ".reg .b32 remAddr32;\n\t"
    "setp.eq.u32 p, %2, 1;\n\t"
    "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
    "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
    "}"
    :
    : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes)
    : "memory");
}

// W14 [fence]  cutlass::arch::fence_barrier_init (barrier.h:712-720)
__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
  asm volatile(
    "{\n\t"
    "fence.mbarrier_init.release.cluster; \n"
    "}"
    ::
    : "memory");
}

// W15 [fence]  cutlass::arch::fence_view_async_shared (barrier.h:728-736)
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile (
    "{\n\t"
    "fence.proxy.async.shared::cta; \n"
    "}"
    ::
    : "memory");
}

// W16/W17 [cluster]  cute::cluster_arrive_relaxed, cute::cluster_wait (cluster_sm90.hpp:48-51, 66-69)
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : : ); }
__device__ __forceinline__ void cluster_wait()           { asm volatile("barrier.cluster.wait.aligned;\n" : : ); }

// W18/W19 [named barrier]  NamedBarrier::arrive_and_wait_internal / arrive_internal (barrier.h:285-288, 305-309)
__device__ __forceinline__ void named_bar_sync(uint32_t barrier_id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(uint32_t barrier_id, uint32_t num_threads) {
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads) : "memory");
}

// W20 [TMA]  cute::prefetch_tma_descriptor (copy_sm90_desc.hpp:302-317): generic address of the parameter-space descriptor
__device__ __forceinline__ void prefetch_tensormap(void const* desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
}

// W21 [TMA]  SM100_TMA_2SM_LOAD_MULTICAST_3D::copy (copy_sm100_tma.hpp:283-305); the barrier operand is already peer-bit-masked by the caller
__device__ __forceinline__ void tma_load_2sm_mcast_3d(void const* desc_ptr, uint32_t smem_int_mbar, uint16_t multicast_mask, uint64_t cache_hint,
                                                      uint32_t smem_int_ptr, int32_t crd0, int32_t crd1, int32_t crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
    " [%0], [%1, {%4, %5, %6}], [%2], %3, %7;"
    :
    : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "h"(multicast_mask),
      "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
    : "memory");
}

// W22 [TMA]  SM100_TMA_2SM_LOAD_3D::copy (copy_sm100_tma.hpp:104-128)
__device__ __forceinline__ void tma_load_2sm_3d(void const* desc_ptr, uint32_t smem_int_mbar, uint64_t cache_hint,
                                                uint32_t smem_int_ptr, int32_t crd0, int32_t crd1, int32_t crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4, %5}], [%2], %6;"
    :
    : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
      "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
    : "memory");
}

// W23 [TMA]  SM90_TMA_STORE_3D::copy (copy_sm90_tma.hpp:1003-1023)
__device__ __forceinline__ void tma_store_3d(void const* desc_ptr, uint32_t smem_int_ptr, int32_t crd0, int32_t crd1, int32_t crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
    :
    : "l"(gmem_int_desc), "r"(smem_int_ptr),
      "r"(crd0), "r"(crd1), "r"(crd2)
    : "memory");
}

// W24/W25 [TMA]  cute::tma_store_arrive, cute::tma_store_wait<Count> (copy_sm90_tma.hpp:1225-1231, 1248-1256)
__device__ __forceinline__ void tma_store_commit_group() { asm volatile("cp.async.bulk.commit_group;"); }
template <int Count>
__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile(
    "cp.async.bulk.wait_group.read %0;"
    :
    : "n"(Count)
    : "memory");
}

// W26-W28 [tcgen05]  cute::TMEM::Allocator2Sm::allocate / free / release_allocation_lock (tmem_allocator_sm100.hpp:134-145, 157-169, 172-179)
__device__ __forceinline__ void tmem_alloc_2sm(uint32_t dst_intptr, uint32_t num_columns) {
  asm volatile(
    "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
    :
    : "r"(dst_intptr), "r"(num_columns));
}
__device__ __forceinline__ void tmem_dealloc_2sm(uint32_t tmem_ptr, uint32_t num_columns) {
  asm volatile(
    "{\n\t"
    "tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1; \n\t"
    "}"
    :
    : "r"(tmem_ptr), "r"(num_columns));
}
__device__ __forceinline__ void tmem_relinquish_2sm() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::);
}

// W29 [tcgen05 MMA]  SM100_MMA_F16BF16_2x1SM_SS::fma (mma_sm100_umma.hpp:549-587); the elect_one_sync() of the source is at the call site
__device__ __forceinline__ void umma_f16_2sm(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t scaleC) {
  uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
    "}\n"
    :
    : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scaleC),
      "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
      "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
}

// W30 [tcgen05 commit]  cutlass::arch::umma_arrive_multicast_2x1SM (barrier.h:848-861); the elect_one_sync() of the source is at the call site
__device__ __forceinline__ void umma_commit_mcast_2sm(uint32_t bar_intptr, uint16_t cta_mask) {
  asm volatile(
    "{\n\t"
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; \n\t"
    "}"
    :
    :"r"(bar_intptr), "h"(cta_mask)
    : "memory");
}

// W31 [tcgen05 ld]  SM100_TMEM_LOAD_32dp32b16x::copy (copy_sm100.hpp:3576-3604): lane l receives 16 consecutive columns of TMEM row (lane) l of the addressed 32-lane group
__device__ __forceinline__ void tmem_ld_32x32b_x16(uint32_t src_addr, uint32_t (&d)[16]) {
  asm volatile ("tcgen05.ld.sync.aligned.32x32b.x16.b32"
                  "{%0, %1, %2, %3,"
                  "%4, %5, %6, %7,"
                  "%8, %9, %10, %11,"
                  "%12, %13, %14, %15},"
                  "[%16];\n"
  :  "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]),
     "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
     "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]),
     "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
  :  "r"(src_addr));
}

// W32 [tcgen05 ld]  cutlass::arch::fence_view_async_tmem_load (barrier.h:923-931)
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile (
    "{\n\t"
    "tcgen05.wait::ld.sync.aligned; \n"
    "}"
    ::
    : "memory");
}

// W33 [CLC]  PersistentTileSchedulerSm100::issue_clc_query (sm100_tile_scheduler.hpp:392-405); elected lane of the scheduler warp
__device__ __forceinline__ void clc_try_cancel(uint32_t result_addr, uint32_t mbarrier_addr) {
  asm volatile(
    "{\n\t"
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
    "}\n"
    :
    : "r"(result_addr), "r"(mbarrier_addr));
}

// W34 [CLC]  PersistentTileSchedulerSm100::work_tile_info_from_clc_response (sm100_tile_scheduler.hpp:409-436); the fence follows in C++.
// For a not-cancelled response (valid == 0) the predicated get_first_ctaid leaves x0/y0/z0 unwritten (D.3): read them only under valid.
__device__ __forceinline__ uint32_t clc_decode(uint32_t result_addr, uint32_t& x0, uint32_t& y0, uint32_t& z0) {
  uint32_t valid = 0;
  asm volatile(
    "{\n"
    ".reg .pred p1;\n\t"
    ".reg .b128 clc_result;\n\t"
    "ld.shared.b128 clc_result, [%4];\n\t"
    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"
    "selp.u32 %3, 1, 0, p1;\n\t"
    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result;\n\t"
    "}\n"
    : "=r"(x0), "=r"(y0), "=r"(z0), "=r"(valid)
    : "r"(result_addr)
    : "memory"
  );
  return valid;
}

// W35/W36 [GDC]  cutlass::arch::wait_on_dependent_grids / launch_dependent_grids (grid_dependency_control.h:95-98, 85-88);
// emitted unconditionally: the fixed build defines CUTLASS_ENABLE_GDC_FOR_SM100=1 for sm_100a (Section 11)
__device__ __forceinline__ void griddep_wait()              { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }
// W37 is the __syncwarp() intrinsic (bar.warp.sync -1), used directly.

///////////////////////////////////////////////////////////////////////////////////////////////////
// 4. [PROD] mbarrier  Pipeline state and descriptor helpers; [TRACE] bounded wait, heartbeat, K_TAIL record
///////////////////////////////////////////////////////////////////////////////////////////////////
// E.6.5  Pipeline state: {index, phase}; producers start {0, 1}, consumers {0, 0} (sm90_pipeline.hpp:254-260); operator++ (:204-213).
// Why the producer starts at phase 1: a freshly initialised mbarrier is in phase 0, and try_wait.parity with parity 1 returns true at once
// ("the previous phase is complete"), so the first pass over the 8 empty[] barriers never blocks; the consumer waits parity 0 on full[]
// and blocks until the first arrival + transaction bytes complete phase 0. After every wrap of the ring the phase bit flips.
template <uint32_t Stages>
__device__ __forceinline__ void advance(uint32_t& idx, uint32_t& phase) {
  if (++idx == Stages) { idx = 0; phase ^= 1; }
}

// 64-bit descriptor formed at the asm boundary from the constant high word and the 32-bit low word (D.2: ULEA/UIADD3 on the low word, UMOV of the high word)
__device__ __forceinline__ uint64_t desc64(uint32_t hi, uint32_t lo) { return (uint64_t(hi) << 32) | uint64_t(lo); }

#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE] everything to the matching #endif is compiled out with the toggle off
__device__ __forceinline__ uint64_t globaltimer_ns() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ HangRec* hang_slot() {
  return g_explicit_hang_buf == nullptr ? nullptr : g_explicit_hang_buf + ((blockIdx.y * 64u + blockIdx.x) * 8u + threadIdx.x / 32u);
}
// B2 heartbeat: lane 0 of a role warp records where it is (tile counter, phase marker)
__device__ __forceinline__ void heartbeat(uint32_t tile, uint32_t phase) {
  HangRec* h = hang_slot();
  if (h != nullptr && (threadIdx.x & 31u) == 0u) { h->hb_tile = tile; h->hb_phase = phase; h->hb_valid = 1u; }
}
// B1 bounded blocking wait (toggle-on only): the same try_wait with the suspend hint, in a C++ loop with a 4 s bound
__device__ __forceinline__ void mbar_wait_traced(uint32_t smem_addr, uint32_t phase, uint32_t site, uint32_t tile, uint32_t ktile) {
  uint64_t const t0 = globaltimer_ns();
  uint32_t ticks = kWaitTicks;   // a local, as in ClusterBarrier::wait (barrier.h:415)
  for (;;) {
    uint32_t done;
    asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2, %3; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(done)
      : "r"(smem_addr), "r"(phase), "r"(ticks)
      : "memory");
    if (done) { return; }
    uint64_t const now = globaltimer_ns();
    if (now - t0 > kHangNs) {
      HangRec* h = hang_slot();
      if (h != nullptr && (threadIdx.x & 31u) == 0u) {
        h->site = site; h->bx = blockIdx.x; h->by = blockIdx.y; h->rank = cluster_ctarank();
        h->warp = threadIdx.x / 32u; h->lane = threadIdx.x & 31u; h->addr = smem_addr; h->parity = phase;
        h->tile = tile; h->ktile = ktile; h->t = now;
      }
      __threadfence_system();
      while (globaltimer_ns() - now < kHangSpinNs) { }
      __trap();
    }
  }
}
// K_TAIL record (E.7.3): one per completed tail step, lane 0 of the tail's warp
__device__ __forceinline__ void trace_tail(uint32_t site, uint32_t idx, uint32_t phase, uint32_t tile) {
  if ((threadIdx.x & 31u) == 0u) { TRACE_RECORD(K_TAIL, 8192, site, idx, phase, tile, 0, 0, 0, 0, 0, 0); }
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// 5. [PROD] CLC LAYOUT  CLC consume (every participating warp, once per tile; Section 7.4, B.3.4 steps 5b-5c)
//   The scheduler warp of rank 0 asks the hardware for another cluster's work through clusterlaunchcontrol.try_cancel (section 7);
//   the 16-byte response is written into every CTA's clc_response[idx] (multicast::cluster::all) and completes every CTA's clc_full[idx].
//   Each participating warp (warp 2, rank-0 warp 1, warp 0, warps 4-7: 32 + 4 x (32 + 32 + 128) = 800 threads cluster-wide) then:
//     wait own full[idx] (parity phase); decode; fence.proxy.async; remote arrive on CTA 0's empty[idx] (count 800); ++state (2 slots);
//   and maps the cancelled cluster's first-CTA id (x0, y0), always even, to this CTA's tile:
//     (tm, tn) = (2 (y0 / 2) + x0 % 2 + x, 2 (x0 / 2) + y0 % 2 + y)     (swizzle_and_rasterize + AlongN transpose, B.4)
//   Example: response (x0, y0) = (10, 6) in CTA (x, y) = (1, 0) -> (tm, tn) = (7, 10): rows [896, 1024) of A, columns [1280, 1408) of D.
//   A not-cancelled response (valid == 0) ends the persistent loop of every warp; its x0/y0/z0 are stale and are not read.
//   Slots: response idx at OFF_CLC_RESPONSE + 16 idx (352, 368); full at 208 + 8 idx; empty at 224 + 8 idx (CTA 0's are the ones used).
//   TraceKind: 0 (no record), K_CLC_SCHED (rank-0 warp 1 lane 0), K_CLC_MMA (warp 0 lane 0); [TRACE] only.
///////////////////////////////////////////////////////////////////////////////////////////////////
template <uint32_t TraceKind>
__device__ __forceinline__ bool clc_consume(uint32_t smem_base, uint32_t x, uint32_t y, uint32_t& clc_idx, uint32_t& clc_phase,
                                            uint32_t& clc_count, int& tm_next, int& tn_next, uint32_t tile) {
  EXPLICIT_WAIT(smem_base + OFF_CLC_FULL + 8u * clc_idx, clc_phase, WAIT_CLC_FULL, tile, 0u);    // consumer_wait (own CTA's full barrier)
  uint32_t x0 = 0, y0 = 0, z0 = 0;
  uint32_t const valid = clc_decode(smem_base + OFF_CLC_RESPONSE + 16u * clc_idx, x0, y0, z0);
  fence_proxy_async_shared_cta();                                                                   // work_tile_info_from_clc_response (:433)
  mbar_arrive_remote(smem_base + OFF_CLC_EMPTY + 8u * clc_idx, 0u);                                 // consumer_release -> CTA 0 (sm100_pipeline.hpp:1132-1135)
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  uint32_t const rec_idx = clc_idx, rec_phase = clc_phase, rec_count = clc_count;
#endif
  advance<kClcStages>(clc_idx, clc_phase);
  ++clc_count;
  (void)clc_count;
  if (valid) {
    tm_next = static_cast<int>(2u * (y0 >> 1) + (x0 & 1u) + x);   // swizzle_and_rasterize + possibly_transpose_work_tile (AlongN), B.4
    tn_next = static_cast<int>(2u * (x0 >> 1) + (y0 & 1u) + y);
  }
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  if constexpr (TraceKind != 0u) {
    if ((threadIdx.x & 31u) == 0u) {
      TRACE_RECORD(TraceKind, TraceKind == K_CLC_SCHED ? 8192 : 16384, x0, y0, z0, valid, rec_idx, rec_phase, rec_count,
                   valid ? uint32_t(tm_next) : 0u, valid ? uint32_t(tn_next) : 0u, 0);
    }
  }
#else
  (void)TraceKind;
#endif
  return valid != 0u;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// 6. [PROD] TMA mbarrier  Mainloop producer: warp 2 of every CTA (Section 7.5, B.3.1; CollectiveMma::load, sm100_mma_warpspecialized.hpp:589-629)
//   Per k-tile q of the current tile (tm, tn), stage s = the producer's ring index (0..7, wraps; phase flips at each wrap):
//     1. wait empty[s] (parity prod_phase; count 2 = the two pair leaders' tcgen05.commit of the previous use of stage s)
//     2. leader lane of the pair leader only: mbarrier.arrive.expect_tx on its own full[s] with 49152 bytes
//     3. one elected lane issues two 2SM TMA loads, both completing on the PAIR LEADER's full[s] ((smem_base + 8 s) & ~(1 << 24)):
//        A  multicast to ranks x and x+2 (mask 0x5 << x), coordinates (64 q, 128 tm + 64 y, 0), destination OFF_SMEM_A + 16384 s + 8192 y
//           in both destination CTAs: this CTA supplies its 64-row half of the 128 x 64 A stage, the other cluster row supplies the other half
//        B  no multicast, coordinates (64 q, 128 tn + 64 x, 0), destination OFF_SMEM_B + 8192 s (this CTA's 64 N-columns)
//     4. advance {s, phase}; peek the next empty[] so that the blocking wait is skipped when the ring is free
//   Example: tile (tm, tn) = (5, 9), q = 7, rank 1 (x = 1, y = 0): A box (448, 640, 0) -> 0x01024800 (rank 1) and 0x03024800 (rank 3),
//   B box (448, 1216, 0) -> 0x01036800; rank 2 (x = 0, y = 1) writes its A half at 0x02026800 / 0x00026800 (offset + 8192).
//   Byte accounting per stage and pair: 2 CTAs x (2 A halves x 8192 + B 8192) = 49152 = the expect_tx of step 2.
//   After the last tile: 8 blocking waits (the tail) so that no stage is still being read when the CTA exits.
///////////////////////////////////////////////////////////////////////////////////////////////////
struct ProducerState {
  uint32_t prod_idx = 0, prod_phase = 1;   // mainloop producer state (8 stages), carried across tiles
  int q = 0;                                // k-tile index inside the current tile
};

// One load() call: n k-tiles starting at state.q. Two calls per tile (8 then k_tiles - 8) give two rolled loops.
__device__ __forceinline__ void load_ktiles(ExplicitGemmParams const& params, ProducerState& st, int n, uint32_t smem_base,
                                            uint32_t a_dst_base, uint32_t b_dst_base, uint32_t full_masked_base, uint32_t empty_base,
                                            uint16_t a_mcast_mask, bool is_leader_lane, int a_row, int b_col, uint32_t tile) {
  uint32_t tok = mbar_try_wait(empty_base + 8u * st.prod_idx, st.prod_phase);            // producer_try_acquire before the loop (:604)
  #pragma unroll 1                                                                          // CUTLASS_PRAGMA_NO_UNROLL (:607)
  while (n > 0) {
    if (!tok) { EXPLICIT_WAIT(empty_base + 8u * st.prod_idx, st.prod_phase, WAIT_MAINLOOP_EMPTY, tile, uint32_t(st.q)); }   // producer_acquire (sm90_pipeline.hpp:531-536)
    if (is_leader_lane) { mbar_arrive_expect_tx_local(smem_base + OFF_MAIN_FULL + 8u * st.prod_idx, kMainloopTxBytes); }   // (:537-539), leader lane, 49152
    uint32_t const s = st.prod_idx;                                                          // write_stage
    advance<kStages>(st.prod_idx, st.prod_phase);
    tok = mbar_try_wait(empty_base + 8u * st.prod_idx, st.prod_phase);                     // peek the next stage, unconditional (:617)
    if (elect_one_sync()) {                                                                  // one election for the A+B pair (:619-622)
      int32_t const c0 = 64 * st.q;
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
      if (TRACE_IN_FIRST_CLUSTER()) {   // K4a: coordinates, destination, elements, unmasked local full[s] (C.4.3)
        TRACE_RECORD(K_TMA_LOAD, 128, uint32_t(c0), uint32_t(a_row), 0u, a_dst_base + kStageBytesA * s, 8192u, smem_base + OFF_MAIN_FULL + 8u * s, 0, 0, 0, 0);
      }
#endif
      tma_load_2sm_mcast_3d(&params.tma_a, full_masked_base + 8u * s, a_mcast_mask, kCacheHintEvictNormal,
                            a_dst_base + kStageBytesA * s, c0, a_row, 0);                     // A: (64q, 128 tm + 64 y, 0), mask 0x5 << x
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
      if (TRACE_IN_FIRST_CLUSTER()) {
        TRACE_RECORD(K_TMA_LOAD, 128, uint32_t(c0), uint32_t(b_col), 0u, b_dst_base + kStageBytesB * s, 4096u, smem_base + OFF_MAIN_FULL + 8u * s, 0, 0, 0, 0);
      }
#endif
      tma_load_2sm_3d(&params.tma_b, full_masked_base + 8u * s, kCacheHintEvictNormal,
                      b_dst_base + kStageBytesB * s, c0, b_col, 0);                          // B: (64q, 128 tn + 64 x, 0)
    }
    --n;
    ++st.q;
  }
}

__device__ __forceinline__ void producer_role(ExplicitGemmParams const& params, uint32_t smem_base, uint32_t x, uint32_t y,
                                              bool is_cta0, bool is_leader_lane, int k_tiles, int tm, int tn) {
  griddep_wait();                                                                              // wait_on_dependent_grids (kernel :620)
  ProducerState st;
  uint32_t thr_idx = 0, thr_phase = 1;      // CLC throttle producer (rank 0 only), 2 stages
  uint32_t clc_idx = 0, clc_phase = 0, clc_count = 0;   // CLC consumer, 2 stages
  uint32_t const a_dst_base       = smem_base + OFF_SMEM_A + kStageBytesAHalf * y;   // + 8192 * y: this CTA's 64-row half of the 128x64 A stage (B.4)
  uint32_t const b_dst_base       = smem_base + OFF_SMEM_B;
  uint32_t const full_masked_base = smem_base & kPeerBitMask;                    // (smem + 8 s) & 0xFEFFFFFF names the pair leader's full[s]
  uint32_t const empty_base       = smem_base + OFF_MAIN_EMPTY;
  uint16_t const a_mcast_mask     = static_cast<uint16_t>(0x5u << x);            // ranks x and x + 2 (create_tma_multicast_mask<2>)
  int tm_next = tm, tn_next = tn;
  uint32_t tile = 0;
  bool valid;
  do {
    int const a_row = 128 * tm + 64 * static_cast<int>(y);   // A rows of this CTA's half; tm % 2 == x for every tile this CTA receives (B.3.1 step 13)
    int const b_col = 128 * tn + 64 * static_cast<int>(x);   // B columns of this CTA's half
    st.q = 0;
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
    heartbeat(tile, HB_TILE_START);
#endif
    if (is_cta0) {                                                                             // throttle producer_acquire + producer_commit (kernel :632-636)
      EXPLICIT_WAIT(smem_base + OFF_THROTTLE_EMPTY + 8u * thr_idx, thr_phase, WAIT_THROTTLE_EMPTY, tile, 0u);   // blocking, no peek (PipelineAsync)
      mbar_arrive_local(smem_base + OFF_THROTTLE_FULL + 8u * thr_idx);                        // all 32 lanes (count 32)
      advance<kThrottleStages>(thr_idx, thr_phase);
    }
    int const k_tile_prologue = k_tiles < kPrologueKTiles ? k_tiles : kPrologueKTiles;        // min(Stages, k_tile_count) (kernel :629)
    load_ktiles(params, st, k_tile_prologue, smem_base, a_dst_base, b_dst_base, full_masked_base, empty_base, a_mcast_mask, is_leader_lane, a_row, b_col, tile);
    // (load_order_barrier.arrive() between the two load() calls is skipped: is_epi_load_needed is false for beta = 0)
    load_ktiles(params, st, k_tiles - k_tile_prologue, smem_base, a_dst_base, b_dst_base, full_masked_base, empty_base, a_mcast_mask, is_leader_lane, a_row, b_col, tile);
    __syncwarp();                                                                              // (kernel :664)
    valid = clc_consume<0u>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // consume response t AFTER this tile's loads (:666-671)
    tm = tm_next; tn = tn_next;                                                                // the fetched tile becomes current only here
    ++tile;
  } while (valid);
  // load_tail = producer_tail (sm90_pipeline.hpp:445-454): 8 blocking waits continuing the producer state (Section 7.10 tail 1)
  #pragma unroll
  for (int i = 0; i < kStages; ++i) {
    EXPLICIT_WAIT(empty_base + 8u * st.prod_idx, st.prod_phase, TAIL_MAINLOOP, tile, uint32_t(i));
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
    heartbeat(tile, HB_TAIL);
    trace_tail(TAIL_MAINLOOP, st.prod_idx, st.prod_phase, tile);
#endif
    advance<kStages>(st.prod_idx, st.prod_phase);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// 7. [PROD] CLC mbarrier  CLC scheduler: warp 1 of rank 0 (Section 7.4, B.3.4; kernel :681-724)
//   Once per tile (so that at most one query is outstanding per cluster, D.3):
//     1. throttle: wait throttle_full[i] (count 32 = the producer warp of rank 0 arrived at the start of its tile), release throttle_empty[i]
//     2. wait own clc_empty[idx] (count 800: every consumer of the previous response of this slot has arrived, section 5)
//     3. lanes 0..3 arm CTAs 0..3's clc_full[idx] with mbarrier.arrive.expect_tx(16 B) through mapa (the response is 16 bytes)
//     4. one elected lane issues clusterlaunchcontrol.try_cancel: the hardware answers into every CTA's clc_response[idx] and
//        completes the 16 transaction bytes on every CTA's clc_full[idx]
//     5. consume the previous response like every other warp (section 5); this warp's 32 arrivals are part of the 800
//   Tail: 2 test_wait/wait pairs on clc_empty so that both slots are released before the CTA exits. Warp 1 of ranks 1-3 does nothing.
///////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void scheduler_role(uint32_t smem_base, uint32_t x, uint32_t y, uint32_t lane, int tm, int tn) {
  griddep_wait();                                                                              // (kernel :688)
  uint32_t thc_idx = 0, thc_phase = 0;      // throttle consumer
  uint32_t clp_idx = 0, clp_phase = 1;      // CLC producer
  uint32_t clc_idx = 0, clc_phase = 0, clc_count = 0;   // CLC consumer (this warp consumes too: its 32 arrivals are part of the 800)
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  uint32_t clp_count = 0;
#endif
  int tm_next = tm, tn_next = tn;
  uint32_t tile = 0;
  bool valid;
  do {
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
    heartbeat(tile, HB_TILE_START);
#endif
    // throttle consumer_wait + consumer_release (all 32 lanes; PipelineAsync, dst_blockid 0) (kernel :693-695)
    EXPLICIT_WAIT(smem_base + OFF_THROTTLE_FULL + 8u * thc_idx, thc_phase, WAIT_THROTTLE_FULL, tile, 0u);
    mbar_arrive_remote(smem_base + OFF_THROTTLE_EMPTY + 8u * thc_idx, 0u);
    advance<kThrottleStages>(thc_idx, thc_phase);
    // advance_to_next_work (sm100_tile_scheduler.hpp:438-451): producer_acquire (sm100_pipeline.hpp:1095-1104) + elected try_cancel
    uint32_t const full_addr = smem_base + OFF_CLC_FULL + 8u * clp_idx;
    EXPLICIT_WAIT(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, WAIT_CLC_EMPTY, tile, 0u);   // wait own empty[idx]
    mbar_arrive_expect_tx_remote_pred(full_addr, lane, lane < 4u ? 1u : 0u, kClcTxBytes);   // lanes 0..3 arm CTAs 0..3's full[idx] with 16 bytes
    if (elect_one_sync()) {
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
      TRACE_RECORD(K_CLC_ISSUE, 8192, clp_idx, clp_phase, clp_count, full_addr, blockIdx.x / 2, blockIdx.y / 2, 0, 0, 0, 0);   // K5a
#endif
      clc_try_cancel(smem_base + OFF_CLC_RESPONSE + 16u * clp_idx, full_addr);
    }
    advance<kClcStages>(clp_idx, clp_phase);
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
    ++clp_count;
#endif
    valid = clc_consume<K_CLC_SCHED_KIND>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // fetch_next_work (:702-706)
    tm = tm_next; tn = tn_next;
    ++tile;
  } while (valid);
  // CLC producer_tail (sm100_pipeline.hpp:1041-1050): test_wait, then the blocking wait only if not done (Section 7.10 tail 2)
  #pragma unroll
  for (int i = 0; i < kClcStages; ++i) {
    uint32_t const done = mbar_test_wait(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, 1u);
    if (!done) { EXPLICIT_WAIT(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, TAIL_CLC, tile, uint32_t(i)); }
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
    heartbeat(tile, HB_TAIL);
    trace_tail(TAIL_CLC, clp_idx, clp_phase, tile);
#endif
    advance<kClcStages>(clp_idx, clp_phase);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// 8. [PROD] tcgen05  MMA role: warp 0 of every CTA (Section 7.6, B.3.2; kernel :726-805; CollectiveMma::mma, sm100_mma_warpspecialized.hpp:651-712)
//   Both CTAs of the pair: tcgen05.alloc 512 TMEM columns (cta_group::2: one allocation for the pair, the result written to smem + 384 of
//   both CTAs), bar.arrive 6,160 to release the epilogue warps, and the accumulator producer state; at the end relinquish + dealloc.
//   The PAIR LEADER only, per tile: wait acc_empty[st] (count 256 = 2 x 128 epilogue threads released stage st), then per k-tile:
//     wait full[s] (count 1 + 49152 tx bytes, section 6) -> 4 x tcgen05.mma.cta_group::2 (one per 16-k block; the first of the tile with
//     scale_c = 0 overwrites the accumulator, the rest accumulate) -> tcgen05.commit to empty[s] of all four CTAs (mask 0xF, count 2 per CTA)
//   then tcgen05.commit to acc_full[st] of the pair (mask 0x3 << 2y) and advance the accumulator stage (both CTAs advance the state).
//   TMEM address = (lane << 16) | column; tmem_c = T + 128 st with T = 0 measured: stage st is columns [128 st, 128 st + 128) of all 128 lanes.
//   The MMA tile is 256 (M) x 128 (N): rows [0,128) come from the leader's A stage, [128,256) from the peer's; N from both B halves.
///////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void mma_role(char* smem, uint32_t smem_base, uint32_t x, uint32_t y, uint32_t rank, bool is_leader,
                                         uint32_t peer, uint32_t lane_pred, int k_tiles, int tm, int tn) {
  (void)tm; (void)tn;   // the MMA warp needs no tile coordinates (the TMEM stage and the smem stages are tile-independent)
  // TMEM allocation: whole warp 0 of both CTAs of the pair, same dst offset (tmem_allocator_sm100.hpp:124-133); result lands in smem
  tmem_alloc_2sm(smem_base + OFF_TMEM_BASE_PTR, kTmemColumns);
  __syncwarp();                                                                                // (kernel :729)
  named_bar_arrive(kBarTmemAlloc, kTmemAllocBarThreads);                                       // bar.arrive 6, 160 (:730)
  uint32_t const T = *reinterpret_cast<uint32_t const*>(smem + OFF_TMEM_BASE_PTR);             // tmem_base_ptr (:731); measured 0, never assumed (TMEM address = lane << 16 | column)
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  heartbeat(0u, HB_ALLOC);
  if (lane_pred) { TRACE_RECORD(K_TMEM, 1024, T, rank, is_leader ? 1u : 0u, 0u, 0, 0, 0, 0, 0, 0); }   // K2 site 0
#else
  (void)lane_pred; (void)rank;
#endif
  // UMMA smem descriptors from this CTA's own stage bases; the 14-bit start-address field drops the rank bits (mma_sm100_desc.hpp:754-755)
  uint32_t const desc_lo_a = kSmemDescLo0 | (((smem_base + OFF_SMEM_A) >> 4) & 0x3FFFu);      // expected 0x10880 (D.7)
  uint32_t const desc_lo_b = kSmemDescLo0 | (((smem_base + OFF_SMEM_B) >> 4) & 0x3FFFu);      // expected 0x12880 (D.7)
  uint32_t const full_base = smem_base + OFF_MAIN_FULL;
  uint32_t const empty_base = smem_base + OFF_MAIN_EMPTY;
  uint32_t const acc_full_base = smem_base + OFF_ACC_FULL;
  uint32_t const acc_empty_base = smem_base + OFF_ACC_EMPTY;
  uint16_t const acc_commit_mask = static_cast<uint16_t>(0x3u << (2u * y));                    // the pair: 0x3 for ranks {0,1}, 0xC for {2,3} (calculate_umma_peer_mask)
  uint32_t cons_idx = 0, cons_phase = 0;     // mainloop consumer state (8 stages), leader only
  uint32_t accp_idx = 0, accp_phase = 1;     // accumulator producer state (4 stages), both CTAs
  uint32_t clc_idx = 0, clc_phase = 0, clc_count = 0;
  int tm_next = tm, tn_next = tn;
  uint32_t tile = 0;
  bool valid;
  do {
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
    heartbeat(tile, HB_TILE_START);
#endif
    valid = clc_consume<K_CLC_MMA_KIND>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // BEFORE computing the current tile (:742-750)
    uint32_t const acc_stage = accp_idx;
    if (is_leader) {                                                                            // mma() (:762-770)
      uint32_t tok = (k_tiles <= 0) ? 1u : mbar_try_wait(full_base + 8u * cons_idx, cons_phase);   // consumer_try_wait(skip_wait) (:670-671)
      uint32_t scale_c = 0;                                                                     // ScaleOut::Zero (:676)
      EXPLICIT_WAIT(acc_empty_base + 8u * acc_stage, accp_phase, WAIT_ACC_EMPTY, tile, 0u);    // accumulator producer_acquire, blocking (:678)
      uint32_t const tmem_c = T + 128u * acc_stage;                                             // lane 0, columns [128 st, 128 st + 128): 0, 128, 256, 384 for T = 0
      int k = k_tiles;
      #pragma unroll 1                                                                          // CUTLASS_PRAGMA_NO_UNROLL (:680)
      while (k > 0) {
        if (!tok) { EXPLICIT_WAIT(full_base + 8u * cons_idx, cons_phase, WAIT_MAINLOOP_FULL, tile, uint32_t(k_tiles - k)); }   // consumer_wait (:684)
        uint32_t const rs = cons_idx;                                                           // read_stage
        advance<kStages>(cons_idx, cons_phase);
        --k;
        tok = (k <= 0) ? 1u : mbar_try_wait(full_base + 8u * cons_idx, cons_phase);             // peek the next stage; skipped on the last k-tile (:694-696)
        uint32_t const lo_a = desc_lo_a + 1024u * rs;                                           // + 16384 B per stage in 16-byte units (32-bit low-word add, mma_sm100_desc.hpp:861-874)
        uint32_t const lo_b = desc_lo_b + 512u * rs;                                            // +  8192 B per stage
        #pragma unroll                                                                          // (:699)
        for (int kb = 0; kb < kKBlocks; ++kb) {
          if (elect_one_sync()) {
            uint64_t const desc_a = desc64(kSmemDescHi, lo_a + 2u * uint32_t(kb));              // + 32 B per k-block
            uint64_t const desc_b = desc64(kSmemDescHi, lo_b + 2u * uint32_t(kb));
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
            if (TRACE_IN_FIRST_CLUSTER()) { TRACE_RECORD(K_MMA, 32, desc_a, desc_b, tmem_c, kInstrDesc, scale_c, 0, 0, 0, 0, 0); }   // K3
#endif
            umma_f16_2sm(tmem_c, desc_a, desc_b, kInstrDesc, scale_c);
          }
          scale_c = 1u;                                                                          // ScaleOut::One after every gemm (:706)
        }
        if (elect_one_sync()) { umma_commit_mcast_2sm(empty_base + 8u * rs, kMainloopReleaseMask); }   // consumer_release, mask 0xF (:708)
      }
      if (elect_one_sync()) { umma_commit_mcast_2sm(acc_full_base + 8u * acc_stage, acc_commit_mask); }   // accumulator producer_commit (:771)
    }
    advance<kAccStages>(accp_idx, accp_phase);                                                  // both CTAs (:773)
    tm = tm_next; tn = tn_next;                                                                 // (:774-775)
    ++tile;
  } while (valid);
  griddep_launch_dependents();                                                                  // launch_dependent_grids (:781)
  tmem_relinquish_2sm();                                                                        // release_allocation_lock (:784)
  if (is_leader) {                                                                              // accumulator producer_tail (:789; sm90_pipeline.hpp:1129-1134) (Section 7.10 tail 3)
    #pragma unroll
    for (int i = 0; i < kAccStages; ++i) {
      EXPLICIT_WAIT(acc_empty_base + 8u * accp_idx, accp_phase, TAIL_ACC, tile, uint32_t(i));
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
      heartbeat(tile, HB_TAIL);
      trace_tail(TAIL_ACC, accp_idx, accp_phase, tile);
#endif
      advance<kAccStages>(accp_idx, accp_phase);
    }
  }
  // TMEM deallocation handshake (:794-796): the peer arrives on the leader's barrier, both wait phase 0, the leader arrives on the peer's (Section 7.10 tail 4)
  if (!is_leader) { mbar_arrive_remote(smem_base + OFF_TMEM_DEALLOC, peer); }
  EXPLICIT_WAIT(smem_base + OFF_TMEM_DEALLOC, 0u, WAIT_TMEM_DEALLOC, tile, 0u);
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  heartbeat(tile, HB_TAIL);
  trace_tail(WAIT_TMEM_DEALLOC, 0u, 0u, tile);
#endif
  if (is_leader) { mbar_arrive_remote(smem_base + OFF_TMEM_DEALLOC, peer); }
  tmem_dealloc_2sm(T, kTmemColumns);                                                            // (:804), both CTAs
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  heartbeat(tile, HB_EXIT);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// 9. [PROD] tcgen05 TMA LAYOUT  Epilogue consumers: warps 4-7 of every CTA (Section 7.7, B.3.3; kernel :869-955; CollectiveEpilogue::store)
//   Warp w = warp_idx - 4 owns TMEM lanes (= accumulator rows) [32 w, 32 w + 32); lane l owns row 32 w + l of the CTA's 128 x 128 tile.
//   Per tile: wait acc_full[st] (count 1: the leader's commit); then 8 subtiles e = 0..7 of 16 columns:
//     1. tcgen05.ld.32x32b.x16 from T + ((32 w) << 16) + 128 st + 16 e: lane l gets D[row 32w + l][16 e .. 16 e + 15] as 16 fp32 registers
//        (example: w = 2, st = 1, e = 3 -> tmem address 0x4000B0 = lane 64, column 176)
//     2. d = fmaf(beta, 0, alpha * acc) per register (the fixed run's beta = 0 keeps the C term at literal 0; the FMUL + FFMA pair is the
//        baseline's own arithmetic, D.7); after the 8th load: tcgen05.wait::ld and one arrive per thread on the pair leader's acc_empty[st]
//     3. 16 x st.shared.b32 into C/D stage (e % 4) with the Swizzle<2,5,2> pattern below; fence.proxy.async; bar.sync 1,128
//     4. warp 4 issues four TMA stores of (32 m, 16 n) boxes from stage (e % 4) + 2048 i to D coordinates (128 tm + 32 i, 128 tn + 16 e, 0),
//        commit_group, wait_group.read 1 (at most one group still reading smem, so stage (e % 4) is free again 4 subtiles later); bar.sync 1,128
//   smem byte offset of value j (column 16 e + j) of lane l in warp w:
//        OFF_SMEM_CD + 8192 (e % 4) + 2048 w + 128 j + 4 ((l & 7) + 8 (((l >> 3) & 3) ^ (j & 3)))
//     = st_off[j & 3] + 512 (j >> 2) + 8192 (e & 3)    with  st_off[c] = 512 + 2048 w + 4 (l & 7) + 128 c + 32 (((l >> 3) & 3) ^ c)
//   Examples: w = 0, l = 0: st_off = {512, 672, 832, 992};  w = 0, l = 9: {548, 644, 868, 964};  w = 3, l = 31: {6780, 6876, 6972, 7068};
//             w = 0, l = 9, j = 5, e = 6 -> 644 + 512 + 16384 = 17540 (0x4484) = stage 2, the TMA store box i = 0 of that stage.
//   D box example: tile (5, 9), e = 3, i = 2 -> coordinates (704, 1200, 0), source stage 3 + 4096 B.
///////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void epilogue_role(ExplicitGemmParams const& params, char* smem, uint32_t smem_base, uint32_t x, uint32_t y,
                                              uint32_t rank, bool is_leader, int warp_idx, uint32_t lane, float alpha, float beta, int tm, int tn) {
  named_bar_sync(kBarTmemAlloc, kTmemAllocBarThreads);                                          // bar.sync 6, 160: wait for the TMEM address (:871)
  uint32_t const T = *reinterpret_cast<uint32_t const*>(smem + OFF_TMEM_BASE_PTR);              // (:872)
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  if (threadIdx.x == 128u) { TRACE_RECORD(K_TMEM, 1024, T, rank, is_leader ? 1u : 0u, 1u, 0, 0, 0, 0, 0, 0); }   // K2 site 1
#else
  (void)rank; (void)is_leader;
#endif
  uint32_t const w = static_cast<uint32_t>(warp_idx) - 4u;                                      // epilogue warp 0..3 (thread_idx / 32)
  uint32_t const l = lane;
  uint32_t const tmem_lane_off = (32u * w) << 16;                                               // TMEM lanes [32 w, 32 w + 32): row 32 w + l of the CTA tile (0, 0x200000, 0x400000, 0x600000)
  // Byte offsets (from the generic smem base) of this thread's 4 column classes inside an 8192-byte epilogue stage (B.4, D.6 item 1):
  //   value j of subtile e (column j of the subtile's stage) -> OFF_SMEM_CD + 8192 (e % 4) + 2048 w + 128 j + 4 ((l & 7) + 8 (((l >> 3) & 3) ^ (j & 3)))
  //   = st_off[j & 3] + 512 (j >> 2) + 8192 (e & 3)      (Swizzle<2,5,2> on byte offsets: bits [5,7) ^= bits [7,9))
  uint32_t st_off[4];
  #pragma unroll
  for (uint32_t c = 0; c < 4u; ++c) {
    st_off[c] = OFF_SMEM_CD + 2048u * w + 4u * (l & 7u) + 128u * c + 32u * (((l >> 3) & 3u) ^ c);
  }
  uint32_t const cd_base = smem_base + OFF_SMEM_CD;
  uint32_t const acc_full_base = smem_base + OFF_ACC_FULL;
  uint32_t const acc_empty_base = smem_base + OFF_ACC_EMPTY;
  bool const issue_tma_store = (w == 0u);                                                        // (:740): warp 4, all 32 lanes, uniform operands, no election
  uint32_t accc_idx = 0, accc_phase = 0;     // accumulator consumer state (4 stages)
  uint32_t clc_idx = 0, clc_phase = 0, clc_count = 0;
  int tm_next = tm, tn_next = tn;
  uint32_t tile = 0;
  bool valid;
  do {
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
    heartbeat(tile, HB_TILE_START);
#endif
    valid = clc_consume<0u>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // BEFORE storing the current tile (:878-886)
    uint32_t const acc_stage = accc_idx;
    uint32_t tok = mbar_try_wait(acc_full_base + 8u * acc_stage, accc_phase);                   // acc_pipeline.consumer_try_wait, once per tile (:813)
    uint32_t const taddr = T + tmem_lane_off + 128u * acc_stage;                                // warp-uniform TMEM address of this warp's 32 rows
    #pragma unroll                                                                               // 8 subtiles fully unrolled (:818-820); DelayTmaStore = false
    for (int e = 0; e < kEpiSubtiles; ++e) {
      if (e == 0 && !tok) { EXPLICIT_WAIT(acc_full_base + 8u * acc_stage, accc_phase, WAIT_ACC_FULL, tile, 0u); }   // consumer_wait on the first subtile (:864-867)
      uint32_t r[16];
      tmem_ld_32x32b_x16(taddr + 16u * uint32_t(e), r);                                        // columns 16e..16e+15 of row 32 w + l (:881-883)
      if (e == kEpiSubtiles - 1) {                                                               // after the last tmem load of the tile (:886-890)
        tmem_wait_ld();                                                                          // fence_view_async_tmem_load
        mbar_arrive_cluster_masked(acc_empty_base + 8u * acc_stage);                             // consumer_release: 128 threads of both CTAs -> 256 on the leader's empty[st]
        advance<kAccStages>(accc_idx, accc_phase);
      }
      // Section 8: t = alpha * acc; d = fmaf(beta, c, t) with the zero C fragment; run-time alpha/beta keep FMUL + FFMA (D.7)
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        float const t = alpha * __uint_as_float(r[j]);
        float const d = fmaf(beta, 0.0f, t);
        *reinterpret_cast<uint32_t*>(smem + st_off[j & 3] + 512u * uint32_t(j >> 2) + kEpiStageBytes * uint32_t(e & 3)) = __float_as_uint(d);   // 16 scalar st.shared.b32 (B.3.3 step 17)
      }
      // tma_store_fn (:766-800)
      fence_proxy_async_shared_cta();                                                            // smem writes visible to the async proxy
      named_bar_sync(kBarEpilogue, kEpilogueThreads);                                           // bar.sync 1, 128
      if (issue_tma_store) {
        uint32_t const src_base = cd_base + kEpiStageBytes * uint32_t(e & 3);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {                                                            // four (32 M, 16 N) boxes per subtile (Section 5.3)
          int32_t const c0 = 128 * tm + 32 * i;
          int32_t const c1 = 128 * tn + 16 * e;
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
          {
            uint32_t const lanes = static_cast<uint32_t>(__popc(__activemask()));   // K4b: every lane evaluates, lane 0 records
            if (TRACE_IN_FIRST_CLUSTER() && l == 0u) {
              TRACE_RECORD(K_TMA_STORE, 256, uint32_t(c0), uint32_t(c1), 0u, src_base + kStoreBoxBytes * uint32_t(i), lanes, 0, 0, 0, 0, 0);
            }
          }
#endif
          tma_store_3d(&params.tma_d, src_base + kStoreBoxBytes * uint32_t(i), c0, c1, 0);
        }
        tma_store_commit_group();                                                                // store_pipeline.producer_commit
        tma_store_wait_read<1>();                                                                // producer_acquire (always_wait): at most 1 group pending on its smem read
      }
      named_bar_sync(kBarEpilogue, kEpilogueThreads);                                           // bar.sync 1, 128
    }
    tm = tm_next; tn = tn_next;                                                                  // (:932-933)
    ++tile;
  } while (valid);
  // store_tail executes nothing for beta = 0 (is_producer_load_needed() is false): no cp.async.bulk.wait_group.read 0 (Section 7.7, D15, E.6.7)
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  heartbeat(tile, HB_EXIT);
#endif
}

} // namespace (anonymous)

///////////////////////////////////////////////////////////////////////////////////////////////////
// 10. [PROD]  The kernel (E.6.3): one __global__ function, __launch_bounds__(256, 1), extern __shared__ without alignment attribute,
//     no static __shared__ anywhere in this translation unit (device_kernel.h:114-127; Section 6.2).
//   Prologue, every CTA: identity -> scalars and tensormap prefetch -> 33 mbarrier.init by the elected lane of four warps
//   (warp 0: mainloop 8 full + 8 empty; warp 4: CLC 2 + 2; warp 5: accumulator 4 + 4; warp 3: throttle 2 + 2; warp 0: tmem_dealloc 1),
//   a fence.mbarrier_init.release.cluster after each of the four groups -> barrier.cluster.arrive -> initial tile -> barrier.cluster.wait
//   (no remote barrier operation before this point) -> role dispatch. Warps: 0 MMA, 1 scheduler (rank 0 only, else exit), 2 producer,
//   3 exits (the beta != 0 epilogue-load warp), 4-7 epilogue.
//   Initial tile from the launch coordinate (AlongN raster, cluster 2 x 2): tm = 2 (blockIdx.y / 2) + blockIdx.x % 2, tn = 2 (blockIdx.x / 2) + blockIdx.y % 2;
//   example blockIdx (5, 2) -> (tm, tn) = (3, 4); x = blockIdx.x % 2, y = blockIdx.y % 2, rank = x + 2 y.
///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void __launch_bounds__(256, 1)
explicit_blackwell_fp16_gemm_kernel(const __grid_constant__ ExplicitGemmParams params) {
  extern __shared__ char smem[];
  uint32_t const smem_base = smem_u32(smem);                                                   // 0x400 + (rank << 24) measured (D.1 D5); computed, never assumed

  // 7.3 step 1: warp index, election, cluster identity (kernel :417-430)
  int const warp_idx = warp_idx_sync();
  uint32_t const lane = threadIdx.x & 31u;
  uint32_t const lane_pred = elect_one_sync();
  uint32_t const rank = cluster_ctarank();
  bool const is_cta0 = (rank == 0u);
  bool const is_leader = ((rank & 1u) == 0u);                                                   // cta_coord_v = rank % 2 == 0
  uint32_t const peer = rank ^ 1u;

  // 7.3 step 2: run-time scalars and k-tile count (every thread); descriptor prefetch (kernel :439-445)
  float const alpha = params.alpha;
  float const beta = params.beta;
  int const k_tiles = (params.k + 63) >> 6;
  if (warp_idx == 1 && lane_pred) {                                                            // Sched warp of every CTA: A, B
    prefetch_tensormap(&params.tma_a);
    prefetch_tensormap(&params.tma_b);
  }
  if (warp_idx == 3 && lane_pred) {                                                            // EpilogueLoad warp of every CTA, regardless of beta: C, D
    prefetch_tensormap(&params.tma_c);
    prefetch_tensormap(&params.tma_d);
  }

  // 7.3 step 3: barrier initialisation in the baseline's constructor order (33-init variant, E.10 item 6):
  // the elected lane of the initialising warp issues the inits (initialize_barrier_array*_aligned, barrier.h:120-150), every thread fences once per group
  if (warp_idx == 0) {                                                                         // mainloop pipeline (kernel :457-472; sm100_pipeline.hpp:555-573)
    if (elect_one_sync()) {
      #pragma unroll
      for (uint32_t i = 0; i < uint32_t(kStages); ++i) {
        mbar_init(smem_base + OFF_MAIN_FULL + 8u * i, kMainFullCount);
        mbar_init(smem_base + OFF_MAIN_EMPTY + 8u * i, kMainEmptyCount);
      }
    }
  }
  fence_mbarrier_init_release_cluster();
  // (epi_load pipeline, warp 1, and load_order barrier, warp 3: beta != 0 only; not initialised, E.10 item 6)
  if (warp_idx == 4) {                                                                         // CLC pipeline (kernel :501-518)
    if (elect_one_sync()) {
      #pragma unroll
      for (uint32_t i = 0; i < uint32_t(kClcStages); ++i) {
        mbar_init(smem_base + OFF_CLC_FULL + 8u * i, kClcFullCount);
        mbar_init(smem_base + OFF_CLC_EMPTY + 8u * i, kClcEmptyCount);
      }
    }
  }
  fence_mbarrier_init_release_cluster();
  if (warp_idx == 5) {                                                                         // accumulator pipeline (kernel :520-536)
    if (elect_one_sync()) {
      #pragma unroll
      for (uint32_t i = 0; i < uint32_t(kAccStages); ++i) {
        mbar_init(smem_base + OFF_ACC_FULL + 8u * i, kAccFullCount);
        mbar_init(smem_base + OFF_ACC_EMPTY + 8u * i, kAccEmptyCount);
      }
    }
  }
  fence_mbarrier_init_release_cluster();
  if (warp_idx == 3) {                                                                         // CLC throttle pipeline (kernel :538-550)
    if (elect_one_sync()) {
      #pragma unroll
      for (uint32_t i = 0; i < uint32_t(kThrottleStages); ++i) {
        mbar_init(smem_base + OFF_THROTTLE_FULL + 8u * i, kThrottleCount);
        mbar_init(smem_base + OFF_THROTTLE_EMPTY + 8u * i, kThrottleCount);
      }
    }
  }
  fence_mbarrier_init_release_cluster();
  if (warp_idx == 0 && lane_pred) {                                                            // tmem_dealloc barrier (kernel :562-565), no fence of its own
    mbar_init(smem_base + OFF_TMEM_DEALLOC, kTmemDeallocCount);
  }

  // 7.3 step 4: cluster rendezvous, part 1 (kernel :580)
  cluster_arrive_relaxed();

  // 7.3 step 5: warp-uniform setup; the initial tile from the launch coordinate (swizzle_and_rasterize + AlongN transpose, sm100_tile_scheduler.hpp:365-371, 666-677)
  uint32_t const x = cluster_ctaid_x();                                                        // == rank & 1
  uint32_t const y = cluster_ctaid_y();                                                        // == rank >> 1
  int tm = static_cast<int>(2u * (blockIdx.y >> 1) + (blockIdx.x & 1u));
  int tn = static_cast<int>(2u * (blockIdx.x >> 1) + (blockIdx.y & 1u));

  // 7.3 step 6: cluster rendezvous, part 2 (kernel :615); no remote barrier operation before this point
  cluster_wait();

#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  if (TRACE_IN_FIRST_CLUSTER() && threadIdx.x == 0u) {   // K1: the Section 6.2 addresses as computed by this kernel (C.4.3)
    TRACE_RECORD(K_SMEM, 16, smem_base, smem_base + OFF_MAIN_FULL, smem_base + OFF_MAIN_EMPTY, smem_base + OFF_CLC_FULL, smem_base + OFF_CLC_EMPTY,
                 smem_base + OFF_ACC_EMPTY, smem_base + OFF_TMEM_DEALLOC, smem_base + OFF_CLC_RESPONSE, smem_base + OFF_SMEM_A, smem_base + OFF_SMEM_B);
    TRACE_RECORD(K_SMEM2, 16, smem_base + OFF_SMEM_CD, smem_base + OFF_TMEM_BASE_PTR, 0u, is_cta0 ? 1u : 0u, rank, peer,
                 smem_base + OFF_ACC_FULL, smem_base + OFF_THROTTLE_FULL, smem_base + OFF_LOAD_ORDER, smem_base + OFF_EPI_LOAD_FULL);
  }
#endif

  // 7.3 step 7: role dispatch in the baseline's branch order (kernel :617, 681, 726, 807, 869); warp 3 and warp 1 of ranks 1-3 fall through and exit
  if (warp_idx == 2) {
    producer_role(params, smem_base, x, y, is_cta0, lane_pred != 0u && is_leader, k_tiles, tm, tn);
  }
  else if (warp_idx == 1 && is_cta0) {
    scheduler_role(smem_base, x, y, lane, tm, tn);
  }
  else if (warp_idx == 0) {
    mma_role(smem, smem_base, x, y, rank, is_leader, peer, lane_pred, k_tiles, tm, tn);
  }
  else if (warp_idx >= 4) {
    epilogue_role(params, smem, smem_base, x, y, rank, is_leader, warp_idx, lane, alpha, beta, tm, tn);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// 11. [HOST] [PROD]  Host function: the mirror of GemmUniversalAdapter::run() for the explicit kernel (E.5).
//     Per call: guards (E.5.1) -> cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 230400) -> grid (64,64,1) / block 256 / cluster (2,2,1)
//     -> check_cluster_dims -> cudaFuncSetAttribute(NonPortableClusterSizeAllowed, 1) -> cudaLaunchKernelExC with one cluster attribute
//     -> cudaGetLastError. Nothing synchronises (the [TRACE] watchdog below runs only for the toggle-on warm-up launch).
//     The first namespace below is [TRACE]: the watchdog that bounds the traced warm-up launch to 20 s and prints the hang records.
///////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
namespace {
bool  g_explicit_trace_armed = false;    // set by trace_begin(), cleared by trace_end(): run() synchronises and checks hang records only then
void* g_hang_host = nullptr;             // pinned, host-mapped HangRec buffer (B1/B2)
char const* wait_site_name(uint32_t s) {
  static char const* names[] = {"NONE", "WAIT_MAINLOOP_FULL", "WAIT_MAINLOOP_EMPTY", "WAIT_ACC_FULL", "WAIT_ACC_EMPTY", "WAIT_CLC_FULL", "WAIT_CLC_EMPTY",
                                "WAIT_THROTTLE_FULL", "WAIT_THROTTLE_EMPTY", "WAIT_TMEM_DEALLOC", "TAIL_MAINLOOP", "TAIL_ACC", "TAIL_CLC"};
  return s < WAIT_SITE_COUNT ? names[s] : "?";
}
// After the traced warm-up launch: poll for completion (20 s bound, B2), then report hang records (B1) and heartbeats.
cutlass::Status explicit_trace_wait_and_check() {
  cudaEvent_t ev;
  if (cudaEventCreate(&ev) != cudaSuccess) { return cutlass::Status::kErrorInternal; }
  cudaError_t e = cudaEventRecord(ev, nullptr);
  auto const t0 = std::chrono::steady_clock::now();
  bool timed_out = false;
  if (e == cudaSuccess) {
    for (;;) {
      e = cudaEventQuery(ev);
      if (e != cudaErrorNotReady) { break; }
      if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count() >= 20) { timed_out = true; break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  (void)cudaEventDestroy(ev);
  HangRec const* h = static_cast<HangRec const*>(g_hang_host);
  if (h != nullptr && (timed_out || e != cudaSuccess)) {
    unsigned n_hang = 0, n_hb = 0;
    for (uint32_t i = 0; i < kHangSlots; ++i) {
      if (h[i].site != 0u) {
        ++n_hang;
        if (n_hang <= 64) {
          std::printf("TRACE_HANG site=%s bx=%u by=%u rank=%u warp=%u lane=%u addr=0x%08x off=%u parity=%u t=%u q=%u\n",
                      wait_site_name(h[i].site), h[i].bx, h[i].by, h[i].rank, h[i].warp, h[i].lane, h[i].addr, h[i].addr & 0x00FFFFFFu, h[i].parity, h[i].tile, h[i].ktile);
        }
      }
    }
    for (uint32_t i = 0; i < kHangSlots; ++i) {
      if (h[i].hb_valid != 0u) {   // B2 (E.7.2): every slot; bx/by/warp come from the slot index ((by*64 + bx)*8 + warp): HangRec.bx/by are written only by the hang path
        ++n_hb;
        std::printf("TRACE_HANG_HB bx=%u by=%u warp=%u tile=%u phase=%u\n", (i / 8u) % 64u, i / 512u, i % 8u, h[i].hb_tile, h[i].hb_phase);
      }
    }
    std::printf("TRACE_HOST explicit_hang_records %u heartbeats %u timed_out %d cuda_error %d %s\n", n_hang, n_hb, timed_out ? 1 : 0, int(e),
                e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    std::fflush(stdout);
    if (timed_out) { std::_Exit(3); }
  }
  return e == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
}
} // namespace
#endif

cutlass::Status run(ExplicitGemmParams const& params) {
  // E.5.1 guards: the kernel hard-codes the Section 0 configuration; refuse anything else loudly (device code has no checks)
  if (params.mode != kModeGemm) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.m != kFixedM || params.n != kFixedN || params.k != kFixedK || params.l != kFixedL) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.raster_order != kRasterOrderAlongN || params.swizzle_divisor != 0) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.tiles_m != kProblemTilesM || params.tiles_n != kProblemTilesN || params.tiles_l != kProblemTilesL) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.alpha_ptr != nullptr || params.beta_ptr != nullptr) { return cutlass::Status::kErrorNotSupported; }   // no pointer path (D.6 item 13)
  if (params.beta != 0.0f) { return cutlass::Status::kErrorNotSupported; }   // the C path is omitted (Section 0; E.10 item 5); -0.0f passes like is_zero()

  // E.5.2: the run() mirror (gemm_universal_adapter.h:375-575 -> cluster_launch.hpp:260-296)
  void const* kernel = reinterpret_cast<void const*>(&explicit_blackwell_fp16_gemm_kernel);
  cutlass::Status launch_result = cutlass::Status::kSuccess;

  // 1. MaxDynamicSharedMemorySize opt-in on the explicit kernel (initialize() does it for the CUTLASS kernel, :342-352); every call, not hoisted (E.5.4)
  launch_result = status_from_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes)));

  // 2. launch geometry (Section 5.4): grid (64,64,1) = the AlongN-transposed (N/128, M/128, 1); block (256,1,1); 230400 bytes of dynamic smem
  dim3 const grid(kGridX, kGridY, kGridZ);
  dim3 const block(kBlockThreads, 1, 1);
  dim3 const cluster(kClusterX, kClusterY, kClusterZ);

  // 3. check_cluster_dims (cluster_launch.hpp:103-112)
  if (launch_result == cutlass::Status::kSuccess) {
    bool const ok = (cluster.x * cluster.y * cluster.z <= 32u) && (grid.x % cluster.x == 0u) && (grid.y % cluster.y == 0u) && (grid.z % cluster.z == 0u);
    if (!ok) { launch_result = cutlass::Status::kInvalid; }
  }

  // 4. ClusterLauncher::init: NonPortableClusterSizeAllowed on every launch (cluster_launch.hpp:141-143, called from :280)
  if (launch_result == cutlass::Status::kSuccess) {
    launch_result = status_from_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  }

  // 5.-6. make_cluster_launch_config (:151-213): one attribute, cudaLaunchAttributeClusterDimension = {2,2,1}; legacy default stream; then cudaLaunchKernelExC (:294)
  if (launch_result == cutlass::Status::kSuccess) {
    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = kSmemBytes;
    launch_config.stream = nullptr;
    cudaLaunchAttribute launch_attribute[1];
    launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
    launch_attribute[0].val.clusterDim.x = cluster.x;
    launch_attribute[0].val.clusterDim.y = cluster.y;
    launch_attribute[0].val.clusterDim.z = cluster.z;
    launch_config.attrs = launch_attribute;
    launch_config.numAttrs = 1;
    void* kernel_params[] = { const_cast<void*>(static_cast<void const*>(&params)) };
    launch_result = status_from_cuda(cudaLaunchKernelExC(&launch_config, kernel, kernel_params));
  }

  // 7. cudaGetLastError -> kSuccess / kErrorInternal (gemm_universal_adapter.h:564-574)
  cudaError_t const result = cudaGetLastError();
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  if (result != cudaSuccess || launch_result != cutlass::Status::kSuccess) {
    std::printf("TRACE_HOST explicit_run_launch_error %d %s status %d\n", int(result), cudaGetErrorString(result), int(launch_result));
  }
  if (g_explicit_trace_armed && result == cudaSuccess && launch_result == cutlass::Status::kSuccess) {
    return explicit_trace_wait_and_check();   // toggle-on warm-up only: bounded wait, hang records, heartbeats (E.7.2 B1/B2)
  }
#endif
  if (cudaSuccess == result && cutlass::Status::kSuccess == launch_result) {
    return cutlass::Status::kSuccess;
  }
  return cutlass::Status::kErrorInternal;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// 12. [HOST] [TRACE]  Toggle-on host entry points called by the harness's guarded blocks (E.7.3): K0 probe kernel, sentinel fill of D (B3),
//     H3 on the explicit kernel, record dump, mismatch dump (B4). All recording happens in this translation unit's buffer.
//     The harness calls trace_begin() before the warm-up launch and trace_end() after verify(); the ten timed launches run with
//     recording disabled. Everything to the final #endif is compiled out with the toggle off.
///////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
void trace_begin(void* d_ptr, size_t d_bytes) {
  trace_reset();                                          // this TU's record buffer (static __device__), enabled from here on
  if (g_hang_host == nullptr) {
    trace_check(cudaHostAlloc(&g_hang_host, size_t(kHangSlots) * sizeof(HangRec), cudaHostAllocMapped), "hang buffer cudaHostAlloc");
  }
  std::memset(g_hang_host, 0, size_t(kHangSlots) * sizeof(HangRec));
  void* dev = nullptr;
  trace_check(cudaHostGetDevicePointer(&dev, g_hang_host, 0), "hang buffer device pointer");
  trace_check(cudaMemcpyToSymbol(g_explicit_hang_buf, &dev, sizeof(dev)), "hang buffer symbol");
  trace_probe_kernel_k0();                                // C.4.2: K0 addresses / mapa / TMEM base, recorded into this TU's buffer
  if (d_ptr != nullptr && d_bytes != 0) {
    trace_check(cudaMemset(d_ptr, 0xFF, d_bytes), "sentinel fill of D");   // B3: never-stored tiles read as NaN
    trace_check(cudaDeviceSynchronize(), "sentinel fill sync");
  }
  std::printf("TRACE_HOST explicit_kernel_name explicit_blackwell_fp16_gemm_kernel\n");
  std::printf("TRACE_HOST sizeof_ExplicitGemmParams %zu\n", sizeof(ExplicitGemmParams));
  std::printf("TRACE_HOST explicit_trace 1\n");
  g_explicit_trace_armed = true;
}

namespace {
// B4: classify the mismatches of D against the reference (column-major, element (m, n) at n * M + m; exact integers, Section 8)
void trace_mismatch_dump(float const* d_dev, float const* ref_dev, int m, int n) {
  size_t const count = size_t(m) * size_t(n);
  std::vector<uint32_t> d(count), ref(count);
  trace_check(cudaMemcpy(d.data(), d_dev, count * sizeof(float), cudaMemcpyDeviceToHost), "B4 copy D");
  trace_check(cudaMemcpy(ref.data(), ref_dev, count * sizeof(float), cudaMemcpyDeviceToHost), "B4 copy ref_D");
  auto as_f = [](uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; };
  size_t total_val = 0, total_bits = 0, sentinel = 0, sign_zero = 0;
  int const tiles_m = m / 128, tiles_n = n / 128;
  std::vector<int> bad_per_tile(size_t(tiles_m) * size_t(tiles_n), 0), nan_per_tile(size_t(tiles_m) * size_t(tiles_n), 0);
  for (size_t idx = 0; idx < count; ++idx) {
    float const fd = as_f(d[idx]), fr = as_f(ref[idx]);
    bool const val_diff = !(fd == fr);
    if (d[idx] != ref[idx]) { ++total_bits; }
    if (val_diff) {
      ++total_val;
      size_t const mm = idx % size_t(m), nn = idx / size_t(m);
      ++bad_per_tile[size_t(mm / 128) * size_t(tiles_n) + nn / 128];
      if (d[idx] == 0xFFFFFFFFu) { ++sentinel; ++nan_per_tile[size_t(mm / 128) * size_t(tiles_n) + nn / 128]; }
    } else if (d[idx] != ref[idx]) { ++sign_zero; }
  }
  std::printf("TRACE_B4 mismatches value=%zu bitwise=%zu sentinel_nan=%zu sign_of_zero_only=%zu of %zu\n", total_val, total_bits, sentinel, sign_zero, count);
  if (total_val == 0) { return; }
  std::printf("TRACE_B4 tile map (rows tm = 0..%d, cols tn = 0..%d): '.' clean, '#' all wrong, 'x' partial, 'N' sentinel\n", tiles_m - 1, tiles_n - 1);
  int first_tm = -1, first_tn = -1;
  for (int tm = 0; tm < tiles_m; ++tm) {
    std::printf("TRACE_B4 ");
    for (int tn = 0; tn < tiles_n; ++tn) {
      int const bad = bad_per_tile[size_t(tm) * size_t(tiles_n) + tn], nan_cnt = nan_per_tile[size_t(tm) * size_t(tiles_n) + tn];
      char c = bad == 0 ? '.' : (nan_cnt == 128 * 128 ? 'N' : (bad == 128 * 128 ? '#' : 'x'));
      if (bad != 0 && first_tm < 0) { first_tm = tm; first_tn = tn; }
      std::printf("%c", c);
    }
    std::printf("\n");
  }
  if (first_tm < 0) { return; }
  // first bad tile: mismatch fraction per 64-row half, 32-row quarter (epilogue warp), 16-column subtile and (row % 32, col % 4) class
  int half[2] = {0, 0}, quarter[4] = {0, 0, 0, 0}, subtile[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::map<int, int> ratio_hist;
  int rowmod_colmod[32][4];
  std::memset(rowmod_colmod, 0, sizeof(rowmod_colmod));
  for (int rr = 0; rr < 128; ++rr) {
    for (int cc = 0; cc < 128; ++cc) {
      size_t const idx = size_t(128 * first_tn + cc) * size_t(m) + size_t(128 * first_tm + rr);
      float const fd = as_f(d[idx]), fr = as_f(ref[idx]);
      if (fd == fr) { continue; }
      ++half[rr / 64]; ++quarter[rr / 32]; ++subtile[cc / 16]; ++rowmod_colmod[rr % 32][cc % 4];
      if (fr != 0.0f && std::isfinite(fd)) {
        double const q = double(fd) / double(fr) * 128.0;
        if (std::fabs(q - std::round(q)) < 1e-6 && std::fabs(q) <= 4096.0) { ++ratio_hist[int(std::round(q))]; }
      }
    }
  }
  std::printf("TRACE_B4 first bad tile (tm=%d, tn=%d): halves %d/%d quarters %d/%d/%d/%d subtiles %d/%d/%d/%d/%d/%d/%d/%d (of 8192/4096/2048)\n",
              first_tm, first_tn, half[0], half[1], quarter[0], quarter[1], quarter[2], quarter[3],
              subtile[0], subtile[1], subtile[2], subtile[3], subtile[4], subtile[5], subtile[6], subtile[7]);
  std::printf("TRACE_B4 first bad tile col%%4 classes (row%%32 summed): %d %d %d %d\n",
              [&] { int s = 0; for (int r = 0; r < 32; ++r) s += rowmod_colmod[r][0]; return s; }(),
              [&] { int s = 0; for (int r = 0; r < 32; ++r) s += rowmod_colmod[r][1]; return s; }(),
              [&] { int s = 0; for (int r = 0; r < 32; ++r) s += rowmod_colmod[r][2]; return s; }(),
              [&] { int s = 0; for (int r = 0; r < 32; ++r) s += rowmod_colmod[r][3]; return s; }());
  int shown = 0;
  for (auto const& kv : ratio_hist) {
    if (shown++ < 8) { std::printf("TRACE_B4 ratio D/ref = %d/128 : %d elements\n", kv.first, kv.second); }
  }
  // does the first bad tile hold the reference of another tile?
  for (int tm2 = 0; tm2 < tiles_m; ++tm2) {
    for (int tn2 = 0; tn2 < tiles_n; ++tn2) {
      bool same = true;
      for (int cc = 0; cc < 128 && same; ++cc) {
        size_t const a = size_t(128 * first_tn + cc) * size_t(m) + size_t(128 * first_tm);
        size_t const b = size_t(128 * tn2 + cc) * size_t(m) + size_t(128 * tm2);
        for (int rr = 0; rr < 128; ++rr) {
          if (!(as_f(d[a + rr]) == as_f(ref[b + rr]))) { same = false; break; }
        }
      }
      if (same) { std::printf("TRACE_B4 tile (%d,%d) holds the reference of tile (%d,%d)\n", first_tm, first_tn, tm2, tn2); }
    }
  }
}
} // namespace

void trace_end(char const* csv_path, bool passed, float const* d_ptr, float const* ref_d_ptr, int m, int n) {
  g_explicit_trace_armed = false;
  trace_host_post_run_fn(reinterpret_cast<void const*>(&explicit_blackwell_fp16_gemm_kernel));   // H3 on the explicit kernel (attributes, occupancy)
  trace_dump(csv_path);                                   // K0 + warm-up launch records of this TU's buffer
  trace_disable();                                        // the 10 timed launches run without probe traffic
  if (!passed && d_ptr != nullptr && ref_d_ptr != nullptr) { trace_mismatch_dump(d_ptr, ref_d_ptr, m, n); }
  std::fflush(stdout);
}
#endif

} // namespace explicit_gemm
