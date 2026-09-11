/***************************************************************************************************
 * De-abstraction trace infrastructure for examples/70_blackwell_gemm
 * (Part C of examples/70_blackwell_gemm/Semantics-preserving-de-abstraction.md).
 *
 * Included by 70_blackwell_fp16_gemm.cu and 70_blackwell_fp16_gemm_explicit.cu BEFORE any CUTLASS header, and
 * only when CUTLASS_DEABSTRACTION_TRACE is defined by the one-line toggle on line 1 of
 * 70_blackwell_fp16_gemm_explicit_util.hpp (Part E).  The guarded probe blocks inside include/cute and
 * include/cutlass reference the macros and symbols declared here.  With the toggle off nothing in this file is
 * compiled.
 *
 * Two translation units (Part E, E.7.3): the device globals are `static __device__` and the host functions that
 * touch them are `static inline`, so every TU owns a private copy (no -rdc).  Recording, K0 and the dump run in the
 * kernel TU (explicit_gemm::trace_begin / trace_end); the harness TU's copies serve only the guarded blocks inside
 * the never-launched CUTLASS kernel.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// One fixed-size record per probe event (128 bytes). Column order of the CSV written by trace_dump():
//   kind,bx,by,rank,warp,lane,smid,seq,t,v0,...,v9
struct TraceRec {
  uint32_t kind;   // TraceKind
  uint32_t bx;     // blockIdx.x
  uint32_t by;     // blockIdx.y
  uint32_t rank;   // %cluster_ctarank
  uint32_t warp;   // threadIdx.x / 32
  uint32_t lane;   // threadIdx.x % 32
  uint32_t smid;   // %smid
  uint32_t seq;    // per-kind sequence number (atomic)
  uint64_t t;      // %globaltimer (ns)
  uint64_t v[10];  // payload, meaning per kind (see below and check_trace.py)
  uint64_t pad_;   // tail padding to 128 bytes (unused, not written to the CSV)
};
static_assert(sizeof(TraceRec) == 128, "TraceRec must be 128 bytes");

// Payload layout per kind (v0..v9):
//  K_SMEM   (K1a): smem base, &mainloop.full[0], &mainloop.empty[0], &clc.full[0], &clc.empty[0],
//                  &accumulator.empty[0], &tmem_dealloc, &clc_response[0], smem_A, smem_B
//                  (all 32-bit shared::cta addresses of the recording CTA)
//  K_SMEM2  (K1b): smem_D, &tmem_base_ptr, is_epi_load_needed, is_first_cta_in_cluster (the CTA whose warp 1 schedules), cta_rank_in_cluster,
//                  mma_peer_cta_rank, &accumulator.full[0], &clc_throttle.full[0], &load_order[0][0], &epi_load.full[0]
//  K_TMEM   (K2) : tmem_base_ptr, cta_rank_in_cluster, is_mma_leader_cta, site (0 = MMA warp, 1 = epilogue warps)
//  K_MMA    (K3) : desc_a, desc_b, tmem_c, idesc (32-bit), scale_c
//  K_TMA_LOAD  (K4a): c0, c1, c2, dst smem address, size(src) in elements (8192 = A, 4096 = B), mbarrier smem address (unmasked)
//  K_TMA_STORE (K4b): c0, c1, c2, src smem address, popc(activemask) at the issue point (32 = every lane of the warp issues)
//  K_TMA_STORE_LANES: reserved (unused)
//  K_CLC_ISSUE (K5a): state.index, state.phase, state.count, mbarrier_addr, blockIdx.x/2, blockIdx.y/2
//  K_CLC_SCHED (K5b), K_CLC_MMA (K5c): raw M_idx, raw N_idx, raw L_idx, valid, state.index, state.phase, state.count,
//                  swizzled M_idx, swizzled N_idx
//  K_PROBE0 (K0) : v9 = 0: smem base, mapa(base, rank ^ 1), mapa(base, 0), base & 0xFEFFFFFF, cluster_ctaid.x, cluster_ctaid.y, smid
//                  v9 = 1: tmem base returned by tcgen05.alloc, rank, smem base
//  K_TAIL   (E.7.3, explicit kernel only): tail site id (TAIL_MAINLOOP 10, TAIL_ACC 11, TAIL_CLC 12, WAIT_TMEM_DEALLOC 9),
//                  pipeline index and parity of the wait that just completed (recorded after the wait returns and BEFORE the
//                  state advance; check_trace.py k_tail_mainloop_parity expects v2 == 1 for every mainloop step), tile counter;
//                  one record per completed tail step, lane 0 (WAIT_TMEM_DEALLOC records idx = phase = 0)
enum TraceKind : uint32_t {
  K_SMEM = 1,
  K_TMEM = 2,
  K_MMA = 3,
  K_TMA_LOAD = 4,
  K_TMA_STORE = 5,
  K_TMA_STORE_LANES = 6,
  K_CLC_ISSUE = 7,
  K_CLC_SCHED = 8,
  K_CLC_MMA = 9,
  K_PROBE0 = 10,
  K_SMEM2 = 11,
  K_TAIL = 13,
  K_COUNT = 16
};

constexpr uint32_t kTraceCapacity = 1u << 16;   // 65536 records = 8 MiB

static __device__ TraceRec g_trace_buf[kTraceCapacity];   // static: one private copy per translation unit (E.7.3)
static __device__ uint32_t g_trace_next = 0;      // number of records appended
static __device__ uint32_t g_trace_enabled = 0;   // 1 between trace_reset() and trace_disable()
static __device__ uint32_t g_trace_seq[K_COUNT] = {};   // per-kind counters (count every event until the kind's cap is reached)
static __device__ uint32_t g_trace_full[K_COUNT] = {};  // set to 1 once a kind reached its cap: later events skip the atomics (a stale 0 only costs one more atomic)

// Host-side counter for the encoder probe H2b (include/cute/atom/copy_traits_sm90_tma.hpp)
inline int g_trace_encode_count = 0;

//
// Device helpers
//
__device__ __forceinline__ uint32_t trace_cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t trace_smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ uint64_t trace_globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// shared::cluster address of `smem_addr` (a shared::cta address of the executing CTA) in CTA `cta_rank`
__device__ __forceinline__ uint32_t trace_mapa(uint32_t smem_addr, uint32_t cta_rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(cta_rank));
  return r;
}
__device__ __forceinline__ bool trace_in_first_cluster() {
  return blockIdx.x < 2 && blockIdx.y < 2;
}
__device__ __forceinline__ TraceRec
trace_make_rec(uint32_t kind, uint32_t seq,
               uint64_t v0 = 0, uint64_t v1 = 0, uint64_t v2 = 0, uint64_t v3 = 0, uint64_t v4 = 0,
               uint64_t v5 = 0, uint64_t v6 = 0, uint64_t v7 = 0, uint64_t v8 = 0, uint64_t v9 = 0) {
  TraceRec r;
  r.kind = kind;
  r.bx = blockIdx.x;
  r.by = blockIdx.y;
  r.rank = trace_cluster_ctarank();
  r.warp = threadIdx.x / 32;
  r.lane = threadIdx.x % 32;
  r.smid = trace_smid();
  r.seq = seq;
  r.t = trace_globaltimer();
  r.v[0] = v0; r.v[1] = v1; r.v[2] = v2; r.v[3] = v3; r.v[4] = v4;
  r.v[5] = v5; r.v[6] = v6; r.v[7] = v7; r.v[8] = v8; r.v[9] = v9;
  r.pad_ = 0;
  return r;
}

// The probe macros expand to device code only in the device compilation pass; in the host pass they are
// empty, so they can be placed inside CUTE_HOST_DEVICE functions without host-side references to device symbols.
#if defined(__CUDA_ARCH__)
#define TRACE_RECORD(kind, cap, ...)                                                        \
  do {                                                                                     \
    if (::g_trace_enabled && !::g_trace_full[(kind)]) {                                    \
      uint32_t trace_s_ = atomicAdd(&::g_trace_seq[(kind)], 1u);                           \
      if (trace_s_ < (cap)) {                                                              \
        uint32_t trace_i_ = atomicAdd(&::g_trace_next, 1u);                                \
        if (trace_i_ < ::kTraceCapacity) {                                                 \
          ::g_trace_buf[trace_i_] = ::trace_make_rec((kind), trace_s_, __VA_ARGS__);       \
        }                                                                                  \
      } else {                                                                             \
        ::g_trace_full[(kind)] = 1u;                                                       \
      }                                                                                    \
    }                                                                                      \
  } while (0)
#define TRACE_COUNT(kind)                                                                  \
  do {                                                                                     \
    if (::g_trace_enabled) {                                                               \
      atomicAdd(&::g_trace_seq[(kind)], 1u);                                               \
    }                                                                                      \
  } while (0)
#define TRACE_IN_FIRST_CLUSTER() (::trace_in_first_cluster())
#else
#define TRACE_RECORD(kind, cap, ...) do { } while (0)
#define TRACE_COUNT(kind) do { } while (0)
#define TRACE_IN_FIRST_CLUSTER() (false)
#endif

//
// Host side
//
inline void trace_check(cudaError_t e, char const* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "TRACE_ERROR %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Clear the buffer and counters and enable recording (call before the launch to be traced).
// static: operates on the calling translation unit's buffer (E.7.3).
static inline void trace_reset() {
  trace_check(cudaDeviceSynchronize(), "trace_reset sync");
  uint32_t zero = 0, one = 1;
  uint32_t zeros[K_COUNT] = {};
  trace_check(cudaMemcpyToSymbol(g_trace_next, &zero, sizeof(zero)), "trace_reset next");
  trace_check(cudaMemcpyToSymbol(g_trace_seq, zeros, sizeof(zeros)), "trace_reset seq");
  trace_check(cudaMemcpyToSymbol(g_trace_full, zeros, sizeof(zeros)), "trace_reset full");
  trace_check(cudaMemcpyToSymbol(g_trace_enabled, &one, sizeof(one)), "trace_reset enable");
}

// Stop recording (call before the timed launches so they run without probe traffic).
static inline void trace_disable() {
  trace_check(cudaDeviceSynchronize(), "trace_disable sync");
  uint32_t zero = 0;
  trace_check(cudaMemcpyToSymbol(g_trace_enabled, &zero, sizeof(zero)), "trace_disable");
}

// Write all records to `path` as CSV; the per-kind counters go to stdout as TRACE_HOST lines and as '#' comment
// lines at the top of the CSV.
static inline void trace_dump(char const* path) {
  trace_check(cudaDeviceSynchronize(), "trace_dump sync");
  uint32_t n = 0;
  uint32_t seq[K_COUNT] = {};
  trace_check(cudaMemcpyFromSymbol(&n, g_trace_next, sizeof(n)), "trace_dump next");
  trace_check(cudaMemcpyFromSymbol(seq, g_trace_seq, sizeof(seq)), "trace_dump seq");
  uint32_t const n_stored = n < kTraceCapacity ? n : kTraceCapacity;
  std::vector<TraceRec> h(n_stored);
  if (n_stored > 0) {
    trace_check(cudaMemcpyFromSymbol(h.data(), g_trace_buf, size_t(n_stored) * sizeof(TraceRec)), "trace_dump buf");
  }
  FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    std::fprintf(stderr, "TRACE_ERROR cannot open %s\n", path);
    std::exit(1);
  }
  std::fprintf(f, "# trace records: appended=%u stored=%u\n", n, n_stored);
  for (uint32_t k = 0; k < K_COUNT; ++k) {
    std::fprintf(f, "# seq[%u]=%u\n", k, seq[k]);
    std::printf("TRACE_HOST trace_seq_%u %u\n", k, seq[k]);
  }
  std::printf("TRACE_HOST trace_records %u\n", n_stored);
  std::fprintf(f, "kind,bx,by,rank,warp,lane,smid,seq,t,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9\n");
  for (uint32_t i = 0; i < n_stored; ++i) {
    TraceRec const& r = h[i];
    std::fprintf(f, "%u,%u,%u,%u,%u,%u,%u,%u,%llu", r.kind, r.bx, r.by, r.rank, r.warp, r.lane, r.smid, r.seq,
                 (unsigned long long)r.t);
    for (int j = 0; j < 10; ++j) {
      std::fprintf(f, ",%llu", (unsigned long long)r.v[j]);
    }
    std::fprintf(f, "\n");
  }
  std::fclose(f);
  std::printf("TRACE_HOST trace_dump %s\n", path);
}
