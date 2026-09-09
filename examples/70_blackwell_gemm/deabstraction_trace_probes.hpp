/***************************************************************************************************
 * De-abstraction trace probes that need the CUTLASS/CuTe types: the standalone address probe kernel
 * (K0), the GDC probe kernel, and the host probes H1/H2/H3/H5 (Part C, C.4.1 and C.4.2 of
 * examples/70_blackwell_gemm/Semantics-preserving-de-abstraction.md).
 *
 * Included by 70_blackwell_fp16_gemm.cu AFTER the CUTLASS headers, only when
 * CUTLASS_DEABSTRACTION_TRACE is defined.  Output lines are prefixed TRACE_* so that check_trace.py
 * can parse them from the captured stdout (host.txt).
 **************************************************************************************************/
#pragma once

#include "deabstraction_trace.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/util.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/grid_dependency_control.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// K0: standalone address / TMEM probe kernel.  Same launch shape as the GEMM kernel (256 threads,
// 230400 bytes of dynamic shared memory, no static shared memory, cluster (2,2,1)) but no CUTLASS
// kernel code, so the hardware facts of Section B.6 are measured before the real kernel is touched.
///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void __launch_bounds__(256, 1) trace_probe_addr_kernel(int /*unused*/) {
  extern __shared__ char trace_probe_smem[];
  uint32_t const rank = cute::block_rank_in_cluster();
  dim3 const cid = cute::block_id_in_cluster();
  uint32_t const base = cute::cast_smem_ptr_to_uint(trace_probe_smem);
  uint32_t const mapa_peer = trace_mapa(base, rank ^ 1u);
  uint32_t const mapa_rank0 = trace_mapa(base, 0u);
  if (threadIdx.x == 0) {
    TRACE_RECORD(K_PROBE0, 64, base, mapa_peer, mapa_rank0, base & 0xFEFFFFFFu, cid.x, cid.y, trace_smid(), 0, 0, 0);
  }
  // TMEM allocation exactly as the GEMM kernel does it: the whole warp 0 of both CTAs of a pair
  // allocates all 512 columns, the result lands in shared memory.
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(trace_probe_smem);
  int const warp = threadIdx.x / 32;
  cute::TMEM::Allocator2Sm allocator;
  if (warp == 0) {
    allocator.allocate(cute::TMEM::Allocator2Sm::Sm100TmemCapacityColumns, tmem_slot);
    __syncwarp();
  }
  cute::cluster_sync();
  uint32_t const tmem_base = *tmem_slot;
  if (warp == 0 && (threadIdx.x % 32) == 0) {
    TRACE_RECORD(K_PROBE0, 64, tmem_base, rank, base, 0, 0, 0, 0, 0, 0, 1);
    printf("TRACE_K0 block %u %u rank %u smem_base 0x%08x mapa_peer 0x%08x mapa_rank0 0x%08x masked 0x%08x tmem_base 0x%08x smid %u\n",
           blockIdx.x, blockIdx.y, rank, base, mapa_peer, mapa_rank0, base & 0xFEFFFFFFu, tmem_base, trace_smid());
  }
  cute::cluster_sync();
  if (warp == 0) {
    allocator.release_allocation_lock();
    allocator.free(tmem_base, cute::TMEM::Allocator2Sm::Sm100TmemCapacityColumns);
  }
}

// Device-side view of the build configuration (the GDC flag is arch-dependent and therefore invisible to host code).
__global__ void trace_probe_gdc_kernel() {
  int feat_sm100_all = 0;
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  feat_sm100_all = 1;
#endif
  int cuda_arch = 0;
#if defined(__CUDA_ARCH__)
  cuda_arch = __CUDA_ARCH__;
#endif
  printf("TRACE_DEV IsGdcGloballyEnabled %d FEAT_SM100_ALL %d CUDA_ARCH %d\n",
         (int)cutlass::arch::IsGdcGloballyEnabled, feat_sm100_all, cuda_arch);
}

inline void trace_probe_kernel_k0() {
  void const* fn = reinterpret_cast<void const*>(&trace_probe_addr_kernel);
  trace_check(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 230400), "K0 cudaFuncSetAttribute smem");
  trace_check(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1), "K0 cudaFuncSetAttribute cluster");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(4, 4, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = 230400;
  cfg.stream = nullptr;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 2;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  trace_check(cudaLaunchKernelEx(&cfg, trace_probe_addr_kernel, 0), "K0 launch");
  trace_check(cudaDeviceSynchronize(), "K0 sync");
  trace_probe_gdc_kernel<<<1, 1>>>();
  trace_check(cudaDeviceSynchronize(), "GDC probe sync");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Host probes
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void trace_print_type(char const* label) {
  std::printf("TRACE_TYPE %s | %s\n", label, __PRETTY_FUNCTION__);
}

template <class L>
void trace_print_layout(char const* label, L const& l) {
  std::printf("TRACE_H1 %s ", label);
  cute::print(l);
  std::printf("\n");
}

inline void trace_dump_tensormap(char const* name, void const* desc) {
  uint64_t const* w = reinterpret_cast<uint64_t const*>(desc);
  std::printf("TRACE_TMAP %s", name);
  for (int i = 0; i < 16; ++i) {
    std::printf(" %016llx", (unsigned long long)w[i]);
  }
  std::printf("\n");
}

// H1 (types, sizes, offsets), H2 (tensor maps), H5 (build macros).  Call after the first gemm.initialize().
template <class Gemm>
void trace_host_probes(Gemm const& gemm) {
  using GemmKernel = typename Gemm::GemmKernel;
  using Mainloop = typename GemmKernel::CollectiveMainloop;
  using Epilogue = typename GemmKernel::CollectiveEpilogue;
  using SharedStorage = typename GemmKernel::SharedStorage;

  // ---- H1: resolved types ----
  trace_print_layout("TiledMma", typename GemmKernel::TiledMma{});
  trace_print_layout("SmemLayoutA", typename Mainloop::SmemLayoutA{});
  trace_print_layout("SmemLayoutB", typename Mainloop::SmemLayoutB{});
  trace_print_layout("SmemLayoutAtomC", typename Epilogue::SmemLayoutAtomC{});
  trace_print_layout("SmemLayoutAtomD", typename Epilogue::SmemLayoutAtomD{});
  trace_print_layout("EpilogueTile", typename GemmKernel::EpilogueTile{});
  trace_print_layout("TileShape", typename GemmKernel::TileShape{});
  trace_print_layout("CtaShape_MNK", typename GemmKernel::CtaShape_MNK{});
  trace_print_layout("AtomThrShapeMNK", typename GemmKernel::AtomThrShapeMNK{});
  trace_print_layout("ClusterShape", typename GemmKernel::ClusterShape{});
  trace_print_type<Mainloop>("CollectiveMainloop");
  trace_print_type<Epilogue>("CollectiveEpilogue");
  trace_print_type<typename GemmKernel::TileScheduler>("TileScheduler");
  trace_print_type<typename GemmKernel::DispatchPolicy>("MainloopDispatchPolicy");
  trace_print_type<typename Epilogue::DispatchPolicy>("EpilogueDispatchPolicy");
  trace_print_type<typename Epilogue::CopyOpT2R>("CopyOpT2R");
  trace_print_type<typename Epilogue::CopyOpG2S>("CopyOpG2S");
  trace_print_type<typename Epilogue::CopyOpS2G>("CopyOpS2G");
  trace_print_type<typename Epilogue::CopyOpS2R>("CopyOpS2R");
  trace_print_type<typename Epilogue::CopyOpR2S>("CopyOpR2S");
  trace_print_type<typename Mainloop::GmemTiledCopyA>("GmemTiledCopyA");
  trace_print_type<typename Mainloop::GmemTiledCopyB>("GmemTiledCopyB");
  trace_print_type<typename GemmKernel::MainloopPipeline>("MainloopPipeline");
  trace_print_type<typename GemmKernel::AccumulatorPipeline>("AccumulatorPipeline");
  trace_print_type<typename GemmKernel::CLCPipeline>("CLCPipeline");
  trace_print_type<typename GemmKernel::CLCThrottlePipeline>("CLCThrottlePipeline");
  trace_print_type<typename Epilogue::LoadPipeline>("EpiLoadPipeline");
  trace_print_type<typename Epilogue::StorePipeline>("EpiStorePipeline");

  // ---- H1: sizes ----
  std::printf("TRACE_HOST sizeof_SharedStorage %zu\n", sizeof(SharedStorage));
  std::printf("TRACE_HOST SharedStorageSize %d\n", int(GemmKernel::SharedStorageSize));
  std::printf("TRACE_HOST alignof_SharedStorage %zu\n", alignof(SharedStorage));
  std::printf("TRACE_HOST sizeof_PipelineStorage %zu\n", sizeof(typename SharedStorage::PipelineStorage));
  std::printf("TRACE_HOST sizeof_TensorStorage %zu\n", sizeof(typename SharedStorage::TensorStorage));
  std::printf("TRACE_HOST sizeof_EpilogueSharedStorage %zu\n", sizeof(typename Epilogue::SharedStorage));
  std::printf("TRACE_HOST sizeof_Params %zu\n", sizeof(typename GemmKernel::Params));
  std::printf("TRACE_HOST sizeof_MainloopParams %zu\n", sizeof(typename GemmKernel::MainloopParams));
  std::printf("TRACE_HOST sizeof_EpilogueParams %zu\n", sizeof(typename GemmKernel::EpilogueParams));
  std::printf("TRACE_HOST sizeof_TileSchedulerParams %zu\n", sizeof(typename GemmKernel::TileSchedulerParams));
  std::printf("TRACE_HOST MaxThreadsPerBlock %u\n", unsigned(GemmKernel::MaxThreadsPerBlock));
  std::printf("TRACE_HOST MinBlocksPerMultiprocessor %u\n", unsigned(GemmKernel::MinBlocksPerMultiprocessor));
  std::printf("TRACE_HOST NumEpilogueSubTiles %u\n", unsigned(GemmKernel::NumEpilogueSubTiles));
  std::printf("TRACE_HOST CLCResponseSize %u\n", unsigned(GemmKernel::CLCResponseSize));
  std::printf("TRACE_HOST SchedulerPipelineStageCount %u\n", unsigned(GemmKernel::SchedulerPipelineStageCount));
  std::printf("TRACE_HOST AccumulatorPipelineStageCount %u\n", unsigned(GemmKernel::AccumulatorPipelineStageCount));
  std::printf("TRACE_HOST IsOverlappingAccum %d\n", int(GemmKernel::IsOverlappingAccum));
  std::printf("TRACE_HOST IsGdcEnabled_host_view %d\n", int(GemmKernel::IsGdcEnabled));

  // ---- H1: member offsets (address differences inside an aligned host buffer; no member is read) ----
  {
    alignas(1024) static char trace_ss_buf[sizeof(SharedStorage)];
    SharedStorage* ss = reinterpret_cast<SharedStorage*>(trace_ss_buf);
    auto off = [&](void const* p) -> size_t { return size_t(static_cast<char const*>(p) - trace_ss_buf); };
    std::printf("TRACE_HOST off_pipelines %zu\n", off(&ss->pipelines));
    std::printf("TRACE_HOST off_mainloop %zu\n", off(&ss->pipelines.mainloop));
    std::printf("TRACE_HOST off_mainloop_full0 %zu\n", off(&ss->pipelines.mainloop.full_barrier_[0]));
    std::printf("TRACE_HOST off_mainloop_empty0 %zu\n", off(&ss->pipelines.mainloop.empty_barrier_[0]));
    std::printf("TRACE_HOST off_epi_load %zu\n", off(&ss->pipelines.epi_load));
    std::printf("TRACE_HOST off_epi_load_full0 %zu\n", off(&ss->pipelines.epi_load.full_barrier_[0]));
    std::printf("TRACE_HOST off_epi_load_empty0 %zu\n", off(&ss->pipelines.epi_load.empty_barrier_[0]));
    std::printf("TRACE_HOST off_load_order %zu\n", off(&ss->pipelines.load_order));
    std::printf("TRACE_HOST off_load_order_b00 %zu\n", off(&ss->pipelines.load_order.barrier_[0][0]));
    std::printf("TRACE_HOST off_load_order_b01 %zu\n", off(&ss->pipelines.load_order.barrier_[0][1]));
    std::printf("TRACE_HOST off_clc %zu\n", off(&ss->pipelines.clc));
    std::printf("TRACE_HOST off_clc_full0 %zu\n", off(&ss->pipelines.clc.full_barrier_[0]));
    std::printf("TRACE_HOST off_clc_empty0 %zu\n", off(&ss->pipelines.clc.empty_barrier_[0]));
    std::printf("TRACE_HOST off_accumulator %zu\n", off(&ss->pipelines.accumulator));
    std::printf("TRACE_HOST off_accumulator_full0 %zu\n", off(&ss->pipelines.accumulator.full_barrier_[0]));
    std::printf("TRACE_HOST off_accumulator_empty0 %zu\n", off(&ss->pipelines.accumulator.empty_barrier_[0]));
    std::printf("TRACE_HOST off_clc_throttle %zu\n", off(&ss->pipelines.clc_throttle));
    std::printf("TRACE_HOST off_clc_throttle_full0 %zu\n", off(&ss->pipelines.clc_throttle.full_barrier_[0]));
    std::printf("TRACE_HOST off_clc_throttle_empty0 %zu\n", off(&ss->pipelines.clc_throttle.empty_barrier_[0]));
    std::printf("TRACE_HOST off_tmem_dealloc %zu\n", off(&ss->pipelines.tmem_dealloc));
    std::printf("TRACE_HOST off_clc_response0 %zu\n", off(&ss->clc_response[0]));
    std::printf("TRACE_HOST off_clc_response1 %zu\n", off(&ss->clc_response[1]));
    std::printf("TRACE_HOST off_tmem_base_ptr %zu\n", off(&ss->tmem_base_ptr));
    std::printf("TRACE_HOST off_tensors %zu\n", off(&ss->tensors));
    std::printf("TRACE_HOST off_tensors_epilogue %zu\n", off(&ss->tensors.epilogue));
    std::printf("TRACE_HOST off_smem_C %zu\n", off(&ss->tensors.epilogue.collective.smem_C));
    std::printf("TRACE_HOST off_smem_D %zu\n", off(&ss->tensors.epilogue.collective.smem_D));
    std::printf("TRACE_HOST off_epilogue_thread %zu\n", off(&ss->tensors.epilogue.thread));
    std::printf("TRACE_HOST off_tensors_mainloop %zu\n", off(&ss->tensors.mainloop));
    std::printf("TRACE_HOST off_smem_A %zu\n", off(&ss->tensors.mainloop.smem_A));
    std::printf("TRACE_HOST off_smem_B %zu\n", off(&ss->tensors.mainloop.smem_B));
  }

  // ---- H2: tensor maps as encoded by the driver, scheduler params, grid ----
  auto const& p = gemm.params();
  trace_dump_tensormap("A", &p.mainloop.tma_load_a.tma_desc_);
  trace_dump_tensormap("B", &p.mainloop.tma_load_b.tma_desc_);
  trace_dump_tensormap("A_fallback", &p.mainloop.tma_load_a_fallback.tma_desc_);
  trace_dump_tensormap("B_fallback", &p.mainloop.tma_load_b_fallback.tma_desc_);
  trace_dump_tensormap("C", &p.epilogue.tma_load_c.tma_desc_);
  trace_dump_tensormap("D", &p.epilogue.tma_store_d.tma_desc_);
  std::printf("TRACE_H2 tma_load_a "); cute::print(p.mainloop.tma_load_a); std::printf("\n");
  std::printf("TRACE_H2 tma_load_b "); cute::print(p.mainloop.tma_load_b); std::printf("\n");
  std::printf("TRACE_H2 tma_load_c "); cute::print(p.epilogue.tma_load_c); std::printf("\n");
  std::printf("TRACE_H2 tma_store_d "); cute::print(p.epilogue.tma_store_d); std::printf("\n");
  std::printf("TRACE_HOST scheduler_problem_tiles %d %d %d\n", int(p.scheduler.problem_tiles_m_), int(p.scheduler.problem_tiles_n_), int(p.scheduler.problem_tiles_l_));
  std::printf("TRACE_HOST scheduler_cluster_divisors %d %d\n", int(p.scheduler.divmod_cluster_shape_m_.divisor), int(p.scheduler.divmod_cluster_shape_n_.divisor));
  std::printf("TRACE_HOST scheduler_swizzle_divisor %d\n", int(p.scheduler.divmod_swizzle_size_.divisor));
  {
    using TraceRasterOrder = decltype(p.scheduler.raster_order_);
    std::printf("TRACE_HOST scheduler_raster_order %s %d\n",
                p.scheduler.raster_order_ == TraceRasterOrder::AlongN ? "AlongN"
                : p.scheduler.raster_order_ == TraceRasterOrder::AlongM ? "AlongM" : "Other",
                int(p.scheduler.raster_order_));
  }
  std::printf("TRACE_HOST scheduler_log_swizzle_size %d\n", int(p.scheduler.log_swizzle_size_));
  {
    dim3 grid = Gemm::get_grid_shape(p);
    dim3 block = GemmKernel::get_block_shape();
    std::printf("TRACE_HOST grid %u %u %u\n", grid.x, grid.y, grid.z);
    std::printf("TRACE_HOST block %u %u %u\n", block.x, block.y, block.z);
  }
  {
    int drv = 0, rt = 0;
    trace_check(cudaDriverGetVersion(&drv), "cudaDriverGetVersion");
    trace_check(cudaRuntimeGetVersion(&rt), "cudaRuntimeGetVersion");
    std::printf("TRACE_HOST driver_version %d\n", drv);
    std::printf("TRACE_HOST runtime_version %d\n", rt);
  }

  // ---- H5: build configuration as seen by the host compiler ----
#if defined(NDEBUG)
  std::printf("TRACE_HOST macro_NDEBUG 1\n");
#else
  std::printf("TRACE_HOST macro_NDEBUG 0\n");
#endif
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
  std::printf("TRACE_HOST macro_CUDA_API_PER_THREAD_DEFAULT_STREAM 1\n");
#else
  std::printf("TRACE_HOST macro_CUDA_API_PER_THREAD_DEFAULT_STREAM 0\n");
#endif
#if defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 1\n");
#else
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 0\n");
#endif
#if defined(CUTLASS_ENABLE_GDC_FOR_SM100)
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_GDC_FOR_SM100 1\n");
#else
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_GDC_FOR_SM100 0\n");
#endif
#if defined(CUTLASS_ENABLE_SYNCLOG)
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_SYNCLOG 1\n");
#else
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_SYNCLOG 0\n");
#endif
#if defined(CUTLASS_ENABLE_CUDA_HOST_ADAPTER)
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_CUDA_HOST_ADAPTER 1\n");
#else
  std::printf("TRACE_HOST macro_CUTLASS_ENABLE_CUDA_HOST_ADAPTER 0\n");
#endif
#if defined(CUTLASS_DEBUG_TRACE_LEVEL)
  std::printf("TRACE_HOST macro_CUTLASS_DEBUG_TRACE_LEVEL %d\n", int(CUTLASS_DEBUG_TRACE_LEVEL));
#else
  std::printf("TRACE_HOST macro_CUTLASS_DEBUG_TRACE_LEVEL undefined\n");
#endif
#if defined(__CUDACC_VER_MAJOR__)
  std::printf("TRACE_HOST nvcc_version %d %d %d\n", int(__CUDACC_VER_MAJOR__), int(__CUDACC_VER_MINOR__), int(__CUDACC_VER_BUILD__));
#endif
  std::printf("TRACE_HOST host_probes_done 1\n");
}

// H3: launch-related facts.  Call after the warm-up gemm.run() and verify(), when both function attributes
// (MaxDynamicSharedMemorySize from initialize(), NonPortableClusterSizeAllowed from run()) have been set.
template <class Gemm>
void trace_host_post_run() {
  using GemmKernel = typename Gemm::GemmKernel;
  void const* fn = reinterpret_cast<void const*>(&cutlass::device_kernel<GemmKernel>);
  cudaError_t last = cudaGetLastError();
  std::printf("TRACE_HOST cudaGetLastError %d %s\n", int(last), cudaGetErrorString(last));
  cudaFuncAttributes a;
  trace_check(cudaFuncGetAttributes(&a, fn), "cudaFuncGetAttributes");
  std::printf("TRACE_HOST kernel_maxDynamicSharedSizeBytes %d\n", a.maxDynamicSharedSizeBytes);
  std::printf("TRACE_HOST kernel_sharedSizeBytes %zu\n", a.sharedSizeBytes);
  std::printf("TRACE_HOST kernel_constSizeBytes %zu\n", a.constSizeBytes);
  std::printf("TRACE_HOST kernel_localSizeBytes %zu\n", a.localSizeBytes);
  std::printf("TRACE_HOST kernel_numRegs %d\n", a.numRegs);
  std::printf("TRACE_HOST kernel_maxThreadsPerBlock %d\n", a.maxThreadsPerBlock);
  std::printf("TRACE_HOST kernel_ptxVersion %d\n", a.ptxVersion);
  std::printf("TRACE_HOST kernel_binaryVersion %d\n", a.binaryVersion);
  std::printf("TRACE_HOST kernel_nonPortableClusterSizeAllowed %d\n", a.nonPortableClusterSizeAllowed);
  std::printf("TRACE_HOST kernel_clusterDimMustBeSet %d\n", a.clusterDimMustBeSet);
  std::printf("TRACE_HOST kernel_requiredClusterDims %d %d %d\n", a.requiredClusterWidth, a.requiredClusterHeight, a.requiredClusterDepth);
  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(64, 64, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = 230400;
    cfg.stream = nullptr;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 2;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int max_active_clusters = -1;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&max_active_clusters, fn, &cfg);
    std::printf("TRACE_HOST max_active_clusters %d %s\n", max_active_clusters, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
  }
  {
    int dev = 0;
    trace_check(cudaGetDevice(&dev), "cudaGetDevice");
    int sms = 0, reserved = 0, optin = 0, cluster = 0, major = 0, minor = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&cluster, cudaDevAttrClusterLaunch, dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceProp prop;
    trace_check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    std::printf("TRACE_HOST device_name %s\n", prop.name);
    std::printf("TRACE_HOST device_sm_count %d\n", sms);
    std::printf("TRACE_HOST device_reserved_smem_per_block %d\n", reserved);
    std::printf("TRACE_HOST device_max_smem_per_block_optin %d\n", optin);
    std::printf("TRACE_HOST device_cluster_launch %d\n", cluster);
    std::printf("TRACE_HOST device_compute_capability %d %d\n", major, minor);
  }
  std::printf("TRACE_HOST post_run_done 1\n");
}
