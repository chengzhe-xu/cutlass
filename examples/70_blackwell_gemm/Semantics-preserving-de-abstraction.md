# Semantics-preserving de-abstraction of `70_blackwell_fp16_gemm.cu`

Task target, equivalence contract, and verified baseline trace.

- Recorded 2026-09-08 and revised the same day for the fixed use case of Section 0: scope narrowed, material duplicated between Part A and Part B merged, and the corrections of three independent source reviews applied (B.8). No replacement kernel has been written yet.
- Baseline source: `examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu`
- Repository state: CUTLASS v4.7.1, commit `cb4247394dd82148787aed73e5dc7cef33cbf862`. All file:line anchors below refer to this commit.
- Inspection host: macOS/arm64 without `nvcc`, without a CUDA driver, and without a build directory. Everything below is derived from source; facts that need a compiler or a B200 are labelled and listed in Section 11.
- `de-abstraction-plan.log` (the earlier draft in this directory) was used only as a checklist. It was treated as untrusted. Section 13 lists where it was wrong or imprecise.
- How this document is organized: **Part A** (Sections 1-14) is the task target, the equivalence contract and the verified baseline facts; **Part B** is the call-stack and lowering trace of `gemm.run()` down to CuTe formulas and PTX, with the statically undeterminable items collected in B.7; **Part C** is the ready-to-execute plan for the debug-print code and the experiments that settle those items on the B200 under the fixed build and run commands of Section 0 (start at C.5 for the run order); **Part D** (to be added after the runs) will hold the measured values.

Evidence labels used throughout:

| Label | Meaning |
|---|---|
| `[S]` | Fixed by the instantiated C++/CuTe source at this commit (compile-time selection or device control flow). |
| `[E]` | Exact arithmetic consequence for the default run `--m=8192 --n=8192 --k=8192`, alpha=1, beta=0, swizzle=0. |
| `[C]` | Depends on the compiler/toolchain/build flags (CUDA version, `-fmad`, ptxas scheduling, SASS lowering, CMake cache). |
| `[R]` | Depends on the runtime, driver, or hardware (addresses, CLC cancellation order, timings, driver acceptance). |

---

## 0. Fixed build and run configuration (user decision, 2026-09-08)

The user builds and runs the example in exactly one way. Everything else the example can do (other shapes, `--alpha`/`--beta`, `--swizzle`, `--iterations`) is out of scope, and the replacement kernel may hard-code the values below.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDAToolkit_ROOT="$CUDA_HOME" \
      -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_ENABLE_EXAMPLES=ON -DCUTLASS_ENABLE_TESTS=ON -DCUTLASS_ENABLE_PROFILER=ON \
  && cmake --build build --target 70_blackwell_fp16_gemm --parallel 16
./build/examples/70_blackwell_gemm/70_blackwell_fp16_gemm --m=8192 --n=8192 --k=8192
```

| Quantity | Fixed value | Source |
|---|---|---|
| `M, N, K, L` | 8192, 8192, 8192, 1 | command line; `Options` defaults are the same (`70_blackwell_fp16_gemm.cu:200-204`) |
| `alpha`, `beta` | `1.0f`, `0.0f` | `Options` defaults (`:202`, `:219-220`) |
| `iterations` | 10 | `Options` default (`:203`); one warm-up launch + 10 timed launches, so 11 `initialize()`/`run()` pairs per process (`:397-419`); each pair performs 6 tensor-map encodes, 2 `cudaFuncSetAttribute` calls (one in `initialize()`, one in `run()`) and 1 `cudaLaunchKernelExC` |
| `swizzle` (`max_swizzle_size`) | 0 -> swizzle disabled, raster order `AlongN` by heuristic | `:204`, `:338`; Section 7.4 |
| Data | A, B, C filled by `BlockFillRandomUniform` with seeds 2023, 2022, 2021 (global `seed = 0`, `:175`, `:323-325`), integer values in `[-8, 8]` (`:299-301`), `bits = 0` | Section 8 |
| Correctness check | `verify()` runs the device reference GEMM with the same `alpha`/`beta`/`C` and `BlockCompareEqual` on all `M*N` outputs (`:343-373`); the harness prints `Disposition: Passed/Failed` and exits on failure | Section 3 |
| Performance number | `Avg runtime` and `GFLOPS` printed by the same binary from `GpuTimer` around the 10 `initialize()+run()` pairs (`:413-432`) | Section 12 |
| Build flags | `Release` (`NDEBUG` defined, `-O3`), `sm_100a` only, `CUTLASS_ENABLE_GDC_FOR_SM100=ON` (CMake default), no `--default-stream per-thread`, no direct driver calls, no host adapter, trace level 0; `CUTLASS_ENABLE_TESTS/PROFILER=ON` only add other targets and do not change this target's flags | Section 11, D14 |
| Toolchain and machine | CUDA 13.3 (`nvcc` 13.3, driver newer than 13010) on a B200 (compute capability 10.0, 148 SMs), as stated by the user; confirmed by H5/H3/D14. Facts that follow from it (the 32-bit low-word descriptor increment, the driver-version fixup not triggering) are treated as fixed | Section 11 |

Consequences for the rest of this document:

- The beta != 0 machinery (warp 3 as C producer, `load_order` barrier arrivals, `epi_load` pipeline traffic, CLC empty count 928, `store_tail` work; B.3.5) is documented for completeness but is **not required** in the replacement. The replacement may hard-code `beta = 0` and omit the C path, but must keep the same arithmetic form for `D` (`alpha * acc` then `beta * 0 + (.)`, Section 8) so that every output bit, including the sign of zero, matches.
- Swizzle and `AlongM` rasterization, residue handling for non-multiple shapes, and `L > 1` (Section 10) are not required. The kernel may assume 32 x 64 full CTA tiles, 128 k-tiles per tile, and the `AlongN` transpose of Section 7.4.
- The trace experiments of Part C must work through this exact build command and this exact run command: no second CMake target, no extra command-line flags, no extra compile definitions. Trace mode is a source-level toggle (Part C, C.1).
- Section 11 items about the CMake cache are now fixed by the command above and only need confirmation (D14).

## 1. What the task is

"Semantics-preserving code de-abstraction and inlining" means: take the GEMM that the example builds through CUTLASS's high-level template machinery (`CollectiveBuilder` for the mainloop and the epilogue, `GemmUniversal`, `GemmUniversalAdapter`, the CuTe `TiledMma`/`TiledCopy` atoms, the pipeline classes, and the tile scheduler), trace what those layers resolve to at compile time and what actually executes at run time, and then re-implement that exact execution as one explicit CUDA kernel whose code shows every mechanism directly.

Required properties of the replacement:

1. **Same computation, same output.** For the Section 0 run the replacement must produce the same D bitwise (`Disposition: Passed`). Section 8 shows that the harness data make the result independent of accumulation order, so a passing compare does not by itself prove that the instruction sequence is the same; equivalence is established structurally, by matching the protocol of Section 7, the trace of Part B and the measured values of Part D.
2. **Same runtime performance.** Measured under identical conditions (Section 12). The replacement must therefore keep the baseline's performance mechanisms: two-SM `tcgen05` MMA with a 2x2x1 cluster, TMA loads with A multicast, an 8-stage A/B smem pipeline, a 4-stage TMEM accumulator pipeline, warp specialization, the persistent Cluster-Launch-Control (CLC) scheduler, and the TMA-store epilogue with 4 smem stages.
3. **No hidden layers.** The replacement must not call, wrap, or re-instantiate the `CollectiveBuilder` outputs, `CollectiveMma`, `CollectiveEpilogue`, `GemmUniversal`, `GemmUniversalAdapter`, `PersistentTileSchedulerSm100`, the `Pipeline*` classes, `TiledMma`, `TiledCopy`/`make_tma_copy`, `FusionCallbacks`, or comparable wrappers that hide the implementation. Scheduling, roles, barriers, TMA issue, `tcgen05` issue, TMEM addressing, the epilogue math, and the tail logic must all be visible in the kernel source.
4. **Allowed low-level tools.** CuTe may be used for layouts, swizzles, and coordinate arithmetic (for example `Swizzle<3,4,3>`, `Layout`, `composition`, index math). Inline PTX may be used for `tcgen05.*`, `cp.async.bulk.tensor.*` (TMA), `mbarrier.*`, `fence.*`, `clusterlaunchcontrol.*`, `griddepcontrol.*`, `mapa`, `elect.sync`, `bar.sync`, `barrier.cluster.*`, and `prefetch.tensormap`. The host may keep using the CUDA driver's `cuTensorMapEncodeTiled` to build tensor maps.
5. **Which instruction is used follows the trace.** The baseline moves A and B with `cp.async.bulk.tensor` (TMA), not with per-thread `cp.async.{ca,cg}`. The replacement must do the same. Permission to use "cp.async" in the task statement does not license replacing TMA with per-thread copies.
6. **Not a different GEMM.** A CUDA-core GEMM, a one-thread-per-output kernel, naive smem tiling, a cuBLAS call, or a different CUTLASS kernel is not acceptable even if numerically correct.
7. **Scope of the host harness.** The CLI parsing, allocation, random initialization, reference GEMM, exact comparison, and timing loop in the example are outside the de-abstraction scope and may stay as they are unless a later step says otherwise. Host code that builds the six tensor maps and launches the kernel is part of the replacement (it must not go through `GemmUniversalAdapter`).

Later steps from the user govern the implementation and validation details. This file records the target so those steps can be executed without re-deriving the baseline.

---

## 2. The baseline instantiation `[S]`

From `70_blackwell_fp16_gemm.cu:93-148`:

| Item | Value |
|---|---|
| ElementA / LayoutA / AlignmentA | `half_t`, `RowMajor`, 8 elements (16 B) |
| ElementB / LayoutB / AlignmentB | `half_t`, `ColumnMajor`, 8 elements (16 B) |
| ElementC = ElementD / LayoutC / AlignmentC | `float`, `ColumnMajor`, 4 elements (16 B) |
| ElementAccumulator | `float` |
| ArchTag / OperatorClass | `arch::Sm100`, `OpClassTensorOp` |
| MmaTileShape_MNK | `(256, 128, 64)` |
| ClusterShape_MNK | `(2, 2, 1)`, static |
| Epilogue builder | `EpilogueTileAuto`, `EpilogueScheduleAuto`, default fusion op (`LinearCombination`) |
| Mainloop builder | `StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>`, `KernelScheduleAuto` |
| Kernel | `GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>` |
| Device adapter | `GemmUniversalAdapter<GemmKernel>` |
| Problem shape | `(M, N, K, L)` runtime ints with `L = 1` |
| Mode | `GemmUniversalMode::kGemm` |
| Scalars | `alpha`, `beta` runtime `float` (defaults 1.0f, 0.0f), `alpha_ptr = beta_ptr = nullptr` |
| Scheduler argument | `scheduler.max_swizzle_size = --swizzle` (default 0); `raster_order = Heuristic` |
| CLI defaults | `m = n = k = 8192`, `iterations = 10` |

Build conditions `[S]`:

- The example target is only added when `CUTLASS_NVCC_ARCHS` matches `100a|100f|101a|101f|103a|103f` (`examples/70_blackwell_gemm/CMakeLists.txt:36`).
- `CUTLASS_ARCH_MMA_SM100_SUPPORTED` requires CUDA 12.8+ (`include/cutlass/arch/config.h:87-88`); the `tcgen05` wrappers additionally need an `sm_100a`/family build (`CUTLASS_ARCH_MMA_SM100A_ENABLED` from `__CUDA_ARCH_FEAT_SM100_ALL`, `config.h:89-93`; `CUTLASS_ARCH_TCGEN_ENABLED` in `include/cutlass/arch/barrier.h:50-58`).
- `main()` requires `__CUDACC_VER_MAJOR__.MINOR >= 12.8` and a device with `major == 10 && minor == 0` (`70_blackwell_fp16_gemm.cu:446-461`). It queries the current device and then overwrites `props` with device 0's properties (lines 454-456), so the gate actually inspects device 0.
- Root `CMakeLists.txt:463-475` defaults `CUTLASS_ENABLE_GDC_FOR_SM100=ON`, adding `-DCUTLASS_ENABLE_GDC_FOR_SM100=1`.
- `CUTLASS_ENABLE_CUDA_HOST_ADAPTER` is not defined by default, so `GemmUniversalAdapter::kEnableCudaHostAdapter == false`.

---

## 3. Functional contract the replacement must preserve `[S]`

Scope note (2026-09-08): Section 0 fixes the configuration, so only the parts of this contract that the fixed run exercises are required (alpha = 1, beta = 0, M = N = K = 8192, L = 1, swizzle 0). Items about other arguments are kept as reference.

Math: `D(m,n) = alpha * sum_k A(m,k) * B(k,n) + beta * C(m,n)` for `0 <= m < M`, `0 <= n < N`, K summed over `0 <= k < K`, single batch.

Memory representation (all packed, from `make_cute_packed_stride`, `tools/util/include/cutlass/util/packed_stride.hpp:75-109`):

| Tensor | CuTe modes | Stride type | Packed stride | Byte address of element |
|---|---|---|---|---|
| A | `(M, K, L)` | `Stride<int64_t, _1, int64_t>` | `(K, 1, 0)` | `2*(m*K + k)` |
| B | `(N, K, L)` | `Stride<int64_t, _1, int64_t>` | `(K, 1, 0)` | `2*(n*K + k)` |
| C | `(M, N, L)` | `Stride<_1, int64_t, int64_t>` | `(1, M, 0)` | `4*(m + n*M)` |
| D | `(M, N, L)` | `Stride<_1, int64_t, int64_t>` | `(1, M, 0)` | `4*(m + n*M)` |

The batch stride is a runtime `int64_t` equal to 0 because `batch_count == 1` (`packed_stride.hpp:84-87`). It is not a static `Int<0>`; this is why every tensor map is rank 3 (Section 5.3).

Host checks: `gemm.can_implement` (`70_blackwell_fp16_gemm.cu:394`; kernel `:300-336`) still executes. Because the collectives test default-constructed strides, it reduces to `K % 8 == 0` (`sm100_mma_warpspecialized.hpp:429-437`, `detail/layout.hpp:401-418`) and `M % 4 == 0` (`sm100_epilogue_tma_warpspecialized.hpp:334-375`); both hold for 8192. `make_tma_copy_desc` asserts 16-byte base alignment only in debug builds (`copy_traits_sm90_tma.hpp:955`); `cudaMalloc` satisfies it.

Semantics that must survive de-abstraction:

- The beta = 0 path: no C reads and an idle warp 3, but the linear-combination multiply-add with a zeroed C fragment still executes (Section 8). The beta != 0 path is reference only (B.3.5).
- `AlongN` rasterization with swizzle disabled and the CLC persistent schedule (Section 7.4).
- The exact `tcgen05` accumulate pattern (first MMA of each output tile clears, all later MMAs accumulate) and the epilogue arithmetic (Section 8).
- Workspace size 0; six tensor-map encodes per `initialize()`; launch geometry and attributes (Section 5).

Harness facts that affect reproduction `[S]`:

- `uint64_t seed;` is a zero-initialized global; A, B, C are filled with seeds 2023, 2022, 2021 (`70_blackwell_fp16_gemm.cu:175, 323-325`).
- `initialize_block` calls `BlockFillRandomUniform(ptr, size, seed, max=8, min=-8, bits=0)` for 16- and 32-bit elements (`lines 285-307`). In `tensor_fill.h:521-545`, `bits = 0` sets `int_scale = 0`, so each value is `llround(uniform_in[-8,8])` — **every A, B, and C element is an integer in [-8, 8]**. Consequence: all FP16 products are exact integers, |acc| <= 64*K, and for K <= 262144 the FP32 sum is exact in any order. The harness's exact compare therefore passes independently of accumulation order; see Section 8 for why a stronger test is needed. The fill kernel assigns element `i` to thread `i mod T` with `T` chosen by `cudaOccupancyMaxPotentialBlockSize` (block capped at 128) and seeds each thread with `curand_init(seed, gtid, 0)`, so the concrete random values depend on the device's occupancy answer `[R]`; the integer-valued property does not, and the harness is kept unchanged, so no experiment is needed.
- `verify()` runs `cutlass::reference::device::Gemm` (accumulating in FP32 with a sequential k loop, epilogue `alpha * acc + beta * C`, `reference/device/thread/gemm.h:170-171`) and then `BlockCompareEqual`, an exact `!=` element compare (`reference/device/tensor_compare.h:56-75`). NaN would fail; -0.0 vs +0.0 passes.
- Timing: `GpuTimer` events on the default stream around a loop of `gemm.initialize(arguments, workspace)` + `gemm.run()` per iteration (`lines 413-427`). Each timed iteration therefore includes six host `cuTensorMapEncodeTiled` calls, two `cudaFuncSetAttribute` calls (`MaxDynamicSharedMemorySize` in `initialize()`, `NonPortableClusterSizeAllowed` in `run()`) and the launch. GFLOP/s = `2*M*N*K / time`.
- One warm-up/correctness launch precedes verification; failure calls `exit(-1)` before profiling.
- Every GPU operation in the harness (fill kernels, the GEMM launch, the reference GEMM, the compare kernel) runs on the legacy default stream and relies on its serialization; the explicit host waits are `cudaDeviceSynchronize()` in `verify()`, `cudaStreamSynchronize` inside the compare, and `cudaEventSynchronize` in the timer.
---

## 4. Compile-time resolution of every `Auto` `[S]`

| Requested | Resolved value | Where |
|---|---|---|
| `KernelScheduleAuto` (mainloop) with static cluster, ClusterM % 2 == 0, TileM % 128 == 0 | two-SM path: `is_2sm = true`; `TiledMma = make_tiled_mma(SM100_MMA_F16BF16_2x1SM_SS<half_t, half_t, float, 256, 128, UMMA::Major::K, UMMA::Major::K>)` | `builders/sm100_umma_builder.inl:208-222`; `builders/sm100_common.inl:456-465, 386-390` |
| UMMA majors | A = K-major (RowMajor A), B = K-major (ColumnMajor B) | `sm100_umma_builder.inl:201-202` |
| `AtomThrShapeMNK` / `CtaTileShape_MNK` | `(2,1,1)` / `(128, 128, 64)` | `sm100_umma_builder.inl:227-230` |
| GmemTiledCopyA | `SM100_TMA_2SM_LOAD_MULTICAST` (cluster N = 2 != 1) | `sm100_common.inl:203-211` |
| GmemTiledCopyB | `SM100_TMA_2SM_LOAD` (cluster M == 2) | `sm100_common.inl:239-247` |
| SmemLayoutAtomA / B | `UMMA::Layout_K_SW128_Atom<half_t>` for both (BLK_K = 64 is a multiple of 64) | `sm100_common.inl:115-118, 256-259` |
| AccumulatorPipelineStageCount | `min(4, 128*512 / (128*128)) = 4` | `sm100_umma_builder.inl:260-267` |
| SchedulerPipelineStageCount | 2 | `sm100_umma_builder.inl:282` |
| KernelSmemCarveout | 192 bytes (Section 6.1) | `builders/sm100_pipeline_carveout.inl:45-83` |
| PipelineStages (`StageCountAutoCarveout`) | `floor((232448 - 192 - 33792) / 24592) = 8` | `sm100_umma_builder.inl:84-99, 292-299`; `arch/arch.h:44` |
| Mainloop DispatchPolicy | `MainloopSm100TmaUmmaWarpSpecialized<8, 2, 4, Shape<_2,_2,_1>, Sm100>` | `sm100_umma_builder.inl:320-326` |
| `EpilogueScheduleAuto` | `TmaWarpSpecialized2Sm` (static even ClusterM and MmaTileM == 256) | `epilogue/collective/builders/sm100_builder.inl:1698-1715` |
| Epilogue CtaTileShape / TmemWarpShape | `(128,128,64)` / `Shape<_4,_1>` | `sm100_builder.inl:1198-1210, 1073-1081` |
| `EpilogueTileAuto` | `(128, 16)`: M = min(128, 32*4) = 128; N_perf = 16 (MaxBits 32, CtaM > 64, CtaN <= 128); N_min_C = N_min_D = 8 (M-major); N = 16. The type is a `cute::Tile` of two rank-1 layouts, `tuple<Layout<_128,_1>, Layout<_16,_1>>`, whose shape is (128,16) | `sm100_builder.inl:999-1055` |
| EpiTiles / FragmentSize | 8 subtiles / 16 floats per thread per subtile | `sm100_builder.inl:1253-1255` |
| Epilogue DispatchPolicy | `Sm100TmaWarpSpecialized<StagesC=4, StagesD=2, FragmentSize=16, ReuseSmemC=true, DelayTmaStore=false>` | `sm100_builder.inl:1107-1148` |
| CopyOpT2R (TMEM to registers) | `SM100_TMEM_LOAD_32dp32b16x` (`op_repeater<SM100_TMEM_LOAD_32dp32b1x, 16*32 bits>`) | `sm100_builder.inl:593-631`; `cute/atom/copy_traits_sm100.hpp:2781+` |
| SmemLayoutAtomC = SmemLayoutAtomD | `UMMA::Layout_MN_SW128_32B_Atom<float>` = `Swizzle<2,5,2> o smem_ptr[32b] o (32,4):(1,32)` (128 B contiguous along M, 4 columns) | `sm100_builder.inl:72-97`; `sm100_common.inl:89-94`; `cute/atom/mma_traits_sm100.hpp:73-76` |
| CopyOpG2S / CopyOpS2G | `SM90_TMA_LOAD` / `SM90_TMA_STORE` | `sm100_builder.inl:1298-1316` |
| CopyOpS2R / CopyOpR2S / CopyOpR2R | `AutoVectorizingCopyWithAssumedAlignment<128>` for all three (no `stmatrix` variant matches a 32dp op with 32-bit M-major output) | `sm100_builder.inl:717-840, 1318-1331` |
| FusionCallbacks | `FusionCallbacks<Sm100TmaWarpSpecialized<4,2,16,true,false>, LinearCombination<float,float,float,float,round_to_nearest>, ...>` deriving from the SM90 `Sm90LinearCombination` tree (Section 8) | `fusion/sm100_callbacks_tma_warpspecialized.hpp:69-82`; `fusion/sm90_callbacks_tma_warpspecialized.hpp:182-241` |
| TileSchedulerTag `void` | `PersistentTileSchedulerSm100<Shape<_2,_2,_1>, 2>`; `IsDynamicPersistent = true` | `kernel/tile_scheduler.hpp:207-219`; `sm100_tile_scheduler.hpp:59-79` |
| `IsOverlappingAccum` | false | `MainloopSm100TmaUmmaWarpSpecialized` (dispatch_policy.hpp:1029+) |
| Threads per CTA | 256 = 5 roles: Sched 32 + MMA 32 + MainloopLoad 32 + EpilogueLoad 32 + Epilogue 128 | `sm100_gemm_tma_warpspecialized.hpp:136-147` |
| `MinBlocksPerMultiprocessor` | 1 (`__launch_bounds__(256, 1)`) | `sm100_gemm_tma_warpspecialized.hpp:147`; `device_kernel.h:114-127` |
| TmemAllocator | `TMEM::Allocator2Sm`, 512 columns | `sm100_gemm_tma_warpspecialized.hpp:178-179, 728` |

The UMMA atom traits (`cute/atom/mma_traits_sm100.hpp:1160-1218`), with M=256, N=128, K=16:

```
ThrID   = Layout<_2>
ALayout = ((2),(128,16)) : ((128),(1,256))     // V x (M/2, K)
BLayout = ((2),( 64,16)) : (( 64),(1,128))     // V x (N/2, K)
CLayout = ((2),(128,128)): ((128),(1,256))     // V x (M/2, N)
FrgTypeA/B = UMMA::smem_desc<Major::K>   (descriptor iterators)
FrgTypeC   = UMMA::tmem_frg_2sm<float>
```

So each CTA of the pair holds a 128x64 A slice and a 64x64 B slice per K stage, and owns a 128x128 quarter of the pair's 256x128 accumulator.

---

## 5. Host path `[S]` `[E]`

### 5.1 Call chain

```
run<Gemm>(options)
  initialize(options)                 // fills A,B,C; allocates D, ref_D
  args_from_options(options)          // Arguments{kGemm, {m,n,k,1}, {A,dA,B,dB}, {{alpha,beta},C,dC,D,dD}}; scheduler.max_swizzle_size
  Gemm::get_workspace_size(args)      // 0
  gemm.can_implement(args)            // Section 3
  gemm.initialize(args, workspace)    // Section 5.2
  gemm.run()                          // Section 5.4
```

`get_workspace_size` is 0 because the fusion callbacks need none (`sm90_visitor_load_tma_warpspecialized.hpp` scalar/src ops return 0) and the CLC scheduler's `Params::get_workspace_size` is 0 for a non-stream-K scheduler.

### 5.2 `initialize()` (`gemm_universal_adapter.h:311-355`)

1. `GemmKernel::initialize_workspace` — no-op success.
2. `params_ = GemmKernel::to_underlying_arguments(args, workspace)` (`sm100_gemm_tma_warpspecialized.hpp:257-298`): builds
   - mainloop Params: four TMA atoms in this order: `tma_load_a`, `tma_load_b`, `tma_load_a_fallback`, `tma_load_b_fallback` (`sm100_mma_warpspecialized.hpp:377-407`). For the static cluster the fallback descriptors are byte-identical to the primaries and are never selected or prefetched on the device (`:341-350`); the host still encodes them, and that cost is inside the timed interval.
   - epilogue Params: `tma_load_c` then `tma_store_d` (`sm100_epilogue_tma_warpspecialized.hpp:300-319`). `tma_load_c` is built even when beta == 0.
   - scheduler Params (Section 7.4).
   Total: **six `cuTensorMapEncodeTiled` calls per `initialize()`**, order A, B, A-fallback, B-fallback, C, D.
3. `cudaFuncSetAttribute(device_kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, 230400)` because `SharedStorageSize >= 48 KiB` — executed on every `initialize()`, hence in every timed iteration.

### 5.3 Tensor maps

All six maps are rank 3. `construct_tma_gbasis` (`cute/atom/copy_traits_sm90_tma.hpp:813-838`) drops a gmem mode only when its shape type is statically `Int<1>` or its stride type is statically `Int<0>`; L = 1 and stride 0 are runtime values here, so the batch mode is kept with extent 1 and stride 0. Device-side TMA coordinates are therefore 3-D.

Encoder inputs (`make_tma_copy_desc`, `copy_traits_sm90_tma.hpp:905-1075`), with modes ordered by increasing stride and mode-0 stride implicit:

| Map | dataType | TMA mode order | globalDim | globalStrides (bytes, modes 1..2) | boxDim | elementStrides | swizzle |
|---|---|---|---|---|---|---|---|
| A, A-fallback | `FLOAT16` | (K, M, L) | (K, M, 1) | (2K, 0) | (64, 64, 1) | (1,1,1) | `CU_TENSOR_MAP_SWIZZLE_128B` |
| B, B-fallback | `FLOAT16` | (K, N, L) | (K, N, 1) | (2K, 0) | (64, 64, 1) | (1,1,1) | `CU_TENSOR_MAP_SWIZZLE_128B` |
| C | `FLOAT32` | (M, N, L) | (M, N, 1) | (4M, 0) | (32, 16, 1) | (1,1,1) | `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` |
| D | `FLOAT32` | (M, N, L) | (M, N, 1) | (4M, 0) | (32, 16, 1) | (1,1,1) | `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` |

Common settings: `CU_TENSOR_MAP_INTERLEAVE_NONE`, `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` (zero fill). For 8192-cubed: A/B strides (16384, 0), C/D strides (32768, 0) `[E]`.

Details that matter for re-implementation:

- The C/D box is (32 M, 16 N, 1) = 2048 bytes, not the whole (128,16) subtile. `construct_tma_gbasis` (`copy_traits_sm90_tma.hpp:757-783`) inverts the non-swizzled smem stage layout `((32,4),(4,4)):((1,512),(32,128))` and truncates at the first mode whose basis stride is not 1: after 32 contiguous M elements and 16 N columns (stride 32 elements), the next M block has stride 512, so the box stops at (32,16). This also matches the TMA rule that the innermost box dimension of a 128-byte-swizzled map is at most 128 bytes. One `copy(tma_store_d, ...)` or `copy(tma_load_c, ...)` per epilogue subtile therefore expands to **four TMA instructions** (i = 0..3) with gmem coordinates `(128*tm + 32*i, 128*tn + 16*e, 0)` and smem addresses `stage_base + 2048*i`. The C-load barrier still expects 8192 bytes per stage (`TmaTransactionBytes = StageCBits/8`).
- A's smem box per CTA is logically (64 K, 128 M) but `make_tma_atom_A_sm100` passes `num_multicast = size<2>(cluster_vmnk) = 2` (`copy_traits_sm100_tma.hpp:767-782`) and `make_tma_copy_desc` truncates the box starting from the last mode: the L box (1) absorbs nothing, then the M box is divided by 2, giving (64, 64, 1) (`copy_traits_sm90_tma.hpp:996-1002`). Each of the two CTAs sharing an A tile loads a different 64-row half and multicasts it to both (Section 7.5).
- The zero L stride is passed to the driver as a literal 0 byte stride; the source asserts only `stride % 16 == 0` and `< 2^40` (`copy_traits_sm90_tma.hpp:938-947`). Driver acceptance is `[R]` but is what the baseline does.
- Swizzle mapping: `Swizzle<3,4,3>` -> 128B; `Swizzle<2,5,2>` -> B128 bits with 32-byte base -> `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`, only available for CUDA > 12.6 (`cute/arch/copy_sm90_desc.hpp:240-262`).
- The driver-version fixup in `make_tma_copy_desc` (`copy_traits_sm90_tma.hpp:1061-1070`: clear bit 21 of the descriptor's second word when `cudaDriverGetVersion <= 13010` and the tensor is smaller than 131072 bytes) cannot trigger for the 128 MiB / 256 MiB tensors of this run, whatever the driver; the replacement need not implement it.
- The 128-byte `CUtensorMap` objects live by value inside `Params`, which is passed `__grid_constant__`; device code prefetches and issues TMA with the generic address of the parameter-space descriptor. In the baseline each `Params` TMA member is a `Copy_Atom<Copy_Traits<Op, C<bits>, AuxTmaParams>>` that embeds the descriptor plus CuTe basis metadata, so the parameter block is not six back-to-back maps. An explicit kernel is free to define its own parameter struct with six plain `CUtensorMap`s (64-byte aligned), as long as the encoder inputs above are identical. Only the four maps actually used (A, B, C, D) are prefetched; the fallback maps are never touched on the device.

### 5.4 `run()` and launch geometry (`gemm_universal_adapter.h:372-575`)

- `block = (256,1,1)`, `smem = 230400`, `grid = GemmKernel::get_grid_shape(params)` (Section 7.4). For 8192-cubed the grid is `(64, 64, 1)` `[E]` (4096 CTAs, 1024 clusters).
- `ArchTag::kMinComputeCapability == 100`, static cluster, not 1x1x1, host adapter disabled, so the path is `ClusterLauncher::launch_with_fallback_cluster(grid, cluster=(2,2,1), fallback=(0,0,0), block, smem, stream=nullptr, device_kernel<GemmKernel>, {&params}, launch_with_pdl=false)` (`gemm_universal_adapter.h:486-521`).
- `make_cluster_launch_config` (`cluster_launch.hpp:151-207`): exactly one active attribute, `cudaLaunchAttributeClusterDimension = {2,2,1}`. The preferred-cluster attribute is dropped because there is no fallback, and the PDL attribute is dropped because `launch_with_pdl == false`.
- `check_cluster_dims` (grid divisible by cluster, cluster size <= 32), then `init(kernel)` = `cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` on every launch, then `cudaLaunchKernelExC`. `run()` returns `kErrorInternal` if `cudaGetLastError()` reports an error; the launch itself is asynchronous.
- Kernel entry (`device_kernel.h:114-127`): `CUTLASS_GLOBAL` (= `__global__ static`) `__launch_bounds__(256, 1) void device_kernel(__grid_constant__ Params const params)` with `extern __shared__ char smem[]` (no alignment attribute) and `GemmKernel op; op(params, smem)`. The `__grid_constant__` qualifier applies in the device pass (`CUTLASS_GRID_CONSTANT_ENABLED`).

---

## 6. Shared memory and TMEM budget `[S]`

### 6.1 Carveout arithmetic used by the builder

- Stage bytes = `128*64*2 (A) + 64*64*2 (B) + sizeof(PipelineTmaUmmaAsync<1>::SharedStorage) = 16` = 24592.
- `KernelSmemCarveout` = accumulator pipeline 64 + CLC pipeline 32 + load-order barrier 16 + TMEM-dealloc barrier 8 + CLC throttle pipeline 32 + CLC responses 2*16 + TMEM base pointers 2*4 = **192** bytes (the builder's estimate; the actual struct has one 4-byte `tmem_base_ptr`, Section 6.2).
- `sizeof(CollectiveEpilogue::SharedStorage)` = TensorStorage 33280 + LoadPipeline storage 64 = 33344, rounded to the 512-byte alignment = **33792**.
- Stages = `floor((232448 - 192 - 33792) / 24592) = floor(198464 / 24592) = 8`.

### 6.2 Actual kernel `SharedStorage` layout (`sm100_gemm_tma_warpspecialized.hpp:181-212`)

| Offset | Bytes | Member | Notes |
|---|---|---|---|
| 0 | 64 | `pipelines.mainloop.full_barrier_[8]` | `ClusterTransactionBarrier`, 8 B each |
| 64 | 64 | `pipelines.mainloop.empty_barrier_[8]` | `ClusterBarrier` |
| 128 | 32 | `pipelines.epi_load.full_barrier_[4]` | |
| 160 | 32 | `pipelines.epi_load.empty_barrier_[4]` | |
| 192 | 16 | `pipelines.load_order.barrier_[1][2]` | `OrderedSequenceBarrier<1,2>` |
| 208 | 16 | `pipelines.clc.full_barrier_[2]` | transaction barriers |
| 224 | 16 | `pipelines.clc.empty_barrier_[2]` | |
| 240 | 32 | `pipelines.accumulator.full_barrier_[4]` | |
| 272 | 32 | `pipelines.accumulator.empty_barrier_[4]` | |
| 304 | 16 | `pipelines.clc_throttle.full_barrier_[2]` | |
| 320 | 16 | `pipelines.clc_throttle.empty_barrier_[2]` | |
| 336 | 8 | `pipelines.tmem_dealloc` | `ClusterBarrier` |
| 344 | 8 | padding | `PipelineStorage` is `aligned_struct<16>` |
| 352 | 32 | `clc_response[2]` | 16-byte responses, `alignas(16)` |
| 384 | 4 | `tmem_base_ptr` | |
| 388 | 124 | padding | `tensors` needs 512-byte alignment |
| 512 | 32768 | `tensors.epilogue.collective` (union `smem_C` / `smem_D`) | 4 stages x 8192 B, `alignas(512)` from `Swizzle<2,5,2>` |
| 33280 | 1 (+511 pad) | `tensors.epilogue.thread` (FusionStorage, empty tuple) | pads to 512 |
| 33792 | 131072 | `tensors.mainloop.smem_A` | 8 x 16384 B; offset is 1024-aligned |
| 164864 | 65536 | `tensors.mainloop.smem_B` | 8 x 8192 B; offset is 1024-aligned |
| 230400 | | end | `SharedStorageSize = 230400` = 225 KiB, 2048 B below 232448 |

The mainloop `TensorStorage` only declares 128-byte alignment; the 1024-byte alignment that `Swizzle<3,4,3>` needs falls out of the member sizes above (C++ pads `sizeof` of the epilogue storage to its own 512-byte alignment, so the mainloop storage starts at 33280 within `tensors`, i.e. at 33792 absolute) and of the dynamic-smem base being 1024-aligned `[R]` (both candidate bases, 0 and 0x400, are; Section B.6, probe D5). The kernel declares no static `__shared__` data, so `extern __shared__ char smem[]` is the first byte of user shared memory; its 32-bit `shared::cta` address differs between even and odd cluster ranks in bit 24 (B.6). An explicit kernel must likewise have zero static shared memory (or re-align its base), must keep `smem_A` and `smem_B` at 1024-aligned offsets and `smem_C/D` at 512-aligned offsets, and must request exactly the dynamic size it lays out.

This shared-memory footprint limits residency to one CTA per SM regardless of `__launch_bounds__`. Every barrier offset above is what an explicit kernel must recreate if it wants identical `mbarrier` placement; only the protocol, not the offsets, is semantically required.

### 6.3 TMEM

- Each CTA pair allocates all 512 columns (`Allocator2Sm::allocate(512, &shared_storage.tmem_base_ptr)`).
- Accumulator tensor (`cutlass/detail/sm100_tmem_helper.hpp:71-73`, `cute/atom/mma_traits_sm100_frag.hpp:197-207`): per CTA `((128,128), 1, 1, 4 stages)` with TMEM address `= base + (m << 16) + (128*stage + n)`, i.e. lane = m, column = 128*stage + n. Stage s occupies columns `[128s, 128s+128)`.
- TMEM address encoding: bits [16,32) = datapath lane (0..127), bits [0,16) = column (`cute/pointer.hpp:314-322`, `tmem_allocator_sm100.hpp:44-52`).

---

## 7. Device execution model `[S]`

Section 7 is the protocol specification that the replacement kernel must implement. Part B holds the derivation of every value in it (CuTe algebra, exact PTX operands, `file:line` anchors); each subsection names the Part B rows that back it, and B.4 lists every formula in one table.

### 7.1 CTA identity and warp roles (`sm100_gemm_tma_warpspecialized.hpp:403-616`; B.2 steps 3-11)

- Cluster `(2,2,1)`; `(x, y) = %cluster_ctaid`, rank `r = x + 2y` (`%cluster_ctarank`, column-major cluster layout). `V = r % 2 = x` is the CTA's position in its MMA pair; `is_mma_leader_cta = (V == 0)`; `mma_peer_cta_rank = r ^ 1`; `is_first_cta_in_cluster = (r == 0)`. Pairs are ranks {0,1} (y = 0) and {2,3} (y = 1); a pair computes one 256x128 UMMA tile, so a cluster covers 256x256 of output per work tile.
- Warp roles by `canonical_warp_idx_sync()` (256 threads):

| Warp | Category | Participates | Duty |
|---|---|---|---|
| 0 | MMA | always | TMEM alloc/dealloc; the leader CTA's elected lane issues all `tcgen05.mma` and commits; the peer only tracks state |
| 1 | Sched | rank 0 only | CLC producer loop; in every CTA its elected lane prefetches the A and B descriptors before exiting or looping |
| 2 | MainloopLoad | always | TMA A/B producer; on rank 0 also the CLC-throttle producer |
| 3 | EpilogueLoad | never for beta = 0 (`is_producer_load_needed()` is false) | still prefetches the C and D descriptors and initializes the load-order and throttle barriers in every CTA, then exits |
| 4-7 | Epilogue | always | 128 threads: TMEM reads, alpha/beta math, smem writes; warp 4 issues the TMA stores |

### 7.2 Barriers, pipelines, counts, and PTX

All `mbarrier` objects are 8-byte `ClusterBarrier`/`ClusterTransactionBarrier` (`arch/barrier.h:342-705`) at the Section 6.2 offsets.

Phase bookkeeping (`pipeline/sm90_pipeline.hpp:170-260`): a pipeline state is `{index, phase, count}`; producers start at `{0, 1, 0}`, consumers at `{0, 0, 0}`; `++state` increments `index` and `count` and flips `phase` when `index` wraps at `Stages`. A freshly initialized barrier is in phase 0, so a producer's first wait (parity 1) passes immediately. For the n-th use of a `Stages`-deep pipeline counted from kernel start, `stage = n % Stages`, the consumer waits parity `(n / Stages) & 1`, the producer waits parity `1 ^ ((n / Stages) & 1)`. In this run: mainloop `n = 128 t + q` with `Stages = 8` (every tile starts at stage 0 with the same phase, since 128 is a multiple of 16); accumulator `n = t`, `Stages = 4`; CLC and throttle `n = t`, `Stages = 2`; the store pipeline has no barrier and reuses smem stage `(8t + e) % 4 = e % 4`.

Wait forms: `producer_try_acquire` and `consumer_try_wait` issue one `mbarrier.try_wait.parity` without a suspend-time hint and return a token (`WaitDone` when the phase has already completed, `WaitAgain` otherwise); the blocking `wait` loops on `mbarrier.try_wait.parity ... 0x989680` and is entered only when the token is `WaitAgain` (or unconditionally where no peek precedes it). `mbarrier.test_wait.parity` appears only in the CLC `producer_tail`. In the epilogue the accumulator token is peeked once per tile and waited on at the first subtile.

| Pipeline | Stages | Full barrier: count / arrival | Empty barrier: count / arrival | Init warp | Part B |
|---|---|---|---|---|---|
| Mainloop A/B `PipelineTmaUmmaAsync<8,(2,2,1),(2,1,1)>` | 8 | 1: the leader CTA's elected lane `mbarrier.arrive.expect_tx` 49152 B; completed by TMA bytes from both CTAs of the pair | 2 = `2/2 + 2/1 - 1`: `tcgen05.commit ... multicast::cluster` from each of the two MMA-leader CTAs with mask 0xF | 0 | B.2.12, B.3.1, B.3.2 |
| Epilogue C load `PipelineTransactionAsync<4>` | 4 | 32 (+ 8192 tx B) | 128 | 1 | B.2.13; initialized but never used for beta = 0 (B.3.5) |
| Load-order `OrderedSequenceBarrier<1,2>` | 2 barriers | count 32 each | | 3 | B.2.15; initialized, never arrived on or waited for beta = 0 |
| CLC `PipelineCLCFetchAsync<2,(2,2,1)>` | 2 | 1 + 16 tx B: lanes 0..3 of rank-0 warp 1 each do a remote `mbarrier.arrive.expect_tx` on CTA i's full barrier; completed by the multicast CLC response | 800 = `32 + 4*(32 + 128 + 32)` (kernel `:509-513`; 928 only when beta != 0): every consuming thread does a remote `mbarrier.arrive` on **CTA 0's** empty barrier | 4 | B.2.16, B.3.4 |
| Accumulator `PipelineUmmaAsync<4,(2,1,1)>` | 4 | 1: the leader's `tcgen05.commit ... multicast::cluster` with pair mask 0x3 or 0xC | 256 = 2 x 128: every epilogue thread of both CTAs does `mbarrier.arrive.shared::cluster` on the **leader's** barrier | 5 | B.2.17, B.3.2, B.3.3 |
| CLC throttle `PipelineAsync<2>` | 2 | 32: rank-0 warp 2 lanes arrive locally | 32: rank-0 warp 1 lanes arrive (remote form to CTA 0) | 3 | B.2.18, B.3.1.12, B.3.4.4 |
| TMEM dealloc `ClusterBarrier` | 1 | count 32; pair handshake at exit (7.6) | | 0 | B.2.19, B.3.2.13 |

Masks (`pipeline/sm100_pipeline.hpp:55-110, 598-604`): the mainloop release mask is `calculate_multicast_mask<kRowCol>` = all CTAs with the same `x/2` (all, since cluster M = 2) or the same `y`, i.e. **0xF** for every CTA; the accumulator commit mask is `calculate_umma_peer_mask` = the pair, **0x3** for ranks {0,1}, **0xC** for ranks {2,3}. Both are 16-bit `"h"` register operands of `tcgen05.commit`; the TMA multicast mask (`0x5 << x`, 7.5) is likewise a 16-bit `"h"` operand.

The peer-bit trick: `Sm100MmaPeerBitMask = 0xFEFFFFFF` (`cute/arch/copy_sm100_tma.hpp:45`). The 2SM TMA loads and the accumulator consumer release apply it to a barrier's 32-bit `shared::cta` address so that the odd CTA of a pair addresses the even (leader) CTA's barrier; this relies on bit 24 of the address encoding the CTA rank within the cluster (B.6, probe D5).

PTX primitives on the executed path (`arch/barrier.h`, `cute/arch/*`; each string was checked against the source, B.5 gives the per-warp inventory):

```
mbarrier.init.shared::cta.b64 [addr], count;                                           // elected lane of the initializing warp
mbarrier.arrive.shared::cta.b64 _, [addr];                                             // local arrive (throttle producer, rank 0 warp 2)
mapa.shared::cluster.u32 rem, addr, cta_id;  mbarrier.arrive.shared::cluster.b64 _, [rem];    // remote arrive (CLC/throttle release to CTA 0; dealloc handshake)
mbarrier.arrive.expect_tx.shared::cta.b64 _, [addr], 49152;                            // leader's arrive_and_expect_tx on mainloop full[s]
setp.eq.u32 p, lane_lt_4, 1;  @p mapa.shared::cluster.u32 rem, addr, lane;  @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [rem], 16;   // CLC full[s] arming, all 32 sched lanes execute, lanes 0..3 act
mbarrier.try_wait.parity.shared::cta.b64 P, [addr], phase;  selp.b32 tok, 1, 0, P;     // try/peek (no suspend hint)
LAB: mbarrier.try_wait.parity.shared::cta.b64 P, [addr], phase, 0x989680;  @P bra DONE;  bra LAB;  DONE:   // blocking wait (ticks in a register)
setp.eq.u32 P2, pred, 1;  @P2 mbarrier.test_wait.parity.shared::cta.b64 P, [addr], phase;  selp.b32 tok, 1, 0, P;   // test_wait: CLC producer_tail only
fence.mbarrier_init.release.cluster;                                                   // all threads, after each of the 6 pipeline constructors
barrier.cluster.arrive.relaxed.aligned;   barrier.cluster.wait.aligned;                // prologue rendezvous
fence.proxy.async.shared::cta;                                                         // before every TMA store issue; after every CLC response decode
bar.sync 1, 128;   bar.sync 6, 160;   bar.arrive 6, 160;                               // named barriers: EpilogueBarrier = 1 (epilogue warps), TmemAllocBarrier = 6 (warp 0 arrives, warps 4-7 sync)
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [addr], mask16;   // mainloop release (0xF) and accumulator commit (0x3/0xC); elected lane
mbarrier.arrive.shared::cluster.b64 _, [addr & 0xFEFFFFFF];                            // accumulator consumer release (umma_arrive_2x1SM_sm0): every epilogue thread; NOT a tcgen05.commit
tcgen05.wait::ld.sync.aligned;                                                         // fence_view_async_tmem_load, once per tile before the release
elect.sync %rx|%px, 0xFFFFFFFF;  @%px mov.s32 pred, 1;  mov.s32 lane, %rx;             // elect_one_sync
```

Not executed for beta = 0: `mbarrier.expect_tx.shared::cta.b64` (C-load producer), `cp.async.bulk.wait_group.read 0` (`store_tail`), the C-load `cp.async.bulk.tensor` form (B.3.5), `fence.release.sync_restrict::shared::cta.cluster` (only in `store_query_response`, never called). No `tcgen05.fence::before_thread_sync`/`after_thread_sync` exists anywhere in the CUTLASS C++ tree.

### 7.3 Prologue executed by every CTA (in this order; anchors, counts and PTX in B.2)

1. Warp category, `elect_one_sync` predicate, cluster rank, leader/peer flags (B.2 steps 3-6).
2. Collectives constructed: every thread reads `alpha` and `beta` from `params`; warp 1's elected lane prefetches the A and B descriptors, warp 3's elected lane the C and D descriptors, in every CTA (`prefetch.tensormap`); descriptors are never modified on the device (B.2 steps 8-10).
3. The six pipeline objects are constructed: the elected lane of each initializing warp issues the `mbarrier.init`s with the 7.2 counts, every thread executes `fence.mbarrier_init.release.cluster` once per constructor (6 times); warp 0's elected lane initializes `tmem_dealloc` with count 32 (B.2 steps 12-19).
4. `barrier.cluster.arrive.relaxed.aligned` (B.2 step 20).
5. `load_init` (the TMA coordinate and smem tensors of 7.5), pipeline states, `%cluster_ctaid`, the two masks, the scheduler with the initial tile from `blockIdx`, the TMEM accumulator tensor with base 0 (B.2 steps 21-25).
6. `barrier.cluster.wait.aligned`; no remote barrier operation precedes this (B.2 step 26).
7. Role dispatch (B.2 step 27). Warp 2 and rank 0's warp 1 start with `griddepcontrol.wait`; warp 1 of ranks 1-3 and warp 3 of every CTA exit here.

### 7.4 Persistent CLC tile scheduler (`sm100_tile_scheduler.hpp`; B.1.1 step 11, B.3.4)

Host parameters for this run: 32 x 32 cluster tiles (`ctas_m = round_up(ceil_div(8192, 256) * 2, 2) = 64`, `ctas_n = round_up(ceil_div(8192, 128), 2) = 64`), raster order **AlongN** (heuristic `tiles_n > tiles_m ? AlongM : AlongN` on equal counts, with the 65535 grid-Y guard not triggering), swizzle disabled (`max_swizzle_size = 0` -> divisor 0), launch grid `(64, 64, 1)` after the AlongN transpose `grid = ((ctas_n/2)*2, (ctas_m/2)*2, 1)`.

Tile mapping in units of the 128x128 CTA tile (`swizzle_and_rasterize` + `possibly_transpose_work_tile`, `sm100_tile_scheduler.hpp:666-677, 695-814`; the serpentine flip mentioned in a source comment is not implemented):

- Initial tile from the launch coordinate `(x_l, y_l) = blockIdx`: `(tm, tn) = (2*(y_l/2) + x_l%2, 2*(x_l/2) + y_l%2)`, `L = 0`.
- CLC tile from the first-CTA id `(x0, y0)` of a cancelled cluster (both even): `(tm, tn) = (y0 + ctaid.x, x0 + ctaid.y)` (the general form `2*(y0/2) + x0%2 + ctaid.x`, `2*(x0/2) + y0%2 + ctaid.y` with the intra-cluster offsets `ctaid.{x,y}`).
- For 8192-cubed `[E]`: every launch coordinate maps to a distinct tile; rank r of the cluster at launch cluster `(CX, CY)` computes rows `[256*CY + 128*(r%2), +128)` and columns `[256*CX + 128*(r/2), +128)`.

Protocol (PTX with operands in B.3.4):

- Scheduler warp (rank 0 only), per work tile: `griddepcontrol.wait` once; then { throttle `consumer_wait` + `consumer_release`; `advance_to_next_work` = `producer_acquire` (wait own CLC `empty[idx]`; all lanes execute the predicated remote `arrive.expect_tx` so that lanes 0..3 arm CTA 0..3's `full[idx]` with 16 bytes) then the elected lane issues `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [response[idx]], [full[idx]]`; `++producer_state`; `fetch_next_work` (consume its own response like every other role) } until the fetched tile is invalid; then `producer_tail` (`test_wait` + `wait` on both empty barriers).
- Every participating warp of every CTA, once per work tile and **before** working on the tile it just fetched: `fetch_next_work` = wait own `full[idx]` (parity `(n>>1)&1`); `ld.shared.b128` the 16-byte response; `clusterlaunchcontrol.query_cancel.is_canceled.pred.b128`, `selp`, `@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x0, y0, z0, _}`; `fence.proxy.async.shared::cta`; remote `mbarrier.arrive` on **CTA 0's** `empty[idx]` (800 arrivals); map `(x0, y0)` to `(tm, tn)` as above. An invalid ("not cancelled") response ends every role loop; its `x0/y0/z0` are undefined and must not be read.
- Slot and parity: the n-th query of a cluster (n = 0, 1, ...) uses `idx = n % 2`; the producer waits parity `1 ^ ((n >> 1) & 1)` on `empty[idx]`, the consumers parity `(n >> 1) & 1` on `full[idx]`. The response lands at the same smem offset in all four CTAs and each CTA's own `full[idx]` receives the 16 transaction bytes; the full barriers of CTAs 1-3 are completed entirely by remote actions (rank 0's `arrive.expect_tx` plus the multicast bytes) and their empty barriers are initialized but never used.
- Throttle: rank 0's warp 2 performs `producer_acquire` + `producer_commit` on the throttle pipeline at the start of every tile and reaches the next tile only after consuming that tile's CLC response; the scheduler warp must `consumer_wait` the throttle before every query. Hence query `i` can be issued only after the loader has begun tile `i`, and at most one query is outstanding beyond the tile currently being loaded. A cluster that processes `T` tiles (the launch tile plus `T-1` CLC tiles) issues exactly `T` queries: `T-1` answered "cancelled" and the last answered "not cancelled". Summed over the launch: 1024 queries, of which `N_native` fail, `N_native` being the number of natively launched clusters (at most `148 / 4 = 37` co-resident clusters on a 148-SM B200 with one 230400-byte CTA per SM) `[R]`.

K coverage: 128 k-tiles per tile starting at 0; `compute_epilogue` is always true; `fixup` is a no-op (this scheduler never splits K). Which clusters become resident, how many tiles each processes and in what order is `[R]`; the mapping and exactly-once coverage are fixed.

### 7.5 Mainloop producer: warp 2 of every CTA (`sm100_mma_warpspecialized.hpp:497-641`, kernel `:617-679`; B.3.1)

Per work tile `(tm, tn)` for this CTA at `(x, y)`, for each k-tile `q = 0..127` (stage `s = q % 8`, producer parity `1 ^ ((q >> 3) & 1)` within the tile, since every tile starts at `n = 128 t`):

1. `producer_acquire`: peek then wait `empty[s]`; if this lane is the leader (`elect_one_sync` predicate, `V == 0`, warp 2): `mbarrier.arrive.expect_tx.shared::cta.b64 _, [full[s]], 49152` (`TmaTransactionBytes = 2 * (16384 + 8192)`).
2. The elected lane issues two TMA loads with `cache_hint = TMA::CacheHintSm100::EVICT_NORMAL = 0x1000000000000000` (`copy_sm90_desc.hpp:193-197`, the default of `with()`):
   - A (`SM100_TMA_2SM_LOAD_MULTICAST_3D`): coordinates `(64q, 128*tm + 64*y, 0)`, box 64 K x 64 M, destination `smem + 33792 + 16384*s + 8192*y` (linear, pre-swizzle), barrier `(smem + 8s) & 0xFEFFFFFF`, multicast mask `0x5 << x` (ranks `x` and `x+2`).
   - B (`SM100_TMA_2SM_LOAD_3D`): coordinates `(64q, 128*tn + 64*x, 0)`, box 64 K x 64 N, destination `smem + 164864 + 8192*s`, barrier `(smem + 8s) & 0xFEFFFFFF`, no multicast.
   The two CTAs with the same `x` (ranks `x` and `x+2`) each load a different 64-row half of A into the corresponding half of the 128x64 stage in both destination CTAs; every completion that lands in either CTA of a pair credits the **leader's** `full[s]`, six 8192-byte boxes per stage = 49152 (B.3.1, last paragraph). The peer's full barriers are never arrived on or waited on.
3. `++producer_state`.

Per work tile the kernel calls `load` twice, for 8 then 120 k-tiles (the load-order `arrive` between them is skipped for beta = 0), then `__syncwarp`, `fetch_next_work` (7.4), and loops while the tile is valid. Rank 0's warp 2 additionally does the throttle `producer_acquire` + `producer_commit` before the loads of every tile. After the loop: `load_tail` = wait on all 8 empty barriers with the producer parity. The k-tile loop is `#pragma unroll 1` (`CUTLASS_PRAGMA_NO_UNROLL`, `:607`). For 8192-cubed `[E]`: for the cluster at `(CX, CY)`, rank `r = x + 2y` issues A rows `256*CY + 128*x + 64*y` and B columns `256*CX + 128*y + 64*x`; 2 TMA instructions per k-tile, 256 per tile, per CTA.

Smem layouts (`sm100_mma_warpspecialized.hpp:181-189`; atom `Layout_K_SW128_Atom<half_t> = Swizzle<3,4,3> o smem_ptr[16b] o (8,64):(64,1)`):

```
SmemLayoutA : Sw<3,4,3> o ((128,16),1,4,8) : ((64,1),0,16,8192)   // half units; stage stride 16384 B
SmemLayoutB : Sw<3,4,3> o (( 64,16),1,4,8) : ((64,1),0,16,4096)   // half units; stage stride  8192 B
```

In plain form: A element `(m, k)` of stage `s` sits at pre-swizzle byte `16384*s + 128*m + 2*k`, B element `(n, k)` at `8192*s + 128*n + 2*k`, and the physical byte is `addr ^ (((addr >> 7) & 7) << 4)` (`Swizzle<3,4,3>`: 16-byte chunk index XOR row-mod-8). The TMA unit applies this XOR on write and the UMMA unit on read through the `SWIZZLE_128B` descriptor, so the kernel never computes it; it only has to keep every stage base 1024-byte aligned.

### 7.6 MMA role: warp 0 of every CTA (kernel `:726-805`; `mma`, `sm100_mma_warpspecialized.hpp:651-712`; B.3.2)

Setup: `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [smem + 384], 512` (whole warp, both CTAs of the pair), `__syncwarp`, `bar.arrive 6, 160`, read `tmem_base_ptr` (`T`, expected 0 `[R]`), build the A/B smem descriptors from this CTA's own `smem_A`/`smem_B` addresses.

Per work tile (both CTAs): `fetch_next_work` **before** computing the tile; `acc_stage = t % 4`; leader only: `producer_acquire` on the accumulator `empty[acc_stage]` (parity `1 ^ ((t >> 2) & 1)`), then for each k-tile: peek/wait `full[s]`, four `tcgen05.mma` (k-block loop unrolled, `:699`; k-tile loop `#pragma unroll 1`, `:680`), `consumer_release` = `tcgen05.commit ... multicast::cluster [empty[s]], 0xF`; after the 128th k-tile `producer_commit` = `tcgen05.commit ... multicast::cluster [acc_full[acc_stage]], 0x3 | 0xC`; both CTAs `++acc_producer_state`. The peer's warp 0 executes only the CLC consume and the state increment.

The `tcgen05.mma` (`cute/arch/mma_sm100_umma.hpp:563-586`), issued by the elected lane: `setp.ne.b32 p, scale_c, 0; tcgen05.mma.cta_group::2.kind::f16 [tmem_c], desc_a, desc_b, idesc, {0,0,0,0,0,0,0,0}, p;` (the eight zero mask operands are registers).

- `tmem_c = T + 128 * acc_stage` (column offset; lane bits 0). TMEM element `(m, n)` of stage `st` is at `T + (m << 16) + 128*st + n` (Section 6.3).
- `idesc` = `InstrDescriptor` (`cute/arch/mma_sm100_desc.hpp:416-443`, built by `make_instr_desc`, `:474-506`): bits [0,2) `sparse_id2 = 0`, [2] `sparse_flag = 0`, [3] `saturate = 0`, [4,6) `c_format = 1` (F32), [6] reserved, [7,10) `a_format = 0` (F16), [10,13) `b_format = 0` (F16), [13] `a_negate = 0`, [14] `b_negate = 0`, [15] `a_major = 0` (K), [16] `b_major = 0` (K), [17,23) `n_dim = 128 >> 3 = 16`, [23] reserved, [24,29) `m_dim = 256 >> 4 = 16`, [29] reserved, [30,32) `max_shift = 0`. Value **0x10200010**, passed in an `"r"` register (`uint32_t(idescE >> 32)`).
- `desc_a`/`desc_b` = `SmemDescriptor` (`mma_sm100_desc.hpp:101-127`, built by `make_umma_desc<Major::K>`, `:737-833`): bits [0,14) `start_address = (addr >> 4) & 0x3FFF` (14-bit field, `uint16_t start_address_ : 14`, assigned from `static_cast<uint16_t>(addr >> 4)` at `:757-758`; it drops the rank bit 24 so leader and peer hold identical descriptors), [14,16) unused, [16,30) `leading_byte_offset = 1` (16 B), [30,32) unused, [32,46) `stride_byte_offset = 64` (1024 B, the 8-row group stride), [46,48) `version = 1`, [48] unused, [49,52) `base_offset = 0`, [52] `lbo_mode = 0`, [53,61) unused, [61,64) `layout_type = 2` (`SWIZZLE_128B`; enum `SWIZZLE_NONE 0, SWIZZLE_128B_BASE32B 1, SWIZZLE_128B 2, SWIZZLE_64B 4, SWIZZLE_32B 6`, `:82-88`). Constant part **0x4000404000010000**. A: `addr = smem + 33792 + 16384*s + 32*kb`; B: `addr = smem + 164864 + 8192*s + 32*kb` (`kb = 0..3`), i.e. the low word advances by `2` per k-block and `1024` (A) / `512` (B) per stage; CUTLASS adds the offset to the low 32-bit word (`DescriptorIterator::operator+`, `:863-874`, the `#else` branch for CUDA 13.3), which cannot carry out of the 14-bit field for any admissible smem base (B.3.2 step 8). The `cta_group::2` MMA reads the peer's halves at the identical CTA-relative addresses, which is why both CTAs use the same `SharedStorage` layout.
- Accumulate predicate `scale_c`: 0 for the first MMA of each work tile (`q = 0, kb = 0`, clears the TMEM stage), 1 for the other 511.

Both commit forms are issued right after the corresponding `tcgen05.mma` instructions with no wait, and rely on `tcgen05.commit` tracking the completion of all prior `tcgen05` operations of the issuing thread; they must be issued by the same elected thread that issued the MMAs and cannot be replaced by a plain `mbarrier.arrive`.

Termination (kernel `:778-805`), after the last valid tile:

1. `griddepcontrol.launch_dependents` (every warp 0, all lanes).
2. `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;`
3. Leader only: `producer_tail` on the accumulator pipeline (wait all 4 empty barriers: both CTAs' epilogues have released every stage).
4. Pair handshake on `tmem_dealloc` (count 32): the peer's 32 lanes remote-arrive on the leader's barrier; both wait parity 0 on their own barrier; the leader's 32 lanes remote-arrive on the peer's barrier. Net effect: the peer cannot deallocate before the leader has finished `producer_tail`.
5. `tcgen05.dealloc.cta_group::2.sync.aligned.b32 T, 512;` in both CTAs.

### 7.7 Epilogue consumers: warps 4-7 of every CTA (`sm100_epilogue_tma_warpspecialized.hpp:573-956`, kernel `:869-955`; B.3.3)

Setup: `bar.sync 6, 160` (wait for the TMEM address), read `tmem_base_ptr`.

Per work tile (all 128 threads): `fetch_next_work` **before** the store; `acc_stage = t % 4`; `store(...)`; advance. Inside `store`, `thread_idx = threadIdx.x % 128`, `w = thread_idx / 32` (0..3), `l = lane`; thread `(w, l)` owns output row `32w + l` of the CTA tile, which satisfies the hardware rule that warp `w` may only read TMEM lanes `[32*(w % 4), +32)`. The C fragment (16 registers) is zeroed once per tile.

For each of the 8 subtiles `e = 0..7` (loop fully unrolled, `:818-820`; smem stage `idx = e % 4`):

1. First subtile only: `consumer_wait` on the accumulator `full[acc_stage]` (parity `(t >> 2) & 1`), after a single peek at the start of the tile.
2. `tcgen05.ld.sync.aligned.32x32b.x16.b32 {r0..r15}, [T + ((32w) << 16) + 128*acc_stage + 16e]`: lane `l` receives columns `16e..16e+15` of row `32w + l`. No `tcgen05.wait::ld` precedes the use of these registers for subtiles 0..6 (B.6, D11).
3. Last subtile only: `tcgen05.wait::ld.sync.aligned;` then the accumulator release: each of the 128 threads `mbarrier.arrive.shared::cluster.b64 _, [(smem + 272 + 8*acc_stage) & 0xFEFFFFFF]` (both CTAs -> 256 on the leader); `++acc_consumer_state`. This happens **before** the last subtile's arithmetic and store.
4. Arithmetic per register `j` (Section 8): `t = alpha * acc[j]`, `d = beta * C[j] + t` with `C[j] = 0`.
5. Sixteen `st.shared.b32`: register `j` (column `n = 16e + j`) goes to byte `smem + 512 + 8192*idx + 2048*w + 128*n + 4*((l & 7) + 8*(((l >> 3) & 3) ^ (j & 3)))` (B.3.3 step 17: `Swizzle<2,5,2> o (32,4):(1,32)` atoms tiled N-first then M over the (128,16) subtile; a warp writes 128 contiguous bytes per column).
6. `fence.proxy.async.shared::cta` (all 128 threads); `bar.sync 1, 128`; warp 4, all 32 lanes with uniform operands (no lane election anywhere in the store path): four `cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [descD, {128*tm + 32*i, 128*tn + 16*e, 0}], [smem + 512 + 8192*idx + 2048*i]` for `i = 0..3`, then `cp.async.bulk.commit_group`, `++store_state`, `cp.async.bulk.wait_group.read 1` (`always_wait = true`), then `bar.sync 1, 128` (all 128). Whether ptxas collapses the 32 lanes' identical store instructions into one uniform-datapath operation is `[C]` (D11).
7. ReuseSmemC bookkeeping: `wait_group.read 1` guarantees that the store issued one subtile earlier has finished reading its smem stage; for beta = 0 no barrier is released (`load_pipe_consumer_state` is dead state).

After the last tile: `store_tail` (`:1269-1291`) executes nothing for beta = 0 (its body is gated on `is_producer_load_needed()`), so there is no `cp.async.bulk.wait_group.read 0`: the CTA exits with at most one bulk group (the last subtile's four stores) whose smem read was not awaited; the global writes complete as part of kernel completion `[R]` (D15). The replacement must preserve or tighten this, never loosen it. `store` also builds residue/coordinate tensors that the linear-combination callbacks never read; D has no per-thread predication (all tiles are full).

### 7.8 Warp 3 C producer (beta != 0 only)

Not exercised by the Section 0 run; the protocol is recorded for reference in B.3.5.

### 7.9 CLC consumption per role

Covered in 7.4 (every role consumes each response once per tile, before working on it; 800 releases per response on CTA 0's empty barrier).

### 7.10 Exit safety

There is no cluster-wide synchronization at kernel exit. The guards that keep a CTA alive while remote CTAs may still signal into its barriers are exactly: (1) every CTA's warp 2 runs the mainloop `producer_tail` (8 empty-barrier waits, absorbing the last multicast `tcgen05.commit` arrivals from both pair leaders); (2) CTA 0's warp 1 runs the CLC `producer_tail` (all 800 releases of the final response); (3) the leader's warp 0 runs the accumulator `producer_tail` (256 arrivals per stage from both CTAs' epilogues); (4) the TMEM deallocation handshake between the pair's MMA warps. The throttle pipeline is CTA-local. Warp 1 of CTAs 1..3 and warp 3 of every CTA fall through to the empty final branch and exit right after `pipeline_init_wait`. An explicit kernel must keep every one of these tails.

---

## 8. Numerical semantics `[S]` `[C]`

- MMA: `tcgen05.mma.cta_group::2.kind::f16` with FP16 inputs and FP32 accumulation, K=16 per instruction, K order 0..K-1 in steps of 16 (stage-major, k_block-minor). The first instruction of a tile overwrites the TMEM stage (`ScaleOut::Zero`); the rest accumulate. The internal summation order/precision inside one instruction is hardware-defined `[R]` but identical for any kernel issuing the same instruction sequence on the same operands.
- Epilogue tree (`Sm90LinearCombination`, `sm90_callbacks_tma_warpspecialized.hpp:182-192`; performance specialization `sm90_visitor_compute_tma_warpspecialized.hpp:224-342`):
  - This specialization is selected because the primary `Sm90Compute` template has a defaulted fifth parameter (`class = void`, `sm90_visitor_compute_tma_warpspecialized.hpp:84-91`) and `Sm90ScalarBroadcast` provides `is_zero()`; the generic tree's OR-over-children predicate (which would report the C load as always needed for a non-void C) is therefore **not** what runs.
  - `is_C_load_needed() = (not beta.is_zero()) && C is non-void` -> for the example, exactly `beta_eff != 0.0f`, where `beta_eff = beta_ptr ? *beta_ptr : beta` is read once per CTA in the `Sm90ScalarBroadcast` constructor (the beta L-stride is 0). `is_zero()` compares `scalar == Element(0)`.
  - `is_producer_load_needed() = (dBeta L-stride != 0 && beta_ptr != nullptr) || is_C_load_needed() || false` = `is_C_load_needed()` for this argument set.
  - `visit`: `frg_added = alpha * acc` (per element `float * float`, `cutlass::multiplies`), then, because C is non-void, **always** `frg_I = multiply_add(beta_frag, C_frag, frg_added)` = per element `beta * C + t` (`cutlass::multiply_add<float>`, `functional.h:585-590`: `C(a) * C(b) + c`). When C is not loaded, `get_consumer_store_callbacks` zero-fills the C register fragment once (`cute::clear(args.tCrC)`, line 336-338), so beta == 0 computes `0 * 0 + t`.
  - Consequences: for beta == 0, D equals `alpha * acc` bitwise for all finite values except that a `-0.0` product becomes `+0.0`. Whether `beta*C + t` is emitted as FMUL+FADD or contracted to FFMA is a compiler decision (`nvcc` default `-fmad=true` contracts) `[C]`. The explicit kernel must use the same expression form and build flags, or an explicit `fmaf` if SASS inspection shows contraction in the baseline.
  - Conversions are identity (float -> float, `round_to_nearest`).
- Harness data: because A, B, C are integer-valued in [-8, 8] (Section 3), for any K <= 262144 the FP32 accumulation is exact in any order and `alpha=1, beta=0` gives D = acc exactly. Passing the harness therefore does not by itself prove that the explicit kernel reproduces the baseline's instruction sequence; that is established structurally (Section 7, Part B, Part D). An optional stronger check, outside the fixed command line, is a non-integer fill at the same shape with alpha = 1, beta = 0 and a bitwise compare of the two kernels' D.

---

## 9. Operation ledger for the default run `[E]`

| Item | Per unit | Total |
|---|---|---|
| Output CTA tiles (128x128) | | 4096 |
| Cluster work tiles (256x256) | | 1024 |
| K tiles per work tile | 128 | |
| TMA A loads (64x64 boxes) | 1 per CTA per k tile | 524288 |
| TMA B loads (64x64 boxes) | 1 per CTA per k tile | 524288 |
| `tcgen05.mma` | 512 per pair per tile (1 clear + 511 accumulate) | 1048576 |
| FLOP per MMA | 2*256*128*16 = 1048576 | 2*8192^3 |
| `tcgen05.ld` warp instructions | 32 per CTA tile (8 subtiles x 4 warps) | 131072 |
| C TMA load instructions | 0 (beta = 0) | 0 |
| D TMA store instructions (2048 B boxes) | 32 per CTA tile (4 per subtile, issued from warp 4) | 131072 (256 MiB) |
| `cp.async.bulk.commit_group` / `wait_group.read 1` | 8 each per CTA tile per lane of warp 4 | |
| `mbarrier` full/empty round trips (mainloop) | 128 per pair per tile | |
| Tensor-map encodes | 6 per `initialize()` | 66 for 1 warm-up + 10 timed runs |
| Kernel launches | | 11 (plus the reference GEMM and the compare kernel) |

CLC queries, resident clusters, and tiles per physical cluster are `[R]`.

---

## 10. Boundary behavior `[S]`

Not exercised by the Section 0 run: all 4096 CTA tiles are full, K = 128 x 64 exactly, L = 1, swizzle disabled, so no residue, clipping or zero-fill path executes and the replacement may assume full tiles. Reference only: the baseline handles partial M/N tiles by TMA clipping and zero fill (`OOB_FILL_NONE`), handles K residues by zero-filled last stages with all four MMAs still issued, and rejects `K % 8 != 0` or `M % 4 != 0` in `can_implement` (Section 3).

---

## 11. Facts that need the toolchain or the machine

Fixed by Section 0 (confirm only, D14/H5): `sm_100a` cubin, `Release`/`NDEBUG`, `CUTLASS_ENABLE_GDC_FOR_SM100=ON` so that `griddepcontrol.wait` / `griddepcontrol.launch_dependents` are emitted (`arch/grid_dependency_control.h:50-54`, root `CMakeLists.txt:463-475`), legacy default stream, no direct driver calls, no host adapter, CUDA 13.3 (hence the 32-bit low-word descriptor increment). The replacement must be compiled by the same CMake target flags (B3 in Part C records the exact `nvcc` line).

`[C]` (compiler decisions, D11):

- SASS lowering of: the warp-wide TMA store issue (all 32 lanes of warp 4 execute each of the four store instructions; expected one uniform-datapath `UTMASTG`-class instruction per store per warp), FFMA contraction of `beta*C + t` in the epilogue, register count and spills under `__launch_bounds__(256,1)`, `st.shared` widths (16 scalar stores per subtile per thread are layout-determined), how the 16 destination registers of `tcgen05.ld` are protected when consumed without an explicit `tcgen05.wait::ld` (subtiles 0..6), how the `__grid_constant__` scheduler parameters are materialized.
- `sizeof`/`offsetof` of `SharedStorage` and its members as compiled (expected values in Section 6.2, D2).

`[R]` (driver/hardware/runtime):

- The raw 128-byte descriptor contents and driver acceptance of the zero L-stride and of `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` (D3, D13).
- The dynamic shared-memory base of each cluster rank (low 24 bits 0 or 0x400, the rank expected in bits [24,28)), its 1024-byte alignment, and the TMEM base returned by `tcgen05.alloc` (D5, D6).
- Number of natively launched clusters, CLC cancellation order, tiles per cluster, SM placement and query latency (D9).
- Baseline correctness output and the timing/noise distribution on the target B200 (D12).
- Completion semantics of the un-awaited final bulk store group at kernel exit (Section 7.7, D15).

Part C describes how each item is measured through the fixed build and run commands.

---

## 12. Performance-equivalence contract

"Same runtime performance" means no material regression under identical conditions: same GPU and clock/power state, same driver and toolkit, same architecture flags, the fixed inputs, shape, alpha/beta and swizzle of Section 0, same warm-up, same stream, same iteration count, same timing method. Both of these intervals must be measured and reported:

1. The example's own interval, printed as `Avg runtime` / `GFLOPS` by the fixed command (it includes the six host encodes, the two `cudaFuncSetAttribute` calls and the launch per iteration). This is the primary number.
2. Optionally, kernel-only durations from `nsys` on the same binary and command, to separate host overhead from kernel time.

Report medians and spreads over enough repetitions, plus registers, spills and dynamic smem from `cuobjdump --dump-resource-usage` (Part C, B1). The mechanisms in Section 1 item 2 must all be present in the replacement, and the instruction-stream shape should match: the producer's and the MMA warp's k-tile loops are `#pragma unroll 1` (`sm100_mma_warpspecialized.hpp:607, 680`), the four k-block MMAs are unrolled (`:699`), and the eight epilogue subtiles are fully unrolled (`sm100_epilogue_tma_warpspecialized.hpp:175, 818-820`). A single wall-clock number is not sufficient evidence.

---

## 13. Corrections and refinements to `de-abstraction-plan.log`

Items in the draft that were wrong or imprecise, with the corrected fact:

1. The accumulator-pipeline consumer release (`umma_arrive_2x1SM_sm0`) is a plain `mbarrier.arrive.shared::cluster.b64` on the peer-bit-masked address, **not** a `tcgen05.commit` (`arch/barrier.h:905-921`). The draft listed it among the `tcgen05.commit` forms.
2. beta == 0 does not skip the linear-combination multiply-add. The `homogeneous_multiply_add` node is compiled in whenever C is non-void; the C fragment is zero-filled and `0*0 + alpha*acc` is computed (Section 8). The draft described only "alpha * accumulator".
3. (reference only, swizzle is disabled here) The swizzle size is used verbatim, not rounded to a power of two.
4. The raster heuristic is `tiles_n > tiles_m ? AlongM : AlongN` with a 65535 grid-Y guard (equal counts give `AlongN` here); the draft stated the outcome without the rule.
5. The CTA grid is computed as `ceil_div(M, 256) * 2` (MMA tile, then times AtomThrShape), not `ceil_div(M, 128)`; identical (64) for this configuration.
6. The CLC full-barrier `expect_tx` is performed remotely by lanes 0..3 of the scheduler warp onto each CTA's own full barrier (one lane per CTA), not by a single arrival; the draft did not say who arrives where.
7. The mainloop empty barrier count is 2 (`cluster_m/atom_m + cluster_n/atom_n - 1`), fed by the two leaders' multicast commits with mask 0xF; the draft had the count right but did not derive the mask from `calculate_multicast_mask<kRowCol>`.
8. The TMA store is issued by all 32 lanes of the first epilogue warp with uniform operands (`issue_tma_store = (warp_idx == 0)`, no elect); the draft said "one warp-uniform store from local epilogue warp 0", which is compatible, but the number of executed PTX instructions per subtile is 32 and the hardware coalescing is `[C]` to confirm.
9. The per-thread register-to-smem copies are 16 scalar 32-bit stores (layout-determined), not a compiler-chosen vector width.
10. The epilogue smem atom is `Layout_MN_SW128_32B_Atom<float>` with `Swizzle<2,5,2>` on byte addresses (32-byte chunks across 128-byte spans, 512-byte period), stated here explicitly.
11. The accumulator release happens before the last subtile's arithmetic and store, not after "the final subtile" completes.
12. The scheduler's serpentine flip exists only in a source comment; the implemented mapping has no flip.
13. `SmemLayoutA/B` are rank 4 `((128,16),1,4,8)` / `((64,16),1,4,8)` descriptor-tiled layouts; the draft's printed form grouped the modes differently.
14. The C and D tensor maps have box (32, 16, 1) = 2048 bytes, not (128, 16, 1); every epilogue subtile is moved by four TMA instructions, so there are 32 D stores (and, for beta != 0, 32 C loads) per CTA tile, not 8. The draft's "eight 8192-byte D TMA stores per logical CTA work tile" and its ledger rows for C/D transfers are wrong (the byte totals are unchanged).
15. (reference only, beta = 0) The C-load producer protocol is: all lanes wait the empty barrier; the elected lane issues the loads and then `mbarrier.expect_tx` (no arrive); all 32 lanes arrive.
16. The reused C/D smem stage is released one subtile after its D store is committed (lag `UnacquiredStages = 1`), not two.
17. The CLC empty-barrier count is 800 for this run (beta = 0), decided at run time by the beta-aware `Sm90TreeVisitor` specialization (`sm90_visitor_compute_tma_warpspecialized.hpp:224-342`); the generic OR-over-children reading that gives 928 is wrong because the specialization replaces it.
18. `can_implement` tests default-constructed strides, so it enforces only `K % 8 == 0` and `M % 4 == 0` (both satisfied); the draft attributed the check to the runtime strides.
19. Everything else the draft claimed about resolved types, stage counts, the SharedStorage total (230400), the A/B TMA coordinates and masks, the instruction descriptor (0x10200010), the smem descriptor constant (0x4000404000010000), and the mainloop operation ledger was confirmed against source.

---

## 14. Source anchors (commit `cb424739`)

- Example and harness: `examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu:93-148, 189-255, 283-341, 343-373, 375-435, 441-484`; `examples/70_blackwell_gemm/CMakeLists.txt:31-56`; `tools/util/include/cutlass/util/reference/device/tensor_fill.h:432-545, 922-939`; `.../reference/device/tensor_compare.h:56-160`; `.../reference/device/thread/gemm.h:150-175`; `.../packed_stride.hpp:60-135`.
- Builders: `include/cutlass/gemm/collective/builders/sm100_umma_builder.inl:47-99, 155-346`; `.../sm100_common.inl:64-133, 197-266, 294-477, 999-1001`; `.../sm100_pipeline_carveout.inl:35-83`; `include/cutlass/epilogue/collective/builders/sm100_builder.inl:72-97, 593-660, 717-840, 967-1148, 1151-1360, 1661-1737`; `include/cutlass/arch/arch.h:44`.
- Kernel: `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:61-401 (types, storage, host), 403-959 (device)`; `include/cutlass/device_kernel.h:114-127`.
- Mainloop: `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:106-240 (types), 303-443 (params/host), 497-547 (load_init), 589-641 (load/tail), 651-712 (mma)`.
- Epilogue: `include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp:106-233 (types/storage), 268-375 (host), 410-450, 461-551 (load), 553-559 (load_tail), 573-956 (store), 1269-1291 (store_tail)`; `include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp:69-82`; `.../sm90_callbacks_tma_warpspecialized.hpp:182-241`; `.../sm90_visitor_compute_tma_warpspecialized.hpp:91-216, 224-342`; `.../sm90_visitor_load_tma_warpspecialized.hpp:86-130, 1010-1120`; `include/cutlass/functional.h:585-590`; `include/cutlass/array.h:658-690, 963-990`.
- Scheduler: `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:55-101, 135-161, 220-250, 340-343, 365-473, 499-528, 590-605, 666-814`; `include/cutlass/gemm/kernel/tile_scheduler_params.h:311-354, 1838-1927`; `include/cutlass/gemm/kernel/tile_scheduler.hpp:207-219`.
- Pipelines/barriers: `include/cutlass/pipeline/sm100_pipeline.hpp:55-110, 118-278, 532-759, 934-1136`; `include/cutlass/pipeline/sm90_pipeline.hpp:170-260, 270-648, 650-707, 765-1010, 1014-1252, 1254-1388`; `include/cutlass/arch/barrier.h:160-335 (named barriers), 342-705 (cluster barriers), 711-755 (fences), 796-947 (umma arrives, tmem fences)`; `include/cute/arch/cluster_sm90.hpp` (cluster arrive/wait, block ids).
- TMA: `include/cute/atom/copy_traits_sm90_tma.hpp:735-870 (gbasis, box truncation), 880-1129 (make_tma_copy_desc), 1400-1520 (multicast masks)`; `include/cute/atom/copy_traits_sm100_tma.hpp:600-830`; `include/cute/arch/copy_sm100_tma.hpp:45-46, 104-128, 283-305`; `include/cute/arch/copy_sm90_tma.hpp:159-191 (C load), 1003-1024 (store), 1214-1260 (fence/arrive/wait)`; `include/cute/arch/copy_sm90_desc.hpp:134-150, 186-197 (cache hints), 200-262, 302-317 (prefetch)`; `include/cute/atom/copy_traits_sm90_tma_swizzle.hpp:45-100`.
- UMMA/TMEM: `include/cute/arch/mma_sm100_desc.hpp:62-127, 239-243, 416-443, 474-545, 690-930`; `include/cute/arch/mma_sm100_umma.hpp:549-587`; `include/cute/atom/mma_traits_sm100.hpp:55-125, 1160-1218`; `include/cute/atom/mma_traits_sm100_frag.hpp:79-211`; `include/cute/atom/mma_traits_sm90_gmma.hpp:84-104 (SW128 atoms)`; `include/cute/arch/tmem_allocator_sm100.hpp:40-183`; `include/cute/arch/copy_sm100.hpp:3576-3604`; `include/cute/atom/copy_traits_sm100.hpp:1594-1604, 325-366, 2777-2830`; `include/cutlass/detail/sm100_tmem_helper.hpp:54-76`; `include/cute/pointer.hpp:277-325`.
- Host launch: `include/cutlass/gemm/device/gemm_universal_adapter.h:311-355, 372-575`; `include/cutlass/cluster_launch.hpp:43, 100-300`; `include/cutlass/arch/grid_dependency_control.h:44-105`; `include/cutlass/arch/config.h:87-93`; root `CMakeLists.txt:463-475`.

---
---

# Part B. Call-stack and lowering trace of `gemm.run()` (user step 1, recorded 2026-09-08)

This part answers: "when `gemm.run()` executes for `./70_blackwell_fp16_gemm --m=8192 --n=8192 --k=8192` on a B200 with CUDA 13.3, what is the exact chain of calls, how do the CuTe layout/coordinate computations resolve, and which inline PTX runs?" It is organized as ordered call chains. Each step names the function and `file:line`, the resolved types/values for this instantiation, the CuTe result where a layout computation happens, and the PTX emitted. Every fact is `[S]` (fixed by source) unless marked `[C]`/`[R]`; Section B.7 collects everything that is not statically determinable together with the experiment that determines it. Part A (Sections 1-14 above) holds the supporting tables (types, barrier counts, smem offsets); this part cross-references them instead of repeating them.

Conventions used below: `smem` is the 32-bit `shared::cta` address of the dynamic shared-memory base (`extern __shared__ char smem[]`); byte offsets from Section 6.2 are added to it. `r = x + 2y` is the CTA rank in the cluster with `(x, y) = %cluster_ctaid.{x,y}`; `V = x` is the CTA's position in its MMA pair; `(CX, CY) = (blockIdx.x/2, blockIdx.y/2)` is the launch-space cluster coordinate; `(tm, tn)` is the CTA's 128x128 output tile index; `q` is the K-tile index (0..127 for K = 8192); `s` is a mainloop smem stage (0..7); `e` is an epilogue subtile (0..7).

## B.1 Host chain: from `gemm.run()` to `cudaLaunchKernelExC`

`gemm.run()` launches with `params_` that `gemm.initialize()` built one call earlier (`70_blackwell_fp16_gemm.cu:397-400` for the warm-up, `417-420` for each timed iteration). The launch therefore depends on the `initialize()` chain, which is listed first.

### B.1.1 `gemm.initialize(arguments, workspace)` (`gemm_universal_adapter.h:311-355`)

| # | Call (file:line) | Resolved values / result | Status |
|---|---|---|---|
| 1 | `GemmKernel::initialize_workspace(args, workspace, stream=nullptr, adapter=nullptr)` (`sm100_gemm_tma_warpspecialized.hpp:354-380`) -> `CollectiveEpilogue::initialize_workspace` -> `FusionCallbacks::initialize_workspace` (each visitor node returns `kSuccess`); `TileScheduler::initialize_workspace` -> `Params::initialize_workspace` returns `kSuccess` | No memory touched; workspace size is 0 | S |
| 2 | `params_ = GemmKernel::to_underlying_arguments(args, workspace)` (`:257-298`): `problem_shape_MNKL = append<4>((8192,8192,8192,1), 1)`; workspace offsets 0 | `Params{mode=kGemm, problem_shape, mainloop, epilogue, scheduler, hw_info}` | S |
| 3 | `CollectiveMainloop::to_underlying_arguments(problem_shape, args.mainloop, nullptr, hw_info)` (`sm100_mma_warpspecialized.hpp:355-418`): `tensor_a = make_tensor(ptr_A, make_layout((M,K,L), dA))` with `dA = (8192, _1, 0)`; `tensor_b` likewise `(N,K,L):(8192,_1,0)`; `cluster_shape = select_cluster_shape((2,2,1), hw_info.cluster_shape)` -> static `(2,2,1)`; `cluster_layout_vmnk = tiled_divide(make_layout((2,2,1)), make_tile(Layout<_2>))` = `((_2),_1,_2,_1):((_1),_0,_2,_4)` (the M-rest mode has extent 1 and stride `_0` because `complement(2:1, 2) = 1:0`; as a function it is `(v,0,n,0) -> v + 2n`) | four TMA atoms built in the order A, B, A-fallback, B-fallback (steps 4-5 twice) | S |
| 4 | `make_tma_atom_A_sm100<half_t>(SM100_TMA_2SM_LOAD_MULTICAST{}, tensor_a, SmemLayoutA{}(_,_,_,0), (256,128,64), TiledMma{}, cluster_layout_vmnk)` (`copy_traits_sm100_tma.hpp:729-782`): `mma_tiler_mk = (256,64)`; `g_tile = make_identity_layout((M,K,L)).compose((256,64))` = `(256,64):(E<0>,E<1>)`; `cta_v_tile = layout<1>(mma.thrfrg_A(g_tile))(_,_)` = the V=0 slice `((128,16),1,4):((E<0>,E<1>),0,16*E<1>)`; `num_multicast = size<2>(cluster_layout_vmnk) = 2` | -> `make_tma_copy_atom<half_t>(op, tensor_a, slayout, 2, cta_v_tile)` | S |
| 5 | `detail::make_tma_copy_atom` (`copy_traits_sm90_tma.hpp:1138-1193`): `smem_swizzle = Swizzle<3,4,3>`, `smem_layout = ((128,16),1,4):((64,1),0,16)`; `construct_tma_gbasis` (`:735-870`): `inv_smem_layout = right_inverse(smem_layout)` enumerates smem indices K-fastest (16 within a k-block, then the next k-block at +16, then M at +64) so `sidx2gmode_full = (64:E<1>, 128:E<0>)`; both bases are unit -> `smem_rank = 2`; the L mode is re-appended with extent 1 -> `tma_gbasis = (64,128,1):(E<1>,E<0>,E<2>)` (TMA dims = K, M, L) | `make_tma_copy_desc<half_t>(tensor_a, tma_gbasis, Swizzle<3,4,3>, 2)` (`:905-1129`): `gmem_prob_shape = {8192, 8192, 1}`, strides in elements `{1, 8192, 0}` -> bytes `{16384, 0}` passed as `gmem_prob_stride.data()+1`; box `{64,128,1}` then divided from the last mode: L box 1 absorbs nothing, M box 128 -> 64 => `{64,64,1}`; `elementStrides {1,1,1}`; `tma_format = FLOAT16`; `INTERLEAVE_NONE`; swizzle `get_tma_swizzle_bits(Swizzle<3,4,3>) = B128`, base 16B -> `CU_TENSOR_MAP_SWIZZLE_128B`; `L2_PROMOTION_L2_128B`; `OOB_FILL_NONE`; **`cuTensorMapEncodeTiled(...)` #1** through `CUTLASS_CUDA_DRIVER_WRAPPER_CALL` (`cuda_host_adapter.hpp:109-127`: `cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &pfn, 12000, cudaEnableDefault, &qres)` and then the call through `pfn`); then `cudaDriverGetVersion(&v)` and the fixup `if (v <= 13010 && tensor_bytes < 131072) desc.word[1] &= ~(1 << 21)` (`:1061-1070`), which a CUDA 13.3 driver does not enter (and the 128 MiB tensors would fail the size test anyway) [R: driver version]; on a non-success `CUresult` the argument tuple is printed to `stderr` (`:1072-1089`); in a non-`NDEBUG` build host `assert`s check the 16-byte base alignment, shapes in `[1, 2^32]`, `stride[0] == 1`, byte strides `< 2^40` and multiples of 16, box extents in `[1, 256]`, element strides in `[1, 8]` and the multicast divisibility (`:955-1024`), not executed in the Release build of Section 0; `aux_params.g_stride_ = (E<1>, E<0>, E<2>)` (gmem mode M -> TMA dim 1, K -> dim 0 with scale 1, L -> dim 2) | S (encode result and raw bytes: R) |
| 6 | Same for B via `make_tma_atom_B_sm100` (`copy_traits_sm100_tma.hpp:785-830`): `cta_v_tile = ((64,16),1,4)`, `num_multicast = size<1>(cluster_layout_vmnk) = 1`; `tma_gbasis = (64,64,1):(E<1>,E<0>,E<2>)`; `globalDim {8192,8192,1}`, strides `{16384,0}`, box `{64,64,1}`, FLOAT16, SWIZZLE_128B; **encode #2**; `g_stride_ = (E<1>,E<0>,E<2>)` | | S |
| 7 | **Encodes #3, #4**: steps 4-6 repeated for the fallback atoms (byte-identical descriptors; never prefetched or used on the device) | `Params{tma_load_a, tma_load_b, tma_load_a_fallback, tma_load_b_fallback, cluster_shape_fallback = {0,0,0}, nullptr, nullptr}` | S |
| 8 | `CollectiveEpilogue::to_underlying_arguments(problem_shape, args.epilogue, workspace)` (`sm100_epilogue_tma_warpspecialized.hpp:300-319`): `problem_shape_mnl = (8192,8192,1)`; `get_tma_load_c` (`:270-274`): `tensor_c = make_tensor(ptr_C, (M,N,L):(_1, 8192, 0))`; `make_tma_copy(SM90_TMA_LOAD{}, tensor_c, SmemLayoutStageC, TmaEpilogueTile = Tile<_128,_16>, _1{})` (`copy_traits_sm90_tma.hpp:~1330-1355`) -> `make_tma_copy_tiled` -> `make_tma_copy_atom` with `cta_v_map = (128,16):(E<0>,E<1>)`, `num_multicast = 1`; `construct_tma_gbasis` on `((32,4),(4,4)):((1,512),(32,128))`: `sidx2gmode_full = (32:E<0>, 16:E<1>, 4:32*E<0>)`, truncated at the third mode (basis value 32 is not 1) -> `tma_gbasis = (32,16,1):(E<0>,E<1>,E<2>)`; desc: `globalDim {8192,8192,1}`, strides bytes `{32768, 0}`, box `{32,16,1}`, FLOAT32, `Swizzle<2,5,2>` -> B128 with `SWIZZLE_BASE_32B` -> `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` (needs nvcc > 12.6); **encode #5**; the TiledCopy tiles the 512-element box 4 times along M (`layout_V`, `:1221-1247`) | `tma_load_c` (built even though beta = 0) | S |
| 9 | `get_tma_store_d` (`:278-282`) with `SM90_TMA_STORE`: identical basis/box; **encode #6** | `tma_store_d` | S |
| 10 | `FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace)` -> nested visitor params: `{beta=0.f, beta_ptr=nullptr, dBeta={_0,_0,0}}`, C fetch `{}`, `{alpha=1.f, alpha_ptr=nullptr, dAlpha={_0,_0,0}}`, acc fetch `{}`, compute `{}` | `Params.thread` | S |
| 11 | `TileScheduler::to_underlying_arguments(problem_shape_MNKL, TileShape (256,128,64), AtomThrShape (2,1,1), ClusterShape (2,2,1), hw_info, args.scheduler)` (`sm100_tile_scheduler.hpp:135-161`): `get_tiled_cta_shape_mnl` (`:590-605`): `tiles_m = ceil_div(8192,256) = 32`, `tiles_n = ceil_div(8192,128) = 64`, `ctas_m = round_nearest(32*2, 2) = 64`, `ctas_n = round_nearest(64*1, 2) = 64`, `ctas_l = 1`; `Params::initialize(problem_blocks=(64,64,1), cluster (2,2,1), hw_info, max_swizzle_size = 0, Heuristic)` (`tile_scheduler_params.h:1910-1927`): `problem_tiles_m_ = 32`, `problem_tiles_n_ = 32`, `problem_tiles_l_ = 1`, `divmod_cluster_shape_m_/n_ = FastDivmod(2)`; `initialize_swizzle` (`:1874-1905`): `get_rasterization_order(32, 32, Heuristic)` -> `tiles_n > tiles_m` is false -> **AlongN**; `32*2 = 64 <= 65535` keeps it; `max_swizzle_size <= 1` -> `divmod_swizzle_size_.divisor = 0` | `Params.scheduler` | S |
| 12 | `cudaFuncSetAttribute(device_kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, 230400)` because `230400 >= 48 KiB` (`gemm_universal_adapter.h:338-353`) | every `initialize()` call, i.e. 11 times per process | S (return code R) |
| 13 | Host-call tally per `initialize()`: 6 x (`cudaGetDriverEntryPointByVersion`, `cuTensorMapEncodeTiled`, `cudaDriverGetVersion`) + 1 `cudaFuncSetAttribute` = 19 runtime/driver calls, no device memory touched. Prelude executed once by `run<Gemm>()` (`70_blackwell_fp16_gemm.cu:377-435`) before the first `initialize()`: `Gemm::get_workspace_size(arguments)` = 0 (`gemm_universal_adapter.h:242-254` -> kernel `:338-352` -> epilogue `:321-325` and `TileScheduler::get_workspace_size` -> `Params::get_workspace_size` = 0, `tile_scheduler_params.h:1988-2004`); `cutlass::device_memory::allocation<uint8_t> workspace(0)` -> `cudaMalloc(&p, 0)`; `gemm.can_implement(arguments)` (`:231-239` -> kernel `:300-336`): `kGemm` mode, `check_alignment<8>` for A/B and `<4>` for C/D on the default strides (all extents multiples of 8), fusion and scheduler `can_implement` true -> `kSuccess`. `GemmKernel::to_underlying_arguments` (`:276, 283`) and `initialize_workspace` (`:363, 372`) each re-invoke the two `get_workspace_size` functions (both 0) to compute offsets | prelude; nothing here reaches the device | S (`cudaMalloc(0)` result R) |

### B.1.2 `gemm.run()` (`gemm_universal_adapter.h:609-616` -> `372-575`)

| # | Call (file:line) | Resolved values | Status |
|---|---|---|---|
| 1 | `GemmUniversalAdapter::run(cudaStream_t stream = nullptr, CudaHostAdapter* = nullptr, bool launch_with_pdl = false)` -> `run(params_, stream, cuda_adapter, launch_with_pdl)` | | S |
| 2 | `block = GemmKernel::get_block_shape()` = `dim3(256,1,1)` (`:398-401`); `grid = get_grid_shape(params)` (`:383-396`) -> `TileScheduler::get_grid_shape` (`sm100_tile_scheduler.hpp:220-232`) = `get_tiled_cta_shape_mnl` = `(64,64,1)` then `possibly_transpose_grid` (`:237-250`) for AlongN: `grid.x = (64/2)*2 = 64`, `grid.y = (64/2)*2 = 64` | `grid = (64,64,1)`, 4096 CTAs, 1024 clusters | S |
| 3 | `smem_size = SharedStorageSize = 230400`; `kMinComputeCapability = 100 >= 90` -> extended launch branch; `is_static_1x1x1 = false`; `cluster = dim3(2,2,1)`; `fallback_cluster = {0,0,0}` (static cluster: dynamic override skipped); `kEnableCudaHostAdapter = false`; `kClusterLaunch = (100 == 90) = false` -> the SM100 branch (`:485-521`) | `ClusterLauncher::launch_with_fallback_cluster(grid, cluster, fallback, block, 230400, nullptr, device_kernel<GemmKernel>, {&params}, false)` | S |
| 4 | `make_cluster_launch_config` (`cluster_launch.hpp:150-213`): `have_fallback = false` -> `attr[0] = {cudaLaunchAttributeClusterDimension, {2,2,1}}`; with `CUDA_ENABLE_PREFERRED_CLUSTER` (CUDA >= 12.8) `numAttrs` 3 -> 2; PDL attribute written to slot 1 but `numAttrs = launch_with_pdl ? 2 : 1` = **1** | one attribute | S |
| 5 | `check_cluster_dims(grid, cluster)` (`:103-112`): 4 <= 32, 64 % 2 == 0 | `kSuccess` | S |
| 6 | `init(kernel)` (`:114-148`): `cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` | every launch | S (return R) |
| 7 | `cutlass::arch::synclog_setup()` (no-op without `CUTLASS_ENABLE_SYNCLOG`) then `cudaLaunchKernelExC(&launch_config, device_kernel<GemmKernel>, kernel_params = {&params})` (`cluster_launch.hpp:293-295`) with `gridDim (64,64,1)`, `blockDim (256,1,1)`, `dynamicSmemBytes 230400`, `stream 0`, `numAttrs 1`; `stream 0` is the legacy default stream (Section 0 build); together with B.1.1 step 12 and B.1.2 step 6 each timed iteration performs two `cudaFuncSetAttribute` calls, six encodes and one launch | asynchronous; `run()` then checks `cudaGetLastError()` and returns `kSuccess`/`kErrorInternal`. The timed loop brackets the 10 `initialize()+run()` pairs with `GpuTimer` (`examples/common/helper.h:72-110`): `cudaEventCreate` x2, `cudaEventRecord(start, 0)`, ..., `cudaEventRecord(stop, 0)`, `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy` x2; whether the 19 host calls of each `initialize()` are hidden behind the previous launch is [R] | S (result R) |

The `Params` object (by value, `__grid_constant__`) contains six `Copy_Atom` objects that embed the 128-byte `CUtensorMap`s plus basis metadata, the fusion scalars, the scheduler `Params` (`problem_tiles 32/32/1`, two `FastDivmod(2)`, swizzle divisor 0, `AlongN`), and `hw_info` (all zero).

## B.2 Device entry and kernel prologue (every CTA, every thread unless stated)

| # | Call (file:line) | Lowering | Status |
|---|---|---|---|
| 1 | `device_kernel<GemmKernel>(__grid_constant__ Params const params)` (`device_kernel.h:111-127`): `extern __shared__ char smem[]`; `GemmKernel op; op(params, smem)` | `__launch_bounds__(256,1)`, `__global__ static` | S; `smem` base address R |
| 2 | `operator()` (`sm100_gemm_tma_warpspecialized.hpp:405-415`): `static_assert(230400 <= 232448)`; `problem_shape_MNKL = (8192,8192,8192,1)` | | S |
| 3 | `warp_idx = canonical_warp_idx_sync()` (`cutlass.h:127-133`) = `__shfl_sync(0xffffffff, threadIdx.x/32, 0)`; `warp_category = warp_idx < 4 ? warp_idx : Epilogue` (`:418-420`) | `shfl.sync.idx.b32` | S |
| 4 | `lane_predicate = cute::elect_one_sync()` (`cluster_sm90.hpp:180-197`) | `elect.sync %rx|%px, 0xFFFFFFFF;` (mask in a register) -> predicate true in exactly one lane; which lane is implementation-defined (lane 0 in practice) and may differ between calls | S (lane R) |
| 5 | `cluster_shape = select_cluster_shape((2,2,1))` (`cutlass/detail/cluster.hpp:59-68`) -> static; `cluster_size = 4`; `cta_rank_in_cluster = block_rank_in_cluster()` (`cluster_sm90.hpp:154-163`) | `mov.u32 %r, %cluster_ctarank;` | S |
| 6 | `is_first_cta_in_cluster = (rank == 0)`; `cta_coord_v = rank % 2`; `is_mma_leader_cta = (v == 0)`; `has_mma_peer_cta = true`; `mma_peer_cta_rank = rank ^ 1` (`:426-430`) | | S |
| 7 | `SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf)` (`:433`) | offsets of Section 6.2 | S (base alignment R) |
| 8 | `CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, rank)` (`sm100_mma_warpspecialized.hpp:336-351`): static cluster -> `observed_tma_load_a_ = &params.mainloop.tma_load_a`, `..._b_ = &params.mainloop.tma_load_b` | pointers into parameter space | S |
| 9 | `CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue)` (`sm100_epilogue_tma_warpspecialized.hpp:436-437`): constructs `FusionCallbacks` -> `Sm90ScalarBroadcast` constructors read `beta` and `alpha` from `params.thread` (`update_scalar`, batch stride 0) | every thread holds `alpha = 1.f`, `beta = 0.f` in registers | S |
| 10 | Warp 1 elected lane **of every CTA** (all four ranks, not only the sched participant rank 0; kernel `:439-442`): `collective_mainloop.prefetch_tma_descriptors()` (`:446-450`) -> `cute::prefetch_tma_descriptor(&tma_desc_)` x2 (`copy_sm90_desc.hpp:302-317`); warp 3 elected lane of every CTA, **regardless of beta** (`:443-445`): `collective_epilogue.prefetch_tma_descriptors(params.epilogue)` (`:426-430`) x2; the fallback descriptors are never prefetched | `prefetch.tensormap [%0];` with the generic address of the parameter-space descriptor (A, B in warp 1; C, D in warp 3) | S |
| 11 | `is_epi_load_needed = collective_epilogue.is_producer_load_needed()` (`:447-450`) -> `Sm90TreeVisitor<...>::is_producer_load_needed()` (`sm90_visitor_compute_tma_warpspecialized.hpp:265-277`) -> `(get<2>(dBeta) != 0 && beta_ptr != nullptr) || is_C_load_needed() || false`; `is_C_load_needed = !beta.is_zero() && true` -> **false** for beta = 0 | `is_participant = {mma: warp0, sched: warp1 && rank0, main_load: warp2, epi_load: false, epilogue: warps 4-7}` (`:449-455`) | S (runtime value of beta) |
| 12 | Mainloop pipeline (`:457-472`): `Params{transaction_bytes = 49152, role = Producer (warp 2) / Consumer (warp 0), is_leader = lane_predicate && v==0 && warp2, initializing_warp = 0}`; `PipelineTmaUmmaAsync(storage, params, cluster_shape, true_type, false_type)` (`:624-638`): the base `PipelineTmaAsync` constructor (`sm90_pipeline.hpp:326-345`) runs `canonical_warp_idx_sync()` (`shfl.sync.idx.b32`) and `cute::elect_one_sync()` (`elect.sync`) in every thread although it initializes nothing; then `init_barriers` (`sm100_pipeline.hpp:559-573`): `canonical_warp_idx_sync()` again and, if `warp_idx == 0`, the elected lane (`initialize_barrier_array_pair_aligned`, `barrier.h:131-141`, `if (elect_one_sync())` around the loop) issues `mbarrier.init.shared::cta.b64 [smem+0+8i], 1` and `[smem+64+8i], 2` for i = 0..7; then **all 256 threads** `fence.mbarrier_init.release.cluster;` (the fence sits outside the `warp_idx` test). Steps 13-18 follow the same pattern: every constructor does a `shfl.sync` in all threads (`PipelineTransactionAsync` also an `elect.sync`), the elected lane of the initializing warp issues the `mbarrier.init`s, and all threads fence, so each thread executes 6 `fence.mbarrier_init.release.cluster` in the prologue (mainloop, epi_load, load_order, clc, accumulator, throttle); the `tmem_dealloc` init of step 19 has no fence of its own | | S |
| 13 | Epilogue load pipeline (`:474-487`): `PipelineTransactionAsync<4>` with `producer_arv_count 32`, `consumer_arv_count 128`, `transaction_bytes 8192`, `dst_blockid = rank`, `initializing_warp = 1` -> warp 1 elected lane inits full `[smem+128+8i]` count 32, empty `[smem+160+8i]` count 128; fence | initialized but never used for beta = 0 | S |
| 14 | Store pipeline `PipelineTmaStore<4,1>{always_wait = true}` (`:487-490`) | no barriers | S |
| 15 | Load-order barrier (`:494-499`): `OrderedSequenceBarrier<1,2>{group_id = (warp2 ? 0 : 1), group_size 32, initializing_warp 3}` -> warp 3 elected lane inits `[smem+192]`, `[smem+200]` count 32; group 0 starts with phase 1, group 1 with phase 0 | initialized; no arrive or wait for beta = 0 | S |
| 16 | CLC pipeline (`:501-518`): role `ProducerConsumer` for warp 1 (all CTAs), else `Consumer`; `producer_blockid 0`, `producer_arv_count 1`, `consumer_arv_count = 32 + 4*(32+128+32) = 800`, `transaction_bytes 16`, `initializing_warp 4` -> warp 4 elected lane inits full `[smem+208+8i]` count 1, empty `[smem+224+8i]` count 800; `cluster_size_ = 4` | | S |
| 17 | Accumulator pipeline (`:520-536`): `PipelineUmmaAsync<4,(2,1,1)>` role `Producer` (warp 0) / `Consumer` (warps 4-7); `producer_arv_count 1`, `consumer_arv_count 256`, `initializing_warp 5` -> warp 5 elected lane inits full `[smem+240+8i]` count 1, empty `[smem+272+8i]` count 256 | | S |
| 18 | Throttle pipeline (`:538-552`): `PipelineAsync<2>` role `Producer` (warp 2) / `Consumer` (warp 1); counts 32/32; `dst_blockid 0`; `initializing_warp 3` -> warp 3 elected lane inits `[smem+304+8i]`, `[smem+320+8i]` count 32; producer state `{0,1,0}`, consumer `{0,0,0}` | | S |
| 19 | `TmemAllocator tmem_allocator{}` (`Allocator2Sm`); `NamedBarrier tmem_allocation_result_barrier(160, TmemAllocBarrier=6)`; warp 0 elected lane: `tmem_deallocation_result_barrier.init(32)` -> `mbarrier.init.shared::cta.b64 [smem+336], 32` (`:554-576`) | | S |
| 20 | `pipeline_init_arrive_relaxed(4)` (`:580`) -> `cute::cluster_arrive_relaxed()` | `barrier.cluster.arrive.relaxed.aligned;` | S |
| 21 | `load_inputs = collective_mainloop.load_init(problem_shape_MNKL, shared_storage.tensors.mainloop)` (`:582-583`) | Section B.3.1 steps 1-9 | S |
| 22 | Pipeline states (`:585-598`): producers `make_producer_start_state` = `{index 0, phase 1, count 0}`; consumers `{0,0,0}` | | S |
| 23 | `block_id_in_cluster = cute::block_id_in_cluster()` (`cluster_sm90.hpp:126-137`): `mov.u32 %r, %cluster_ctaid.{x,y,z};`; `mainloop_pipeline.init_masks(cluster_shape, block_id)` (`sm100_pipeline.hpp:598-604`, consumers only) -> `block_id_mask_ = calculate_multicast_mask<kRowCol>` = **0xF**; `accumulator_pipeline.init_masks` (`:151-161`, producers only) -> `tmem_sync_mask_ = calculate_umma_peer_mask` = **0x3** (y = 0) or **0xC** (y = 1) | | S |
| 24 | `TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster)`; `work_tile_info = scheduler.initial_work_tile_info(cluster_shape)`; `cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info)` = `(M_idx, N_idx, _, 0)` (`:607-609`) | Section B.3.4 steps 1-2 | S |
| 25 | `tmem_storage = collective_mainloop.init_tmem_tensors<EpilogueTile,false>(EpilogueTile{})` (`:613`) | Section B.3.2 step 3 (TMEM tensor with base 0) | S |
| 26 | `pipeline_init_wait(4)` (`:615`) -> `cute::cluster_wait()` | `barrier.cluster.wait.aligned;` — no remote barrier operation precedes this | S |
| 27 | Role dispatch `if (is_participant.main_load) ... else if (sched) ... else if (mma) ... else if (epi_load) ... else if (epilogue) ... else {}` (`:617-958`) | warp 1 of ranks 1-3 and (beta = 0) warp 3 of every CTA exit here | S |

## B.3 Role chains

### B.3.1 Mainloop producer: warp 2 (`sm100_gemm_tma_warpspecialized.hpp:617-679`; `sm100_mma_warpspecialized.hpp:497-641`)

`load_init` (executed by all threads in the prologue, step B.2.21):

| # | Call | CuTe result | Status |
|---|---|---|---|
| 1 | `mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L))` (`copy_traits_sm100_tma.hpp:~212-218`) = `make_coord_tensor(make_layout((8192,8192,1), (E<1>,E<0>,E<2>)))` (`tensor_impl.hpp:481-487`) | a tensor whose element `(m,k,l)` **is** the TMA coordinate tuple `(k, m, l)` (an `ArithmeticTupleIterator`, `arithmetic_tuple.hpp:183-206`); `mB_nkl(n,k,l) = (k, n, l)` | S |
| 2 | `gA_mkl = local_tile(mA_mkl, (256,128,64), make_coord(_,_,_), Step<_1,X,_1>)` (`:512`) | shape `((256,64), 32, 128, 1)` = (BLK_M, BLK_K, m-tile, k-tile, l); `gB_nkl` = `((128,64), 64, 128, 1)` | S |
| 3 | `cta_mma = TiledMma{}.get_slice(blockIdx.x % 2)` (`:516`; `mma_atom.hpp:355-364`): `thr_vmnk = (V, 0, 0, 0)`; `tCgA_mkl = cta_mma.partition_A(gA_mkl)` (`mma_atom.hpp:476-484` via `thrfrg_A` `:291-313`): the (256,64) block is divided by the atom (256,16) into `((AtomM,AtomK),(1,4))`, the atom mode is composed with `ALayout_TV = ((2),(128,16)):((128),(1,256))` so `V` selects rows `[128V, 128V+128)`, giving `((128,16),1,4, m,k,l)`; `tCgB_nkl = partition_B` -> `((64,16),1,4, n,k,l)` selecting B columns `[64V, 64V+64)` of the 128-wide block | element `(0,0)` of `tCgA_mkl(_,_,_,m,q,0)` is the coordinate `(64q, 256m + 128V, 0)`; for B `(64q, 128n + 64V, 0)` | S |
| 4 | `sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA)`; `sB` (`:521-522`) | `Sw<3,4,3> o smem+33792 o ((128,16),1,4,8):((64,1),0,16,8192)`; B at `smem+164864`, stage stride 4096 | S |
| 5 | `cta_layout_vmnk = tiled_divide(make_layout((2,2,1)), make_tile(Layout<_2>))` = `((_2),_1,_2,_1):((_1),_0,_2,_4)` (the M-rest mode has extent 1 and stride `_0` because `complement(2:1, 2) = 1:0`; as a function it is `(v,0,n,0) -> v + 2n`); `cta_coord_vmnk = get_flat_coord(rank)` = `(V, 0, y, 0)` (`:525-527`) | | S |
| 6 | `tma_partition(*tma_a, y, make_layout(_2), group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl))` (`:530-532`; `copy_traits_sm90_tma.hpp:1417-1458`): `inv_smem_layout = right_inverse(((128,16),1,4):((64,1),0,16))`; `layout_v` covers the 8192 smem elements; `NumValSrc = 131072 bits / 16 = 8192` (the whole 128x64 CTA tile per TMA "atom"); `layout_V = ((8192, 1))`; `multicast_offset = y * (8192 / 2) = 4096*y` elements; `domain_offset` shifts both tensors by that element offset | `tAgA_mkl` = `((8192), m, k, l)` with element 0 = coordinate `(64q, 256m + 128V + 64y, 0)`; `tAsA` = `((8192), 8)` with data pointer `smem_A + 8192*y bytes + 16384*s` | S |
| 7 | `tma_partition(*tma_b, 0, make_layout(_1), sB grouped, tCgB grouped)` (`:535-537`) -> offset 0 | `tBgB_nkl` element 0 = `(64q, 128n + 64V, 0)`; `tBsB` = `smem_B + 8192*s` | S |
| 8 | `mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk)` (`:540`; `copy_traits_sm90_tma.hpp:1471-1511`): slice the layout at all coords except mode 2 (N) -> ranks `{V, V+2}` | `0x5 << V` (0x5 for x=0, 0xA for x=1); `mcast_mask_b = create_tma_multicast_mask<1>` = `1 << rank` (ignored by the non-multicast op) | S |
| 9 | return `LoadParams{k_tiles = 128, tAgA_mkl, tBgB_nkl, tAsA, tBsB, masks}` | | S |

Role loop (warp 2, all 32 lanes unless noted):

| # | Call | Lowering | Status |
|---|---|---|---|
| 10 | `wait_on_dependent_grids()` (`:620`; `grid_dependency_control.h:95-99`) | `griddepcontrol.wait;` is emitted because `CUTLASS_GDC_ENABLED` is defined: the top-level `CMakeLists.txt:463-475` defaults `CUTLASS_ENABLE_GDC_FOR_SM100` to `ON` and adds `-DCUTLASS_ENABLE_GDC_FOR_SM100=1` to `CUTLASS_CUDA_FLAGS`, and the `sm_100a` target defines `__CUDA_ARCH_FEAT_SM100_ALL` with `__CUDA_ARCH__ == 1000` (`grid_dependency_control.h:50-54`); all 32 lanes execute it | S for the default CMake build (C only if the cache variable was set to OFF; probe D11) |
| 11 | per tile: `k_tile_iter = scheduler.get_k_tile_iterator(...)` (`sm100_tile_scheduler.hpp:482-497`; rank-1 K mode -> plain `make_coord_iterator(k_tiles)`, `stride.hpp:593-596`) = `ForwardCoordIterator<int,int,tuple<E<>>>{coord 0, shape&}`: `*it` is the int `q`, `++it` is `++q` (`detail::increment`, `stride.hpp:489-501`); the member `Shape const& shape` binds to the callee's local `k_tiles` and dangles after return, harmlessly, because the rank-1 increment never reads it and the loops are driven by `k_tile_count`; `k_tile_count = 128`; `k_tile_prologue = min(8, 128) = 8` (`:627-629`) | the iterator lowers to a plain int counter 0..127 | S |
| 12 | rank 0 only: `clc_throttle_pipeline.producer_acquire(state)` -> wait `empty[smem+320+8i]` parity; `producer_commit` -> each lane `mbarrier.arrive.shared::cta.b64 _, [smem+304+8i]` (`:631-637`; `sm90_pipeline.hpp`) | | S |
| 13 | `collective_mainloop.load(pipeline, producer_state, load_inputs, cta_coord_mnkl, k_tile_iter, 8)` (`:640-646`; `sm100_mma_warpspecialized.hpp:589-629`): `tAgA = tAgA_mkl(_, tm/2, _, 0)` (iterator coordinate `(0, 256*(tm/2) + 128V + 64y, 0)` = `(0, 128*tm + 64y, 0)`, because `tm % 2 == x == V` for every tile this CTA ever receives: the initial mapping gives `M_idx = 2CY + x` and CLC tiles give `M_idx = cy_first + x` with `cy_first` even), `tBgB = tBgB_nkl(_, tn, _, 0)` (coordinate `(0, 128*tn + 64V, 0)`); `barrier_token = producer_try_acquire(state)` -> single `mbarrier.try_wait.parity.shared::cta.b64 P, [smem+64+8*idx], phase` | | S |
| 14 | loop body: `producer_acquire(state, token)` (`sm90_pipeline.hpp:531-549`): if token says wait -> loop `mbarrier.try_wait.parity ... 0x989680`; if `is_leader` (V = 0, elected lane) -> `mbarrier.arrive.expect_tx.shared::cta.b64 _, [smem+0+8*idx], 49152` | | S |
| 15 | `tma_barrier = producer_get_barrier(state)` = `&full[idx]`; `write_stage = idx`; `++state`; `token = producer_try_acquire(next)` | | S |
| 16 | `if (elect_one_sync())` -> `copy(tma_a->with(*tma_barrier, mcast_mask_a), tAgA(_, q), tAsA(_, s))` (`:620`): `with()` (`copy_traits_sm100_tma.hpp:~195-200`) yields `Copy_Traits<SM100_TMA_2SM_LOAD_MULTICAST_OP>{opargs = (&tma_desc_, &full[s], mask, 0x1000000000000000)}`; `cute::copy(Copy_Atom, src, dst)` (`copy.hpp:185-235`): rank-1 tensors -> `copy_atom.call(src, dst)` (`copy_atom.hpp:94-114`): `size(src) == NumValSrc` -> `copy_unpack(traits, src, dst)` = `TMA_LOAD_Unpack` (`copy_traits_sm90_tma.hpp:67-92`): `src_coord = src(Int<0>{})` = the tuple `(64q, 256*(tm/2) + 128V + 64y, 0)`, `dst_ptr = raw_pointer_cast(dst.data())` = `smem + 33792 + 16384*s + 8192*y` (linear, pre-swizzle); `explode_tuple(CallCOPY<Op>{}, opargs, dst_ptr, coords)` (`cute/arch/util.hpp:154-215`) -> `SM100_TMA_2SM_LOAD_MULTICAST_3D::copy(desc, mbar, mask, hint, smem_ptr, c0, c1, c2)` (`copy_sm100_tma.hpp:283-305`) | `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [smem+33792+16384s+8192y], [descA, {64q, 256*(tm/2)+128V+64y, 0}], [(smem+8s) & 0xFEFFFFFF], 0x5<<V, 0x1000000000000000;` | S |
| 17 | `copy(tma_b->with(*tma_barrier, mcast_mask_b), tBgB(_, q), tBsB(_, s))` (`:621`) -> `SM100_TMA_2SM_LOAD_3D::copy` (`copy_sm100_tma.hpp:104-128`) | `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [smem+164864+8192s], [descB, {64q, 128tn+64V, 0}], [(smem+8s) & 0xFEFFFFFF], 0x1000000000000000;` | S |
| 18 | `--k_tile_count; ++k_tile_iter`; after 8 tiles return `(state, iter)`; the kernel then (beta = 0) skips `load_order_barrier.arrive()` and calls `load(...)` again for the remaining 120 k-tiles (`:649-661`) | | S |
| 19 | `__syncwarp()`; `fetch_next_work` (Section B.3.4 step 5); `cta_coord_mnkl = work_tile_to_cta_coord(...)`; loop while valid (`:664-676`) | | S |
| 20 | `load_tail` -> `producer_tail` (`sm90_pipeline.hpp:445-454`): 8 x `mbarrier.try_wait.parity` loops on `empty[smem+64+8*idx]` | | S |

For 8192-cubed with the cluster at launch `(CX, CY)` and CTA `(x, y)`: `tm = 2CY + x`, `tn = 2CX + y`, so A rows start at `256CY + 128x + 64y`, B columns at `256CX + 128y + 64x`, for `q = 0..127`; each CTA issues exactly 2 TMA instructions per k-tile, 256 per tile. On odd ranks the 32-bit `shared::cta` address carries the CTA-rank bit 24 (Section B.6), so `(smem+8s) & 0xFEFFFFFF` names rank `r-1`'s `full[s]`; the leader's `full[s]` is therefore credited by six 8192-byte boxes per stage: the two A boxes landing in the leader (issued by the two CTAs with the leader's `x`, ranks `x` and `x+2`), the two A boxes landing in the peer (issued by ranks `x^1` and `(x^1)+2`, redirected by the mask), and the B boxes of the leader and of the peer, totalling the 49152 bytes of `expect_tx`.

### B.3.2 MMA role: warp 0 (`sm100_gemm_tma_warpspecialized.hpp:726-805`; `sm100_mma_warpspecialized.hpp:550-579, 651-712`)

| # | Call | CuTe result / lowering | Status |
|---|---|---|---|
| 1 | `tmem_allocator.allocate(512, &shared_storage.tmem_base_ptr)` (`tmem_allocator_sm100.hpp:134-145`) | `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [smem+384], 512;` (whole warp, both CTAs of the pair) | S; returned base R |
| 2 | `__syncwarp(); tmem_allocation_result_barrier.arrive()` -> `bar.arrive 6, 160;`; `tmem_base_ptr = shared_storage.tmem_base_ptr`; `set_tmem_offsets(tmem_storage, base)` (`:728-732`) | | S |
| 3 | (prologue) `init_tmem_tensors` (`sm100_mma_warpspecialized.hpp:468-480`): `acc_shape = partition_shape_C(TiledMma, (256,128))` (`mma_atom.hpp:563-570`) = `((128,128),1,1)`; `make_sm100_accumulator<4,false>` (`sm100_tmem_helper.hpp:58-75`) = `make_fragment_C(((128,128),1,1,4))` -> `make_tensor<tmem_frg_2sm<float>>(shape)` -> `tmem_frg::make` (`mma_traits_sm100_frag.hpp:101-207`, `N_SM == 2, M_MMA == 128` branch): atom `(128,128):(1,128)` tiled by `(1,1,4)` then restrided by `(128,16384):(DP<float> = 65536, 1)` | `((128,128),1,1,4):((65536,1),0,0,128)` over `tmem_ptr<float>`; element `(m,n)` of stage `st` at `base + (m << 16) + 128*st + n`; `slice_accumulator(tmem_storage, st) = accumulators(_,_,_,st)` (`:464-466`) | S |
| 4 | `mma_init` (`:552-579`): `sA/sB` as in B.3.1; `tCrA = TiledMma::make_fragment_A(sA)` (`mma_atom.hpp:148-170`) -> `make_tensor<UMMA::smem_desc<K>>(sA)` -> `MakeTensor<smem_desc<K>>` (`mma_sm100_desc.hpp:899-912`): `DescriptorIterator{make_umma_desc<K>(tensor<0>(sA))}` with layout `replace<0>(recast<uint128_t>(sA).layout(), Layout<_1,_0>)` | `make_umma_desc<K>` (`:737-833`) on the `(128,16)` half tile recast to `uint128_t` -> canonical `((8,16),(2,1)):((8,64),(1,0))` in 16-byte units: `start_address_ = uint16_t((smem+33792) >> 4)` stored in a **14-bit** field (`mma_sm100_desc.hpp:107, 757-758`), `LBO = 1`, `SBO = 64`, `version 1`, `layout_type 2`, `base_offset 0` -> `0x4000404000010000 | (((smem+33792) >> 4) & 0x3FFF)`; the truncation drops the CTA-rank bit 24 of the `shared::cta` address, so leader and peer hold bit-identical descriptors (as `tcgen05.mma.cta_group::2` requires: it reads the peer's smem at the same CTA-local offset), and the field spans 256 KiB, enough for `smem + 230400` as long as the dynamic base is below 31744 [R, probe D5]; iterator layout A `(1,1,4,8):(0,0,2,1024)`, B `(1,1,4,8):(0,0,2,512)` with `(smem+164864) >> 4` | S |
| 5 | `tiled_mma` copy with `idesc_ = make_instr_desc<half_t,half_t,float,256,128,K,K>()` = **0x10200010** (`mma_sm100_desc.hpp:482-506`); `accumulate_` default `One` | | S |
| 6 | per tile (both CTAs): `k_tile_count = 128`; `fetch_next_work` (B.3.4 step 5) before computing; `acc_stage = acc_producer_state.index()` (`:739-760`) | | S |
| 7 | leader only: `mma(...)` (`:762-770`): `tiled_mma.accumulate_ = ScaleOut::Zero` (`:676`); `accumulator_pipeline.producer_acquire(acc_state)` (`sm100_pipeline.hpp:207-209` -> `PipelineAsync::producer_acquire`) -> `mbarrier.try_wait.parity ... [smem+272+8*acc_stage], phase` loop | | S |
| 8 | per k-tile: `consumer_try_wait/consumer_wait(state)` on `full[smem+0+8*s]` (`:671, 684`); `read_stage = s`; `++state`; `for k_block in 0..3: cute::gemm(tiled_mma, tCrA(_,_,k_block,s), tCrB(_,_,k_block,s), accumulators)` (`:700-707`) | `cute::gemm` 3-arg -> `gemm(mma, C, A, B, C)`; D/C = `((128,128),1,1)` TMEM (counts as `is_rmem` because CuTe classifies non-gmem/non-smem as rmem, `pointer.hpp:226-230`), A/B = `(1,1)` descriptor tensors -> Dispatch [4] `(V,M)x(V,N)` (`gemm.hpp:263-300`): `size<0>(A)*sizeof(SmemDescriptor) = 8` -> 64-bit serpentine branch with M = N = 1 -> one `gemm(mma, D(_,0,0), A(_,0), B(_,0), C(_,0,0))` whose arguments are prvalue slices, so overload resolution first takes the "accept mutable temporaries" forwarder `gemm(MMA_Atom const&, Tensor&& D, A const&, B const&, C const&)` (`gemm.hpp:135-149`) and then Dispatch [1] (`:178-197`) -> `mma.call(D,A,B,C)` (`mma_atom.hpp:94-104`) -> `mma_unpack` (`mma_traits_sm100.hpp:1195-1217`): `desc_a = A[0]` = base descriptor `+ 2*k_block + 1024*s`, `desc_b = B[0]` = `+ 2*k_block + 512*s` (`DescriptorIterator::operator+`, `mma_sm100_desc.hpp:861-874`: for CUDA 13.3 the test `(MAJOR > 13) || (MAJOR == 13 && MINOR > 3)` is false, so the `#else` branch adds the offset to the low 32-bit word only; no carry can leave the 14-bit field here: the largest sum is `((smem+164864) >> 4) + 6 + 3584 = 13894 + (smem >> 4) < 16384` for `smem < 0x9BA0`, a weaker condition than the whole-buffer bound of step 4), `tmem_c = base + 128*acc_stage`, `idesc = make_runtime_instr_desc(idesc_) = 0x10200010 << 32` -> `fma(desc_a, desc_b, tmem_c, uint32_t(accumulate_), idesc)` (`mma_sm100_umma.hpp:562-586`) | `if (elect.sync) { setp.ne.b32 p, %scale_c, 0; tcgen05.mma.cta_group::2.kind::f16 [tmem_c], desc_a, desc_b, 0x10200010, {0,0,0,0,0,0,0,0}, p; }` — `scale_c = 0` only for `q = 0, k_block = 0` of each tile | S |
| 9 | `tiled_mma.accumulate_ = One` after each gemm (`:706`); `consumer_release(saved_state)` (`:708`; `sm100_pipeline.hpp:725-758`) -> `umma_arrive_multicast_2x1SM(&empty[s], 0xF)` | `if (elect.sync) tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem+64+8s], 0xF;` | S |
| 10 | after 128 k-tiles `mma()` returns; `accumulator_pipeline.producer_commit(acc_state)` (`:771`; `sm100_pipeline.hpp:212-214, 259-269`) | `if (elect.sync) tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem+240+8*acc_stage], (y==0 ? 0x3 : 0xC);` | S |
| 11 | both CTAs: `++acc_producer_state`; next tile (`:773-776`) | | S |
| 12 | after the last tile: `launch_dependent_grids()` -> `griddepcontrol.launch_dependents;` (all 32 lanes of warp 0 in every CTA; emitted for the default build, see B.3.1 step 10); `release_allocation_lock()` -> `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;` (`:781-784`) | | S |
| 13 | leader: `accumulator_pipeline.producer_tail(state)` -> 4 x wait on `empty[smem+272+8i]`; handshake (`:792-797`): peer: `arrive(leader, pred=true)` -> `mapa.shared::cluster.u32 r, smem+336, leader_rank; mbarrier.arrive.shared::cluster.b64 _, [r];` (32 lanes); both: `wait(phase 0)` on own `[smem+336]`; leader: `arrive(peer, true)` | | S |
| 14 | `tmem_allocator.free(base, 512)` (`tmem_allocator_sm100.hpp:157-169`) | `tcgen05.dealloc.cta_group::2.sync.aligned.b32 base, 512;` | S |

Per pair per tile: 512 `tcgen05.mma` (1 with `scale_c = 0`, 511 accumulating), 128 mainloop commits, 1 accumulator commit, 128 non-blocking `mbarrier.try_wait.parity` peeks on `full[s]` (1 before the loop + 127 in-loop; the peek after the 128th k-tile is elided by `skip_wait`) plus a blocking wait loop only when a peek failed, 1 blocking wait on the accumulator `empty`, 641 `elect.sync` (one per `fma`/`commit`); warp 0 of both CTAs additionally performs one CLC consume per tile (B.3.4 steps 5b-5c). The follower's warp 0 executes everything except `mma()` and `producer_commit`, which keeps its `acc_producer_state` in lockstep with the leader.

### B.3.3 Epilogue consumers: warps 4-7 (`sm100_gemm_tma_warpspecialized.hpp:869-955`; `sm100_epilogue_tma_warpspecialized.hpp:573-956, 1269-1291`)

| # | Call | CuTe result / lowering | Status |
|---|---|---|---|
| 1 | `tmem_allocation_result_barrier.arrive_and_wait()` -> `bar.sync 6, 160;`; read `tmem_base_ptr`; `set_tmem_offsets` (`:871-873`) | | S |
| 2 | per tile: `fetch_next_work` (B.3.4 step 5); `acc_stage = acc_consumer_state.index()`; `accumulator = get<0>(slice_accumulator(tmem_storage, acc_stage))`; `scheduler.fixup<false>(TiledMma{}, work_tile_info, accumulator, accumulator_pipeline, state, CopyOpT2R{})` returns the state unchanged (`sm100_tile_scheduler.hpp:546-566`; this scheduler never splits K) and `scheduler.compute_epilogue(work_tile_info)` returns `true` (`:524-528`) (`:878-911`) | | S |
| 3 | `store(...)` setup (`:600-605`): `thread_idx = threadIdx.x % 128`, `warp_idx = thread_idx / 32` (= kernel warp - 4), `lane_idx` | | S |
| 4 | `mD_mn = tma_store_d.get_tma_tensor((M,N,L))` = coord tensor with `g_stride_ = (E<0>,E<1>,E<2>)`; `mD = coalesce(mD_mn, (128,128))`; `gD = local_tile(mD, (128,128), (tm, tn, 0))` (`:613-615`) | `gD(m',n')` = TMA coordinate `(128tm + m', 128tn + n', 0)` | S |
| 5 | `tAcc = accumulators(make_coord(_,_),0,0)` = `(128,128):(65536,1)` at `base + 128*acc_stage`; `tAcc_epi = flat_divide(tAcc, EpilogueTile)` = `(128,16,1,8)`; `gD_epi = flat_divide(gD, EpilogueTile)` (`:617-621`) | subtile `e` = columns `[16e, 16e+16)` | S |
| 6 | `sC_epi/sD_epi = as_position_independent_swizzle_tensor(make_tensor(smem_C/D, SmemLayoutC/D))` (`:624-629`; `pointer_flagged.hpp:119-140`): the `Swizzle<2,5,2>` (byte units) is recast to `Swizzle<2,3,2>` in float units (`recast_layout<uint8_t,float>`, `swizzle_layout.hpp:442-455`), moved from the pointer into the layout and applied to the *float offset* from the 512-byte-aligned buffer base (`smem+512`): `o ^ ((o & 0x60) >> 2)` | layout `((32,4),(4,4),4):((1,512),(32,128),2048)` floats composed with `Swizzle<2,3,2>`; the TMA unit applies the same swizzle to absolute addresses, which is why `smem+512` must be 512-byte aligned (`alignas(512)` union; asserted only in debug builds) | S |
| 7 | `tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b16x{}, tAcc_epi(_,_,0,0))` (`copy_traits_sm100.hpp:284-303`): `atom_t_layout = (32,4):(0, 32*65536)`, `atom_v_layout = upcast<32>(ValID)` = 16 columns; `make_cotiled_copy` -> thread `t` of the 128 owns TMEM lane `32*(t/32) + t%32`, i.e. row `t`, and 16 consecutive columns | `thread_t2r.partition_S(tAcc_epi)` = `((16),1,1,1,8)`; the per-thread view is only bookkeeping: the load is warp-collective and its single address operand is warp-uniform, `base + ((32w) << 16) + 128*acc_stage + 16e`; `tTR_sD = partition_D(sD_epi(_,_,0))` = 16 smem elements per thread | S |
| 8 | `tiled_s2r = make_tiled_copy_D(Copy_Atom<AutoVectorizing...<128>, float>, tiled_t2r)`; `tiled_r2r`, `tiled_r2s` likewise (`:651-685`); `IsDirectR2S = true`, `IsDirectS2R = true` | each thread's 16 register values map one-to-one to the 16 smem elements `(m = 32w + l, n = 16e + j)` | S |
| 9 | `thrblk_s2g = tma_store_d.get_slice(0)`; `bSG_sD = partition_S(sD_epi)` = `(512, 4, 1, 4 stages)`; `bSG_gD = partition_D(gD_epi)` = `(512, 4, 1, 1, 8)` (`:688-690`) | four 32x16 boxes per subtile | S |
| 10 | residue tensors (`:694-702`) computed but unused by this fusion; `synchronize = NamedBarrier::sync(128, EpilogueBarrier)`; `issue_tma_store = (warp_idx == 0)` (`:723, 740`); `load_wait_state = store_state with phase ^ 1` (`:747-751`) | | S |
| 11 | `cst_callbacks = fusion_callbacks.get_consumer_store_callbacks<false>(cst_args)` (`sm90_visitor_compute_tma_warpspecialized.hpp:328-341`): `is_C_load_needed = false` -> `cute::clear(args.tCrC)` (C fragment zeroed once per tile) | | S |
| 12 | `epi_loop_fn`: `acc_wait_token = acc_pipeline.consumer_try_wait(acc_state)` (`:813`) -> one `mbarrier.try_wait.parity [smem+240+8*acc_stage], phase` | | S |
| 13 | for `e = 0..7` (`iter_n` outer, `iter_m` = 0; `#pragma unroll`): `cst_callbacks.begin_loop` and, on every subtile, `cst_callbacks.previsit(0, e, load_wait_state.count(), false)` (`:852`, unconditional, a no-op for this fusion); first subtile only: `acc_pipeline.consumer_wait(acc_state, token)` (`:864-867`) -> loop `mbarrier.try_wait.parity ... 0x989680` if the peek of step 12 returned WaitAgain; the producer-load blocks `:837-849` and `:854-862` are skipped (`is_producer_load_needed = false`) | | S |
| 14 | `copy(tiled_t2r, tTR_tAcc(_,_,_,0,e), tTR_rAcc)` (`:881-883`) -> `SM100::TMEM::LOAD::copy_unpack` (`copy_traits_sm100.hpp:380-410`) -> `explode(SM100_TMEM_LOAD_32dp32b16x::copy, &tmem_addr, rD[0..15])` (`copy_sm100.hpp:3576-3604`) | `tcgen05.ld.sync.aligned.32x32b.x16.b32 {r0..r15}, [base + ((32w)<<16) + 128*acc_stage + 16e];` lane `l` receives columns `16e..16e+15` of row `32w + l` | S |
| 15 | `e == 7` only: `fence_view_async_tmem_load()` -> `tcgen05.wait::ld.sync.aligned;` then `acc_pipeline.consumer_release(acc_state)` (`sm100_pipeline.hpp:241-249, 271-277`) -> `umma_arrive_2x1SM_sm0(&empty[acc_stage])` (`barrier.h:905-921`); `++acc_consumer_state` (`:886-890`) | `mbarrier.arrive.shared::cluster.b64 _, [(smem+272+8*acc_stage) & 0xFEFFFFFF];` from each of the 128 threads (both CTAs -> 256 on the leader; relies on the bit-24 rank encoding of B.6) | S |
| 16 | `tTR_rD_frg(0) = cst_callbacks.visit(tTR_rAcc_frg(0), 0, 0, e)` (`:894-896`) -> specialized `Sm90TreeVisitor::visit` (`sm90_visitor_compute_tma_warpspecialized.hpp:295-326`): `frg_added = multiplies<Array<float,16>>(alpha_frag, acc)` -> `t[j] = alpha*acc[j]`; then `multiply_add<Array<float,16>>(beta_frag, C_frag(=0), t)` -> `d[j] = beta*C[j] + t[j]` (`functional.h:585-590`, `array.h:963-975`) | 16 FMUL + 16 (FMUL+FADD or FFMA `[C]`) per thread | S/C |
| 17 | `tRS_rD_frg(0) = NumericArrayConverter<float,float,16>{}(...)` identity (`:913`); `copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,store_idx))` (`:921-923`) -> `copy(AutoVectorizingCopyWithAssumedAlignment<128>, ...)` (`copy.hpp:245-262`): `max_common_vector = 1` -> `copy_if` -> 16 scalar `float` assignments through the swizzled smem pointer | thread `(w,l)`, value `j` (column `n = 16e + j`) writes to byte `smem + 512 + 8192*store_idx + 2048*w + 128*n + 4*((l & 7) + 8*(((l >> 3) & 3) ^ (j & 3)))` (derivation: pre-swizzle word offset `(m%32) + 32n + 512(m/32)` with `m = 32w + l`, then byte bits [5,7) ^= bits [7,9), and `(128n) >> 7 = n` so the XOR term is `j & 3`) | S (SASS width C) |
| 18 | `tma_store_fn(0, e)` (`:766-800`): `fence_view_async_shared()` -> `fence.proxy.async.shared::cta;` (all 128); `synchronize()` -> `bar.sync 1, 128;`; if `warp_idx == 0`: `copy(params.tma_store_d, bSG_sD(_,_,_,store_idx), bSG_gD(_,_,_,0,e))` -> `copy(Copy_Atom, src (512,4), dst (512,4))` rank 2 -> loop over the 4 M-boxes -> `Copy_Traits<SM90_TMA_STORE>::copy_unpack` (`copy_traits_sm90_tma.hpp:404-420`): `dst_coord = dst(Int<0>{})` = `(128tm + 32i, 128tn + 16e, 0)`, `src_ptr = smem + 512 + 8192*store_idx + 2048*i` -> `explode_tuple(CallCOPY<SM90_TMA_STORE>, (desc, src), coords)` -> `SM90_TMA_STORE_3D::copy` (`copy_sm90_tma.hpp:1003-1023`) | 4 x `cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [descD, {128tm+32i, 128tn+16e, 0}], [smem+512+8192*store_idx+2048i];` executed by all 32 lanes of warp 4 | S (lane coalescing C) |
| 19 | `store_pipeline.producer_commit(state)` -> `cp.async.bulk.commit_group;` (warp 4 lanes); `++store_state`; `producer_acquire(state)` -> `always_wait` -> `cp.async.bulk.wait_group.read 1;` (warp 4); `synchronize()` -> `bar.sync 1, 128;`; ReuseSmemC bookkeeping: `store_finished = count > 1` -> `++load_pipe_consumer_state` (no release for beta = 0) (`:777-799`) | | S |
| 20 | `cst_callbacks.end_loop`, loop; after `e = 7`: `cst_callbacks.end()`; return states (`:933-955`) | | S |
| 21 | after the last tile: `store_tail` (`:1269-1291`): `ReuseSmemC && is_producer_load_needed()` is false for beta = 0 -> nothing executes, in particular no `cp.async.bulk.wait_group.read 0`; the epilogue warps therefore exit with at most one bulk async-group (the last subtile's four stores) possibly still reading smem, bounded by the `wait_group.read 1` of step 19 — a property the replacement must preserve or tighten, never loosen (probe D15) | | S |

### B.3.4 Scheduler: warp 1 of rank 0 as producer, all roles as consumers (`sm100_gemm_tma_warpspecialized.hpp:681-724`; `sm100_tile_scheduler.hpp`)

| # | Call | Lowering / result | Status |
|---|---|---|---|
| 1 | (prologue, all threads) `initial_work_tile_info(cluster_shape)` (`:369-371`) -> `swizzle_and_rasterize(blockIdx.x, blockIdx.y, blockIdx.z, true, 0, 0)` (`:697-814`): `divmod_swizzle_size_.divisor == 0` -> skip; `divmod_cluster_shape_m_/n_(x)` = `FastDivmod{divisor 2, multiplier 0x80000000, shift_right 0}` (`fast_math.h:368-376`) -> `q = __umulhi(x, 0x80000000) >> 0 = x >> 1`, `r = x - 2q = x & 1` (`:337, 343`; the `params_` fields live in the `__grid_constant__` parameter space and how nvcc materializes them is [C]); `possibly_transpose_work_tile(AlongN, x, y)` (`:666-677`): `M_idx = (y/2)*2 + x%2`, `N_idx = (x/2)*2 + y%2`; offsets 0 | tile `(tm, tn) = (2CY + x%2, 2CX + y%2)`, `L_idx = 0`, valid | S |
| 2 | `work_tile_to_cta_coord` = `(M_idx, N_idx, _, L_idx)` (`:373-377`) | | S |
| 3 | sched warp: `wait_on_dependent_grids()` -> `griddepcontrol.wait;` (`:688`; emitted for the default build, B.3.1 step 10) | | S |
| 4 | loop: `clc_throttle_pipeline.consumer_wait(state)` -> `mbarrier.try_wait.parity [smem+304+8i], phase` loop; `consumer_release(state)` -> `mapa.shared::cluster.u32 r, smem+320+8i, 0; mbarrier.arrive.shared::cluster.b64 _, [r];` (32 lanes); `++state` (`:693-695`) | | S |
| 5a | `clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, state)` (`:698`; `:438-451`): for the n-th query of the cluster (n = 0, 1, ...) `idx = n % 2`, the producer waits parity `1 ^ ((n >> 1) & 1)` and the consumers of 5b wait parity `(n >> 1) & 1`; `mbarrier_addr = producer_get_barrier(state)` = `smem+208+8*idx`; `producer_acquire(state)` (`sm100_pipeline.hpp:1095-1104`): wait `empty[smem+224+8*idx]` parity, then every lane executes `setp.eq.u32 p, (lane < 4), 1; @p mapa.shared::cluster.u32 r, smem+208+8*idx, lane; @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [r], 16;` (lanes 0..3 arm CTAs 0..3); `if (elect_one_sync()) issue_clc_query(state, mbarrier_addr, clc_response_ptr)` (`:392-407`) | `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [smem+352+16*idx], [smem+208+8*idx];` then `++state` | S (response R) |
| 5b | consumers (every participating warp of every CTA, once per tile): `fetch_next_work(work_tile_info, clc_pipeline, state)` (`:457-473`): `consumer_wait(state)` -> `mbarrier.try_wait.parity [smem+208+8*idx], phase` loop (own CTA's full barrier); `work_tile_info_from_clc_response(smem+352+16*idx)` (`:409-436`; for a "not cancelled" response `valid = 0` and `M_idx/N_idx/L_idx` are left undefined, so only `is_valid()` may be read) | `ld.shared.b128 r, [addr]; clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, r; selp.u32 valid, 1, 0, p1; @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x, y, z, _}, r;` then `fence.proxy.async.shared::cta;` | S (values R) |
| 5c | `consumer_release(state)` (`sm100_pipeline.hpp:1132-1135`) -> `mapa.shared::cluster.u32 r, smem+224+8*idx, 0; mbarrier.arrive.shared::cluster.b64 _, [r];` (every consuming thread, 800 total); `swizzle_and_rasterize(x, y, z, valid, cluster_ctaid.x, cluster_ctaid.y)` -> `(M_idx, N_idx) = ((y/2)*2 + x%2 + ctaid.x, (x/2)*2 + y%2 + ctaid.y)` with `x, y` the returned first-CTA id (even) | returns `(work_tile, true)` -> `++clc_pipe_consumer_state` | S |
| 6 | sched loop continues while valid; then `clc_pipeline.producer_tail(state)` -> 2 x (`test_wait` then `wait`) on `empty[smem+224+8i]` (`:722`; `sm100_pipeline.hpp:1041-1050`) | | S |

Per cluster processing `T` tiles: `T` queries are issued (`T-1` cancelled clusters, one "not cancelled"). Summed over the launch: exactly 1024 queries (one per cluster tile), of which `N_native` fail, where `N_native` is the number of clusters the hardware launched natively; it is bounded by the number of co-resident clusters, at most `148 / 4 = 37` on a 148-SM B200 with one 230400-byte CTA per SM (the exact value is `[R]`, `cudaOccupancyMaxActiveClusters`). Which first-CTA ids come back, in what order, and how many tiles each resident cluster gets are `[R]`.

### B.3.5 Epilogue C producer: warp 3 (beta != 0 only)

Not exercised by the default run. Chain: `griddepcontrol.wait`; per tile `fetch_next_work`; first tile `load_order_barrier.wait()` (`mbarrier.try_wait.parity [smem+200], 0` loop; warp 2 arrived after its first prologue); `load(...)` (`sm100_epilogue_tma_warpspecialized.hpp:461-550`): per subtile `producer_acquire` (wait `empty[smem+160+8*idx]`), elected lane: `copy(tma_load_c.with(*full, 0), bGS_gC(_,_,_,0,e), bGS_sC(_,_,_,idx))` -> 4 x `cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [smem+512+8192idx+2048i], [descC, {128tm+32i, 128tn+16e, 0}], [smem+128+8*idx], 0x1000000000000000;` then `mbarrier.expect_tx.shared::cta.b64 [smem+128+8*idx], 8192;`; all lanes `mapa ... rank; mbarrier.arrive.shared::cluster.b64` on the same full barrier; `++state`; `load_tail` = 4 empty waits. In `store()` the consumer then waits `full[smem+128+8*idx]`, reads C with 16 scalar loads from the same swizzled addresses as step B.3.3.17, and releases `empty[smem+160+8*idx]` one subtile later (128 arrivals); `store_tail` adds `cp.async.bulk.wait_group.read 0;` and one final release.

## B.4 Where the CuTe algebra turns into integers (summary of the coordinate/offset formulas)

| Quantity | Formula for this run | Source of the formula |
|---|---|---|
| CTA tile from launch coords `(x_l, y_l)` | `tm = 2*(y_l/2) + x_l%2`, `tn = 2*(x_l/2) + y_l%2` | `swizzle_and_rasterize` + `possibly_transpose_work_tile` (B.3.4.1) |
| CTA tile from a CLC first-CTA id `(x0, y0)` | `tm = 2*(y0/2) + x0%2 + ctaid.x`, `tn = 2*(x0/2) + y0%2 + ctaid.y` | B.3.4.5c |
| A TMA coordinate, k-tile `q` | `(64q, 256*(tm/2) + 128*(tm%2) + 64*ctaid.y, 0)` = `(64q, 128*tm + 64*ctaid.y, 0)` (since `tm % 2 == ctaid.x`); box 64x64 | B.3.1.3 + multicast `domain_offset` (B.3.1.6) |
| A smem destination | `smem + 33792 + 16384*s + 8192*ctaid.y` | `tAsA(_, s)` |
| B TMA coordinate | `(64q, 128*tn + 64*(tm%2), 0)`; box 64x64 | B.3.1.3 |
| B smem destination | `smem + 164864 + 8192*s` | `tBsB(_, s)` |
| Mainloop barrier operands | full `smem + 8s` (masked with 0xFEFFFFFF in TMA), empty `smem + 64 + 8s` | Section 6.2 |
| A/B multicast masks | `0x5 << (tm%2)` for A; B not multicast | `create_tma_multicast_mask<2>` |
| UMMA A descriptor | `0x4000404000010000 | (((smem + 33792 + 16384s + 32*kb) >> 4) & 0x3FFF)` (14-bit start-address field) | B.3.2.4, B.3.2.8 |
| UMMA B descriptor | `0x4000404000010000 | (((smem + 164864 + 8192s + 32*kb) >> 4) & 0x3FFF)` | same |
| Instruction descriptor | `0x10200010` | `make_instr_desc` |
| Accumulator TMEM address (MMA D operand) | `tmem_base + 128*acc_stage` | B.3.2.3 |
| Epilogue TMEM load address (warp `w`, subtile `e`) | `tmem_base + ((32w) << 16) + 128*acc_stage + 16e` | B.3.3.7, B.3.3.14 |
| Epilogue smem D byte address (thread `(w,l)`, value `j`) | `smem + 512 + 8192*idx + 2048w + 128*(16e+j) + 4*((l&7) + 8*(((l>>3)&3) ^ (j&3)))` | B.3.3.17 |
| D TMA store coordinates / smem source | `(128tm + 32i, 128tn + 16e, 0)`, `smem + 512 + 8192*idx + 2048i`, `i = 0..3` | B.3.3.18 |
| CLC response slot / barriers | response `smem + 352 + 16*idx`, full `smem + 208 + 8*idx`, empty `smem + 224 + 8*idx` (CTA 0) | B.3.4.5 |

## B.5 Complete PTX inventory on the executed path (beta = 0)

Prologue (all CTAs): `shfl.sync`, `elect.sync`, `mov %cluster_ctarank`, `mov %cluster_ctaid.*`, `prefetch.tensormap` (4), `mbarrier.init.shared::cta.b64` (8+8, 4+4, 2, 2+2, 4+4, 2+2, 1 barriers), `fence.mbarrier_init.release.cluster` (6 times, all threads: one per pipeline constructor), `shfl.sync.idx.b32` (once in the kernel and once or twice per pipeline constructor), extra `elect.sync` in the `PipelineTmaAsync` and `PipelineTransactionAsync` constructors, `bar.warp.sync` (`__syncwarp`) in warp 0 (warp 2 executes one per tile), `barrier.cluster.arrive.relaxed.aligned`, `barrier.cluster.wait.aligned`. Ancillary instructions inside the wrappers: `mov.u32 %r, %cluster_ctarank` is executed four times per thread (kernel `:426` plus the default initializers of three pipeline `Params` structs, `sm90_pipeline.hpp:791, 1036`), every peek ends in `selp.b32`, every blocking wait is a `@P bra DONE; bra LAB` loop, every `elect.sync` is followed by `@%px mov.s32 pred, 1; mov.s32 lane, %rx`, and the CLC `arrive.expect_tx` carries a `setp.eq.u32` lane predicate.
Warp 2: `griddepcontrol.wait`; per k-tile `mbarrier.try_wait.parity` (empty), leader `mbarrier.arrive.expect_tx.shared::cta.b64` 49152, two `cp.async.bulk.tensor.3d.cta_group::2...` (A multicast, B); rank 0 per tile: `mbarrier.arrive.shared::cta.b64` (throttle full); per tile one CLC consume (`mbarrier.try_wait.parity`, `ld.shared.b128`, `clusterlaunchcontrol.query_cancel.*`, `fence.proxy.async.shared::cta`, `mapa` + `mbarrier.arrive.shared::cluster.b64`); tail 8 waits.
Warp 0: `tcgen05.alloc.cta_group::2`, `bar.arrive 6,160`; leader per k-tile `mbarrier.try_wait.parity` (full), 4 x `tcgen05.mma.cta_group::2.kind::f16`, `tcgen05.commit...multicast::cluster` 0xF; per tile `mbarrier.try_wait.parity` (acc empty), `tcgen05.commit...multicast::cluster` 0x3/0xC, one CLC consume; end: `griddepcontrol.launch_dependents`, `tcgen05.relinquish_alloc_permit.cta_group::2`, 4 waits (leader), `mapa` + `mbarrier.arrive.shared::cluster` handshake, `mbarrier.try_wait.parity`, `tcgen05.dealloc.cta_group::2`.
Warp 1 (rank 0): `griddepcontrol.wait`; per query: throttle wait + `mapa`/`mbarrier.arrive.shared::cluster`, CLC empty wait, 4 x `mapa`/`mbarrier.arrive.expect_tx.shared::cluster.b64` 16, `clusterlaunchcontrol.try_cancel...multicast::cluster::all.b128`, one CLC consume; tail 2 waits.
Warps 4-7: `bar.sync 6,160`; per tile one CLC consume, `mbarrier.try_wait.parity` (acc full), 8 x `tcgen05.ld.sync.aligned.32x32b.x16.b32`, 1 x `tcgen05.wait::ld.sync.aligned`, 1 x `mbarrier.arrive.shared::cluster.b64` (masked), 8 x (16 FP32 multiplies + 16 multiply-adds + 16 scalar `st.shared`), 8 x `fence.proxy.async.shared::cta`, 16 x `bar.sync 1,128`, warp 4 only: 32 x `cp.async.bulk.tensor.3d.global.shared::cta.bulk_group`, 8 x `cp.async.bulk.commit_group`, 8 x `cp.async.bulk.wait_group.read 1`.
Not on the path: `cp.async.{ca,cg}`, `wgmma`, `tcgen05.fence`, `tcgen05.st`, `cp.async.bulk.wait_group.read 0` (beta = 0), any `stmatrix`/`ldmatrix`.

## B.6 Hardware semantics the source relies on (not derivable from CUTLASS; PTX ISA is authoritative)

The kernel's correctness depends on these instruction semantics. They were cross-checked against secondary write-ups (Colfax Research Blackwell tutorials, the MLC "Modern GPU Programming" TMEM chapter, the CUDA Programming Guide CLC page); the PTX ISA 9.3 sections to consult are given by number since the single-page ISA could not be fetched here.

- `cp.async.bulk.tensor` with `.cta_group::2` (PTX ISA 9.7.9.26.5.2): the mbarrier operand may designate the executing CTA's or its peer CTA's barrier; CUTLASS forms the peer (leader) address by clearing bit 24 of the `shared::cta` address (`Sm100MmaPeerBitMask`), and secondary sources confirm "a CTA can find the address of its leader CTA's mbarrier by taking its own mbarrier address and setting the 24th bit to 0". With `.multicast::cluster`, the data lands at the same CTA-relative smem offset in every CTA in the mask and the transaction bytes are credited to the destination pair's leader barrier. The 49152-byte expectation only works if all six 8192-byte boxes that land in a pair credit that pair's leader.
- `tcgen05.mma.cta_group::2` (9.7.17.10.9.1): issued by one thread of the even (leader) CTA; A rows `[0,128)` and B columns `[0,64)` of the 256x128 instruction come from the leader's smem and the remaining halves from the peer's smem at the same CTA-relative addresses; each CTA's TMEM receives its 128 rows of D. Both CTAs must therefore have identical smem layouts (they do: same `SharedStorage`).
- `tcgen05.commit` (9.7.17.12.1): arrives on the named mbarrier(s) when all previously issued asynchronous `tcgen05` operations of the executing thread complete; `.multicast::cluster` with a 16-bit mask arrives on the barrier at the same offset in every masked CTA. The mainloop uses mask 0xF, the accumulator commit 0x3/0xC.
- `tcgen05.ld` / `tcgen05.wait::ld` (9.7.17.8.3, 9.7.17.8.5): `tcgen05.ld` is asynchronous and the ISA requires `tcgen05.wait::ld` before the destination registers are read; a warp may only address TMEM lanes `[32*(warp_id%4), +32)`. CUTLASS's epilogue issues one `tcgen05.wait::ld` per tile (before the accumulator release) and consumes the registers of subtiles 0..6 without an explicit wait, so it relies on ptxas/hardware register dependency tracking `[C]`; the explicit kernel should mirror the baseline and the SASS should be inspected (Section B.7).
- `tcgen05.alloc/dealloc/relinquish_alloc_permit` with `.cta_group::2` (9.7.17.7.1): executed by one full warp in each CTA of the pair (same warp id), columns a power of two >= 32, result written to smem; relinquish only forbids further allocation; dealloc must follow all TMEM reads.
- `clusterlaunchcontrol.try_cancel ... .multicast::cluster::all` (9.7.14.18-19): cancels one not-yet-launched cluster; the 16-byte response is written to the given smem offset in every CTA of the requesting cluster and each CTA's mbarrier at the given offset receives 16 transaction bytes; `get_first_ctaid` returns the launch-grid id of the cancelled cluster's first CTA (cluster-aligned). A failed request means no unstarted clusters remain; issuing another request after observing failure is undefined (CUTLASS stops).
- `mbarrier.*.shared::cluster` with `mapa`: remote arrivals on peer CTAs' barriers; a CTA must not exit while peers may still arrive on its barriers (the tails in Section 7.10).
- `shared::cta` addresses inside a cluster: the 32-bit result of `cvta.to.shared` (`cast_smem_ptr_to_uint`) is also a valid `shared::cluster` address of the executing CTA, and CUTLASS relies on the CTA rank appearing in bits [24,28) of that address (`Sm100MmaPeerBitMask = 0xFEFFFFFF` clears the rank-parity bit to reach the pair leader). Consequently the dynamic smem base `smem` is **not** identical across the four CTAs: with the rank in bits [24,28) the expectation is `smem_r = smem_0 | (r << 24)`, so within a pair `smem_odd = smem_even | 0x01000000`, the low 24 bits are the same in every CTA, all offsets in this document are relative to each CTA's own base, and the 14-bit descriptor field and `mapa` are what keep the CTA-local operands rank-independent. Whether the low 24 bits of the base are 0 or 0x400 (the 1 KiB of driver-reserved shared memory per CTA on sm_90+ may precede the dynamic buffer) is `[R]` (probe D5).
- `barrier.cluster.arrive.relaxed.aligned` / `barrier.cluster.wait.aligned`: cluster-wide rendezvous making the `mbarrier.init`s (fenced by `fence.mbarrier_init.release.cluster`) visible before any remote use.
- `elect.sync`: exactly one lane of a converged warp gets the predicate; the elected lane may differ between calls (CUTLASS re-elects per TMA/MMA issue).

## B.7 What cannot be determined statically, and the experiment for each

Everything below is `[C]` (compiler/build) or `[R]` (driver/hardware/runtime). The experiments run in the trace build of the example itself (line-1 toggle on, C.1; never in the performance binary) on the B200 with CUDA 13.3. Part C holds the implementation: the probe names in this table (H*, K*, B*, R*) refer to C.4.

| # | Item | Why not static | Experiment / debug print |
|---|---|---|---|
| D1 | Exact resolved types (TiledMma, SmemLayoutA/B, SmemLayoutC/D, EpilogueTile, DispatchPolicies, TMA atom types) as the compiler sees them | source-derived above, but confirmation is cheap and removes doubt about the CuTe algebra | Host: `cute::print(typename CollectiveMainloop::TiledMma{})`, `cute::print(typename CollectiveMainloop::SmemLayoutA{})`, `SmemLayoutB`, `print(typename CollectiveEpilogue::SmemLayoutC{})` (exposed via a friend/derived helper or `decltype`), `print(GemmKernel::EpilogueTile{})`, `printf("%s", __PRETTY_FUNCTION__)` in a template helper instantiated with `CollectiveMainloop`, `CollectiveEpilogue`, `GemmKernel::TileScheduler` |
| D2 | `sizeof`/`offsetof` of `GemmKernel::SharedStorage` and members (expected 230400 and the Section 6.2 offsets); `sizeof(CollectiveEpilogue::SharedStorage)` (33792); `sizeof(Params)` | ABI | Host `static_assert`/`printf` of `sizeof(...)`, `offsetof(SharedStorage, pipelines.clc)`, `offsetof(SharedStorage, tensors.mainloop.smem_A)` etc. (members are public in these structs) |
| D3 | The six encoder tuples and the raw 128-byte tensor maps | driver encoding | H2b (guarded block in `copy_traits_sm90_tma.hpp`): the argument tuple and `CUresult` of the first six encodes; H2: the 16 `uint64_t` words of each map and `driver_version` (the `<= 13010` fixup cannot trigger for these tensor sizes) |
| D4 | `cudaFuncSetAttribute`/`cudaLaunchKernelExC` return codes; actual launch config | runtime | H3: `cudaGetLastError()` after the warm-up launch and the kernel's function attributes (`CUTLASS_CHECK` already aborts on a non-success `Status`); H4: grid and block |
| D5 | Dynamic smem base address per rank and its alignment; actual `mbarrier`, `smem_A/B`, `smem_C/D` addresses; validity of the bit-24 rank encoding behind `0xFEFFFFFF` | runtime / hardware address window | K0 (standalone probe kernel: base, `mapa` to the peer and to rank 0, masked base, per CTA) and K1 (the real kernel: base and the addresses of every barrier and tensor buffer, thread 0 of each of the four ranks of cluster (0,0)); expect rank 0's base in {0x0, 0x400}, identical low 24 bits in every rank, the CTA rank in bits [24,28) so that within a pair `base_odd == base_even | 0x01000000` and `base_odd & 0xFEFFFFFF == base_even`, `mapa(base_r, t) == base_t`, `smem_A % 1024 == 0`, `smem_D % 512 == 0`, and `(base & 0xFFFFFF) + 230400 < 262144` so the 14-bit descriptor field cannot overflow |
| D6 | TMEM base address returned by `tcgen05.alloc` in each CTA of a pair | hardware allocator | K0 (probe kernel, both CTAs of each pair) and K2 (the real kernel, MMA warp and epilogue of every CTA); expect equal values within a pair, 0 |
| D7 | Values of `desc_a`, `desc_b`, `tmem_c`, `idesc`, `scale_c` at the first `tcgen05.mma` instructions of a tile | derived above; confirm | K3 (guarded block in `fma`, cluster (0,0), about 16 records per leader) checked against B.4 by `check_trace.py` |
| D8 | TMA coordinates, smem destinations and barrier operands of the first A and B loads per rank; D store coordinates and sources; the number of lanes issuing each store | derived above; confirm | K4a and K4b (the former `#if 0` print blocks of `copy_traits_sm90_tma.hpp`, cluster (0,0)) checked against B.4; a passing run also confirms the hardware rule that each multicast completion credits the destination pair's leader barrier (the 49152-byte `expect_tx` would never complete otherwise) |
| D9 | CLC behavior: which first-CTA ids each resident cluster receives, in what order, how many tiles per cluster, response latency | hardware scheduling | K5a (every query issued), K5b (every response consumed by rank 0's scheduler warp), K5c (every response consumed by every CTA's MMA warp), all with `%smid` and `%globaltimer`; `check_trace.py` verifies exactly-once coverage of the 1024 cluster tiles, 1024 queries with `N_native` failures (`N_native <= 37`), the slot/parity schedule and the mapping of B.3.4, and reports tiles per cluster and latency |
| D10 | (dropped 2026-09-08: only the fixed command line of Section 0 is run) Whether `--swizzle` values 1/2/5 and the uneven 4096x16384 case take the expected `AlongM`/swizzle paths | host arithmetic; confirm | Host: print `params.scheduler.raster_order_`, `divmod_swizzle_size_.divisor`, `problem_tiles_m_/n_`, and the grid returned by `get_grid_shape` |
| D11 | SASS lowering: one `UTMASTG`-class instruction per warp-wide store or 32; FFMA vs FMUL+FADD in the epilogue; `st.shared` widths; confirmation that the descriptor increment is the 32-bit low-word add selected for CUDA 13.3; register count, spills, stack; presence of `griddepcontrol.*` (expected present for the default CMake build); how the `tcgen05.ld` destination registers are protected before their first use; how the `__grid_constant__` scheduler parameters are materialized | ptxas | `cuobjdump --dump-resource-usage` and `--dump-sass` on the toggle-off binary of the Section 0 build (Part C, B1-B3): locate `device_kernel<...GemmUniversal...>`, record the tensor-core/TMA/barrier/CLC mnemonics and counts, `FFMA`/`FMUL`/`FADD` and `STS` counts in the epilogue, the `griddepcontrol` lowering, and the ELF arch `sm_100a` |
| D12 | Correctness/perf baseline on this machine | measurement | Run the unmodified Release binary with the fixed Section 0 command (warm-up + 10 iterations) several times, record `Disposition`, `Avg runtime`, `GFLOPS`; capture `nvidia-smi -q` clocks and `nvcc --version` (other shapes and `--beta` values are out of scope since 2026-09-08) |
| D13 | Driver acceptance of the zero L-stride and of `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` | driver | Covered by D3's `result` print (`CUDA_SUCCESS` expected) |
| D14 | (values fixed by the Section 0 cmake command; confirm only) Build-configuration macros that change host behavior without changing results: `NDEBUG` (host `assert`s in `make_tma_copy_desc`, `Return_Status` verbosity), `CUDA_API_PER_THREAD_DEFAULT_STREAM` (`--default-stream per-thread`), `CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL` (direct `cuTensorMapEncodeTiled` instead of `cudaGetDriverEntryPointByVersion`), `CUTLASS_ENABLE_GDC_FOR_SM100`, `CUTLASS_DEBUG_TRACE_LEVEL`, `CUTLASS_ENABLE_SYNCLOG` | CMake cache / compile flags | Host: inspect `build/CMakeCache.txt` and the example target's `flags.make`, or `printf` each macro under `#ifdef` in the trace build; the replacement must be compiled with the same flags for a fair timing comparison |
| D15 | The epilogue exits with at most one bulk async-group not waited on (beta = 0); completion of the last TMA stores is guaranteed only by kernel completion | PTX/hardware semantics of CTA exit with pending `cp.async.bulk` groups | `Disposition: Passed` on every R1 run of the fixed command (primary evidence: the pending group is not observable in the compared output); optionally R3 (`compute-sanitizer`, outside the fixed command). No variant with an added `tma_store_wait<0>` is built; the replacement mirrors the baseline's exit condition |

Shape of the trace build: see Part C. Under the fixed build and run commands of Section 0 there is no scratch target and no extra flag; trace mode is the one-line toggle `// #define CUTLASS_DEABSTRACTION_TRACE 1` at the top of `70_blackwell_fp16_gemm.cu`, with guarded probe blocks in four `include/` headers, address and operand records limited to cluster (0,0) and capped per kind, recording switched off before the timed launches, and no timing ever taken from the toggle-on binary.

## B.8 Cross-check of this trace against Part A

Every PTX string in B.3 and B.5 matches Section 7's tables; the smem offsets are Section 6.2; the C/D four-box store, the 800/928 CLC count, the mask values, the descriptor constants and the accumulate pattern are as corrected in Section 13. No new contradictions with Part A were found while building the chains; the additions here are (i) the explicit host chain of the six encodes with their intermediate CuTe bases, (ii) the derivation that the TMA "coordinate tensors" are `ArithmeticTuple` iterators whose element *is* the coordinate tuple, (iii) the `cute::gemm` dispatch path (Dispatch [4] with the 8-byte descriptor branch, then Dispatch [1], then `mma_unpack`), (iv) the exact swizzled smem address formula for the epilogue stores, and (v) the experiment list in B.7.

Reconciliation with the independent verification of this trace (five tracer agents re-deriving the host, mainloop, MMA, epilogue and scheduler chains from source, five skeptics checking them): no contradiction with Part A or with the chains above was found. The skeptics' corrections that touched this document were folded in: the exact type of `cluster_layout_vmnk` (stride `_0` M-rest mode; B.1.1 step 3, B.3.1 step 5), the 14-bit truncation of the UMMA start address (B.3.2 step 4, B.4, Section 7.6), the CUDA 13.3 selection of the low-word descriptor add (B.3.2 step 8), six rather than seven `fence.mbarrier_init` per thread (B.5), `griddepcontrol.*` being emitted by the default build (B.3.1 step 10, B.3.2 step 12, B.3.4 step 3), the unconditional no-op `previsit` (B.3.3 step 13), the scheduler `fixup`/`compute_epilogue` no-ops (B.3.3 step 2), the per-CTA descriptor prefetch by warps 1 and 3 (B.2 step 10), the extra `shfl.sync`/`elect.sync` in the pipeline constructors (B.2 step 12), the driver-version gate of the descriptor fixup and the host prelude calls (B.1.1 steps 5 and 13), the CLC slot/parity schedule and the 1024-query total (B.3.4), the pending-bulk-group exit condition (B.3.3 step 21, D15), and the expectation that the dynamic smem base differs between even and odd ranks (B.6, D5). The remaining disagreements were line-number slips inside the tracers' own reports, not in this document.

Revision for the fixed use case (Section 0), checked by three further independent source reviews (PTX-string exactness, completeness for writing the kernel, scope and consistency): every quoted PTX string, constant, count, offset and formula was re-confirmed against the source; the reviews found no missing fact needed to write the kernel. Corrections applied: the Section 7.2 primitive list conflated `mbarrier.try_wait` (every peek) with `mbarrier.test_wait` (CLC `producer_tail` only) and lacked the lane predicate of the remote `arrive.expect_tx`; the descriptor-increment bound in B.3.2 step 8 is `13894`/`0x9BA0`, not `13958`/`0x9800`; each timed iteration performs two `cudaFuncSetAttribute` calls, not one (Sections 3, 12); the cache hint is `CacheHintSm100::EVICT_NORMAL`; anchor slips (`to_underlying_arguments :257`, adapter `:486-521`, `make_cluster_launch_config :151-207`, box truncation `:996-1002`, alignment assert `:955`, store pipeline `:487-490`, seeds `:323-325`, store `copy_unpack` `:414-420`); C.5's CLC record count (1024 + 4096, every query consumed once per warp); the `CUTLASS_SWIZZLE_DEVICE_DEBUG_PRINT` hint (the header forces it to 0). Additions: per-pipeline stage/parity formulas, the complete `SmemDescriptor` and `InstrDescriptor` bit maps, the plain-form A/B smem byte formula, the loop-unrolling shape, the 16-bit mask operands, and the ancillary PTX inside the wrappers (B.5). Simplifications: Section 7 is now the protocol specification with pointers into Part B instead of a second copy of the PTX; Sections 3, 10, 11 and 13 and the beta != 0, swizzle and residue material were condensed to what the fixed run exercises.

---

# Part C. Execution plan for the dynamic items: debug-print code and experiments (implemented and reviewed 2026-09-08; not yet run)

This part turns Section B.7 (D1-D15) and Section 11 into a plan that is executed on the B200 by following C.5 in order, under the fixed build and run commands of Section 0. **Status: implemented in the working tree** (files in C.2; every probe block is guarded by `CUTLASS_DEABSTRACTION_TRACE`), the run script `examples/70_blackwell_gemm/deabstraction_trace_run.sh` performs C.5, and the checker `examples/70_blackwell_gemm/check_trace.py` performs C.6. The implementation was reviewed by six independent source-reading passes with adversarial verification of every finding (49 agents); the one build-breaking defect they found (a wrong `static_assert` on the record size) and every plan/implementation mismatch were fixed, and this part now describes the code as it stands. **Nothing has been run yet.** Every "expected" value is the prediction of Parts A and B and is what the run must confirm or refute. Line numbers of CUTLASS files refer to commit `cb424739` before the probe blocks were inserted.

## C.1 Rules the plan follows

1. **One target, one command line.** Only `70_blackwell_fp16_gemm` is built, with the cmake command of Section 0, and only `--m=8192 --n=8192 --k=8192` is run. There is no second CMake target and no extra `-D` flag. Timings (D12) and SASS (D11) come from the binary built with trace mode **off**.
2. **Trace mode is a one-line source toggle.** Line 1 of `70_blackwell_fp16_gemm.cu` is `// #define CUTLASS_DEABSTRACTION_TRACE 1`; uncommenting it and rebuilding with the same command produces the trace binary at the same path. With the toggle off every probe is removed by the preprocessor, so the performance binary compiles exactly the pristine code.
3. **Probes inside CUTLASS headers are in-place, guarded blocks.** Four headers under `include/` receive blocks wrapped in `#if defined(CUTLASS_DEABSTRACTION_TRACE) && defined(__CUDA_ARCH__) ... #endif` (the nine device-side blocks K1, K2 x2, K3, K4a, K4b, K5a, the `trace_raw_tile` capture and K5b/K5c) or `#if defined(CUTLASS_DEABSTRACTION_TRACE) ... #endif` (the host-side H2b block). `git diff include/` shows nothing else except the removal of the two dead `#if 0 ... #endif` print blocks that K4a/K4b replace. C.5 step 5 (`restore`) proves that the toggle-off binary equals the pristine one (`cmp`, with a SASS diff as fallback if the toolchain output is not bit-reproducible). If a pristine `include/` is preferred, the same blocks can live in copies under an override directory named first in `target_include_directories(70_blackwell_fp16_gemm BEFORE PRIVATE ...)`; the cmake command stays the same, but the include order then has to be verified.
4. **Records, not text, on the device.** Device probes append fixed-size records to a global ring buffer (C.3). The host dumps it to CSV after the warm-up launch. Device `printf` is limited to the standalone probe kernels (16 + 1 lines).
5. **Gates and caps.** Address and operand probes fire only in the launch-space cluster `(0,0)` (`blockIdx.x < 2 && blockIdx.y < 2`); every kind has a cap, and once a kind reaches its cap a per-kind flag makes later events skip the atomics. CLC probes record every query of every cluster. Recording is switched off before the 10 timed launches, whose output in trace mode is ignored anyway.
6. **Every record has a prediction.** `check_trace.py` (C.6) prints one PASS/FAIL line per item against the B.4 formulas and Section 6.2 offsets, and the document section that made the prediction.
7. **Code paths unchanged.** Probes never add `__syncwarp`, named or cluster barriers, `elect.sync` or warp shuffles, never move statements, and sit inside existing `elect_one_sync()` regions where the code already has one (K3, K5a); warp identity in K5b/K5c is `threadIdx.x / 32`, not the shuffle-based `canonical_warp_idx_sync()`. The probe macros expand to nothing in the host compilation pass.

## C.2 Files (implemented)

```
examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu        edited: line 1 toggle, guarded includes of the two trace headers, guarded calls in run<Gemm>()
examples/70_blackwell_gemm/deabstraction_trace.hpp          new (included before any CUTLASS header): TraceRec, TraceKind, device ring buffer, TRACE_RECORD/TRACE_COUNT/TRACE_IN_FIRST_CLUSTER, trace_reset/trace_dump/trace_disable
examples/70_blackwell_gemm/deabstraction_trace_probes.hpp   new (included after the CUTLASS headers): probe kernels K0 and GDC, trace_probe_kernel_k0(), trace_host_probes<Gemm>() (H1, H2, H4, H5), trace_host_post_run<Gemm>() (H3)
examples/70_blackwell_gemm/check_trace.py                   new: post-processing (C.6)
examples/70_blackwell_gemm/deabstraction_trace_run.sh       new: the run matrix of C.5 (build, inspect, baseline, trace, restore, check)
include/cute/atom/copy_traits_sm90_tma.hpp                  H2b (host, after the driver-version fixup at :1070), K4a (TMA_LOAD_Unpack::copy_unpack, replacing the `#if 0` print at :80-86), K4b (Copy_Traits<SM90_TMA_STORE>::copy_unpack, replacing the `#if 0` print at :414-420)
include/cute/arch/mma_sm100_umma.hpp                        K3 (SM100_MMA_F16BF16_2x1SM_SS::fma, five-argument overload, after `uint32_t mask[8]` at :571 inside the existing elect_one_sync region)
include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp   K1 (after `pipeline_init_wait` at :615), K2 (after the `tmem_base_ptr` reads at :731 and :872)
include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp        K5a (advance_to_next_work, before `issue_clc_query` at :446), trace_raw_tile capture (after :464), K5b/K5c (after swizzle_and_rasterize, :467-469)
```

Edits to the example (`70_blackwell_fp16_gemm.cu`):

- Line 1 is the toggle `// #define CUTLASS_DEABSTRACTION_TRACE 1`, followed by the guarded `#include "deabstraction_trace.hpp"`, **before** `#include "cutlass/cutlass.h"`, so the record buffer and the `TRACE_*` macros are declared before any CUTLASS header that uses them. The guarded `#include "deabstraction_trace_probes.hpp"` follows `#include "helper.h"`. Both quoted includes resolve against the example's own directory.
- In `run<Gemm>()`, all inside `#if defined(CUTLASS_DEABSTRACTION_TRACE)`: after `initialize(options)` call `trace_reset()` then `trace_probe_kernel_k0()` (the K0 records are the first entries of the buffer); after the first `gemm.initialize(...)` call `trace_host_probes<Gemm>(gemm)`; after `verify()` call `trace_host_post_run<Gemm>()`, `trace_dump("launch0.csv")` and `trace_disable()`. The timed loop is untouched; its printed timing is meaningless in trace mode.
- Output conventions consumed by `check_trace.py`: records in `launch0.csv` (header `kind,bx,by,rank,warp,lane,smid,seq,t,v0..v9`, `#` comment lines with the per-kind counters); stdout lines `TRACE_HOST <key> <values>` (sizes, offsets, attributes, macros, counters, scheduler params, grid), `TRACE_ENCODE <n> key=value...` (H2b; enumerators printed as `NAME(number)`), `TRACE_TMAP <name> <16 hex words>`, `TRACE_K0 ...` (probe kernel `printf`), `TRACE_DEV ...` (device-side GDC flag). Lines the checker does not parse and that are compared by hand when Part D is written: `TRACE_H1 <name> <cute::print output>` (layouts), `TRACE_TYPE <label> | <__PRETTY_FUNCTION__>` (resolved types), `TRACE_H2 <name> <cute::print of a TMA atom>`. `TRACE_ERROR ...` on stderr aborts the run. The run script merges stderr into `host.txt`.

## C.3 Record infrastructure (`deabstraction_trace.hpp`)

```cpp
struct TraceRec {                 // 128 bytes
  uint32_t kind, bx, by, rank;     // launch block, cluster rank (%cluster_ctarank)
  uint32_t warp, lane, smid, seq;  // recording thread, %smid, per-kind sequence number
  uint64_t t;                      // %globaltimer (ns)
  uint64_t v[10];                  // payload, meaning per kind (table at the top of the header and C.4)
  uint64_t pad_;                   // tail padding, not written to the CSV
};
enum TraceKind : uint32_t { K_SMEM = 1, K_TMEM, K_MMA, K_TMA_LOAD, K_TMA_STORE, K_TMA_STORE_LANES /*reserved*/,
                            K_CLC_ISSUE, K_CLC_SCHED, K_CLC_MMA, K_PROBE0, K_SMEM2 = 11, K_COUNT = 16 };
constexpr uint32_t kTraceCapacity = 1u << 16;                  // 65536 records = 8 MiB
__device__ TraceRec g_trace_buf[kTraceCapacity];
__device__ uint32_t g_trace_next, g_trace_enabled;             // append index; 1 between trace_reset() and trace_disable()
__device__ uint32_t g_trace_seq[K_COUNT], g_trace_full[K_COUNT];   // per-kind counters; per-kind "cap reached" flags
#define TRACE_RECORD(kind, cap, ...)   /* device pass only; empty in the host pass */                       \
  do { if (::g_trace_enabled && !::g_trace_full[(kind)]) {                                                  \
         uint32_t s_ = atomicAdd(&::g_trace_seq[(kind)], 1u);                                               \
         if (s_ < (cap)) { uint32_t i_ = atomicAdd(&::g_trace_next, 1u);                                    \
                           if (i_ < ::kTraceCapacity) ::g_trace_buf[i_] = ::trace_make_rec((kind), s_, __VA_ARGS__); } \
         else { ::g_trace_full[(kind)] = 1u; } } } while (0)
```

`trace_make_rec` fills `bx, by` from `blockIdx`, `rank` from `%cluster_ctarank`, `warp/lane` from `threadIdx.x`, `smid` from `mov.u32 %0, %%smid;` (as `cutlass/arch/arch.h:61`), `t` from `mov.u64 %0, %%globaltimer;` (as `cutlass/arch/synclog.hpp:82`), and the payload from up to ten `uint64_t` arguments. `TRACE_IN_FIRST_CLUSTER()` is `blockIdx.x < 2 && blockIdx.y < 2` (false in the host pass); `TRACE_COUNT` (a record-less counter increment) exists but is unused. The `g_trace_full` flag is a plain load: a stale 0 costs one more returning atomic, a stale 1 cannot occur, so the steady-state cost of a saturated probe is one L1-cached load. `trace_reset()` synchronizes the device, zeroes the counters, flags and index, and sets `g_trace_enabled = 1` (`cudaMemcpyToSymbol`); `trace_disable()` clears the flag; `trace_dump(path)` synchronizes, copies the `g_trace_next` records with `cudaMemcpyFromSymbol`, writes the CSV (with `# seq[k]=n` comment lines) and prints `TRACE_HOST trace_seq_<k> <n>` and `TRACE_HOST trace_records <n>`. A host-side `inline int g_trace_encode_count` counts the tensor-map encodes for H2b. The buffer holds 65536 records; K0 plus the warm-up launch produce about 6900 (C.5).

## C.4 Probe catalogue

### C.4.1 Host probes (`trace_host_probes<Gemm>` after the first `gemm.initialize()`, `trace_host_post_run<Gemm>` after the warm-up `run()` and `verify()`)

| Probe | Covers | What is printed | Prediction |
|---|---|---|---|
| H1 types and sizes | D1, D2 | `TRACE_H1 <name>` + `cute::print` (helper `trace_print_layout`) of `GemmKernel::TiledMma{}`, `CollectiveMainloop::SmemLayoutA{}`/`SmemLayoutB{}`, `CollectiveEpilogue::SmemLayoutAtomC{}`/`SmemLayoutAtomD{}` (the composed `SmemLayoutC/D` are `private`, `sm100_epilogue_tma_warpspecialized.hpp:132, 162-163`; they follow from the atom, `EpilogueTile` and the stage count), `GemmKernel::EpilogueTile{}`, `TileShape{}`, `CtaShape_MNK{}`, `AtomThrShapeMNK{}`, `ClusterShape{}`; `TRACE_TYPE <label> | __PRETTY_FUNCTION__` (helper `trace_print_type<T>`) for `CollectiveMainloop`, `CollectiveEpilogue`, `TileScheduler`, both `DispatchPolicy`s, `CopyOpT2R/G2S/S2G/S2R/R2S`, `GmemTiledCopyA/B`, the six pipeline types; `TRACE_HOST sizeof_*` for `SharedStorage` (and its `PipelineStorage`, `TensorStorage`), `CollectiveEpilogue::SharedStorage`, `Params`, `MainloopParams`, `EpilogueParams`, `TileSchedulerParams`, plus `SharedStorageSize`, `MaxThreadsPerBlock`, `MinBlocksPerMultiprocessor`, `NumEpilogueSubTiles`, `CLCResponseSize`, the two stage counts, `IsOverlappingAccum`, `IsGdcEnabled` (host view, always false); `TRACE_HOST off_<member>` = absolute byte offsets of every `SharedStorage` member, computed as address differences inside a 1024-byte-aligned host buffer reinterpreted as `SharedStorage` (no member is read) | Section 4 types (compared by hand); 230400; epilogue storage 33792; offsets exactly as in Section 6.2 (`off_smem_A 33792`, `off_smem_B 164864`, `off_smem_C = off_smem_D = 512`, ...); `Params` below 32764 bytes |
| H2 tensor maps | D3, D13 | `TRACE_TMAP <A|B|A_fallback|B_fallback|C|D>` = the 16 `uint64_t` words of each `tma_desc_` (`p = gemm.params()`; `tma_desc_` is a public member); `TRACE_H2 <name>` + `cute::print` of the four used atoms; `TRACE_HOST driver_version`, `runtime_version` | fallback maps byte-identical to the primaries (checked); A vs B differ only in the base address (by hand); driver version above 13010 |
| H2b encoder tuples | D3, D13 | guarded host block in `copy_traits_sm90_tma.hpp` right after the driver-version fixup: `TRACE_ENCODE <n> format=NAME(v) dim=3 gmem=<ptr> shape=a,b,c stride_bytes=s1,s2 box=x,y,z elem_stride=1,1,1 interleave=NAME(v) swizzle=NAME(v) l2promo=NAME(v) oobfill=NAME(v) result=<CUresult> driver=<v>`; the process-wide `g_trace_encode_count` limits it to the first six encodes (66 happen per process) | encodes 0-3 (A, B, A-fallback, B-fallback): `FLOAT16, 3, 8192,8192,1, 16384,0, 64,64,1, 1,1,1, INTERLEAVE_NONE, SWIZZLE_128B, L2_128B, OOB_FILL_NONE, result=0`; encodes 4-5 (C, D): `FLOAT32, ..., 32768,0, 32,16,1, ..., SWIZZLE_128B_ATOM_32B, ...` (Section 5.3, B.1.1 steps 5-9) |
| H3 launch (`trace_host_post_run`) | D4 | `cudaGetLastError()` after the warm-up GEMM, reference GEMM and compare; `cudaFuncGetAttributes(cutlass::device_kernel<GemmKernel>)` -> `maxDynamicSharedSizeBytes, sharedSizeBytes, constSizeBytes, localSizeBytes, numRegs, maxThreadsPerBlock, ptxVersion, binaryVersion, nonPortableClusterSizeAllowed, clusterDimMustBeSet, requiredCluster*`; `cudaOccupancyMaxActiveClusters` with the real launch config (grid 64x64, block 256, smem 230400, cluster (2,2,1)); device name, SM count, reserved smem per block, opt-in smem, cluster-launch support, compute capability. The `Status` values of `can_implement`/`initialize`/`run` are not printed: `CUTLASS_CHECK` already aborts on anything but `kSuccess` | `cudaSuccess`; `maxDynamicSharedSizeBytes = 230400`, `sharedSizeBytes = 0` (no static smem), `nonPortableClusterSizeAllowed = 1`; max active clusters `<= 37` (the `N_native` bound of C.6); 148 SMs, 1024 reserved bytes, 232448 opt-in, capability 10.0. `numRegs` and `localSizeBytes` of the trace build are informational only (D11 measures the pristine binary) |
| H4 scheduler params | (formerly D10) | `TRACE_HOST scheduler_problem_tiles m n l`, `scheduler_cluster_divisors`, `scheduler_swizzle_divisor`, `scheduler_raster_order AlongN|AlongM <int>`, `scheduler_log_swizzle_size`, `grid x y z` (`Gemm::get_grid_shape(params)`), `block x y z` | `32 32 1`, `2 2`, `0`, `AlongN`, `0`, `64 64 1`, `256 1 1` (Section 7.4, B.1.1 step 11); the other `--swizzle` runs of the original D10 stay dropped |
| H5 build macros | D14 | `TRACE_HOST macro_<NAME> 0|1` for `NDEBUG`, `CUDA_API_PER_THREAD_DEFAULT_STREAM`, `CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL`, `CUTLASS_ENABLE_GDC_FOR_SM100`, `CUTLASS_ENABLE_SYNCLOG`, `CUTLASS_ENABLE_CUDA_HOST_ADAPTER`, `CUTLASS_DEBUG_TRACE_LEVEL`; `nvcc_version major minor build`; the one-thread `trace_probe_gdc_kernel` prints `TRACE_DEV IsGdcGloballyEnabled <0|1> FEAT_SM100_ALL <0|1> CUDA_ARCH <n>` | Section 0: `NDEBUG 1`, per-thread stream 0, direct driver call 0, GDC 1, synclog 0, host adapter 0, trace level 0; nvcc 13.3; device `1 1 1000` |

### C.4.2 Standalone probe kernel K0 (`trace_probe_addr_kernel(int)`, launched by `trace_probe_kernel_k0()` before the CUTLASS kernel)

Covers D5 and D6 independently of CUTLASS's kernel and measures the hardware facts of Section B.6 directly.

- Launch: `cudaLaunchKernelEx` with grid `(4,4,1)` (four clusters), block 256, dynamic smem 230400, one attribute `cudaLaunchAttributeClusterDimension = (2,2,1)`, after `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 230400)` and `NonPortableClusterSizeAllowed = 1`, as the real launch does (B.1.2 steps 6-7). The kernel is `__global__ void __launch_bounds__(256,1) trace_probe_addr_kernel(int)` (the unused parameter avoids a zero-length argument array in the runtime's launch template) with `extern __shared__ char trace_probe_smem[];` and no static shared variables, matching `device_kernel` (`device_kernel.h:121`).
- Thread 0 of every CTA records `K_PROBE0` (`v9 = 0`): `base = cast_smem_ptr_to_uint(trace_probe_smem)`, `mapa(base, rank ^ 1)`, `mapa(base, 0)`, `base & 0xFEFFFFFF`, `cluster_ctaid.x`, `.y`, `%smid`.
- Warp 0 of every CTA: `cute::TMEM::Allocator2Sm{}.allocate(512, (uint32_t*)trace_probe_smem)` (both CTAs of a pair execute it, as the real kernel does), `__syncwarp()`, `cute::cluster_sync()` (all threads), lane 0 records `K_PROBE0` (`v9 = 1`): TMEM base, rank, smem base, and prints one `TRACE_K0 block bx by rank r smem_base 0x.. mapa_peer 0x.. mapa_rank0 0x.. masked 0x.. tmem_base 0x.. smid n` line; `cute::cluster_sync()`; then `release_allocation_lock()` and `free(base, 512)`.
- The one-thread `trace_probe_gdc_kernel` follows (H5).

Records: 16 CTAs x 2 = 32 (cap 64 not reached). Predictions: within each pair `base_odd == base_even | 0x01000000` and `base_odd & 0xFEFFFFFF == base_even`; `mapa(base_r, t)` equals CTA `t`'s own `base_t` for every `r, t` (the `shared::cta` address is the CTA's own `shared::cluster` address, which is what the peer-bit trick relies on); the low 24 bits of every base are equal and are 0 or 0x400 (B.6); the CTA rank is expected in bits [24,28) (`base_r == base_0 | (r << 24)`, reported as INFO); TMEM base equal in both CTAs of a pair, expected 0. `(base & 0xFFFFFF) + 230400 < 262144` must hold for the 14-bit descriptor field (B.3.2 step 4).

### C.4.3 Kernel probes (guarded blocks in the four headers)

| Probe | Covers | Insertion point (pristine line numbers) | Recorder and gate | Payload `v[]` | Prediction (C.6 checks) |
|---|---|---|---|---|---|
| K1 addresses | D5 | `sm100_gemm_tma_warpspecialized.hpp`, after `pipeline_init_wait(cluster_size);` (`:615`) | `threadIdx.x == 0` in every CTA with `blockIdx.x < 2 && blockIdx.y < 2`; two kinds, cap 16 each (8 records in total) | `K_SMEM`: `cast_smem_ptr_to_uint` of `smem_buf`, `&pipelines.mainloop.full_barrier_[0]`, `&...mainloop.empty_barrier_[0]`, `&...clc.full_barrier_[0]`, `&...clc.empty_barrier_[0]`, `&...accumulator.empty_barrier_[0]`, `&...tmem_dealloc`, `&clc_response[0]`, `tensors.mainloop.smem_A.begin()`, `smem_B.begin()`. `K_SMEM2`: `tensors.epilogue.collective.smem_D.begin()`, `&tmem_base_ptr`, `is_epi_load_needed`, `is_first_cta_in_cluster` (the CTA whose warp 1 schedules; `is_participant.sched` is a per-warp value and would be 0 in thread 0), `cta_rank_in_cluster`, `mma_peer_cta_rank`, `&...accumulator.full_barrier_[0]`, `&...clc_throttle.full_barrier_[0]`, `&...load_order.barrier_[0][0]`, `&...epi_load.full_barrier_[0]` | offsets 0, 64, 208, 224, 272, 336, 352, 33792, 164864; 512, 384, 240, 304, 192, 128 from each CTA's own base; bases per rank as in K0; `is_epi_load_needed = 0`, first-CTA flag `= (rank == 0)`, peer `= rank ^ 1`; `smem_A % 1024 == 0`, `smem_D % 512 == 0`; the odd rank's accumulator-empty address masked with `0xFEFFFFFF` equals the even rank's |
| K2 TMEM base | D6 | same file, after `uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;` in the MMA branch (`:731`) and in the epilogue branch (`:872`) | MMA: the `lane_predicate` lane of warp 0 (variable from `:422`); epilogue: `threadIdx.x == 128`; **all CTAs** (this probe is not cluster-gated so that every pair is checked), one cap of 1024 shared by both sites (`2 x 4 x N_native <= 296` records expected) | `tmem_base_ptr`, `cta_rank_in_cluster`, `is_mma_leader_cta`, site (0 = MMA warp, 1 = epilogue) | equal within a pair; expected 0 (informational) |
| K3 MMA operands | D7 | `mma_sm100_umma.hpp`, `SM100_MMA_F16BF16_2x1SM_SS::fma` (five-argument overload, `:563-586`), inside the existing `if (cute::elect_one_sync())` after `uint32_t mask[8] = {...}` | the elected lane; `blockIdx.x < 2 && blockIdx.y < 2`; cap 32 on `K_MMA`, shared by the two leaders of the cluster (about 16 records each, i.e. the first 4 k-tiles x 4 k-blocks; `seq` is the shared counter, the checker orders records per CTA) | `desc_a`, `desc_b`, `tmem_c`, `uint32_t(idescE >> 32)`, `scaleC` | high words `0x40004040`; low word `0x00010000 | (((base + 33792) >> 4) & 0x3FFF) + 2*kb + 1024*s` (A) and `... 164864 ... + 512*s` (B), `(s,kb)` = (0,0),(0,1),(0,2),(0,3),(1,0),... per CTA; `tmem_c = T` (first tile, stage 0); idesc `0x10200010`; `scaleC = 0` only for the first record of each leader |
| K4a TMA loads | D8 | `copy_traits_sm90_tma.hpp`, `TMA_LOAD_Unpack::copy_unpack` (`:80-86`): the `#if 0` block becomes `#if defined(CUTLASS_DEABSTRACTION_TRACE) && defined(__CUDA_ARCH__)` and its `printf` a record | the calling thread (the elected lane of warp 2); `blockIdx.x < 2 && blockIdx.y < 2`; cap 128 on `K_TMA_LOAD`, shared by the four CTAs (about 16 k-tiles of A+B per CTA) | `c0, c1, c2` (`append<5>` of the coordinate, `get<0..2>`), `cast_smem_ptr_to_uint(dst_ptr)`, `size(src)` (8192 = A, 4096 = B), `cast_smem_ptr_to_uint(get<1>(traits.opargs_))` (the barrier before masking) | first tile of each CTA (`tm = 2*(by/2) + bx%2`, `tn = 2*(bx/2) + by%2`), `q` counted per CTA and per operand: A `(64q, 128*tm + 64*(by%2), 0)` into `base + 33792 + 16384*(q%8) + 8192*(by%2)`; B `(64q, 128*tn + 64*(bx%2), 0)` into `base + 164864 + 8192*(q%8)`; barrier `base + 8*(q%8)` (B.4). A `size(src)` of 512 would be a C load and is reported (beta must be 0) |
| K4b TMA stores | D8 | same file, `Copy_Traits<SM90_TMA_STORE>::copy_unpack` (`:414-420`), same treatment | every lane executes the call (B.3.3 step 18): all lanes evaluate `__popc(__activemask())`, lane 0 records; `blockIdx.x < 2 && blockIdx.y < 2`; cap 256 (the first tile's 4 x 32 stores plus the start of the next tiles) | `c0, c1, c2`, `cast_smem_ptr_to_uint(src_ptr)`, `popc(activemask)` | record `i` of a CTA: `e = (i % 32) / 4`, `k = i % 4`, tile `i / 32` from the CTA's K5c tile list: `(128*tm + 32k, 128*tn + 16e, 0)` from `base + 512 + 8192*(e%4) + 2048k`; `popc(activemask) == 32` (all 32 lanes issue) |
| K5a CLC issue | D9 | `sm100_tile_scheduler.hpp`, `advance_to_next_work` (`:438-451`), inside the existing `if (cute::elect_one_sync())`, before `issue_clc_query` | elected lane of the sched warp (rank 0); all clusters; cap 8192 | `state.index()`, `.phase()`, `.count()`, `mbarrier_addr`, `blockIdx.x/2`, `blockIdx.y/2` | slot `count % 2`, phase `1 ^ ((count >> 1) & 1)`, barrier `base_0 + 208 + 8*slot` (B.3.4 step 5a); 1024 records in total |
| K5b CLC consume, sched warp | D9 | same file, `fetch_next_work` (`:457-473`): `WorkTileInfo trace_raw_tile = work_tile;` right after `work_tile_info_from_clc_response` (`:464`), and the recording block right after `swizzle_and_rasterize` (`:467-469`) | `threadIdx.x / 32 == 1 && threadIdx.x % 32 == 0 && cute::block_rank_in_cluster() == 0`; cap 8192 | raw `M_idx, N_idx, L_idx, is_valid()` of the response, `state.index()/.phase()/.count()`, swizzled `M_idx, N_idx` | valid responses carry an even first-CTA id `(x0, y0)` and map to `(tm, tn) = (y0, x0)` on rank 0 (B.4); the last record of each cluster has `valid = 0` (its raw fields are undefined and are not compared); consumer parity `(count >> 1) & 1` |
| K5c CLC consume, MMA warp | D9 | same block | `threadIdx.x / 32 == 0 && threadIdx.x % 32 == 0` in every CTA; cap 16384 | same payload | all four CTAs of a cluster decode identical responses in the same order (valid flag and, for valid ones, the raw ids); `(tm, tn) = (y0 + ctaid.x, x0 + ctaid.y)`; supplies each CTA's tile list for the K4 checks |

Existing hook that can serve as a cross-check (text output, not records): `#define CUTLASS_DEBUG_TRACE_LEVEL 1` in the toggle block of the `.cu` prints the host launch lines (`trace.h:52`, `cluster_launch.hpp:178-179, 286-291`). The scheduler's own `CUTLASS_SWIZZLE_DEVICE_DEBUG_PRINT` print (`sm100_tile_scheduler.hpp:804-811`) is not usable from the `.cu`: the header defines the macro to 0 unconditionally (`:34`); K5b/K5c make it unnecessary.

### C.4.4 Binary inspection (toggle-off binary, no code)

| Probe | Covers | Command (run by `deabstraction_trace_run.sh inspect`) | What to extract | Prediction |
|---|---|---|---|---|
| B1 resources | D11 | `cuobjdump --dump-resource-usage <pristine binary>` -> `resources.txt` | for the function whose mangled name contains `GemmUniversal` and `KernelTmaWarpSpecializedSm100`: REG, STACK, SHARED, LOCAL, CONSTANT | STACK 0, LOCAL 0 (no spills), SHARED 0 (all smem dynamic); REG as measured (the replacement must not exceed it) |
| B2 SASS | D11 | `cuobjdump --dump-sass` -> `sass.txt`; `--list-elf` -> `elf.txt`; `--dump-ptx` -> `ptx.txt` (empty if no PTX is embedded); `sass_summary.txt` = the GemmUniversal function's mnemonic-family counts (`UTC*`, `UTMA*`, `SYNCS*`, `CLC*`, `ACQBULK`, `PREEXIT`, `FFMA`, `FMUL`, `FADD`, `STS`, `LDS`, `BAR`, `ELECT`, predicated lines included) and the `griddepcontrol` count in the PTX | (a) ELF arch `sm_100a`; (b) exact mnemonics and counts of the tensor-core, TMA, barrier, CLC and `griddepcontrol` families inside the kernel; (c) TMA-store instructions in the unrolled epilogue loop (one per box, 32 per tile); (d) `FFMA`/`FMUL`/`FADD` counts in the epilogue (16 FMUL + 16 FFMA per subtile per thread, or 32 FMUL + 16 FADD if not contracted); (e) `STS.32` x16 per subtile and the swizzle XOR form; (f) the `griddepcontrol` lowering is present; (g) how the 16 `tcgen05.ld` destination registers are protected before first use; (h) the descriptor increment is a 32-bit add on the low word | all recorded; (c) one per box; (f) present |
| B3 build config | D14 | `grep` of `CUTLASS_NVCC_ARCHS`, `CMAKE_BUILD_TYPE`, `CUTLASS_ENABLE_GDC_FOR_SM100`, `CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL`, `CUTLASS_ENABLE_CUDA_HOST_ADAPTER`, `CMAKE_CUDA_FLAGS`, `CMAKE_CUDA_COMPILER` in `build/CMakeCache.txt`; `build/examples/70_blackwell_gemm/CMakeFiles/70_blackwell_fp16_gemm.dir/flags.make`; `nvcc --version`; `nvidia-smi` name/driver/compute capability -> `config.txt` | the exact nvcc command line of the target | `100a`, `Release`, GDC `ON`, others `OFF`; `-O3`, `-DCUTLASS_ENABLE_GDC_FOR_SM100=1`; kept verbatim as the build line of the replacement |

### C.4.5 Runs of the toggle-off binary

| Probe | Covers | Command | Record |
|---|---|---|---|
| R1 baseline | D12, D15 | `deabstraction_trace_run.sh baseline`: the Section 0 run command five times -> `baseline.txt`, with `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -lms 100` sampled alongside -> `clocks.csv`, and `nvidia-smi -q -d CLOCK` -> `clocks_query.txt` | `Disposition`, `Avg runtime`, `GFLOPS` per run; min/median/max; clocks. `Disposition: Passed` on every run is the primary evidence for D15 (the un-awaited final bulk group is not observable in the compared output) |
| R3 sanitizer (optional, not in the script) | D15 | `compute-sanitizer --tool memcheck ./build/examples/70_blackwell_gemm/70_blackwell_fp16_gemm --m=8192 --n=8192 --k=8192` | zero errors, `Passed`; if the tool rejects cluster or TMEM features, record the message. This wraps the binary in a different launcher and is outside the fixed command; skip it if the fixed command is the only permitted way to run |

## C.5 Run matrix and order

`examples/70_blackwell_gemm/deabstraction_trace_run.sh` executes the steps below (`all`, or one of `build inspect baseline trace restore check`). It requires `CUDACXX` and `CUDA_HOME`, uses only the Section 0 cmake command and run command, flips the toggle with `sed` on line 1 of the example, restores it, and writes everything to `./trace_out` (override with `TRACE_OUT`). An exit trap warns if a failure left the toggle on.

```
1. build     toggle off: Section 0 configure+build; copy the binary to trace_out/70_blackwell_fp16_gemm.pristine
2. inspect   B3 -> config.txt; B1/B2 on the pristine binary -> resources.txt, elf.txt, sass.txt, ptx.txt, sass_gemm.txt, sass_summary.txt
3. baseline  R1: the Section 0 run command five times -> baseline.txt, clocks.csv, clocks_query.txt
4. trace     toggle on; same configure+build; Section 0 run command once -> host.txt (stdout and stderr: K0, H1-H5, H2b lines,
             Disposition, ignored timing) and launch0.csv (K0 + warm-up launch records)
5. restore   toggle off; same configure+build; cmp against the pristine binary (fallback: diff of cuobjdump --dump-sass);
             git diff --stat of include/ and the example -> git_diff_stat.txt, git diff of include/ -> git_diff_include.patch
6. check     python3 examples/70_blackwell_gemm/check_trace.py trace_out/launch0.csv trace_out/host.txt -> checklist.txt
```

Steps 1-3 need no source change beyond the already-present guarded blocks (inert with the toggle off). Step 5 is the proof that trace mode leaves the performance binary untouched. Expected record volume in step 4: 32 (K0) + 8 (K1) + up to 296 (K2: 2 per launched CTA) + 32 (K3) + 128 (K4a) + 256 (K4b) + 1024 (K5a) + 1024 (K5b) + 4096 (K5c), about 6900 (every query is consumed exactly once per warp, including the failing one), far below the 65536-record buffer.

## C.6 Post-processing checks (`check_trace.py`)

The script reads `launch0.csv` and `host.txt`, derives the per-rank bases from the K1 records (cross-checked with K0) and the TMEM base `T` from the cluster-(0,0) K2 records, uses the K5c records to know which tile each CTA was working on when a load or store was recorded, and evaluates (each check prints `PASS`/`FAIL`, observed vs expected, and the document section that made the prediction; `INFO` lines report measured values without a prediction; the exit code is non-zero if any check failed):

1. **Addresses (D5).** Rank 0's base is 0 or 0x400; every rank's base has the same low 24 bits; within each pair the odd base has bit 24 set and `base_odd & 0xFEFFFFFF == base_even`; `(base & 0xFFFFFF) + 230400 < 262144`; every K1/K1b offset equals its Section 6.2 value; `smem_A % 1024 == 0`, `smem_D % 512 == 0`; the masked odd accumulator-empty address equals the even one; K0: `mapa(base_r, r^1)` and `mapa(base_r, 0)` equal the target CTA's own base, the rank field is reported (INFO, expected `r << 24`), and the K0 and K1 bases of rank 0 agree.
2. **TMEM (D6).** All K2 records of a pair carry the same base; the value is reported (expected 0); a missing cluster-(0,0) record is a FAIL.
3. **MMA operands (D7).** Per leader CTA, records in `seq` order: `desc >> 32 == 0x40004040`; `desc & 0xFFFFFFFF == 0x00010000 | (((base + off) >> 4) & 0x3FFF) + 2*kb + 1024*s` (A) or `+ 512*s` (B) with `(s, kb)` advancing (0,0)..(0,3),(1,0)...; `tmem_c == T`; idesc `0x10200010`; `scaleC == 0` exactly for the first record.
4. **TMA operands (D8).** Using each CTA's tile list (initial mapping, then the valid K5c responses), every K4a record matches the A/B formulas of B.4 and every K4b record matches the store formula; every store record has `popc(activemask) == 32`; a `size(src)` of 512 in a load record is reported.
5. **CLC (D9).** From K5a/K5b: per cluster `count` runs 0..T_c-1, slot `count % 2`, producer phase `1 ^ ((count >> 1) & 1)`, consumer phase `(count >> 1) & 1`, barrier `base_0 + 208 + 8*slot`; exactly one `valid = 0` record per cluster and it is the last; the valid first-CTA ids are even and map to `(y0, x0)`; the natively launched cluster ids (clusters that produced any record, each owning cluster tile `(cy, cx)`) plus the cancelled clusters' tiles `(y0/2, x0/2)` cover the 32x32 cluster-tile grid exactly once; `sum(T_c) == 1024` and 1024 queries were issued; issues and consumes per cluster agree; `N_native <= cudaOccupancyMaxActiveClusters` (H3); K5c: every CTA of a cluster holds the same response sequence (valid flags, and raw ids for valid responses) and `(tm, tn) = (y0 + ctaid.x, x0 + ctaid.y)`. INFO: tiles per native cluster (min/max/mean), query-to-consume latency (min/median/p90/max). SM placement is in the `smid` column of every record but is not summarized.
6. **Host (D1-D4, D13, D14).** Sizes and offsets against Section 6.2; six `TRACE_ENCODE` lines with the Section 5.3 tuples and `result = 0`; fallback maps identical to the primaries; grid, block, scheduler params; driver version above 13010; kernel attributes; `cudaGetLastError == 0`; SM count 148 and 1024 reserved bytes (informational facts of Section 0, checked as PASS/FAIL so a different machine is noticed); build macros as in Section 0; `TRACE_DEV` GDC flag and `sm_100a` feature macro; nvcc 13.3. The printed types (`TRACE_H1`, `TRACE_TYPE`, `TRACE_H2`) are compared with Section 4 by hand when Part D is written.

## C.7 Recording the results in this document

After the runs, add **Part D "Measured values"**: one row per D item with the observed value, its source file (`launch0.csv`, `host.txt`, `sass_summary.txt`, `resources.txt`, `baseline.txt`, `checklist.txt`) and PASS/FAIL against the prediction. Then mark each item in B.7 and Section 11 as confirmed or replace the prediction with the measured fact (the per-rank bases, `T`, `N_native`, register count, SASS mnemonics, FFMA/FMUL split, baseline runtime distribution). A failed prediction is traced to the Part B step that produced it, the step is corrected, and the correction is listed in B.8. The replacement kernel is written only after Part D exists, so it can target the measured register budget, instruction mix and exit condition (D15), not just the derived ones.

## C.8 Cautions

- The trace binary has more registers and different timing; nothing measured with the toggle on is a performance number, and its `Avg runtime` line is discarded.
- A recorded event costs two returning atomics and a 128-byte store; after a kind reaches its cap the `g_trace_full` flag reduces later events to one cached load. Probes in the per-k-tile paths (K3, K4a) are limited to cluster (0,0), so that cluster runs slightly slower and may process fewer tiles; the CLC distribution is `[R]` anyway.
- The lane that `elect.sync` picks is implementation-defined; K3 and K5a record from inside existing `elect_one_sync()` regions, K5b/K5c gate on lane 0 because every lane executes `fetch_next_work`, K2 uses the kernel's own `lane_predicate`. K1's single-lane record precedes warp 0's `tcgen05.alloc ... .sync.aligned` by the same shape as the kernel's own single-lane `mbarrier.init`; the `.aligned` contract requires a warp-uniform guard, which `warp == 0` is, not a reconvergence point.
- `fetch_next_work` is `CUTLASS_HOST_DEVICE`, `copy_unpack` is `CUTE_HOST_DEVICE constexpr`: their probes are device-pass only (`__CUDA_ARCH__`) and contain only literal-type locals and calls, no `asm`, no statics. The H2b block is host-only code inside `make_tma_copy_desc`.
- The guarded blocks in `include/` are removed by the preprocessor when the toggle is off; C.5 step 5 verifies this on the binary and records the source diff.
- If a step fails, the exit trap says whether the toggle is still on; run `deabstraction_trace_run.sh restore` before taking any performance number.
- `compute-sanitizer` may not support every Blackwell feature; R3 is optional and outside the fixed command line.
- The `.gitignore` hunk staged in the working tree (`*.log`) predates this work and is not part of it.
