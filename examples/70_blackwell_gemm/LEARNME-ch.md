# LEARNME: Explicit Blackwell FP16 GEMM Kernel，从设计到每一个字符

一篇关于 `examples/70_blackwell_gemm/70_blackwell_fp16_gemm_explicit.cu` 的讲义，该文件是 CUTLASS 示例 70 的去抽象化重新实现（一个非模板的 `__global__` kernel，设备端代码中没有 CUTLASS/CuTe 抽象），针对 NVIDIA B200 上的固定运行 `--m=8192 --n=8192 --k=8192`。

更新于 2026/10/16

用以下命令取回该 Claude 会话：`claude --resume aa5f968f-7641-4a17-be49-09c44708be24`

**目标。** 学完这十四章之后，你应当能够 (1) 阅读 kernel 的每一行、每一个字符，并说出它为什么在那里，(2) 从一个空白文件出发重写这个 kernel，顶层设计在先、PTX 在后，以及 (3) 把这套方法带到其他算子上，写出你自己的 speed-of-light kernel。

**组织。** 第 1-4 章是自顶向下的：这个 kernel 是什么、它所驱动的硬件、设计决策及其 speed-of-light 算术，以及把各个 warp 联系在一起的屏障协议。第 5-11 章是自底向上的：逐节解读源文件、每一个常量、每一个 PTX wrapper、每一个角色函数、prologue 与主机端启动器。第 12-14 章是精通：编译器把代码变成了什么、测得了什么，如何自己构建并泛化这样的 kernel，以及带答案的练习和一份术语表。

**标签。** 在这种区分重要的地方，一条陈述会被标注为实测（笔记的 D 部分和 G 部分）、源自源码或分析（在第 12 章中缩写为 [M]、[S]、[H]，在第 14 章中缩写为 [M]、[S]、[A]）；一条我无法佐证的论断被标为（未验证）；关于硬件、工具链或工具，且我得自 NVIDIA 内部信息而非公开文档的事实，被标为（NVIDIA 内部）。

**记法。** rank `r = x + 2y`，其中 `(x, y) = %cluster_ctaid`。pair leader 是偶数 rank；peer 是 `r ^ 1`；CTA 0 是 rank 0。tile 坐标 `(tm, tn)` 以 128 x 128 的 CTA tile 为单位计数。`q` 是 k-tile (0..127)，`s` 是 A/B smem stage (0..7)，`kb` 是 k-block (0..3)，`st` 是 TMEM 累加器 stage (0..3)，`e` 是 epilogue subtile (0..7)，`i` 是存储 box (0..3)，`j` 是寄存器或列 (0..15)，`w = warp_idx - 4` 是 epilogue warp (0..3)，`l` 是 lane (0..31)。`smem_base` 是动态共享内存的 32 位 `shared::cta` 地址；`T` 是 TMEM 基地址。

## 目录

- [第 1 章 导览：这个 kernel 是什么，以及如何阅读它](#第-1-章-导览这个-kernel-是什么以及如何阅读它)
  - [1.1 这个 kernel 是什么](#11-这个-kernel-是什么)
  - [1.2 文件集](#12-文件集)
  - [1.3 注释图例：`[PROD]`、`[TRACE]`、`[CHECK]`、`[HOST]`](#13-注释图例prodtracecheckhost)
  - [1.4 FILE MAP：12 个节](#14-file-map12-个节)
  - [1.5 三层图景](#15-三层图景)
  - [1.6 关于 NUMBERS AT A GLANCE 块的说明 (explicit.cu:52-70)](#16-关于-numbers-at-a-glance-块的说明-explicitcu52-70)
  - [1.7 如何使用本讲义](#17-如何使用本讲义)
  - [1.8 要记在脑中的事实](#18-要记在脑中的事实)
- [第 2 章 机器：面向 GEMM 编写者的 Blackwell SM100](#第-2-章-机器面向-gemm-编写者的-blackwell-sm100)
  - [2.1 硅片层级：GPU、GPC、TPC、SM、cluster、pair](#21-硅片层级gpugpctpcsmclusterpair)
  - [2.2 带数字的内存层级](#22-带数字的内存层级)
  - [2.3 张量内存（TMEM）](#23-张量内存tmem)
  - [2.4 tcgen05 族：异步、单线程发射、由 commit 跟踪](#24-tcgen05-族异步单线程发射由-commit-跟踪)
  - [2.5 2SM MMA：`cta_group::2`](#25-2sm-mmacta_group2)
  - [2.6 描述符：概念](#26-描述符概念)
  - [2.7 TMA：tensor map、box、swizzle、multicast、bulk group、异步 proxy](#27-tmatensor-mapboxswizzlemulticastbulk-group异步-proxy)
  - [2.8 深入 `mbarrier`](#28-深入-mbarrier)
  - [2.9 地址：`shared::cta`、`shared::cluster`、`mapa` 与第 24 位](#29-地址sharedctasharedclustermapa-与第-24-位)
  - [2.10 cluster 屏障、命名屏障、`elect.sync` 与收敛](#210-cluster-屏障命名屏障electsync-与收敛)
  - [2.11 uniform 数据通路，以及为什么 warp-uniform 发射很重要](#211-uniform-数据通路以及为什么-warp-uniform-发射很重要)
  - [2.12 Cluster 启动控制（CLC）](#212-cluster-启动控制clc)
  - [2.13 程序化依赖启动：`griddepcontrol`](#213-程序化依赖启动griddepcontrol)
  - [2.14 映射：硬件特性 -> PTX -> wrapper -> warp](#214-映射硬件特性---ptx---wrapper---warp)
- [第 3 章 设计：从 8192^3 到 256x128x16，以及 Speed of Light](#第-3-章-设计从-81923-到-256x128x16以及-speed-of-light)
  - [3.1 计算及其 layout](#31-计算及其-layout)
  - [3.2 tile 层级](#32-tile-层级)
  - [3.3 数据复用算术：为什么是 (2,2,1)](#33-数据复用算术为什么是-221)
  - [3.4 预算](#34-预算)
  - [3.5 流水线深度及其理由](#35-流水线深度及其理由)
  - [3.6 epilogue tile (128, 16) 与 (32, 16, 1) 存储 box](#36-epilogue-tile-128-16-与-32-16-1-存储-box)
  - [3.7 warp 特化：8 个 warp、5 种角色、256 个线程](#37-warp-特化8-个-warp5-种角色256-个线程)
  - [3.8 使用 CLC 的持久化调度](#38-使用-clc-的持久化调度)
  - [3.9 Speed-of-light 分析](#39-speed-of-light-分析)
  - [3.10 设计决策表](#310-设计决策表)
  - [3.11 “设计先于代码”这一课](#311-设计先于代码这一课)
- [第 4 章 协议：屏障、流水线、parity 与 per-tile 时间线](#第-4-章-协议屏障流水线parity-与-per-tile-时间线)
  - [4.1 为什么需要协议：五个异步引擎与一条规则](#41-为什么需要协议五个异步引擎与一条规则)
  - [4.2 一页纸的 mbarrier](#42-一页纸的-mbarrier)
  - [4.3 流水线状态与 parity 公式](#43-流水线状态与-parity-公式)
  - [4.4 逐行解读屏障表](#44-逐行解读屏障表)
  - [4.5 先 peek 再 wait](#45-先-peek-再-wait)
  - [4.6 数据流图](#46-数据流图)
  - [4.7 per-tile 时间线](#47-per-tile-时间线)
  - [4.8 退出协议](#48-退出协议)
  - [4.9 冒险、是什么防止了它们，以及没有它们会发生什么](#49-冒险是什么防止了它们以及没有它们会发生什么)
- [第 5 章 基础：文件、参数块、tensor map 与第 1 节的常量](#第-5-章-基础文件参数块tensor-map-与第-1-节的常量)
  - [5.1 文件集](#51-文件集)
  - [5.2 逐行解读 `util.hpp`](#52-逐行解读-utilhpp)
  - [5.3 逐行解读 `explicit.hpp`](#53-逐行解读-explicithpp)
  - [5.4 tensor map](#54-tensor-map)
  - [5.5 `explicit.cu` 第 1 节，第 1 组：配置常量](#55-explicitcu-第-1-节第-1-组配置常量)
  - [5.6 LAYOUT 注释块，逐 mode 解码](#56-layout-注释块逐-mode-解码)
  - [5.7 `explicit.cu` 第 1 节，第 2 组：共享内存映射](#57-explicitcu-第-1-节第-2-组共享内存映射)
  - [5.8 `explicit.cu` 第 1 节，第 3 组：UMMA 描述符](#58-explicitcu-第-1-节第-3-组umma-描述符)
  - [5.9 `explicit.cu` 第 1 节，第 4 组：协议常量](#59-explicitcu-第-1-节第-4-组协议常量)
  - [5.10 `explicit.cu` 第 1 节，第 5 组：CuTe swizzle 交叉检查](#510-explicitcu-第-1-节第-5-组cute-swizzle-交叉检查)
  - [5.11 第 2 节简述：仅用于 trace 的类型](#511-第-2-节简述仅用于-trace-的类型)
- [第 6 章 词汇：PTX wrapper W1-W37，逐字符解读](#第-6-章-词汇ptx-wrapper-w1-w37逐字符解读)
  - [6.0 为什么需要一个词汇层，以及如何阅读一条内联汇编语句](#60-为什么需要一个词汇层以及如何阅读一条内联汇编语句)
  - [6.1 第 0 族：地址桥梁，`smem_u32`](#61-第-0-族地址桥梁smem_u32)
  - [6.2 第 1 族：warp 与 cluster 身份，W1-W4](#62-第-1-族warp-与-cluster-身份w1-w4)
  - [6.3 第 2 族：mbarrier，W5-W13](#63-第-2-族mbarrierw5-w13)
  - [6.4 第 3 族：fence，W14-W15](#64-第-3-族fencew14-w15)
  - [6.5 第 4 族：cluster 屏障与命名屏障，W16-W19](#65-第-4-族cluster-屏障与命名屏障w16-w19)
  - [6.6 第 5 族：TMA，W20-W25](#66-第-5-族tmaw20-w25)
  - [6.7 第 6 族：tcgen05，W26-W32](#67-第-6-族tcgen05w26-w32)
  - [6.8 第 7 族：CLC，W33-W34](#68-第-7-族clcw33-w34)
  - [6.9 第 8 族：网格依赖控制，W35-W36，以及 W37](#69-第-8-族网格依赖控制w35-w36以及-w37)
  - [6.10 汇总表](#610-汇总表)
- [第 7 章 分发器：流水线辅助函数、clc_consume 与 scheduler warp](#第-7-章-分发器流水线辅助函数clc_consume-与-scheduler-warp)
  - [7.1 本章在文件中的位置](#71-本章在文件中的位置)
  - [7.2 第 4 节：流水线状态约定、`desc64` 与 wait 形式](#72-第-4-节流水线状态约定desc64-与-wait-形式)
  - [7.3 硬件契约：CLC 承诺了什么](#73-硬件契约clc-承诺了什么)
  - [7.4 第 5 节：逐行解读 `clc_consume`](#74-第-5-节逐行解读-clc_consume)
  - [7.5 从 `(x0, y0)` 到 `(tm, tn)`：从 CUTLASS 推导映射](#75-从-x0-y0-到-tm-tn从-cutlass-推导映射)
  - [7.6 第 7 节：逐行解读 `scheduler_role`](#76-第-7-节逐行解读-scheduler_role)
  - [7.7 一次 CLC 往返：时序](#77-一次-clc-往返时序)
  - [7.8 设计注记：为什么分发器是这个形状，以及如何复用它](#78-设计注记为什么分发器是这个形状以及如何复用它)
- [第 8 章 供给者：producer_role 与 load_ktiles（Warp 2）](#第-8-章-供给者producer_role-与-load_ktileswarp-2)
  - [8.1 一张图看 Warp 2](#81-一张图看-warp-2)
  - [8.2 逐步解读该节的横幅注释 (explicit.cu:708-722)](#82-逐步解读该节的横幅注释-explicitcu708-722)
  - [8.3 ProducerState (explicit.cu:723-726)](#83-producerstate-explicitcu723-726)
  - [8.4 逐行解读 load_ktiles (explicit.cu:728-760)](#84-逐行解读-load_ktiles-explicitcu728-760)
  - [8.5 跨 cluster 装配一个 stage](#85-跨-cluster-装配一个-stage)
  - [8.6 逐行解读 producer_role (explicit.cu:762-807)](#86-逐行解读-producer_role-explicitcu762-807)
  - [8.7 算例与计数](#87-算例与计数)
  - [8.8 TMA 单元在写入时做什么：SWIZZLE_128B](#88-tma-单元在写入时做什么swizzle_128b)
  - [8.9 warp 2 的占用，以及为什么它的循环长度无关紧要](#89-warp-2-的占用以及为什么它的循环长度无关紧要)
  - [8.10 把这些带到其他 kernel](#810-把这些带到其他-kernel)
- [第 9 章 引擎：mma_role（Warp 0）](#第-9-章-引擎mma_rolewarp-0)
  - [9.1 引擎位于何处](#91-引擎位于何处)
  - [9.2 逐句解读横幅注释 (explicit.cu:871-880)](#92-逐句解读横幅注释-explicitcu871-880)
  - [9.3 pair、2SM 指令与 TMEM 映射](#93-pair2sm-指令与-tmem-映射)
  - [9.4 设置：分配、发布、描述符基址、掩码、状态 (explicit.cu:881-908)](#94-设置分配发布描述符基址掩码状态-explicitcu881-908)
  - [9.5 tile 循环 (explicit.cu:909-949)](#95-tile-循环-explicitcu909-949)
  - [9.6 终止 (explicit.cu:950-975)](#96-终止-explicitcu950-975)
  - [9.7 算例](#97-算例)
  - [9.8 ptxas 生成的热循环（笔记 G.3，实测）](#98-ptxas-生成的热循环笔记-g3实测)
  - [9.9 设计注记：哪些要带到其他 kernel](#99-设计注记哪些要带到其他-kernel)
- [第 10 章 收尾者：epilogue_role（Warp 4-7）](#第-10-章-收尾者epilogue_rolewarp-4-7)
  - [10.1 epilogue 位于何处](#101-epilogue-位于何处)
  - [10.2 横幅注释 (explicit.cu:977-994)](#102-横幅注释-explicitcu977-994)
  - [10.3 设置：TMEM 地址、warp 与 lane 身份 (explicit.cu:995-1006)](#103-设置tmem-地址warp-与-lane-身份-explicitcu995-1006)
  - [10.4 st_off[4] 表 (explicit.cu:1007-1014) 与四个基址](#104-st_off4-表-explicitcu1007-1014-与四个基址)
  - [10.5 推导 swizzled smem 公式](#105-推导-swizzled-smem-公式)
  - [10.6 per-tile 循环头 (explicit.cu:1024-1034)](#106-per-tile-循环头-explicitcu1024-1034)
  - [10.7 一个 subtile 的步骤 1：tcgen05.ld（explicit.cu:1035-1036，W31 位于 542-555）](#107-一个-subtile-的步骤-1tcgen05ldexplicitcu1035-1036w31-位于-542-555)
  - [10.8 e == 7 时的步骤 2：wait::ld、释放 TMEM stage、推进（explicit.cu:1037-1041；W32 位于 557-565；W11 位于 369-379）](#108-e--7-时的步骤-2waitld释放-tmem-stage推进explicitcu1037-1041w32-位于-557-565w11-位于-369-379)
  - [10.9 算术运算与 16 次标量存储 (explicit.cu:1042-1048)](#109-算术运算与-16-次标量存储-explicitcu1042-1048)
  - [10.10 fence、第一个屏障、warp 4 的四次存储、commit、wait (explicit.cu:1049-1070)](#1010-fence第一个屏障warp-4-的四次存储commitwait-explicitcu1049-1070)
  - [10.11 循环之后：没有 store_tail (explicit.cu:1076-1080)](#1011-循环之后没有-store_tail-explicitcu1076-1080)
  - [10.12 per-tile 核算、带宽，以及为什么 epilogue 不在关键路径上](#1012-per-tile-核算带宽以及为什么-epilogue-不在关键路径上)
  - [10.13 一个 subtile 的完整数据路径：tile (5, 9)，e = 3](#1013-一个-subtile-的完整数据路径tile-5-9e--3)
  - [10.14 如果你自己写这个角色](#1014-如果你自己写这个角色)
- [第 11 章 框架：kernel prologue、分发与主机端启动镜像](#第-11-章-框架kernel-prologue分发与主机端启动镜像)
  - [11.1 我们在文件中的位置](#111-我们在文件中的位置)
  - [11.2 签名 (explicit.cu:1095-1098)](#112-签名-explicitcu1095-1098)
  - [11.3 步骤 1：身份 (explicit.cu:1100-1107)](#113-步骤-1身份-explicitcu1100-1107)
  - [11.4 步骤 2：标量、k-tile 计数、描述符预取 (explicit.cu:1109-1120)](#114-步骤-2标量k-tile-计数描述符预取-explicitcu1109-1120)
  - [11.5 步骤 3：33 次 `mbarrier.init` 调用 (explicit.cu:1122-1167)](#115-步骤-333-次-mbarrierinit-调用-explicitcu1122-1167)
  - [11.6 步骤 4-6：拆分的 cluster 会合与初始 tile (explicit.cu:1169-1179)](#116-步骤-4-6拆分的-cluster-会合与初始-tile-explicitcu1169-1179)
  - [11.7 步骤 7：分发与贯穿而过的 warp (explicit.cu:1190-1203)](#117-步骤-7分发与贯穿而过的-warp-explicitcu1190-1203)
  - [11.8 第 11 节：主机端启动镜像 `run()` (explicit.cu:1265-1329)](#118-第-11-节主机端启动镜像-run-explicitcu1265-1329)
  - [11.9 作为时间线的框架，以及一份重写检查清单](#119-作为时间线的框架以及一份重写检查清单)
- [第 12 章 产物：ptxas 生成了什么，以及测得了什么](#第-12-章-产物ptxas-生成了什么以及测得了什么)
  - [12.1 三个产物，一条工具链](#121-三个产物一条工具链)
  - [12.2 PTX 到 SASS 词典](#122-ptx-到-sass-词典)
  - [12.3 uniform 数据通路，或者说为什么选举消失了](#123-uniform-数据通路或者说为什么选举消失了)
  - [12.4 寄存器：26 对 68](#124-寄存器26-对-68)
  - [12.5 四个热循环](#125-四个热循环)
  - [12.6 测得了什么](#126-测得了什么)
  - [12.7 如何复现与检视](#127-如何复现与检视)
  - [12.8 经验：“性能等价”要求了什么，以及哪些是自由的](#128-经验性能等价要求了什么以及哪些是自由的)
- [第 13 章 自己动手写：构建顺序、陷阱、泛化与一份 SOL 配方](#第-13-章-自己动手写构建顺序陷阱泛化与一份-sol-配方)
  - [13.1 一页纸的方法](#131-一页纸的方法)
  - [13.2 构建计划：从空白文件开始的八个步骤](#132-构建计划从空白文件开始的八个步骤)
  - [13.3 按症状排列的陷阱检查清单](#133-按症状排列的陷阱检查清单)
  - [13.4 泛化：每种情形都作为相对于本 kernel 的差异](#134-泛化每种情形都作为相对于本-kernel-的差异)
  - [13.5 SOL 设计配方](#135-sol-设计配方)
  - [13.6 算例：256 x 256 pair-tile FP16 GEMM](#136-算例256-x-256-pair-tile-fp16-gemm)
- [第 14 章 练习与术语表](#第-14-章-练习与术语表)
  - [14.1 如何使用本章](#141-如何使用本章)
  - [14.2 练习](#142-练习)
  - [14.3 术语表](#143-术语表)
  - [14.4 数字卡片](#144-数字卡片)

---
# 第 1 章 导览：这个 kernel 是什么，以及如何阅读它

本章以及整篇讲义所用的约定：`explicit.cu` 即 `examples/70_blackwell_gemm/70_blackwell_fp16_gemm_explicit.cu`，`explicit.hpp` 和 `util.hpp` 是它旁边的两个头文件，“harness”即 `70_blackwell_fp16_gemm.cu`，而“笔记”即 `examples/70_blackwell_gemm/Semantics-preserving-de-abstraction.md`。在这种区分重要的地方，各条陈述会标注为 **实测**（笔记的 D 部分和 G 部分）、**源自源码**、**我的分析** 或 **（NVIDIA 内部）**（未公开记载的 NVIDIA 工程信息）。

---

## 1.1 这个 kernel 是什么

### 1.1.1 一段话的定义

`explicit.cu` 以它自己的定义开篇 (explicit.cu:5-23)：

```
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
```

把那段横幅注释中的三条断言读作整篇讲义的三条轴线：

1. **相同的计算。** 该 kernel 对固定问题计算 `D = alpha * A * B + beta * C`，与 CUTLASS kernel 的计算完全一致，逐值相同。它所替换的参考 kernel 在 harness:102-156 实例化：`half_t` A（RowMajor = K-major）、`half_t` B（ColumnMajor = K-major）、`float` C/D（ColumnMajor = M-major）、`MmaTileShape_MNK = Shape<_256,_128,_64>`、`ClusterShape_MNK = Shape<_2,_2,_1>`、`KernelScheduleAuto`、`EpilogueScheduleAuto`、`GemmUniversal`、`GemmUniversalAdapter`（笔记，第 2 节）。
2. **相同的性能机制。** 带 2 x 2 x 1 cluster 的双 SM `tcgen05` MMA、带 A multicast 的 TMA 加载、8 个 stage 的 A/B 共享内存流水线、4 个 stage 的 TMEM 累加器流水线、warp 特化、持久化的 Cluster 启动控制（CLC）scheduler，以及带 4 个 smem stage 的 TMA 存储 epilogue（笔记，第 1 节第 2 项）。这些都不可以被近似掉。正是这一点使该 kernel 成为一个名副其实的 "speed-of-light" 研究对象，而不是一个玩具。
3. **没有隐藏层。** 没有 `CollectiveMma`、`CollectiveEpilogue`、`GemmUniversal`、`PersistentTileSchedulerSm100`、`Pipeline*`、`TiledMma`、`TiledCopy`、`FusionCallbacks`（笔记，第 1 节第 3 项）。硬件协议的每一条 PTX 指令（横幅注释所列的那些指令族，从 `tcgen05.*` 到 `prefetch.tensormap`）都以 `asm volatile` 字符串的形式可见于文件的第 3 节（37 个 wrapper，W1-W37；W1 和 W37 是 `__shfl_sync` 和 `__syncwarp()` 内建函数，而不是 asm）；普通的加载、存储和算术运算则是纯 C++。每一个“谁在哪个屏障上、以哪个计数和 parity 调用哪个 wrapper”的决策都可见于第 5-10 节。在 `explicit.cu` 中，CuTe 只在编译期出现（:78 处的 `cute/swizzle.hpp` include、:198 处的 `EpiSwizzle` 别名以及 :199-204 处的各个 `static_assert`）：`cute::Swizzle<2,5,2>` 交叉检查手写的 epilogue swizzle 公式 (explicit.cu:198-202)，`cute::Swizzle<3,4,3>` 钉住 A/B stage 的 swizzle 掩码 (explicit.cu:204)。`explicit.hpp` 还在仅主机端的 `make_params` 模板内部使用了 `cute::get<i>(p.problem_shape)`（explicit.hpp:39, 107-110），该模板在 harness 翻译单元中实例化。没有任何 CuTe 类型到达设备端代码。

### 1.1.2 固定的运行

该 kernel 硬编码了一种配置（笔记，第 0 节；util.hpp:29-39），而主机端函数在启动之前拒绝其他一切 (explicit.cu:1266-1272)。配置中的两行，用于导览（第 5 章有完整的表）：

| 量 | 值 | 在何处固定 |
|---|---|---|
| A | fp16，M x K，K-major (`RowMajor`) | harness:102-104（`ElementA = half_t`、`LayoutA = cutlass::layout::RowMajor`；B 在 :107-108，C/D 在 :112-113；:117-156 是 kernel 功能配置、tile shape 与 cluster shape、两个 `CollectiveBuilder` 以及 :150-156 处的 `GemmUniversal`/`GemmUniversalAdapter` 组合）；tensor map A：`FLOAT16`，维度 (K, M, L) = (8192, 8192, 1)，box (64, 64, 1)，`SWIZZLE_128B` (explicit.hpp:50) |
| alpha、beta | 1.0f、0.0f，从参数块中 **在运行时读取** | `alpha(1.f), beta(0.f),` harness:210；explicit.hpp:55；为什么是运行时：第 10 章，第 10.9 节 |

为什么要硬编码？笔记的第 0 节记录了用户的决定：该示例只以恰好一种方式构建和运行，因此替代品可以把 shape、tile、stage 和 scheduler 常量折叠为 `constexpr` 值。收益是一个每个数字都可见、可解释的 kernel（本讲义）；代价是通用性（不同的 shape 会以 `kErrorInvalidProblem` 被拒绝，explicit.cu:1268）。

---

## 1.2 文件集

一切都位于 `examples/70_blackwell_gemm/` 中，只有两个证据目录例外，它们位于仓库根目录（因此下文用 `../../`）。该表列出你将用到的文件、它们的大小和角色；`check_trace.py` 与 `examples/common/helper.h` 只是顺带提及。第 1.4 节的 FILE MAP 所带有的 `[PROD]`/`[TRACE]`/`[CHECK]`/`[HOST]` 标签在第 1.3 节中解释。

| 文件 | 行数 | 角色 |
|---|---|---|
| `70_blackwell_fp16_gemm_explicit.cu` | 1451 | **本讲义的讲解对象。** 第 1-12 节，列于 FILE MAP 中（第 1.4 节）。唯一定义设备端代码的文件；harness 的翻译单元仍会从 CUTLASS 头文件实例化那个从不启动的 CUTLASS kernel 以及参考 GEMM、填充和比较 kernel（笔记 E.3.3；`trace_out_v2/resources.txt`）。 |
| `70_blackwell_fp16_gemm_explicit.hpp` | 134 | 两个翻译单元之间的 ABI：`ExplicitGemmParams`（四个 `CUtensorMap`、m/n/k/l、alpha/beta、仅主机端的验证字段；explicit.hpp:49-63）、钉住设备端所读取的每个偏移的各个 `static_assert`（explicit.hpp:66-76）、`run(ExplicitGemmParams const&)` 的声明（explicit.hpp:80），以及模板桥 `make_params(gemm)` / `run(gemm)`（explicit.hpp:99-132）。没有设备端代码。第 5 章。 |
| `70_blackwell_fp16_gemm_explicit_util.hpp` | 63 | **第 1 行是 trace 开关**（util.hpp:1）。然后是不需要任何 CUDA 类型的第 0 节常量（`kFixedM/N/K/L`、`kCtaTileM/N`、`kTileK`、`kClusterX/Y/Z`、`kGridX/Y/Z`、`kBlockThreads`、`kSmemBytes`、`kProblemTilesM/N/L`、`kRasterOrderAlongN`、`kModeGemm`、`kSmemDescConst`、`kInstrDescConst`；util.hpp:29-39）、它们的 `static_assert`（util.hpp:41-46），以及 `status_from_cuda`（util.hpp:51-53）。第 5 章。 |
| `70_blackwell_fp16_gemm.cu`（harness） | 506 | 除四处无守卫的编辑和五个受开关守卫的块（两个 include、三处调用）之外未改动的 CUTLASS 示例。第 5.1.1 节（两个翻译单元）、笔记 E.4（这些编辑）；关于它所调用的 `run()`，第 11.8 节。 |
| `Semantics-preserving-de-abstraction.md`（笔记） | 2059 | 知识库。A-I 部分。本讲义以“笔记，第 7.5 节”或“笔记 D.2”的形式引用它。 |
| `70_blackwell_fp8_gemm.cu` | 672 | 同一目录下的姊妹 FP8 示例；未被去抽象化触及。 |
| `../../trace_out/` | - | CUTLASS kernel 的 2026-09-09 baseline 运行的证据，以及未改动的 baseline 二进制文件 `70_blackwell_fp16_gemm.pristine`。绝不能被覆盖（笔记 E.9 第 17 项）。 |
| `../../trace_out_v2/` | - | explicit kernel 的 2026-09-11 验收运行的证据（笔记的 G 部分）：`host.txt`、`launch0.csv`、`checklist.txt`、`sass_explicit.txt`、`ptx_explicit.txt`、`resources.txt`、`sass_checklist.txt`、`timing_summary.txt`、`clocks.csv`，以及开关关闭的二进制文件 `70_blackwell_fp16_gemm.explicit`。 |

讲解对象的 Git 历史（`git log --oneline -- 70_blackwell_fp16_gemm_explicit.cu`）：`829f0645` "add explicit kernels for de-abstraction"（代码：1298 行 `.cu`、112 行 `.hpp`、56 行 `_util.hpp`、harness 与 CMake 的编辑、工具），然后是 `6726adc0` "add comments for better readability"（文档层：仅注释，代码逐字节相同，笔记 H.4）。G 部分的实测二进制文件是 `829f0645` 的代码；注释层不需要重新构建。

---

## 1.3 注释图例：`[PROD]`、`[TRACE]`、`[CHECK]`、`[HOST]`

三个源文件中每个节的横幅注释和许多行尾注释都带有标签。图例在文件头部 (explicit.cu:26-36)：

```
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
```

阅读时如何使用这些标签：

- 当你想要 **GPU 所运行的 kernel** 时，读 `[PROD]`，并跳过每一个 `#if defined(CUTLASS_DEABSTRACTION_TRACE)` 块。`explicit.cu` 有 29 个这样的预处理器行（`grep -c '^#if defined(CUTLASS_DEABSTRACTION_TRACE)'`；单纯的 `grep -c` 会给出 30，因为第 29 行的图例文字也匹配），每一行都与附近的一个 `#endif`（有时还有一个 `#else`）配对。开关关闭时，第 2 节缩减为它的 `#else` 分支（explicit.cu:235-240：`EXPLICIT_WAIT` 是普通的阻塞等待 `mbar_wait`，而 CLC 记录种类为 0），第 12 节则消失。`clc_consume<TraceKind>` 仍是一个模板，以 `K_CLC_SCHED_KIND = K_CLC_MMA_KIND = 0u` 实例化 (explicit.cu:238-239)；模板参数是编译期常量，因此 `clc_consume<0u>` 编译出的结果与一个非模板函数完全相同。
- 当你想要 **知道某个数字是对的** 时，读 `[CHECK]`。
- 当你想要 **看 kernel 是如何被启动的** 时，读 `[HOST]`：第 11 节（`run()`，explicit.cu:1265-1329）和头文件桥。
- 硬件族的词（`tcgen05`、`TMA`、`mbarrier`、`CLC`、`GDC`、`LAYOUT`）告诉你一个节属于第 6 章的哪个 wrapper 组和第 2 章的哪个硬件单元。`LAYOUT` 标出展开写出的整数算术取代了 CuTe layout 代数的地方：第 1 节（它的横幅注释、所引用的 CuTe layout、字节偏移、UMMA 描述符常量以及 CuTe 交叉检查；explicit.cu:98, 122, 137, 168, 194）、`clc_consume` 中的坐标公式（第 5 节，explicit.cu:665），以及 epilogue 中的 swizzled 存储偏移（第 9 节，explicit.cu:978）。producer 中的 TMA 坐标（第 6 节）和 MMA 角色中的描述符增量（第 8 节）也是展开写出的算术，但它们的横幅注释只带有 `TMA mbarrier` 和 `tcgen05`（explicit.cu:709, 871）。

---

## 1.4 FILE MAP：12 个节

横幅注释列出了各节 (explicit.cu:38-50)：

```
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
```

各节按这个顺序出现，是因为 C++ 需要先定义后使用：常量在嵌入它们的 wrapper 之前，wrapper 在辅助函数之前，辅助函数在角色之前，角色在分发它们的 kernel 之前，kernel 在取其地址的主机端函数之前。同一顺序也是一个合理的*自底向上*的阅读顺序，而且，除一个例外之外，它就是第 5-11 章的顺序（第 7 章连同第 4-5 节一起讲第 7 节，即 scheduler warp，然后第 8 章才讲第 6 节）。

### 1.4.1 每一节位于何处，以及哪一章拥有它

| 节 | 行 | 内容（要找什么） | 讲义章 |
|---|---|---|---|
| 头部 | 1-91 | include（先是 util.hpp，然后是带守卫的 trace 头文件，然后是 `<cuda.h>`、`<cuda_runtime.h>`、`<cstdint>`、`<cstdio>`、仅用于 `cutlass::Status` 的 `cutlass/cutlass.h`、仅用于编译期检查的 `cute/swizzle.hpp`、`explicit.hpp`，然后是 explicit.cu:82-91 处的第二个带守卫的 trace 块）、图例、FILE MAP、NUMBERS AT A GLANCE | 第 1 章（本章）、第 5 章 |
| 1 | 98-204 | `kThreads`、`kMmaM/N/K`、`kStages = 8`、`kAccStages = 4`、`kClcStages = 2`、`kEpiStages = 4`、`kEpiSubtiles = 8`、`kTmemColumns = 512`、stage 字节大小；为参考而引用的 CUTLASS kernel 的 CuTe layout；`OFF_*` 共享内存映射；UMMA 描述符字；屏障计数与掩码；`epi_swz` 与 `cute::Swizzle` 交叉检查 | 第 5 章（常量与映射）、第 3 章（为什么是这些值）、第 4 章（协议常量） |
| 2 | 207-240 | `WaitSite`、`HangRec`、`EXPLICIT_WAIT`、`K_CLC_*_KIND`；开关关闭时只有 `#else` 分支留存（explicit.cu:235-240）：一条注释、`EXPLICIT_WAIT` 宏和两个为零的记录种类（源码横幅注释中的 "two #else lines" 把该宏和这两个种类归为一组） | 第 1 章（图例）、第 5 章（第 5.11 节）、第 7 章（辅助函数中的 `EXPLICIT_WAIT`） |
| 3 | 243-601 | `smem_u32`、W1-W37：身份 (W1-W4)、mbarrier (W5-W13)、fence (W14-W15)、cluster 屏障与命名屏障 (W16-W19)、TMA (W20-W25)、tcgen05 (W26-W32)、CLC (W33-W34)、GDC (W35-W36)、W37 = `__syncwarp()` | 第 6 章 |
| 4 | 604-662 | `advance<Stages>`（流水线状态）、`desc64`；仅用于 trace 的 `mbar_wait_traced`、`heartbeat`、`trace_tail` | 第 7 章 |
| 5 | 665-706 | `clc_consume<TraceKind>`：等待自己的 `clc_full[idx]`、解码、fence、对 CTA 0 的 `clc_empty[idx]` 远程 arrive、推进、把 `(x0, y0)` 映射到 `(tm, tn)` | 第 7 章 |
| 6 | 709-807 | `ProducerState`、`load_ktiles`、`producer_role`（warp 2） | 第 8 章 |
| 7 | 810-868 | `scheduler_role`（rank 0 的 warp 1）：throttle、CLC 装填、`try_cancel`、消费、尾段 | 第 7 章 |
| 8 | 871-975 | `mma_role`（warp 0）：TMEM 分配、描述符、带 4 条 UMMA 和 commit 操作的 k-tile 循环、累加器 commit、尾段、dealloc 握手 | 第 9 章 |
| 9 | 978-1080 | `epilogue_role`（warp 4-7）：TMEM 地址、`st_off[4]`、8 个 subtile 的 `tcgen05.ld` -> FMUL/FFMA -> 16 次 `st.shared` -> fence -> `bar.sync` -> 4 次 TMA 存储 -> commit -> `wait_group.read 1` -> `bar.sync` | 第 10 章 |
| 10 | 1085-1203 | `explicit_blackwell_fp16_gemm_kernel`：身份、标量、tensormap 预取、分五组的 33 次 `mbarrier.init`（前四组每组之后各跟一条 `fence.mbarrier_init.release.cluster`，单独的 `tmem_dealloc` 初始化则不带 fence；第 1.5.1 节）、`barrier.cluster.arrive`、初始 tile、`barrier.cluster.wait`、角色分发 | 第 11 章 |
| 11 | 1206-1329 | `run(ExplicitGemmParams const&)`：守卫、`cudaFuncSetAttribute` x2、`check_cluster_dims`、带一个 cluster 属性的 `cudaLaunchKernelExC`、`cudaGetLastError`；仅用于 trace 的看门狗 | 第 11 章 |
| 12 | 1332-1449 | `trace_begin`、`trace_mismatch_dump`、`trace_end`（仅开关开启时） | 第 1 章（文件集）、第 12 章（测量是如何进行的） |

每个范围从该节的横幅注释行（`////` 分隔线之后的那一行）起，到它的最后一行代码止（对第 3 节而言，到 explicit.cu:601 处的行尾 `W37` 注释）；空行、开头的 `////` 分隔线以及外层命名空间的行（explicit.cu:93, 95, 1082, 1451）不属于任何节。主匿名命名空间横跨 explicit.cu:95-1082（第 1-9 节，kernel 之前的一切）；两个仅用于 trace 的匿名命名空间位于第 11 节（explicit.cu:1213-1262，看门狗）和第 12 节（explicit.cu:1358-1439，`trace_mismatch_dump`）之内；三者都位于 `namespace explicit_gemm` 之内（explicit.cu:93-1451）。

---

## 1.5 三层图景

### 1.5.1 主机端 `run()` -> kernel prologue -> 五种 warp 角色

```
LEVEL 0  HOST            explicit_gemm::run(gemm)                               explicit.hpp:129-132 -> explicit.cu:1265
                         guards (shape, mode, raster, tiles, pointers, beta == 0)
                         cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 230400)
                         cudaFuncSetAttribute(NonPortableClusterSizeAllowed, 1)
                         cudaLaunchKernelExC: grid (64,64,1), block (256,1,1), cluster (2,2,1), smem 230400, stream 0
                         cudaGetLastError
                                                   |
                                                   v
LEVEL 1  PROLOGUE        explicit_blackwell_fp16_gemm_kernel(params)            explicit.cu:1095-1188, all 256 threads of each of the 4096 CTAs
                         smem_base = shared::cta address of the dynamic smem   (0x400 + (rank << 24), measured)
                         warp_idx, lane, lane_pred (elect.sync), rank, is_cta0, is_leader (rank even), peer (rank ^ 1)
                         alpha, beta, k_tiles = (k + 63) >> 6 = 128; prefetch.tensormap A, B (warp 1 of EVERY CTA) and C, D (warp 3 of every CTA)
                         33 mbarrier.init: four fenced groups (warp 0: 8 full + 8 empty mainloop; warp 4: 2 + 2 CLC; warp 5: 4 + 4 accumulator;
                         warp 3: 2 + 2 throttle; one fence.mbarrier_init.release.cluster after each) + the tmem_dealloc init by warp 0 after the
                         fourth fence, with no fence of its own (explicit.cu:1165-1167); 16 + 4 + 8 + 4 + 1 = 33
                         barrier.cluster.arrive.relaxed -> (x, y) = %cluster_ctaid, initial (tm, tn) from blockIdx -> barrier.cluster.wait
                                                   |
                                                   v
LEVEL 2  ROLES           if (warp_idx == 2)            producer_role   explicit.cu:762   TMA A (multicast) + B per k-tile into the 8-stage ring
         (explicit.cu:   else if (warp_idx == 1 && is_cta0) scheduler_role explicit.cu:820   CLC queries + throttle (rank 0 only)
          1190-1202)     else if (warp_idx == 0)       mma_role        explicit.cu:881   TMEM alloc; leader: 4 UMMAs per k-tile + commits; dealloc
                         else if (warp_idx >= 4)       epilogue_role   explicit.cu:995   tcgen05.ld -> alpha/beta -> swizzled smem -> TMA store
                         (warp 3, and warp 1 of ranks 1-3: fall through and exit)
```

分发本身 (explicit.cu:1190-1202) 在第 11.7 节中与 kernel 的结束花括号（第 1203 行）一同被引述。

不带 fence 的 `tmem_dealloc` 初始化 (explicit.cu:1165-1167) 照搬了 CUTLASS kernel，后者在其所有流水线构造函数之后初始化该屏障，并依赖随后的 cluster 会合（`pipeline_init_arrive_relaxed`，`cb424739` 处的 kernel :578-580，附注释 "We need this to guarantee that the Pipeline init is visible to all producers and consumer threadblocks in the cluster"）。

### 1.5.2 一个 CTA：8 个 warp 及其角色

```
 one CTA = one SM: 256 threads = 8 warps; 230400 B dynamic smem (so exactly 1 CTA per SM); allocates all 512 columns of its OWN SM's
 Tensor Memory through one tcgen05.alloc.cta_group::2 issued jointly with its pair peer (one paired allocation, two separate 256 KiB memories)
 +--------+----------------------+--------------------------------------------------------------------------------------------+
 | warp 0 | MMA                  | tcgen05.alloc (both CTAs); pair LEADER only: per k-tile wait full[s], 4 x tcgen05.mma, commit   |
 |        | mma_role             | empty[s] (mask 0xF); per tile commit acc_full[st] (mask 0x3 << 2y); tail; both: dealloc handshake|
 | warp 1 | Sched                | EVERY rank: prefetch tma_a/tma_b (one lane) in the prologue; rank 0 then: throttle wait/release, |
 |        | scheduler_role       | arm clc_full of CTAs 0..3 (lanes 0..3), clusterlaunchcontrol.try_cancel (one lane), consume;    |
 |        |                      | ranks 1-3 then fall through the dispatch and exit                                                |
 | warp 2 | MainloopLoad         | per k-tile wait empty[s]; leader lane arrive.expect_tx 49152 on full[s]; one lane issues TMA A   |
 |        | producer_role        | (multicast 0x5 << x) and TMA B; rank 0 also feeds the throttle; 8 tail waits                     |
 | warp 3 | EpilogueLoad         | prefetch tma_c/tma_d (one lane), init the 4 throttle barriers (one lane), then EXIT: the C       |
 |        | (no role function)   | producer is needed only for beta != 0                                                            |
 | warp 4 | Epilogue w = 0       | TMEM lanes [0, 32): rows 0..31 of the CTA's 128 x 128 tile; ALSO issues all TMA stores of D       |
 | warp 5 | Epilogue w = 1       | TMEM lanes [32, 64): rows 32..63                                                                  |
 | warp 6 | Epilogue w = 2       | TMEM lanes [64, 96): rows 64..95                                                                  |
 | warp 7 | Epilogue w = 3       | TMEM lanes [96, 128): rows 96..127                                                                |
 +--------+----------------------+--------------------------------------------------------------------------------------------+
 named barriers: bar 6 (160 threads = warp 0 + warps 4-7): TMEM address handshake; bar 1 (128 threads = warps 4-7): epilogue subtile sync
```

### 1.5.3 一个 cluster：4 个 CTA、rank、pair、multicast 伙伴

```
 cluster (2,2,1): (x, y) = %cluster_ctaid, rank r = x + 2y (x = r & 1, y = r >> 1)
 cluster tile = 256 (M) x 256 (N) of D; CTA (x, y) owns the 128 x 128 CTA tile (tm, tn) with tm % 2 == x

                     x = 0 (M half 0)                       x = 1 (M half 1)
              +----------------------------------+----------------------------------+
   y = 0      | rank 0  PAIR LEADER              | rank 1  peer of rank 0           |   MMA pair {0, 1}: the leader's warp 0 issues
   (N half 0) | tile (tm, tn)                    | tile (tm + 1, tn)                |   tcgen05.mma.cta_group::2 over M 256 = both
              | rows [128 tm, +128)              | rows [128 tm + 128, +128)        |   CTAs' A stages, N 128 = both CTAs' B halves
              | cols [128 tn, +128)              | cols [128 tn, +128)              |   (columns [128 tn, +128))
              +----------------------------------+----------------------------------+
   y = 1      | rank 2  PAIR LEADER              | rank 3  peer of rank 2           |   MMA pair {2, 3}: same, for
   (N half 1) | tile (tm, tn + 1)                | tile (tm + 1, tn + 1)            |   columns [128 (tn + 1), +128)
              +----------------------------------+----------------------------------+

 A multicast partners (same x, hence same tm, hence the same 128 rows of A):   {0, 2} with mask 0x5 and {1, 3} with mask 0xA (= 0x5 << x)
   each CTA loads ITS OWN 64-row half of the 128 x 64 A stage (rows 128 tm + 64 y .. +64) and the TMA writes it at
   OFF_SMEM_A + 16384 s + 8192 y in BOTH partners, so after both halves land every CTA holds the whole 128 x 64 A stage.
 B is not multicast: each CTA loads its own 64 columns (128 tn + 64 x .. +64) into OFF_SMEM_B + 8192 s; the pair's 2SM MMA
   reads the leader's B half (columns 128 tn .. +64) and the peer's B half (128 tn + 64 .. +64) = the full N 128 of the MMA.
 Barriers that cross CTA boundaries: the leader's full[s] (credited by TMA bytes from all four CTAs, the peer's through the peer-bit mask), every
   CTA's empty[s] (both leaders' tcgen05.commit, mask 0xF), the leader's acc_empty[st] (256 epilogue threads of both CTAs),
   CTA 0's clc_empty[idx] (800 threads of all four CTAs), the pair's acc_full[st] (leader's commit, mask 0x3 << 2y), the
   tmem_dealloc handshake (leader <-> peer), and CTAs 1-3's clc_full[idx] (CTA 0's remote arrive.expect_tx + the multicast CLC response).
```

**启动坐标是经 AlongN 转置的。** CUTLASS 应用 AlongN 转置 `grid = ((ctas_n/2)*2, (ctas_m/2)*2, 1)`（笔记 7.4，第 13 节第 5 项）；util.hpp 的捷径 `N/128 x M/128` 之所以正确，仅仅是因为 M 和 N (8192) 是 256 x 256 cluster tile 的倍数：此时 `ctas_m = ceil_div(M, 256) * 2` 和 `ctas_n = round_up(ceil_div(N, 128), 2)` 都是 64 = 8192/128，而转置后的网格恰好是 (N/128, M/128, 1)。

### 1.5.4 数据路径

```
   global A (fp16, M x K, K-major)        global B (fp16, N x K, K-major)                    global D (fp32, M x N, M-major)
            |                                      |                                                     ^
            | TMA 2SM load, multicast 0x5 << x     | TMA 2SM load (no multicast)                         | TMA store (warp 4): 4 boxes of
            | box (64 k, 64 rows) = 8192 B         | box (64 k, 64 rows) = 8192 B                        | (32 m, 16 n) = 2048 B each per subtile,
            | completes on the LEADER's full[s]    | completes on the LEADER's full[s]                   | commit_group, wait_group.read 1
            v                                      v                                                     |
   +------------------------------------------------------------------+     +----------------------------------------------------+
   | smem A ring  OFF_SMEM_A = 33792: 8 stages x 16384 B (128 x 64)   |     | smem C/D  OFF_SMEM_CD = 512: 4 stages x 8192 B      |
   | smem B ring  OFF_SMEM_B = 164864: 8 stages x  8192 B ( 64 x 64)  |     | (128 x 16 fp32), Swizzle<2,5,2>, stage = e % 4      |
   | both Swizzle<3,4,3> (SWIZZLE_128B), written by the TMA unit       |     | written by 16 st.shared.b32 per lane per subtile    |
   +------------------------------------------------------------------+     +----------------------------------------------------+
            |                                                                                            ^
            | tcgen05.mma.cta_group::2.kind::f16 (pair leader, one elected lane): descriptors name       | d = fmaf(beta, 0.0f, alpha * acc)
            | the A stage (both CTAs, 256 rows) and the B stage (both halves, 128 columns), K 16;        | (FMUL + FFMA per value, run-time
            | 4 per k-tile (kb = 0..3, +32 B each); first of a tile with scale_c = 0, then accumulate    |  alpha and beta), 16 values per lane
            v                                                                                            |
   +----------------------------------------------------+   tcgen05.ld.sync.aligned.32x32b.x16.b32   +------------------------------+
   | TMEM: 128 lanes (rows) x 512 columns, base T = 0   | -----------------------------------------> | registers: 16 x fp32 per lane |
   | accumulator stage st = columns [128 st, +128)      |   warp w reads lanes 32 w .. 32 w + 31,      | lane l of warp w holds row    |
   | = one 128 x 128 fp32 tile; 4 stages = 512 columns  |   columns 128 st + 16 e .. +15 (subtile e)   | 32 w + l, columns 16 e .. +15 |
   +----------------------------------------------------+                                              +------------------------------+
```

---

## 1.6 关于 NUMBERS AT A GLANCE 块的说明 (explicit.cu:52-70)

**problem.** 硬件中没有任何东西是为 TN 设计的（CUTLASS 对 K-major A 配 K-major B 的 BLAS 风格命名，第 3.1.1 节）：`media/docs/cpp/blackwell_functionality.md:15` 将 TN、NT、TT 和 NN 列为对 `kind::f16` 均受支持，而 `include/cute/arch/mma_sm100_desc.hpp` 为每一种 swizzle 模式给出了规范的 K-major (:724-727) 与 MN-major (:701-705) 描述符 layout（除 `SWIZZLE_128B_BASE32B` 之外全为 K-major，:811）。

**cluster.** 该 explicit kernel 保留了 CUTLASS 中由偶数 rank 领导的约定，因为另外两种机制都与 pair 的 parity 绑定：peer 位屏障寻址，它只能指名一个 pair 中的偶数 CTA（第 2 章，第 2.9.3 节），以及 `cta_group::2` TMA 完成规则，它根据持有 mbarrier 的 CTA 的 parity 向每个 pair 中的偶数或奇数 CTA 发出信号（第 2 章，第 2.7.4 节）。

**smem.** 基址 `0x400 + (rank << 24)` 是实测的，从未假定 (explicit.cu:1098)；保持它要求翻译单元中任何地方都不得有静态 `__shared__`，而且该 kernel 还镜像了 CUTLASS 的 `__launch_bounds__(256, 1)` 和不带对齐属性的 `extern __shared__ char smem[]` (explicit.cu:1085-1086, 1095-1097)。

---

## 1.7 如何使用本讲义

### 1.7.1 三遍阅读

- **自顶向下（第 2-4 章）。** 这三章之后，你应当能够凭记忆画出第 1.5.3 节和第 1.5.4 节，并说出每个箭头为何存在。
- **自底向上（第 5-11 章）。** 文件的第 1-11 节，按章划分：常量与 tensor map (5)，37 个 wrapper 的逐字符解读 (6)，流水线辅助函数、`clc_consume` 与 scheduler warp (7)，producer (8)，MMA warp (9)，epilogue (10)，kernel prologue 与主机端镜像 (11)。第 1-11 节中位于 `[TRACE]` 块之外的每一行代码都在这些章的某处被引述并解释；第 12 节，即仅用于 trace 的主机端入口点，属于第 12 章（第 1.4.1 节）。
- **精通（第 12-14 章）。** ptxas 生成了什么，以及测得了什么 (12)；如何以一种让 kernel 在每一步都保持可调试的构建顺序自己动手写出它、各个陷阱，以及一份面向其他算子的 speed-of-light 配方 (13)；练习与术语表 (14)。

### 1.7.2 先修知识

你需要能熟练使用 CUDA C++（网格、线程块、warp、共享内存、`__shfl_sync`、带有 `"r"`、`"l"`、`"h"`、`"n"` 约束的内联 `asm volatile`），并知道 GEMM tile 循环是什么。你**不**需要事先具备 Blackwell 知识：TMEM、`tcgen05`、2SM MMA、TMA multicast 和 CLC 在第 2 章引入，并在其后被具体地使用。Hopper 时代的概念（TMA、`mbarrier`、warp 特化、cluster、`elect.sync`）会让第 2 章和第 4 章读起来更快，但它们也同样被涵盖在内。

---

## 1.8 要记在脑中的事实

1. **代码纪律。** 在 kernel TU 中，CuTe 仅出现在编译期 swizzle 交叉检查中（一个 `cute::Swizzle` 别名和若干 `static_assert`，explicit.cu:198-204），以及作为 `cute::get` 出现在仅主机端的头文件桥中。
2. **测得了什么。** 六次运行中 `D` 逐值相等（每次都是同一个带种子的整数输入，精确比较）。

---

# 第 2 章 机器：面向 GEMM 编写者的 Blackwell SM100

本章中的每一条陈述都被标记为三类之一：**实测**（笔记的 D 部分和 G 部分：trace 记录、SASS、`cudaOccupancyMaxActiveClusters`）、**来源**（PTX ISA 9.4、CUDA Programming Guide 13.4、CUDA driver API、提交 cb424739 处的 CUTLASS 头文件、explicit.cu 本身）或 **分析**（我的推理或假设）。凡我无法佐证的内容都标为“（未验证）”；凡依赖 NVIDIA 内部信息而非公开文档的内容（例如第 2.1.1 节的 B200 TPC 数与每 GPC 的 TPC 数、第 2.1.2 节的 `block_rank()` 线性化、Blackwell 的 uniform 寄存器数、第 2.5.3 节的 `cta_group::2` 语义以及第 2.10.2 节的屏障 0 规则）都标为“（NVIDIA 内部）”。

---

## 2.1 硅片层级：GPU、GPC、TPC、SM、cluster、pair

### 2.1.1 各层级与数字

| 层级 | 运行该 kernel 的那块 B200 上的数量 | 来源 | 本 kernel 用它做什么 |
|---|---|---|---|
| GPU 封装 | 由 NV-HBI 连接的 2 个裸片 | nvidia.com 的 Blackwell 架构页面："two reticle-limited dies connected by a 10 terabytes per second (TB/s) chip-to-chip interconnect"；这个名称是 NVIDIA 的："NVIDIA High-Bandwidth Interface (NV-HBI)"（developer.nvidia.com，Blackwell Ultra 博客），而对于 B200，"MLOPart devices on the same B200 chip ... can instead use the much faster NV-HBI"（developer.nvidia.com，CUDA MPS 博客）；Cornell CVW 的 B200 页面："a pair of silicon chiplets or dies joined along one edge by a 10 TB/s link" | 没有任何显式的处理：网格由硬件跨两个裸片调度 |
| SM（streaming multiprocessor） | 148 个已启用（实测，笔记第 0 节与 D.3）；每个封装物理上有 160 个（第三方数字，未对照 NVIDIA 文档验证）：chipsandcheese "Each B200 die physically contains 80 Streaming Multiprocessors (SMs)"、"B200 enables 74 SMs per die, giving 148 SMs across the GPU" | 实测 + chipsandcheese | 每个 SM 一个 CTA（230400 B 的 smem 不允许两个），132 个 SM 忙碌，16 个空闲（实测，D.3） |
| TPC（texture processing cluster） | 每个 TPC 2 个 SM：NVIDIA RTX Blackwell 白皮书 "Each TPC includes one PolyMorph Engine and two SMs"（消费级裸片）；B200 同样是每个 TPC 2 个 SM，148 个 SM = 74 个 TPC（NVIDIA 内部），且实测的放置与之相符 | 来源（白皮书，经由搜索）+ NVIDIA 内部 + 实测放置 | **MMA pair** `{2y, 2y+1}` 总是落在 SM `(2j, 2j+1)` 上（实测，D.3） |
| GPC（GPU processing cluster） | Cornell CVW："each of the 8 GPCs above contains exactly 20 SMs (in 10 TPCs)"，针对完整的 160-SM 封装（第三方页面，据其所述，其图示展示的是 B300 "Ultra"；NVIDIA 声明 Blackwell Ultra 在八个 GPC 中有 160 个 SM，developer.nvidia.com；对 B200 未验证）。160-SM 的封装数字与随后的已启用 skyline 并不必然冲突：SM 或 TPC 可能默认被隐藏，从而一个 kernel 只能使用封装的一部分（NVIDIA 内部；作为一种可能性陈述）；八个 GPC 的封装计数并非这块 B200 的，其下文所示的已启用 skyline 跨越 11 个 GPC；在该下界之外，没有任何来源给出此封装的物理 GPC 数量。B200 器件出厂时带有两种 skyline 之一，即每个 GPC 启用的 TPC 数：10/10/9/9/9/9/9/9（8 个 GPC）或 10/10/10/9/9/9/9/5/1/1/1（11 个 GPC），两者都是 74 个 TPC = 148 个 SM；此处所用的 B200 是 11-GPC 的那一种（两者均为 NVIDIA 内部）。在一个 GPC 内每个 cluster 占两个完整 TPC 的条件下，11-GPC 的 skyline 给出 33 个 cluster 和 8 个空闲 TPC，正是实测值；8-GPC 的变体则会给出 34 和 6（我的算术）。实测的驻留 cluster 的 SM-id 组为 0-31、32-63、66-93、94-121、130-141 (D.3) | Cornell CVW（每封装）、NVIDIA 内部（skyline）、实测分组 | **一个 cluster 必须容纳于一个 GPC 之内**（CUDA Programming Guide："all thread blocks in a cluster are executed in a single GPC"） |
| Cluster | (2,2,1) = 4 个 CTA；可移植的最大值为 8，"NVIDIA Blackwell B200 GPU allows for a nonportable cluster size of 16"（Blackwell Tuning Guide） | explicit.cu:1284, 1305-1308 (`kClusterX/Y/Z` = 2, 2, 1, util.hpp:31)，Tuning Guide | 4 个 CTA = 2 个 MMA pair；A 跨两个 pair 进行 multicast |
| CTA pair | 仅在第 0 位上不同的 rank | PTX ISA 9.4 第 9.7.18.5.1 节 | 两个 CTA 每个 k-block 共用一条 `tcgen05.mma.cta_group::2` |
| 驻留 cluster | 33（= `cudaOccupancyMaxActiveClusters`，实测 D9/D.3），而不是朴素的 148/4 = 37 | 实测 | 33 个持久化工作者通过 CLC 拉取 1024 个 cluster tile |

为什么是 33 而不是 37。这个 33 仅依据测量：`cudaOccupancyMaxActiveClusters` 对此启动配置返回了 33（笔记 D4、D9），而 trace 证实了 132 个 SM 上有 33 个原生启动的 cluster，且 16 个 SM (64, 65, 122-129, 142-147) 在整个启动期间空闲 (D.3)。结合表中的实测放置（每个 MMA pair 位于一个 TPC `(2j, 2j+1)` 上，一个 cluster 的两个 pair 位于同一个 SM-id 组中），共使用了 66 个 TPC，而 8 个 TPC（64-65 处一个、122-129 处四个、142-147 处三个）从未被使用。算术如下：148 个 SM = 74 个 TPC；132 个忙碌的 SM = 66 个 TPC = 33 个 cluster x 2 个 TPC；16 个空闲的 SM = 8 个 TPC。笔记的解释（D.3，在那里标为 "Inferred"）是：一个 2 x 2 cluster 需要同一个 GPC 内的两个完整 TPC，而剩余的 TPC 无法在各自所属的那一个之内配对。GPC skyline，即每个 GPC 启用的 TPC 数，将其补全。这块 B200 每个 TPC 有 2 个 SM，148 个 SM = 74 个 TPC，skyline 为 10/10/10/9/9/9/9/5/1/1/1：11 个 GPC，合计 74 个 TPC（NVIDIA 内部）。推导：一个 (2,2,1) cluster 占据一个 GPC 的两个完整 TPC，所以拥有 t 个 TPC 的 GPC 容纳 floor(t / 2) 个 cluster 并空闲 t mod 2 个 TPC；每个奇数大小的 GPC 空闲一个 TPC = 2 个 SM。这一 skyline 空闲 4（四个 9）+ 1（那个 5）+ 3（三个 1）= 8 个 TPC = 16 个 SM，并在 132 个 SM 上容纳 5 + 5 + 5 + 4 + 4 + 4 + 4 + 2 = 33 个 cluster，与测量结果完全一致；另一种 B200 skyline，10/10/9/9/9/9/9/9（8 个 GPC，同样 74 个 TPC；NVIDIA 内部），则会空闲 6 个 TPC = 12 个 SM，并在 136 个 SM 上容纳 34 个 cluster。因此测量结果只与 11-GPC 的 skyline 相符；这块 B200 具有该 skyline 一事属于 NVIDIA 内部信息，与实测的 132 个忙碌 SM 一致。还有两点需要注意。第一，在这一 skyline 中一个 GPC 至多有 10 个 TPC = 20 个 SM，所以一个 32-SM 的组（16 个 TPC）不可能是单个 GPC；这些 SM-id 组是硬件中某种没有公开文档记载的放置单元，而哪些 SM id 属于哪个 GPC 同样没有文档记载，我也无法以 NVIDIA 内部信息的形式获得；上面的推导使用的是 skyline，而不是那张映射。第二，8 个空闲 TPC 正是 8 个奇数大小的 GPC 各自无法用于一个 (2,2,1) cluster 的那一个 TPC（NVIDIA 内部），所以 SM 122-129 处四个连续的空闲 TPC 属于四个不同的 GPC；相邻的 SM id 并不意味着同一个 GPC。

pair 的 PTX 定义是整个 kernel 几何结构赖以成立的那一个事实（PTX ISA 9.4，9.7.18.5.1，引文如下）：

> Any 2 CTAs within the cluster whose `%cluster_ctarank` differs by the last bit only is said to form a CTA pair. Within a CTA pair, the CTA whose last bit in the `%cluster_ctarank` is: 0 is termed the even numbered CTA within the CTA pair. 1 is termed as the odd numbered CTA within the CTA pair.

以及 (9.7.18.5.2)："The peer CTA of the odd CTA within the CTA pair is the even CTA in the same pair. Similarly, the peer CTA of the even CTA within the CTA pair is the odd CTA in the same pair."

### 2.1.2 rank、坐标、pair：本 kernel 的 2 x 2 cluster

`%cluster_ctaid` 是 "the CTA identifier in a cluster in each dimension"，而 `%cluster_ctarank` 是 "the CTA rank within a cluster across all dimensions"（PTX ISA 9.4，10.14 与 10.16）。对于 (2,2,1) cluster，rank 是线性化 `r = x + 2y`，其中 `x` 变化最快。PTX ISA 并未定义这一线性化（10.16 只说 "CTA rank within a cluster across all dimensions"）。它是 `cluster.block_rank()` 所遵循的线性化，`r = x + nx y + nx ny z`（NVIDIA 内部）；Programming Guide 对 `block_rank()` 的定义仅为 "Rank of the calling block within [0, num_blocks)"，并未给出公式。CUTLASS 在 `calculate_umma_peer_mask` (cutlass/pipeline/sm100_pipeline.hpp:96-110) 中用 column-major 的 `make_layout(cluster_shape)` 复现了它，而 kernel 自己的注释将其表述为 `x == rank & 1`、`y == rank >> 1` (explicit.cu:1173-1174)。D8 中每个 rank 的 TMA 坐标证实了这一点：rank 1 从 128 开始加载 A 的行 (`x = 1, y = 0`)，rank 2 从 64 开始 (`x = 0, y = 1`)，rank 3 从 192 开始 (`x = 1, y = 1`)，恰好是 `128 x + 64 y`。D5 的基址 `0x00000400`、`0x01000400`、`0x02000400`、`0x03000400` 仅表明基址是按 rank 索引的（第 2.9.2 节）。pair `{2y, 2y+1}` 与 `peer = rank ^ 1` (explicit.cu:1107) 依赖于这一约定。kernel 在 explicit.cu:1104 和 :1173-1174 读取全部三个特殊寄存器，并在 :1106-1107 推导出 pair 角色（引文见第 11.3 节）。

```
            x = 0  (%cluster_ctaid.x)         x = 1
         +------------------------------+------------------------------+
 y = 0   | rank 0   PAIR LEADER         | rank 1   peer of 0           |  pair {0,1}: one 2SM MMA of
         | CTA tile (tm = 2TM,   tn=2TN)| CTA tile (tm = 2TM+1, tn=2TN)|  M 256 (both tm) x N 128 (tn = 2TN)
         +------------------------------+------------------------------+
 y = 1   | rank 2   PAIR LEADER         | rank 3   peer of 2           |  pair {2,3}: M 256 x N 128 (tn = 2TN+1)
         | CTA tile (tm = 2TM,  tn=2TN+1)| CTA tile (tm=2TM+1, tn=2TN+1)|
         +------------------------------+------------------------------+
   A rows of tm = 2TM   are needed by ranks 0 and 2  -> A multicast mask 0x5 (x = 0)
   A rows of tm = 2TM+1 are needed by ranks 1 and 3  -> A multicast mask 0xA (x = 1)
   B columns of tn = 2TN   are needed by ranks 0 and 1 (each loads its own 64-column half, the 2SM MMA reads both)
   B columns of tn = 2TN+1 are needed by ranks 2 and 3
   rank 0 = "CTA 0": the only CTA whose scheduler warp runs; owner of the CLC empty barriers
```

这里 `(TM, TN)` 是 256 x 256 的 cluster tile；`x` 选择 M 半部 (`tm % 2 == x`)，`y` 选择 N 半部。因此 A 的 multicast 掩码 `0x5 << x` (explicit.cu:772) 指定的是 `x` 相同的两个 CTA，而累加器 commit 掩码 `0x3 << 2y` (explicit.cu:902) 指定的是 `y` 相同的两个 CTA。第 3 章解释设计为何需要这一取向。本章只需要其背后的两个硬件事实：按 PTX 的定义，pair 是 `{2y, 2y+1}`，而 multicast 是按 rank 位寻址的。

### 2.1.3 cluster 落在了哪里（实测）

笔记 D.3 为每个 CTA 记录了 `%smid`。每个 pair `{rank 0, rank 1}` 和 `{rank 2, rank 3}` 都位于 SM `(2j, 2j+1)` 上；一个 cluster 的两个 pair 位于同一个 SM-id 组中；并且在两次独立的启动中，前四个 cluster id 落在了 SM 0,1,16,17 / 2,3,18,19 / 4,5,20,21 / 6,7,22,23 上。分析：一个 TPC 的两个 SM 共享一条互连结构，`tcgen05.mma.cta_group::2` 在每条指令期间都经由它读取 peer 的共享内存并写入 peer 的 TMEM，所以硬件把一个 pair 放在一个 TPC 上。GPC 决定 cluster 的第二个 pair 落在哪里，这里对 cluster id 0-15 是相隔 16 个 SM（id 16-29 为 14，id 30-32 为 6；D.3），而这一选择对 kernel 没有任何代价：跨 pair 的流量是 A 的 multicast（掩码 `0x5 << x`，explicit.cu:772）、mainloop release 对全部四个 CTA 的 `empty[s]` 所做的 commit（掩码 `0xF`，explicit.cu:186、:942）、CLC 的装填、响应与释放到达 (explicit.cu:842, :847, :684) 以及那一次启动时的 `barrier.cluster` (explicit.cu:1170, :1179)：都是屏障信号和异步的 bulk 写入，没有一项位于 MMA 的每指令操作数路径上。

---

## 2.2 带数字的内存层级

| 存储器 | 容量 | 来源 | 本 kernel 如何使用它 |
|---|---|---|---|
| 向量寄存器 | 每个 SM 64 K x 32-bit = 256 KiB；每线程最多 255 个 | CUDA Programming Guide，compute-capabilities 表（CC 10.0："Number of 32-bit registers per SM: 64 K"、"Maximum number of 32-bit registers per thread: 255"） | 在 `__launch_bounds__(256, 1)` 之下每线程 26 个寄存器，0 次溢出（实测 G.3）；那个 `1` 不花任何代价，因为 230400 B 的动态 smem 已经把 occupancy 钉在每个 SM 一个 CTA，所以 CC 10.0 对驻留块与线程的每 SM 上限在这里并不构成约束（我没有在此环境中验证它们的值） |
| uniform 寄存器 | 在引入它们的 Turing 上，每个 warp 有 63 个通用的 `UR0`-`UR62` + `URZ`；在 Blackwell 上，每个 warp 有 80 x 32-bit 可寻址的 uniform 寄存器 `UR0`-`UR79` = 每个 warp 320 B，在 16 个 warp 时每个 SM 子分区 5 KB，其中 `URZ` 是 80 个之外的一个单独的恒零寄存器（NVIDIA 内部），所以 Turing 的 64 计入了 `URZ` 而 Blackwell 的 80 没有；baseline 的 Blackwell SASS 最高用到 `UR77`，与此一致 | Jia 等人，arXiv:1903.07486（Turing）；NVIDIA 内部（Blackwell） | 描述符、屏障地址、掩码、循环状态都存放于此（实测 D.2、G.3） |
| 共享内存 | 每个 SM 228 KB；每个块 227 KB = 232448 B（可选启用）；"CUDA reserves 1 KB of shared memory per thread block" | Blackwell Tuning Guide 13.4；util.hpp:46 断言 `kSmemBytes <= 232448` | 230400 B 动态 = 225 KiB，比可选启用的最大值低 2048 B（笔记 6.2） |
| 统一的 L1 / 纹理 / 共享 | 每个 SM 256 KB | Tuning Guide："the same the maximum capacity of the combined L1 cache, texture cache, and shared memory of 256 KB" | 划出 228 KB 的 smem 之后，剩下 28 KB 的 L1（分析：256 - 228） |
| 张量内存（TMEM） | 128 个 lane x 512 列 x 32 位 = 每个 SM 256 KiB（在 sm_100a 上即每个 CTA） | PTX ISA 9.4，9.7.18.1 | 每个 CTA pair 分配了全部 512 列：四个 128 x 128 的 fp32 累加器 stage |
| L2 | 每个封装 126 MB | Tuning Guide（"The NVIDIA GB200 GPU increases the L2 cache capacity to 126 MB"）；chipsandcheese 对 B200 的 "126 MB of total capacity" | A (128 MiB) + B (128 MiB) 放不下；AlongN 光栅 + CLC 分发把 working set 保持在 A 的两到三个 256 行条带（实测 D.3） |
| HBM3e | 180 GB；7.7 TB/s（Lenovo 产品指南表格），市场宣传为每 GPU 8 TB/s（NVIDIA 的 DGX B200 页面：八个 GPU 的 "64 TB/s HBM3e bandwidth"） | Lenovo LP2226 | D 为 256 MiB，只写一次；A 从 L2 重读（两到三个条带驻留，实测 D.3）；B 是否在条带之间存活于 L2 中尚无定论（第 3.9.5 节） |
| Tensor 吞吐量 | 每个 B200 FP16/BF16 稠密 2.25 PFLOPS（Lenovo LP2226 表格，NVIDIA 数据手册数值） | Lenovo LP2226 | 分析：2.25e15 / (148 x 1.965e9) = 每时钟每 SM 7737 FLOP，为第 3 章所用的 8192 FLOP/clk/SM (4096 FMA x 2 FLOP) 稠密 FP16 速率的 94%。因此数据手册的数字对应于 2.25e15 / (148 x 8192) = 1.856 GHz 的时钟，低于此处实测的 1965 MHz。所以第 3 章从实测时钟下的指令速率 (148 x 8192 x 1.965e9 = 2.382 PFLOPS) 推导 speed-of-light 界，而不是从这个数字 |

共享内存那一行的算术核对：228 KiB = 233472 B；减去保留的 KiB = 232448 B = 227 KiB，即 harness 与 util.hpp:46 所假定的可选启用最大值。kernel 的 230400 B = 225 KiB 留下 2048 B 未用。stage 数是 8 而不是 9，因为第九个 stage 放不下（笔记 6.1）。在 CUTLASS builder 的核算中，预算是 232448 - 192 - 33792 = 198464 B，而一个 stage 花费 24592 B = 16384 A + 8192 B + 16 B 的流水线屏障；8 个 stage 占用 196736 B 并留下 1728 B，所以第九个 stage 还差 24592 - 1728 = 22864 B。在 explicit kernel 的实际 layout 中，同一结论写作 230400 + 24576 = 254976 B，即比 232448 字节的可选启用最大值超出 22528 B。第 3 章会回到 stage 数。

**保留的千字节在哪里（实测，D5 与 D.2）。** rank `r` 的动态共享内存缓冲区始于 32 位的 `shared::cta` 地址 `0x400 + (r << 24)`：低 24 位是 `0x400` = 1024 = 位于用户共享内存之前的系统保留千字节，而 `cuobjdump` 对二进制中的每个 kernel 都将其报告为 `SHARED:1024`。`tcgen05.alloc` 与 `tcgen05.dealloc` 的下沉实现使用该千字节中的若干字 (`(rank << 24) + 0x40..0x5c`) 来完成 alloc 的 pair 握手以及 dealloc 由 ptxas 插入的一致性检查（实测 D.2）。explicit.cu 从不假定这个值：它在运行时计算 `smem_base` (explicit.cu:1098)。它确实断言实测的基址使 14 位的描述符字段保持在范围之内（explicit.cu:165；该断言展示于第 2.6.1 节，并在第 2.9.4 节回顾）：

```
  uint32_t const smem_base = smem_u32(smem);                                                   // 0x400 + (rank << 24) measured (D.1 D5); computed, never assumed
```

**这 230400 字节是如何花掉的**（笔记 6.2；常量在 explicit.cu:139-156）：屏障、CLC 响应槽位和 TMEM 基址指针位于最前面的 512 B，四个 8192 字节的 C/D epilogue stage 位于 `OFF_SMEM_CD = 512`，八个 16384 字节的 A stage 位于 `OFF_SMEM_A = 33792`，八个 8192 字节的 B stage 位于 `OFF_SMEM_B = 164864`。每个 A/B stage 基址按 1024 对齐，每个 C/D stage 基址按 512 对齐 (explicit.cu:162-163)，原因是第 2.7.3 节所解释的 swizzle 周期。第 5 章将逐字节走读这张映射。

---

## 2.3 张量内存（TMEM）

### 2.3.1 几何与地址

PTX ISA 9.4，9.7.18.1："On architecture sm_100a/sm_100f, the 5th generation TensorCore's Tensor Memory has a two-dimensional structure of 512 columns and 128 rows per CTA, with each cell being 32-bits in size." 行被称为 *lane*。每个 SM 上 128 x 512 x 4 B = 262144 B = 256 KiB，与寄存器堆的大小相同。

寻址（9.7.18.1.1）："Tensor Memory addresses are 32-bit wide and specify two components. Lane index. Column index." 其中 lane 索引占据位 [16,32)，列索引占据位 [0,16)。CuTe 以掩码的形式陈述了同一件事（`cute/pointer.hpp:314-316`）："TMEM 'Address' with active mask 0x007F.01FF: the upper 16 bits, the 0x007F portion, refers to the 128 DP lanes; the lower 16 bits, the 0x01FF portion, refers to the 512 COL lanes." Colfax 的 TMEM 教程说法相同：TMEM 是 "256KB per SM in size, and is organized 2-dimensionally in 512 columns and 128 rows, or lanes, of 32-bit cells"，并且 "bits 31-16 denote the lane ID while 15-0 denote the column"。因此一个 lane 的步长是 `1 << 16 = 65536`，`cute/arch/tmem_allocator_sm100.hpp:45-53` 将其编码为 `DP = 1 << 16`（以 32 位为单位）；笔记（第 6.3 节）引用了同一个头文件，而 Colfax 的教程把它表述为 "bits 31-16 denote the lane ID while 15-0 denote the column"。

```
  31              16 15               0
  +-----------------+-----------------+
  |   lane (0..127) |  column (0..511)|      address = (lane << 16) | column
  +-----------------+-----------------+
  a 128 x 128 fp32 tile occupies 128 lanes x 128 columns = 64 KiB = one quarter of TMEM
```

### 2.3.2 分配

PTX ISA 9.4，9.7.18.1.2："The Tensor Memory is dynamically allocated. The Tensor Memory must be allocated by a single warp in a CTA ... The unit of allocation is 32 columns and the number of columns being allocated must be a power of 2. When a column is allocated, all 128 lanes of the column are allocated. All of the Tensor Memory that was allocated in a kernel, must be explicitly deallocated before the kernel exits."

该指令（9.7.18.7.1）：`tcgen05.alloc` "is a blocking instruction which dynamically allocates the specified number of columns in the Tensor Memory and writes the address of the allocated Tensor Memory into shared memory at the location specified by address operand dst. The tcgen05.alloc blocks if the requested amount of Tensor Memory is not available"。非独占的 `nCols` "must be a power of 2, in the range [32, 512]"。对于 `.cta_group::2`："one warp from each of the peer CTAs must collectively perform the allocation and de-allocation"、"the issuing warp must make sure that peer CTA is launched and its warps eventually participate in collective operations"，而对于 `tcgen05.dealloc` 则是 "issuing warp and peer CTA warp must synchronize Tensor Memory accesses before attempting to collectively deallocate the Tensor Memory"。`tcgen05.relinquish_alloc_permit` "specifies that the CTA of the executing thread is relinquishing the right to allocate Tensor Memory"，也就是说此后再次 `alloc` 是非法的；它并不释放任何张量内存，只释放该 CTA 分配张量内存的许可（NVIDIA 内部；它所释放的调度资源，即 Virtual Resource Counter，在第 9.6.2 节中描述）。CUTLASS 补充了一条实用规则 "The 2 warps from participating CTAs have the same logical warp ID"（`cute/arch/tmem_allocator_sm100.hpp:124-133`）。

在 explicit.cu 中，这是两个 CTA 的 warp 0（explicit.cu:885-888，引文见第 9.4 节）。

该调用传入 `kTmemColumns = 512`（explicit.cu:114）：即整个 TMEM，因为反正每个 SM 上只驻留一个 CTA（第 2.2 节），而且累加器流水线需要四个 128 列的 stage（第 3 章）。结果落在**两个** CTA 的 `OFF_TMEM_BASE_PTR = 384` 中（"Both CTAs _must_ provide the exact same dst_ptr"，CUTLASS :128）。实测的基址在全部 264 条 K2 记录中都是 `T = 0`（D6），正如一个 SM 上唯一的分配所应有的那样，但 kernel 是读取它而不是假定它。实测的下沉结果（D.2）：分配原子操作 `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN` 在 leader 被选中的 lane 中以 `0x10` 个 32 列单位执行，结果通过保留的那个千字节中的一次握手转发给 peer；`dealloc` 在两个 CTA 中都是 `UTCATOMSWS.AND`。PTX 指令仍然是你所书写的契约。

9.7.18.7.1 中还有三句话解释了分配代码（explicit.cu:885-888）的细节。(i) "The number of columns allocated should not increase between any two allocations in the execution order within the CTA"：这是在单条 `tcgen05.alloc` 中一次分配全部 512 列、而不是一个 stage 一个 stage 地增长分配的一个原因。(ii) "The store operation in tcgen05.alloc is treated as a weak memory operation in the Memory Consistency Model"：该指令本身并不把写入 `OFF_TMEM_BASE_PTR` 的地址发布给其他 lane 或 warp，这就是为什么（我的分析，第 9.4.2 节）warp 0 在读取它之前执行 `__syncwarp()`（explicit.cu:886-888），也是为什么 epilogue warp 只在 `bar.sync 6, 160` 之后才读取它（explicit.cu:997-998）。(iii) "The behavior of the instruction is undefined if all the threads in the warp do not use the same values of nCols, or if any thread in the warp has exited"：`kTmemColumns` 是编译期常量，而且 warp 0 在 alloc 和 dealloc 时都是完整的。

满足 "synchronize Tensor Memory accesses before ... deallocate" 规则的退出序列，是位于 explicit.cu:950-971 的、在 `OFF_TMEM_DEALLOC`（计数 32）上进行的 pair 握手：`griddep_launch_dependents()` 和 `tmem_relinquish_2sm()`，leader 的四次累加器尾段等待（两个 CTA 中 epilogue 对 TMEM 的每一次读取都已被释放），然后是在 `OFF_TMEM_DEALLOC` 上的两次远程 arrive 以及集体 dealloc（explicit.cu:963-971，引文见第 9.6 节）。握手中仅 peer 执行的部分，即它的远程 arrive，只有一行（explicit.cu:964）；:965 处的等待由两个 CTA 执行：

```
  if (!is_leader) { mbar_arrive_remote(smem_base + OFF_TMEM_DEALLOC, peer); }
```

第 9 章追踪这一顺序；这里，硬件上的原因就足够了。peer 不得在 leader 观察到两个 epilogue 都已读完 TMEM 之前解除分配。PTX 中关于非确定性挂起的表（9.7.18.5，表 57）确切地展示了当一侧在另一侧同步之前发射 `.2cta` dealloc 时会出什么问题。表 56 周围的文字给出了握手所依赖的两条保证："Before attempting to deallocate Tensor Memory, it suffices to ensure that there are no concurrent Tensor Memory accesses from the Peer CTA" 和 "Warps can immediately exit after deallocating Tensor Memory; no extra synchronization required"。因此 `tmem_dealloc_2sm` 是 warp 0 的最后一条语句（explicit.cu:971），其后再无任何内容。表 55 中 `.alloc`/`.dealloc` 配合 `cta_group::2` 的那一行只说 "the first warp to perform the operation could block until the second warp in the Peer CTA also performs the operation"。这正是握手之所以存在的原因：除程序自己建立的顺序之外，ISA 并不给出 peer 的最后几次 TMEM 读取与 leader 的 dealloc 之间的任何顺序，所以 `OFF_TMEM_DEALLOC` 交换（explicit.cu:963-970）在任一侧发射集体 dealloc 之前建立了这一顺序。

### 2.3.3 warp 象限规则

ISA（9.7.17.1，"Warpgroup"）把 *warpgroup* 定义为 "a set of four contiguous warps such that the warp-rank of the first warp is a multiple of 4"：即 warp 4k..4k+3，warpgroup 索引为 `warp_idx / 4`。在本 kernel 中，warp 0-3 构成 warpgroup 0，四个 epilogue warp 4-7 构成 warpgroup 1。PTX ISA 9.4，9.7.18.8.1（"Access restrictions"，位于 tcgen05 ld/st 指令之下）："Not all threads of the CTA can access the entire Tensor Memory via the tcgen05.ld and tcgen05.st operations. The Tensor Memory of a CTA is divided into 4 equal chunks such that each warp of a warpgroup in the CTA can access a chunk of the Tensor Memory. All the columns of the Tensor Memory can be accessed by all the four warps of a warpgroup. A lane of the Tensor Memory can be accessed by a single warp in the warpgroup." 该表：warpgroup 的 warp 0 访问 lane 0-31，warp 1 访问 lane 32-63，warp 2 访问 lane 64-95，warp 3 访问 lane 96-127；发射粒度表补充道 "Issue from a warp in the current CTA can access only 1/4 of the Tensor Memory of the current CTA. So, a warpgroup is needed to access the entire Tensor Memory of the current CTA."

为什么（分析，并得到 chipsandcheese 的推断 "That implies each SM sub-partition has a 512 column by 32 row TMEM partition" 和 "Each wave can only access 32 TMEM rows" 的支持）：TMEM 在物理上是四个 32-lane 的切片，每个 SM 子分区（warp scheduler）一个，而一个 warp 只能读取它所运行的那个子分区的切片。warpgroup 内的 warp id（`warp_idx % 4`）选择子分区，所以规则是 "warp `w` of the warpgroup sees lanes `[32 (w % 4), 32 (w % 4) + 32)`"。它对*每一种* `tcgen05.ld`/`tcgen05.st` shape 都成立。笔记 B.6、MLC 的 TMEM 章节中的表格以及 Colfax 的教程（"Each warp within a warpgroup can only access 32 lanes (with warp 0 associated to lanes 0-31, warp 1 to lanes 32-63, and so forth)"）都重述了它。

这就是为什么 epilogue 恰好是四个 warp（4-7，一个 warpgroup），以及为什么线程 `(w, l)` 拥有输出行 `32 w + l`：explicit.cu:1006 与 :1031

```
  uint32_t const tmem_lane_off = (32u * w) << 16;                                               // TMEM lanes [32 w, 32 w + 32): row 32 w + l of the CTA tile (0, 0x200000, 0x400000, 0x600000)
```
```
    uint32_t const taddr = T + tmem_lane_off + 128u * acc_stage;                                // warp-uniform TMEM address of this warp's 32 rows
```

`w = warp_idx - 4` 的取值为 0..3，并且对 warp 4..7 等于 `warp_idx % 4`，所以象限规则在构造上即已满足。加载本身是 W31，`tcgen05.ld.sync.aligned.32x32b.x16.b32`（explicit.cu:543-555）。它的 shape："A warp executing tcgen05{.ld,.st}.32x32b will access 32 lanes of the Tensor Memory. It loads from or stores to each of the lane (32 * .num)-bits of data"（9.7.18.2.3.1.1）。因此该 warp 的 lane `l` 接收 TMEM lane `32 w + l` 的 `.num = 16` 个连续 32 位列，即当前 subtile `e` 的 `D[32 w + l][16 e .. 16 e + 15]`。ISA 的表 59 为 `.32x32b.x16` 给出 16 个目的寄存器，也就是 wrapper 的 `uint32_t (&d)[16]`。

### 2.3.4 本 kernel 如何划分 TMEM（算例）

累加器 stage `st`（0..3）是全部 128 个 lane 的列 `[128 st, 128 st + 128)`（笔记 6.3："TMEM address = base + (m << 16) + (128*stage + n)"）。MMA 写入 `tmem_c = T + 128 st`（explicit.cu:919 计算 `T + 128u * acc_stage`；lane 位为 0，因为该指令写入其 CTA 的全部 128 个 lane）。epilogue 在 `taddr + 16 e` 处读取 warp `w` 的 subtile `e`（explicit.cu:1036）。

示例。explicit.cu:982 处第 9 节的横幅注释写着 `example: w = 2, st = 1, e = 3 -> tmem address 0x4000B0 = lane 64, column 176`；该加载是 :1036 处的 `tmem_ld_32x32b_x16(taddr + 16u * uint32_t(e), r)` 调用。`w = 2, st = 1, e = 3` 的算术：`T + (64 << 16) + 128 + 48 = 0x400000 + 0xB0 = 0x4000B0`，即 lane 64、列 176：CTA tile 的第 64..95 行、128 x 128 tile 的第 48..63 列、累加器 stage 1。四个 stage x 128 列 = 512 列 = 整个分配。为什么是四个（源自源码）：CUTLASS 的 builder 填满 TMEM，`AccumulatorPipelineStageCount_ = TotalTmem / (CtaTile_M x CtaTile_N) = (128 x 512) / (128 x 128) = 4`，然后用 `cute::min(4u, ...)` 把它封顶，并附上注释 "4 accumulator stages works well to buffer the accumulators, while also preventing overhead in the epilogue tail on small tile sizes"（cutlass/gemm/collective/builders/sm100_umma_builder.inl:260-267）。这些 stage 让 tile `t` 的 epilogue 与后续 tile 的 mainloop 重叠，而不使 MMA 停顿。两个 stage 就已经能把二者解耦，因为一次 epilogue（8 个 subtile）远短于一个 128-k-tile 的 mainloop（分析；第 3 章量化了实际需要多少重叠）。

---

## 2.4 tcgen05 族：异步、单线程发射、由 commit 跟踪

### 2.4.1 发射粒度

PTX ISA 9.4，表 55（9.7.18.5），本 kernel 用到的那些行：

| 指令 | 发射粒度（引文；凡表 55 有 `::2` 条目之处取该条目） | Wrapper，调用点 |
|---|---|---|
| `.mma`、`.commit` | "Issue from a single thread from the CTA-Pair would initiate the base operation. When the current CTA issues the operation, the peer CTA should be active and should not have exited." | W29 explicit.cu:938、W30 :942 与 :944，leader 的 warp 0 中的一个被选中的 lane |
| `.alloc`、`.dealloc`、`.relinquish_alloc_permit` | "Issue from two warps, one in each of the current CTA and its Peer CTA, in order to collectively perform the operation, i.e., the first warp to perform the operation could block until the second warp in the Peer CTA also performs the operation" | W26-W28 :885、:951、:971，两个 CTA 的整个 warp 0 |
| `.ld`、`.wait::ld` | "Issue from a warp in the current CTA can access only 1/4 of the Tensor Memory of the current CTA." | W31 :1036、W32 :1038，每个 epilogue warp |

同样具有约束力的还有："All tcgen05 instructions within a kernel must specify the same value for the .cta_group qualifier"（9.7.18.7.1 与 9.7.18.10.10.1）。这就是为什么 explicit.cu 第 3 节中每一个带 `.cta_group` 限定符的 `tcgen05.*` 字符串（W26-W30：alloc、dealloc、relinquish_alloc_permit、mma、commit）都写着 `cta_group::2`；`tcgen05.ld` 和 `tcgen05.wait::ld`（W31、W32）则不带该限定符。

### 2.4.2 单线程语义与异步性

9.7.18.10.10.1："The instruction tcgen05.mma has single thread semantics, unlike the collective instructions mma.sync or wgmma.mma_async. So, a single thread issuing the tcgen05.mma will result in the initiation of the whole matrix multiply and accumulate operation." Colfax 的 TMEM 教程用一句话陈述了这一点："Only one thread is used to launch UMMA"。该指令 "is an asynchronous instruction which initiates an MxNxK matrix multiply and accumulate operation, D = A*B+D"，并且 "The operation of the form D = A*B is issued when the input predicate argument enable-input-d is false." 操作数："The address operand d-tmem specifies the address of the destination and the accumulation matrix D in the Tensor Memory ... The 64-bit register operand a-desc and b-desc are the matrix descriptors which represent the matrices A and B in shared memory respectively"、"The 32-bit register operand idesc is the instruction descriptor"，以及 "The vector operand disable-output-lane specifies the lane(s) in the Tensor Memory that should be not be updated"，对于 `::2` 有 8 个元素（W29 的八个为零的 `mask[]` 寄存器；实测 D.2：它们没有存活到 SASS 中）。

该 wrapper 是 W29（explicit.cu:517-529，引文见第 6.7.2 节）。

W29 的谓词 `p` 由 `setp.ne.b32 p, %4, 0` 根据 `scaleC` 参数设置，它就是 `enable-input-d` 谓词：一个 tile 的第一个 k-block 上的 `scaleC = 0` 覆写 TMEM stage (`D = A*B`)；该 tile 其余 511 条 MMA 上的 `scaleC = 1` 则进行累加。没有线程等待结果。"Asynchronous" 的意思是，发射 lane 立即继续执行，与此同时 tensor core 执行已经发射的那些 MMA。

流水线化规则 (9.7.18.6.2) 正是使一个 tile 的 512 条 MMA——它们在同一个累加器上发射、且不等待其中任何一条——无需任何 fence 便保持正确的原因："tcgen05.mma.cta_group::N (D writes) -> tcgen05.mma.cta_group::N (D writes) (same N and accumulator address and shape)"，并且 C 读/D 写配对是*流水线化的*，即 "guaranteed to access certain operands in tensor memory in the same order as the instructions that issued them"。来自不同 warp 的指令从不被流水线化，这是所有 MMA 都来自一个线程的又一个理由。

共享内存一侧是通过异步 proxy 读取的 (9.7.18.6.5)："The shared memory accesses by tcgen05.mma and tcgen05.cp operations are performed asynchronously and via the asynchronous proxy (async proxy)." TMA 通过同一个 proxy 写入同样的这些 stage，而交接是一个 mbarrier，其 `complete_tx` "has .release semantics at the .cluster scope" (9.7.10.28.5.3)。因此，在一次 TMA 加载与消费它的 MMA 之间没有任何 proxy fence。

### 2.4.3 完成：tcgen05.commit

9.7.18.12.1：`tcgen05.commit` "is an asynchronous instruction which makes the mbarrier object, specified by the address operand mbar, track the completion of all the prior asynchronous tcgen05 operations ... initiated by the executing thread. Upon the completion of the tracked asynchronous tcgen05 operations, the signal specified by the .completion_mechanism is triggered by the system on the mbarrier object." 带有 `.mbarrier::arrive::one` 时，"an arrive-on operation, with the count argument of 1, is signaled on the mbarrier object. The scope of the arrive-on operation is the cluster scope." 带有 `.multicast::cluster` 时，"Operand ctaMask specifies the CTAs in the cluster such that each bit position in the 16-bit ... ctaMask operand corresponds to the %cluster_ctarank of the destination CTA ... The mbarrier signal is multicast to the same offset as mbar in the shared memory of each destination CTA." 还有 9.7.18.6.2.1："Instructions tcgen05.commit and tcgen05.wait are implicitly pipelined with respect to previously issued tcgen05.{mma,cp,shift} and tcgen05.{ld,st} instructions respectively that they track from the same thread."

三个后果塑造了 explicit.cu：

1. "initiated by the executing thread"：commit 必须来自发射了那些 MMA 的同一个 lane。explicit.cu 在每条 MMA 和每次 commit 之前重新选举 (:932, :942, :944)。`elect.sync` 是确定性的（"the same leader thread is elected for the same membermask every time"，9.7.15.15），因此是同一个 lane。笔记 7.6 陈述了该规则："they must be issued by the same elected thread that issued the MMAs and cannot be replaced by a plain `mbarrier.arrive`"。
2. 一次 commit 是一个*延迟的* arrive。kernel 在一个 k-tile 的第四条 MMA 之后随即对 `empty[s]` 发射 `tcgen05.commit` (:942)，在一个 tile 的最后一条 MMA 之后随即对 `acc_full[st]` 发射 (:944)，并不等待。对 `empty[s]` 的 arrive 发生在该 k-tile 的四条 MMA 以及该线程所有更早的 MMA 都已完全*完成*、D 已写入之时，这意味着它们的 smem 读取已经结束。对 `acc_full[st]` 的 arrive 同样发生在该 tile 的最后一条 MMA 完成之时。ISA 跟踪的是完成，而不是操作数读取的结束。一个只针对 A 读取的信号仅以 `.sync_restrict::shared::read::mma::a` 变体的形式存在，它 "performs an arrive-on operation on the barrier upon completion of read of A matrix from shared memory for all prior tcgen05.mma operations. This does not signal the overall completion of any prior tcgen05.mma operations" (9.7.18.12.1)；ISA 只为 "sm_107f or higher in the same family" 列出该变体，因此它对本 kernel 的 sm_100a 构建不可用。
3. mainloop release 的掩码是 `0xF` (:942, `kMainloopReleaseMask`, explicit.cu:186)。每个 CTA 的 stage `s` 只被它自己所在 pair 的 MMA 读取（pair leader 在其上发射的那四条 k-block MMA），因此单就这个 pair 而言只需要其 leader 的 commit。计数 2 和掩码 `0xF` 来自 A 的 multicast。CTA `r` 的 A box 落在 rank `x` 和 `x + 2` 中，每个 pair 中各一个 CTA（掩码 `0x5 << x`，explicit.cu:772）。在 `r` 可以用它的下一个 A box 覆写 stage `s` 之前，两个 pair 的 MMA 都必须已经读完它们所收到的副本。因此，每个 leader 的 commit 都 multicast 到全部四个 CTA，而每个 `empty[s]` 等待两次 commit：`count = cluster_m/atom_m + cluster_n/atom_n - 1 = 2/2 + 2/1 - 1 = 2`（explicit.cu:182 处的注释；CUTLASS 的 `multicast_consumer_arrival_count`，sm100_pipeline.hpp:565-566，以及用于掩码的 `calculate_multicast_mask<kRowCol>`，:57-91；笔记 13 第 7 项）。累加器 commit (:944) 使用 `0x3 << 2y`，因为只有这个 pair 的两个 epilogue 等待它。

W30 (explicit.cu:532-540) 的引文见第 6.7.3 节。

W30 在一个 `"h"`（16 位寄存器）约束下传递 16 位的 `ctaMask`。这个字符串的 CUTLASS 来源是 `umma_arrive_multicast_2x1SM` (cutlass/arch/barrier.h:848-861)，其中包含 `elect_one_sync()`；explicit.cu 把选举移到了调用点。笔记第 13 节第 1 项记录了一个陷阱：epilogue 对累加器的 *release* (W11) 是一个普通的 `mbarrier.arrive`，而不是 `tcgen05.commit`，因为跟踪 epilogue 的 TMEM 读取的是 `tcgen05.wait::ld`，而不是 mbarrier 机制。

### 2.4.4 tcgen05.ld 与 tcgen05.wait::ld

9.7.18.8.3："Instruction tcgen05.ld asynchronously loads data from the Tensor Memory at the location specified by the 32-bit address operand taddr into the destination register r, collectively across all threads of the warps. All the threads in the warp must specify the same value of taddr"。9.7.18.8.5："Instruction tcgen05.wait::ld causes the executing thread to block until all prior tcgen05.ld operations issued by the executing thread have completed." ISA 将 `tcgen05.ld -> tcgen05.wait::ld` 列为一种 "implicitly pipelined tcgen05 instruction pairing"，`tcgen05.wait` 是 `tcgen05.ld` 的完成机制 (9.7.18.6.2.1.2)，并且它自己的示例在一条会覆写所加载区域的 `tcgen05.mma` 之前放了一个 `wait::ld`："tcgen05.ld ...; // Prevents subsequent tcgen05.mma from racing ahead of the tcgen05.ld; tcgen05.wait::ld.sync.aligned; tcgen05.mma ..."。

explicit.cu 完全遵循 CUTLASS：每个 tile 八次加载 (:1036)，一个 `tcgen05.wait::ld` (W32, :1038) 位于第八次之后、并且在累加器 release（:1039，引文见第 10.8 节）**之前**。

正是这个 wait 使 release 合法：在它之后，这个 warp 不再有任何加载仍在读取 stage `st`，因此 leader 可以让 MMA 覆写它。对于 subtile 0..6，加载到的寄存器在没有显式 wait 的情况下被消费 (`FMUL`/`FFMA`/`STS`)。这在 ISA 上是合法的，而不是一条捷径（来源：PTX ISA 9.4，9.7.18.6.4.5 "Register dependencies, same thread"）："For tcgen05.ld, an intra-thread ordering through true register dependency will be respected regardless of the presence or absence of other forms of synchronization." 读取 `r[j]` 的 `FMUL` 就是这样一个依赖，因此在每次加载之后放一个 `tcgen05.wait::ld` 对寄存器在 ISA 层面的正确性不会增添任何东西。寄存器依赖所不提供的是内存排序："This form of register dependency does not imply any other form of ordering"，并且特别地，"a register dependency does not imply that a dependee instruction's memory accesses will be performed before a dependent instruction's memory accesses. To enforce such memory orderings and avoiding anti-dependency hazards around tcgen05.ld, tcgen05.wait::ld must be used." 这就是每个 tile 那一个 `wait::ld` 的角色。它关闭了这个 warp 对 stage `st` 的 TMEM 读取与 MMA 对该 stage 的下一次覆写之间的读后写冒险，而那次覆写是紧随其后的 release 将要允许的。笔记 B.6 的措辞 "the ISA requires tcgen05.wait::ld before the destination registers are read"，以及 MLC 章把紧随其 TMEM 加载之后发射的 `tcgen05.wait.ld()` 解读为 "confirms that the asynchronous TMEM load has populated its destination registers" 的读法，都比 ISA 原文更强：ISA 要求它在*内存*被交接之前，而不是在寄存器被使用之前。实测 (D.2, G.3)：ptxas 根本不发出任何 wait 操作码。`LDTM.x16` 的目的寄存器由 scoreboard 保护，并且每条 `LDTM` 之后的第一条 `FMUL` 在其上等待；subtile 7 的 `SYNCS.ARRIVE` 在最后一条 `LDTM` 的 scoreboard 上等待。至于控制字所编码的就是这一点，则是从 SASS 编码推断出来的。放在别处、即一条 `LDTM` 与其 FMUL 之间的 wait 是否也会被折叠掉，未经实测。这里的 *scoreboard* 是 SASS 控制字所设置并等待的 per-warp 依赖计数器之一。控制字在某个 scoreboard 上等待的指令会停顿，直到设置它的 producer 交付了其结果；这是针对可变延迟指令的硬件互锁。该术语由 NVIDIA 内部信息确认（scoreboard token：producer 上的 `&wr=x`，consumer 上的 `&req={x}`）；对这份清单的控制字的解读（第 12.1.3 节）出自笔记（D.2，推断）。

**ISA 的规范模式增添了什么，以及 kernel 省略了什么。** 9.7.18.6.4.4（"Non-pipelined instructions, different thread"）给出了两种跨线程模式，它们比 kernel 所写的更严格。示例 2：线程 0 `tcgen05.mma; tcgen05.commit.mbarrier::arrive::one [mbar]`；线程 1 `mbarrier.try_wait.relaxed.cluster [mbar]; tcgen05.fence::after_thread_sync; tcgen05.ld`。示例 1：线程 0 `tcgen05.ld; tcgen05.wait::ld; tcgen05.fence::before_thread_sync; mbarrier.arrive.relaxed.cluster`；线程 1 `mbarrier.try_wait.relaxed.cluster; tcgen05.fence::after_thread_sync; tcgen05.mma`。这些 fence 的定义是 "tcgen05.fence::before_thread_sync orders all the prior asynchronous tcgen05 operations with respect to the subsequent tcgen05 and the execution ordering operations" 和 "tcgen05.fence::after_thread_sync orders all the subsequent asynchronous tcgen05 operations with respect to the prior tcgen05 and the execution ordering operations" (9.7.18.11.1)。与 CUTLASS 一样，explicit.cu 两个都不发出：在 `acc_full` wait 与第一条 `tcgen05.ld` 之间没有 `tcgen05.fence::after_thread_sync`，在 `tcgen05.wait::ld` 与 `acc_empty` arrive 之间也没有 `tcgen05.fence::before_thread_sync`。笔记 7.2 记录了 "No tcgen05.fence::before_thread_sync/after_thread_sync exists anywhere in the CUTLASS C++ tree, and none appears in the PTX"，而 G.3 的盘点将 `tcgen05.fence` 列为 0。该 kernel 经实测是正确的且数值相等 (G.1)。有一个差异或许可以解释原因（分析，未验证）：ISA 的示例使用 `.relaxed.cluster` 的 wait 并让 fence 来提供排序，而 kernel 的 wait 带有默认的 `.acquire` 语义（第 2.8.4 节）。一个遵循 ISA 字面规定的编写者会加上这两个 fence。省略它们是 CUTLASS 的做法，而非 ISA 的保证，并且第 13 章（第 13.3 节末尾的列表）把它列入需要在另一套工具链上重新验证的假设之中。

---

## 2.5 2SM MMA：`cta_group::2`

### 2.5.1 该指令计算什么，以及操作数位于何处

9.7.18.10.10.1："The qualifier .cta_group::2 specifies that the matrix multiply and accumulate operation is performed on the Tensor Memory of the executing thread's CTA and its peer CTA." 操作数列表（9.7.18.10）："the A matrix has shape MxK, in either Tensor Memory or Shared Memory; the B matrix has shape KxN, in Shared Memory of the current CTA and optionally in peer CTA; the D matrix is of the shape MxN, in Tensor Memory." 对于 M = 256，数据通路组织方式为 "Layout A (M = 256)"，其数据通路组织为 "4x1"，lane 对齐为 0（9.7.18.10.5 中的表）：每个 CTA 的 128 个 lane 保存 D 的 128 行。发射该指令的线程可以属于该 pair 中的任一 CTA（mlc.ai 的章节 "Blackwell Tensor Core: tcgen05.mma"："That thread may belong to either CTA, but the peer CTA must remain active"）；按 CUTLASS 的约定，由偶数 CTA 领头（第 2.1.2 节；`is_leader = ((rank & 1u) == 0u)`，explicit.cu:1106）。

ISA 留给其 layout 图示去表达的东西，CUTLASS 用类型写明，而笔记通过实测予以确认：`LayoutC_TV (_2,(_128,_128)):(_128,(_1,_256))`（explicit.cu:130-131 注释，D1），即 "CTA v of the pair owns accumulator rows [128 v, 128 v + 128) of the 256 x 128 MMA tile, i.e. its own 128 x 128 CTA tile"。笔记 B.6："A rows [0,128) and B columns [0,64) of the 256x128 instruction come from the leader's smem and the remaining halves from the peer's smem at the same CTA-relative addresses; each CTA's TMEM receives its 128 rows of D."

```
 tcgen05.mma.cta_group::2.kind::f16   M 256 x N 128 x K 16   issued by ONE lane of the EVEN CTA's warp 0

      A  (256 x 16 fp16, K-major)       B (16 x 128 fp16, K-major)        D (256 x 128 fp32)
      +---------------------------+     +-------------+-------------+     +---------------------------+
      | rows   0..127 = tm even   |<----| cols 0..63  | cols 64..127|     | rows   0..127             |---> LEADER TMEM
      | leader smem_A[s] + 32 kb  |     | leader      | peer        |     | lanes 0..127, cols 128st..|     stage st
      +---------------------------+     | smem_B[s]   | smem_B[s]   |     +---------------------------+
      | rows 128..255 = tm odd    |<----| + 32 kb     | + 32 kb     |     | rows 128..255             |---> PEER TMEM
      | peer   smem_A[s] + 32 kb  |     +-------------+-------------+     | lanes 0..127, cols 128st..|     stage st
      +---------------------------+                                       +---------------------------+
      desc_a names "smem_A[s] + 32 kb" ONCE; the hardware reads that CTA-relative address in both CTAs
```

每个 CTA、每个 k-tile，smem 保存一个 128 x 64 fp16 的 A stage（16384 B，得益于 multicast，两个 64 行的半块都在）和一个 64 x 64 fp16 的 B stage（8192 B，本 CTA 的 64 列）。四条 MMA 以 `kb = 0..3` 在 stage 内部沿 K 以 16 个元素（32 字节）为步长步进（作用于以 16 字节为单位的低位字的 `+ 2u * uint32_t(kb)`，explicit.cu:933-934）。这就是 "4 MMAs per k-tile"（`kKBlocks = kTileK / kMmaK = 64 / 16`，explicit.cu:106）的由来。

### 2.5.2 为什么两个 CTA 必须有完全相同的共享内存 layout

该指令接受**一个** A 描述符和**一个** B 描述符，并在两个 CTA 中应用同一个 CTA 相对起始地址，因此 peer 的 stage `s` 必须位于与 leader 相同的偏移处。explicit.cu 平凡地保证了这一点：两个 CTA 运行同一个 kernel，使用同样的 `OFF_*` 常量。描述符的 14 位起始地址字段（第 2.6.1 节）丢弃了 `shared::cta` 地址的 rank 位 [24,32)，因此 `desc_lo_a`/`desc_lo_b` 在 leader 和 peer 中逐位相同（explicit.cu:896-897，引文见第 9.4.4 节）。

实测 D7：在**两个** leader（rank 0 和 2）的第一条 MMA 处，`desc_a = 0x4000404000010880`、`desc_b = 0x4000404000012880`。算术：`(0x400 + 33792) >> 4 = 34816 >> 4 = 0x880`；`(0x400 + 164864) >> 4 = 165888 >> 4 = 0x2880`。

### 2.5.3 为什么要用 2SM（分析；第 3 章给出量化）

每个 CTA、每个 k-tile，kernel 从 L2 向自己的 smem 搬入 16 KiB 操作数：自己的 64 x 64 A 半块 = 8 KiB 和自己的 64 x 64 B 半块 = 8 KiB。另一半 A 由 multicast 送达，其代价由同胞 CTA 的加载支付（即另一个 pair 中 `x` 相同的那个 CTA，A multicast 掩码 `0x5 << x`，第 2.1.2 节；不是 2SM peer），所以写入 smem 的是 24 KiB（16 KiB A + 8 KiB B）。要精确区分是什么换来了什么。A 的减半靠的是 **multicast**，一个 1SM 的 2 x 2 cluster 同样可以利用它。B 的减半来自 **2SM MMA 直接读取 peer 的 64 列**；1SM 设计则需要改用 B multicast。因此每个 CTA、每个 k-tile，两种设计都可以降到从 L2 取 8 KiB 的 A 加 8 KiB 的 B，对应同样的每 CTA 2 x 128 x 128 x 64 = 2.1 MFLOP。`cta_group::2` 所特有的是共享内存占用和进入每个 SM 的 TMA 写带宽。一个计算 128 x 128 x 64 tile 的 1SM CTA 必须保存完整的 128 x 64 B stage：每 stage 32 KiB（16384 A + 16384 B）而不是 24 KiB，在 builder 的 198464 字节预算（第 2.2 节）中放得下 `floor(198464 / (32768 + 16)) = 6` 个 stage 而不是 8 个。而且 256 x 128 的指令 shape 本可以让 tensor core 为 A 的 256 行（而不是 128 行）只从 smem 读取每个 B 元素一次，使每 FLOP 的 B smem 读带宽减半，但 `cta_group::2` 并不意味着 B 只读一次：它为该操作把 CTA pair 编为一组，并允许完成屏障位于任一 CTA 中（NVIDIA 内部），而 ISA 只说明了 B 可以位于何处（9.7.18.10），没有说明两条数据通路如何取它。第 3.3.1 节和第 3.9.3 节建立在同一个只读一次的模型上；它们的读取数字是界，而不是经过验证的流量（NVIDIA 内部）。代价是跨 SM 的耦合：leader 的发射线程必须知道两个 CTA 的 stage 都已满，而这正是第 4 章，第 4.4.1 节的 "six boxes credit the leader's `full[s]`" 规则所提供的。

---

## 2.6 描述符：概念

第 5 章给出位图和那些 `static_assert`；这里讲的是这两个 64 位/32 位的字*意味着*什么。

### 2.6.1 共享内存矩阵描述符

PTX ISA 9.4，9.7.18.4.1："The shared memory descriptor describes the properties of multiplicand matrix in shared memory including its location in the shared memory of the current CTA. It is a 64-bit value"。其表 49 的各字段连同 kernel 的取值、编码函数 `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4` 以及 ISA 的三条规则见第 5 章，第 5.8.1 节。

kernel 所依赖的三个事实：

- **地址字段覆盖 256 KiB 并丢弃 rank。** 该编码保留地址位 [4,18)，因此对于有效的编码，15 位字段的第 14 位始终为 0；CUTLASS 声明为 `uint16_t start_address_ : 14`（cute/arch/mma_sm100_desc.hpp:107），explicit.cu 用 `0x3FFF` 做掩码（:896-897），而 kernel 不丢弃任何编码所保留的位。从地址位 18 往上的一切，包括位 [24,32) 中的 rank，都被丢弃。由此得到第 2.5.2 节的 leader/peer 一致性（第 9 章，第 9.4.4 节），以及 explicit.cu:165 处的断言，即实测基址加上整个缓冲区保持在 `1 << 18` 以下（`0x400 + 230400 = 231424 < 262144`）：

```
static_assert(kSmemBaseLow24 + kSmemBytes < (1u << 18), "14-bit UMMA start-address field spans 256 KiB (B.3.2 step 4)");
```

- **K-major 的 stride 字节偏移 = "The offset from the first 8 rows to the next 8 rows"**（9.7.18.3.2）。A/B stage 的一行是 64 个 fp16 = 128 B，所以 8 行是 1024 B，编码为 `1024 >> 4 = 64`（`kSmemDescHi = 0x40004040` 在 [32,46) 携带 SBO = 64，在 [46,48) 携带版本 1，在 [61,64) 携带 layout 类型 2 = SWIZZLE_128B；explicit.cu:171）。
- **swizzle 图案必须从其自然边界开始，否则 base offset 必须另作说明。** 表 50："128-Byte swizzle: 1024-Byte boundary"，"The value of base offset is 0 when the repeating pattern of the specified swizzling mode starts as per shown in Table 50." "Otherwise, the base offset must be a non-zero value, computed using the following formula: base offset = (pattern start addr >> 0x7) & 0x7"（9.7.18.4.1）。explicit.cu 在 `kSmemDescHi` 中把 base offset 保持为 0，转而让每个 A/B stage 的基址都按 1024 对齐（explicit.cu:162；stage stride 16384 和 8192 都是 1024 的倍数；实测基址 `0x400` 是 1024 对齐的）。TMA 一侧从它那端陈述了同一条规则（第 2.7.3 节），这就是该对齐的全部理由。

该描述符所描述的规范 K-major 128B-swizzled layout，用 CuTe 的术语说，是作用在 `((8,m),(T,2k))` 上的 `Swizzle<3,4,3>`（9.7.18.3.3，表 "Canonical Layouts"），它在每个 stage 上恰好就是笔记的 `SmemLayoutA/B`（第 7.5 节），也正是 TMA `SWIZZLE_128B` 模式所写出的东西（第 2.7.3 节）。kernel 从不计算这个 swizzle：TMA 在写入时施加它，MMA 在读取时撤销它，而描述符就是二者之间的约定。

### 2.6.2 指令描述符

PTX ISA 9.4，9.7.18.4.2，表 51 定义了 `.kind::f16` 的 32 位指令描述符。CuTe 的 `InstrDescriptor`（`cute/arch/mma_sm100_desc.hpp:416-443`）与之对应，所用的字段名正是本讲义和 kernel 自己的注释所用的（explicit.cu:173：`c_format F32 (bit 4), n_dim 128 >> 3 = 16 at [17,23), m_dim 256 >> 4 = 16 at [24,29)`）。kernel 的常量 `kInstrDesc = 0x10200010` 是 F32 累加（第 4 位）、F16 输入（零）、两者均为 K-major（零）、`n_dim = 128 >> 3 = 16`、`m_dim = 256 >> 4 = 16`；实测 D7 在每条 K3 记录中 `idesc = 0x10200010`。它是 W29 的一个运行时 `"r"` 操作数，但在整个 kernel 中是常量，因为指令 shape 从不改变。带有 kernel 取值的完整位图在第 5 章，第 5.8.2 节。

---

## 2.7 TMA：tensor map、box、swizzle、multicast、bulk group、异步 proxy

### 2.7.1 Tensor map

tensor map 是一个 128 字节的不透明描述符：即驱动 API 的 `CUtensorMap`，一个由 16 个不透明的 64 位字组成的数组。它在 CUDA 12 的头文件中按 64 字节对齐，在 CUDA 13 的头文件中按 128 字节对齐（explicit.hpp:66 注释和笔记 G.3）。CuTe 的 `using TmaDescriptor = CUtensorMap;` 位于 cute/arch/copy_sm90_desc.hpp:292；往下三行的后备定义 `struct alignas(64) { char bytes[128]; }`（:295）把大小明确写了出来，而 explicit.hpp:67 用 `static_assert(sizeof(CUtensorMap) == 128, ...)` 将其固定。主机端用 `cuTensorMapEncodeTiled` 编码它，并作为 `const __grid_constant__` 按值传给 kernel（explicit.cu:1096：`explicit_blackwell_fp16_gemm_kernel(const __grid_constant__ ExplicitGemmParams params)`），因此设备端使用的是它在参数空间中的地址。驱动 API 的规则（CUDA 驱动 API，`cuTensorMapEncodeTiled`）：`globalAddress` "must be 16 byte aligned"；`globalDim` "must be non-zero and less than or equal to 2^32"；`globalStrides` "must be a multiple of 16 and less than 2^40"；`boxDim` "must be non-zero and less than or equal to 256"；`boxDim[0] * elementSizeInBytes` "must be a multiple of 16 bytes"；对于带 swizzle 的 map，"the bounding box inner dimension (computed as boxDim[0] multiplied by element size derived from tensorDataType) must be less than or equal to the swizzle size"（32/64/128 字节）。所有这些在这里都平凡地成立：`cudaMalloc` 指针至少按 256 字节对齐（CUDA Programming Guide，"Device Memory Accesses"），各维大小为 8192，stride 为 16384 和 32768 B。

本次运行的六个 map（A、B、C、D，加上 CUTLASS 对 A 和 B 的两份后备副本，它们与前者逐字节相同，且从未传给 explicit kernel；笔记 5.3，实测 D3；全部为 rank 3，带一个大小为 1、stride 为 0 的 batch mode）：

| Map | dtype | 维度（最内层在前） | stride（字节） | box | swizzle | 内层 box 字节数 |
|---|---|---|---|---|---|---|
| A | FLOAT16 | (K, M, L) = (8192, 8192, 1) | (16384, 0) | (64, 64, 1) | `CU_TENSOR_MAP_SWIZZLE_128B` | 64 x 2 = 128 = swizzle 跨度 |
| B | FLOAT16 | (K, N, L) | (16384, 0) | (64, 64, 1) | `SWIZZLE_128B` | 128 |
| C, D | FLOAT32 | (M, N, L) | (32768, 0) | (32, 16, 1) | `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` | 32 x 4 = 128 |

共同项：`INTERLEAVE_NONE`、`L2_PROMOTION_L2_128B`、`OOB_FILL_NONE`。每个 map 的内层 box 都是 128 字节，因为 128 字节 swizzle 规则把它封顶在那里：一个 64 x 64 fp16 的 box 是 8192 B，一个 32 x 16 fp32 的 box 是 2048 B。这就是为什么一个 A/B stage 在每个 CTA 上是两个 box（一个 A 半块、一个 B 半块：第 2.7.4 节），以及为什么一个 128 x 16 fp32 的 epilogue subtile 是四个 D box（笔记 13 第 14 项；`kStoreBoxBytes = 32 * 16 * 4 = 2048`，explicit.cu:120）。

`prefetch.tensormap`（W20，explicit.cu:440-447）"brings the cache line containing the specified address in the .const or .param memory state space for subsequent use by the cp.async.bulk.tensor instruction"（PTX ISA 9.4，9.7.10.16）。kernel 在 prologue 中每个 CTA 预取全部四个 map 一次（explicit.cu:1114-1119；实测 SASS `UTMACCTL.PF` x 4）。

### 2.7.2 box、坐标与越界填充

PTX ISA 9.4，5.5.3.1："A tensor can be accessed in chunks known as Bounding Box ... Size of each bounding Box must be a multiple of 16 bytes. The address of the bounding Box must also be aligned to 16 bytes ... The tensor-coordinates, specified in the PTX tensor instructions, specify the starting offset of the bounding box." 坐标是元素索引，最内层在前：A/B 用 `(k, row, 0)`，C/D 用 `(m, n, 0)`（explicit.cu:134-135 注释；实测 K4a/K4b）。5.5.3.3："Zero fill mode: Elements in the Bounding Box which fall outside of the tensor boundary are set to 0. OOB-NaN fill mode: ... set to a special NaN called OOB-NaN." `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` 就是零填充模式（驱动 API）。在本次运行中没有任何 box 越过边界（8192 是 64、32 和 16 的倍数），所以 OOB 填充虽已配置但从未被用到；一个通用的 kernel 则可借此免费获得余数处理（第 13 章）。

### 2.7.3 Swizzle 模式：`SWIZZLE_128B` 与 `SWIZZLE_128B_ATOM_32B`

为什么要 swizzle。对 MMA 操作数而言，原因是共享内存的 bank 冲突。一个 K-major 核心矩阵是 8 行 x 位于同一列位置的一个 16 字节 chunk；不做 swizzling 时，这八个 chunk 相隔 128 B，落在 32 x 4 字节 bank 中相同的四个里，而把 chunk 索引与行索引 mod 8 做 XOR 就把它们散布到 128 字节行的全部八个位置上（对 layout 的分析；硬件实际的访问分组未见文档记载；第 8 章，第 8.8.2 节）。对 epilogue stage 而言，原因则不同。它的模式由 D tensor map 决定（`CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`，由 CUTLASS 的 epilogue builder 通过 `Layout_MN_SW128_32B_Atom<float>` 选定，笔记 13 第 10 项），而通用 proxy 写入方必须产生与 TMA 存储将要读回的完全一致的字节图案。对这个 kernel 的存储而言，它*不是*一个 bank 冲突的修复：每条 warp 宽的 `st.shared.b32` 覆盖一个 128 字节的行，全部 32 个 bank 各一次，无论有没有这个 XOR 都是如此（源自 explicit.cu:1013；第 10 章，第 10.5.5 节）：

```
    st_off[c] = OFF_SMEM_CD + 2048u * w + 4u * (l & 7u) + 128u * c + 32u * (((l >> 3) & 3u) ^ c);
```

在这两种模式下，TMA 单元都在数据进入时对每个 128 字节行内部的 16 字节（或 32 字节）chunk 做置换，而 consumer 在数据出去时施加同样的置换：MMA 通过它的描述符（第 2.6.1 节），TMA 存储通过它的 map。

**带 16 字节原子性的 `SWIZZLE_128B`**（PTX ISA 9.4，5.5.7 "128 byte swizzle mode ... 16-byte atomicity sub-mode: In this sub-mode, the 16-byte of data is kept intact while swizzling"；图案表是针对 1024 字节对齐的起点给出的）。行 `r` 的 chunk `c`（0..7）被存放在 chunk 位置 `c ^ (r % 8)`：

```
  16-byte chunk position inside the 128-byte line:   0  1  2  3  4  5  6  7
  line r = 0  holds chunks                           0  1  2  3  4  5  6  7
  line r = 1                                         1  0  3  2  5  4  7  6
  line r = 2                                         2  3  0  1  6  7  4  5
  line r = 3                                         3  2  1  0  7  6  5  4
  line r = 4                                         4  5  6  7  0  1  2  3
  line r = 5                                         5  4  7  6  1  0  3  2
  line r = 6                                         6  7  4  5  2  3  0  1
  line r = 7                                         7  6  5  4  3  2  1  0
  ... pattern repeats every 8 lines = 1024 bytes
```

用地址来说：字节地址的位 [4,7)，即 chunk 索引，与位 [7,10)，即行索引 mod 8，做 XOR。这就是 CuTe 的 `Swizzle<3,4,3>`（explicit.cu:204 断言 `yyy_msk = 0x380`、`num_shft = 3`；explicit.cu:127 处的注释："the 16-byte chunk inside a 128-byte row is XOR-ed with (row % 8)"）。该图案每 1024 B 重复一次，所以一个 stage 必须从 1024 字节边界开始，TMA 写入与描述符读取（base offset 0）才能一致（explicit.cu:162；第 8 章，第 8.8.1 节和第 8.8.3 节）。TMA 的表（5.5.7）说 "The value of swizzle base offset is 0 when the dstMem shared memory address is located at the following boundary: 128-Byte swizzle: 1024-Byte boundary"，而描述符的表 50（第 2.6.1 节）对读取方说了同样的话。

**`SWIZZLE_128B_ATOM_32B`** 保持 32 字节单元完整不动（"This sub-mode requires 32 byte alignment at shared memory"，5.5.7）。CuTe 把它写成作用在字节偏移上的 `Swizzle<2,5,2>`：位 [5,7) 与位 [7,9) 做 XOR，周期为 512 B。explicit.cu 在编译期把自己的公式与 CuTe 交叉核对（explicit.cu:197；`static_assert` 位于 :199-202）：

```
__host__ __device__ constexpr uint32_t epi_swz(uint32_t o) { return o ^ ((o & 0x180u) >> 2); }
```

例子（explicit.cu:196 注释）：偏移 `0x1E4` 变为 `0x1E4 ^ ((0x1E4 & 0x180) >> 2) = 0x1E4 ^ 0x60 = 0x184`。512 B 的周期就是 C/D stage 按 512 对齐的原因（explicit.cu:163）；UMMA 描述符的枚举把这一模式单独列为一种 layout 类型，`SWIZZLE_128B_BASE32B = 1`（PTX ISA 9.4 表 49："128-Byte with 32B atomic swizzling"；epilogue 从不构建描述符，所以这里只有 TMA 一侧使用这种模式）。第 10 章从这个 swizzle 推导出 epilogue 的字节公式；第 5 章展示那些 `static_assert`。

### 2.7.4 加载：`complete_tx`、`cta_group::2`、multicast

加载字符串是 W21（explicit.cu:450-460，引文见第 6.6.3 节）。

逐个限定符来看（PTX ISA 9.4，9.7.10.28.5.3）：

- `cp.async.bulk.tensor` "is a non-blocking instruction which initiates an asynchronous copy operation of tensor data from the location in .src state space to the location in the .dst state space." `.3d`：三个坐标。`.shared::cluster.global`：目的地可以是 cluster 中的任意 CTA，源是经由 tensor map 访问的全局内存。
- `.mbarrier::complete_tx::bytes`："Upon the completion of the asynchronous copy operation, the complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar." 以及："When the mbarrier object mbar resides in the shared memory of a different CTA within the cluster (a remote barrier), .dst must be specified as .shared::cluster"。
- `.cta_group::2`："The modifier .cta_group is used to signal either the odd numbered CTA or the even numbered CTA among the CTA-Pair ... When .cta_group::2 is specified, the mbarrier object mbar can be in shared memory of either the same CTA as the shared memory destination dstMem or in its peer-CTA." 正是这一点让 peer 的 B 加载（W22，无 multicast）得以指名 **leader 的** `full[s]`（Colfax，cluster 教程："Setting this to `cta_group::2` allows the TMA copy to arrive at the mbarrier of either the executing CTA or its peer CTA"）。
- `.multicast::cluster`："Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit ... ctaMask operand corresponds to the %cluster_ctarank of the destination CTA ... The source data is multicast to the same offset as dstMem in the shared memory of each destination CTA." 配合 `.cta_group::2`："the mbarrier signal is multicast either to all the odd-numbered CTAs or even-numbered CTAs for each possible CTA-Pair within a cluster. In particular, for each destination CTA specified in the ctaMask, the mbarrier signal is sent either to the destination CTA or its peer-CTA based on the %cluster_ctarank parity of the shared memory where the mbarrier object mbar resides." kernel 传入的是一个**偶数**屏障地址（第 24 位被清零，第 2.9.3 节），因此每个目的地的信号都发往该目的地所在 pair 的偶数 CTA。
- `.L2::cache_hint`："The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access ... treated as a performance hint only"。值 `0x1000000000000000` 是 CUTLASS 的 `CacheHintSm100::EVICT_NORMAL` (`cute/arch/copy_sm90_desc.hpp:193-194`, = `Sm100MemDescDefault`, copy_sm100_tma.hpp:46; explicit.cu:188)；策略字的编码在 ISA 中是不透明的，且我没有找到对其各个比特位的公开拆解（除 CUTLASS 常量之外未验证）。

每个 stage `s`，每个 CTA 发射一个 A box（8192 B，它自己的 64 行那一半，在其自身 tile 坐标中为第 `128 tm + 64 y .. + 64` 行，explicit.cu:777，multicast 到 rank `x` 和 `x + 2`）和一个 B box（8192 B，不 multicast）。

实测 K4a (D8)：对于 128 条记录（第 8 章，第 8.7.2 节），全部四个 rank 的每一个坐标和目的地都与每个 rank 的行、列及地址公式（第 8 章，第 8.5.1 节和第 8.7 节）相符。*发射出去的*操作数是 leader 的掩码地址，这一点由 SASS（`sass_gemm.txt` `0x1b90`/`0x1e80`：baseline 每 k-tile 对屏障寄存器执行的 `ULOP3 ..., 0xfefffff8`，其 PTX 形式被 D.2 列为 `and.b32 %r, %r, -16777224`；G.3：explicit kernel 的掩码折叠进外提的基址，成为 `& 0xfefffff0`；第 2.9.3 节）以及每次运行都以数值相等告终 (G.1) 所证明，而未加掩码的操作数会在 `q = 0` 处死锁（第 8 章，第 8.5.3 节）。

### 2.7.5 存储：bulk group

存储（W23，explicit.cu:476-484，引文见第 6.6.5 节）使用另一种完成机制。

`.bulk_group`："cp.async.bulk.commit_group instruction creates a new per-thread bulk async-group and batches all prior cp{.reduce}.async.bulk{.prefetch}{.tensor} ... instructions ... initiated by the executing thread but not committed to any bulk async-group" (9.7.10.28.6.1)。"cp.async.bulk.wait_group instruction will cause the executing thread to wait until only N or fewer of the most recent bulk async-groups are pending and all the prior bulk async-groups committed by the executing threads are complete ... The optional .read modifier indicates that the waiting has to be done until all the bulk async operations in the specified bulk async-group have completed: reading from the tensormap, the reading from their source locations" (9.7.10.28.6.2)。

因此 kernel 每个 subtile 的三元组 (explicit.cu:1066-1069)：来自 stage `e % 4` 的四条 `tma_store_3d`、一条 `cp.async.bulk.commit_group` (W24)、一条 `cp.async.bulk.wait_group.read 1` (W25)；引文见第 10.10 节。

为什么用 `.read`，以及为什么 `wait_group.read 1` 中是 `1`。epilogue 只需要把 smem stage 拿回来，而不需要全局写入变得可见，所以 `.read` 是更便宜的等待。允许一个组保持未完成，让 subtile `e` 的存储与 subtile `e + 1` 的算术运算重叠，同时仍然保证：四个 subtile 之前读取过 stage `e % 4` 的那个组，在 subtile `e + 4` 复用它时已经完成（笔记 13 第 16 项；第 10 章，第 10.10.4 节走一遍这个环）。组按线程归属，这就是为什么发射存储的 warp 就是进行等待的 warp：一个组属于发射它的线程。由单个 warp，即 warp 4，发射存储，这一点与 CUTLASS 一致（"one warp issues TMA store"，`sm100_epilogue_tma_warpspecialized.hpp:739-740`；`issue_tma_store = (w == 0u)`，explicit.cu:1018），而 warp 4 的全部 32 个 lane 都以 uniform 操作数发射（实测 K4b：32 个活跃 lane，每个 box 一条 `UTMASTG.3D`）。

### 2.7.6 异步 proxy 与 `fence.proxy.async`

PTX ISA 9.4，8.6 ("Proxies")："A memory proxy, or a proxy is an abstract label applied to a method of memory access. When two memory operations use distinct methods of memory access, they are said to be different proxies. A proxy fence is required to synchronize memory operations across different proxies." 不属于通用 proxy 的操作包括 "async-proxy and tensormap-proxy accesses by .async.bulk operations"，以及如第 2.4.2 节所引的 `tcgen05.mma` 对 smem 的读取。CLC 响应同样是以 "using weak async-proxy writes" 的方式写入的 (9.7.15.18)。9.7.15.4："Value .async of the .proxykind qualifier specifies that the memory ordering is established between the async proxy and the generic proxy. The memory ordering is limited only to operations performed on objects in the state space specified."

在本 kernel 中，通用与异步两种 proxy 触及相同字节之处，以及为它们排序的 fence（W15，explicit.cu:418-425，引文见第 6.4.2 节）：

1. Epilogue：每个线程向 stage `e % 4` 写入 16 条通用 `st.shared.b32`，然后 TMA 存储通过异步 proxy 读取该 stage。每个线程都执行 fence，然后 `bar.sync 1, 128` 使全部 128 个线程经此排序的写入都先于 warp 4 的存储发射（explicit.cu:1050-1051，引文见第 10.10 节）。

2. CLC：对响应槽位的通用 `ld.shared.b128` 必须排在对同一槽位的**下一次**异步 proxy 写入之前（两次查询之后的下一个 `try_cancel`）。explicit.cu:683 处解码之后的那条 fence，正位于 ISA 示例放置其 "before next iteration async-proxy write to handle" fence 的位置；该示例把那条 fence 写成单向的 `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`（与其 cluster 屏障之后的一条 `.acquire` fence 配对），而 CUTLASS 与 explicit.cu 使用的是双向的 `fence.proxy.async.shared::cta`。

什么**不**需要 *proxy* fence：TMA 加载 -> MMA（两者都是异步 proxy，通过 mbarrier 的 release/acquire 排序）；MMA -> `tcgen05.ld`（由 `tcgen05.commit` -> mbarrier 排序；ISA 对这一交接的规范模式中的 `tcgen05.fence::after_thread_sync`，以及反向的 `tcgen05.ld` -> MMA 交接中的 `tcgen05.fence::before_thread_sync`，二者 CUTLASS 和 explicit.cu 都省略了，它们是 *tcgen05* fence，而不是 proxy fence，第 2.4.4 节）；`mbarrier.init` -> 远程使用（即 `fence.mbarrier_init.release.cluster`，第 2.10.1 节，这是一条操作 fence，而不是 proxy fence）。实测 (D.2)：`fence.proxy.async.shared::cta` 下沉为 `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`；在 explicit kernel 的 PTX 中有 12 处（8 个 epilogue subtile + 4 次 CLC 解码），G.3。

### 2.7.7 L2 提升与缓存提示

`CUtensorMapL2promotion` 指定 "the byte granularity at which L2 requests are filled from DRAM"（驱动 API）；这些 map 使用 `L2_128B`，因此一个未命中 L2 的 TMA 请求会填充 128 字节的缓存行。每条指令的 `.L2::cache_hint` 策略（这里是 `EVICT_NORMAL`）是关于逐出优先级的提示。之所以保留 CUTLASS 的默认值，是因为实测的 kernel 已经以 baseline 的速度运行 (G.2)；换一种提示将是一次优化实验，而不是语义要求（分析）。

---

## 2.8 深入 `mbarrier`

### 2.8.1 对象

PTX ISA 9.4，9.7.15.16.1：一个 mbarrier 是 `.shared` 中一个按 8 字节对齐的 `.b64`。9.7.15.16.3 ("Contents of the mbarrier object")："irrespective of the layout"，它 "keeps track of the following information: Current primary and conditional phases of the mbarrier object; Count of pending arrivals for the current phase of the mbarrier object; Count of expected arrivals for the next phase of the mbarrier object; Count of pending asynchronous memory operations (or transactions) tracked by the current phase of the mbarrier object. This is also referred to as tx-count." 取值范围（表 43，`.layout::v0`）：期望到达计数为 1 .. 2^20 - 1，待到达计数为 0 .. 2^20 - 1，**tx-count 为 -(2^20 - 1) .. 2^20 - 1**（有符号）。

初始化 (9.7.15.16.12, `mbarrier.init`)："The primary and conditional phases are initialized to zero. The expected arrival and pending arrival counts are initialized to count. The initial transaction count tx-count is initialized to zero." W5 (explicit.cu:286-294) 就是这条指令；kernel 在 prologue 中初始化 33 个屏障 (explicit.cu:1128-1166)。

### 2.8.2 操作

- **arrive-on** (9.7.15.16.9)："the pending arrival count is decremented by count. If the count argument is not specified, then it defaults to 1"，然后 "If the current phase has been completed then the mbarrier transitions to the next phase." W9 为本地形式，W10 为远程形式（`mapa` + `.shared::cluster`），W11 为经由掩码地址的远程形式。对于 `.shared::cluster` 屏障，"Sink symbol '_' is mandatory for the destination operand" (9.7.15.16.16)，因此 W10/W11 中有 `_`。
- **expect-tx** (9.7.15.16.6.1)："increases the tx-count of an mbarrier object by the value specified by expectCount"。`mbarrier.arrive.expect_tx` 执行 "an expect-tx operation ... prior to the arrive-on operation ... the count argument of the arrive-on operation is assumed to be 1" (9.7.15.16.16)。W12 是本地形式（那次 49152 字节的装填）；W13 是远程的、带谓词的形式（CTA 0 的 scheduler warp 的 lane 0..3 以 16 字节装填全部四个 CTA 的 `clc_full[idx]`，explicit.cu:842；第 2.12.2 节解释了为什么每个 CTA 的屏障都要装填）。这一族还有第三个成员，kernel 没有用到：裸的 `mbarrier.expect_tx`（不带 arrive-on 的 expect-tx），CUTLASS 中 beta != 0 时的 C 加载 producer 在其加载*之后*发射它，随后全部 32 个 lane 各自单独到达（笔记 7.2 和笔记 13 第 15 项）。在 beta = 0 时该路径在 explicit.cu 中不存在，所以 W12/W13 就是 kernel 的全部 expect-tx 词汇，二者都是 `arrive.expect_tx` 形式。
- **complete-tx** (9.7.15.16.6.2)："tx-count is decremented by completeCount"，然后该 phase 可以完成。由硬件对每个 TMA 加载 box（`.mbarrier::complete_tx::bytes`）和每个 CLC 响应（16 字节）执行。
- **phase 完成** (9.7.15.16.8)：phase 在 "when all of the following conditions are met: The count of the pending arrivals has reached zero. The tx-count has reached zero." 时完成。然后原子地 "The mbarrier object transitions to the next primary and conditional phases. The pending arrival count is reinitialized to the expected arrival count."

有符号的 tx-count 起着承重作用（对上述规则的分析）。一个 TMA box 可能在 leader 的 `arrive.expect_tx` 加上 49152 **之前**就已落地，并把 tx-count 驱动到零以下。这是合法的，而且 phase 仍然不能提前完成，因为待到达计数在 leader 的 arrive 之前一直保持为 1。CLC 屏障以同样的方式工作：它的远程 `arrive.expect_tx(16)` 和硬件的 `complete_tx(16)` 无论以哪种顺序都恰好一次地完成该 phase（第 4 章，第 4.2.2、4.4.1 和 4.4.4 节）。

### 2.8.3 phase、parity 与一个状态机算例

9.7.15.16.19（`test_wait`/`try_wait`）："The .parity variant of the instructions test for the completion of the phase indicated by the integer parity of the operand phaseParity, which denotes either the current phase or the immediately preceding phase of the mbarrier object. An even phase has integer parity 0 and an odd phase has integer parity of 1 ... The test_wait and try_wait operations are valid only for: the current incomplete phase, for which waitComplete returns False. the immediately preceding phase, for which waitComplete returns True." 以及："the use of the .parity variants of the instructions requires tracking the phase of an mbarrier object throughout its lifetime." 同一节中的操作数描述："Indicated by the 32-bit unsigned integer operand phaseParity, whose integer parity denotes either the current phase or the immediately preceding phase of the mbarrier object ... An even phase has integer parity 0 and an odd phase has integer parity of 1"。

算例：rank 0 的 mainloop `full[3]`（计数 1，49152 个 tx 字节），由 tile 0 的 k-tile `q = 3, 11, 19, ...` 使用（stage `s = q % 8`）。第一次使用的一种可能的交错次序：

```
 event                                              phase | pending | expected | tx-count | consumer try_wait.parity(0)
 ---------------------------------------------------+------+---------+----------+----------+----------------------------
 mbarrier.init [full[3]], 1                          |  0   |    1    |    1     |     0    | false (phase 0 current, incomplete)
 rank 2's A half lands in rank 0 (complete_tx 8192)  |  0   |    1    |    1     |  -8192   | false   (tx-count may go negative)
 rank 0 leader lane: arrive.expect_tx 49152          |  0   |    0    |    1     |  40960   | false   (pending 0 but tx != 0)
 rank 0's A half, rank 0's B half land (2 x 8192)    |  0   |    0    |    1     |  24576   | false
 rank 1's A, rank 3's A, rank 1's B land in rank 1   |  0   |    0    |    1     |  0       | -> phase completes:
   (all three credit rank 0's full[3])               |  1   |    1    |    1     |  0       | true    (phase 0 is now "immediately preceding")
 MMA warp issues 4 MMAs on stage 3, commits empty[3] |  1   |    1    |    1     |  0       |
 second use (q = 11): consumer waits parity 1 ...     |  1   |    1    |    1     |  0       | try_wait.parity(1) false until the second 49152 land
 ... completes                                       |  2   |    1    |    1     |  0       | try_wait.parity(1) true; next use waits parity 0
```

第 2 行是来自 rank 2 的 A multicast，后者是 rank 0 在另一个 pair 中的 `x` 伙伴，而不是它的 peer。peer 的 box，即落在 rank 1 中的那三个（它自己的 A 半块和 B 半块，以及 rank 3 的 A 半块，multicast 到 rank 1 和 3），是第 5 行的那些；三者全都通过加了掩码的操作数计入 rank 0 的屏障。

第 4 章，第 4.3 节推导了环的第 n 次使用的 stage 与 parity 公式；第 4.7 节为全部五条流水线（第 4.1 节的五个环）画出了 per-tile 时间线。

### 2.8.4 `try_wait`、挂起时间提示、`test_wait`

9.7.15.16.19："mbarrier.test_wait is a non-blocking instruction which tests for the completion of the phase. mbarrier.try_wait is a potentially blocking instruction which tests for the completion of the phase. If the phase is not complete, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes OR before the phase completes following a system-dependent time limit. The optional 32-bit unsigned integer operand timeHint specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit."

阻塞等待（W6，explicit.cu:297-311）就是 CUTLASS 的 `ClusterBarrier::wait`（barrier.h:410-430），其 asm 文本逐字相同（引文见第 6.3.3 节）。

`kWaitTicks = 0x989680 = 10,000,000` ns = 10 ms 就是 `timeHint` 操作数，在更早的 ISA 版本中名为 `suspendTimeHint`（explicit.cu:189；CUTLASS 注释，`include/cutlass/arch/barrier.h:414`："Arbitrarily large timer value after which try-wait expires and re-tries"）。为什么取这么大：线程应当保持挂起直到 phase 完成，而不是自旋并抢走同一 SM 上 MMA/TMA warp 的发射槽；时间限制（若提示被遵守则为 10 ms）只保证该循环最终会重新轮询（phase 完成时的唤醒是如何实现的：第 12 章，第 12.2.2 节）。实测的下沉结果（D.2）：`SYNCS.PHASECHK.TRANS64.TRYWAIT P, [addr], R`，parity 位于 `R` 的第 31 位，而重试循环 `@!P NANOSLEEP.SYNCS 0x989680; @!P SYNCS.PHASECHK.TRANS64; @!P BRA` 被放到了非内联位置。

kernel 中出现三种 wait 形式：阻塞的 W6；一次性的 **peek** W7（不带提示的 `try_wait.parity`，返回一个 token），用于在某个 stage 已经空闲时跳过阻塞等待（`producer_try_acquire`/`consumer_try_wait`，explicit.cu:732、:739、:916、:927、:1030）；以及 W8 `test_wait.parity`（非阻塞，从不挂起），只用在 CLC producer 尾段（explicit.cu:860）。三者都是 `.shared::cta`：等待总是在 CTA 自己的屏障上进行；只有 arrive 才会跨 CTA。

**默认的 `.sem` 与 `.scope`。** 三种形式写出来时都不带 `.sem`/`.scope`。9.7.15.16.19："If the .sem qualifier is absent, .acquire is assumed by default" 以及 "If the .scope qualifier is not specified then it defaults to .cta. In contrast, the .shared::<scope> indicates the state space where the mbarrier resides." 因此 kernel 的等待都是在 `.shared::cta` 屏障上、`.cta` 作用域下的 `.acquire` 等待，即 CUTLASS 的 `ClusterBarrier::wait` 形式（cutlass/arch/barrier.h:410-430）。释放的一侧则跨越 CTA：TMA 的 `complete_tx` "has .release semantics at the .cluster scope"，`tcgen05.commit` 的 arrive 处于 "cluster scope"，而 W10/W11/W13 的 arrive 是在 `shared::cluster` 地址上进行的，且大多来自其他 CTA；由于写出来时不带 `.sem`/`.scope`，这些 arrive 本身带的是 `mbarrier.arrive` 的默认值，即 `.cta` 作用域下的 `.release`（9.7.15.16.16："If the .sem qualifier is absent, .release is assumed by default" 以及 "If the .scope qualifier is not specified then it defaults to .cta"）。acquire 等待返回 true 时保证了什么，同一节中写得很明确："All cp.async.bulk asynchronous operations using the same mbarrier object requested prior, in program order, to mbarrier.arrive having release semantics during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread"（以及针对通用访问的类似条款）。ISA 没有说明为什么 `.cta` 作用域的 acquire 在这里就足够了。我能给出的理由（分析，未验证）是：等待者在等待之后读取的一切（A/B stage、CLC 响应槽位、TMEM stage、被释放的 smem）都位于等待方 CTA 自己的 SM 中，而远程的 arrive 本身并不携带数据。CUTLASS 的头文件没有提供其他 wait 形式（在 cb424739 处，`include/` 下全部四条 `mbarrier.try_wait`/`test_wait` asm 字符串都是不带 `.sem`/`.scope` 的 `.shared::cta`），所以每个基于它们构建的 Hopper 和 Blackwell kernel 使用的都正是这一形式，而 ISA 自己的示例展示的则是 `.relaxed.cluster` 等待搭配显式的 fence（第 2.4.4 节的 tcgen05 模式，以及 9.7.15.4 的 `membar`/`fence` 示例 `mbarrier.try_wait.relaxed.cluster.shared::cta.b64 complete, [addr], parity; fence.acquire.sync_restrict::shared::cluster.cluster;`，它来自第 2.10.1 节所引用的 `fence.mbarrier_init` 片段所在的那个示例块）。一个从零开始、相比 CUTLASS 先例更偏好 ISA 字面规定的编写者，可以写 `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64`，这是 ISA 列在其示例之中的一种形式（`mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 complete, [shMem], 0;`）。

---

## 2.9 地址：`shared::cta`、`shared::cluster`、`mapa` 与第 24 位

### 2.9.1 两个窗口，一个地址空间

PTX ISA 9.4，5.1.7："Additional sub-qualifiers ::cta or ::cluster can be specified on instructions with .shared state space to indicate whether the address belongs to the shared memory window of the executing CTA or of any CTA in the cluster respectively. The addresses in the .shared::cta window also fall within the .shared::cluster window ... Instruction mapa gives the .shared::cluster address of the corresponding variable in another CTA in the cluster." 9.7.10.26（`mapa`）："Get address in the CTA specified by operand b which corresponds to the address specified by operand a ... b is a 32-bit integer operand representing the rank of the target CTA."

因此，一个 CTA 自己的 32 位 `shared::cta` 地址已经是该 CTA 的一个有效的 `shared::cluster` 地址，而 `mapa` 把它变成另一个 CTA 中的同一偏移。这个地址就是 `__cvta_generic_to_shared` 所返回的；`smem_u32`（explicit.cu:258）镜像了 `cute::cast_smem_ptr_to_uint`（cute/arch/util.hpp:93-108）。W10（explicit.cu:357-367）的引文见第 6.3.6 节。

**分布式共享内存（DSMEM）。** CUDA Programming Guide（"Distributed Shared Memory"）使整个 cluster 的共享内存都可寻址：对通过 `mapa`（或 cooperative groups 中的 `cluster.map_shared_rank()`）得到的 `shared::cluster` 地址进行通用 `ld`/`st`/`atom` 是合法的。本 kernel 从不执行通用的 DSMEM 加载或存储。它只通过硬件单元或屏障操作触及远程共享内存：对另一个 CTA 的屏障执行 `mbarrier.arrive{.expect_tx}.shared::cluster`（W10、W11、W13），`cta_group::2` TMA 加载（A 的 multicast 写入，W21，以及两次加载在 pair leader 的 `full[s]` 上的 `complete_tx`，W21/W22，explicit.cu:770），`tcgen05.mma` 对 A/B stage 中 peer 的半块的读取（W29），`tcgen05.commit` 向其掩码中各 CTA 的屏障发出的 multicast arrive（W30），以及 CLC `::all` multicast 响应（W33）。第 2.9.4 节的退出规则适用于所有这些。

### 2.9.2 这台机器做什么（实测）

ISA 没有说 rank 在 `shared::cluster` 地址中是如何编码的。针对这套编译器和硬件，测量结果（D5、D.2、G.3）表明：

- rank `r` 的动态 smem 基址是 `0x400 + (r << 24)`：`0x00000400`、`0x01000400`、`0x02000400`、`0x03000400`；
- 对测试过的每一对 rank，都有 `mapa(base_r, t) == base_t`；
- ptxas 把基址物化为 `0x400 | (SR_CgaCtaId << 24)`（`S2UR UR6, SR_CgaCtaId; UMOV UR29, 0x400; ULEA UR29, UR6, UR29, 0x18`，笔记 D.1 D5 从 baseline 的 `sass_gemm.txt` 的 `0x440`/`0x470`/`0x4c0` 处引用了它；explicit kernel 有相同的下沉，带两条 `S2UR SR_CgaCtaId`，笔记 G.3），并把 `mapa` 下沉为将 rank 插入字节 3（`PRMT ..., 0x654, ...`），所以这一约定是编译器可见的：**rank 占据地址位 [24,32)**，而对于一个 4-CTA cluster，只有第 24-25 位会被置位。对更大的 cluster 这个字段有多宽，笔记并未确立：它在 D5 中的预测是 [24,28)，而笔记 6.2 根据四个实测基址写的是 "rank in bits [24,26)"。

后果：每个地址的低 24 位在四个 CTA 中都相同，所有 `OFF_*` 偏移都是相对于各 CTA 自己的基址的，而基址在四个 CTA 中并**不**是同一个数，这就是为什么 kernel 在运行时计算 `smem_base` 而从不把 `0x400` 硬编码（第 2.2 节）。

### 2.9.3 peer 位

第 24 位是 rank 的第 0 位，也就是 pair parity `x`。在一个奇数 CTA 的任何地址上把它清零，得到的就是其 pair 中偶数 CTA 里的同一偏移：即 pair leader，可用作 `shared::cluster` 地址（笔记 B.6）。CUTLASS 的常量与注释（`cute/arch/copy_sm100_tma.hpp:45` 和 :61-62）：`Sm100MmaPeerBitMask = 0xFEFFFFFF`；"Executed by both CTAs. Set peer bit to 0 so that the transaction bytes will update CTA0's barrier." Colfax 以同样的方式陈述这条规则："a CTA can find the address of its leader CTA's mbarrier by taking its own mbarrier address and setting the 24th bit to 0"。explicit.cu:187：

```
constexpr uint32_t kPeerBitMask      = 0xFEFFFFFFu;                  // cute::Sm100MmaPeerBitMask (copy_sm100_tma.hpp:45); nvcc folds it to 0xFEFFFFF8
```

两个使用者：

1. producer 用于两次 TMA 加载的屏障操作数（explicit.cu:770）：`full_masked_base = smem_base & kPeerBitMask`，因此无论由哪个 CTA 发射，`full_masked_base + 8 s` 都是 leader 的 `full[s]`（第 2.7.4 节）。
2. 累加器释放（W11，explicit.cu:370-379；cutlass/arch/barrier.h:905-921 `umma_arrive_2x1SM_sm0`；引文见第 6.3.7 节）。

算例。rank 3 的 `full[5]`：`0x03000400 + 8 x 5 = 0x03000428`；加掩码后：`0x03000428 & 0xFEFFFFFF = 0x02000428` = rank 2 的 `full[5]`。rank 2 的 `acc_full[1]`（explicit.cu:138 处的注释）：`0x02000400 + 240 + 8 = 0x020004F8`。编译器并不按原样发出这个掩码，而且折叠结果因位置而异（实测，笔记 D.6 第 2 项和 G.3）。在 epilogue 释放处（W11），PTX 常量是 `-16777224` = `0xFEFFFFF8`：nvcc 把 8 字节对齐的屏障地址中三个已知为零的对齐位折叠进了常量（SASS `0x44c0`）。在 producer 位置（explicit.cu:770），它把掩码施加在外提的、1024 字节对齐的 `smem_base` 上，并进一步多折叠了一位，得到 `0xFEFFFFF0`（PTX `-16777232`，SASS 中位于 `0x0c00` 的 `ULOP3 ... & 0xfefffff0`）；随后由一条 `ULEA` 加上 `8 s`，因此 baseline 中逐 k-tile 执行的 `ULOP3 ..., 0xfefffff8` 在 explicit kernel 中没有对应物，后者的 SASS 恰好只带有一次字面量 `0xfefffff8`，就在 epilogue 释放处（实测，G.3）。这三个掩码对本 kernel 的每个屏障地址的效果都与 `0xFEFFFFFF` 相同，因为它们额外清除的低位本来就是零：屏障是 8 字节对齐的，`smem_base = 0x400 + (r << 24)` 的第 0..9 位为零，而且 producer 的折叠是在 `ULEA` 加上 `8 s` 之前施加于 `smem_base` 的，所以 `8 s` 这一项（至多 56）从不会被掩码。baseline CUTLASS kernel 在其全部三个 PTX 位置都施加 `0xFEFFFFF8`：在两个 producer 位置是每个 k-tile 一次，在 epilogue 释放处是每个 tile 一次（D.2：ptx.txt:898、:1004、:4190 处的 `and.b32 %r, %r, -16777224`）。在 baseline 的 SASS 中，该释放是 `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R62+URZ], RZ`，其中 `R62 = (base + 0x110 + 8 st) & 0xFEFFFFF8`（笔记 D.2，baseline 的词典；`0x110 = 272`）。explicit kernel 在 `0x44f0` 处对 `[UR24]` 发出同一个操作码，位于 `0x44c0` 处的 `ULOP3 ... 0xfefffff8` 之后三条指令，二者之间是 subtile 7 的 `LDTM.x16` 和一条 `NOP`（实测，`sass_explicit.txt`；第 12 章，第 12.2 节的 W11 行）。explicit.cu:187 处的源码注释只写出了第一个值；核对 PTX 或 SASS 的读者应当预期看到这两个常量。

为什么这是安全的而非取巧之举（分析）：TMA 的 `cta_group::2` 规则只需要屏障所在 CTA 的 **parity**（"based on the %cluster_ctarank parity of the shared memory where the mbarrier object mbar resides"），而 pair 的定义是 "differs by the last bit only"。无论 rank 在地址中如何编码，leader 的屏障都是 rank 第 0 位被清零的那个 CTA 中同一偏移处的屏障；本实现恰好把 rank 的第 0 位放在了地址的第 24 位。针对另一套工具链编写的 kernel 应当重新测量这一点（第 13 章第 13.3 节末尾把它列在需要重新验证的假设之中）。

### 2.9.4 对描述符与退出的后果

- 描述符会丢掉 rank（第 2.6.1 节）：`((smem_base + OFF) >> 4) & 0x3FFF` 只看到地址位 [4,18)；explicit.cu:165 处的断言检查基址与大小保持在 2^18 以下，因此该字段不可能回绕。
- DSMEM 退出规则：CUDA Programming Guide（"Writing SIMT Kernels"、"Distributed Shared Memory"）："The user also needs to ensure that all distributed shared memory operations happen before the exit of a thread block." 而对于带 `.multicast::cluster::all` 的 CLC："The behavior of instruction with .multicast::cluster::all qualifier is undefined if any CTA in the cluster is exited"（9.7.15.18）。落到一个已退出 CTA 上的远程 mbarrier 到达，就是退出之后的 DSMEM 操作。这就是笔记 7.10 中四个尾段的硬件原因：每个 CTA 的 warp 2 把最后的 multicast commit 吸收进它的 `empty[]`（8 次等待），CTA 0 的 warp 1 吸收最后几次 800 次到达的释放（2 对 test/wait），leader 的 warp 0 吸收最后几次 256 次到达的释放（4 次等待），而 pair 握手为解除分配定序。第 8 章、第 7 章和第 9 章给出了它们的位置（第 8.6.8、7.6.8、9.6.3 和 9.6.4 节）。

---

## 2.10 cluster 屏障、命名屏障、`elect.sync` 与收敛

### 2.10.1 `barrier.cluster` 与 `fence.mbarrier_init`

其形式为 `barrier.cluster.arrive{.sem}{.aligned}`，其中 `.sem = {.release, .relaxed}`（默认 `.release`），以及 `barrier.cluster.wait{.acquire}{.aligned}`（默认 `.acquire`）。PTX ISA 9.4，9.7.15.3："barrier.cluster.arrive instruction marks warps' arrival at barrier without causing executing thread to wait for threads of other participating warps. barrier.cluster.wait instruction causes the executing thread to wait for all non-exited threads of the cluster to perform barrier.cluster.arrive ... The barrier.cluster.wait instruction guarantees that when it completes the execution, memory accesses (except asynchronous operations) requested, in program order, prior to the preceding barrier.cluster.arrive by all threads in the cluster are complete and visible to the executing thread ... The optional .relaxed qualifier on barrier.cluster.arrive specifies that there are no memory ordering and visibility guarantees provided for the memory accesses performed prior to barrier.cluster.arrive." 完成："When all non-exited threads in the cluster have executed `barrier.cluster.arrive`, the barrier completes and is automatically reinitialized"。`.aligned`："all threads in the warp must execute the same barrier.cluster instruction."

explicit.cu 使用 relaxed 的 arrive 与默认的（acquire）wait（同一节："If the optional .acquire qualifier is absent for barrier.cluster.wait, .acquire is assumed by default"）（W16/W17，explicit.cu:428-429，在第 6.5.1 节中引用）。

用 relaxed 就够了，因为在 wait 返回之前唯一必须在 cluster 范围内可见的东西，是那 33 个 `mbarrier.init` 的集合，而一个专用的操作 fence 会发布它们：`fence.mbarrier_init.release.cluster`（W14，explicit.cu:408-415），其语义是（9.7.15.4）"When .op_restrict is .mbarrier_init, the synchronizing effect of the fence only applies to the prior mbarrier.init operations executed by the same thread on mbarrier objects in .shared::cta state space." ISA 自己的示例（9.7.15.4）包含了这个 kernel 的 release 那一半，即同样的三个操作（ISA 的 `mbarrier.init [bar]` 省略了 `.shared::cta.b64` 与计数操作数，其 arrive 省略了 `.aligned`）：`// Release local barrier state via generic proxy. mbarrier.init [bar]; fence.mbarrier_init.release.cluster; barrier.cluster.arrive.relaxed;`。与之配对的 `barrier.cluster.wait.acquire;` 出现在同一示例中前面一个单独的片段里（`// Acquire remote barrier state via async proxy.`，其后跟着一条 `fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster`）；这两个片段合起来说明了 prologue 所用的模式。ISA 的 `barrier.cluster` 示例，标题为 "use memory fence prior to arrive for relaxed barrier"，给出了同一模式的一般形式 `fence.cluster.acq_rel; barrier.cluster.arrive.relaxed.aligned; ... barrier.cluster.wait.aligned;`：先用一个 fence 发布，然后以 relaxed 方式 arrive。拆分开的 arrive/wait 让 prologue 得以在二者之间计算 `%cluster_ctaid` 与初始 tile（explicit.cu:1170-1179），而 "no remote barrier operation before this point"（explicit.cu:1178 的注释）就是这次 wait 所建立的不变量。实测（D.2）：该 fence 不发出任何 SASS 指令（每组 init 之前有一条裸的 `FENCE.VIEW.ASYNC.S`；它们属于 init 的下沉而非属于该 fence，这一点是推断），arrive 是 `UCGABAR_ARV`，wait 是 `UCGABAR_WAIT; CCTL.IVALL`。

9.7.15.3 的另外两条规则对退出很重要："Each thread must arrive at the barrier only once before the barrier completes"，以及 "barrier.cluster instructions cause the executing thread to wait for all non-exited threads from its warp"。prologue 的会合是这个 kernel 唯一的 cluster 屏障：每个线程到达一次、等待一次。随后 rank 1-3 的 warp 1 以及每个 CTA 的 warp 3 退出（G.3：`0x1f00` 处的 `EXIT`），之后不再有 cluster 屏障；"non-exited threads" 这一措辞使之无害，而且后面也不存在任何可能等待它们的 `barrier.cluster`。

### 2.10.2 命名屏障

9.7.15.1："Each CTA instance has sixteen barriers numbered 0..15 ... Operand b specifies the number of threads participating in the barrier ... the value must be a multiple of the warp size ... barrier{.cta}.arrive does not cause executing thread to wait for threads of other participating warps ... When a barrier completes, the waiting threads are restarted without delay, and the barrier is reinitialized so that it can be immediately reused." `bar.sync` = `barrier.sync.aligned`，`bar.arrive` = `barrier.arrive.aligned`。

CUTLASS 保留了若干 id（cutlass/arch/barrier.h:169-178）：`EpilogueBarrier = 1`、`TransposeBarrier = 2`、`TransformBarrier = 3`、`StreamkBarrier0/1 = 4/5`、`TmemAllocBarrier = 6`、`Sm120MainloopBarrier = 7`，并警告（barrier.h:163-164）"ID 0 should not be used for safety, as other driver APIs (i.e. __syncthreads) may use it"。这条警告背后的规则是：编译器通常把 `__syncthreads()` 下沉为一个 CTA 屏障，一般是屏障 0，因此要把屏障 0 视为归编译器所有，不要在同一区域里把手写的 `bar.sync 0` 协议与 `__syncthreads()` 混用（NVIDIA 内部）。explicit.cu 使用了两个（explicit.cu:190-191）：`kBarEpilogue = 1`，128 个线程（warp 4-7，每个 subtile 两次：explicit.cu:1051 与 :1071），以及 `kBarTmemAlloc = 6`，160 个线程 = warp 0 的 32 次 `bar.arrive`（explicit.cu:887）+ warp 4-7 的 128 次 `bar.sync`（explicit.cu:997）。ISA 所描述的 producer/consumer 形式（"The producer threads execute barrier{.cta}.arrive to announce their arrival at the barrier and continue execution without delay"）正是 TMEM 地址的交接：warp 0 发布分配结果后继续去做它的 MMA，而 epilogue warp 则等待它一次。

W18/W19（explicit.cu:432-437）在第 6.5.2 节中引用。

### 2.10.3 `elect.sync` 与 warp 收敛

9.7.15.15："elect.sync elects one predicated active leader thread from among a set of threads specified by membermask. laneid of the elected thread is returned in the 32-bit destination operand d ... The predicate destination p is set to True for the leader thread, and False for all other threads ... The behavior is undefined if the executing thread is not in membermask. Election of a leader thread happens deterministically, i.e. the same leader thread is elected for the same membermask every time. The mandatory .sync qualifier indicates that elect causes the executing thread to wait until all threads in the membermask execute the elect instruction before resuming execution."

W2（explicit.cu:264-278），即 CUTLASS 的 `elect_one_sync`（cute/arch/cluster_sm90.hpp:180-197），在第 6.2.2 节中引用。

在 `membermask = 0xFFFFFFFF` 下，每次调用都必须由一个完全收敛的 warp 发出，explicit.cu 正是这样调用它的：`if (elect_one_sync())` 守卫只出现在角色 warp 的全部 32 个 lane 都会到达的位置（explicit.cu:740、:843、:932、:942、:944、:1125、:1136、:1146、:1156，以及 prologue 中 :1103 处由每个 warp 计算一次的 `lane_pred`）。确定性正是让发射 MMA 的 lane 与发射 commit 的 lane 成为同一个线程（第 2.4.3 节）的原因，即使这个 kernel 为二者分别重新调用 `elect_one_sync()`。

这个 kernel 所用的每一条 `.sync.aligned` 指令都要求同样的收敛纪律：`tcgen05.alloc/dealloc/relinquish_alloc_permit`、`tcgen05.ld`、`tcgen05.wait::ld`、`barrier.cluster.*.aligned`、`bar.sync`（= `barrier.sync.aligned`）。对于 tcgen05 族，ISA 说（9.7.18.7.1）："all threads in the warp must execute the same instruction. In conditionally executed code, the instruction should only be used if it is known that all threads in the warp evaluate the condition identically"。对于 `barrier{.cta}`，`.aligned` 规则是在 CTA 层面陈述的（9.7.15.1）："all threads in CTA will execute the same barrier{.cta} instruction. In conditionally executed code, an aligned barrier{.cta} instruction should only be used if it is known that all threads in CTA evaluate the condition identically, otherwise behavior is undefined"。这个 kernel 满足它，因为执行命名屏障的是由 warp-uniform 的角色索引选出的整个 warp，线程数（128、160）都是 32 的倍数。由此得出这个 kernel 结构上的两条规则：kernel 以一个 **warp-uniform** 的索引分发角色（`warp_idx_sync()`，W1，explicit.cu:261：`__shfl_sync(0xffffffff, static_cast<int>(threadIdx.x / 32), 0)` 让编译器把该值视为 uniform），并且没有任何角色函数*在这些指令周围*依据 lane 相关的控制流拆分其 warp。

lane 相关的 C++ 分支本身并不被禁止。被禁止的是在这样的分支里放一条 `.sync.aligned` 或 `elect.sync` 指令，或者一个在下一次集体操作之前尚未重新收敛的分支。explicit.cu 有若干这样的分支，它们全都只包含非集体的工作，并且全都立即重新收敛。`if (is_leader_lane) { mbar_arrive_expect_tx_local(...); }`（explicit.cu:736；`is_leader_lane` 在调用处 :1192 是 `lane_pred != 0u && is_leader`，其中 `lane_pred = elect_one_sync()` 在 :1103 处计算一次）位于 :740 处的 `if (elect_one_sync())` 之前（二者之间是三条非集体语句，:737-739）。上面列出的九个位置上的每一个 `if (elect_one_sync())` 分支体，本身就是一个由单个 lane 执行的 lane 相关分支。`if (warp_idx == 1 && lane_pred)` / `if (warp_idx == 3 && lane_pred)`（:1113、:1117）守卫 tensormap 预取，`if (warp_idx == 0 && lane_pred)`（:1165）守卫 `tmem_dealloc` 屏障的 init。CLC 装填中的 `lane < 4u`（explicit.cu:842）位于一对 `asm` `setp`/`@p` 之中（W13），并不是因为收敛需要如此，而是因为那是 CUTLASS 的 `ClusterTransactionBarrier::arrive_and_expect_tx(smem_ptr, bytes, cta_id, pred)` API（cutlass/arch/barrier.h:605-621），被逐字复制过来；在一个 W10 风格的 arrive 外面套一个 C++ 的 `if (lane < 4)` 同样合法。

---

## 2.11 uniform 数据通路，以及为什么 warp-uniform 发射很重要

自 Turing 起，SM 在向量数据通路之外还有一条标量的 *uniform 数据通路*（Turing 上仅限整数；Blackwell 的还有 FP32 算术），Turing 上每个 warp 有 64 个 uniform 寄存器（SASS 记法中为 `UR0`-`UR62` 外加 `URZ`），Blackwell 上每个 warp 有 80 x 32-bit 的可寻址 uniform 寄存器 `UR0`-`UR79`，`URZ` 则是一个不在该计数之内的独立常零寄存器，而 Turing 的数字把它包含在内（NVIDIA 内部；baseline 的 Blackwell SASS，`trace_out_v2/sass_gemm.txt`，使用了直到 `UR77` 的 uniform 寄存器，与此一致），在这条数据通路上，一条指令对 warp-uniform 的操作数每个 warp 执行一次（转述自 Jia、Maggioni、Smith、Scarpazza，*Dissecting the NVidia Turing T4 GPU via Microbenchmarking*，arXiv:1903.07486，第 3.5.2 节 "Uniform Registers"；在第 12.3.1 节中引用）。

在 Blackwell 上实测（笔记 D.2 与 G.3），TMA、MMA、commit 与 CLC 指令都是 **uniform 数据通路指令**：`UTMALDG.3D.MULTICAST.2CTA`、`UTMALDG.3D.2CTA`、`UTMASTG.3D`、`UTCHMMA.2CTA`、`UTCBAR.2CTA.MULTICAST`、`UGETNEXTWORKID.BROADCAST`、`UTMACCTL.PF`，带有 `UR` 操作数，每个 warp 发射一次 "regardless of how many lanes reach it"。因此 ptxas 把 producer、MMA 与 scheduler 循环中的每一个 `elect.sync` 都折叠掉了，而 warp 4 由 32 个 lane 执行的 TMA 存储每个 box 花费一条 `UTMASTG`（第 12 章，第 12.3 节）。

由此得出三个设计上的后果，每一个都在 explicit.cu 中可见：

1. **让发射操作数保持 warp-uniform。** 描述符基址、屏障地址、掩码、坐标与循环状态在 producer/MMA/scheduler warp 的每个 lane 中都相同，因此它们存放在 uniform 寄存器里，而每 k-tile 的循环保持很短：explicit kernel 的 producer 循环 37 条指令（快速路径上 33 条），其 MMA 循环 47 条，对比 baseline 的 31 条与 44 条（G.3），全部都在 tensor core 忙于前一个 k-tile 的 MMA 时发射（第 12 章，第 12.5.5 节的发射模型，在那里标为 [H]）。
2. **稀缺资源是 uniform 寄存器堆，而不是向量寄存器堆**，就这些 warp 而言（笔记 D.7，由 baseline 的 uniform 到向量的溢出推断；第 12 章，第 12.3.2 节）。explicit.cu 让循环不变量保持很少：它的热循环里没有这样的搬移，而且这个 kernel 使用 26 个向量寄存器（G.3）。
3. **选举仍然写在 PTX 里。** `elect.sync` 是 ISA 层面的契约（为 `tcgen05.mma` 这条被 ISA 赋予 "single thread semantics" 的指令提供一个发射 lane）；ptxas 在这一架构上把它移除，是一项你观察到的优化，而不是你所依赖的优化（分析，笔记 D.6 第 6 项）。

---

## 2.12 Cluster 启动控制（CLC）

### 2.12.1 语义

PTX ISA 9.4，9.7.15.18："The clusterlaunchcontrol.try_cancel instruction requests atomically cancelling the launch of a cluster that has not started running yet. It asynchronously writes an opaque response to shared memory indicating whether the operation succeeded or failed. The completion of the asynchronous operation is tracked using the mbarrier completion mechanism at .cluster scope ... On success, the opaque response contains the ctaid of the first CTA of the canceled cluster; no other successful response from other clusterlaunchcontrol.try_cancel operations from the same grid will contain that id." 响应 "will be 16-byte opaque value"；带有 `.mbarrier::complete_tx::bytes` 时，"complete-tx operation, with completeCount argument equal to amount of data stored in bytes, will be performed on the mbarrier"（因此 `kClcTxBytes = 16`，explicit.cu:181）。`.multicast::cluster::all`："the response is asynchronously written using weak async-proxy writes to the corresponding local shared memory addr of each CTA in the requesting cluster. The completion of the writes to addr of a particular CTA is signaled via a complete-tx operation to the mbarrier object on the shared memory of that CTA." 未定义行为规则："If the executing CTA has already observed the completion of a clusterlaunchcontrol.try_cancel instruction as failed, then the behavior of issuing a subsequent clusterlaunchcontrol.try_cancel instruction is undefined"，以及 (9.7.15.19) "If the request fails the behavior of clusterlaunchcontrol.query_cancel.get_first_ctaid is undefined." CUDA Programming Guide（"Work Stealing with Cluster Launch Control"）用 C++ 的措辞说了同样的事："Submitting another cancellation request after **observing** a previously failed request is _undefined behavior_" 以及 "Retrieving the thread block index of a failed cancellation request is Undefined Behavior"。它还给出了这一特性的引入时间："Cluster Launch Control is a feature introduced in the NVIDIA Blackwell GPU architecture (compute capability 10.0)."

CUTLASS 文档（"Blackwell Cluster Launch Control"）陈述了调度模型：一个 `ClcID` 是 "a coordinate from the 3D grid launched on GPU"；"A ClcID will be launched as a Worker when there are available resources. A ClcID can be queried by an existing Worker via clusterlaunchcontrol.try_cancel instruction. Every ClcID is guaranteed to be processed by either (1) or (2)"；"The most common reason of a decline is that all ClcIDs have been processed"；"Cluster launch control works on the granularity of clusters. For example, a 2x2 persistent worker cluster's query will consume 2x2 ClcIDs at once."

### 2.12.2 kernel 中的两条指令

W33 (explicit.cu:568-575)，即查询，由 CTA 0 的 warp 1 的被选中的 lane 每个 tile 发出一次（引文见第 6.8.2 节）。

两个操作数都是 CTA 0 自己的 `shared::cta` 地址：位于 `OFF_CLC_RESPONSE + 16 idx` 的 `clc_response[idx]`，按 ISA 的要求 16 字节对齐（"naturally aligned address of the 16-byte wide shared memory location"），以及位于 `OFF_CLC_FULL + 8 idx` 的 `clc_full[idx]`。`::all` multicast 把响应和 16 字节的 `complete_tx` 投递到全部四个 CTA 中的相同偏移处。这就是为什么 CTA 0 的 scheduler 必须先用 `arrive.expect_tx(16)` 装填 **每个** CTA 的 `clc_full[idx]`（W13，explicit.cu:842：lane 0..3，每个 rank 一个，经由 `mapa`），也是为什么 rank 1-3 根本不需要 scheduler warp。

W34（explicit.cu:579-595；:578 处的注释记录了 `x0/y0/z0` 的注意事项），即解码，由每个 CTA 的每个消费 warp 执行（引文见第 6.8.3 节）。

在 W34 中，`get_first_ctaid` 以 `p1`（`is_canceled` 谓词）为谓词条件，原因是 UB 规则。对于失败的响应，它的输出保持未写入（实测 D.3：在全部 165 次未取消的解码中，它们保留着上一个响应的 id），所以 `clc_consume` 只在 `valid` 之下读取 `x0/y0`（explicit.cu:691-694）。一个被取消的 2 x 2 cluster 的首 CTA id 是 cluster 对齐的（两者都是偶数，实测 D.3），并通过 `tm = 2 (y0 >> 1) + (x0 & 1) + x`、`tn = 2 (x0 >> 1) + (y0 & 1) + y` 映射到本 CTA 的 tile（explicit.cu:692-693；第 7 章从 AlongN 光栅推导出它）。按 ISA 的说法，响应是不透明的，但 SASS 显示了这个编译器如何解码它（D.2，从提取代码推断）：`is_canceled` = 32 位字 2 的第 0 位（`LOP3.LUT P, RZ, Rword2, 0x1`），`x0` = 字 0，`y0` = 字 1 & 0xffff，`z0` = 字 1 >> 16；在 MMA warp 中只有 valid 位保留了下来（它的 `get_first_ctaid` 在那里是死代码）。这对阅读 `launch0.csv` 有用，但绝不是 kernel 可以依赖的东西。

### 2.12.3 循环如何遵守 UB 规则

scheduler 在同一次迭代中发出查询 `t`，然后消费响应 `t`（explicit.cu:847，然后是 :853），并在第一个无效响应处退出。因此，查询绝不会在失败被*观察到*之后发出；当失败*发生*时最多只有一个在途，这是 ISA 所允许的。其他每个角色每个 tile 恰好消费一个响应，并在同一个无效响应处结束。throttle 流水线（计数 32/32）使 CTA 0 的加载者在查询 `t` 发出之前就开始 tile `t`，所以除了正在加载的 tile 之外，每个 cluster 最多只有一个查询在途（实测 D.3：1024 次查询中 0 次违规，GPU 范围内最多 25 个在途；第 7 章，第 7.3、7.6.7 和 7.6.9 节）。

实测的动态在第 12 章，第 12.6.3 节：33 个原生 cluster，991 个被取消的响应，其 id 为 33..1023 且每个恰好出现一次，外加 33 个未取消的，分发顺序大致按启动线性顺序递增，从查询到消费的延迟中位数在 CUTLASS trace 中为 1408 ns、在 explicit trace 中为 1024 ns，隐藏在一个 tile 之后（每个 tile 的时间也在那里），以及 trace 构建中每个 cluster 32/31/19 个 tile。kernel 绝不能依赖其中任何一项（D.7），只能依赖恰好一次的保证。

---

## 2.13 程序化依赖启动：`griddepcontrol`

PTX ISA 9.4，9.7.15.14："The griddepcontrol instruction allows the dependent grids and prerequisite grids as defined by the runtime, to control execution in the following way: .launch_dependents modifier signals that specific dependents the runtime system designated to react to this instruction can be scheduled as soon as all other CTAs in the grid issue the same instruction or have completed. The dependent may launch before the completion of the current grid. There is no guarantee that the dependent will launch before the completion of the current grid. Repeated invocations of this instruction by threads in the current CTA will have no additional side effects past that of the first invocation ... .wait modifier causes the executing thread to wait until all prerequisite grids in flight have completed and all the memory operations from the prerequisite grids are performed and made visible to the current grid." CUDA Programming Guide（"Programmatic Dependent Launch and Synchronization"）："when the secondary kernel is configured with _Programmatic Dependent Launch_, it must always use cudaGridDependencySynchronize or other means to verify that the result data from the primary is available." `cudaGridDependencySynchronize()` 本身 "will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory"。同一份指南说 `cudaGridDependencySynchronize` "block[s] until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory"，并且说 "The CUDA driver can launch the secondary kernel when all primary thread blocks have launched and executed cudaTriggerProgrammaticLaunchCompletion"。

W35/W36 (explicit.cu:599-600)，即 CUTLASS 的 `wait_on_dependent_grids` / `launch_dependent_grids` (cutlass/arch/grid_dependency_control.h:95-98, 85-88)，引文见第 6.9.1 节。

放置位置：`griddep_wait()` 是两个角色的第一条语句，即 producer（explicit.cu:764，在任何对 A/B 的 TMA 加载之前，而这些加载是 kernel 对全局内存仅有的读取）和 scheduler（explicit.cu:821，它本身不读取任何全局内存），恰好是 CUTLASS 放置它的位置（`sm100_gemm_tma_warpspecialized.hpp:618-620`："Ensure that the prefetched kernel does not touch unflushed global memory prior to this instruction"）。因此，前一个 kernel 对 A 和 B 的写入在第一次 TMA 读取之前就是可见的，而屏障初始化、TMEM 分配和 cluster 会合则与前驱的尾段重叠。warp 0 在它的最后一个 tile 之后发射 `griddep_launch_dependents()`（explicit.cu:950），即在本 CTA 没有更多 MMA 要发射的时候，让后继 kernel 在各 epilogue 排空的同时开始它的 prologue（CUTLASS 注释："Launching a dependent kernel too early can compete with current kernels, while launching too late can lead to a long latency"，以及位于 `sm100_gemm_tma_warpspecialized.hpp:778-780`（按 cb424739 编号）的 "Hint on an early release of global memory resources. The timing of calling this function only influences performance, not functional correctness."）。构建定义了 `CUTLASS_ENABLE_GDC_FOR_SM100=1`（笔记第 0 节；B.3.1 步骤 10，实测 D14），所以两者都被无条件地生成。explicit.cu:1299-1312 中的主机端启动只传递了 cluster 属性，没有传递 programmatic-stream-serialization 属性，所以在实测运行中不存在前置网格：`griddepcontrol.wait` 立即返回，而 `launch_dependents` 没有被指定的依赖方可以通知（分析：在没有 "as defined by the runtime" 的前置网格或依赖网格时，没有什么可等待或可释放的；指南没有明说这一点，作为引文未验证）。即使配置了 PDL，`launch_dependents` 也只是一个提示，"There is no guarantee that the dependent will launch before the completion of the current grid"，所以 kernel 只可能从中获益，而绝不能依赖它。实测的下沉结果（D.2，G.3）：`ACQBULK` 2 条，两条都在被执行的路径上（每个 CTA 的 warp 2 以及 CTA 0 的 warp 1），`PREEXIT` 1 条（warp 0）；baseline 的源码把这个 wait 放在 **三个** warp 角色中，即 warp 2（`sm100_gemm_tma_warpspecialized.hpp:620`）、scheduler（`:688`）和负责 epilogue *加载* 的 warp 3（`:810`，在 `else if (is_participant.epi_load)` 之下，在它为 `beta != 0` 对 C 的 TMA 加载之前），所以它在那条 warp-3 路径中有第三条从未执行的 `ACQBULK`（D.2），这就是那个 3 对 2 的差值。

---

## 2.14 映射：硬件特性 -> PTX -> wrapper -> warp

第 6 章的汇总表（第 6.10 节）逐列复现的那些行在此省略；那张表是完整的 W1-W37 清单。下面的各行承载第 6.10 节没有的细节：操作数立即数、每个 warp 触碰哪个屏障，以及 fence 的放置位置。

| 硬件特性 | PTX 指令（按 explicit.cu 中的写法） | Wrapper（explicit.cu 行号） | 使用者 |
|---|---|---|---|
| warp 身份，uniform 角色索引 | `shfl.sync.idx.b32`（经由 `__shfl_sync`） | W1 `warp_idx_sync` (:261) | 每个线程，prologue |
| warp leader 选举 | `elect.sync %rx\|%px, 0xffffffff` | W2 `elect_one_sync` (:264) | prologue；warp 2、warp 1（CTA 0）、warp 0 在每次 TMA/CLC/MMA/commit 发射时 |
| 带 10 ms 挂起提示的 mbarrier 阻塞等待 | `mbarrier.try_wait.parity.shared::cta.b64 ..., 0x989680` 循环 | W6 `mbar_wait` (:297) | 所有角色 |
| mbarrier peek（token） | `mbarrier.try_wait.parity.shared::cta.b64` 一次 | W7 `mbar_try_wait` (:314) | warp 2（empty）、warp 0（full）、warp 4-7（acc full） |
| mbarrier 非阻塞测试 | `mbarrier.test_wait.parity.shared::cta.b64` | W8 `mbar_test_wait` (:329) | CTA 0 的 warp 1，CLC 尾段 |
| 本地 arrive | `mbarrier.arrive.shared::cta.b64 _` | W9 `mbar_arrive_local` (:346) | CTA 0 的 warp 2（throttle full） |
| 经由 `mapa` 的远程 arrive | `mapa.shared::cluster.u32` + `mbarrier.arrive.shared::cluster.b64 _` | W10 `mbar_arrive_remote` (:357) | 每个消费 warp（CLC empty -> CTA 0）、warp 1（throttle empty）、warp 0（dealloc 握手） |
| 经由 peer 位在 pair leader 上 arrive | `and 0xFEFFFFFF` + `mbarrier.arrive.shared::cluster.b64 _` | W11 `mbar_arrive_cluster_masked` (:370) | warp 4-7（acc empty 释放） |
| 通用 -> 异步 proxy 排序 | `fence.proxy.async.shared::cta` | W15 (:418) | warp 4-7（TMA 存储之前）、每个消费 warp（CLC 解码之后） |
| TMA 2SM multicast 加载 (A) | `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint` | W21 (:450) | warp 2 的被选中的 lane，每个 CTA |
| TMA 2SM 加载 (B) | `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` | W22 (:463) | warp 2 的被选中的 lane，每个 CTA |
| 2SM tensor-core MMA | `tcgen05.mma.cta_group::2.kind::f16 [d], a-desc, b-desc, idesc, {8 x 0}, p` | W29 `umma_f16_2sm` (:517) | pair leader 的 warp 0 的被选中的 lane |
| MMA 完成时的延迟 arrive，multicast | `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [bar], mask` | W30 `umma_commit_mcast_2sm` (:532) | 同一个 lane（掩码 0xF 和 0x3 << 2y） |
| warp 级同步 | `bar.warp.sync -1`（`__syncwarp()`） | W37（:601 处的注释） | 分配之后的 warp 0 (:886)，每个 tile 的 warp 2 (:792) |

---

# 第 3 章 设计：从 8192^3 到 256x128x16，以及 Speed of Light

本章是自底向上的第 5-11 章的自顶向下配套篇。第 2 章引入了本章所依赖的机器事实（TMEM、tcgen05、TMA multicast、CLC）。

---

## 3.1 计算及其 layout

### 3.1.1 数学与内存

对于这次固定运行，kernel 计算 `D(m, n) = alpha * sum_k A(m, k) * B(k, n) + beta * C(m, n)`，其中 `M = N = K = 8192`、`L = 1`、`alpha = 1.0f`、`beta = 0.0f` 在运行时从参数块读取（笔记，第 0 节；第 5 章解释了为什么它们必须保持为运行时值）。harness 在 `70_blackwell_fp16_gemm.cu:103`、`:108`、`:113` 处声明这些类型：

```
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
```

用 CuTe 的词汇来说（笔记，第 3 节，源自 `make_cute_packed_stride`）：

| 张量 | 逻辑 shape | 紧凑 stride | 元素的字节地址 | 连续（"major"）mode |
|---|---|---|---|---|
| A | (M, K, L) | (K, 1, 0) | `2 * (m * K + k)` | K |
| B | (N, K, L) | (K, 1, 0) | `2 * (n * K + k)` | K |
| C, D | (M, N, L) | (1, M, 0) | `4 * (m + n * M)` | M |

"K-major" 的意思是 K 索引是单位 stride 的索引：A 的一行（固定 m）和 B 的一列（固定 n）各为 8192 个连续的 fp16 = 16384 字节。D 是 M-major 的：D 的一列（固定 n）是 8192 个连续的 fp32 = 32768 字节。这就是 BLAS 中经典的 "TN" GEMM（A 行主序，B 列主序，二者都以 K 为最内层），而它正是第 3.1.2 节称为天然的 UMMA 操作数 layout 的那对 layout：对于 swizzle 过的 K-major 操作数，描述符的 LBO 字段为 "not used, assumed to be 1"（PTX ISA 9.4，第 9.7.18.3.1.1 节）。

### 3.1.2 为什么 K-major 是天然的 UMMA 操作数 layout

一条 `tcgen05.mma` 并不从寄存器读取 A 和 B。本 kernel 使用 `_SS` 原子：两个操作数都位于共享内存中，指令通过两个 64 位的*共享内存描述符*读取它们（第 6 章，W29）。每个描述符指明一个起始地址、两个 stride（LBO 和 SBO）以及一个 layout 类型（即 swizzle）。ISA 也允许 A 来自张量内存（`_TS` 形式）："the A matrix has shape MxK, in either Tensor Memory or Shared Memory; the B matrix has shape KxN, in Shared Memory of the current CTA and optionally in peer CTA; the D matrix is of the shape MxN, in Tensor Memory"（PTX ISA 9.4，第 9.7.18.10 节）。该 kernel 不使用 TMEM-A 形式，因此下文全部针对 `_SS` 描述符路径。用 CUTLASS 自己的记法（`cute/arch/mma_sm100_desc.hpp:720-732`，注释块），硬件针对 128 字节 swizzle 的规范 K-major layout 是：

```
LayoutType::B128       : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))
```

读它时取 `T = 8`，即每 16 字节单元所含的 fp16 元素数。八个连续的行（M 或 N 索引）、每行 128 字节，构成一个 1024 字节的 swizzle 原子；`m` 个这样的原子彼此相距 `SBO`；在一行之内，各 16 字节 chunk 是连续的。该 kernel 的 128 x 64 fp16 A stage 恰好是 16 个这样的原子沿 M 堆叠而成，`SBO = 1024` B；而 64 x 64 的 B 半 stage 是 8 个这样的原子。由此得出四个事实（源自源码）：

- A 和 B 的 TMA box 是 (64 k, 64 行, 1)，带 `CU_TENSOR_MAP_SWIZZLE_128B`（笔记，第 5.3 节）。box 的内维是 64 个 fp16 = 128 字节，即 128 字节 swizzle 所允许的最大值（"the bounding box inner dimension to be <= 128"，`cuTensorMapEncodeTiled` 文档）；64 行 = 8 个原子；一个 box 是 8192 字节；
- 描述符的常量高字是 `0x40004040`（`explicit.cu:171`）：`SBO = 64` 个十六字节单元 = 1024 B、`version = 1`、`layout_type = 2 = SWIZZLE_128B`；低字携带 `LBO = 1`（对 swizzle 过的 K-major 不使用；`mma_sm100_desc.hpp:831-832` 将其设为规范的 `stride_10 = 1`）以及 14 位起始地址；
- 在一行之内推进一个 k-block（16 个 k 值 = 32 字节）是对低字做 `+2`，而推进一个 stage 是 `+1024` (A, 16384 B) 或 `+512` (B, 8192 B)，`explicit.cu:928-929` 和 `:933-934`（引文见第 9.5 节）；

- 指令描述符 `0x10200010`（`explicit.cu:173`）在第 15 位和第 16 位有 `a_major = 0` 和 `b_major = 0`（笔记，第 7.6 节），这正是 `Major::K` 所编码的。

CollectiveBuilder 从 layout 标签解析出 major 方向，`sm100_umma_builder.inl:201-202`（`tag_to_umma_major_A<RowMajor>` 为 K，`tag_to_umma_major_B<ColumnMajor>` 为 K），并从 `sm100_smem_selector` 的 K-major 分支解析出 smem 原子（`sm100_common.inl:115-118`：当 k-tile 是该原子的 64 个元素的倍数时为 `Layout_K_SW128_Atom`）。

**对于 MN-major 操作数会有什么变化**（源自源码注释块 `mma_sm100_desc.hpp:697-717` 和 MN 分支 `:767-798`；该 kernel 并未实现这一点，所以请把这份清单当作设计注记，而不是经过验证的代码）：

| 项目 | K-major（本 kernel） | MN-major（例如 ColumnMajor 的 A） |
|---|---|---|
| 规范描述符 layout (SW128) | `((8,m),(T,2)):((8T,SBO),(1,T))`：沿 M/N 的 8 行、每行 128 B，K 在一行之内连续 | `((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))`：64 个 M/N 元素（128 B）连续，每个原子 8 个 k 行 |
| 起作用的 stride 字段 | SBO = 1024 B（8 行一组）；LBO 固定为 1 | LBO = 64 元素 MN 块之间的 stride，SBO = 各 8-k 组之间的 stride（`:797-798` 相对于 `SWIZZLE_NONE` 交换了两者的角色） |
| k-block 推进 | 128 字节行之内的 `+32 B`（对低字做 `+2`） | SBO 的倍数，即由 8 个 k 行组成的各组之间的 stride：每个 16-k 块两组。SBO 取决于 1024 字节原子被平铺的顺序，而不只取决于 MN 范围。CUTLASS 的 SM100 collective 把 MN-major 操作数先沿 K 划分为 tile（`sm100_mma_warpspecialized.hpp:178-184`："Tile along K mode first before tiling over MN"，当 `is_mn_major<StrideA>()` 时取 `Step<_2,_1,_3>`），所以对任何 MN 范围，相邻的 8-k 组都保持相距 1024 B：SBO = 1024 B，一个 k-block 步进是 2 x 1024 = 2048 B，而 64 元素的 MN 块（对 64 宽的 k-tile 而言相距 8 x 1024 = 8192 B）进入 LBO。只有当一个 128 行的 A tile 先沿 MN 平铺（在 k 组之前先沿 M 放两个原子）时，SBO 才会是 2 x 1024 = 2048 B，k-block 步进才会是 4096 B。（`mma_sm100_desc.hpp:704`：在 `((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))` 中，m-mode 的 stride 为 LBO，k-mode 的 stride 为 SBO；`:797-798` 对 swizzle 过的 MN layout 把 `stride_11`，即 k 组 stride，赋给 SBO。） |
| TMA box | (64 k, 64 行)：内维 = K | (64 m, 64 k) 或 (64 n, 64 k)：内维 = M/N，仍为 128 B |
| builder 选择的 smem 原子 | `Layout_K_SW128_Atom`（`sm100_common.inl:115-118`） | `Layout_MN_SW128_Atom`（同一选择器的 MN 分支，`:89-114`） |
| 指令描述符 | 第 15/16 位 = 0 | 第 15 位 (A) 或第 16 位 (B) = 1 |
| `SWIZZLE_128B_BASE32B` | 对 Major-K 无效（`:811` static_assert） | 被 `make_umma_desc<Major::MN>` 接受（`:705`、`:775`），但对于 MN-major 操作数，硬件只允许 32 位类型使用它（PTX ISA 9.4，第 9.7.18.10.3 节，表 65：16 位 MN-major 为 "All except 128B swizzling with 32B atomicity"，32 位为 "Only 128B swizzling with 32B atomicity"），并且 builder 只在 `sizeof(ElementType) == 4` 时选择 `Layout_MN_SW128_32B_Atom`（`sm100_common.inl:91-93`），所以 fp16 的 ColumnMajor A 不会使用它。epilogue 的 fp32 M-major D stage（第 3.6 节）使用同一个 `Swizzle<2,5,2>` 原子，但作为 TMA 模式 `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`；它从不构建 UMMA 描述符（第 2.7.3 节） |

要记住的要点：对于 K-major 操作数，一个 k-tile 的每个 k-block 都是同一 swizzle 行之内的一个固定 32 字节偏移，所以一个 k-tile 的四条 MMA 只在一个 32 位描述符字中相差 `+2`。正是这一点把 MMA warp 的热循环保持在 47 条指令，其中 37 条是 uniform 数据通路的（笔记 G.3；第 12 章解释 uniform 寄存器、uniform 数据通路以及本章引述的 SASS 助记符）。硬件支持 MN-major，但代价是一种不同的描述符遍历和一张不同的 TMA map。

---

## 3.2 tile 层级

### 3.2.1 从问题到指令

图中用到的两个名字：`CX = blockIdx.x >> 1` 是 cluster 列（它索引 N），`CY = blockIdx.y >> 1` 是 cluster 行（它索引 M），因此 `blockIdx = (2 CX + x, 2 CY + y)`，并且由 `explicit.cu:1175-1176`（引文见第 11.6 节）可得 `tm = 2 CY + x`、`tn = 2 CX + y`。

```
  8192 x 8192 (M x N) output, K = 8192
  |
  |  32 x 32 = 1024 cluster tiles of 256 x 256     (one per CLC work item; note 7.4)
  v
  +----------------- 256 columns (N) -----------------+
  |  rank 0 (x=0,y=0)       |  rank 2 (x=0,y=1)       |   128 rows: tm even   = 2*CY
  |  CTA tile (tm, tn)      |  CTA tile (tm, tn+1)    |
  |-------------------------+-------------------------|
  |  rank 1 (x=1,y=0)       |  rank 3 (x=1,y=1)       |   128 rows: tm odd    = 2*CY + 1
  |  CTA tile (tm+1, tn)    |  CTA tile (tm+1, tn+1)  |
  +---------------------------------------------------+
        128 columns: tn even        128 columns: tn odd
        = 2*CX                      = 2*CX + 1

  x = %cluster_ctaid.x selects the M half (tm % 2 == x), y = %cluster_ctaid.y selects the N half.
  MMA pairs: {0,1} (y = 0) and {2,3} (y = 1): a pair shares tn and covers 256 rows x 128 columns.
  A-sharing: ranks x and x+2 share tm, i.e. the same 128 rows of A -> A is multicast across y.
  |
  |  128 k-tiles of 64 (q = 0..127), one smem stage s = q % 8 each
  v
  per k-tile per pair: 4 x tcgen05.mma.cta_group::2  256 (M) x 128 (N) x 16 (K), kb = 0..3
                       FLOP per MMA = 2 * 256 * 128 * 16 = 1048576
```

数字（推导）：1024 个 cluster tile x 128 个 k-tile x 4 条 MMA x 2 个 pair = 1048576 条 MMA，每条 2^20 FLOP，总计 2^40 = 2 x 8192^3 FLOP，即笔记第 9 节的台账行。

编码这一层级的常量位于 `util.hpp:29-35`：

```
constexpr int      kFixedM = 8192, kFixedN = 8192, kFixedK = 8192, kFixedL = 1;   // Section 0 shape
constexpr int      kCtaTileM = 128, kCtaTileN = 128, kTileK = 64;                  // Section 4 CTA tile / k-tile
constexpr int      kClusterX = 2, kClusterY = 2, kClusterZ = 1;                    // Section 2 cluster (2,2,1)
constexpr unsigned kGridX = kFixedN / kCtaTileN, kGridY = kFixedM / kCtaTileM, kGridZ = 1;   // (64,64,1) after the AlongN transpose (Section 7.4, B.1.2 step 2)
constexpr unsigned kBlockThreads = 256;                                             // Section 4
constexpr unsigned kSmemBytes = 230400;                                             // Section 6.2 SharedStorageSize
constexpr unsigned kProblemTilesM = kFixedM / 256, kProblemTilesN = kFixedN / 256, kProblemTilesL = 1;   // 32 x 32 cluster tiles (B.1.1 step 11)
```

以及 `explicit.cu:105-106`（`kMmaM = 256, kMmaN = 128, kMmaK = 16` 和 `kKBlocks = kTileK / kMmaK`；引文见第 5.5 节）。

harness 请求 `MmaTileShape_MNK = Shape<_256,_128,_64>` 和 `ClusterShape_MNK = Shape<_2,_2,_1>`（`70_blackwell_fp16_gemm.cu:123`、`:125`）。builder 通过除以 `AtomThrShapeMNK = (2,1,1)` 把 MMA tile 变成 CTA tile（`sm100_umma_builder.inl:229-230`）：2SM 原子沿 M 跨越两个 CTA，所以每个 CTA 拥有 128 x 128 x 64（笔记，第 4 节的表）。

**为什么每条指令 K = 16。** UMMA 的 K 范围固定为操作数类型的 256 位（`mma_traits_sm100.hpp:1174-1175`："Size of instructions's K extent is always 256bits"），即 16 个 fp16，所以 64 的 k-tile 是 4 条指令。权威表述是 PTX ISA 9.4 的表 48 "Various combinations of .kind and shapes"（第 9.7.18.2.1 节）。对于 `.kind::f16`、`cta_group::2`、稠密的情形，合法的 shape 是 128xNxK 和 256xNxK，其中 N = {16, 32, ..., 256} 以 16 为步长，K = 16（稀疏时 K = 32）。对于不带 `.ws` 的 `cta_group::1`，它们是 64xNxK 和 128xNxK，其中 N = {8, 16, 24, ..., 256} 以 8 为步长，K = 16。对于 `.kind::tf32`，同一张表给出 K = 8：在 sm_100a 上，每一种 kind 的 K 范围都是 256 位（表 48 还列出了更宽的稠密 K 值，`.kind::f8f6f4` 和 `.kind::mxf8f6f4` 的 K = 64，以及 `.kind::mxf4` 和 `.kind::mxf4nvf4` 的 K = 96 和 128，但其 Target ISA Note，9.7.18.2.1.1，把它们限定于 sm_103a、sm_107a 和 sm_107f 或更高），这正是 CuTe 注释为 f16 traits 所编码的内容。该 kernel 的 256 x 128 x 16 是 2SM 表中 N = 8 x 16 时最大的 M。

**为什么 k-tile 是 64 而不是 16。** TMA box 的内维必须是 16 字节的倍数，而 swizzle 原子需要完整的 128 字节行。64 个 fp16 = 128 字节恰好填满 SW128 行。这给出了 swizzle 所允许的最大 box，以及每字节最少的屏障往返次数：每 48 KiB 的 pair 数据一个 `full`/`empty` 周期，而不是每 12 KiB 一个。

### 3.2.2 每条 UMMA 读哪些字节

固定一个 pair {2y, 2y+1}、stage `s`、k-block `kb`。

| 操作数的一半 | 来源 CTA | 读取的共享内存字节（swizzle 前的地址，相对于该 CTA 的 `smem_base`） | MMA 行 / 列 |
|---|---|---|---|
| A 的行 0..127 | leader（rank 2y） | `OFF_SMEM_A + 16384 s + 128 m + 32 kb .. +32`，m = 0..127 (4096 B) | M 0..127 |
| A 的行 128..255 | peer（rank 2y+1） | peer 的 smem 中的相同偏移 (4096 B) | M 128..255 |
| B 的列 0..63 | leader | `OFF_SMEM_B + 8192 s + 128 n + 32 kb .. +32`，n = 0..63 (2048 B) | N 0..63 |
| B 的列 64..127 | peer | peer 的 smem 中的相同偏移 (2048 B) | N 64..127 |

**算例。** 位于启动线性坐标 `(CX, CY) = (4, 2)` 的 cluster：rank 0 计算 CTA tile `(tm, tn) = (4, 8)`，rank 1 `(5, 8)`，rank 2 `(4, 9)`，rank 3 `(5, 9)`（由 `explicit.cu:1175-1176` 结合 blockIdx = (2 CX + x, 2 CY + y) 得出）。pair {0,1} 的指令覆盖 D 的行 512..767（tm 4 和 5）和列 1024..1151（tn 8）；pair {2,3} 覆盖行 512..767 和列 1152..1279（tn 9）。A 的行（`a_row` 和掩码，explicit.cu:777 和 :772）：rank 0 (x = 0, y = 0) 把行 `128 * 4 + 0 = 512..575` 加载到 rank 0 和 2 的槽位 `+0`。rank 2 (x = 0, y = 1) 把行 `576..639` 加载到 rank 0 和 2 的槽位 `+8192`。rank 1 把 `640..703` 加载到 rank 1 和 3 的槽位 `+0`。rank 3 把 `704..767` 加载到 rank 1 和 3 的槽位 `+8192`。B 的列：rank 0 加载 `128 * 8 + 0 = 1024..1087`，rank 1 `1088..1151`，rank 2 `1152..1215`，rank 3 `1216..1279`（explicit.cu:778）。于是 {0,1} 的指令读取 A 512..767（leader 512..639，peer 640..767）和 B 1024..1151（leader 1024..1087，peer 1088..1151）：恰好就是它所累加的那个 256 x 128 块。

（`a_row` 和 `b_col`：`explicit.cu:777-778`，引文见第 8.6 节。）第 8 章逐条指令地走读这些加载；这里的要点是几何结构。同一条指令的 TMEM 一侧在第 2.5.1 节；TMEM 映射在第 9.3.2 节。

---

## 3.3 数据复用算术：为什么是 (2,2,1)

### 3.3.1 每 cluster 每 k-tile 的字节数

每个 k-tile，每个 CTA 都需要在自己的共享内存中拥有完整的 128 x 64 fp16 A stage (16384 B) 以及它供给 pair 的 64 x 64 B 半 stage (8192 B)。问题在于，为了把它们送到那里，必须*向 L2 请求*多少字节。四种设计（推导；该 kernel 实现的是最后一行）：

| 设计 | 每 CTA 向 L2 请求的 A 字节数 | 每 CTA 请求的 B 字节数 | 每 cluster（4 个 CTA）每 k-tile 的 L2 字节数 | 每 cluster 每 k-tile 的 FLOP | 每 L2 字节的 FLOP | 每 SM 每 k-tile 写入 + 读取的 smem 字节数（读取：一个下界，至少是 TMA 送达该 SM 的字节数，第 2.5.3 节） | 每 SM 每 smem 字节的 FLOP（每 CTA 每 k-tile 2097152 FLOP；在读取计数是一个界的情况下为上界） |
|---|---|---|---|---|---|---|---|
| 1SM MMA 128x128，无 multicast（"(1,1,1)"） | 16384 | 16384（完整的 128 x 64 B） | 131072 | 8388608 | 64 | 32768 + 32768 = 65536 | 2097152 / 65536 = 32 |
| 1SM MMA，A 在共享 tm 的 2 个 CTA 之间 multicast（"(1,2,1)"） | 8192 | 16384 | 98304 | 8388608 | 85.3 | 65536 | 32 |
| 2SM MMA 256x128，无 multicast（"(2,1,1)"） | 16384 | 8192 | 98304 | 8388608 | 85.3 | 24576 + 24576 = 49152 | 2097152 / 49152 = 42.7 |
| 2SM MMA + A 跨 y multicast（本 kernel，(2,2,1)） | 8192 | 8192 | 65536 | 8388608 | 128 | 49152 | 42.7 |

如何解读这张表：

- **2SM MMA** 使一个 CTA 必须持有的 B 字节数减半：N = 128 在 pair 内按 64/64 划分，每个 CTA 只存储自己的 64 列。这既节省了 L2 请求，*也*节省了共享内存写入：TMA 送达每个 SM 的操作数字节数从每 k-tile 32 KiB 降到 24 KiB。表中 smem 列的读取项（以及那个 42.7）沿用第 2.5.3 节的只读一次模型，而 `cta_group::2` 对 B 并不意味着该模型，所以它是一个界，而不是经过验证的流量（NVIDIA 内部）。Colfax 对 256 x 256 shape 陈述了同样的效应："the 256x256 MMA does the same number of FLOPs but transfers half the operand data"，相对于两条 128 x 256 MMA 而言。
- **A multicast** 使 A 的 L2 请求减半：共享 tm 的两个 CTA（rank x 和 x + 2）各自从 L2 取回一个 64 行的半块，TMA 单元把它写入两个 CTA 的 smem。它并不减少 smem 流量：两个 CTA 每 k-tile 仍然各收到 16 KiB 的 A。请把两个每字节 FLOP 列放在一起读。multicast 改变的是每 L2 字节 FLOP 列（64 -> 85.3 -> 128），而不动每 smem 字节 FLOP 列（32 仍是 32）；2SM MMA 是这个设计空间中唯一能改善共享内存复用的机制（32 -> 至多 42.7：读取计数假定第 2.5.3 节的只读一次模型，而 `cta_group::2` 并不意味着该模型，NVIDIA 内部信息）。第 3.9.3 节解释了为什么对 128 x 128 的 CTA tile 而言，重要的是那第二列。
- 二者合起来达到每 L2 字节 128 FLOP，是朴素设计的两倍。实现该 multicast 的掩码是 `explicit.cu:772`：

```
  uint16_t const a_mcast_mask     = static_cast<uint16_t>(0x5u << x);            // ranks x and x + 2 (create_tma_multicast_mask<2>)
```

  `0x5 = 0b0101` 选中 rank 0 和 2；按 x = 1 移位后它变成 `0xA = 0b1010`，即 rank 1 和 3。只要 cluster 的 N 范围不为 1，builder 就为 A 选择 multicast 拷贝（`sm100_common.inl:203-214`）；而当 cluster 的 M 范围恰为 2 时，为 B 选择普通的 2SM 拷贝（`:239-250`）：沿 M 只有一个 pair 时，不存在拥有相同 tn 的第二个 pair 可与之共享 B。

### 3.3.2 为什么不选其他 cluster shape（分析）

- **(1,1,1)**：没有 pair，没有 multicast；每 L2 字节 64 FLOP，每 SM 每 k-tile 32 KiB 的操作数 smem 写入。第 3.9 节论证（假设 3.9.3 和 3.9.4），即使是 (2,2,1) 设计也已接近共享内存与 L2 带宽的极限，因此 (1,1,1) 将大幅受限于带宽。
- **(2,1,1)**：pair 存在，但没有任何东西被 multicast。每 FLOP 的 L2 字节数是 (2,2,1) 的 1.5x。它还把 cluster 大小减半，这对驻留略有帮助（第 3.8 节），但实测的 SM 放置提示，驻留受限于 GPC 内部的 TPC 配对，而不是 cluster 数量。术语：一个 TPC（texture processing cluster）是一对 SM `(2j, 2j+1)`；一个 GPC（graphics processing cluster）是一组 TPC，构成一个 cluster 启动域；一个 2 x 2 cluster 需要同一个 GPC 内的两个完整 TPC。每 TPC 2 个 SM 以及每 GPC 的 TPC 数量（GPC skyline）是 NVIDIA 内部信息，与 pair 放置以及笔记 D.3 中的 16 个空闲 SM 一致；第 2.1.1 节给出了机器图景与算术。
- **(1,2,1)**：有 multicast 而没有 pair。L2 字节数与 (2,1,1) 相同，但每 SM 每 k-tile 有全部 32 KiB 的操作数 smem 写入，以及 M = 128、N = 128 的 1SM MMA（除 64 之外，1SM F16 原子所允许的唯一 M 值，`mma_sm100_umma.hpp:88`）。
- **(4,1,1)**：沿 M 堆叠的两个 pair 共享 tn，因此可以 multicast B 而不是 A。每个 CTA 将请求 A 16384（它自己的 128 行，无共享）+ B 4096（128 列的四分之一，multicast 给另一个 pair）= 20480 B，比 16384 更差，因为在这个 tile shape 中 A 是每 CTA 更大的那个操作数，而从共享中获益的正是 A。builder 的规则（`sm100_common.inl:239-250`）确实会在 cluster M = 4 时把 B 切换为 multicast，所以这个设计点是真实存在的；它只是对 128 x 128 的 CTA tile 而言不是更好的那个。
- **更大的 cluster**（(2,4,1)、(4,2,1)）：更多共享，但 8 个 CTA 必须共同驻留在同一个 GPC 中。实测的放置（笔记 D.3）表明，即使是 4-CTA 的 cluster 也会让 148 个 SM 中的 16 个空闲（原因，鉴于这块 B200 的 skyline，第 2.1.1 节：每个奇数大小的 GPC 都会留下一个未配对的 TPC；NVIDIA 内部信息）；8-CTA 的 cluster 会量化得更差。假设，此处未实测。

### 3.3.3 CollectiveBuilder 解析出什么（笔记，第 4 节），以及每个 `Auto` 背后的规则

| 请求项 | 解析结果 | 规则（已在源码中验证） |
|---|---|---|
| 带静态 cluster 的 `KernelScheduleAuto` | 2SM 路径，`SM100_MMA_F16BF16_2x1SM_SS<half_t, half_t, float, 256, 128, Major::K, Major::K>` | 当调度标签派生自 `KernelSchedule2Sm` 时，或者当它既不派生自 `KernelSchedule1Sm` 也不派生自 `KernelSchedule2Sm`，并且 cluster 为静态且 cluster M 为偶数时，`is_2sm` 为真（`sm100_umma_builder.inl:208-212`）。harness 的 `KernelScheduleAuto` 属于第二种情况。在 `KernelScheduleAuto` 下，当 cluster M % 2 == 0 且 tile M % 128 == 0 时选用 2SM 原子（`sm100_common.inl:455-464`） |
| UMMA major | A: K, B: K | 对 RowMajor / ColumnMajor 应用 `tag_to_umma_major_A/B`（`sm100_umma_builder.inl:201-202`） |
| `AtomThrShapeMNK`、`CtaTileShape_MNK` | (2,1,1), (128, 128, 64) | 原子的线程 layout 的第一个 mode；tile / 原子 shape（`:229-230`） |
| GmemTiledCopyA / B | `SM100_TMA_2SM_LOAD_MULTICAST` / `SM100_TMA_2SM_LOAD` | A：cluster N != 1 -> multicast（`sm100_common.inl:203-214`）；B：cluster M == 2 -> 无 multicast（`:239-250`） |
| SmemLayoutAtomA / B | `Layout_K_SW128_Atom<half_t>` | K-major，且 BLK_K (64) 能被原子的 64 个元素整除（`:115-118`） |
| AccumulatorPipelineStageCount | 4 | `min(4, 128 * 512 / (128 * 128)) = min(4, 4)`（`sm100_umma_builder.inl:260-267`；第 3.4.2 节） |
| SchedulerPipelineStageCount | 2 | 非分组 GEMM 为 2（`:282`） |
| KernelSmemCarveout | 192 B | kernel 级屏障、CLC 响应与 TMEM 基址指针存储之和（`sm100_pipeline_carveout.inl:45-83`；第 3.4.1 节） |
| PipelineStages（`StageCountAutoCarveout`） | 8 | `floor((232448 - 192 - 33792) / 24592)`（`sm100_umma_builder.inl:84-98`，容量见 `arch.h:44`） |
| `EpilogueScheduleAuto` | `TmaWarpSpecialized2Sm` | 静态且为偶数的 cluster M，并且 MMA tile M 属于 {128, 256}（`sm100_builder.inl:1697-1715`）；对于 `OpClassBlockScaledTensorOp`，只有 MMA tile M = 256 才符合条件（`:1708-1710`），当你转向 FP8/FP4 块缩放 GEMM 时这一点很重要 |
| `EpilogueTileAuto` | (128, 16) | 第 3.6 节（`sm100_builder.inl:1006-1046`） |
| EpiTiles、FragmentSize | 8, 16 | `(128 * 128) / (128 * 16)`、`(128 * 16) / 128`（`:1253-1255`） |
| Epilogue DispatchPolicy | `Sm100TmaWarpSpecialized<StagesC = 4, StagesD = 2, 16, ReuseSmemC = true, DelayTmaStore = false>` | `StagesD = min(EpiTiles, 2)`、`StagesC = max(min(EpiTiles, 4), StagesD + 1)`，因为 C 宽于 8 位（`:1107-1115`） |
| TileScheduler | `PersistentTileSchedulerSm100<Shape<_2,_2,_1>, 2>`，动态持久化 | `void` 标签映射到 SM100 上的 CLC scheduler（笔记，第 4 节） |

explicit kernel 把这些结果硬编码了下来，其中大多数是 `explicit.cu:104-120` 处的常量（其余在别处，例如描述符常量 `:171-173` 中的 K-major SW128 layout，以及掩码 `:772` 中的 multicast 选择），并用 `static_assert` 检查其中那些有算术后果的项（`:158-166`、`:174-177`）。对你自己的 kernel 而言，教训不在于这些数字，而在于*规则*：每个 `Auto` 都是操作数类型、tile 与 cluster 的一个小函数，在信任 builder 之前，你应当能够手工把它算出来。

---

## 3.4 预算

### 3.4.1 共享内存：230400 字节，以及为什么是 8 个 stage

B200 为每个 CTA 提供 232448 字节的可选启用的动态共享内存（在笔记第 0 节中实测；`cutlass::arch::sm100_smem_capacity_bytes = 232448`，`arch.h:44`），其下方还有 1024 字节系统保留的共享内存（实测基址 `0x400`，笔记 D.1 D5）。builder 的 carveout 算术（笔记，第 6.1 节；`sm100_compute_stage_count_or_override`，`sm100_umma_builder.inl:84-98`，由 `:284-290` 处的 `KernelSmemCarveout` 提供输入；`sm100_pipeline_carveout.inl:45-83`）：

```
stage bytes      = 128*64*2 (A) + 64*64*2 (B) + 16 (one PipelineTmaUmmaAsync<1> barrier pair) = 24592
kernel carveout  = acc pipeline 64 + CLC pipeline 32 + load-order 16 + tmem-dealloc 8 + throttle 32
                   + CLC responses 2*16 + TMEM base pointers 2*4                            = 192
epilogue storage = TensorStorage 33280 + LoadPipeline 64 = 33344, rounded to 512-B alignment = 33792
stages           = floor((232448 - 192 - 33792) / 24592) = floor(198464 / 24592) = floor(8.07) = 8
```

这个 16 是 `sizeof(PipelineTmaUmmaAsync<1>::SharedStorage)`：每个 stage 一个 8 字节的 `full` 屏障和一个 8 字节的 `empty` 屏障。TMEM 指针的 `2*4` 是 builder 的估计；真实的结构体有一个 4 字节的 `tmem_base_ptr`。界限：8 x 24592 = 196736 <= 198464 < 9 x 24592 = 221328。

真实的 layout（笔记，第 6.2 节）复现于 `explicit.cu:139-155`（`OFF_*` 常量，引文见第 5.7 节）；以字节映射的形式：

```
  byte     0 .. 343   43 mbarrier slots (8 B each): 33 initialised; the 10 at bytes 128..207 (epi_load full/empty 4 + 4, load_order 2)
                      are reserved for the beta != 0 path and never initialised (explicit.cu:141-143; note E.10 item 6)
  byte   344 .. 351   padding (PipelineStorage is aligned_struct<16>)
  byte   352 .. 383   2 x 16 B CLC response slots
  byte   384 .. 387   TMEM base pointer, written by tcgen05.alloc
  byte   388 .. 511   padding to the epilogue's 512-B alignment                            [ 388 B used ]
  byte   512 .. 33279 4 epilogue C/D stages x 8192 B  (Swizzle<2,5,2>, needs 512-B alignment)
  byte 33280 .. 33791 FusionStorage (empty) padded to the next 512-B boundary
  byte 33792 .. 164863  8 A stages x 16384 B          (Swizzle<3,4,3>, needs 1024-B alignment)
  byte 164864 .. 230399 8 B stages x  8192 B          (idem)
  byte 230400           end; 232448 - 230400 = 2048 B unused
```

`explicit.cu:158-163` 处的各个 `static_assert`（在第 5.7 节中引用）锁定了这些算术与对齐。

为什么是这些对齐（推导）：`Swizzle<3,4,3>` 把地址位 [4,7) 与位 [7,10) 做 XOR，因此其模式每 1024 字节重复一次。两个硬件单元都以绝对地址位为键来做 XOR，而 kernel 的 UMMA 描述符把 3 位的 `base_offset` 字段留为 0（`kSmemDescHi = 0x40004040`，`explicit.cu:171`），因此 stage 基址必须 1024 字节对齐，否则 TMA 单元写入侧的 XOR 与 UMMA 单元读取侧的 XOR 就不会作用于相同的位（第 8 章，第 8.8.3 节）。`Swizzle<2,5,2>` 把位 [5,7) 与位 [7,9) 做 XOR：周期 512 字节，因此 epilogue stage 需要 512 字节对齐。上面的偏移同时满足二者，但这仅仅是因为动态基址 `0x400` 本身就是 1024 对齐的。这就是 kernel 不得有任何静态 `__shared__` 数据的原因：其放置并不是源码级的 ABI 契约（NVIDIA 内部），因此它可能会移动动态基址（笔记，第 6.2 节；第 11 章，第 11.2.3 节）。另一个约束是 14 位的描述符起始地址字段，它覆盖 2^18 = 256 KiB 的 shared::cta 地址空间：`static_assert(kSmemBaseLow24 + kSmemBytes < (1u << 18), ...)`（`explicit.cu:165`；第 9 章，第 9.4.4 节）。

**为什么 230400 恰好剩下 2048 字节**（推导）。把 builder 的估计与真实 layout 并排放在一起：

```
                      builder estimate                              real SharedStorage
  non-stage bytes     192 (kernel carveout: barriers, CLC slots,     512 (bytes 0..511: 43 barrier slots, CLC slots,
                           TMEM pointers)                                 TMEM pointer, padding)
                    + 33792 (epilogue: 33344 rounded up to 512)    + 33280 (bytes 512..33791: 4 x 8192 D stages
                                                                          + FusionStorage pad to 512)
                    = 33984                                        = 33792
  stage bytes         8 x 24592 = 196736 (each stage carries         8 x 24576 = 196608 (the 8 full + 8 empty
                      16 B of PipelineTmaUmmaAsync<1> storage)        barriers live in the 512-B head instead)
  total               230720                                        230400
  capacity 232448     - 230720 = 1728 (floor remainder)             - 230400 = 2048
```

320 字节的差额是 192 + 128。192 是 carveout，真实的头部把它吸收了：512 + 33280 等于 builder 仅 epilogue 一项的 33792 估计值。128 是 8 x 16 B 的 per-stage 屏障存储，它被计入 stage 大小之内，却被布置在头部。1728 + 320 = 2048。这一余量是量化的结果，而非设计选择：第 9 个 stage 将需要再多 24576 字节。

**究竟为什么要用 stage 把共享内存填满？** 一旦占用超过 SM 容量的一半，occupancy 就是每 SM 1 个 CTA，而这个 kernel 的 232448 字节中的 230400 字节确实如此，因此剩余的共享内存没有其他 consumer。每增加一个 stage 都免费多出 24 KiB 的在途预取（第 3.5.1 节）。这就是 `StageCountAutoCarveout` 所编码的规则：拿走 epilogue 与屏障剩下的一切。

### 3.4.2 TMEM：512 列 = 4 个 stage x 128 列

张量内存是每 SM 128 个 lane x 512 列、每列 32 位（第 2.3.1 节；`tmem_allocator_sm100.hpp:45-46`："128 DP x 512 COL x uint32_t-addressing"；`:120` 处的 `Sm100TmemCapacityColumns = 512`；MLC 一章："Each CTA has 128 lane rows and up to 512 32-bit columns"）。kernel 分配了它的全部（`explicit.cu:114`）：

```
constexpr uint32_t kTmemColumns   = 512;             // tcgen05.alloc / dealloc column count
```

stage 大小的推导：CTA 的累加器 tile 是 128 行 x 128 列的 fp32。MMA 把第 m 行写到 TMEM 的 lane m（lane = 行，第 2.5.1 节），因此 128 行恰好用满 128 个 lane，而每个 fp32 列占据一个 32 位的 TMEM 列：每 stage 128 列。512 / 128 = 4 个 stage。builder 以 `TotalTmem / (CtaTileM * CtaTileN) = 65536 / 16384 = 4` 计算出同样的结果，然后以 4 为上限（`sm100_umma_builder.inl:260-267`）。该结果是 mainloop 分发策略 `MainloopSm100TmaUmmaWarpSpecialized<8,2,4,(2,2,1),Sm100>` 的第三个参数：`include/cutlass/gemm/dispatch_policy.hpp:1025` 处的 `int AccumulatorPipelineStageCount_,`（范围 `:1021-1035`；笔记 D.1 D1）。kernel 的地址算术是同一个陈述，`explicit.cu:1006` 与 `:1031`：

```
  uint32_t const tmem_lane_off = (32u * w) << 16;                                               // TMEM lanes [32 w, 32 w + 32): row 32 w + l of the CTA tile (0, 0x200000, 0x400000, 0x600000)
```

```
    uint32_t const taddr = T + tmem_lane_off + 128u * acc_stage;                                // warp-uniform TMEM address of this warp's 32 rows
```

256 x 256 的 pair tile 会有什么代价：CTA tile 变为 128 x 256，即每 stage 256 个 fp32 列，且 512 / 256 = 2 个累加器 stage。这就是更大 tile 的 TMEM 代价；第 3.9.7 节把它与其带宽收益放在一起权衡。

---

## 3.5 流水线深度及其理由

### 3.5.1 八个 mainloop stage：Little 定律

producer warp（第 8 章）让最多 8 个 k-tile 的 TMA 加载在途，领先于 MMA warp（第 9 章）。第一次 `load_ktiles` 调用覆盖 `min(8, k_tiles)` 个 k-tile，第二次覆盖其余的（`explicit.cu:788-791`，在第 8.6 节中引用，并在第 8.6.5 节中讨论）。

所需的深度由 Little 定律得出：*在途字节数 = 摄入速率 x 延迟*。实测的各项输入（笔记 D.4，CUTLASS baseline；explicit kernel 自己的中位数给出 27.79 us，多 0.14 %，第 3.9.2 节）：在 32-tile 的关键路径上每 tile 至多 27.75 us，即在 1965 MHz 下每 k-tile 216.8 ns = 426 个周期。推导：

```
ingest rate per SM     = 24576 B per k-tile / 216.8 ns          = 113 GB/s
bytes in flight        = 7 stages ahead x 24576 B               = 172032 B   (8 stages: 196608 B)
latency tolerance      = 172032 B / 113 GB/s                    = 1.52 us    (8 stages: 8 x 216.8 ns = 1.73 us)
prefetch horizon       = 8 x 426 cycles                         = 3408 cycles = 1.73 us
```

在理想的每 k-tile 256 个周期下（第 3.9.2 节），同样的 8 个 stage 跨越 8 x 256 = 2048 个周期 = 1.04 us。

因此这个环能够隐藏任何短于约 1.5 us 的 TMA 往返：发射 -> L2 或 HBM -> smem 落地 -> `full[s]` 完成，再加上 `tcgen05.commit` -> `empty[s]` 的返回路径。Chips and Cheese 在 B200 上实测，到直连 L2 分区约 150 ns（"L2 latency to the directly attached L2 partition is similar to prior generations at about 150 ns"；这是第三方测量，并非在这台机器上）。跨分区命中与 HBM 未命中更长：该文章报告，当占用溢出到另一个分区时延迟 "dramatically increases"，而 HBM 未命中则是微秒量级（数量级陈述，未实测）。因此两个 stage（0.22-0.43 us，一到两个 k-tile 时间）就已足以覆盖本地 L2 命中；4 个 stage（0.65-0.87 us）对远分区与 HBM 流量而言处于临界；8-stage 的环（1.5-1.7 us）是按远分区与 HBM 未命中、再加上 commit 操作 -> `empty` -> producer -> TMA 返回路径来定尺寸的。

这个环除了内存延迟之外还吸收第二样东西：跨 CTA 的偏斜。leader 的 `full[s]` 由来自全部四个 CTA 的 producer 的六个 box 计入（它自己的 A 半边和 B 半边、同胞的 multicast A 半边，以及 peer 经由 peer 位掩码地址送来的三个 box；第 3.11 节），因此在四个 producer 中*最慢*的那个发射它之前，一个 stage 不会变满。正是这个深的环让每 CTA *单个* producer 线程就能喂饱 tensor core。

你将如何确认：Nsight Compute 针对 warp 0 的 warp 停顿分解。一个经常失败的 `full[s]` 等待会表现为 mainloop 等待的 `SYNCS.PHASECHK.TRANS64.TRYWAIT` / `NANOSLEEP.SYNCS` 重试循环上的停顿采样：`NANOSLEEP` 重试归在 `sleeping` 原因下（Profiling Guide 在那里点名了 `NANOSLEEP`）；`TRYWAIT` 等待路径归在 `long scoreboard` 下（NVIDIA 内部：在一个被考察的 SM100 warp 特化 GEMM 中，Nsight 把它采样在那里；这是一种依赖/等待路径的归因，而不是 `TRYWAIT` 指令本身就是根本原因的证明，因为 PC 采样可能转而记到其后带谓词的指令或控制指令头上，在这里就是重试循环的 `@!P0 NANOSLEEP.SYNCS` 或内联尝试之后的 `@!P0 BRA`，二者都在第 12.2.2 节中；第 12 章解释这些 SASS 名称）。TMA 单元的“在途字节数”并没有被直接暴露出来，因此实际可行的测试是用 `kStages = 4` 重新构建并观察每 k-tile 的时间。

### 3.5.2 四个累加器 stage

TMEM 环把 MMA warp 与 epilogue warp 解耦。MMA warp 用 `tcgen05.commit` 把 stage `st` 提交到 `acc_full[st]`，并且如果 `acc_empty[st + 1]` 空闲，就立即在 stage `st + 1` 中开始下一个 tile；epilogue 则并发地排空 stage `st`（第 9 章和第 10 章；屏障见第 4 章）。稳态要求 tile t 的 epilogue 在 tile t + (stages - 1) 的 mainloop 结束之前结束，即 `epilogue time per tile < (stages - 1) x mainloop time per tile`。

mainloop 每个 tile 大约耗时 27.75 us（实测上界）。每个 CTA 每个 tile 的 epilogue 由 128 个线程把 128 x 128 x 4 = 64 KiB 从 TMEM 搬到 smem（每个线程每个 subtile 16 条 `st.shared.b32`，8 个 subtile），并由 32 次 TMA 存储把 64 KiB 从 smem 搬到全局内存。按标称的 128 B/clk 共享内存带宽，这至少是 2 x 65536 / 128 = 1024 个周期 = 0.52 us 的纯数据搬运；算上八对 `bar.sync 1, 128`、fence 以及 TMA 发射，则很可能是几微秒。第 10 章（第 10.12 节）在刻意悲观的假设下把它界定在大约 11-15 us。两者都是估计；笔记中没有测量过任何 epilogue 时长。

即使 15 us 的 epilogue 也低于一个 27.75 us 的 mainloop 周期，所以**两个** stage 就已经能让 MMA warp 在稳态下不必等待。四个是 TMEM 所施加的上限（第 3.4.2 节），也是 builder 的上限（"4 accumulator stages works well to buffer the accumulators, while also preventing overhead in the epilogue tail on small tile sizes"，`sm100_umma_builder.inl:266`）。多出的两个 stage 换来了对抗抖动的余量：一次缓慢的 epilogue（某个 SM pair 的 TMA 存储在 L2 中发生争用）、一次迟到的 CLC 响应（中位数 1.4 us，第 3.5.3 节），或者某个 A 条带未命中 L2 的 tile。TMEM 在这个 kernel 中没有其他用途，所以这份余量是免费的。

如何确认：用 trace 构建给 MMA warp 中的 `acc_empty` 等待打上时间戳；在生产版本中，对 `acc_empty` 的 `TRYWAIT` 重试次数（`NANOSLEEP.SYNCS` 循环）是一个替代指标（proxy）。

### 3.5.3 两个 CLC stage 与两个 throttle stage

scheduler warp（第 7 章）在 cluster 计算当前 tile 的同时，用 `clusterlaunchcontrol.try_cancel` 向硬件索要下一个 cluster tile。响应落入一个 16 字节的槽位，并使全部四个 CTA 中的 `clc_full[idx]` 完成。有了两个槽位，就可以在 cluster 的 800 个消费线程仍在读取上一个响应时，让一次查询处于在途状态。throttle 流水线（计数 32，2 个 stage）把 scheduler 与 rank 0 的 producer warp 绑在一起：scheduler 只有在加载者已经*开始* tile i 之后才发出查询 i，因此在加载者所处的 tile 之外最多只有一次查询在途（笔记，第 7.4 节）。

在 CUTLASS kernel 的 trace 构建中实测（笔记 D.3；完整分布见第 12 章，第 12.6.3 节）：从查询到消费的延迟中位数为 1408 ns，每个 tile 31.9 us，以及 scheduler 每个 tile 花 28.8-37.7 us 等待 throttle。explicit kernel 的 trace 构建测得中位数 1024 ns，并因其额外的诊断而每个 tile 为 35.46 us（笔记 G.2）。Release 版本的 per-tile 数字为 27.75 us（CUTLASS，笔记 D.4）与 27.79 us（explicit，笔记 G.2）。该延迟占一个 tile 的 5 %，且被完全隐藏。更深的 CLC 环不会带来任何收益；更浅的环（一个槽位）则会让全部 800 个线程对响应 t 的消费与查询 t + 1 的发出串行化，而这正是第二个槽位所避免的。throttle 之所以存在，是因为没有它 scheduler 就会跑到前面，一次取消许多尚未启动的 cluster，囤积本可由其他 cluster 领取的 tile（负载不均衡），并让响应滞留在没有人准备好去消费的槽位里。

### 3.5.4 四个 8 KiB 的 epilogue smem stage，配 `wait_group.read 1`

warp 4-7 把每个 128 x 16 subtile 暂存（stage）在一个 8192 字节的 smem 缓冲区中（stage `e % 4`）；随后 warp 4 用四次 TMA 存储把它写到全局内存，把这些存储作为一个 bulk group 提交，并执行 `explicit.cu:1069`：

```
        tma_store_wait_read<1>();                                                                // producer_acquire (always_wait): at most 1 group pending on its smem read
```

`cp.async.bulk.wait_group.read 1` 在至多还有一个已提交的组（刚刚提交的那个）可能仍在读取 smem 时返回，因此 subtile e - 1 的那个组已经读完了 stage `(e - 1) % 4`。下一个 subtile 写入 stage `(e + 1) % 4`，它的上一个读取者是 subtile e - 3 的那个组，早已完成。对于这种 `read 1` 纪律，两个 stage 就足够了。四个来自设计中加载 C 的那一侧：`StagesC = max(min(EpiTiles, 4), StagesD + 1) = 4`，因为对 32 位的 C 而言 `ReuseSmemC` 为真（`sm100_builder.inl:1107-1115`），而 D 复用 C 的缓冲区（`StorePipeline = PipelineTmaStore<StagesC, StagesD - 1>`，`sm100_epilogue_tma_warpspecialized.hpp:202-205`；`UnacquiredStages = 1` 给出了 `.read 1`，`sm90_pipeline.hpp:697-700`）。对于 beta = 0 的运行，多出的两个 stage 是未被使用的余量：单凭 `.read 1` 这一纪律就能让 epilogue warp 比 TMA 存储对 smem 的读取领先一个 subtile，而这次 wait 是否会阻塞并不取决于 stage 的数量。

---

## 3.6 epilogue tile (128, 16) 与 (32, 16, 1) 存储 box

epilogue 的工作单元是一个 128 x 16 的 fp32 subtile；每个 CTA tile 有 8 个（`kEpiSubtiles = 8`、`kEpiTileN = 16`：`explicit.cu:112-113`，引文见第 5.5 节）。

推导（来自 `sm100_builder.inl:1006-1046`，已验证；`:1047` 是守卫它的 `static_assert(CtaN >= N_min_C && CtaN >= N_min_D, "CTA tile too small")`）：

1. **M = 128** 来自 TMEM 加载的 shape。`tcgen05.ld.sync.aligned.32x32b.x16.b32` 是 "32 data path lanes, 32-bit pattern, repeated 16 times" 这一操作（`copy_sm100.hpp:3575-3576`）：一个 warp 的 32 个 lane 各读取一个 TMEM lane（= 一个累加器行），每个 lane 读 16 个连续的 32 位列。一个 warp 只能寻址它自己子分区的 32 个 TMEM lane，即 lane `[32 (w % 4), +32)`（在未定义 `NDEBUG` 时 CuTe 会对此断言，`copy_traits_sm100.hpp:400-402`；epilogue 检查 "this tile's tmem subpartition is accessible by this warp"，`sm100_epilogue_tma_warpspecialized.hpp:874`）。因此四个 warp x 32 个 lane 覆盖 128 行，而 builder 的 `M = min(CtaM, 32 * WarpM) = min(128, 128) = 128`（`sm100_builder.inl:1006-1007`，`DpFull = 32`）。
2. **N = 16** 来自性能表。在存在残差 C（`DisableSource` 为假）、32 位 C/D、`CtaM > 64` 且 `CtaN <= 128` 的情况下，`N_perf = 16`（`:1024-1025`）。M-major 的 D 的最小 N 是 `8 * WarpN = 8`（`:1039-1044`）。`N_tmp = min(CtaN, max(N_perf, N_min_C, N_min_D)) = min(128, max(16, 8, 8)) = 16`（`:1045`），而 `N = CtaN % N_tmp == 0 ? N_tmp : CtaN` 保留了它，因为 128 % 16 == 0（`:1046`）。于是 subtile 为 128 x 16 x 4 B = 8192 B，即一个 smem stage，而每个线程的 `FragmentSize = 128 * 16 / 128 threads = 16` 个浮点数（`:1255`），这恰好就是加载的 `.x16`，`explicit.cu:1036`：

```
      tmem_ld_32x32b_x16(taddr + 16u * uint32_t(e), r);                                        // columns 16e..16e+15 of row 32 w + l (:881-883)
```

3. **存储 box (32, 16, 1)** 来自 swizzle 限制。D 是 M-major 的，所以 TMA map 的最内维是 M。epilogue smem stage 在一个 `(32, 4):(1, 32)` 的 fp32 原子上使用 `Swizzle<2,5,2>`，编码为 `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`。驱动的规则原文为 "CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128"。星号代表 `_ATOM_*` 各变体，所以这里使用的 `_ATOM_32B` 没有单独的限制；该规则只在 swizzle 不是 `CU_TENSOR_MAP_SWIZZLE_NONE` 时适用。一般约束同样适用："boxDim array ... must be non-zero and less than or equal to 256"，以及对 `CU_TENSOR_MAP_INTERLEAVE_NONE` 而言，"{ boxDim[0] * elementSizeInBytes( tensorDataType ) } must be a multiple of 16 bytes"。128 字节的 fp32 是 32 个元素。因此 box 为 32 (m) x 16 (n) = 2048 B，而一个 128 x 16 subtile 需要四个（`explicit.cu:1055-1057`，引文见第 10.10 节）。

每个 CTA tile 即 8 个 subtile x 4 个 box = 32 次 2048 B 的 TMA 存储 = 64 KiB = 128 x 128 x 4 B，也就是台账行 "D TMA store instructions: 32 per CTA tile"（笔记，第 9 节；笔记第 13 节的更正 14 修正了草稿中的 "eight 8192-byte D TMA stores per logical CTA work tile"）。

为什么不用更宽的 subtile（N = 32，4 个 subtile）？每个线程将携带 32 个而不是 16 个活跃的累加器寄存器，而且 epilogue smem 会翻倍：EpiTiles = 4，StagesD = min(4, 2) = 2，StagesC = max(min(4, 4), 3) = 4（`sm100_builder.inl:1113-1115`），于是 4 x 16384 + 512（FusionStorage 填充）+ 64（LoadPipeline）= 66112 字节，按 512 取整 -> 66560。carveout 公式随后给出 floor((232448 - 192 - 66560) / 24592) = floor(165696 / 24592) = floor(6.74) = 6 个 mainloop stage 而不是 8 个：多出的 32 KiB 是 1.33 个 stage 的大小，但 floor 把这一损失向上取整为整整两个 stage。为什么不用更窄的（N = 8）？同样的字节数下，每个 tile 的屏障/fence/commit 轮次会翻倍。builder 的 `N_perf` 表是 NVIDIA 的经验性答案；explicit kernel 继承了它。

---

## 3.7 warp 特化：8 个 warp、5 种角色、256 个线程

```
 warp 0  MMA        both CTAs allocate TMEM; the pair LEADER's elected lane issues all 512 tcgen05.mma per tile
                    and the commits; the PEER's warp 0 only consumes CLC responses and advances state
 warp 1  scheduler  in every CTA its elected lane prefetches the A and B tensor maps in the prologue; on rank 0 it then
                    runs the CLC try_cancel + throttle loop, on ranks 1-3 it falls through the dispatch and exits
 warp 2  producer   one elected lane issues 2 TMA loads per k-tile (A multicast, B); on rank 0 also the throttle producer
 warp 3  idle       the CUTLASS kernel's beta != 0 C-load producer (no role function here); prefetches the C and D maps, initialises the throttle barriers, exits
 warps 4-7 epilogue 128 threads = 128 TMEM lanes = 128 rows; warp w owns rows [32 w, 32 w + 32); warp 4 issues the TMA stores
```

（`explicit.cu:104`；笔记，第 7.1 节；CUTLASS kernel 的线程核算位于 `sm100_gemm_tma_warpspecialized.hpp:136-147`。）

```
constexpr uint32_t kThreads       = kBlockThreads;   // 8 warps: 0 MMA, 1 Sched (rank 0), 2 MainloopLoad, 3 EpilogueLoad (exits), 4-7 Epilogue
```

为什么每个数目是现在这样（推导）：

- **一个 producer warp。** 一次 TMA 加载由一个线程发射；TMA 单元负责寻址、swizzling 与 multicast。来自一个 lane 的每个 k-tile 两条指令（`explicit.cu:747`、`:754`）就是一个 CTA 的全部数据路径。更多的 producer 线程除了在 `empty[s]` 等待上增加争用之外不会带来任何东西。这个 warp 之所以存在（而不是单个线程），是因为 `elect.sync` 和 warp 级的 `mbarrier` 协议需要一个收敛的 warp。
- **一个 MMA warp，且只有 leader 的 lane 发射。** 因此 peer 的 warp 0 在整个 mainloop 期间空闲；它仍然必须存在，既为了 pair 范围的 `tcgen05.alloc`/`dealloc`，它们要求 pair 中每个 CTA 各出一个 warp（PTX ISA 9.7.18.5，表 55；CUTLASS 的分配器还额外要求两者中逻辑 warp ID 相同，`tmem_allocator_sm100.hpp:132`，笔记 B.6），也为了让两个 CTA 的 stage 索引保持同步的累加器状态簿记。
- **四个 epilogue warp。** 没有别的 warp 数目能读取一个 128 行的 CTA tile；8 个 epilogue warp 只会两两配对到相同的 32 个 lane 上。
- **一个 scheduler warp，仅 rank 0。** 每个 CTA 的 warp 1（包括 rank 0）都在 prologue 中预取 A 和 B 的 tensor map（`explicit.cu:1113-1116`）；在 rank 1-3 上，它随后未通过分发测试 `warp_idx == 1 && is_cta0`（`explicit.cu:1194`），于是贯穿而过并退出，正如分发注释所说（`explicit.cu:1190`："warp 3 and warp 1 of ranks 1-3 fall through and exit"）。第 11 章涵盖 prologue。
- **一个空闲的 warp。** 保留 warp 3 是为了让 warp 编号、屏障初始化顺序（它初始化 throttle 屏障，`explicit.cu:1155-1163`）以及 C 和 D 的 tensor-map 预取与 CUTLASS kernel 保持完全一致（第 11 章）。
- **256 个线程，occupancy 为每个 SM 1 个 CTA。** 寄存器并不稀缺：explicit kernel 使用 26 个（实测，笔记 G.3；CUTLASS kernel 为 68 个）。这一设计后果对你自己的 kernel 很重要：在这台机器上，一个 GEMM CTA 是其 SM 上*唯一*的驻留单元，所以每一项延迟都必须由 CTA 内部的异步性（TMA、UMMA、TMEM 环）来隐藏，而不是靠切换到另一个 CTA。

---

## 3.8 使用 CLC 的持久化调度

### 3.8.1 为什么要动态分派

启动网格为 (64, 64, 1) = 4096 个 CTA = 1024 个 cluster（`explicit.cu:1282-1284`，引文见第 11.8.3 节）。

但只有能装进机器的那些 cluster 会原生启动：实测为 33 个（笔记 D.3，等于 `cudaOccupancyMaxActiveClusters`），在 132 个 SM 上。16 个 SM 保持空闲（SM id 64、65、122-129、142-147），因为一个 2 x 2 cluster 需要同一个 GPC 内的两个完整 TPC，而每个奇数大小的 GPC（NVIDIA 内部）都会剩下一个未配对的 TPC（这一配对是笔记中的推断；算术与两条注意事项见第 2.1.1 节；第 3.3.2 节定义了 TPC 和 GPC）。随后每个驻留的 cluster 用 `clusterlaunchcontrol.try_cancel` *取消*尚未启动的 cluster 并接管它们的工作；1024 次查询中有 991 次成功、33 次失败（每个 cluster 一个终止符）。

尾段算术（推导）：1024 个 tile / 33 个 cluster = 每个 cluster 31.03 个 tile。无论怎样分配，总有某个 cluster 必须做 32 个 tile，所以完工时间是 32 个 tile 时间，而理想值为 31.03：这是 3.0 % 的量化损失，在这一 shape 与驻留数下没有任何 scheduler 能消除它。动态方案消除的是*变动*。下面的计数来自 CUTLASS kernel 的 trace 构建（笔记 D.3）；Release 版本的分布没有被追踪。13 个 cluster 做了 32 个 tile，19 个做了 31 个；带探针的 cluster (0,0) 因其逐指令探针而慢 1.7x（笔记中的推断），做了 19 个。做了 32 个 tile 的那 13 个恰好就是 per-tile 时间最短的 13 个。per-tile 时间分成一个快组（31.1-31.8 us）和一个慢组（32.0-32.7 us）。笔记*推断*（而非测量）这一分组跟随的是在 GPC 内的位置而非 cluster id，因为快慢 cluster 在每个 SM-id 组中都在相同的 SM 偏移处重复出现。每个 cluster 31 或 32 个 tile 的静态分配会给慢 cluster 分到和快的那些一样多的 tile，并在最慢者完成时才结束；动态分派自动吸收了两组之间大约 3 % 的差距（从最快到最慢的不带探针 cluster 则是 5 %）。

### 3.8.2 AlongN 光栅与 L2 局部性

光栅顺序是 `AlongN`，因为启发式规则是 `tiles_n > tiles_m ? AlongM : AlongN`，而两个计数相等（`tile_scheduler_params.h:1810-1816`）。该启发式规则的一般原则是沿较短的 tile 轴光栅化（`tiles_n > tiles_m` -> `AlongM`）：驻留的 cluster 组成的 wave 于是轮流遍历较小操作数的各个面板，后者整体上更有机会保持驻留于 L2，同时一串连续的 cluster 复用较大操作数的每个面板。对于方形问题，两个轴相等，`AlongN` 是打破平局的选择（我的分析）。CLC 硬件按启动线性顺序分发 cluster id（实测：近似递增，991 个中有 933 个处于其排序后的位置，同一个 32 ns 计时器刻度内的并列按 id 升序打破；笔记 D.3："ascending except for a few local swaps"），因此*网格*在主机端被转置，使该顺序先沿 N 扫过：`possibly_transpose_grid` 交换 x 与 y（`sm100_tile_scheduler.hpp:237-250`），而每个 tile id 在设备端被转置回来（`possibly_transpose_work_tile`，`:668-677`）。kernel 的那两个公式就来源于此（`explicit.cu:1175-1176`，在第 11.6 节中引用；以及 `:692-693`，在第 7.4 节中引用）。

网格 x 索引 N（64 个 CTA 列），网格 y 索引 M；二者各自的最低位是其在 cluster 内的位置。硬件在转置后的 (64, 64) 网格上按网格线性顺序给 cluster 编号，因此启动线性的 cluster id 为 `L = (blockIdx.x >> 1) + 32 (blockIdx.y >> 1) = CX + 32 CY`。（笔记 D.3 陈述的是其逆：id L 是 cluster tile `(m, n) = (L / 32, L % 32)`，即 `m = CY`、`n = CX`，并且它把每个 cluster 的连续 id 记作 `(y0/2)*32 + x0/2`。）由于 `tn = 2 CX + y` 且 `tm = 2 CY + x`，连续的 id 推进的是 CX：它们在 A 的一个 256 行条带内沿 N 行走。实测的后果（笔记 D.3）：33 个驻留的 cluster 持有大约 33 个连续的 id，因此它们处理一个完整的条带加上下一个条带的一个 tile，即 "two (834 of 991 hand-outs) or three (157) adjacent 256-row bands of A at any moment"。对 L2 而言这意味着：一个条带的全部 32 个 cluster 共享其 A 条带（256 行 x 8192 k x 2 B = 4 MiB），因此 HBM 大约每个条带供给它一次；每个条带恰好有一个 cluster 使用每个 B 列块（也是 4 MiB），而下一个条带会再次触及它。第 3.9.5 节将其转化为字节数。

Swizzle 已禁用（`max_swizzle_size = 0`、`swizzle_divisor = 0`，笔记第 0 节以及主机端守卫 `explicit.cu:1269`）。CUTLASS 的 swizzle 会把 id 分组为每组几列的块，使得驻留的 cluster 除 A 条带之外也共享 B 块；它是 harness 保留为默认值的一个旋钮，而 explicit kernel 把该默认值硬编码了。第 3.10 节的设计表把它列为显而易见的、有待尝试的替代方案。

---

## 3.9 Speed-of-light 分析

### 3.9.1 Tensor-core 峰值

| 量 | 值 | 状态与来源 |
|---|---|---|
| Blackwell 每 SM 每时钟的稠密 FP16/BF16 张量速率 | 8192 FLOP/clk/SM | SemiAnalysis："for FP16 we know the hardware can do 8192 MMA FLOPs per cycle per SM"；Chips and Cheese："1024 16-bit MAC operations per cycle, per partition" x 4 个分区 x 2 FLOP/MAC = 8192。与下面的数据手册一致：2.25e15 / 148 个 SM / 8192 = 1.86 GHz，一个合理的规格时钟 |
| 同上，Hopper H100 | 4096 FLOP/clk/SM | NVIDIA H100 产品页面为 H100 SXM 列出的 FP16 Tensor Core 为 "1,979 teraFLOPS"，并带脚注 "With sparsity"，即稠密 989.5 TFLOPS；它没有给出时钟。在 132 个 SM 下，该稠密数字*隐含* 989.5e12 / (132 x 4096) = 1.83 GHz，低于通常列出的 1.98 GHz boost 时钟（该 boost 数字未在此处打开的任何来源中得到验证）；4096 FLOP/clk/SM 是上文 Blackwell 数字的一半，是本推导所假设的值，而非数据手册中的数字 |
| B200 数据手册稠密 FP16/BF16 | 每 GPU 2.25 PFLOPS | HGX B200：8 个 GPU 的 "FP16/BF16 Tensor Core: 36 PFLOPS"，脚注 "Specification in Sparse. Dense is 1/2 sparse spec shown"：36 / 8 / 2 = 2.25 |
| 实测的 1965 MHz、148 个 SM 下的峰值 | 148 x 8192 x 1.965e9 = **2.382 PFLOPS** | 推导；高于数据手册的数字，因为数据手册假设了更低的持续时钟（假设） |
| 1965 MHz、132 个忙碌 SM 下的峰值 | 132 x 8192 x 1.965e9 = **2.125 PFLOPS** | 推导 |
| 1965 MHz 下每 SM | 16.1 TFLOPS | 推导 |
| HBM3e 带宽 | 每 GPU 8 TB/s | DGX B200：8 个 GPU 的 "64 TB/s HBM3e bandwidth"，即 64 / 8；Lenovo HGX B200 产品指南列出每 GPU 7.7 TB/s（第 2.2 节） |
| L2 | 总计 126 MB，两个分区；本地 21 TB/s，跨分区 16.8 TB/s（Chips and Cheese）；读带宽全 GPU 23.9 TB/s，每分区约 12.0 TB/s；写带宽全 GPU 10.8 TB/s，每分区约 5.4 TB/s（NVIDIA 内部） | 容量以及本地与跨分区的数字来自 Chips and Cheese 的测量；读带宽与写带宽来自 NVIDIA 内部信息 |
| 每 SM 的共享内存带宽 | 128 B/clk | SemiAnalysis："the SMEM bandwidth is 128 B/cycle (per SM)"，这是主要来源。同一数字被 NVIDIA Developer Forums 帖子 "About Shared-Memory Bandwidth Usage Of Tensor-Core In Blackwell Architecture" 中的论坛参与者重复（两人都不是 NVIDIA 员工）：用户 Curefab，"Each SM can do 128B/cycle of shared memory, which is shared between all 4 SM partitions"，以及用户 Nanodeoclus，"The SMEM pipeline shared between the four SM partitions can do either a load or a store per cycle, but not both"——后者在同一条发言中补充说，这 "doesn't necessarily mean that TMA writes to SMEM, or TC reads from SMEM, have the same limitation"，而 Curefab 回忆道 "the TC smem acceses are routed through the async engine, which is per lane"。因此，tensor-core 操作数读取与 TMA 写入是否计入同一个 128 B/clk，**在该帖子本身中就悬而未决**，并且此处打开的任何来源都**未予证实** |

### 3.9.2 kernel 所处的位置

实测（笔记 G.2）：中位数 `Avg runtime` 0.889354 ms，2 x 8192^3 / 0.889354 ms = 1.236 PFLOPS（1.2363e15 FLOP/s；harness 在五次 explicit 运行中打印出 1.23498e6-1.23687e6 GFLOPS，而笔记 D.4 的 1.237e15 是 CUTLASS baseline）。推导出的比例：

```
1.236 / 2.382 = 51.9 % of the 148-SM peak at 1965 MHz
1.236 / 2.125 = 58.2 % of the 132-SM peak (the SMs that actually run CTAs)
```

相对于 2.25 PFLOPS 的数据手册数字，kernel 达到 55 %；按每个*活跃* SM（148 个中的 132 个）计，它在 1965 MHz 下维持每时钟 4770 FLOP。

每 k-tile 预算（笔记 D.4；这是一个实测上界，因为计时区间还包含主机端工作）：关键路径上 32 个 tile -> 每 tile 27.75 us -> 每 k-tile 216.8 ns = 426 个周期 -> 每条 256 x 128 x 16 MMA 106.5 个周期。27.75 us 是笔记 D.4 给出的 CUTLASS baseline 的数字，即 0.888122 ms (0.888 ms) / 32。explicit kernel 自身的中位数给出 0.889354 ms / 32 = 27.79 us = 217.1 ns = 每 k-tile 427 个周期（对一个 31-tile 的 cluster 为 28.69 us），这 0.14 % 的差异不改变任何结论，因此 426 个周期是本章通篇使用的工作数值。1965 MHz 本身是推断的，并非在循环内部实测：时钟是在 9 ms 的计时循环前后每 100 ms 采样一次（笔记 D.4、G.2）。在峰值时，pair 的两个 SM 提供 2 x 8192 FLOP/clk，因此一条这样的 MMA（2^20 FLOP）需要 64 个周期，一个 k-tile 需要 256 个周期 = 130 ns。因此在 mainloop 期间，kernel 以 64 / 106.5 = 60 % 的峰值速率运行 tensor core（推导；58 % 的整体运行数字还加上了 3 % 的尾段量化与爬升）。

所以大约 40 % 的 tensor-core 周期没有被使用。下面的候选项连同各自的算术一并列出；第一个是首要假设。

### 3.9.3 候选 1：共享内存带宽（假设，首要）

每 SM 每 k-tile，共享内存看到的是（由第 3.2-3.3 节推导）：

```
TMA writes   : A stage 16384 B (own half by own load, other half by multicast) + B half 8192 B = 24576 B
UMMA reads   : A 128 rows x 16 k x 2 B x 4 k-blocks = 16384 B + B 64 cols x 16 k x 2 B x 4 = 8192 B = 24576 B  (a lower bound: at least the bytes delivered to this SM; cta_group::2 does not imply read-once B (NVIDIA internal); Section 2.5.3)
total        : 49152 B per k-tile per SM
at measured 426 cycles : 49152 / 426 = 115 B/clk  = 90 % of 128 B/clk
at peak     256 cycles : 49152 / 256 = 192 B/clk  > 128 B/clk
```

假设 TMA 写入与 UMMA 操作数读取共用一个 128 B/clk 的共享内存端口（论坛用户 Nanodeoclus 针对 SMEM 流水线所说的 "either a load or a store per cycle, but not both"，已连同其注意事项在第 3.9.1 节中引用）。UMMA 那一行是读取量的下界（`cta_group::2` 并不意味着 B 只读一次，NVIDIA 内部，第 2.5.3 节）；高于它的读取量会降低上限。TMA 那一行（落入此 SM 的 smem 的字节，即复制流量）与 UMMA 那一行（tensor core 所消费的量）是两个不同的量：leader 发射 MMA，而两个 CTA 都接收 smem 数据（NVIDIA 内部）。那么 128 x 128 的 CTA tile *不可能*超过 tensor 峰值的 128 / 192 = 67 %，即 132 个 SM 上的 1.417 PFLOPS。实测的 1.236 PFLOPS 是该上限的 87 %。其余的 13 % 被归因于（并非实测）3 % 的尾段量化、首个 tile 的爬升（在 trace 构建中，每个 cluster 的第一个区间为 37.9 us 对 31.9 us，笔记 D.3）、最后一个 tile 未被重叠的 epilogue，以及计时区间内的主机端工作。epilogue 每 tile 再增加 2 % 的 smem 流量（64 KiB 的 `st.shared` + 64 KiB 的 TMA 存储读取 = 128 KiB，对比每 tile 6 MiB 的 mainloop 流量）。

同一范围内的一则轶事，而非对 60 % 的测量：该论坛帖子的原发帖人（用户 tugrul_192bit）报告 "a pipeline with 7 stages of 128x128x64 tile that has only 1100 tflops performance of bf16. But cublas does 1670 tflops"，后来又说 "128 x128 tile size doesn't reach 60% of peak tensor core performance"。1100 TFLOPS 是 2.25 PFLOPS 稠密数据手册数字的 49 %，是该 cuBLAS 数字的 66 %。cuBLAS 的数字也是本章中最有用的*可达成*目标：1670 TFLOPS 是 2.25 PFLOPS 数据手册峰值的 74 %，是 1965 MHz 下 148-SM 峰值（2.382 PFLOPS）的 70 %。这限定了 40 % 的差距中有多少在实践中可以挽回（用户在未指明的 B200 与时钟上的测量；视为指示性的）。

反假设就在同一帖子中：如果 tensor-core 操作数读取是 "routed through the async engine" 而不是经过 128 B/clk 的 LSU 流水线，那么上面 192 B/clk 的总和从不争抢同一个端口，候选 1 就是错的。对该端口总流量的一次测量就能区分二者（没有任何 Nsight Compute 计数器直接暴露 tensor core 的操作数读取字节，NVIDIA 内部；见下文）。如果操作数读取*确实*使用该端口，那么 mainloop 的共享内存吞吐量至少为 49152 B / 426 个周期 = 115 B/clk = 128 的 90 %，这是一个下界，因为上面的 UMMA 读取行假设 B 只读一次，而 `cta_group::2` 并不意味着这一点（NVIDIA 内部）；如果它们不使用，那么只有 TMA 写入计入：24576 B / 426 个周期 = 57.7 B/clk = 128 的 45 %。

如果有计数器暴露它，用 Nsight Compute 确认的做法是：测量 mainloop 期间共享内存端口的总流量——它并不是 Memory Workload Analysis 表中以峰值百分比表示的共享内存吞吐量（Profiling Guide 的 Shared Memory 表按访问类型显示 Wavefronts 与 % Peak；共享内存 wavefront 计数器是 `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum`（NVIDIA 内部）；bank 冲突计数器 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` 统计共享内存加载/存储的 bank 冲突，在比较 kernel 时应当按那些 wavefront 计数归一化，而不是除以源码级的加载指令数（NVIDIA 内部；该指南的 Metrics Examples 列出的是 `l1tex__data_bank_conflicts_pipe_lsu.sum` 族，而非 `_mem_shared_op_*` 这些名字）；这些 `pipe_lsu` 计数器覆盖的是 LSU/L1TEX 路径，而非异步 proxy 流量（NVIDIA 内部）：它们能看到 epilogue 的 `st.shared`，但既看不到 mainloop 的 TMA 写入，也看不到 tensor core 的操作数读取，这两者都是异步 proxy 操作（第 2.7.6 节）；TMA 一侧有其专用指标，它们只覆盖写入一侧（NVIDIA 内部；请求所安装的 Nsight Compute 提供的 `.sum` 形式）：`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` 统计送入共享内存的字节数，对本次运行的预期值为 CTA 数 x k-tile 数 x (A stage + B stage) = 4096 x 128 x 24576 B = 12 GiB（每 k-tile 每 SM 落地的 24576 B 就是上面推导出的 45 %），并且不因 multicast 而改变，因为每个接收 CTA 仍然得到一个完整的 tile（一个内部实测的 2-SM GEMM——并非本次运行——在其 multicast 与非 multicast 两个分支中都恰好显示 1,073,741,824 B）；`lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_tma*` 族是一个 L2 tag-stage 请求/扇区指标（按 Profiling Guide，`lts__t` 是 L2 切片的 tag stage），其预期值是 L2 一侧的流量，即第 3.9.4 节中每 SM 每 k-tile 的 16384 个唯一字节，并且预期会在 multicast 下随唯一请求流量一起下降（NVIDIA 内部）；没有任何已知的 Nsight Compute 计数器直接暴露 tensor core 的 A/B 共享内存操作数读取字节，而 UMMA 是通过其描述符从共享内存读取它们、绕过了 `ldmatrix`（NVIDIA 内部），因此该流量必须从 tile 几何与指令计数推导出来，正如上面的 UMMA 行所做的那样）——连同 tensor-pipe 利用率一起（`sm__pipe_tensor_cycles_active`，即 UMMA 数学活动的度量；`sm__pipe_tc_cycles_active` 是 Profiling Guide 中更宽泛的 TC pipe，即 MMA 数学加上 `UTCCP`、`UTCBAR` 以及 TMEM 分配原子操作，二者是不同的；第 12.7.5 节列出了两者；`sm__pipe_tensor_op_hmma_cycles_active` 是 `sm__pipe_tensor_*` 族中的一个 tensor-pipe 活跃指标，其本身并不能证明是 `UTCHMMA` 使其递增；全部为 NVIDIA 内部）。端口总量接近 90 % 且 `sm__pipe_tensor_cycles_active` 接近 60 % 将证实该假设，而接近 45 %（仅 TMA 写入）则将其推翻；但由于没有针对操作数读取的计数器（见上文），该总量无法直接从 Nsight Compute 读出，而第 3.9.7 节的 256 x 256 实验仍然是决定性的检验。

### 3.9.4 候选 2：L2 -> SM 带宽（假设，峰值时接近极限）

每个 SM 每个 k-tile，TMA 单元请求 8192 B 的 A（自己的那一半；另一半由同胞的请求经 multicast 送达）和 8192 B 的 B：共 16384 个唯一字节，也就是第 3.9.3 节的 `lts__t` TMA 指标预期会报告的 L2 侧流量（NVIDIA 内部）。GPU 范围（推导）：

```
measured rate : 132 SMs x 16384 B / 216.8 ns =  9.98 TB/s
at peak       : 132 SMs x 16384 B / 130.3 ns = 16.6  TB/s
without A multicast          (24576 B/SM) : 15.0 TB/s at the measured rate
without multicast and 2SM    (32768 B/SM) : 20.0 TB/s at the measured rate
```

与之对照的是实测的 L2 带宽：分区内 21 TB/s，跨分区 16.8 TB/s（Chips and Cheese），以及 L2 读带宽 GPU 范围 23.9 TB/s、每分区约 12.0 TB/s（NVIDIA 内部）。按现状构建，这个 kernel 在其实测速度下使用了 L2 带宽的 42 到 48 %（9.98 TB/s 对 23.9 或 21 TB/s），而在 tensor 峰值时将需要其 69 到 99 %（16.6 TB/s 对 23.9 TB/s，或对 16.8 TB/s 的跨分区数字）。第 3.3 节的两种共享机制使这个候选不至于成为第一堵墙。用 `lts__t_bytes.sum.per_second` 及其占峰值的百分比来确认：`lts` 是一个 L2 缓存切片，`lts__t` 是它的 tag stage（Profiling Guide，单元描述），而 `lts__t_bytes.sum` 是宽泛的 L2 tag-stage 字节流量，既不是 DRAM 字节，也不一定是交付给消费它们的指令的有效载荷字节，因为命中、来自多个客户端的请求以及请求侧效应都可能计入其中（NVIDIA 内部），所以要对照上表把它读作 L2 流量，而不是有效载荷字节计数。

### 3.9.5 候选 3：光栅所隐含的 HBM 流量（假设，并非限制因素）

唯一字节数：A 128 MiB、B 128 MiB、D 256 MiB = 512 MiB。在第 3.8.2 节的 AlongN 扫描下，A 从 HBM 大约读取一次（每个条带由处理它的 32 个 cluster 共享），D 写入一次，而 B 每个条带重读一次，除非它在条带之间仍留存于 L2 中。上下界（推导）：

```
best  (B read once)           : 128 + 128 + 256 MiB = 0.54 GB -> 0.54 GB / 0.889 ms = 0.60 TB/s =  8 % of 8 TB/s
worst (B re-streamed 32 times): 128 + 4096 + 256 MiB = 4.70 GB -> 5.28 TB/s                      = 66 % of 8 TB/s
```

B 是否留存：一个 B 列块要到一个条带之后才被复用，也就是在 B 的全部 128 MiB 已经流过一个被分成两个分区的 126 MB L2 之后（若均分则每个分区 63 MB；Chips and Cheese 只给出了总量），所以在类 LRU 策略下其大部分会被逐出（假设）。真实值介于上下界之间，可用 `dram__bytes_read.sum` 和 `dram__bytes_write.sum` 测得。即便是最坏情况也还留有三分之一的 HBM 带宽空闲，所以在这个 shape 下 HBM 不是限制因素。对于 tensor 峰值下的 128 x 128 CTA tile，它会成为限制因素（5.28 TB/s x 426 / 256 = 8.8 TB/s）；tile swizzle（第 3.8.2 节）是标准的补救办法。

### 3.9.6 候选 4：屏障与发射开销（推导，并非限制因素）

MMA warp 的 k-tile 循环是 47 条指令，其中 37 条在 uniform 数据通路上（实测，笔记 G.3），这当中 4 条是 MMA、1 条是 commit。按大约每周期一条指令计，它只需要 426 个周期中远不到 100 个，而且它是在 tensor core 执行上一个 k-tile 的工作的同时发射的。producer 的 37 条指令循环最多领先 8 个 stage 运行。`full[s]` 等待通常是一次成功的 peek：数据早就到了（第 3.5.1 节）。关键路径上唯一的每个 k-tile 的同步是 `tcgen05.mma` -> `tcgen05.commit` -> `empty[s]` -> producer -> TMA -> `full[s]` 这个环路的完成，而有了 7 个 stage 的余量，它不在路径上。

commit 操作是一次延后的 arrive：`tcgen05.commit` 让 mbarrier 跟踪发射线程此前所有异步 tcgen05 操作的完成（PTX ISA 9.4，第 9.7.18.12.1 节，在第 2 章，第 2.4.3 节和第 4 章，第 4.2.2 节中引用）。因此它必须由发射了那四条 MMA 的同一个被选中的线程发射。它不能用一条普通的 `mbarrier.arrive` 代替，后者会在 tensor core 读完该 stage 之前就触发（笔记，第 7.6 节）。于是 empty 侧的释放完全不需要 MMA warp 等待：硬件在四条被跟踪的 MMA 完成时发出 `empty[s]` 信号，而这意味着它们对该 stage 的读取已经完成（第 2.4.3 节）。

*确实*在路径上的开销每个 tile 发生一次：`acc_empty` 等待（被隐藏，第 3.5.2 节）、CLC 消费（1.4 us，隐藏在 throttle 之后），以及第一个 tile 的冷启动。这些都解释不了 40 %。用 Nsight Compute 中对 warp 0 的 warp 状态采样来确认：如果它在 mainloop 期间的主导状态是 `selected`（该指南的 "Warp was selected by the micro scheduler and issued an instruction"）或是在等待 tensor pipe，而不是在一次失败的 `SYNCS` try-wait 之后的 `NANOSLEEP` 重试循环中处于 `sleeping`，也不是 try-wait 路径本身上的 `long scoreboard`（Nsight Compute 没有 `SYNCS` 状态；`long scoreboard` 这一归因及其注意事项是 NVIDIA 内部信息，第 3.5.1 节），那么屏障就不是问题所在。

### 3.9.7 256 x 256 pair tile 会改变什么（推导）

256 x 256 pair tile（每个 CTA 128 x 256）是本分析预测在 8192^3 上最快的配置。这里没有对它做过剖析，而且为本章打开的任何来源都没有对 CUTLASS 配置进行排名，所以请把这一预测当作一个假设。

收益：每条 256 x 256 x 16 MMA，每个 SM 供给 A 128 x 16（4 KiB）和 B 128 x 16（4 KiB）= 8 KiB，而 pair 执行 2^21 FLOP = 峰值时 128 个周期。每个 k-tile（4 条 MMA，峰值时 512 个周期）SM 通过 TMA 写入 32 KiB，而 tensor core 至少读取 32 KiB，即第 3.9.3 节的只读一次计数，现在它是一个下界，因为 `cta_group::2` 并不意味着 B 只读一次（NVIDIA 内部；实际计数未验证）：64 KiB / 512 个周期 = 在该界上**峰值时 128 B/clk**，恰好是标称的共享内存带宽，相对于这个 kernel 的 tile 的 192 B/clk。每 smem 字节的 FLOP，在同一界上，从 42.7 升到 64；每个 SM 每 L2 字节的 FLOP 从 128 升到 171（为 4.19 MFLOP 请求 24 KiB）。

代价：每个 stage 256 个 TMEM 列，于是累加器 stage 是 2 个而非 4 个（第 3.4.2 节）；一个 32 KiB + 16 B 的 stage 和一个更大的 epilogue 缓冲区，于是按 carveout 公式 mainloop stage 是 5 个而非 8 个（手工推导，未实例化）；以及一个 256 x 256 的 CTA-pair tile 对一般问题的量化更差。

这个 5 是这样得出的。当 CtaN = 256 时，epilogue builder 的 `N_perf` 变为 32（`sm100_builder.inl:1025`：`(CtaM > 64 && CtaN <= 128) ? 16 : 32`），所以 epilogue tile 是 (128, 32) = 每个 stage 16 KiB。EpiTiles = 128 x 256 / (128 x 32) = 8，StagesD = min(8, 2) = 2，StagesC = max(min(8, 4), 3) = 4（`:1113-1115`），所以 epilogue 存储与第 3.6 节为 (128, 32) subtile 推导出的 66560 B 相同（4 x 16384 + 512 + 64 = 66112，按 512 取整）。累加器 stage 数为 65536 / (128 x 256) = 2，于是累加器流水线的屏障存储从 64 减半到 32 B（`PipelineAsyncSharedStorage`：两个 `ClusterBarrier[Stages]` 数组，`sm90_pipeline.hpp:1005-1011`），而 kernel carveout 为 192 - 32 = 160。每个 CTA 的 stage 字节数为 A 128 x 64 x 2 = 16384 加 B 128 x 64 x 2 = 16384（2SM 原子的 `BLayout` 给每个 CTA N/2 列，`mma_traits_sm100.hpp:1181`，而 builder 按那个 `BlockTileB_N` 来确定 B smem 的大小，`sm100_umma_builder.inl:241`、`:295`）再加 16 = 32784。Stages = floor((232448 - 160 - 66560) / 32784) = floor(165728 / 32784) = floor(5.05) = 5。对于 8192^3，这些代价没有一项会真正造成影响，这就是本分析预测更大的 tile 在那里会胜出的原因；harness 是一个*教学*示例（其文件头部：它 "demonstrates minimal set of changes needed to transition from a Hopper CUTLASS 3.x GEMM kernel"），而且在任何地方都没有声称 256 x 128 tile 是最快的那个。

### 3.9.8 16 个空闲的 SM

- **空闲的 SM（实测，笔记 D.3）。** 148 个 SM 中有 132 个承载 CTA：相对于一台完美的机器有 11 % 的损失，与 kernel 无关（在这块 B200 的 skyline 上，第 2.1.1 节，它来自奇数大小的 GPC 和 (2,2,1) 的 cluster shape；NVIDIA 内部信息）。一个以不同方式配对 TPC 的 cluster shape，或者一个能够利用剩余 TPC 的 1SM kernel（cluster (1,1,1) 或 (1,2,1)），会以第 3.3 节的复用为代价换回其中一部分。

### 3.9.9 SOL 图景小结

```
                       tensor peak, 148 SMs, 1965 MHz                  2.382 PFLOPS   (100 %)
  16 idle SMs      ->  tensor peak, 132 SMs                             2.125 PFLOPS   ( 89 %)
  smem bandwidth   ->  x 128/192 (read-once bound) if reads share port  1.417 PFLOPS   ( 59 %)   [hypothesis]
  tail quantisation->  x 31.03 / 32                                     1.374 PFLOPS   ( 58 %)
  measured                                                              1.236 PFLOPS   ( 52 %)
```

1.374 与 1.236 之间的一切（10 %）都是归因而非实测，归于爬升、最后一个 tile 的 epilogue、epilogue 的 smem 流量和主机端开销。这就是必须由 Nsight Compute 而非算术接手的地方。决定性的实验是 256 x 256 pair tile：如果共享内存假设是对的，它会把上限从 148-SM 峰值的 59 % 提升到 89 %（2.382 PFLOPS 中的 1.417 -> 2.125），也就是从 2.125 PFLOPS 这一 132-SM 峰值的 67 %（128 / 192）提升到 100 %，二者都基于第 3.9.3 节和第 3.9.7 节的只读一次界（NVIDIA 内部；实际读流量未验证）。

---

## 3.10 设计决策表

| 决策 | 取值 | 为什么 | 备选方案 | 备选方案的代价 |
|---|---|---|---|---|
| 操作数 major | A K-major，B K-major | 规范的 UMMA SW128 layout；32 字节的 k-block 步进；TMA box 64 x 64 | MN-major 的 A 或 B | 不同的描述符遍历（LBO 生效，k 步进以 SBO 的倍数计），不同的 TMA map 与 smem 原子（第 3.1.2 节） |
| MMA shape | 256 x 128 x 16，`cta_group::2` | pair 共享 B（按 64/64 拆分，每个 CTA 加载一半，指令读取两半）并沿 M 堆叠 A（每个 CTA 各自的 128 行）；每个 64 宽的 k-tile 4 条 MMA | 1SM 128 x 128 x 16 | 每个 k-tile 每个 SM 32 KiB 而非 24 KiB 的操作数 smem；在第 3.3.1 节的 (1,2,1) 设计中 L2 字节为 1.5x，无 multicast 时为 2x |
| CTA tile | 128 x 128 x 64 | TMEM 容纳 4 个 stage；放得下 8 个 smem stage | 128 x 256（pair 256 x 256） | 2 个 TMEM stage、5 个 smem stage（在第 3.9.7 节中推导），在小问题上量化更差；但峰值时 smem 为 128 B/clk 而非 192，二者都在只读一次界上（NVIDIA 内部） |
| Cluster | (2,2,1) | 沿 M 的 2SM pair 加上沿 N 的 A multicast：每 L2 字节 128 FLOP | (1,1,1) 64；(2,1,1) 和 (1,2,1) 85；(4,1,1) 比 (2,2,1) 更差 | 每 FLOP 更多的 L2 流量，而且对于 1SM 的各 shape，写入每个 SM 的 smem 的 TMA 字节也更多（第 3.3 节） |
| k-tile | 64 | 填满 128 字节的 SW128 行；每 48 KiB 一轮屏障 | 16 或 32 | 相同字节数下 4x 或 2x 的屏障/TMA 指令速率；更小的 TMA box |
| Mainloop stage 数 | 8 | carveout 剩下的全部：1.5-1.7 us 的预取 | 4 | 0.65-0.87 us 的延迟容忍（领先 3-4 个 stage x 24576 B / 113 GB/s）；远分区与 HBM 未命中暴露出来（第 3.5.1 节） |
| 累加器 stage 数 | 4 | 每个 stage 128 列时 TMEM 的最大值；稳态下 2 个就够 | 2 | 对 epilogue 或 CLC 抖动的余量更少 |
| CLC stage 数 / throttle | 2 / 2 | 一个查询处于未决状态，在下一个在途时消费其响应；1.4 us 延迟对 27.75 us 的 tile | 1 个槽位；无 throttle | 串行化的消费/发射；囤积 tile 与负载不均衡 |
| Epilogue tile | (128, 16) | 32x32b.x16 加载 shape，4 个 warp x 32 个 lane，每线程 16 个寄存器 | (128, 32) | 32 个活跃寄存器，4 x 16 KiB 的 stage，mainloop 环按 carveout 的下取整从 8 -> 6 个 stage（第 3.6 节） |
| 存储 box | (32, 16, 1) | 128B swizzle 对 M-major fp32 D 的 128 字节内维限制 | 未经 swizzle 的 stage，配 (128, 16, 1) 的 box | 合法：`SWIZZLE_NONE` 没有 128 字节的内 box 限制，boxDim[0] = 128 <= 256，而且 128 x 4 B 是 16 B 的倍数。每线程的 16 条 `st.shared.b32` 两种方式下都没有 bank 冲突：对于固定的值索引，32 个 lane 带着 `explicit.cu:1013` 的 XOR 写入同一个 128 字节跨度内的 32 个不同的字，不带它则写入 128 个连续字节。真正的代价是离开 CUTLASS builder 的 `Layout_MN_SW128_32B_Atom` 选择（笔记，第 4 节）并把台账改为每个 CTA tile 8 次 8 KiB 的存储。对 TMA 存储的 smem 读取侧的任何影响都未验证 |
| Epilogue smem stage 数 | 4 x 8 KiB，`wait_group.read 1` | 从带 `ReuseSmemC` 的 `StagesC` 继承而来；beta = 0 时 2 个就够 | 2 个 stage | 没有余量；一次缓慢的 TMA 存储读取会使下一个 subtile 停顿 |
| Warp | 1 MMA + 1 sched + 1 producer + 1 空闲 + 4 epilogue = 8 | 单发射者单元（TMA、UMMA、CLC）；128 个 TMEM lane | 2 个 producer warp；8 个 epilogue warp | 无事可发射；TMEM 的 lane 规则无论如何都给每个 warp 32 个 lane |
| Occupancy | 1 CTA/SM，`__launch_bounds__(256, 1)` | 230400 B 的 smem | 2 CTA/SM，各 3 个 stage（各 4 个将需要 2 x (33792 + 4 x 24576) = 264192 B > 每个 SM 228 KB，算上这个 kernel 的屏障与 epilogue 区域；各 3 个，2 x 107520 B，放得下） | 每个 CTA 3/8 的预取深度；两个 MMA 发射者争用一个 tensor core |
| 调度 | 动态 CLC，持久化 | 吸收不同 cluster 放置位置之间每个 tile 3 % 的速度差异（SM 在 GPC 内的位置，推断；trace 构建）；尾段 32 对 31.03 | 静态持久化（tile = f(cluster id, iteration)） | 慢的 cluster 最后完成；同样的量化 |
| 光栅 | AlongN，swizzle 0 | tile 数相等时的启发式规则；驻留的 cluster 扫过一个 A 条带 | swizzle 4-8 | 从 HBM 对 B 的重读更少（第 3.9.5 节）；harness 默认值为 0 |
| alpha、beta | 运行时参数 | 保留 FMUL + FFMA 这一对以及按位精确的 D（第 5 章） | 编译期 1.0f / 0.0f | 128 条 FMUL + 128 条 FFMA 被折叠掉，而 alpha、beta 被冻结；D 保持按位一致，只有单独的 `alpha * acc` 才会产生 -0.0 的符号差异（第 10.9 节） |

---

## 3.11 “设计先于代码”这一课

在 `explicit.cu` 的第一行存在之前，笔记就已经产出了五件产物。它们就是设计；代码是它们的誊写。对于一个新 kernel，你应当能够根据算子的 shape 与机器的数字填写其中每一件，然后才去写 PTX。

**1. tile 表**（第 3.2 节）：问题 -> cluster tile -> CTA tile -> k-tile -> 指令，附有每一层级上的 FLOP 数与字节数，以及坐标公式（`tm, tn` 来自启动 id 和来自 CLC id；A 的行 `128 tm + 64 y`；B 的列 `128 tn + 64 x`；D box `(128 tm + 32 i, 128 tn + 16 e)`）。

**2. 共享内存映射**（第 3.4.1 节）：每一个偏移、大小、对齐与所有者，以及证明总量的算术。在代码中，它变成了 `OFF_*` 以及 `explicit.cu:139-166` 的那些 `static_assert`。

**3. 屏障表**（笔记，第 7.2 节；第 4 章）：对 33 个已初始化屏障（layout 中有 43 个槽位，第 3.4.1 节）中的每一个，其计数、其事务字节数、谁到达（本地、经 `mapa` 的远程、经 peer 位掩码、经 `tcgen05.commit` 并带哪个 multicast 掩码）、谁以哪个 parity 等待，以及在退出时排空它的尾段。该 kernel 的协议常量位于 `explicit.cu:180-192`（引文见第 5.9 节）。

每一个计数都是设计中的一句话（第 4 章，第 4.4 节逐一解读）。注意，姊妹类 `PipelineTmaTransformAsync` 在 `sm100_pipeline.hpp:357-359` 处把 `kMainEmptyCount` 表达式（`multicast_consumer_arrival_count`，第 4.4.2 节）乘以 `ceil_div(num_consumers, 128)`，因此一个 consumer 侧是一个 warpgroup 而非单个 MMA warp 的设计，必须重新推导该计数。

**4. 角色表**（第 3.7 节）：warp -> 角色 -> 哪些 CTA -> 它生产与消费哪些屏障 -> 它在每个 tile 上以及退出时做什么。

**5. per-tile 操作台账**（笔记，第 9 节），即此后每一项优化都据以衡量的预算：

| 项目 | 每单位 | 整次运行的总计 |
|---|---|---|
| 输出 CTA tile (128 x 128) | | 4096 |
| Cluster 工作 tile (256 x 256) | | 1024 |
| TMA A 加载 (64 x 64 box) | 每 CTA 每 k-tile 1 次 | 524288 |
| TMA B 加载 (64 x 64 box) | 每 CTA 每 k-tile 1 次 | 524288 |
| `tcgen05.mma` | 每 pair 每 tile 512 条（1 条清零 + 511 条累加） | 1048576 |
| 每条 MMA 的 FLOP | 2 x 256 x 128 x 16 = 1048576 | 2 x 8192^3 |
| `tcgen05.ld` warp 指令 | 每 CTA tile 32 条（8 个 subtile x 4 个 warp） | 131072 |
| D TMA 存储 (2048 B box) | 每 CTA tile 32 次（每 subtile 4 次，warp 4） | 131072 (256 MiB) |

手握这五件产物，第 3.9 节的分析就可以在 kernel 存在*之前*完成：台账给出每 k-tile 的字节数，机器表给出各项带宽，而这些比值告诉你会先撞上哪一堵墙。对这个 kernel，它们说的是：共享内存为 192 B/clk（在只读一次界上）对比 128，L2 为 16.6 TB/s 对比约 17-21——二者都在 tensor core 饱和之前。这就是在手写 `tcgen05.mma` 之前应当进行的设计对话；第 13 章把它变成了一份配方。

---

# 第 4 章 协议：屏障、流水线、parity 与 per-tile 时间线

这里的一切都关乎一条指令*何时*可以被发射。

---

## 4.1 为什么需要协议：五个异步引擎与一条规则

kernel 的 256 个线程自身几乎不做任何数据搬运。它们向五个异步引擎*发射*工作并*等待*其完成：

| 引擎 | 写入 | 读取 | 由谁发射 | 完成信号由谁发出 |
|---|---|---|---|---|
| TMA 加载单元（`cp.async.bulk.tensor ... cta_group::2`，W21/W22） | smem A/B stage s | 全局内存中的 A、B | warp 2，一个被选中的 lane | 一个 mbarrier 上的 `complete_tx` 字节（pair leader 的 `full[s]`） |
| Tensor core（`tcgen05.mma.cta_group::2`，W29） | TMEM 累加器 stage st | 两个 CTA 的 smem A/B stage s | pair leader 的 warp 0，一个被选中的 lane | `tcgen05.commit`（W30）在各 mbarrier 上 arrive |
| TMEM 加载路径（`tcgen05.ld`，W31） | 寄存器 | TMEM | warp 4-7，每个 lane | `tcgen05.wait::ld`（W32） |
| TMA 存储单元（`cp.async.bulk.tensor ... bulk_group`，W23） | 全局内存中的 D | smem C/D stage | warp 4，全部 lane | 一个 bulk group 上的 `cp.async.bulk.wait_group.read`（W25） |
| CLC 单元（`clusterlaunchcontrol.try_cancel`，W33） | 每个 CTA 中的 16 字节响应槽位 | 网格中尚未启动的 cluster | rank 0 warp 1，一个被选中的 lane | 在每个 CTA 的 `clc_full[i]` 上 `complete_tx` 16 字节 |

每个引擎都在一个*缓冲区*上工作，而另一个引擎或线程随后会复用这个缓冲区。使这一点安全的规则对每个缓冲区都相同：它是一个 **由 S 个 stage 组成的环**，每个 stage 由一个 **full** 屏障（producer 宣布“数据已就位”）和一个 **empty** 屏障（consumer 宣布“我已用完它”）守护。producer 在写入 stage s 之前等待 `empty[s]`；consumer 在读取它之前等待 `full[s]`。kernel 的五个环（`k*Stages` 常量，explicit.cu:107-111，引文见第 5.5 节）：

| 环 | S | Stage 内容 | Producer | Consumer | Full 屏障 | Empty 屏障 |
|---|---|---|---|---|---|---|
| Mainloop A/B smem | 8 | 每个 CTA 128 x 64 的 A（16384 B）+ 64 x 64 的 B（8192 B） | warp 2（TMA） | pair leader 的 warp 0（UMMA） | 位于 `OFF_MAIN_FULL` 的 `full[8]` | 位于 `OFF_MAIN_EMPTY` 的 `empty[8]` |
| 累加器（TMEM） | 4 | 128 个 lane x 128 列 fp32 | pair leader 的 warp 0（UMMA） | 两个 CTA 的 warp 4-7 | `acc_full[4]` | `acc_empty[4]` |
| CLC 响应 | 2 | 每个 CTA 中一个 16 字节的响应 | rank 0 warp 1 | 所有 CTA 的 warp 0、2、4-7 + rank 0 warp 1 | `clc_full[2]` | `clc_empty[2]`（CTA 0 的） |
| CLC throttle | 2 | 无（一个纯粹的 token） | rank 0 warp 2 | rank 0 warp 1 | `thr_full[2]` | `thr_empty[2]` |
| Epilogue C/D smem | 4 | 128 x 16 的 fp32 subtile（8192 B） | warp 4-7（`st.shared`） | TMA 存储单元 | （无：`bar.sync 1`） | （无：bulk-group `wait_group.read 1`） |

**为什么是 8、4、2、2 和 4 个 stage？** 这些深度是预算决策。本章把它们视为给定；下面的协议对任何深度都适用。每个深度各有一条一行长的理由（源自源码）。8 个 mainloop stage 是动态共享内存所容纳的内容：`230400 B = 512 B`（位于偏移 0..343 的屏障、两个 16 字节的 CLC 响应槽位和 TMEM 基指针，带填充）`+ 4 x 8192 B`（C/D stage）`+ 512 B`（填充）`+ 8 x (16384 + 8192) B`（A/B stage），explicit.cu:139-160。4 个累加器 stage 是 512 个 TMEM 列（`kTmemColumns`，explicit.cu:114）除以一个 128 x 128 tile 的 128 个 fp32 列。2 个 CLC 槽位足够，因为 throttle（4.4.5）允许在正在加载的 tile 之外最多有一个查询在途；throttle 环复制了这个深度。4 个 C/D stage 是 `Sm100TmaWarpSpecialized<4,2,16,true,false>` epilogue 策略的 `StagesC`（4.4.8）。第 3 章说明了为什么这些预算对 Speed of Light 而言是正确的（每个 stage 的 smem 对比 k-tile 时间，TMEM 深度对比 epilogue 的滞后）；第 5 章给出字节 layout 以及把它钉住的 `static_assert`。

为什么 C/D 环没有 mbarrier？它的 consumer 是 TMA 存储单元，而硬件已经通过 *bulk group*（`commit_group` / `wait_group`）跟踪 TMA 存储；一个 mbarrier 会与之重复。为什么 throttle 环有屏障却没有数据？它是一个纯粹的流控 token。复用 mbarrier 机制（每个 stage 两个 8 字节的字，一个 full 屏障和一个 empty 屏障）比发明一种不同的机制更便宜，而且它让每个角色的循环保持一致。两者都在 4.4 中详述。

---

## 4.2 一页纸的 mbarrier

### 4.2.1 对象

一个 `mbarrier` 是共享内存中的一个 8 字节的字（本 kernel 的每个屏障都位于 `smem_base + OFF_*`，偏移 0..343，explicit.cu:139-150）。它保存当前 phase（kernel 只会观察主 phase 的 parity，0 或 1）、待到达计数与期望到达计数，以及 **tx-count**。`mbarrier.init` 设置 phase 0，pending = expected = `count`，tx = 0。当待到达计数*和* tx-count 都为零时，当前 phase 完成；随后待到达计数被重置为期望计数（PTX ISA 9.4，9.7.15.16.3、9.7.15.16.8、9.7.15.16.12，引文见第 2 章，第 2.8.1-2.8.2 节）。表 43 给出的 tx-count 范围是 `-(2^20 - 1) .. 2^20 - 1`：它是有符号的，可以变为负数，4.2.2 和 4.4.1 依赖于这一点。实测（笔记 D.2，`mbarrier.init` 行）：ptxas 把每个 `mbarrier.init` 下沉为一条 `SYNCS.EXCH.64`，其 64 位初始化字为 `{(0x100000 - count) << 1, (0x100000 - count) << 11}`。表 43 的界限是硬件限制：`count` 操作数是 32 位的，但其值必须放得进硬件的字段/范围（NVIDIA 内部）。该字保存的是 `2^20 - count`，即一个 20 位字段中一个递减计数器的补码，这一点是从这一个 SASS 形式读出来的（我的分析；ISA 本身只陈述了范围）。

### 4.2.2 kernel 使用的操作

| Wrapper | PTX | 对对象的效果 | 使用者 |
|---|---|---|---|
| W5 `mbar_init` (explicit.cu:286) | `mbarrier.init.shared::cta.b64 [addr], count` | phase 0，pending = count，tx = 0 | prologue 中 warp 0、3、4、5 的被选中的 lane |
| W9 `mbar_arrive_local` (:346) | `mbarrier.arrive.shared::cta.b64 _, [addr]` | pending -= 1 | throttle producer（rank 0 warp 2） |
| W10 `mbar_arrive_remote` (:357) | `mapa.shared::cluster.u32 rem, addr, cta; mbarrier.arrive.shared::cluster.b64 _, [rem]` | 在 CTA `cta` 的位于相同偏移处的屏障上 pending -= 1 | CLC 与 throttle 向 CTA 0 的释放；向 peer 的 dealloc 握手 |
| W11 `mbar_arrive_cluster_masked` (:370) | `mbarrier.arrive.shared::cluster.b64 _, [addr & 0xFEFFFFFF]` | 在 pair leader 的屏障上 pending -= 1 | 累加器释放，每个 epilogue 线程 |
| W12 `mbar_arrive_expect_tx_local` (:382) | `mbarrier.arrive.expect_tx.shared::cta.b64 _, [addr], bytes` | tx += bytes，然后 pending -= 1（一条指令，expect-tx 先于 arrive-on：9.7.15.16.16） | mainloop producer，仅 leader lane |
| W13 `mbar_arrive_expect_tx_remote_pred` (:393) | `@p mapa ...; @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [rem], bytes` | 相同，在 CTA `cta` 的屏障上 | CLC 装填，rank 0 warp 1 的 lane 0..3 |
| （硬件）TMA / CLC `complete_tx` | 隐含在 `cp.async.bulk.tensor ... mbarrier::complete_tx::bytes` 与 `clusterlaunchcontrol.try_cancel ... mbarrier::complete_tx::bytes` 之中 | tx -= 已落地的字节数 | TMA 与 CLC 单元 |
| （硬件）`tcgen05.commit` (W30, :532) | `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [addr], mask` | 当发射线程之前的所有 `tcgen05` 操作都已完成时，在 `mask` 中每个 CTA 的位于 `addr` 偏移处的屏障上 pending -= 1 | mainloop 释放（掩码 `0xF`），累加器 commit（掩码 `0x3 << 2y`） |
| W7 `mbar_try_wait` (:314) | `mbarrier.try_wait.parity.shared::cta.b64 P, [addr], parity` | 若该 parity 的 phase 已完成则返回 1 | peek |
| W6 `mbar_wait` (:297) | 相同，带有 `0x989680` 的 `timeHint`，置于循环中 | 阻塞直到该 parity 的 phase 完成 | 阻塞等待 |
| W8 `mbar_test_wait` (:329) | `@P2 mbarrier.test_wait.parity.shared::cta.b64 P, [addr], parity` | 非阻塞探针 | 仅 CLC producer 尾段 |

协议建立在这些操作的两个性质之上：

1. **`arrive.expect_tx` 是一条指令。** 一个事务屏障的 phase 只有在两个计数都为零时才完成（9.7.15.16.8，引文见 4.2.1）。mainloop 的 `full[s]` 有 `count = 1`，而那唯一的一次到达就是 leader 的 `arrive.expect_tx(49152)`。在它执行之前，待到达计数为 1 且 phase 无法完成，无论已经落地了多少字节。这使协议对*谁先到*不敏感。来自另一个 CTA 的 TMA box 可能在 leader 宣布预期之前就已落地；提前到来的 `complete_tx` 只是把有符号的 tx-count 推向负数，k 个提前的 box 使其变为 -8192 k（表 43，引文见 4.2.1）。phase 仍然等待 arrive，arrive 给 tx-count 加上 49152（剩下 49152 - 8192 k），剩余的字节把它恰好推到零。协议依赖于这一点，这是源自源码的：`load_ktiles` 中没有任何东西把 leader 的 `arrive.expect_tx` 排在*其他*三个 CTA 对同一 stage 的 TMA 发射之前（4.4.1）。
2. **`tcgen05.commit` 是一次自带等待的 arrive。** 它不在发射时 arrive；它在发射线程更早的 tensor-core 工作完成时才 arrive。PTX ISA 9.4（9.7.18.12.1）：`tcgen05.commit` "Makes the mbarrier object track the completion of all prior async-tcgen05 operations initiated by the executing thread"；"the instruction tcgen05.commit.cta_group::2 tracks for the completion of all prior asynchronous tcgen05 operations with .cta_group::2 issued by the current thread"；以及 "Upon the completion of the tracked asynchronous tcgen05 operations, the signal specified by the .completion_mechanism is triggered by the system on the mbarrier object"（此处为 `.mbarrier::arrive::one`：一次到达）。这就是为什么 mainloop 释放和累加器 commit 是 `tcgen05.commit` 而不是 `mbarrier.arrive`（笔记，第 13 节第 1 项是反向的警告：epilogue 对累加器的*释放*是一次普通的 arrive，而不是 commit 操作）。

### 4.2.3 parity：我在等待哪个 phase？

CUDA Programming Guide（"Asynchronous Barriers"，4.10.3 Explicit Phase Tracking）陈述："When we initialize a barrier, its phase has parity 0"，以及 "The `phaseParity` operand is the integer parity of either the current phase or the immediately preceding phase of the barrier object. An even phase has integer parity 0 and an odd phase has integer parity 1." PTX 的 `try_wait.parity` 在*具有该 parity 的* phase 已完成时返回 true。把两者放到一条时间线上：

```
   phase number :   (-1)        0           1           2           3
   parity       :    1          0           1           0           1
                  complete | current  |           |           |
   after init:        ^
   wait(parity 0): current phase 0 has parity 0 and is incomplete  -> BLOCK until phase 0 completes
   wait(parity 1): the phase with parity 1 among {current, preceding} is phase -1, deemed complete -> RETURN AT ONCE

   after phase 0 completes (pending == 0 and tx == 0):
                   (-1)        0           1           2
                    1          0           1           0
                  complete  complete   current
   wait(parity 0): phase 0 is the preceding phase and is complete -> RETURN AT ONCE
   wait(parity 1): phase 1 is current and incomplete -> BLOCK
```

所以在一个刚初始化的屏障上，**等待 parity 1 会立即通过，而等待 parity 0 会阻塞**。这就是不对称起始状态的全部原因（explicit.cu:606-609，逐字复制）：

```
// E.6.5  Pipeline state: {index, phase}; producers start {0, 1}, consumers {0, 0} (sm90_pipeline.hpp:254-260); operator++ (:204-213).
// Why the producer starts at phase 1: a freshly initialised mbarrier is in phase 0, and try_wait.parity with parity 1 returns true at once
// ("the previous phase is complete"), so the first pass over the 8 empty[] barriers never blocks; the consumer waits parity 0 on full[]
// and blocks until the first arrival + transaction bytes complete phase 0. After every wrap of the ring the phase bit flips.
```

producer 等待 `empty[s]`；没有人在它上面 arrive 过，这些 stage *确实是*空的，而且第一遍不能阻塞：parity 1。consumer 等待 `full[s]`；没有人填充过它，而它必须阻塞：parity 0。CUTLASS 把这一点编码在 `make_producer_start_state` (sm90_pipeline.hpp:254-260: `InitialProducerPhase = 1`) 和默认构造的 consumer 状态 `{0, 0, 0}` 中。

**为什么 1 位的 parity 就足够（我的分析）。** 等待者从不计数 phase；它每当自己的环回绕一圈就翻转一个位。只有当屏障从不领先等待者超过一个 phase 时，这才是无歧义的。环的纪律保证了这一点。producer 不能在 consumer 释放第 n 次使用之前开始对一个 stage 的第 n + S 次使用（它等待 `empty`），而 consumer 不能在 producer 填充它之前看到第 n + S 次使用（它等待 `full`）。所以每个屏障在每一时刻都要么处于等待者所询问的 phase，要么处于紧随其后的那一个。打破这个环（例如让 producer 跳过一次 `empty` 等待），parity 方案就会悄无声息地把相隔两个的 phase 混叠在一起：那是数据损坏，而不是挂起。

**ISA 的每 phase 一次等待规则。** PTX ISA 9.4（9.7.15.16.5.1，"Primary phase"）增加了一条每个环都必须遵守的规则："For each primary phase of the mbarrier object, at least one test_wait or try_wait operation must be performed which returns True for waitComplete before an arrive-on operation in the subsequent primary phase." kernel 通过环的纪律本身满足了它：phase k + 1 的每一次到达在因果上都位于所有者对 phase k 的成功等待之后（源自源码）。`full[s]`：leader 的 warp 0 每个 k-tile 都等待它。它上面的下一次 `arrive.expect_tx` 来自 leader lane，发生在 warp 2 通过了 `empty[s]` 之后，而 `empty[s]` 在 leader 于那次等待之后发射的 commit 之前无法完成。`empty[s]`：warp 2 在每次重新填充之前都等待它，而接下来的 commit 需要重新填充后的 stage。`acc_full[st]`：两个 CTA 的 epilogue 每个 tile 都等待它，而下一次 commit 需要 `acc_empty[st]`，后者需要那些 epilogue 的释放。`acc_empty[st]`：leader 的 warp 0 在每个 MMA tile 之前都等待它。`clc_full[i]`：每个 consumer 都在自己的 CTA 中等待它。`clc_empty[i]`：scheduler 在每次重新装填之前都等待它。两个 throttle 屏障：rank 0 的 warp 1 和 2 在下一个 phase 的每次到达之前都等待它们。`tmem_dealloc` 完成一个 phase 并被等待一次。那些从不被等待的屏障（peer 的 `full[]` 和 `acc_empty[]`，rank 1-3 的 `clc_empty[]` 和 throttle 屏障）也从不被 arrive。4.8 的尾段*不是*这条规则所要求的，因为它们之后没有 arrive 跟随；它们是为了退出安全而存在的（我的分析）。

### 4.2.4 两种 wait 形式

阻塞形式，即 W6 `mbar_wait`（explicit.cu:297-311，复制自 `ClusterBarrier::wait`，barrier.h:410-430），在第 6.3.3 节中引用。

peek 形式，即 W7 `mbar_try_wait`（explicit.cu:314-326，复制自 `ClusterBarrier::try_wait`，barrier.h:461-478），在第 6.3.4 节中引用。

阻塞形式的第三个操作数 `kWaitTicks = 0x989680`（explicit.cu:189）是 PTX 的 `timeHint`：`0x989680 = 10,000,000` ns，即 10 ms，是单次 `try_wait` 在返回 false 并由循环（`bra LAB_WAIT`，第 6.3.3 节）重试之前，可用于挂起线程的时间上限（ISA 的定义，PTX ISA 9.4，9.7.15.16.19，在第 2 章，第 2.8.4 节中引用）。peek 形式*仍然*是 `try_wait`，ISA 将其定义为可能阻塞、最长可达一个系统相关的时限；该提示只是提供 10 ms 作为一个 "may be used ... instead of the system-dependent limit"（ISA 措辞）的时限。没有该提示时，peek 可能免费吸收一次短暂的等待，这就是为什么 CUTLASS 说 try 函数 "opportunistically waits for an implementation-dependent timeout"（sm90_pipeline.hpp:410，在 4.5.1 中引用）。kernel 使用的三种 wait 形式，以及每种各用在何处：

| 形式 | PTX | 是否阻塞？ | 用途 | 为何在那里 |
|---|---|---|---|---|
| (a) `mbar_test_wait` (W8) | `mbarrier.test_wait.parity` | 从不（非阻塞探针） | 仅 CLC producer 尾段 (4.8.2) | `PipelineCLCFetchAsync::producer_tail` 在阻塞之前先用 `test_wait` 探测 (sm100_pipeline.hpp:1041-1050)；explicit kernel 保留了这一点，因此 SASS 保留了它的两条不带 `.TRYWAIT` 的无谓词 `SYNCS.PHASECHK.TRANS64`（0x1d70、0x1e50；重试循环中带谓词的再检查使用同一助记符） |
| (b) `mbar_try_wait` (W7)，无提示 | `mbarrier.try_wait.parity` | 可能阻塞，最长可达一个系统相关的时限 | 每次 peek (4.5) | 结果只在稍后才被消费，因此一次短暂的免费等待不花任何代价，还可能省掉 finalize |
| (c) `mbar_wait` (W6)，提示 `0x989680`，循环 | `mbarrier.try_wait.parity ..., timeHint` | 直到 phase 完成（每次尝试 10 ms 时间提示，然后重试） | 每次阻塞 wait 与每次 finalize | `ClusterBarrier::wait` (barrier.h:410-430) 是 CUTLASS 各流水线中唯一的阻塞 mbarrier wait |

在 SASS 中，每条 `try_wait.parity` 都是 `SYNCS.PHASECHK.TRANS64.TRYWAIT P, [addr], R`，parity 位于 `R` 的第 31 位；`.TRYWAIT` 是最初的非阻塞探针，未命中时标准序列用 `NANOSLEEP` 退避，然后在跳回之前把裸的 `SYNCS.PHASECHK.TRANS64` 用作重试路径上的阻塞检查；这两种形式测试的是同一个 phase，但并不等价，而且该睡眠不是周期性轮询：一次失败的 `.TRYWAIT` 会装填一个唤醒触发器，屏障的 phase 完成会唤醒正在睡眠的 warp（NVIDIA 内部）。只有阻塞形式 (W6) 得到那个循环 `TRYWAIT; @!P NANOSLEEP.SYNCS 0x989680; @!P SYNCS.PHASECHK.TRANS64; @!P BRA`，通常为非内联，而 `test_wait.parity` (W8) 则是不带 `.TRYWAIT` 的普通 `SYNCS.PHASECHK.TRANS64`（笔记 D.2，`mbarrier.try_wait.parity` 行，在 CUTLASS kernel 上实测；explicit kernel 见笔记 G.3；第 12 章，第 12.2.2 节有两者的清单以及循环计数与指令计数）。`mbar_wait` 与整个 `EXPLICIT_WAIT(...)` 宏（explicit.cu:237，开关关闭时：`mbar_wait((addr), (parity))`）是同一条指令。

### 4.2.5 fence 与会合

另有三条同步指令是该协议的一部分，尽管它们并不是 mbarrier 操作：

- W14 `fence.mbarrier_init.release.cluster`（explicit.cu:408；barrier.h:712-720），在每组 `mbarrier.init` 之后：它以 release 语义在 cluster 范围内把已初始化的屏障字发布给将在其上 arrive 的远程 CTA。默认为 `.acquire` 的 `barrier.cluster.wait` 提供 acquire 一侧；release 一侧必须来自该 fence，因为 `barrier.cluster.arrive.relaxed` 什么也不发布（ISA 引文见第 2 章，第 2.10.1 节；CUTLASS 的 "must be composed with an appropriate sync instruction" 注释见第 11 章，第 11.5.4 节）。一个 release fence 只覆盖*先于*它的内容，所以 explicit.cu:1166 处位于第四个 fence 之后的 `tmem_dealloc` init 不被其中任何一个覆盖 (4.4.6)。实测：该 fence 没有自己的 SASS 指令；每组 init 之前的裸 `FENCE.VIEW.ASYNC.S` 经推断归属于 init 的下沉（笔记 D.2、G.3；计数见 11.5.4）。

- W16/W17 `barrier.cluster.arrive.relaxed.aligned` / `barrier.cluster.wait.aligned`（explicit.cu:428-429；cluster_sm90.hpp:48-51, 66-69）：cluster 范围的会合，即那条 "appropriate sync instruction"。每个 CTA 在它的各个 init 与 fence 之后到达（explicit.cu:1170），进行它的 warp-uniform 设置，然后等待（explicit.cu:1179）。在该 wait 返回之前不会发射任何远程屏障操作（笔记，第 7.3 节步骤 6）。实测（笔记 D.2）：`UCGABAR_ARV` 与 `UCGABAR_WAIT; CCTL.IVALL`。
- W15 `fence.proxy.async.shared::cta`（explicit.cu:418；barrier.h:728-736）：把本线程的*通用 proxy* 共享内存访问（`st.shared`、`ld.shared`）排序在后续的*异步 proxy* 访问（TMA 存储的读取、CLC 响应的写入）之前。用在每次 TMA 存储之前（explicit.cu:1050）以及每次 CLC 解码之后（explicit.cu:683）。实测：`MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`。

---

## 4.3 流水线状态与 parity 公式

### 4.3.1 状态及其递增

每个角色为它触及的每个环携带一对 `{index, phase}`，并用一个函数推进它（`advance<Stages>`，explicit.cu:610-613，在第 7.2.1 节中引用）。

这就是 `PipelineState::operator++`（sm90_pipeline.hpp:204-213）减去 `count_` 成员，CUTLASS kernel 保留该成员只是为了 trace 探针、epilogue 的 fusion 回调迭代计数器以及 TMA 存储簿记（sm100_epilogue_tma_warpspecialized.hpp:538, 775, 791, 852, 926）。explicit.cu 中的各状态及其起始值：

| 环 | 角色 | 变量 | 起始值 | 行 |
|---|---|---|---|---|
| Mainloop | warp 2 (producer) | `st.prod_idx, st.prod_phase` | `{0, 1}` | 724 |
| Mainloop | warp 0 leader (consumer) | `cons_idx, cons_phase` | `{0, 0}` | 903 |
| 累加器 | warp 0（producer；两个 CTA 都推进它） | `accp_idx, accp_phase` | `{0, 1}` | 904 |
| 累加器 | warp 4-7 (consumer) | `accc_idx, accc_phase` | `{0, 0}` | 1019 |
| CLC | rank 0 warp 1 (producer) | `clp_idx, clp_phase` | `{0, 1}` | 823 |
| CLC | 每个消费 warp | `clc_idx, clc_phase` | `{0, 0}` | 767, 824, 905, 1020 |
| Throttle | rank 0 warp 2 (producer) | `thr_idx, thr_phase` | `{0, 1}` | 766 |
| Throttle | rank 0 warp 1 (consumer) | `thc_idx, thc_phase` | `{0, 0}` | 822 |

mainloop 的 producer 状态是*跨 tile* 携带的（`ProducerState`，explicit.cu:723-726，在第 8.3 节中引用）。

为什么是携带而不是每个 tile 重置？smem 环并不知道 tile 的存在。tile t + 1 的 k-tile 0 取用紧接在 tile t 的 k-tile 127 所用的那个之后的 stage，而屏障的 phase 继续计数。对于不是 16 的倍数的 k-tile 数量，重置后的状态会复用屏障已经越过的 parity 值：数据损坏。（对于本次运行的 128 个 k-tile，携带的 producer 状态恰好在每个 tile 边界回到 `{0, 1}`，4.3.3，所以重置会与之重合。）

### 4.3.2 公式

对于一个有 S 个 stage 的环的、从 kernel 启动起算的第 n 次使用（n = 0, 1, 2, ...）：

```
stage            s(n) = n mod S
consumer parity  c(n) = floor(n / S) mod 2        -- waits on full[s]
producer parity  p(n) = 1 xor c(n)                 -- waits on empty[s]
```

推导。环是轮转使用的，所以第 n 次使用落在 stage `n mod S` 上，并且是该 stage 的第 `floor(n / S)` 次使用。一个 stage 的 full 屏障每次使用完成一个 phase。当 consumer 等待第 n 次使用时，它等待的是 `full[s]` 的 phase `floor(n / S)`，其 parity 为 `floor(n / S) mod 2`。一个 stage 的 empty 屏障每次*释放*完成一个 phase。要写入第 n 次使用，producer 需要第 n - S 次使用的释放，即 `empty[s]` 的 phase `floor(n / S) - 1`，parity 为 `(floor(n / S) - 1) mod 2 = 1 xor c(n)`。对于 `floor(n / S) = 0`，“第 -S 次使用的释放”就是 4.2.3 中那个虚构的已完成 phase -1，parity 为 1；这就是为什么公式对第一遍给出的 `p = 1` 能不阻塞地通过。这就是笔记第 7.2 节关于 phase 簿记那一段的内容；kernel 的 `advance<Stages>` 以增量方式计算它。

### 4.3.3 算例表：mainloop 环 (S = 8)

mainloop 使用 `n = 128 t + q`（每个 tile 128 个 k-tile，tile t 的 k-tile q）。对于 tile 0，q = 0..17：

| q | n | stage s | s 的第几次使用 = n / 8 | consumer 在 `full[s]` 上的 parity | producer 在 `empty[s]` 上的 parity |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 | 0 | 1 |
| 2 | 2 | 2 | 0 | 0 | 1 |
| 3 | 3 | 3 | 0 | 0 | 1 |
| 4 | 4 | 4 | 0 | 0 | 1 |
| 5 | 5 | 5 | 0 | 0 | 1 |
| 6 | 6 | 6 | 0 | 0 | 1 |
| 7 | 7 | 7 | 0 | 0 | 1 |
| 8 | 8 | 0 | 1 | 1 | 0 |
| 9 | 9 | 1 | 1 | 1 | 0 |
| 10 | 10 | 2 | 1 | 1 | 0 |
| 11 | 11 | 3 | 1 | 1 | 0 |
| 12 | 12 | 4 | 1 | 1 | 0 |
| 13 | 13 | 5 | 1 | 1 | 0 |
| 14 | 14 | 6 | 1 | 1 | 0 |
| 15 | 15 | 7 | 1 | 1 | 0 |
| 16 | 16 | 0 | 2 | 0 | 1 |
| 17 | 17 | 1 | 2 | 0 | 1 |

读第一行：producer 以 parity 1 等待 `empty[0]`（立即通过，该 stage 是空的）；consumer 以 parity 0 等待 `full[0]`（阻塞，直到 leader 的 arrive 与 49152 字节完成 phase 0）。q = 8 这一行：producer 再次想要 stage 0，并以 parity 0 等待 `empty[0]`，即等待 `empty[0]` 的 phase 0，它由两个 leader 为 k-tile 0 发出的 `tcgen05.commit` 完成。consumer 以 parity 1 等待 `full[0]`，等待第二次填充。

**为什么每个 tile 都从 stage 0 以相同的 parity 开始。** 对于 tile t，`n = 128 t + q`，所以 `s = q mod 8`（128 是 8 的倍数），并且 `floor(n / 8) = 16 t + floor(q / 8)`，其 parity 等于 `floor(q / 8) mod 2`，因为 `16 t` 是偶数。用文字来说：128 个 k-tile 是 8 个 stage 的环的 16 次完整绕圈；phase 位翻转 16 次；16 是偶数；所以上表是*每个* tile 的表。这是这次固定运行的性质（`128 % 16 == 0`），不是协议的性质。一般地，这里用 K 记每个 tile 的 k-tile 数量（`k_tiles`，本次运行中为 128；不是 GEMM 维度），tile t 从 stage `(K t) mod 8` 开始，phase 位为 `floor(K t / 8) mod 2`。所以“所有 tile 看起来都一样”需要两个条件：`K mod 8 == 0`（stage 0）与 `floor(K / 8)` 为偶数（相同的 phase 位），合起来就是 `K mod 16 == 0`（笔记，第 7.2 节："every tile starts at stage 0 with the same phase, since 128 is a multiple of 16"）。tile 1 的三种情形：K = 100 从 stage 4（100 mod 8）开始，phase 位*相同*（12 圈，偶数）；K = 104 从 stage 0 开始，phase 位*翻转*（13 圈，奇数）；K = 108 从 stage 4 开始，phase 位翻转。在每种情形下携带的 `ProducerState` 都是正确的，因为 `advance<8>` 每绕一圈恰好把该位翻转一次（explicit.cu:611-613）；丢失的只是那条方便的陈述。

### 4.3.4 算例表：累加器环 (S = 4)，n = t

| tile t | stage st = t % 4 | 第几次使用 = t / 4 | epilogue 在 `acc_full[st]` 上的 parity | MMA 在 `acc_empty[st]` 上的 parity |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 1 |
| 1 | 1 | 0 | 0 | 1 |
| 2 | 2 | 0 | 0 | 1 |
| 3 | 3 | 0 | 0 | 1 |
| 4 | 0 | 1 | 1 | 0 |
| 5 | 1 | 1 | 1 | 0 |

tile 4 复用 TMEM stage 0：pair leader 的 MMA warp 以 parity 0 等待 `acc_empty[0]`（explicit.cu:915-918；peer 的 warp 0 只推进状态，:946），即等待释放了 tile 0 的累加器的那 256 次 epilogue 到达；tile 4 的 epilogue 以 parity 1 等待 `acc_full[0]`，即进入该屏障的第二次 commit。

### 4.3.5 算例表：CLC 环与 throttle 环 (S = 2)，n = t

| tile t | 槽位 i = t % 2 | 第几次使用 = t / 2 | consumer parity（`clc_full[i]`、`thr_full[i]`） | producer parity（`clc_empty[i]`、`thr_empty[i]`） |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 1 |
| 1 | 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 1 | 0 |
| 3 | 1 | 1 | 1 | 0 |
| 4 | 0 | 2 | 0 | 1 |
| 5 | 1 | 2 | 0 | 1 |

实测（笔记 D.3，"Slot, parity, coverage"）：被追踪的那次运行的全部 1024 条查询记录都符合 `slot = count % 2`、producer parity `1 ^ ((count >> 1) & 1)` 以及屏障地址 `base + 208 + 8 slot`，并且所有消费记录都符合 `slot = count % 2` 与 consumer parity `(count >> 1) & 1`。

---

## 4.4 逐行解读屏障表

协议常量 (explicit.cu:180-191) 在第 5.9 节中引用。

表格。偏移是从 `smem_base` 起算的字节数；“谁的屏障”指的是到达落在其共享内存中的那个 CTA。

| # | 流水线（CUTLASS 类型） | S | 偏移 | Full：计数 / tx / 到达 | Empty：计数 / 到达 | 初始化 warp (explicit.cu) |
|---|---|---|---|---|---|---|
| 1 | Mainloop A/B `PipelineTmaUmmaAsync<8,(2,2,1),(2,1,1)>` | 8 | full 0 + 8s, empty 64 + 8s | **1 + 49152 tx**：pair leader 的被选中的 lane，`mbarrier.arrive.expect_tx.shared::cta`（W12，本地，在 leader 自己的 `full[s]` 上）；落在该 pair 中的六个 8192 字节的 box 通过 peer 位掩码后的屏障地址完成这些字节 | **2**：来自两个 pair leader 各自的 `tcgen05.commit ... multicast::cluster` 掩码 `0xF`（W30）落在每个 CTA 的 `empty[s]` 上 | 0 (:1124-1133) |
| 2 | Epilogue C 加载 `PipelineTransactionAsync<4>` | 4 | 128 + 8i, 160 + 8i | 32 + 8192 tx | 128 | （仅 beta != 0 时；已保留，未初始化：:141-142） |
| 3 | 加载顺序 `OrderedSequenceBarrier<1,2>` | 2 | 192 + 8i | 各 32 | | （仅 beta != 0 时；未初始化：:143） |
| 4 | CLC `PipelineCLCFetchAsync<2,(2,2,1)>` | 2 | full 208 + 8i, empty 224 + 8i | **1 + 16 tx**：rank 0 warp 1 的 lane 0..3，经由 `mapa` 的 `mbarrier.arrive.expect_tx.shared::cluster`（W13），lane c（c = 0..3）落到 CTA c 的 `clc_full[i]` 上，对应当前槽位 i；multicast 响应在每个 CTA 的屏障上完成 16 字节 | **800** = 32 + 4 x (32 + 32 + 128)：每个消费线程，`mapa` + `mbarrier.arrive.shared::cluster`（W10）落到 **CTA 0 的** `clc_empty[i]` 上 | 4 (:1135-1144) |
| 5 | 累加器 `PipelineUmmaAsync<4,(2,1,1)>` | 4 | full 240 + 8st, empty 272 + 8st | **1**：leader 的带 pair 掩码 `0x3 << 2y` 的 `tcgen05.commit ... multicast::cluster`（W30），落在两个 CTA 的 `acc_full[st]` 上 | **256** = 2 x 128：两个 CTA 的每个 epilogue 线程，经由 peer 位掩码在 **leader 的** `acc_empty[st]` 上 `mbarrier.arrive.shared::cluster`（W11） | 5 (:1145-1154) |
| 6 | CLC throttle `PipelineAsync<2>` | 2 | full 304 + 8i, empty 320 + 8i | **32**：rank 0 warp 2，所有 lane，`mbarrier.arrive.shared::cta`（W9，本地） | **32**：rank 0 warp 1，所有 lane，`mapa` 到 CTA 0（它自己）+ `mbarrier.arrive.shared::cluster`（W10） | 3 (:1155-1164) |
| 7 | TMEM dealloc `ClusterBarrier` | 1 | 336 | **32**：该 pair 的另一个 MMA warp，`mapa` + arrive（W10） | （单个屏障；握手，4.8.3） | 0，`lane_pred` (:1165-1167) |
| 8 | 命名屏障 1（`EpilogueBarrier`） | | 硬件 | `bar.sync 1, 128`：四个 epilogue warp，每个 subtile 两次 | | |
| 9 | 命名屏障 6（`TmemAllocBarrier`） | | 硬件 | warp 0 执行 `bar.arrive 6, 160`；warp 4-7 执行 `bar.sync 6, 160` | | |
| 10 | TMA 存储 `PipelineTmaStore<4,1>` | 4 | 无 | 每个 subtile 的四次存储之后的 `cp.async.bulk.commit_group` | `cp.async.bulk.wait_group.read 1` | |

第 2 行和第 3 行之所以列出，是因为它们存在于 CUTLASS kernel 的共享内存 layout 中（这些偏移被保留下来，使得之后的每个偏移都完全相同，笔记第 6.2 节）；explicit kernel 并不初始化它们（33 条 `mbarrier.init`，而 baseline 为 43 条，笔记 D.7）。下面是原因。

### 4.4.1 第 1 行，mainloop full 屏障：计数 1，49152 字节，leader 的屏障

**谁来填充一个 pair 的 stage s。** 取 pair {0, 1} (y = 0)。该 pair 的两个 smem 映像接收到：

```
                 lands in rank 0                      lands in rank 1
   A rows  0.. 63 of the 128x64 stage:  box from rank 0 (slot +0)        box from rank 1 (slot +0)
   A rows 64..127:                      box from rank 2 (slot +8192)     box from rank 3 (slot +8192)
   B, this CTA's 64 columns:            box from rank 0                  box from rank 1
                                        -------------------              -------------------
                                        3 boxes x 8192 B                 3 boxes x 8192 B      = 6 x 8192 = 49152 B
```

六个 box 中的每一个都把屏障操作数指定为 `(smem_base + 8 s) & 0xFEFFFFFF`（explicit.cu:770，`full_masked_base`；调用点 :747 和 :754 加上 `+ 8 s`）。对于 rank 1（`smem_base = 0x01000400`，实测，笔记 D.1 D5），`full[3]` 是 `0x01000418`，而被掩码后的操作数是 `0x00000418`，即 rank 0 的 `full[3]`。硬件规则是基于 parity 的，而不是基于 leader 的：带 `.cta_group::2` 时，mbarrier 可以位于目标 CTA 或其 peer 中；带 `.multicast::cluster` 时，每个目标的信号按持有 `mbar` 的 CTA 的 `%cluster_ctarank` parity 送往其 pair 中的偶数或奇数 CTA（PTX ISA 9.4，`cp.async.bulk.tensor`，引文见第 2 章，第 2.7.4 节）。硬件对 leader 一无所知；是 kernel 的掩码使 parity 规则与 leader 重合（笔记 B.6；第 2 章，第 2.9.3 节）。

**为什么计数是 1。** 恰好只有一个线程宣告期望：leader 的 `is_leader_lane`（`lane_pred != 0u && is_leader`，explicit.cu:1192）以 `kMainloopTxBytes` 执行 W12（explicit.cu:736）。CUTLASS：`producer_arv_cnt = 1`（sm100_pipeline.hpp:563）以及 `mainloop_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load`（kernel :465）。

**为什么是 49152。** `TmaTransactionBytes`（sm100_mma_warpspecialized.hpp:238-240）把每个 CTA 的 A 和 B 的 stage 字节数乘以 `size(AtomThrShapeMNK{}) = 2`，因为 "Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly"（:237 处的注释）：`2 x (16384 + 8192) = 49152`（explicit.cu:180；`static_assert` 位于 :192）。Colfax 的 cluster 教程推导出了同一个因子：`int tma_transaction_bytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA)) + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));`。

**为什么无论谁先落地都行得通。** 那四个并非由 leader 自己发射的 box 中的任何一个都可能先落地：peer 的 A 和 B box，以及另一个 pair 的两次 A multicast（rank 2 的那一半落入 leader 的 smem，rank 3 的那一半则落入 peer 的；上表和 4.6 的字节台账）；为 1 的待到达计数使 phase 保持打开，而早到的字节只会把有符号的 tx-count 驱动为负值（4.2.2）。计数 1 是一个设计选择，而不是正确性上的必需。一种计数为 2 的拆分方案，即每个 CTA 的 producer 各自宣告自己的 24576 字节，在 ISA 语义下同样是正确的（完成需要 pending == 0 *且* tx == 0，PTX ISA 9.4，9.7.15.16.8；tx 可以为负，表 43）：到达 A 期望 24576（pending 1，tx 24576）；全部 49152 字节落地（tx -24576，pending 1，未完成）；到达 B 期望 24576（tx 0，pending 0），并恰好在那时完成该 phase。计数 1 换来的是一个*单边*协议。只有 leader 的 lane 才会触碰 leader 的 `full[s]`，所以 peer 的 producer 从不发射远程 arrive（每个 stage 都没有 `mapa` 或 peer 位 arrive），而那单独一次 `expect_tx` 携带了该 pair 的全部字节总数。这就是 CUTLASS 的 `producer_arv_cnt = 1`（sm100_pipeline.hpp:563）配合 `is_leader`（kernel :465）以及 `TmaTransactionBytes` 中的 x2（sm100_mma_warpspecialized.hpp:237-240）（源自源码；与计数为 2 的方案的比较是我的分析）。

**consumer 看到的是什么。** 在它的 `full[s]` wait 之后（explicit.cu:916, 923），leader 的 warp 0 可以发射四条 `tcgen05.mma`，它们在相同的 CTA 相对地址上读取 A 的第 0..255 行（两个 CTA 的 stage）和 B 的第 0..127 列（两个 CTA 的两半）；硬件的 `cta_group::2` MMA 替它读取 peer 的 smem（笔记 B.6）。这就是为什么 leader 中的单个屏障就足够了：该 pair 的 stage s 恰好只有一个读取者，而它就位于 leader 中。

### 4.4.2 第 1 行，mainloop empty 屏障：计数 2，掩码 0xF

**谁读取 CTA r 写入的内容。** CTA r 的 producer (i) 把它的那一半 B 写入 r 自己的 stage s，(ii) 把它的那一半 A 写入 r 自己的 stage s 以及 rank `r ^ 2`（另一个 `y`，同一个 `x`）的 stage s。这些字节的读取者是 r 的 pair leader 的 UMMA（读取 r 的 A 和 B）和 `r ^ 2` 的 pair leader 的 UMMA（读取 r multicast 到 `r ^ 2` 中的那一半 A）。它们就是两个不同的 leader，rank 0 和 2。在*两者*都完成之前，CTA r 不得覆写 stage s，因此**计数 2**。CUTLASS 把它计算为 `multicast_consumer_arrival_count = cluster_m / atom_m + cluster_n / atom_n - 1 = 2/2 + 2/1 - 1 = 2`（sm100_pipeline.hpp:565-566）。各项：收到了这个 CTA 的 A multicast 的每个 MMA-pair 行各一个 leader（`cluster_n / atom_n = 2`）；将会收到 B multicast 的每个 pair 列各一个（`cluster_m / atom_m = 1`）；再减去自己所在的 pair 的那一个，它被两项都计入了。

**为什么掩码是 0xF。** leader 的释放是带 `kMainloopReleaseMask = 0xF` 的 `tcgen05.commit`（explicit.cu:942, 186），即 `calculate_multicast_mask<kRowCol>` 的值（sm100_pipeline.hpp:57-91）：同一 MMA 列（`x / 2` 相等：全部，因为 `atom_m = 2`）或同一行（`y` 相等）中的每个 CTA。对于 2 x 2 的 cluster，那就是全部四个 CTA。commit 操作落在每个被掩码选中的 CTA 中相同偏移处的 `empty[s]` 上，所以每个 CTA 每个 k-tile 从两个 leader 各恰好收到一次到达。两次 multicast commit 满足计数 2；公式和掩码是同一事实的两种视角。

**为什么用 `tcgen05.commit` 而不是 `mbarrier.arrive`。** 这次释放必须意味着 "the tensor core has finished *reading* stage s"，而不是 "warp 0 has finished *issuing* the MMAs"。`tcgen05.commit` 只有在发射线程此前的 `tcgen05` 操作完成之后才会 arrive（4.2.2）。它在第四条 MMA 之后立即发射，没有任何等待（explicit.cu:942；SASS：`UTCBAR.2CTA.MULTICAST` 是第四条 `UTCHMMA` 之后的那条指令，笔记 D.2）；等待由硬件来做。把它换成一次 arrive，producer 就会覆写 tensor core 仍在读取的 stage：数据损坏。

### 4.4.3 第 5 行，累加器屏障：full 1 配掩码 0x3 << 2y，empty 256 在 leader 上

**Full。** 在第 128 个 k-tile 之后，leader 的被选中的 lane 以 `acc_commit_mask = 0x3 << (2 y)` 在 `acc_full[st]` 上发射 `tcgen05.commit`（explicit.cu:944, 902）：pair {0, 1} 为 `0x3`，pair {2, 3} 为 `0xC`，即 `calculate_umma_peer_mask` 的值（sm100_pipeline.hpp:96-110），绝不会是 `0xF`。计数 1（`producer_arv_count = 1`，kernel :528-529："Only one producer thread arrives on this barrier"）。为什么要 multicast 到该 pair 的两个 CTA？`mbarrier.try_wait` 和 `test_wait` 只接受 `.shared{::cta}` 状态空间（PTX ISA 9.4，9.7.15.16.19：`.ss = { .shared{::cta} }`），所以每个 CTA 的 epilogue 只能对*它自己的*屏障执行 wait（explicit.cu:1030, 1034）。同一个 MMA 序列写了两个 CTA 各自的那一半 TMEM，所以两者都需要这个信号。为什么不是 `0xF`？另一个 pair 的 epilogue 对这个 pair 的累加器毫无兴趣；一次误入那里的到达会破坏那个屏障的计数。

**Empty。** 每个 epilogue 线程在它对该 tile 的最后一次 `tcgen05.ld` 之后，并在 `tcgen05.wait::ld` 之后，执行 W11（explicit.cu:1037-1041，引文见第 10.8 节）。

W11（explicit.cu:370-379，引文见第 6.3.7 节）复制自 `umma_arrive_2x1SM_sm0`，barrier.h:905-921，其注释写道 "Always arrive on even SM of collaborating 2 SMs"。

它清除地址的第 24 位，所以 peer 的 128 个线程在 *leader 的* `acc_empty[st]` 上 arrive，而 leader 的 128 个线程在自己的屏障上 arrive：**256** = `size(AtomThrShapeMNK{}) * NumEpilogueThreads`（kernel :530）。为什么只在 leader 上？只有 leader 的 warp 0 发射 MMA，所以只有它会问 "is TMEM stage st free in both CTAs"（explicit.cu:918）。peer 的 `acc_empty[]` 被初始化但从未使用。为什么是每个线程各一次到达，而不是每个 warp 一次被选中的到达（计数 8）？并非出于必要：`tcgen05.wait::ld.sync.aligned`（W32，explicit.cu:558-565）本身就是 warp 集体的，所以在它之后 warp 是收敛的，每个 warp 一次被选中的 arrive 同样安全。决定性的原因是 CUTLASS 定义了 `consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads`（kernel :530），并让每个线程不经选举地执行 `umma_arrive_2x1SM_sm0`；explicit kernel 为了 SASS 等价而镜像了它（每个线程一条 `SYNCS.ARRIVE.TRANS64.RED.A1T0`，笔记 G.3）。两种做法的代价都可以忽略不计（源自源码；关于代价的评论是我的分析）。为什么要在 `tcgen05.wait::ld` 之后？`tcgen05.ld` 是异步的（PTX ISA 9.4，9.7.18.8.3，引文见第 2.4.4 节；gau-nernst，tcgen05 帖子）。这次 wait 保证在该 stage 被宣告可复用之前，寄存器已持有这些值，即 TMEM 读取已经完成。注意这发生在*何时*：在最后一个 subtile 的算术运算和存储之前（笔记第 13 节第 11 项），所以 MMA warp 可以在 epilogue 仍从寄存器存储时就开始覆写该 stage。

### 4.4.4 第 4 行，CLC 屏障：full 1 + 16 字节由远程装填，empty 800 在 CTA 0 上

**Full。** PTX 操作数是 `.b128`，并且 "The response of clusterlaunchcontrol.try_cancel instruction will be 16-byte opaque value"（PTX ISA 9.4，9.7.15.18）；CUTLASS 把它的大小定为 `CLCResponseSize = sizeof(typename TileScheduler::CLCResponse)`（kernel :155，在 :516 处用作 `transaction_bytes`）；explicit.cu:181 硬编码了 `kClcTxBytes = 16`。CLC 单元把它写入*每个* CTA 的 `OFF_CLC_RESPONSE + 16 i`（`.multicast::cluster::all`，W33），并在*每个* CTA 的 `clc_full[i]` 上完成 16 个事务字节（笔记 B.6；CUDA Programming Guide："The result received by all thread blocks corresponds to the local block index {0, 0, 0} within a cluster"）。谁来装填它们？rank 0 的 scheduler warp 的 lane 0..3，每个各装填到一个不同的 CTA 上（explicit.cu:842；`arrive_and_expect_tx(params_.transaction_bytes, lane_idx_, uint32_t(lane_idx_ < cluster_size_))`，sm100_pipeline.hpp:1103）。W13 在全部 32 个 lane 中执行；`setp.eq.u32 p, %2, 1` 谓词（explicit.cu:398）让 lane 0..3 行动，且 `cta_id = lane`。为什么不让每个 CTA 装填自己的屏障？只有 scheduler 知道查询何时发出。从 producer 装填使“一次 producer 到达，计数 1”的形状与 mainloop 环和累加器环保持完全相同。远程 arrive 与响应字节的先后顺序无关紧要，原因在 4.2.2 中给出：来自 CTA 0 的 lane `i` 的远程 `arrive.expect_tx(16)` 与硬件的 `complete_tx(16)` 可以按任一顺序到达，而 `count 1` 加上 `+16/-16` 这一对恰好让 phase 完成一次，因为为 1 的待到达计数使 phase 保持打开。

**Empty。** 把这 800 次到达汇集到 CTA 0 的屏障，而不是每个 CTA 各一个屏障，使 producer 只需一次 wait 而不是四次。（CUTLASS 的 CLC 文档 media/docs/cpp/blackwell_cluster_launch_control.md 说 "The CLC pipeline has a depth of 3"；这个 kernel 实例化的是 `PipelineCLCFetchAsync<2, ...>`，在笔记 D.1 D1 中实测，所以这里的深度是 2。）

**该 fence。** 解码之后的 `fence.proxy.async.shared::cta`（explicit.cu:683）来自 `work_tile_info_from_clc_response`（sm100_tile_scheduler.hpp:430）。解码是对一个槽位的通用 proxy `ld.shared.b128`，而*下一次* `try_cancel`——一个异步 proxy 写入方——将覆写该槽位；fence 把这次读取排在允许那次写入的释放之前。

### 4.4.5 第 6 行，throttle：32 / 32，仅 rank 0

`PipelineAsync<2>`，带 `producer_arv_count = NumMainloopLoadThreads = 32`、`consumer_arv_count = NumSchedThreads = 32`、`dst_blockid = 0`（kernel :546-548）。在每个 tile 的加载之前，rank 0 的 producer warp 等待 `thr_empty[i]`（explicit.cu:784），并以全部 32 个 lane 在本地对 `thr_full[i]` arrive（explicit.cu:785；`full_barrier_ptr_[stage].arrive()`，sm90_pipeline.hpp:1195）。这次 wait 是阻塞式的，没有 peek：CUTLASS kernel 以默认的 `WaitAgain` token 调用 `clc_throttle_pipeline.producer_acquire(state)`，前面没有 `producer_try_acquire`，尽管 `PipelineAsync` 提供了一个（kernel :633；sm90_pipeline.hpp:1105, 1110）。scheduler 等待 `thr_full[i]`，并以全部 32 个 lane 经由 `mapa` 对 CTA 0——也就是它自己——的 `thr_empty[i]` arrive（explicit.cu:836-837；`empty_barrier_ptr_[stage].arrive(params_.dst_blockid)`，sm90_pipeline.hpp:1238）。为什么是 32 而不是 1？`PipelineAsync` 没有 leader 选举；每个 lane 都 arrive，所以计数就是 warp 大小。这不花费任何额外指令：无论是一个 lane 还是全部 32 个执行它，warp 都只发射一条 `mbarrier.arrive`（我的分析）。这个环为什么存在？为了限制未完成查询的数量：查询 t 只有在 rank 0 的加载者已经*开始* tile t 之后才能发出，所以一个 cluster 预留的未开始 tile 绝不会超过一个（笔记第 7.4 节，D.7）。没有它，scheduler 会领先两个 CLC 槽位，为一个可能是网格中最慢的 cluster 预留 tile：尾段的负载不均衡（我对其目的的分析，与 kernel :692 处的 CUTLASS 注释一致："Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers"）。实测（笔记 D.3，CUTLASS kernel 的 trace 构建）：每个 cluster 未完成的查询从未超过一个；scheduler 从消费响应 t 到发出查询 t + 1 的间隔，即这次 wait，每个 tile 为 28.8-37.7 us（中位数 30.8）。

### 4.4.6 第 7 行，tmem_dealloc：计数 32

`NumMMAThreads = NumThreadsPerWarp = 32`（kernel :138 处是定义；:565 处是 `tmem_deallocation_result_barrier.init(NumMMAThreads)`）。warp 0 的 `lane_pred` lane 在第四个也是最后一个 `fence.mbarrier_init.release.cluster`（explicit.cu:1164-1167）*之后*才初始化它，因此没有任何 fence 覆盖它。release fence 只对在它之前的操作排序，而随后的 `barrier.cluster.arrive.relaxed`（:1170）没有 release 语义 (4.2.5)。explicit kernel 仿照 CUTLASS，后者在 `tmem_deallocation_result_barrier.init`（kernel :565）与 `pipeline_init_arrive_relaxed`（:580，即 `cute::cluster_arrive_relaxed()`，sm90_pipeline.hpp:1377-1384）之间同样没有 `fence_barrier_init`。实际上的保证是：这个屏障上仅有的到达只发生一次，在 kernel 退出时 (4.8.3)，此前 pair 之间已经历了数千次 cluster 范围的同步。这是否在形式上把初始化字发布给了 peer，我标为（未验证）。

### 4.4.7 第 8-9 行，命名屏障

硬件命名屏障以线程计数，按 32 的倍数，存在 id 0..15；CUTLASS 把 1 留给 `EpilogueBarrier`、把 6 留给 `TmemAllocBarrier`（barrier.h:170, 175），并避开 0，因为编译器通常把 `__syncthreads()` 下沉为一个 CTA 屏障，一般是屏障 0，所以屏障 0 被视为编译器所有（NVIDIA 内部）；barrier.h:163-164 陈述了同一规则。

- **屏障 6，160 个线程 = 32 + 128。** Warp 0 分配 TMEM（W26，一条 warp 集体的 `.sync.aligned` 指令，把基址写入 `OFF_TMEM_BASE_PTR`），像 baseline 一样执行 `__syncwarp()`（kernel :729），然后 `bar.arrive 6, 160`（explicit.cu:885-887）。warp 4-7 在读取该地址之前 `bar.sync 6, 160`（explicit.cu:997）。`bar.arrive` 不等待，所以 warp 0 立即进入它的循环。四个 epilogue warp 等到全部 160 次到达都存在为止，这也把它们对该地址的读取排序在 warp 0 的分配之后。没有它，epilogue 会读到未初始化的 TMEM 基址（实测为 0，但 kernel 从不假定这一点）并寻址到错误的 TMEM：数据损坏或故障。
- **屏障 1，128 个线程，每个 subtile 两次**（explicit.cu:1051 与 :1071）。第一次同步：全部 128 个线程都已执行了它们的 16 条 `st.shared` *以及*它们的 `fence.proxy.async`（explicit.cu:1050），然后 warp 4 才发射读取该 stage 的四次 TMA 存储。第二次同步：warp 4 已执行 `wait_group.read 1`，然后才有任何线程写入下一个 subtile 的 stage (4.4.8)。CUTLASS 的 `tma_store_fn` 有同样的两次 `synchronize()` 调用（sm100_epilogue_tma_warpspecialized.hpp:769, 787），并附有注释 "ensure all threads have issued their async fence"。

### 4.4.8 第 10 行，TMA 存储 bulk group

epilogue 的 C/D smem 环有四个 8192 字节的 stage（explicit.cu:111, 153）；subtile e 写入 stage `e % 4`。warp 4 从该 stage 发射四个 2048 字节的 box，然后执行 `cp.async.bulk.commit_group`（W24）与 `cp.async.bulk.wait_group.read 1`（W25）（explicit.cu:1066-1069）。这个 `1` 是 `PipelineTmaStore<StagesC = 4, StagesD - 1 = 1>` 的 `UnacquiredStages`（sm100_epilogue_tma_warpspecialized.hpp:202-204）。`StagesD = 2` 来自笔记 D.1 D1 中实测到的 `Sm100TmaWarpSpecialized<4,2,16,true,false>` 策略。由于 `always_wait = true`，这个等待在每个 subtile 上都会执行（kernel :491；sm90_pipeline.hpp:698-699：`if (params_.always_wait || count > UnacquiredStages) tma_store_wait<UnacquiredStages>()`）。含义：提交第 e 组之后，至多还有一组（e 自身）可能仍在读 smem；第 e - 1 组及其之前的所有组都已读完。下一个 subtile 写入 stage `(e + 1) % 4`，它上一次被第 e - 3 组读取，所以 WAR 冒险被有余量地覆盖。ReuseSmemC 设计付出这一保守等待的代价，是为了在加载 C 时（beta != 0）能够尽早告知 C 加载 producer 哪个 stage 是空闲的（sm100_epilogue_tma_warpspecialized.hpp:742-744, 789-792）。对于 beta = 0，这份簿记是死代码，但等待仍然保留。SASS（笔记 D.2）：`UTMACMDFLUSH` 与 `DEPBAR.LE SB0, 0x1`。

---

## 4.5 先 peek 再 wait

### 4.5.1 模式

CUTLASS 的流水线把一个 "try" 函数与一个 "finalize" 函数配对（sm90_pipeline.hpp:402-416，注释："The 'try' function ... opportunistically waits for an implementation-dependent timeout. Whether or not the barrier has flipped yet, the try function will return a token. If the token indicates that the barrier has not flipped, then the token must be passed into the corresponding 'finalize' function. The finalize function will then block until the barrier has flipped."）。在 explicit kernel 中，token 是一个携带 CUTLASS 自身编码的 `uint32_t tok`，`BarrierStatus::WaitAgain = 0u`、`WaitDone = 1u`（sm90_pipeline.hpp:113-116），这就是为什么 1 表示完成，而 finalize 是 `if (!tok) EXPLICIT_WAIT(...)`。

Producer：explicit.cu:732-739，引文见第 8.4 节。

MMA（explicit.cu:916，逐字照录；k 循环头 explicit.cu:922-927 的引文见第 9.5 节）：

```
      uint32_t tok = (k_tiles <= 0) ? 1u : mbar_try_wait(full_base + 8u * cons_idx, cons_phase);   // consumer_try_wait(skip_wait) (:670-671)
```

Epilogue（explicit.cu:1030, 1034，逐字照录）：

```
    uint32_t tok = mbar_try_wait(acc_full_base + 8u * acc_stage, accc_phase);                   // acc_pipeline.consumer_try_wait, once per tile (:813)
```
```
      if (e == 0 && !tok) { EXPLICIT_WAIT(acc_full_base + 8u * acc_stage, accc_phase, WAIT_ACC_FULL, tile, 0u); }   // consumer_wait on the first subtile (:864-867)
```

### 4.5.2 它为何存在，代价是什么

第 n + 1 次使用的 peek 在第 n 次使用的工作（两次 TMA 发射，或四条 MMA 加 commit 操作）*之前*发射。等到循环回到顶部时，屏障已经有了整个循环体那么长的时间来翻转；在常见情况下 `tok == 1`，阻塞等待被完全跳过。收益不在等待本身（对已完成的 phase 立即执行 `try_wait` 同样很快），而在于*代码的形状*：快速路径没有非内联分支，也没有 `NANOSLEEP`，每个 k-tile 循环都保持为一个直线的循环体。实测（explicit kernel 见笔记 G.3，CUTLASS baseline 见笔记 D.2 与 G.3）：

| 循环 | Explicit kernel（笔记 G.3） | CUTLASS baseline（笔记 D.2、G.3） | 差值的原因（笔记 G.3） |
|---|---|---|---|
| MMA warp，k-tile 循环 | 47 条指令（`0x51f0-0x54d0`：12 条 `UMOV`、9 条 `UIADD3`、5 条 `ULEA`、4 条 `UTCHMMA`、2 条 `TRYWAIT`、1 条 `UTCBAR`、……） | 44（`0x3890-0x3b40`） | +3 `UMOV`、+1 `ULEA`、-1 `UIADD3`：ptxas 仅在 baseline 中做了外提的那一对每 k-block `mov.b32`（指令描述符，以及馈入八个 disable-output-lane 操作数的零）、编译期的 commit 掩码 `0xF`（baseline 在其 tile 循环之前于运行时构造它）、每 k-tile 对 `empty[s]` 地址的一次重新计算，以及一个而非两个循环计数器；全部为 uniform 数据通路指令，在 tensor core 执行前一个 k-tile 期间发射 |
| Producer warp 2，主 k-tile 循环 | 37（快速路径上 33；`0x10b0-0x12f0`） | 31（28） | +6：每 k-tile 对四个循环不变量的五次重新物化（`0xc000` 字节计数、B tensor-map 地址对、multicast 掩码插入、z 坐标），一条为仅 leader 的 arrive 地址做谓词化的 `VOTEU.ALL`，以及一条用于带掩码的 `full[s]` 地址的 `ULEA`，它取代了 baseline 的 `ULOP3 ..., 0xfefffff8`；producer 最多领先 8 个 stage 运行，所以这不在关键路径上 |
| Producer warp 2，8 个 k-tile 的 prologue 循环 | 37（`0x0de0-0x1020`） | 32 | 同样的原因 |
| Epilogue warp，每 tile | 696（设置 56，各 subtile 77/78/78/78/78/78/81/92） | 1265（设置 309） | baseline 的死 C 路径：每 subtile 16 条 `LDS`，位于一个从不成立的谓词之后，以及 `epi_load` 流水线的簿记 |
| Scheduler warp 1，CLC 循环 | 50 | 50 | 计数与结构相同；八个加法/搬移操作码不同（这里是 `IADD3`/`MOV`/`LEA`，那里是 `VIADD`/`IMAD`；第 12.5.3 节） |

挂起提示在另一种情况下很重要。当 phase 尚未完成时，带提示的 `try_wait` 让 warp 进入睡眠而不是在屏障字上自旋，这为 CTA 中其余七个 warp 省出了发射槽（关于目的，这是我的分析；机制 `NANOSLEEP.SYNCS 0x989680` 是实测的；睡眠中的 warp 如何被唤醒：第 4.2.4 节以及第 12 章，第 12.2.2 节）。转述 4.2.4 所引用、第 2 章，第 2.8.4 节所引述的那段 ISA 文本：被挂起的 warp 在 phase 完成或时间限制到期时返回，以先到者为准。因此 10 ms 这个数字是单次尝试的上限（ISA 说该提示 "may be used" 用于该限制），而不是一个延迟。

代价：每次 peek 一条 `SYNCS.PHASECHK.TRANS64.TRYWAIT`。producer **每 tile 做 130 次 peek，128 次被消费，2 次被丢弃**。`load_ktiles` 每 tile 被调用两次（8 个和 120 个 k-tile，explicit.cu:788-791）；每次调用在其循环之前 peek 一次（:732），并在每次迭代中于 `advance` 之后 peek 一次（:739），即 9 + 121 = 130。每次调用的最后一次循环内 peek 看的是该调用不会使用的 stage，而下一次调用（或下一个 tile）会在自己的循环之前重新 peek，所以有 2 次被丢弃。MMA warp **每 tile peek 128 次，全部被消费**：循环之前一次（:916），循环之内 127 次。:927 处的 `k <= 0` 守卫跳过最后一个 k-tile 之后的 peek；在 SASS 中，对下一个 stage 的 peek 是一条带谓词的 `TRYWAIT`，在最后一个 k-tile 上被关闭（explicit kernel 中 `0x51f0-0x54d0` 内的 `@P2 SYNCS...TRYWAIT`，笔记 G.3；baseline 中 `0x39f0` 处的 `@P0`，笔记 D.2）。为什么只在那里跳过？这个 tile 中没有下一个 stage，而下一个 tile 的 `mma()` 以它自己的循环前 peek 开始，所以多余的一次 peek 会是白费的工作。为什么 producer *不*跳过？CUTLASS 的 `load()` 无条件地 peek（`producer_try_acquire(state)` 使用默认的 `skip_wait = false`，sm100_mma_warpspecialized.hpp:617）：加载循环在 prologue 调用与主调用之间共享，最简单的代码胜出，而每 tile 两次浪费掉的 peek 只是噪声。epilogue **每 tile peek 一次**，并在 subtile 0 处 finalize（:1030, :1034）：subtile 0 的 `tcgen05.ld` 需要这些数据，而它之前的任何操作都不需要。throttle、CLC producer 与 CLC consumer **不** peek。它们直接调用阻塞等待（:784, :836, :841, :680），因为在那些循环中没有有用的工作可以放在 peek 与其 finalize 之间。

---

## 4.6 数据流图

图例：`r0..r3` 是 rank（`r = x + 2y`）；`W0`、`W1`、`W2`、`W4-7` 是 warp；`->` 是一次到达或一次字节完成；`[barrier]@rK` 是 CTA K 的共享内存中的一个屏障。四个 CTA 运行同样的代码；箭头仅通过 `x`、`y`、`is_leader`、`is_cta0` 而不同。

```
 MAINLOOP SMEM RING (per stage s; repeated 128 times per tile)
 ------------------------------------------------------------------------------------------------------
   r0.W2  A box (mcast 0x5: r0,r2)  ---------+--------------------------------+
   r2.W2  A box (mcast 0x5: r0,r2)  ---------+---> complete_tx 8192 x2 into   |---> complete_tx 8192 x2 into
   r0.W2  B box (r0 only)           -------------> [full[s]]@r0 (masked addr) |     [full[s]]@r2 (masked addr)
                                                    + r0.W2 leader lane: arrive.expect_tx 49152 (count 1)
   r1.W2  A box (mcast 0xA: r1,r3)  ---------+---> copy landing in r1: complete_tx 8192 into [full[s]]@r0   (r1 is r0's peer: bit 24 cleared)
                                             +---> copy landing in r3: complete_tx 8192 into [full[s]]@r2   (r3 is r2's peer)
   r3.W2  A box (mcast 0xA: r1,r3)  ---------+---> copy landing in r1: complete_tx 8192 into [full[s]]@r0
                                             +---> copy landing in r3: complete_tx 8192 into [full[s]]@r2
   r1.W2  B box (r1 only)           -------------> complete_tx 8192 into [full[s]]@r0
   r2.W2  B box, r3.W2 B box        -------------> complete_tx 8192 each into [full[s]]@r2
                                                    + r2.W2 leader lane: arrive.expect_tx 49152

   r0.W0  waits [full[s]]@r0 -> 4 x tcgen05.mma (reads r0+r1 smem) -> tcgen05.commit mask 0xF ---> [empty[s]]@r0,r1,r2,r3 (1 arrival each)
   r2.W0  waits [full[s]]@r2 -> 4 x tcgen05.mma (reads r2+r3 smem) -> tcgen05.commit mask 0xF ---> [empty[s]]@r0,r1,r2,r3 (1 arrival each)
   rK.W2  waits [empty[s]]@rK (count 2 = one commit from each leader) before reusing stage s

 ACCUMULATOR RING (per tile t, stage st = t % 4)
 ------------------------------------------------------------------------------------------------------
   r0.W0  waits [acc_empty[st]]@r0 (count 256) ... after k-tile 127: tcgen05.commit mask 0x3 ---> [acc_full[st]]@r0, @r1
   r2.W0  waits [acc_empty[st]]@r2 (count 256) ... after k-tile 127: tcgen05.commit mask 0xC ---> [acc_full[st]]@r2, @r3
   r0.W4-7 wait [acc_full[st]]@r0 -> 8 x tcgen05.ld -> wait::ld -> 128 x arrive (masked) ---> [acc_empty[st]]@r0
   r1.W4-7 wait [acc_full[st]]@r1 -> 8 x tcgen05.ld -> wait::ld -> 128 x arrive (masked) ---> [acc_empty[st]]@r0   (bit 24 cleared)
   r2.W4-7 / r3.W4-7: the same into [acc_empty[st]]@r2

 CLC RING (per tile t, slot i = t % 2)                       THROTTLE RING (rank 0 only)
 ------------------------------------------------------------  --------------------------------------------
   r0.W2 (start of tile t): wait [thr_empty[i]]@r0 -> 32 x arrive (local) ---> [thr_full[i]]@r0
   r0.W1: wait [thr_full[i]]@r0 -> 32 x arrive (mapa 0) ---> [thr_empty[i]]@r0
   r0.W1: wait [clc_empty[i]]@r0 (count 800)
   r0.W1 lanes 0..3: arrive.expect_tx 16 (mapa lane) ---> [clc_full[i]]@r0, @r1, @r2, @r3 (count 1 each)
   r0.W1 elected lane: try_cancel ... multicast::cluster::all ---> response bytes into slot i of r0..r3, complete_tx 16 on each [clc_full[i]]@rK
   rK.W0, rK.W2, rK.W4-7 (all K) and r0.W1: wait own [clc_full[i]]@rK -> decode -> fence.proxy.async -> arrive (mapa 0) ---> [clc_empty[i]]@r0
                                                                                          32 + 4 x (32 + 32 + 128) = 800 arrivals

 EXIT HANDSHAKE (once)
 ------------------------------------------------------------------------------------------------------
   r1.W0: 32 x arrive (mapa peer) ---> [tmem_dealloc]@r0 ; r0.W0 waits parity 0 ; r0.W0: 32 x arrive (mapa peer) ---> [tmem_dealloc]@r1 ; r1.W0 waits parity 0
   r3.W0 / r2.W0: the same
```

每 stage 的字节台账（每个 box 都是 8192 B；"via peer bit" 的意思是：拷贝落在奇数 CTA 中，而带掩码的操作数指名的是偶数 CTA 的屏障）：

```
 [full[s]]@r0 = A(r0, landed in r0) + A(r2, landed in r0) + A(r1, landed in r1, via peer bit) + A(r3, landed in r1, via peer bit) + B(r0) + B(r1, via peer bit) = 6 x 8192 = 49152
 [full[s]]@r2 = A(r0, landed in r2) + A(r2, landed in r2) + A(r1, landed in r3, via peer bit) + A(r3, landed in r3, via peer bit) + B(r2) + B(r3, via peer bit) = 6 x 8192 = 49152
 [full[s]]@r1 = [full[s]]@r3 = 0 (initialised, never arrived on, never waited)
```

这个 kernel 使用三种不同的寻址机制：

| 机制 | 指令操作数 | 谁计算目标 | 用于 |
|---|---|---|---|
| Peer 位（`& 0xFEFFFFFF`） | 一个 `shared::cluster` 地址，即第 24 位被清零的本地 `shared::cta` 地址 | 发射该指令的线程，以算术方式 | TMA 加载的屏障操作数（`full_masked_base`，:770）；累加器释放（W11，:370-379） |
| Multicast 掩码（16 位） | `tcgen05.commit ... multicast::cluster [addr], mask`；`cp.async.bulk.tensor ... multicast::cluster ..., mask` | 硬件投递到每个被掩码选中的 CTA 中的同一偏移 | mainloop 释放 `0xF`；累加器 commit `0x3 << 2y`；A 加载 `0x5 << x` |
| `mapa` | `mapa.shared::cluster.u32 rem, addr, cta_id` | 硬件把一个本地地址映射到 CTA `cta_id` 的副本 | CLC 释放与 throttle 释放（`cta_id = 0`）；CLC 装填（`cta_id = lane`）；dealloc 握手（`cta_id = peer`） |

为什么是三种而不是一种？peer 位只花一条 `ULOP3`，但只能指名一个 pair 中的偶数 CTA。`mapa` 可以指名任何 CTA，但每个目标要花一条指令，并且产生一次到达。multicast 掩码用一条指令就能到达多个 CTA，但只能到达同一偏移。这个 kernel 在每个位置都保留 CUTLASS 的选择（explicit.cu:17），不论它是否最便宜：rank 0 通过 `mapa` 释放它自己的 throttle 屏障（:837；第 7 章，第 7.6.4 节）。nvcc 按位置折叠 peer 掩码：在 epilogue 的累加器释放处（`0x44c0`）是 `0xFEFFFFF8`（explicit.cu:187 的注释），在 producer 外提的、1024 字节对齐的 `smem_base` 上是 `0xFEFFFFF0`（`0x0c00`，笔记 G.3；baseline 每 k-tile 施加一次 `0xfefffff8`，笔记 D.2）。两者在这个 kernel 的屏障地址上都具有 `0xFEFFFFFF` 的效果，`0xFEFFFFF0` 这种形式之所以如此，是因为它施加在尚未加上 `8 s` 的外提基址上（第 2 章，第 2.9.3 节）。

---

## 4.7 per-tile 时间线

### 4.7.1 甘特图

各段时长只是大致按比例。“实测”数字来自笔记（D.3、G.2）；推导得出的数字都有标注。固定运行中的一个 tile 每 cluster 约需 0.889 ms / 32 tiles = 27.8 us（由实测的 `Avg runtime` 0.889354 ms 以及实测的每 cluster 32/31 个 tile 推导得出），即每 k-tile 约 217 ns；CLC 从查询到消费的延迟中位数为 1408 ns，scheduler 的 throttle 等待为每 tile 28.8-37.7 us（二者均在 CUTLASS kernel 的 trace 构建中实测，笔记 D.3，其中一个不带探针的 tile 耗时 31.90 us）；explicit kernel 自己的 trace 构建实测到中位数 1024 ns 以及每 tile 35.46 us，后者被它额外的等待诊断所抬高，而笔记 G.2 把二者都标为与开关关闭的二进制文件无关。本章凡为 CLC 往返写 "~1.4 us" 之处，用的都是 CUTLASS trace 的数字。

```
 time ->        tile t (128 k-tiles, ~27.8 us)                                                 | tile t+1
 ============================================================================================|==========================
 W2 (loads)     [thr]|L0 L1 .. L7|L8 ..... Lq issued when empty[q%8] flips (<= 8 ahead) ..... L127|[CLC t]|[thr]|L0 L1 .. L7|L8 ...
                 ^rank 0 only           ^ L8 waits for the commit of k-tile 0                     ^ consume AFTER issuing tile t's loads
                                                                                             tile t+1's first 8 loads start ~8 k-tiles
                                                                                             before the tensor core finishes tile t
 ---------------------------------------------------------------------------------------------------------------------------
 W0 (MMA issue) [CLC t]|[acc_empty t%4]|F0 MMMM C|F1 MMMM C| ... |F127 MMMM C|[acc_full commit]|[CLC t+1]|[acc_empty]|F0 MMMM C| ...
                 ^ consume BEFORE computing tile t          ^ Fq = wait full[q%8]; MMMM = 4 x tcgen05.mma; C = commit mask 0xF
 tensor core          ......|MMA q=0 |MMA q=1 | ... (executes behind the issue; the commit of q fires when q's 4 MMAs are done)
 ---------------------------------------------------------------------------------------------------------------------------
 W1 (sched,     [thr_full t ... wait until rank 0's W2 starts tile t]|[rel]|[clc_empty t%2]|[arm x4]|[QUERY t]|[CLC t consume]|[thr_full t+1 .....
  rank 0 only)                                                                              ^ ~1.4 us later the response lands in all 4 CTAs
 ---------------------------------------------------------------------------------------------------------------------------
 W4-7 (epilogue)[CLC t]|[acc_full t%4 ...... waits the whole mainloop of tile t ......................]|e0 e1 e2 e3 e4 e5 e6 e7|[CLC t+1]|[acc_full ...
                 ^ consume BEFORE storing tile t                                                       ^ epilogue of tile t runs during the
                                                                                                         mainloop of tile t+1 (TMEM stage ring)
                                                                                           e7: tcgen05.ld, wait::ld, arrive acc_empty, then math+stores
 TMA store unit                                                                                        |g0|g1|g2|g3|g4|g5|g6|g7|   (bulk groups, read 1 behind)
```

如何阅读它：

1. **加载跑在前面。** Warp 2 一旦 `empty[q % 8]` 翻转就发射 k-tile q，也就是一旦两个 leader 在 k-tile q - 8 上的 MMA 都已完成（它们的 commit 在那时到达，第 4.2.2 节）。在稳态下，加载领先 MMA 发射最多 8 个 stage（约 1.7 us，推导）。对 smem 环来说 tile 边界没有任何特殊之处 (4.3.1)：tile t + 1 的 k-tile 0 的加载在 tensor core 仍在处理 tile t 的 k-tile 121..127 时就已发射。
2. **MMA warp 在稳态下从不等待 epilogue。** 它对 epilogue 的唯一依赖是 `acc_empty[t % 4]`，由 tile t - 4 的 epilogue 释放。有了 4 个 TMEM stage，epilogue 最多可以落后三个 tile，MMA warp 才会停顿；它在稳态下实际落后多少没有实测（trace 记录的是 CLC 的消费以及 TMA/MMA 的发射，而不是 epilogue 的完成；第 10.8 节），所以只知道结构性的界：这三个 tile 的余量，以及 CLC 环施加的约两个 tile 的更紧的界（第 4.9 节）。
3. **tile t 的 epilogue 与 tile t + 1 的 mainloop 重叠。** epilogue 在 tile t 几乎整个 mainloop 期间都在等待 `acc_full[t % 4]`，然后在 MMA warp 已经在把 tile t + 1 发射到 TMEM stage `(t + 1) % 4` 中时，运行它的八个 subtile。这就是 TMEM 环换来的东西：在 epilogue 期间 tensor core 不会空闲。（epilogue 的时长在笔记中没有实测；我的估计是几微秒，远小于一个 mainloop。）
4. **CLC 的位置。** scheduler 一旦 rank 0 的加载者开始了 tile t（throttle）就发出查询 t，这最多比 tensor core 完成 tile t - 1 早 8 个 k-tile，并在约 1.4 us 之后消费响应 t。warp 0 与 warp 4-7 在 tile t 的*开始*处消费响应 t；warp 2 在发射完 tile t 的 256 次加载*之后*才消费它。在 CUTLASS kernel 的 trace 中实测（笔记第 7.6 节，cluster (0,0)；笔记 D.3）：warp 0 的第一次消费在 rank 0 上比它的第一条 `tcgen05.mma` 早 1248 ns，在 rank 2 上早 576 ns（第一个 stage 的 TMA box 必须先落地；我的解读），而 rank 0 的 warp 0 消费每个响应的时间比 scheduler 晚 96 ns（中位数）。我把这 96 ns 解读为 "the MMA warp is usually already at its CLC wait when the response lands"。那是*发射*流中的一个气泡，不一定是 tensor core 中的气泡，后者异步地执行前几个 k-tile 中已排队的 MMA（我的分析）。

### 4.7.2 每个角色在哪里消费，以及为什么取回的 tile 直到最后才成为当前的

| 角色 | 在循环体中的消费位置 | 行号 | 为什么在那里（源自源码 + 我的分析） |
|---|---|---|---|
| warp 0 (MMA) | 循环体的第一条语句，在 `acc_empty` 等待与 k 循环之前 | 913 | 循环体用当前的 `(tm, tn)` 计算*当前的* tile；先消费使退出决定在结尾处可用，让该 warp 在整个循环体期间持有下一个 tile 的坐标，并尽早释放 CLC 槽位（800 中的 32）。到那时，跑在前面的 producer 已经在 tile t 中待了一阵子，而查询 t 在途；4.7.1 第 4 项的 96 ns 这个数字表明，当响应落地时 warp 0 通常已经在等待了（我的分析）。 |
| warp 4-7 (epilogue) | 第一条语句，在 `acc_full` peek 之前 | 1028 | 同上。此外，epilogue 的 128 次到达是 800 中最大的一份；如果它们发生在 tile t 的存储*之后*，CLC 槽位就会晚大约一个 mainloop 才被释放，而 scheduler 的查询 t + 2 将停顿在 epilogue 之后（只有两个槽位）。 |
| warp 2 (producer) | 在两次 `load_ktiles` 调用和一次 `__syncwarp()` 之后 | 793 | producer 最早需要下一组坐标：它对 tile t + 1 的加载在 MMA 完成 tile t 之前就开始了。在 rank 0 上它还是 throttle producer（4.4.5），所以查询 t 只在它已经*开始了* tile t 之后才发出；若在加载之前消费，就会让每个 tile 在其第一次 TMA 之前都等待一次 CLC 往返（~1.4 us）。`__syncwarp()`（CUTLASS kernel 在工作树中为 :691-692，在 cb424739 中为 :663-664："Sync warp to prevent non-participating threads entering next wave early"）在被选中的 lane 发射 TMA 之后让该 warp 重新收敛。 |
| rank 0 warp 1 (scheduler) | 紧接在发出查询 t 之后 | 853 | 它没有别的事可做，而且它必须得知有效性才能停下：在观察到一次失败的查询之后再发出查询是未定义行为（CUDA Programming Guide，CLC 页面："Submitting another cancellation request after observing a previously failed request is undefined behavior"）。 |

每个角色都把取回的坐标暂存在 `tm_next, tn_next` 中，并且只在循环体的结尾才采用它们（explicit.cu:794, 854, 947, 1073：`tm = tm_next; tn = tn_next;`）。循环条件是*取回的*响应的有效性（`while (valid)`）。由此产生的后果，其中笔记 D.3/G.1 测量了查询计数、scheduler 与 warp 0 的消费计数以及陈旧的 `x0, y0, z0`，而笔记第 7.6 节只对 warp 0 在其第一次 MMA 之前的第一次消费计了时；其余的顺序关系均由源码推出。（在这里以及第 4.8 节中，T 是 cluster 的 tile 数，即笔记的尾段公式和第 8 章 `n_tiles` 所用的符号；不是第 9-10 章的 TMEM 基址 T。）一个处理 T 个 tile 的 cluster 恰好发出 T 次查询：T - 1 次的答复是“已取消”，最后一次是“未取消”。每个参与的 warp 恰好消费 T 个响应。warp 0 在消费响应 0 *之后*才计算初始 tile（来自 `blockIdx`，explicit.cu:1175-1176），而 MMA 与 epilogue 各 warp 在消费未取消的响应*之后*才计算并存储最后一个有效 tile。没有任何角色读取未取消响应的 `x0, y0, z0`（`if (valid)` 守护着映射，explicit.cu:691-694），因为带谓词的 `get_first_ctaid` 让它们保持未写入（笔记 D.3）。

---

## 4.8 退出协议

在 kernel 退出时没有 cluster 范围的同步（笔记，第 7.10 节）。一个 CTA 的共享内存在它的最后一个 warp 退出时就消失了，而任何仍在途、进入该内存的远程到达都违反了 PTX ISA 的规则，即执行中的 CTA 必须确保 "the peer CTA hasn't exited before completing the shared memory operation"（PTX ISA 9.4，2.2.2）。（CUDA Programming Guide 的分布式共享内存规则要求目标 CTA 在其共享内存被访问时仍在运行，此为转述；笔记 B.6 把 kernel 层面的规则表述为 "a CTA must not exit while peers may still arrive on its barriers"。）因此，kernel 让每一个*拥有其他 CTA 会到达的屏障*的角色在离开之前等待最后的那些此类到达。哪些屏障接收的远程到达是任何普通等待都尚未覆盖的？

| 屏障 | 远程到达来自 | 是否已被普通等待覆盖？ | 尾段 |
|---|---|---|---|
| `empty[s]`@每个 CTA | 两个 leader 的 `tcgen05.commit`（掩码 `0xF`） | 否：在其最后一次加载之后，producer 再也不等待 `empty` | **尾段 1**，warp 2，8 次等待 |
| `clc_empty[i]`@CTA 0 | 每个响应 800 次到达，来自全部四个 CTA | 否：在其最后一次查询之后，scheduler 再也不等待 `clc_empty` | **尾段 2**，rank 0 warp 1，2 次等待 |
| `acc_empty[st]`@leader | peer 的 128 个 epilogue 线程（+ 自己的 128 个） | 否：在其最后一个 tile 之后，leader 的 warp 0 再也不等待 `acc_empty` | **尾段 3**，leader warp 0，4 次等待 |
| `tmem_dealloc`@两者 | 另一个 MMA warp 的 32 次到达 | （它*就是*那次握手） | **尾段 4** |
| `full[s]`@leader、`acc_full[st]`@两者、`clc_full[i]`@全部 | 字节 / commit / 装填 | 是：在该角色能够结束之前已由本地等待消费掉 | 无需 |
| `thr_*`@CTA 0 | 无（CTA 本地） | 不适用 | 无需 |

rank 1-3 的 warp 1 以及每个 CTA 的 warp 3 在会合之后立即退出（explicit.cu:1190-1202：它们不匹配任何分支）；没有任何东西以它们为目标。rank 1 和 rank 3 的 `acc_empty`、除 CTA 0 之外每个 CTA 的 `clc_empty`，以及各 peer 的 `full[]` 从未被到达过（4.2.3），所以它们不需要尾段。

### 4.8.1 尾段 1：producer 的 8 次 empty 等待（explicit.cu:797-806，引文见第 8.6 节）

这就是 `PipelineTmaAsync::producer_tail`（sm90_pipeline.hpp:448-454：`for (count < Stages) { empty_barrier_ptr_[state.index()].wait(state.phase()); ++state; }`），经由 `load_tail`（sm100_mma_warpspecialized.hpp:632-641，调用在 :640 处）到达。尾段*延续着 producer 的状态*：在 T 个 tile 之后，producer 已经走了 N = 128 T 步，所以尾段步骤 i 在 `empty[(128 T + i) % 8] = empty[i]` 上以 parity `1 ^ (((128 T + i) / 8) & 1) = 1 ^ ((16 T) & 1) = 1` 等待，对每个 T 都是如此，也就是直到两个 leader 对 stage i 最后一次使用（k-tile 127 - (7 - i)）的 commit 都已落地为止（第 8 章，第 8.6.8 节证明了这一点，并展示了为什么硬编码的 parity 对其他 k-tile 数会是错的）。实测（笔记 G.1）：在全部 33 个 cluster 中，每个 mainloop 尾段步骤等待的都是 parity 1（`k_tail_mainloop_parity` PASS）。

### 4.8.2 尾段 2：CLC producer 的 2 次等待（explicit.cu:857-867，引文见第 7.6 节）

这就是 `PipelineCLCFetchAsync::producer_tail`（sm100_pipeline.hpp:1041-1050），出于 4.2.4 中给出的 SASS 方面的原因（笔记 D.2 的 `mbarrier.test_wait.parity` 行），它保持着先探针后阻塞的形状。在已发出 T 次查询的情况下，步骤 i 以 parity `1 ^ (((T + i) >> 1) & 1)` 等待 `clc_empty[(T + i) % 2]`：即响应 T - 2（槽位 `T % 2`）的 800 次释放，然后是响应 T - 1 的。

### 4.8.3 尾段 3 与尾段 4：leader 的 4 次累加器等待与 dealloc 握手（explicit.cu:952-971，引文见第 9.6 节）

尾段 3 是 `PipelineAsync::producer_tail`（sm90_pipeline.hpp:1129-1134，`producer_acquire(state); ++state;` 执行 S 次），仅由 leader 调用（kernel :787-790："Leader MMA waits for leader + peer epilogues to release accumulator stage"）。在有 T 个 tile 的情况下，步骤 i 以 parity `1 ^ (((T + i) >> 2) & 1)` 等待 `acc_empty[(T + i) % 4]`：即每个 TMEM stage 最后一次使用的 256 次释放，来自两个 CTA 的 epilogue。

尾段 4 是 kernel :792-796 处的 CUTLASS 序列（"Leader does wait + arrive, follower does arrive + wait"；`arrive(mma_peer_cta_rank, not is_mma_leader_cta); wait(dealloc_barrier_phase); arrive(mma_peer_cta_rank, is_mma_leader_cta)`）。peer 的 32 个 lane 到达 leader 的 `tmem_dealloc`（计数 32，一个 phase）；两者都在自己的屏障上等待 parity 0；然后 leader 的 32 个 lane 到达 peer 的屏障。为什么？`tcgen05.dealloc.cta_group::2` 必须在两个 CTA 的*所有* TMEM 读取之后（笔记 B.6），而只有 leader 的 warp 0 在尾段 3 之后才知道两个 epilogue 都已完成：peer 的 warp 0 从不等待 `acc_empty`（explicit.cu:915-918）。因此这次握手 (a) 阻止 peer 的 warp 0 在 leader 完成尾段 3 之前解除分配，并且 (b) 阻止 leader 在 peer 的 warp 0 到达握手点之前解除分配，使得 pair 范围的解除分配以及 peer 对 leader 屏障的远程 arrive 都能遇到一个仍然活着的 CTA（源自源码；顺序安排的理由是我的分析；第 9 章，第 9.6.4 节证明了这两个论断）。实测（笔记 G.1）：1518 条 `K_TAIL` 记录 = 每个 cluster 46 条 = rank 0：8 + 2 + 4 + 1，rank 1：8 + 1，rank 2：8 + 4 + 1，rank 3：8 + 1。

### 4.8.4 针对 T = 32、31 与 19 算出的 parity

这些是在被追踪的运行中观察到的三个 tile 数（笔记 D.3：13 个 cluster 为 32 个 tile，19 个为 31，被探针拖慢的 cluster (0,0) 为 19）。

| 尾段 | T = 32 | T = 31 | T = 19 |
|---|---|---|---|
| 1，mainloop：`empty[(128T+i)%8]`，i = 0..7 | 索引 0..7，全部 parity 1 | 索引 0..7，parity 1 | 索引 0..7，parity 1 |
| 2，CLC：`clc_empty[(T+i)%2]`，i = 0, 1 | (0, parity 1), (1, parity 1) | (1, parity 0), (0, parity 1) | (1, parity 0), (0, parity 1) |
| 3，累加器：`acc_empty[(T+i)%4]`，i = 0..3 | (0,1), (1,1), (2,1), (3,1) | (3,0), (0,1), (1,1), (2,1) | (3,1), (0,0), (1,0), (2,0) |
| 4，dealloc：自己的屏障 | parity 0 | parity 0 | parity 0 |

验算一例：T = 19，尾段 3，i = 1：`n = 20`，索引 `20 % 4 = 0`，`20 / 4 = 5`，parity `1 ^ (5 & 1) = 0`。TMEM stage 0 被 tile 0、4、8、12、16 使用过：五次使用，五次释放。所以一旦 tile 16 的全部 256 次到达都已到齐，`acc_empty[0]` 就已完成了 phase 0..4，此时当前 phase 为 5（奇数）。等待 parity 0 要求的是 phase 4，即前一个：它恰好在 tile 16 的释放完成时返回。这张表的教训（笔记第 7.2 节）：*只有当 `N % S == 0` 时，全部 S 次尾段等待才使用同一个 parity*。一个为整个尾段硬编码单一 parity 的 kernel，每当 acquire 计数 `N` 不是 stage 数的倍数（对累加器环和 CLC 环而言，即 tile 数）时，就会挂起或提前退出。dealloc 等待总是 parity 0，因为那个屏障在 kernel 的生命期内恰好只完成一个 phase。

### 4.8.5 最后一个 TMA 存储组

对于 beta = 0，epilogue 没有尾段。`store_tail`（sm100_epilogue_tma_warpspecialized.hpp:1271 ff.）受 `is_producer_load_needed()` 门控，而后者为假，所以没有任何 `cp.async.bulk.wait_group.read 0` 被执行（explicit.cu:1076 处的注释；笔记 D.2：`DEPBAR.LE SB0, 0x0` 存在于 `0xb760` 处，但 epilogue 各 warp 在它之前就走了 `EXIT`）。因此退出时的不变量就是最后一个 subtile 的 `wait_group.read 1` 所留下的那个：**当 epilogue 各 warp 退出时，至多一个 bulk group（最后一个 subtile 的四次存储）可能仍在读取其 smem stage。** 实测（笔记 D.1 D15）：固定命令的每次运行都在此条件下通过。笔记对任何重写的规则：保持或收紧它，绝不放松它（笔记第 7.7 节）。退出前的一次 `wait_group.read 0`（或 `wait_group 0`）会收紧它；explicit kernel 没有添加任何一个，因为笔记的验收检查把 `wait_group.read 0 = 0` 和 `DEPBAR.LE SB0, 0x0 = 0` 算作对这个 kernel 的硬性要求（笔记 E.6.7、E.7.4；第 10 章，第 10.11 节）。

---

## 4.9 冒险、是什么防止了它们，以及没有它们会发生什么

“挂起”指的是一次其 phase 永不完成的等待；“损坏”指的是 kernel 正常完成但 D 错误。失效列是我对每个缺失元素的分析；这些元素本身源自源码。

| # | 冒险 | 协议元素 | 若缺失 |
|---|---|---|---|
| 1 | smem **WAR**：TMA 在 tensor core 仍在读取第 n - 8 次使用时写入 stage s | `empty[s]` 计数 2；两个 leader 在第四条 MMA 之后的 `tcgen05.commit` 掩码 `0xF`；producer 以 parity `1 ^ ((n/8)&1)` 等待 `empty[s]` | 损坏（UMMA 读到两个 k-tile 的混合）。用 `mbarrier.arrive` 代替 `tcgen05.commit`：损坏，因为 arrive 在发射时就触发，而不是在完成时。计数 1 而非 2：损坏出现在其 multicast 一半在*另一个* leader 读到它之前就被覆盖的那个 CTA 中。 |
| 2 | smem **RAW**：UMMA 在全部六个 box 落地之前读取 stage s | `full[s]` 计数 1 + 49152 tx；leader 的 `arrive.expect_tx`；每个 box 都指名 leader 的屏障（peer 位） | tx 太小（例如 24576，忘了 x2）：phase 在一半字节之后就完成，损坏。tx 太大：挂起。box 被计入错误的屏障（peer 位未清除）：leader 的 phase 永不完成，挂起。 |
| 3 | TMEM **RAW**：epilogue 在 512 条 MMA 完成之前加载 stage st | `acc_full[st]` 计数 1，由 leader 的 `tcgen05.commit` 掩码 `0x3 << 2y` 完成 | 损坏（部分和）。两个 pair 都用掩码 `0x3`：pair {2,3} 的 epilogue 永远看不到它的 commit，挂起；pair {0,1} 在一个计数为 1 的屏障上每个 phase 收到两次到达，未定义。 |
| 4 | TMEM **WAR**：tile t + 4 的 MMA 在 tile t 的 `tcgen05.ld` 仍在途时覆盖 stage st | `acc_empty[st]` 计数 256；每个线程的 arrive 之前先 `tcgen05.wait::ld`；leader 在该 tile 的第一条 MMA 之前等待 | 没有 `wait::ld`：损坏（arrive 先于异步加载的完成）。计数 128：当只有一个 CTA 的 epilogue 释放了时 leader 就继续前进，在另一个 CTA 的行中出现损坏。 |
| 5 | **通用 proxy 写入 vs TMA 存储**：TMA 单元在 128 个线程的 `st.shared` 对异步 proxy 可见之前就读取 C/D stage | 每个线程执行 `fence.proxy.async.shared::cta`，然后 `bar.sync 1, 128`，然后是那些存储 | D 中的陈旧 smem：损坏。有 fence 而无 `bar.sync`：warp 4 可能在其他三个 warp 各自执行 fence 之前就发射，损坏。 |
| 6 | **C/D stage WAR**：线程写入 stage `(e+1)%4`，而一次 TMA 存储仍在读取它 | warp 4 执行 `cp.async.bulk.commit_group` + `wait_group.read 1`，然后 `bar.sync 1, 128` | D 中损坏（存储读到一个被覆盖了一半的 stage）。正是第二个 `bar.sync` 把 warp 4 的等待传播到了其他 warp。 |
| 7 | **CLC 响应的可见性与 WAR**：一个 consumer 在那 16 个字节落地之前读取槽位 i，或者下一次进入槽位 i 的 `try_cancel` 在某个 consumer 读取它之前就覆盖了它 | `clc_full[i]` 1 + 16 tx（arrive.expect_tx 来自 rank 0 的 lane 0..3，字节来自 CLC 单元）；解码之后的 `fence.proxy.async`；重新发出之前 `clc_empty[i]` 计数 800 | 过早读取：一个陈旧或撕裂的响应，一个 tile 被计算两次或从未被计算。计数小于 800：producer 在某个慢 warp 仍在读取时就重新发出，效果相同。`fence.proxy.async` 关闭了通用读/异步写的窗口。 |
| 8 | **屏障 init 的可见性**：一次远程 arrive 或 `complete_tx` 命中了在该 CTA 中尚未初始化的屏障字 | 四个流水线 init 组每组之后各一条 `fence.mbarrier_init.release.cluster`（单独的 `tmem_dealloc` init 不带 fence，照搬 CUTLASS；4.2.5，第 11.5.4 节）；任何远程操作之前的 `barrier.cluster.arrive.relaxed` + `barrier.cluster.wait` | 到达丢失或落在垃圾数据上：挂起（一个永不完成的 phase），或者，如果 init 在到达之后才落地，则计数多出一个：挂起。 |
| 9 | **TMEM 基址的可见性**：epilogue 在 `tcgen05.alloc` 写入 `tmem_base_ptr` 之前就读取它 | warp 0 中先 `__syncwarp()` 然后 `bar.arrive 6, 160`；warp 4-7 中 `bar.sync 6, 160` | 错误的 TMEM 地址：损坏或非法地址错误。 |
| 10 | **未取消的响应被解码为一个 tile** | `(tm, tn)` 映射外围的 `if (valid)`；循环在 `!valid` 时退出 | 前一个响应的陈旧 `x0, y0` 被复用：一个 tile 被计算两次（并且该 CTA 越过网格末尾继续运行）。在一次失败的查询之后再发出另一次查询是未定义行为。 |
| 11 | **tile 的过度预订** | throttle 环（只有在 rank 0 的加载者开始 tile t 之后才发出查询 t） | 不是正确性冒险：kernel 保持正确，但一个 cluster 可能持有多达两个尚未开始的 tile，拉长网格的尾段。 |
| 12 | **在远程到达尚未完成时 CTA 退出** | 尾段 1-4（4.8） | 未定义行为：一次进入已退出 CTA 的共享内存的 arrive；实际上表现为到达方挂起，或者接下来占据该 SM 的任何 CTA 的屏障损坏（我的分析）。 |
| 13 | **`scale_c` 没有逐 tile 重置**（不是屏障，但属于 per-tile 契约的一部分） | 每个 tile 的第一条 MMA 之前 `scale_c = 0`（explicit.cu:917），之后为 1 | 新 tile 累加到前一个 tile 的 TMEM 内容之上：损坏。 |

**无死锁性（我的分析，源自源码结构）。** 每个环都是有界的并按序使用，而跨环依赖在每个 tile 上构成一个有向无环图。`query t` 需要 {rank 0 的加载者已开始 tile t；全部 800 个 consumer 已释放响应 t - 2}。`response t` -> {warp 0 与 warp 4-7 开始 tile t；warp 2 开始 tile t + 1}。tile t 的 k-tile q 的 `full[s]` 需要 {四个 producer 已发射 q，而这需要 `empty[s]` = 两个 leader 对 q - 8 的 commit}。`acc_full[t % 4]` 需要 leader 在 tile t 的 128 个 k-tile（512 次 MMA）之后向它发射的那一次 `tcgen05.commit`。tile t 的 `acc_empty[t % 4]` 需要 tile t - 4 的 epilogue。查询 t 的 `clc_empty` 需要 epilogue 已经*消费*了响应 t - 2。epilogue 把这件事作为 tile t - 2 循环体的第一条语句来做（explicit.cu:1028，在 :1030 处的 `acc_full` peek 之前）。因此，这一要求仅仅是 epilogue 已经完成了 tile t - 3 的*存储*，而这需要 `acc_full[(t - 3) % 4]`，亦即 tile t - 3 的 mainloop，它早已完成。每一条依赖都指向严格更早的工作，因此不存在环路。两处耦合限定了各角色之间能相互漂移多远。4 个 stage 的 TMEM 环允许 epilogue 落后 MMA 至多 3 个 tile。2 个槽位的 CLC 环允许它落后至多约 2 个（查询 t 等待 epilogue 释放响应 t - 2，而 MMA 等待响应 t）。因此在本 kernel 中，对 MMA-epilogue 偏斜更紧的界是 CLC 环，而非 TMEM 环。最后，四个 CTA 保持一致，因为响应是 multicast 的：四者解码出同一个 tile 序列，所以 pair 的 MMA（它读取两个 CTA 的 smem）与 A 的 multicast（它写入两个 cluster 行）在每个 CTA 中总是指向同一个 `(tm, tn)`。

---

# 第 5 章 基础：文件、参数块、tensor map 与第 1 节的常量

## 5.1 文件集

### 5.1.1 四个文件，两个翻译单元

```
                 70_blackwell_fp16_gemm.cu  (harness TU)          70_blackwell_fp16_gemm_explicit.cu  (kernel TU)
                 line 1: #include "..._explicit_util.hpp"          line 1: #include "..._explicit_util.hpp"
                 line 88: #include "..._explicit.hpp"              line 80: #include "..._explicit.hpp"
                 CUTLASS headers (CollectiveBuilder, Gemm, ...)    cutlass/cutlass.h, cute/swizzle.hpp (+ cute/container/tuple.hpp via explicit.hpp)
                        |                                                 |
                        |  instantiates make_params<Gemm>,                |  defines __global__ kernel,
                        |  run<Gemm>(gemm) (templates)                    |  defines run(ExplicitGemmParams const&)
                        v                                                 v
            +--------------------------------------------------------------------------+
            |  70_blackwell_fp16_gemm_explicit.hpp  : struct ExplicitGemmParams (the ABI) |
            |  70_blackwell_fp16_gemm_explicit_util.hpp : toggle (line 1), constants       |
            +--------------------------------------------------------------------------+
```

kernel TU 的 include 集合（`explicit.cu:72-80`）为 `<cuda.h>`、`<cuda_runtime.h>`、`<cstdint>`、`<cstdio>`、`cutlass/cutlass.h`、`cute/swizzle.hpp` 与 `explicit.hpp`；经由 `explicit.hpp`（`:33`、`:39`）它还得到了供 `CUtensorMap` 使用的 `<cuda.h>` 以及供 `cute::get` 使用的 `cute/container/tuple.hpp`。`cute/swizzle.hpp` 的传递闭包是 39 个头文件（笔记，第 E.2.1 节），其中没有一个是 CUTLASS 的 kernel 头文件或 collective 头文件：开关关闭时的 kernel TU 从不会看到 `cutlass/gemm/*`（开关开启时的 trace 头文件 `explicit.cu:83` 会拉入 CUTLASS 的头文件），这就是为什么枚举量 `AlongN` 与 `kGemm` 在 `util.hpp` 中以字面量形式出现（5.2.1）。

## 5.2 逐行解读 `util.hpp`

### 5.2.1 常量 (`util.hpp:29-39`) 以及每个数字从何而来

| 常量 | 值 | 推导 | 谁使用它 |
|---|---|---|---|
| `kFixedM/N/K/L` | 8192, 8192, 8192, 1 | 固定的命令行 `--m=8192 --n=8192 --k=8192`；`Options` 的默认值与之相同（harness 第 207-211 行） | `run()` 的守卫（`explicit.cu`，第 11 节） |
| `kCtaTileM/N` | 128, 128 | `CtaShape_MNK = (128,128,64)` = 256x128 的 MMA tile 沿 M 除以 2-CTA 原子（笔记，第 4 节；第 3 章推导了为什么是 256x128） | `kGridX`/`kGridY`（`util.hpp:32`）以及经由它们的网格断言（`:41-42`） |
| `kTileK` | 64 | `BLK_K = 64` 个 fp16 = 每个 smem 行 128 字节，即 SW128 swizzle 的跨度（`Layout_K_SW128_Atom_Bits`：8 行 x 1024 位，`mma_traits_sm90_gmma.hpp:84`；其带类型的形式 `Layout_K_SW128_Atom<half_t>` = 8 行 x 64 个 fp16，`:104`） | `kKBlocks = 64 / 16 = 4` |
| `kClusterX/Y/Z` | 2, 2, 1 | harness 中的 `ClusterShape_MNK = Shape<_2,_2,_1>`（第 125 行） | 启动属性 |
| `kGridX, kGridY` | 64, 64 | `N / 128`、`M / 128`：网格的 **x 索引 N** 而 **y 索引 M**，因为 CLC scheduler 的 `AlongN` 启发式规则会转置网格（`possibly_transpose_grid`、`sm100_tile_scheduler.hpp:216`；第 7 章） | `run()` |
| `kBlockThreads` | 256 | 8 个 warp = 5 个角色：Sched 32 + MMA 32 + MainloopLoad 32 + EpilogueLoad 32 + Epilogue 128（`sm100_gemm_tma_warpspecialized.hpp:136-147`，经由笔记第 4 节） | `kThreads`（`explicit.cu:104`，在 `:166` 处断言 `== 256`）以及启动的 `block` 维度（`:1283`）；`__launch_bounds__(256, 1)`（`:1095`）把该字面量硬编码了进去 |
| `kSmemBytes` | 230400 | CUTLASS kernel 的 `sizeof(SharedStorage)`（实测 `sizeof_SharedStorage 230400`，笔记 D.1 D2）；第 5.7 节逐字节地重建了它 | `cudaFuncSetAttribute`、启动 |
| `kProblemTilesM/N/L` | 32, 32, 1 | 8192 / 256：scheduler 计数的是 256 x 256 的 **cluster tile**（2 x 2 个 CTA tile），而不是 CTA tile | `run()` 针对 `p.scheduler.problem_tiles_*` 的守卫 |
| `kRasterOrderAlongN` | 1 | `enum class RasterOrder { AlongM, AlongN }`（`tile_scheduler_detail.hpp:38-41`），所以 `AlongN == 1`；kernel TU 不得 include `cutlass/gemm/*`（笔记，E.2.1），因此用字面量 | `run()` 守卫 |
| `kModeGemm` | 0 | `enum class GemmUniversalMode { kGemm, ... }`（`gemm_enumerated_types.h:57-64`） | `run()` 守卫 |
| `kSmemDescConst` | `0x4000404000010000` | 两个 UMMA 描述符的常量部分，即除去 14 位起始地址字段之外的一切，该字段在 A 与 B 之间不同，并随 stage 与 k-block 变化（第 5.8 节） | `explicit.cu:174` 交叉检查 |
| `kInstrDescConst` | `0x10200010` | 指令描述符（第 5.8 节） | `explicit.cu:173` |

为什么它们放在这里而不是放在 `explicit.cu` 中：它们不需要任何 CUDA 类型，而一个被两个 TU 都 include 的头文件会把一处笔误变成*两个* TU 中的构建错误。否则 harness 侧的 `make_params` 与 kernel 侧的守卫会悄无声息地不一致。

### 5.2.2 各个 `static_assert` (`util.hpp:41-46`)

| 断言 | 它保护什么 | 没有它会出什么问题 |
|---|---|---|
| 网格 `== (64,64,1)` | `run()` 硬编码的网格等于这个 shape 下 baseline 的 `get_grid_shape` | 错误的网格会启动过少或过多的 cluster；更小的网格会让 1024 个 cluster tile 中的一些永远不被计算，更大的网格则会发出被 kernel 映射到问题的 32 x 32 个 cluster tile 之外的 cluster id，因为设备端没有 tile 边界检查（我的分析） |
| 网格可被 cluster 整除 | 运行时会拒绝网格在任一维度上不是 cluster 整数倍的 cluster 启动；这是编译期的 `ClusterLauncher::check_cluster_dims`（`cluster_launch.hpp:103-112`） | `run()` 自己的运行时 `check_cluster_dims` 镜像（`explicit.cu:1287-1290`）会设置 `launch_result = kInvalid` 并跳过 `cudaLaunchKernelExC`（`:1298`） |
| cluster 大小 `<= 32` | `MaxClusterSize = 32`（`cluster_launch.hpp:82`） | 同上 |
| `K % 64 == 0`、`K / 64 == 128` | 文档所述的循环形状：`kPrologueKTiles = kStages = 8`、producer 的两次 `load_ktiles` 调用的 `8 + 120` 划分（`explicit.cu:788-791`）以及 MMA warp 的 k-tile 计数器，全都假定 128 个完整的 64 宽 k-tile。设备端在运行时仍然计算 `k_tiles = (params.k + 63) >> 6`（`explicit.cu:1112`），而主机端守卫拒绝任何 `k != 8192`（`:1268`）；该断言保护的是文档所述的形状，而不是 kernel 对其他 64 的倍数的正确性 | 若 `K % 64 != 0`，最后一个 k-tile 将是一个部分 box：TMA 会把越界元素零填充（数值上正确），但 `k_tiles` 与循环划分会与文档所述不同；若 `K / 64 != 128`，每条注释中的 8 + 120 这些数字都会是错的 |
| 32 x 32 个 cluster tile | `tiles_*` 守卫值 | 与 scheduler 的 `problem_tiles_*` 不匹配 |
| `kSmemBytes <= 232448` | 232448 B = 227 KiB 是 B200 上可选启用的动态共享内存最大值（实测，笔记第 0 节） | `MaxDynamicSharedMemorySize` 属性调用会失败 |

## 5.3 逐行解读 `explicit.hpp`

### 5.3.1 以字节看参数块

在 CUDA 13.3 的头文件下 `alignof(CUtensorMap) = 128`（实测：`sizeof(ExplicitGemmParams) = 640` 且 `CONSTANT[0] = 0x380 + 640 = 1536`，笔记 G.3）。于是 layout 为：

```
offset  size  field            device use                                   host use
------  ----  ---------------  -------------------------------------------  -----------------------------
   0    128   tma_a            prefetch.tensormap; cp.async.bulk.tensor A    copied from p.mainloop.tma_load_a
 128    128   tma_b            prefetch.tensormap; cp.async.bulk.tensor B    copied from p.mainloop.tma_load_b
 256    128   tma_c            prefetch.tensormap only (beta = 0)            copied from p.epilogue.tma_load_c
 384    128   tma_d            prefetch.tensormap; bulk.tensor stores of D   copied from p.epilogue.tma_store_d
 512      4   m                -                                             guard  == 8192
 516      4   n                -                                             guard  == 8192
 520      4   k                ld.param.b32 [+520]  -> k_tiles = (k+63)>>6   guard  == 8192
 524      4   l                -                                             guard  == 1
 528      4   alpha            ld.param.v2.b32 [+528] (FMUL operand)         -
 532      4   beta             (same v2 load; FFMA operand)                   guard  == 0.0f
 536      4   mode             -                                             guard  == 0 (kGemm)
 540      4   raster_order     -                                             guard  == 1 (AlongN)
 544      4   swizzle_divisor  -                                             guard  == 0
 548      4   tiles_m          -                                             guard  == 32
 552      4   tiles_n          -                                             guard  == 32
 556      4   tiles_l          -                                             guard  == 1
 560      8   alpha_ptr        -                                             guard  == nullptr
 568      8   beta_ptr         -                                             guard  == nullptr
 576     64   tail padding     sizeof rounds up to a multiple of alignof(CUtensorMap) = 128  -> 640
```

这些数字是如何得出的。四个 128 字节的 map 共 512 字节。十二个 4 字节的字段共 48 字节（512..560）。`alpha_ptr` 需要 8 字节对齐，而 560 是 8 的倍数，所以没有空隙；两个指针止于 576。`sizeof` 必须是该结构体对齐值的倍数，而该对齐值就是其对齐要求最高的成员 `CUtensorMap` 的对齐值：在 128 字节对齐下 576 向上取整到 640，在 64 字节对齐下则保持 576。`explicit.hpp:74-75` 处的 `static_assert` 两种大小都接受。`:68-73` 处的偏移断言钉住了设备端读取的一切，即位于 0/128/256/384 的四个 128 字节 map 以及 12 字节的标量（位于 520 的 `k`，位于 528/532 的 `alpha` 与 `beta`），且与对齐值无关。这正是让该头文件能在 CUDA 12.x 与 13.x 之间移植的原因：驱动 API 的 `CUtensorMap` 页面在 v12.6.0 与 v12.9.1 中写着 "Requires compiler support for aligning to 64 bytes"，在 v13.0.0 与 v13.4 中写着 "Requires compiler support for aligning to 128 bytes"（CUDA Driver API，`CUtensorMap` 结构体）。`<= 4096` 断言（`:76`）防止把参数块增长到超过经典的 kernel 参数上限；较新的工具包允许更大的参数块（在 Volta 及之后的架构上，配合 12.1 工具包与 r530 驱动为 32,764 字节，在更老的架构上为 4 KB：CUDA Programming Guide 13.0，"Function Parameters"；第 11 章，第 11.8.4 节），所以这个数字是一个保守的界限，而不是此处适用的上限。

设备端的实测使用情况（笔记 G.3，explicit kernel 的 PTX 清单）：恰好两条 `ld.param` 指令，即用于 `k` 的 `ld.param.b32 [+520]` 与用于 `alpha, beta` 的 `ld.param.v2.b32 [+528]`，外加四条用于 map 地址的 `cvta.param`。`m`、`n`、`l` 以及从偏移 536 起的一切在设备端从未被读取。在 SASS 中参数块起始于 `c[0x0][0x380]`，所以 `alpha, beta` 是 `LDCU.64 UR26, c[0x0][0x590]`（= 0x380 + 528），而 D map 的通用地址是位于 `c[0x0][0x348]` 的参数基址加上 `0x180`（= 384），实测见笔记 G.3（5.3.2 讨论 map 地址）。

该清单对字段列表的三个推论：

- **shape 从未到达任何设备端指令。** 那唯一一条 `ld.param.b32 [+520]` 馈入 `int const k_tiles = (params.k + 63) >> 6;`（`explicit.cu:1112`）= 128。`m`、`n`、`l` 存在的唯一目的是让主机端守卫能在启动前强制 8192/8192/1（`explicit.cu:1268`，笔记 E.5.1）。tile 坐标来自 CLC 响应与 `blockIdx`，从不来自 `m`/`n`（第 7 章）。
- **为什么每个仅主机端使用的字段是现在这个类型。** `tiles_m/n/l` 是 `unsigned`，因为它们复制的是 `PersistentTileSchedulerSm100Params` 的 `uint32_t problem_tiles_m_/n_/l_`（`tile_scheduler_params.h:1845-1847`）。`mode` 与 `raster_order` 是 `enum class GemmUniversalMode`（`gemm_enumerated_types.h:57-64`）与 `enum class RasterOrder`（`tile_scheduler_detail.hpp:38-41`）的 `int` 强制转换，因为该头文件不得提及那些 CUTLASS 类型（5.1.1）。`swizzle_divisor` 是一个 `FastDivmod` 的 `int32_t divisor`（`fast_math.h:327`）。`alpha_ptr`/`beta_ptr` 是 `float const*`，因为 `ElementScalar = float`，并且 `Sm90ScalarBroadcast::Arguments` 持有 `Element const* scalar_ptrs[BroadcastCount]`，其默认 `BroadcastCount = 1`（`sm90_visitor_load_tma_warpspecialized.hpp:1017-1023`）。
- **这两个指针在 baseline 中意味着什么。** 当 `alpha_ptr`/`beta_ptr` 非空时，`Sm90ScalarBroadcast` 从全局内存加载标量而不是使用 `scalars[0]`；baseline 的 PTX 会测试该指针，并为那条路径设有五处 `ld.global.b32`（笔记 D.6 第 13 项）。explicit kernel 只实现了立即数形式，所以对于非空指针 `run()` 返回 `kErrorNotSupported`（`explicit.cu:1271`）。这些字段的存在是为了让这一拒绝明确而非无声。

### 5.3.2 为什么按值传递四个 `CUtensorMap`，以及为什么用 `const __grid_constant__`

一条 TMA 指令（`cp.async.bulk.tensor`）接受的不是一个指向张量的全局指针加一个 shape。它接受的是驱动在主机端编码好的*一个 128 字节 tensor map 的地址*。PTX ISA 限制了该对象可以位于何处："The tensor-map is a 128-byte opaque object either in .const space or .param (kernel function parameter) space or .global space"（PTX ISA，tensor-map 一节）。CUDA Programming Guide 推荐的那一个是 `.param`（"The recommended approach is to pass the tensor map as a const `__grid_constant__` parameter to a kernel"，4.12.2.2 "Using TMA to transfer multi-dimensional arrays"，"Host-to-device transfer"）：启动把该参数块复制到常量 bank 0，设备端从常量 bank 0 读取它（参数块起始于 `c[0x0][0x380]`），而 `prefetch.tensormap` 预热该缓存行，是在 tensor-map 描述符缓存中而不是在 L2 数据缓存中（NVIDIA 内部；ISA 没有指名任何缓存："brings the cache line containing the specified address in the .const or .param memory state space for subsequent use by the cp.async.bulk.tensor instruction"，同一来源）。这恰恰就是 CUTLASS 所做的：`Params` 以 `__grid_constant__` 传递，设备端用参数空间描述符的通用地址进行预取并发射 TMA（笔记，第 5.3 节，最后一条；`device_kernel.h:118`）。

`__grid_constant__` 正是让 `&params.tma_a` 既合法又廉价的东西。CUDA C++ Programming Guide（5.4 "C/C++ Language Extensions"，"`__grid_constant__` Parameters"）写道："Annotating a `__global__` function parameter with `__grid_constant__` prevents the compiler from creating a per-thread copy of the parameter" 以及 "Kernel parameters annotated with `__grid_constant__` must have `const`-qualified non-reference types." 没有这个注解，对一个按值传递的参数取地址会让编译器在线程局部内存中物化出一份副本：一个普通的按值参数是可修改的对象，所以它的地址必须不同于只读的参数 bank。对于一个 640 字节的结构体，这份副本会是每线程一份（`LOCAL` > 0，`STL`/`LDL` 流量），而 TMA 会被交给一个局部内存的通用地址，它不在 ISA 允许 tensor map 所处的三个状态空间中的任何一个，所以这次启动在最好的情况下会很慢，在最坏的情况下则无效（线程局部副本是 Programming Guide 自己的说法，13.0 版，10.2.4 "`__grid_constant__`"："the compiler will ordinarily make a copy of the kernel parameter in thread local memory and use the address of the copy"；TMA 方面的后果是我的分析）。实测的 `LOCAL:0 STACK:0`（笔记 G.3）是不存在任何副本的证据。这是 C++ 的语义，而不是 PTX 的限制；PTX 自 ISA 7.7 起就有了 `cvta.param`，所以一个 kernel 参数确实拥有通用地址。有了这个注解，该参数对整个网格都是只读的，编译器便发出 `cvta.param` 来形成一个指向参数 bank *内部*的通用地址。explicit kernel 的 PTX 恰好有四条 `cvta.param`（笔记 G.3），全部转换同一个参数基址：每个使用点一条（warp 1 的 A/B 预取、warp 3 的 C/D 预取、producer 的 A/B 加载、epilogue 的 D 存储），随后由 `add.s64` 加上 map 偏移 128/256/384。SASS 用参数基址 `c[0x0][0x348]` 加上该 map 在*这个*块中的偏移 0x0/0x80/0x100/0x180 来形成 `UTMACCTL.PF` 与 `UTMALDG`/`UTMASTG` 的 map 操作数；G.3 把 B map 记录为 `UIADD3 ..., 0x80`，把 D map 记录为 `+0x180`。笔记 D.2 中列出的 `+0x80/0x180/0x580/0x680` 操作数属于 baseline 的 2048 字节 `Params`。`const` 是该限定符所要求的。

为什么是四个 map 的*副本*而不是 CUTLASS 的 `Params` 本身：baseline 的 `Params` 有 2048 字节（`.param .align 128 .b8 [2048]`，实测于笔记 D.2）。它之所以这么大，是因为它的每个 TMA 成员都是一个内嵌了描述符外加 CuTe basis 元数据的 `Copy_Atom`，也因为它携带六个 map（A、B、A-fallback、B-fallback、C、D）。explicit kernel 需要四个不透明的二进制块与三个标量，所以 `ExplicitGemmParams` 只保留这些。在未改动的 `gemm.initialize()` 之后从 `gemm.params()` 中复制出这些 128 字节的二进制块，保证了这些字节与 CUTLASS kernel 本会使用的字节完全相同（笔记，第 5.3 节）；回退 map 与主 map 逐字节相同，且在设备端从未被触及（笔记 D.1 D3）。“map 在前”的字段顺序让每个 map 都位于其自身对齐值的倍数处，彼此之间没有填充空洞（`explicit.hpp:48`）。

为什么在 C 从未被读取的情况下仍然携带并预取 `tma_c`：explicit kernel 在每个 CTA 中都保留了 baseline 的四处 `prefetch.tensormap`，与 beta 无关（warp 1：A 与 B；warp 3：C 与 D；`explicit.cu:1113-1120`；笔记 D.2 的四条 `UTMACCTL.PF`），因此 PTX 清单保持 `prefetch.tensormap` 为 4（笔记 G.3），指令流也保持可比（第 11 章，第 11.4.3 节）。一个从零开始的 beta = 0 kernel 可以去掉 C map 及其预取，并把参数块缩减到 512 字节。

为什么 `alpha` 与 `beta` 是运行时字段，尽管这次运行把它们固定为 1 和 0：若用编译期常量，编译器会折叠掉 baseline 所发出的每元素 `FMUL` + `FFMA` 对，而这一对固定了零的符号（推断；笔记第 0 节与 D.7 "Arithmetic"）；从参数块读取它们使 baseline 的算术形式逐位保持不变（第 10 章，第 10.9 节）。

## 5.4 tensor map

### 5.4.1 tensor-map 编码器的输入

固定运行的各项输入（笔记，第 5.3 节；实测为六条 `TRACE_ENCODE` 行，笔记 D.1 D3）：

| Map | dataType | TMA mode 顺序（最内层优先） | globalDim | globalStrides（字节，mode 1..2） | boxDim | elementStrides | swizzle |
|---|---|---|---|---|---|---|---|
| A、A-fallback | `FLOAT16` | (K, M, L) | (8192, 8192, 1) | (16384, 0) | (64, 64, 1) | (1, 1, 1) | `CU_TENSOR_MAP_SWIZZLE_128B` |
| B、B-fallback | `FLOAT16` | (K, N, L) | (8192, 8192, 1) | (16384, 0) | (64, 64, 1) | (1, 1, 1) | `CU_TENSOR_MAP_SWIZZLE_128B` |
| C | `FLOAT32` | (M, N, L) | (8192, 8192, 1) | (32768, 0) | (32, 16, 1) | (1, 1, 1) | `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` |
| D | `FLOAT32` | (M, N, L) | (8192, 8192, 1) | (32768, 0) | (32, 16, 1) | (1, 1, 1) | `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` |

六者共有的：`CU_TENSOR_MAP_INTERLEAVE_NONE`、`CU_TENSOR_MAP_L2_PROMOTION_L2_128B`、`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`（`copy_traits_sm90_tma.hpp:1039-1041`）。每一列为什么是这个值：

- **dataType。** TMA 元素类型：fp16 操作数用 `FLOAT16`，fp32 输出用 `FLOAT32`。swizzle 跨度与 box 限制都以字节表示，因此元素大小会进入下面的每一处。
- **Rank 为 3，以及为什么保留 L。** CUTLASS 只有当一个 gmem mode 的 shape 静态地为 `Int<1>` 或其 stride 静态地为 `Int<0>` 时，才会把它从 tensor map 中丢掉（`construct_tma_gbasis`，`copy_traits_sm90_tma.hpp:806-831`）。harness 的 stride 来自 `make_cute_packed_stride(StrideA{}, {m, k, 1})`（harness 第 320 行）：`L = 1` 与 batch stride 0 都是*运行时*的 `int`/`int64_t` 值，因此 batch mode 存活了下来，范围为 1、stride 为 0。因此在设备端，每个 TMA 坐标都是 3-D 的，kernel 把字面量 `0` 作为第三个坐标传入（`explicit.cu:747-748`、`:754-755`、`:1066`）。驱动接受字面量 0 的字节 stride：源码只断言 `stride % 16 == 0` 与 `< 2^40`（`copy_traits_sm90_tma.hpp:977-984`），而且全部六次编码都返回了 `CUDA_SUCCESS`（笔记 D.1 D13）。
- **Mode 顺序与 stride。** mode 按最内层优先排序：先是按 smem tile 顺序排列的 box mode，然后是追加的 batch mode（`construct_tma_gbasis`，上一条要点）；mode 0 的 stride 是隐含的（1 个元素）。A 是 M x K 行主序，所以 K 在最内层：mode 为 (K, M, L)，字节 stride 为 (2K = 16384, 0)。B 在 harness 的 `LayoutB` 中是 N x K 的“列主序”，也就是说同样是 K-major：(K, N, L)，(16384, 0)。C 与 D 是 M x N 列主序，也就是 M-major：(M, N, L)，(4M = 32768, 0)。stride 必须是 16 字节倍数的规则（驱动 API）平凡地成立。
- **globalDim** 是完整的张量，(8192, 8192, 1)。TMA 在这些边界处裁剪，并对越界元素填零：`OOB_FILL_NONE` 的意思是零填充，而不是“不填充”；另一个选项 `NAN_REQUEST_ZERO_FMA` 填入 NaN。本次运行中没有任何 box 越界，因此该填充模式从未被用到。
- **boxDim** 是一条 TMA 指令所搬运的 smem 映像；第 5.4.2 节推导这两个 shape。
- **elementStrides** (1, 1, 1)：稠密的 box，box 内部没有跨步。
- **Swizzle。** `Swizzle<3,4,3>`（SW128 K-major 原子）映射到 `SWIZZLE_128B` = "Swizzle 16B chunks within 128B span"；`Swizzle<2,5,2>`（SW128_32B M-major fp32 原子）映射到 `SWIZZLE_128B_ATOM_32B` = "Swizzle 32B chunks within 128B span"（`to_CUtensorMapSwizzle`，`copy_sm90_desc.hpp:240-262`；`ATOM_32B` 枚举量仅在 CUDA > 12.6 中存在：12.6 驱动 API 的 `CUtensorMapSwizzle` 只列出 `NONE`/`32B`/`64B`/`128B`，12.8 有它，而 `:257` 相应地以 nvcc > 12.6 为条件守卫这一情形）。TMA 单元在写入（加载）时施加 XOR，并在读取（存储）时撤销它。因此 kernel 从不计算 `Swizzle<3,4,3>`，而计算 `Swizzle<2,5,2>` 只是因为它自己写入 C/D stage（第 5.10 节）。
- **L2 promotion 128 B。** 驱动 API 把 `CUtensorMapL2promotion` 描述为这样一个参数："specifies L2 fetch size which indicates the byte granularity at which L2 requests are filled from DRAM"（CUDA Driver API，Tensor Map Object Management）。`CU_TENSOR_MAP_L2_PROMOTION_L2_128B` 选择 128 字节，因此每个 128 字节的 box 行都作为一整条 L2 行被取回（L2 的行是 128 字节："The L1 and L2 both have 128 byte cache lines"，Nsight Compute Profiling Guide；推导）。CUTLASS 对每一个 tiled map 都硬编码了这一点（`copy_traits_sm90_tma.hpp:1040`）。
- **Interleave NONE。** box 是一个普通的稠密子张量，而不是交错格式之一（`CU_TENSOR_MAP_INTERLEAVE_16B` / `_32B`）。

编码器的各项约束（CUDA Driver API，`cuTensorMapEncodeTiled`），即你在为另一个算子编码 map 时必须满足的约束，以及这六个 map 如何满足它们：

| 约束（驱动 API） | A、B map | C、D map |
|---|---|---|
| `tensorMap` "must be aligned to 64 bytes"；该对象为 128 字节 | 被编码进一个局部的 `CUtensorMap`（`make_tma_copy_desc` 中的 `TmaDescriptor tma_desc{}`），按其类型对齐（`alignof(CUtensorMap) >= 64`，`explicit.hpp:66`）；kernel 的副本位于一个按 128 对齐的块的偏移 0/128 处 | 256/384 |
| `globalAddress` "must be 16 byte aligned" | `cudaMalloc` 指针（CUDA Programming Guide 保证至少 256 字节对齐） | 相同 |
| `globalDim` 各项 "must be non-zero and less than or equal to 2^32" | (8192, 8192, 1) | (8192, 8192, 1) |
| `globalStrides` "must be a multiple of 16 and less than 2^40" | 16384, 0 | 32768, 0 |
| `boxDim` 各项 "must be non-zero and less than or equal to 256"；在 `INTERLEAVE_NONE` 下，`boxDim[0]` x 元素大小 "must be a multiple of 16 bytes" | (64, 64, 1)；128 B | (32, 16, 1)；128 B |
| 对 32B / 64B / 128B swizzle，以字节计的 box 内维 `<= 32` / `<= 64` / `<= 128`（"`CU_TENSOR_MAP_SWIZZLE_128B*` require the bounding box inner dimension to be <= 128"） | 64 x 2 B = 128 | 32 x 4 B = 128 |
| `elementStrides` 各项 "must be non-zero and less than or equal to 8" | (1, 1, 1) | (1, 1, 1) |

内维规则决定了两个 box 的 shape：5.4.2 推导它们，5.4.3 (1) 则陈述它对 k-tile 和 D box 的后果。

### 5.4.2 两条 box 截断规则

两个 box 都比逻辑上的 per-CTA tile 更*小*，出于两个不同的原因。

**A：(64 K, 64 M, 1)，来自 multicast 拆分。** per-CTA 的 A stage 在逻辑上是 (64 K, 128 M) = 16384 字节。但 `make_tma_atom_A_sm100` 传入 `num_multicast = size<2>(cluster_shape) = 2`（collective 所传入的 VMNK cluster layout `cluster_layout_vmnk` 的 N 范围，`sm100_mma_warpspecialized.hpp:374-383`），因为该操作是 `SM100_TMA_2SM_LOAD_MULTICAST`（`copy_traits_sm100_tma.hpp:754-758`："Use only the N-CTAs in the Multicast"），而 `make_tma_copy_desc` 从*最后一个* mode 开始把 box 除以 multicast 数目（`copy_traits_sm90_tma.hpp:996-1002`；工作树中为 `:1003-1009`）：

```
for (i = tma_dim-1, multicast = 2; multicast > 1; --i):      i = 2 (L): box 1   -> ceil_div(1, 2) = 1, multicast = ceil_div(2, 1) = 2 (nothing absorbed)
    new_mult = ceil_div(multicast, box[i])                     i = 1 (M): box 128 -> ceil_div(128, 2) = 64, multicast = ceil_div(2, 128) = 1 (stop)
    box[i]   = ceil_div(box[i], multicast)                     result: (64, 64, 1) = 8192 bytes
    multicast = new_mult
```

共享同一个 `tm` 的两个 CTA（rank `x` 与 `x + 2`：`x` 相同，`y` 不同）各自加载一个不同的 64 行半块（第 `128 tm + 64 y .. +64` 行），并把它 multicast 给双方。这个半块落在 16384 字节 stage 内的 `+ 8192 y` 处（`explicit.cu:768`）。这种拆分之所以合法，仅仅因为 64 行是 8 行 swizzle 原子的整数倍。SW128 图案把 16 字节 chunk 索引与 `row % 8` 做 XOR，而 `(64 y + r) % 8 == r % 8`，所以写在 `+8192 y` 处的半块恰好带有 UMMA 描述符对该 stage 第 `64 y + r` 行所期望的 swizzle 相位。

**C/D：(32 M, 16 N, 1)，来自 128 字节的 swizzle 跨度。** epilogue subtile 是 (128 M, 16 N) 的 fp32 = 8192 字节，但 box 是 2048 字节。`construct_tma_gbasis` 对*未 swizzle 的* smem stage layout 求逆，以元素计为 `((32,4),(4,4)):((1,512),(32,128))`：32 个 M 连续，4 个 N 以 stride 32 个元素 = 128 B，然后下一个 32 行块以 stride 512 个元素 = 2048 B，然后接下来的 4 列以 stride 128 个元素 = 512 B。它在其 gmem basis 的 stride 不为 1 的第一个 smem mode 处截断（`copy_traits_sm90_tma.hpp:760` 与 `:776-783`；工作树中为 `:767` 与 `:783-790`）。在 32 个连续的 M 元素和 16 个 N 列之后，下一个 M 块跳跃 512 个元素，这不再延续 M mode，所以 box 止于 (32, 16)。这与驱动关于 128 字节 swizzle 的规则（box 内维至多 128 字节）一致：32 个 fp32 = 128 B（驱动 API 文档，`cuTensorMapEncodeTiled`）。因此一个 subtile 的一次 `copy(tma_store_d, ...)` 展开为**四**次 TMA 存储 `i = 0..3`，坐标为 `(128 tm + 32 i, 128 tn + 16 e, 0)`，源自 smem `stage_base + 2048 i`（`explicit.cu:1053-1066`）：每个 CTA tile 32 次存储（笔记，第 13 节第 14 项记录了一个早先的计划在这一点上弄错了）。

### 5.4.3 一个 TMA box 是什么

一个 box 是全局内存中的一个稠密子张量，也是共享内存中的一个*连续*映像，其各行是最内层维度（这里是 128 字节），其 16 字节（或 32 字节）chunk 由 swizzle 置换：

```
A box, map A, coordinates (64 q, 128 tm + 64 y, 0): 64 rows of 64 fp16 = 64 x 128 B = 8192 B

global (row-major A, K contiguous)                 smem image (SWIZZLE_128B): line r at +128 r,
row 128tm+64y+r, k = 64q .. 64q+63                 16-byte chunk c of the row stored at chunk position c ^ (r % 8)
       k = 64q         ...        64q+63           position:  0  1  2  3  4  5  6  7
 r=0  [c0 c1 c2 c3 c4 c5 c6 c7]                     r=0      [c0 c1 c2 c3 c4 c5 c6 c7]
 r=1  [c0 c1 c2 c3 c4 c5 c6 c7]                     r=1      [c1 c0 c3 c2 c5 c4 c7 c6]
 r=2  [c0 c1 c2 c3 c4 c5 c6 c7]                     r=2      [c2 c3 c0 c1 c6 c7 c4 c5]
 ...                                                ...
 r=7  [c0 c1 c2 c3 c4 c5 c6 c7]                     r=7      [c7 c6 c5 c4 c3 c2 c1 c0]
 r=8  (same as r=0)                                 r=8      [c0 c1 c2 c3 c4 c5 c6 c7]   (period 8 lines = 1024 B = one swizzle atom)
 ...
 r=63

D box, map D, coordinates (128 tm + 32 i, 128 tn + 16 e, 0): 16 columns of 32 fp32 = 16 x 128 B = 2048 B

global (column-major D, M contiguous)              smem image (SWIZZLE_128B_ATOM_32B): line n at +128 n,
column 128tn+16e+n, m = 128tm+32i .. +31           32-byte chunk q of the line stored at chunk position q ^ (n % 4)
       m -> 32 values                              position:  0  1  2  3
 n=0  [q0 q1 q2 q3]                                 n=0      [q0 q1 q2 q3]
 n=1  [q0 q1 q2 q3]                                 n=1      [q1 q0 q3 q2]
 n=2  [q0 q1 q2 q3]                                 n=2      [q2 q3 q0 q1]
 n=3  [q0 q1 q2 q3]                                 n=3      [q3 q2 q1 q0]
 n=4  (same as n=0)                                 n=4      [q0 q1 q2 q3]   (period 4 lines = 512 B = one swizzle atom)
 ...
 n=15
```

有三件事需要内化，因为它们决定了第 1 节的常量。(1) 一个 box 行至多是 swizzle 跨度，所以 k-tile 是 64 个 fp16，而 D box 的 M 范围是 32 个 fp32。(2) swizzle 周期是 1024 B（SW128）或 512 B（SW128_32B）。硬件根据*绝对*共享地址位计算 XOR，因此基址未对齐会使图案发生偏移；这就是为什么 A/B stage 基址必须 1024 字节对齐、C/D stage 基址必须 512 字节对齐（`explicit.cu:162-163`）。(3) swizzle 是对一行之内 chunk 的纯置换，因此字节总数与事务数都不受影响。

## 5.5 `explicit.cu` 第 1 节，第 1 组：配置常量

```
$ sed -n 104,120p explicit.cu
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
```

| 常量 | 值 | 为什么是这个值（除非另有标注，均为推导） |
|---|---|---|
| `kThreads` | 256 | 8 个 warp，每个承担一种角色：0 MMA，1 scheduler（仅 rank 0），2 mainloop producer，3 epilogue-load（beta = 0 时退出），4-7 epilogue。warp 特化的 kernel 给每个异步引擎（TMA、tensor core、TMEM 读取者）各自一个 warp，因此 TMA 发射、MMA 发射与 TMEM 读取作为相互独立的指令流运行，而不是在一个 warp 中串行化（第 4 章） |
| `kMmaM, kMmaN, kMmaK` | 256, 128, 16 | 原子 `SM100_MMA_F16BF16_2x1SM_SS`，其 `Shape_MNK (256,128,16)`（实测，笔记 D.1 D1；`mma_sm100_umma.hpp:552`）。M = 256 是最大的 `cta_group::2` M（`m_dim` 字段允许 64、128、256，`mma_sm100_desc.hpp:435`）；pair 的两个 128 行 CTA tile 叠成它。N = 128 是 CTA tile 的 N，由两个 CTA 的 64 列 B 半块装配而成。K = 16 个 fp16 是该原子对 `kind::f16` 的固定 K（一个 128 字节 smem 行中的一个 32 字节切片） |
| `kKBlocks` | 4 | 64 / 16：每个 k-tile 四条 MMA，每条把描述符起始地址推进 32 B（第 5.8 节） |
| `kStages` | 8 | builder 的 carveout 公式（笔记，第 6.1 节）：`floor((232448 - 192 - 33792) / 24592) = floor(198464 / 24592) = 8`，其中 232448 是可选启用的 smem，192 是 kernel 级的 carveout（屏障、CLC 响应槽位、TMEM 基址指针），33792 是 epilogue 存储，24592 = 16384 (A) + 8192 (B) + 16（一个 `PipelineTmaUmmaAsync<1>` 的存储）是每个 stage 的量（`sm100_umma_builder.inl:84-99`）。深度 8 是能放得下的数，而不是调优出来的数字；第 3 章表明它有余量地覆盖了 TMA 延迟 |
| `kAccStages` | 4 | builder 的 2SM 分支：`AccumulatorPipelineStageCount_ = TotalTmem / (CtaTileM x CtaTileN) = (128 x 512) / (128 x 128) = 65536 / 16384 = 4`，然后 `AccumulatorPipelineStageCount = min(4u, 4) = 4`（`sm100_umma_builder.inl:260-267`）。它在数值上等于 512 / 128，仅仅因为 CTA tile 有 128 行 = `TotalTmemRows`；1SM 的 M = 64 情形走的是另一条分支，`(512 / CtaTileN) x 2`。整个 512 列的 TMEM 被切成四个 128 x 128 的 fp32 累加器，使 MMA warp 最多可以领先 epilogue 3 个 tile |
| `kClcStages` | 2 | `SchedulerPipelineStageCount = 2`（`sm100_umma_builder.inl:282`）：两个 16 字节的 CLC 响应槽位；在前一个响应被消费的同时有一个查询在途（第 7 章） |
| `kThrottleStages` | 2 | throttle 流水线复用 scheduler 的 stage 计数（`sm100_tile_scheduler.hpp:82`） |
| `kEpiStages` | 4 | `StagesC = ReuseSmem ? max(min(EpiTiles, 4), StagesD + 1) : min(EpiTiles, 4)`，其中 `EpiTiles = 8`、`StagesD = min(8, 2) = 2`、`ReuseSmem = (sizeof_bits(C) = 32) > 8`（`sm100_builder.inl:1109-1115`）：`max(4, 3) = 4` 个复用的 C/D stage，每个 8192 B |
| `kEpiSubtiles` | 8 | 每个 128 x 128 tile 有 128 / 16 个 subtile |
| `kEpiTileN` | 16 | `EpilogueTileAuto` -> (128, 16)（`sm100_builder.inl:999-1046`）。`M = min(CtaM, 32 x WarpM) = min(128, 32 x 4) = 128`，其中对一个 MMA M 为 256 的 2SM 原子，`TmemWarpShape_MN = (4, 1)`（`:1074-1079`）。`N_perf` 走的是*残差加载*分支，因为 `DisableSource = is_void_v<ElementC_> = false`（`:984`；尽管运行时 beta = 0，`ElementC = float`），而在该分支中 `MaxBits = 32`、`CtaM > 64`、`CtaN <= 128` 给出 16（`:1024-1025`）。`N_min_C = N_min_D = 8 x WarpN = 8`，因为 C 与 D 是 M-major（`:1039-1042`）。`N_tmp = min(128, max(16, 8, 8)) = 16`，且 `128 % 16 == 0`，所以 `N = 16`（`:1045-1046`）。16 个 fp32 列 = 一条 warp 宽的 `tcgen05.ld.32x32b.x16`，每个 lane 16 个值。若 `ElementC = void`，同样的公式给出 `N_perf = 4096 / 128 = 32`（`:1018-1019`）：一个 EpilogueTile (128, 32)，4 个 16384 B 的 subtile，以及一个重新推导的 C/D stage 计数（我对同一段代码的解读） |
| `kTmemColumns` | 512 | 整个 TMEM：在 sm_100a/sm_100f 上它具有 "a two-dimensional structure of 512 columns and 128 rows per CTA, with each cell being 32-bits in size"（PTX ISA 9.7.18.1），每个 SM 256 KiB；pair 用一次 `cta_group::2` alloc 分配它的全部。该计数受 ISA 约束："The unit of allocation is 32 columns and the number of columns being allocated must be a power of 2"（9.7.18.1.2），而对 `tcgen05.alloc`，"If .exclusive is not specified, the operand nCols must be a power of 2, in the range [32, 512]"；512 是最大值 |
| `kPrologueKTiles` | 8 | baseline 的 `load()` 每个 tile 被调用两次，第一次针对 `min(Stages, k_tiles)` 个 k-tile，然后针对其余的（kernel 在工作树中为 `:657`，在 cb424739 处为 `:629`，即 `explicit.cu:115` 所引用的数字），原本是为了在二者之间穿插 C 的加载；在 beta = 0 时两次调用之间什么也没有，但保留这一拆分就保留了 baseline 的 SASS 所具有的两个未展开的循环 (8 + 120)（笔记 D.7） |
| `kStageBytesA` | 16384 | 128 行 x 64 个 fp16 x 2 B |
| `kStageBytesAHalf` | 8192 | 一个 CTA 的 64 行半块（A box），见 5.4.2。源码注释警告说它 "equal to kStageBytesB by coincidence of the tile shape"（`explicit.cu:117`）：64 行的 A 半块，64 x 64 x 2 B，等于 64 列的 B 半块，64 x 64 x 2 B，仅仅因为 `CtaM / 2 = CtaN / 2 = 64`。换一种 tile 长宽比，比如 `CtaN = 256`（一个 256 x 256 x 16 的 MMA，最大的 `cta_group::2` N），per-CTA 的 B 半块就会是 128 x 64 x 2 B = 16384 B，而 A 半块仍为 8192 B。这就是为什么 kernel 保留了各自独立的名字，而不是复用一个 |
| `kStageBytesB` | 8192 | 64 行 x 64 个 fp16 x 2 B：per-CTA 的 B 半块是 64 个 N 行，因为 pair 的 MMA N = 128 被拆分到两个 CTA 上 |
| `kEpiStageBytes` | 8192 | 128 x 16 个 fp32 x 4 B：一个 subtile |
| `kStoreBoxBytes` | 2048 | 32 x 16 个 fp32 x 4 B：一个 D box，每个 subtile 四个 |

## 5.6 LAYOUT 注释块，逐 mode 解码

```
$ sed -n 122,135p explicit.cu
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
```

一个 CuTe layout `Shape : Stride` 通过坐标与 stride 的点积把坐标映射到元素索引。`Sw<B,M,S> o L` 把 swizzle 应用于 `L` 所产生的*偏移*，单位就是 `L` 书写时所用的单位。这里组合后的 layout 以元素为单位打印，但完整的 `TRACE_H1` 打印输出（`trace_out_v2/host.txt:37-40`）读作 `Sw<3,4,3> o smem_ptr[16b](unset) o ...` 和 `Sw<2,5,2> o smem_ptr[32b](unset) o ...`；注释省去了 `smem_ptr` 这一项，而附着在 `smem_ptr_flag` 上的 swizzle 作用于*字节*地址，因为 `make_tensor` 把它移到了共享内存指针上（`cute/pointer_flagged.hpp:65`、`cute/pointer.hpp:180-182`、`cute/pointer_swizzle.hpp:86-87`；第 10.5.1 节）。这就是注释以字节重述它的原因。逐个解读：

**`SmemLayoutA = Sw<3,4,3> o ((_128,_16),_1,_4,(_1,_8)) : ((_64,_1),_0,_16,(_0,_8192))`** 有四个 mode：

| Mode | Shape | Stride（fp16 元素） | 含义 |
|---|---|---|---|
| 0 | `(128, 16)` | `(64, 1)` | 一个 MMA 原子的 A 切片：行 `m` (0..127)、16 宽 k-block 内的 `k` (0..15) -> `64 m + k`；一行是 64 个元素 = 128 B（整个 k-tile 行），k-block 占其中 16 个 |
| 1 | `1` | `0` | 沿 M 的原子 tile 数（1：原子的 per-CTA M 为 128，就是 CTA tile） |
| 2 | `4` | `16` | k-block 索引 `kb` (0..3)：每个 k-block `+16` 个元素 = `+32` B |
| 3 | `(1, 8)` | `(0, 8192)` | stage 索引 `s` (0..7)，以嵌套形式打印；每个 stage `8192` 个元素 = 16384 B（合并为 `8:8192`，笔记 D.6 第 18 项） |

所以，以相对 `OFF_SMEM_A` 的字节计，在 swizzle 之前：`o(m, k, kb, s) = 128 m + 2 k + 32 kb + 16384 s`。`Swizzle<3,4,3>` 随后把位 [4,7) 与位 [7,10) 做 XOR：`phys = o ^ ((o & 0x380) >> 3)`。由于 `128 m` 是一个 stage 内唯一能到达位 [7,10) 的项，位 [7,10) 就是 `m % 8`，于是物理字节为

```
phys(m, k, kb, s) = 16384 s + 128 m + 16 * ( ((32 kb + 2 k) >> 4) ^ (m & 7) ) + (2 k & 15)
```

算例：`m = 5, k = 3, kb = 2, s = 0`：`o = 640 + 6 + 64 = 710 = 0x2C6`；chunk 位 `(0x2C6 >> 4) & 7 = 4`，行位 `(0x2C6 >> 7) & 7 = 5`，新 chunk `4 ^ 5 = 1`；`phys = 640 + 16 + 6 = 662 = 0x296`。kernel 从不对此求值。TMA 正是以这个置换写入 box (5.4.3)，而 UMMA 单元通过一个描述符读取它，该描述符的 `layout_type = SWIZZLE_128B` 告诉它应用同样的 XOR（第 5.8 节）。这段注释的存在是为了让你能检查两边是否一致。

**`SmemLayoutB = Sw<3,4,3> o ((_64,_16),_1,_4,(_1,_8)) : ((_64,_1),_0,_16,(_0,_4096))`** 与之相同，只是每个 CTA 有 64 个 N 行，所以 `o(n, k, kb, s) = 128 n + 2 k + 32 kb + 8192 s`，且一个 stage 为 4096 个元素 = 8192 B。

**`SmemLayoutAtomC/D = Sw<2,5,2> o (_32,_4) : (_1,_32)`** 以 fp32 元素计：32 x 4 原子的元素 `(m, n)` 位于 `m + 32 n`，即字节 `4 m + 128 n`；一列 32 个 fp32 是一个 128 字节行，四行构成 512 字节的原子。`Swizzle<2,5,2>` 把位 [5,7)（行内的 32 字节 chunk，`(m >> 3) & 3`）与位 [7,9)（行索引，`n & 3`）做 XOR：

```
phys_atom(m, n) = 4 (m & 7) + 32 * ( ((m >> 3) & 3) ^ (n & 3) ) + 128 n
```

完整的 8192 字节 stage 把这个原子当作 tile 沿 M 平铺 4 x（stride 2048 B）、沿 N 平铺 4 x（stride 512 B）：即 5.4.2 中未 swizzle 的 `((32,4),(4,4)):((1,512),(32,128))`。对于 0..127 中的 `m`、0..15 中的 `n`：`byte(m, n) = 2048 (m >> 5) + 512 (n >> 2) + phys_atom(m & 31, n & 3)`。代入 `m = 32 w + l`（epilogue warp `w`，lane `l`）和 `n = j`，得到 `2048 w + 4 (l & 7) + 128 (j & 3) + 32 (((l >> 3) & 3) ^ (j & 3)) + 512 (j >> 2)`。这正是 epilogue 组装其存储地址的方式（第 10 章）：per-lane 表 `st_off[c] = OFF_SMEM_CD + 2048u * w + 4u * (l & 7u) + 128u * c + 32u * (((l >> 3) & 3u) ^ c);`，其中 `c = j & 3`（`explicit.cu:1013`），再加上在存储本身处加入的 `512u * uint32_t(j >> 2)` 和 stage 项 `kEpiStageBytes * uint32_t(e & 3)`（`explicit.cu:1047`）。

**`TiledMMA = MMA_Atom SM100_MMA_F16BF16_2x1SM_SS 256x128x16, ThrLayoutVMNK (2,1,1,1)`**：该原子是一个 256 x 128 x 16 的 MMA，由两个“线程”协同执行（V = pair 的两个 CTA）。平铺沿 M、N 和 K 各把它重复一次，所以一个原子*就是* MMA tile。**`LayoutC_TV (_2,(_128,_128)) : (_128,(_1,_256))`** 把 `(v, (m, n))` 映射到 256 x 128 累加器中的 `128 v + m + 256 n`，按列主序索引，M stride 为 1、N stride 为 256：CTA `v` 拥有行 `[128 v, 128 v + 128)`，覆盖每一列。这就是为什么 pair leader 的单条 `tcgen05.mma` 填满两个 CTA 的 TMEM，以及为什么每个 CTA 的 epilogue 只会读取它自己的 128 行（第 9 章和第 10 章）。

**`EpilogueTile = (128, 16)`** 是每一轮 TMA 存储所搬运的 subtile。**`TileShape (256, 128, 64)`** 是 pair 每个 k-tile 的 MMA tile，**`CtaShape_MNK (128, 128, 64)`** 是 per-CTA 的份额，**`ClusterShape (2, 2, 1)`** 是启动 cluster。最后三行注释（`explicit.cu:133-135`）重述了 5.4.1 的 tensor map，并固定了 kernel 在每次 TMA 调用中使用的坐标约定：A/B 为 `(k, row, 0)`，C/D 为 `(m, n, 0)`，以元素为单位，最内层在前。

## 5.7 `explicit.cu` 第 1 节，第 2 组：共享内存映射

```
$ sed -n 137,166p explicit.cu
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
```

### 5.7.1 字节映射

这些偏移是把 `sm100_gemm_tma_warpspecialized.hpp:182-212` 中 CUTLASS 的 `SharedStorage` 展平后的结果：一个按声明顺序容纳全部屏障数组的 `PipelineStorage`（`aligned_struct<16>`）、CLC 响应槽位、TMEM 基址字，然后是 `TensorStorage`（先是 epilogue 存储，再是 mainloop 存储）。每个屏障都是一个 8 字节的 `mbarrier` 对象（`ClusterBarrier`/`ClusterTransactionBarrier`，`barrier.h`）。这张映射，连同每一个屏障（计数是第 5.9 节的；初始化 warp 是 prologue 的，第 11 章）：

```
offset   size   region / object                            count or alignment           initialised by
------   -----  -----------------------------------------  ---------------------------  --------------------
     0     64   OFF_MAIN_FULL    full[s], s = 0..7, 8 B    1 (+ 49152 tx bytes)         warp 0 (elected lane)
    64     64   OFF_MAIN_EMPTY   empty[s], s = 0..7        2                            warp 0
   128     32   OFF_EPI_LOAD_FULL  [0..3]                  (C-load pipeline, beta != 0)  not initialised
   160     32   OFF_EPI_LOAD_EMPTY [0..3]                  (idem)                       not initialised
   192     16   OFF_LOAD_ORDER   [0..1]                    (idem)                       not initialised
   208     16   OFF_CLC_FULL     full[0..1]                1 (+ 16 tx bytes)            warp 4
   224     16   OFF_CLC_EMPTY    empty[0..1]               800                          warp 4  (only CTA 0's are arrived on)
   240     32   OFF_ACC_FULL     full[st], st = 0..3       1                            warp 5
   272     32   OFF_ACC_EMPTY    empty[st], st = 0..3      256                          warp 5  (only the leader's are arrived on)
   304     16   OFF_THROTTLE_FULL  [0..1]                  32                           warp 3  (rank 0 uses them)
   320     16   OFF_THROTTLE_EMPTY [0..1]                  32                           warp 3
   336      8   OFF_TMEM_DEALLOC                           32                           warp 0 (lane_pred lane)
   344      8   padding                                    PipelineStorage is aligned_struct<16> -> sizeof 352
   352     32   OFF_CLC_RESPONSE  response[0..1], 16 B     alignas(16); read by ld.shared.b128
   384      4   OFF_TMEM_BASE_PTR                          written by tcgen05.alloc, read by warps 0 and 4-7
   388    124   padding                                    tensors.epilogue is alignas(512)
   512  32768   OFF_SMEM_CD   C/D stage (e % 4), 4 x 8192  Swizzle<2,5,2>; 512 % 512 == 0
 33280    512   FusionStorage (an empty tuple, 1 B) + pad  the epilogue TensorStorage is padded to its own 512-B alignment
 33792 131072   OFF_SMEM_A    A stage s, 8 x 16384         Swizzle<3,4,3>; 33792 = 33 x 1024
164864  65536   OFF_SMEM_B    B stage s, 8 x 8192          Swizzle<3,4,3>; 164864 = 161 x 1024
230400          end = kSmemBytes                           2048 B below the 232448-B opt-in maximum
```

关于总量有两点说明。第一，stage 数。builder 的公式 (5.5, `kStages`) 剩下 `198464 - 8 x 24592 = 1728` B，远少于第九个 stage 在该公式中所需的 24592 B（24576 B 的张量 + 16 B 的屏障）。*物理*余量是 `232448 - 230400 = 2048` B，多出 320 B，因为 builder 的估计 `192 + 33792 + 8 x 24592 = 230720` 把真实的 `SharedStorage` 高估了 320 B。它把 epilogue 按其完整的 33792 B 的 `SharedStorage` 来做预算（33280 字节的 `TensorStorage` 加上一条 64 字节的加载流水线，向上取整到 512；笔记，第 6.1 节），而 kernel 只嵌入了 33280 字节的 `TensorStorage`：多算了 512 B。在另一个方向上，它的 192 字节 kernel carveout 加上 stage 内部的 8 x 16 B mainloop 屏障，把张量之前真实的 512 字节区域少算了 192 B；净值 512 - 192 = 320 B。无论怎样，第九个 stage 都无法企及。

第二，这两处填充从何而来。`tensors.epilogue` 从 512 开始而不是从 388 开始，是因为 epilogue 的 `CollectiveStorageReuseC` 把它的 `smem_C`/`smem_D` 联合体成员声明为 `alignas(MaxSmemAlignment)`，其中对 `Swizzle<2,5,2>` 有 `SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{}) = 1 << (B + M + |S|) = 1 << (2 + 5 + 2) = 512`（`sm100_epilogue_tma_warpspecialized.hpp:170-172`、`:190-193`；`cutlass/detail/layout.hpp:420-426`）。kernel 的 `TensorStorage : cute::aligned_struct<128, _1>`（`sm100_gemm_tma_warpspecialized.hpp:203`）从它的第一个成员继承了对齐 512，所以编译器把 388 向上取整到 512。33280 处的 512 字节是 epilogue 的 `TensorStorage { CollectiveStorage collective; FusionStorage thread; }`（`sm100_epilogue_tma_warpspecialized.hpp:208-215`）的 `sizeof` 取整：32768 B 的 stage 加上线性组合树那 1 字节的空 `FusionStorage` = 32769，向上取整到该结构体自身的 512 字节对齐 = 33280。因此 mainloop 存储从 512 + 33280 = 33792 开始。mainloop 的 `TensorStorage` 只声明了 `aligned_struct<128>`（`sm100_mma_warpspecialized.hpp:224`）。`Swizzle<3,4,3>` 所需的 1024 字节对齐（`alignment_for_swizzle = 1 << (3 + 4 + 3) = 1024`）*没有在任何地方声明*；它成立只是因为 33792 = 33 x 1024 恰好是其倍数（笔记，第 6.2 节）。这就是 explicit kernel 在 `:162` 处的 `static_assert` 存在的原因：它把一个巧合变成了一个受检查的不变量。实测确认：`alignof 512`、`PipelineStorage 352`、`EpilogueSharedStorage 33792`、`off_epilogue_thread 33280`、`off_smem_A 33792`（笔记 D.1 D2）。

关于基址的三个事实使这张映射可用（事实 1 和 2 为实测，笔记 D.1 D5；事实 3 读自 kernel 的源码；在 `explicit.cu:138` 和 `:156` 处重述，peer 位规则在 `:187` 处）：

1. rank `r` 的动态共享内存从 32 位的 `shared::cta` 地址 `0x400 + (r << 24)` 开始。最低的一个千字节是系统保留的共享内存（二进制文件中每个 kernel 资源行里的 `SHARED:1024`，笔记 G.3）。SASS 显示了那里放着什么（笔记 D.2 的 `tcgen05.alloc` 行，以及 D.6 第 7 项）：ptxas 把 `cta_group::2` 的 TMEM 分配下沉为 leader 被选中的 lane 中的一次原子操作，随后是与 peer 的一次握手。握手把结果存储在本地（`IMAD.SHL.U32 R8, R9, 0x20, RZ; STS [UR29+0x180], R8`，原子操作的结果从 32 列单位重新换算为列），用 `st.async` 把它转发出去，并在 peer 的保留千字节中 `(rank << 24) + 0x48` 处完成一个屏障，簿记字位于 `+0x50/+0x54/+0x58`。kernel 没有声明任何静态 `__shared__`（`sharedSizeBytes 0`，笔记 D.1 D4），所以 `extern __shared__ char smem[]` 是第一个用户字节，位于 `0x400`。因此 `kSmemBaseLow24 = 0x400`。添加任何静态 `__shared__` 变量都可能移动动态基址：静态存储相对于动态数组位于何处不是源码层面的契约（NVIDIA 内部）。那会使那些使用 `kSmemBaseLow24` 的编译期交叉检查失效，但不会使 kernel 本身失效，后者在运行时计算基址（事实 3）。rank 位于地址的字节 3（位 [24,32)）；对于 4 个 CTA 的 cluster，只有位 24-25 被置位。
2. 因此 `smem_base + OFF_*` 是任何对象的共享地址，而清除这样一个地址的位 24 就指向 *pair leader*（偶数 rank）中的同一个对象：`0xFEFFFFFF`（第 5.9 节）。
3. kernel 在运行时（`explicit.cu:1098`）*计算* `smem_base = smem_u32(smem)`（`:258` 处包裹 `__cvta_generic_to_shared` 的 wrapper），且从不假设 `0x400`；`kSmemBaseLow24` 只为 `:165` 和 `:177` 处的编译期交叉检查而存在。

由此得出的地址空间规则（陈述于 `explicit.cu:22-23`）：每个指名共享内存的 `asm` 操作数都是 32 位的 `smem_base + OFF_*`；每个 C++ 解引用都使用通用指针 `smem + OFF_*`。两种经典错误有不同的症状。把裸的 `OFF_*` 作为 `shared::cta` 操作数传入而不加 `smem_base`，在 rank 0 中恰好偏差 `0x400` = 1 KiB；低于 `0x400` 的偏移（控制区和 `OFF_SMEM_CD`）落在保留的千字节中；`OFF_SMEM_A` 和 `OFF_SMEM_B` 落在低了 1 KiB 的位置，即最后一个 C/D stage 和最后一个 A stage 中。在 rank 1-3 中它偏差 `0x1000400` / `0x2000400` / `0x3000400`（16 / 32 / 48 MiB + 1 KiB），并且携带的是 rank 0 的 rank 位而不是本 CTA 的（一个 `shared::cluster` 操作数会指向 rank 0 的窗口；对于位于执行 CTA 窗口之外的 `shared::cta` 操作数，PTX ISA 5.1.7 未作任何规定）。在期望 32 位共享操作数之处传入 64 位通用指针则根本不是偏移错误：通用地址被截断为 32 位，变成一个任意的共享地址或者一次故障。

### 5.7.2 每个 `static_assert` 保护什么

| 行 | 断言 | 保护什么 | 没有它会出什么问题 |
|---|---|---|---|
| 158 | `OFF_SMEM_A == 512 + 4 * 8192 + 512` | C/D stage 之后的 512 字节填充（空的 `FusionStorage` 被填充到 epilogue 存储的 512 字节对齐） | 偏差 512 的 A stage 基址会破坏 1024 字节的 swizzle 对齐以及每一个 A 地址 |
| 159 | `OFF_SMEM_B == OFF_SMEM_A + 8 * 16384` | A 与 B 的 stage 是连续的 | B 地址以及 B 描述符基址 |
| 160 | `kSmemBytes == OFF_SMEM_B + 8 * 8192` | 请求的动态大小等于布局所得的大小 | 太小：最后几个 B stage 越界；太大：浪费 smem，只有在跨过某个边界时才会减少驻留 CTA 数（这里仍是 1 CTA/SM） |
| 161 | `kStageBytesAHalf == 64 * 64 * 2` | multicast 的那一半等于一个 A box | `+ 8192 y` 目的地和 49152 事务计数 |
| 162 | `OFF_SMEM_A % 1024 == 0`、`OFF_SMEM_B % 1024 == 0` | SW128 swizzle 周期 | TMA 写入和 UMMA 读取都用绝对地址位 [7,10) 做 XOR；未按 1024 对齐的基址会使写方与读方之间的模式错位 -> 垃圾操作数 |
| 163 | `OFF_SMEM_CD % 512 == 0` | SW128_32B 周期 | 对 epilogue 同理：kernel 的 `st.shared` XOR 与 TMA 的反 swizzle 会不一致 |
| 164 | `OFF_CLC_RESPONSE % 16 == 0`、`OFF_TMEM_BASE_PTR % 4 == 0` | 对响应的 `ld.shared.b128` 需要 16 字节对齐的地址；`tcgen05.alloc` 的 b32 存储需要 4 | 未定义行为（PTX ISA 9.4，6.4.1：访问可能悄无声息地屏蔽掉低位地址位，或者该指令可能发生故障） |
| 165 | `0x400 + 230400 < 2^18` | UMMA 起始地址字段保存 `(addr & 0x3FFFF) >> 4`：编码掩码覆盖 `2^18` B = 256 KiB 的 `shared::cta` 空间 (5.8.1) | B200 上任何可启动的东西都无法违反它：最大的窗口是 `0x400 + 232448 = 233472 < 262144`，而一个有 `230400 + 30720 = 261120` B 共享内存的 kernel 根本无法启动。这条断言记录了 256 KiB 的跨度。只有当 `kSmemBaseLow24` 或偏移表被改成假定一个高到足以回绕的基址时，它才会触发；位于 256 KiB 或更高处的地址会在描述符内悄无声息地回绕，MMA 就会读取错误的 stage |
| 166 | `kThreads == 256`、`kMmaM == 256`、`kMmaN == 128`、`kKBlocks == 4` | 文件其余部分硬编码的第 4 节配置 | 每一个循环边界和每一个描述符增量 |

## 5.8 `explicit.cu` 第 1 节，第 3 组：UMMA 描述符

```
$ sed -n 168,177p explicit.cu
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
```

`tcgen05.mma` 不接受指向 A 和 B 的指针。它接受两个 64 位的*共享内存描述符*，其中编码了一个操作数在共享内存中从何处开始以及它的 8 行 swizzle 原子如何排布，再加上一个 32 位的*指令描述符*，其中编码了类型、shape 以及 major 属性。MMA 单元仅凭描述符遍历操作数；kernel 唯一的工作是构造出正确的位。下表复现了 `mma_sm100_desc.hpp:101-127` 的 `SmemDescriptor` 联合体（CUTLASS 的位域名称和范围）。权威定义是 PTX ISA 表 49，"Shared memory descriptor layout"（9.7.18.4.1），它对某些字段边界的划分有所不同。两者在 kernel 设置的每一位上都一致；差异以及 ISA 的额外规则在表后给出。

### 5.8.1 64 位共享内存描述符

| 位 | 字段（`mma_sm100_desc.hpp`） | 此处的值 | 为什么 |
|---|---|---|---|
| [0,14) | `start_address_` = (共享地址 >> 4) | A：`(smem_base + 33792 + 16384 s + 32 kb) >> 4`，掩码截取为 14 位；B：`(smem_base + 164864 + 8192 s + 32 kb) >> 4` | 以 16 字节为单位（4 个 LSB 是 "not included" 的，因此操作数必须 16 字节对齐）。编码的这 14 位，即 ISA 的 `(x & 0x3FFFF) >> 4`，覆盖 256 KiB（5.7.2，`:165`）。掩码 `& 0x3FFF` 丢弃了 `smem_base` 的 rank 位 [24,26)，因此描述符与 rank 无关：leader 的 MMA 在 pair 的两个 CTA 中读取*相同的 CTA 相对偏移*（笔记 B.6） |
| [14,16) | 未使用 | 0 | |
| [16,30) | `leading_byte_offset_` (LBO) >> 4 | 1 (= 16 B) | `make_umma_desc` 为 swizzled K-major 规范 layout 设置 `leading_byte_offset_ = stride_10 = 1`（`mma_sm100_desc.hpp:831-832`，上面几行处断言了 `expected_stride_10 = 1`）；对 K-major SW128 而言，一次 MMA 的 K 范围（16 个 fp16 = 32 B）位于一个 128 字节的行内，因此该字段承载的是规范占位值而非真实的 stride（我对 `:801-809` 处规范 layout 注释的分析） |
| [30,32) | 未使用 | 0 | |
| [32,46) | `stride_byte_offset_` (SBO) >> 4 | 64 (= 1024 B) | 沿 M/N 方向相邻 8 行 swizzle 原子之间的字节距离：8 行 x 128 B = 1024 B（`Layout_K_SW128_Atom_Bits`：`Shape<_8,_1024>`，`mma_traits_sm90_gmma.hpp:84`）；MMA 的 128 行（A）或 64 行（B）即 16 个或 8 个这样的原子 |
| [46,48) | `version_` | 1 | "Set the version for blackwell"（`mma_sm100_desc.hpp:746`）；Hopper 的 wgmma 描述符在此处为 0 |
| [48,49) | 未使用 | 0 | |
| [49,52) | `base_offset_` | 0 | 重复模式起始地址的 swizzle 相位（9.7.18.4.1）；0 是有效的，因为每个 stage 基址都是 1024 字节对齐的（`:757-758`，`constexpr uint8_t base_offset = 0`），而且为 `kb` 把起始地址推进 32/64/96 B 并不触及地址位 [7,10) |
| [52,53) | `lbo_mode_` | 0 | "legacy mode by default"（`:747`） |
| [53,61) | 未使用 | 0 | |
| [61,64) | `layout_type_` | 2 = `SWIZZLE_128B` | `enum class LayoutType { SWIZZLE_NONE = 0, SWIZZLE_128B_BASE32B = 1, SWIZZLE_128B = 2, SWIZZLE_64B = 4, SWIZZLE_32B = 6 }`（`:82-88`）；它告诉 MMA 单元施加与 TMA 写入时相同的 16 字节 chunk XOR |

**PTX ISA 表 49 怎么说、它在何处与联合体不同，以及你在其他 layout 下需要的规则。** ISA 把位 0-14（15 位）定义为 "matrix-descriptor-encode (Matrix start address)"，位 15 为 "Reserved - Must be 0"，位 16-30（15 位）为编码后的前导维字节偏移（相对）*或*字节地址（绝对），位 31 保留。位 32-45 保存编码后的 SBO；位 46-48（3 位）为 "Fixed constant value of 0b001"；位 49-51 为 "Matrix base offset"；位 52 为前导维 stride 模式（0 为相对偏移，1 为绝对地址）；位 53 为 `lut::b` 的 "Matrix B K_Segment start offset"（仅用于解压缩，此处为 0）；位 54-60 固定为 0；位 61-63 为 swizzle 模式（"0. No swizzling 1. 128-Byte with 32B atomic swizzling 2. 128-Byte swizzling 4. 64-Byte swizzling 6. 32-Byte swizzling"；3、5、7 无效），"where matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4"。CUTLASS 的 14 位 `start_address_` 加 2 个未使用的位，以及它的 14 位 LBO 加 2 个未使用的位，是同一组位在更窄名称下的表述；它位于 [46,48) 的 `version_ = 1` 是 ISA 固定值 `0b001` 的位 46。因此起始地址的 256 KiB 跨度是编码掩码 `0x3FFFF` 的属性，而不是一个 "14-bit field" 本身的属性；`:165` 处的断言是针对这个跨度写的。ISA 增加了三条规则。kernel 从构造上就满足它们，但当你编码一种不同的 layout 时必须检查它们。(1) "The following must be 16-byte aligned: Matrix start address, Leading dimension byte offset, Stride dimension byte offset"，因此有 `>> 4`。(2) base offset "is 0 when the repeating pattern of the specified swizzling mode starts" 于 1024 字节边界（对 128 字节 swizzle）、512 字节边界（对 64 字节 swizzle）、256 字节边界（对 32 字节 swizzle）（表 50）；"Otherwise, the base offset must be a non-zero value, computed using the following formula: base offset = (pattern start addr >> 0x7) & 0x7"。每个 A/B stage 基址的 1024 字节对齐（5.7.2，第 162 行）正是让 CUTLASS 得以硬编码 `base_offset = 0` 的原因。(3) "The byte address mode for the leading dimension stride is supported on sm_103a"，因此在 B200 上 LBO 始终是相对偏移（`lbo_mode_ = 0`）。

组装常量高字：SBO `64 = 0x40` 位于高字的位 [0,14)；版本 `1 << 14 = 0x4000`；3 位字段 [61,64) 中的 layout 类型 `2` 是 `0b010`，即全局位 62 = 高字的位 30 = `0x40000000`。求和：`0x40004040`，这就是第二条 `static_assert`（`64u | (1u << 14) | (2u << 29)`：`2 << 29 = 1 << 30`）。常量低字是 LBO `1 << 16 = 0x10000`。合起来是 `0x4000404000010000 = kSmemDescConst`（第一条断言）。

运行时部分是起始地址。对实测基址 `0x400`：`(0x400 + 33792) >> 4 = 0x8800 >> 4 = 0x880` 和 `(0x400 + 164864) >> 4 = 0x28800 >> 4 = 0x2880`（第四条断言），因此 stage 0、k-block 0 的低字是 `0x10880` 和 `0x12880`，并且

```
lo_a(s, kb) = 0x10880 + 1024 s + 2 kb        (16384 B per stage = 1024 units; 32 B per k-block = 2 units)
lo_b(s, kb) = 0x12880 +  512 s + 2 kb        ( 8192 B per stage =  512 units)
```

**算例，重新计算（rank 0，`s = 3`，`kb = 2`）。** A：共享地址 `0x400 + 33792 + 3 x 16384 + 2 x 32 = 0x400 + 0x8400 + 0xC000 + 0x40 = 0x14840`；`>> 4 = 0x1484`；`& 0x3FFF = 0x1484`（放得下）；`lo = 0x10000 | 0x1484 = 0x11484`；`desc_a = 0x40004040_00011484`。用增量形式核对：`0x10880 + 3 x 1024 + 4 = 0x10880 + 0xC00 + 0x4 = 0x11484`。B：`0x400 + 164864 + 3 x 8192 + 64 = 0x400 + 0x28400 + 0x6000 + 0x40 = 0x2E840`；`>> 4 = 0x2E84`；`lo = 0x12E84`；`desc_b = 0x40004040_00012E84`；增量形式 `0x12880 + 3 x 512 + 4 = 0x12880 + 0x600 + 0x4 = 0x12E84`。两者都与注释和实测的 K3 记录相符（`desc_a = 0x4000404000010880` 在 `s = 0, kb = 0` 处，每个 k-block `+2`，每个 stage A `+0x400` / B `+0x200`；笔记 D.1 D7）。

kernel 如何构建它们（`explicit.cu:896-897`、`:928-929`、`:933-934`）：每次进入角色时一次，`desc_lo_a = kSmemDescLo0 | (((smem_base + OFF_SMEM_A) >> 4) & 0x3FFFu)`；每个 k-tile，`uint32_t const lo_a = desc_lo_a + 1024u * rs;`（`rs` 为读 stage）；每个 k-block，`desc64(kSmemDescHi, lo_a + 2u * uint32_t(kb))`。加法作用于 32 位低字而非 64 位值，与 CUTLASS 的 `DescriptorIterator::operator+` 对 nvcc <= 13.3 的做法完全一致（`mma_sm100_desc.hpp:843-875`，加法在 `:870`，`:868` 处的注释："Use 32-bit calculation rather than 64-bit calculation as we only update part of desc"）。在 SASS 中，这是每个操作数每个 k-tile 一条 `ULEA`（A 为 `ULEA UR10, UR8, UR30, 0xa`，x1024；B 为 `ULEA UR12, UR8, UR31, 0x9`，x512），k-block 1-3 每个操作数一条 `UIADD3 +2/+4/+6`（k-block 0 直接使用 `ULEA` 的结果），以及每个描述符一条 `UMOV 0x40004040` 高字，每个 k-tile 八条（实测，`sass_explicit.txt` `0x5270-0x5470`，笔记 G.3；baseline 显示相同的模式，笔记 D.2）。为什么起始地址在一个 swizzled 行内每个 k-block 推进 32 B：描述符的地址是元素（行 0，k = 16 kb）的 *unswizzled* 位置。硬件通过与行索引做 XOR 推导出其他每个元素的物理 chunk，因此让基址步进穿过 128 字节行的四个 32 字节切片是迭代 K 的规范方式；这正是 `UMMA::smem_desc` 迭代器所做的。

### 5.8.2 32 位指令描述符

PTX ISA 9.4，9.7.18.4.2，表 51（"Instruction descriptor format for .kind::tf32, .kind::f16, .kind::f8f6f4, .kind::i8 and .kind::ti16"）用 ISA 的行名定义了这个 32 位字：位 4-5 的 `dtype (Matrix D type)`（"F16 = 0, F32 = 1"），位 7-9 的 `atype (Matrix A type)`（对 `.kind::f16` 为 "F16 = 0, BF16 = 1"），等等。下表使用 CuTe 的镜像 `InstrDescriptor`（`mma_sm100_desc.hpp:416-443`）的字段名，即 kernel 自身注释（explicit.cu:173）的词汇；其值为 `0x10200010`（十进制 270532624，PTX 中 `mov.b32` 的操作数，笔记第 7.2 节）：

| 位 | 字段 | 值 | 此处的含义 |
|---|---|---|---|
| [0,2) | `sparse_id2_` | 0 | 无稀疏性元数据 |
| [2,3) | `sparse_flag_` | 0 | 稠密 |
| [3,4) | `saturate_` | 0 | 无饱和（反正仅适用于 S8） |
| [4,6) | `c_format_` | 1 = F32 | TMEM 中的 fp32 累加器（`0x10` = 位 4）；"0 = F16. 1 = F32, 2 = S32" |
| [6,7) | 未使用 | 0 | |
| [7,10) | `a_format_` | 0 = F16 | `F32F16Format`："0 = F16, 1 = BF16, 2 = TF32" |
| [10,13) | `b_format_` | 0 = F16 | |
| [13,14) | `a_negate_` | 0 | |
| [14,15) | `b_negate_` | 0 | |
| [15,16) | `a_major_` | 0 = K-major | A 是 M x K 行主序（K 连续）；"0 = K-major. 1 = MN-major" |
| [16,17) | `b_major_` | 0 = K-major | B 是 N x K 且 K 连续（harness 的 `ColumnMajor` B） |
| [17,23) | `n_dim_` = N >> 3 | 16 -> N = 128 | `0x200000` = 16 << 17；"3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256)" |
| [23,24) | 未使用 | 0 | |
| [24,29) | `m_dim_` = M >> 4 | 16 -> M = 256 | `0x10000000` = 16 << 24；"4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)" |
| [29,30) | K 维度（联合体中一个匿名的 `: 1`，`:436`） | 0 | PTX ISA 表 51 把位 29 命名为 "K Dimension"：对 `.kind::f16` 其值为 "(Dense K=16 / Sparse K=32) = 0"，对 `.kind::tf32` 同样为 0（"(Dense K=8 / Sparse K=16) = 0"）；`.kind::f8f6f4` 单元格列出两个选项，"(Dense K=32 / Sparse K=64) = 0" 和 "(Dense K=64) = 1"（表 48 仅对 `.e4m3`/`.e5m2` 允许的稠密 K = 64 shape），而合并的 `.kind::i8`/`.kind::ti16` 单元格写的是 "NA"；9.7.18.4.2.1 的 Target ISA Note 把位 29 限制在 "sm_107f or higher in the same family"，因此 K = 64 选项在 B200（sm_100a）上不存在。CUTLASS 没有给这一位命名；`make_instr_desc`（`:482-506`）从不设置它，并且用这个联合体构建的每一个 MMA（`mma_traits_sm100.hpp` 的 f16/bf16、tf32、i8 和 f8f6f4 traits：每条指令稠密 K 为 256 位、稀疏 K 为 512 位）的 K 范围在表 51 中的条目都是 "= 0" 或 "NA"，因此这一位对它们全都为 0（我的分析）。因此 K 被编码为固定值 0，而不是一个自由参数 |
| [30,32) | `max_shift_` | 0 | 权重驻留移位，未使用 |

`0x10000000 + 0x200000 + 0x10 = 0x10200010`，即第三条断言（`(1u << 4) | (16u << 17) | (16u << 24)`）。位 29 把 `kind::f16` 的 K 固定为每条指令 16 个元素（原子的 `Shape_MNK (256,128,16)`），这就是为什么四条指令覆盖一个 64 宽的 k-tile。该描述符对一个 tile 的全部 512 条 MMA 以及对每个 tile 都相同；除操作数描述符和 TMEM 地址外，唯一的逐指令变化是 `scale_c` 谓词（对一个 tile 的第一条 MMA 为 0，之后为 1；第 9 章）。

## 5.9 `explicit.cu` 第 1 节，第 4 组：协议常量

```
$ sed -n 179,192p explicit.cu
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
```

第 4 章解释了这些数字所实现的协议；下面的各行保留第 4 章未曾推导的内容（全部源自所列的源码，与笔记第 7.2 节相符）：

| 常量 | 值 | 推导 |
|---|---|---|
| `kClcTxBytes` | 16 | 一个 CLC 响应是一个 16 字节的 `b128`（`struct CLCResponse { uint32_t data[4] = {0}; };`，`sm100_tile_scheduler.hpp:74`） |
| `kMainloopReleaseMask` | `0xF` | `calculate_multicast_mask<kRowCol>`（`sm100_pipeline.hpp:57-91`）：同一 MMA 行*或*列中的每个 CTA；对一个 2 x 2 的 cluster 和一个 2 x 1 的原子而言，那就是全部四个 rank，因此每个 leader 的 commit 操作都会释放每个 CTA 中的该 stage。之所以是 16 位，是因为 `tcgen05.commit ... multicast::cluster` 接受一个 `.b16` 的 CTA 掩码（`"h"` 操作数） |
| `kPeerBitMask` | `0xFEFFFFFF` | `cute::Sm100MmaPeerBitMask`（`copy_sm100_tma.hpp:45`）：清除一个 `shared::cta` 地址的位 24，即 rank 的位 0，把“我的屏障”变成“pair leader 的屏障”（rank `2y+1 -> 2y`；在偶数 rank 上是空操作）。nvcc 在可能之处把屏障地址中已知为零的低位折叠进该常量。在 explicit kernel 中，字面量 `0xFEFFFFF8`（`and.b32 ..., -16777224`，其中折叠进了 8 字节对齐屏障地址的三个对齐位）只在 epilogue 的累加器释放处幸存（SASS `0x44c0`）。在 producer warp 中，该掩码被折叠到外提的 1024 字节对齐的屏障基址上，成为 `& 0xFEFFFFF0`（PTX `-16777232`，SASS `ULOP3 ... 0xfefffff0` 位于 `0x0c00`），这对所有 `s <= 7` 都等价，因为基址的低四位为零，而 `8 s <= 56` 不可能进位到位 24（折叠已实测，笔记 G.3）。笔记 D.6 第 2 项的 `-16777224` 描述的是 CUTLASS baseline，它每个 k-tile 都带着这条 `and`。`0xFEFFFFFF` 和 `0xFEFFFFF8` 两种形式对每个 8 字节对齐的屏障地址在语义上完全相同；`0xFEFFFFF0` 形式则不然，它在此处只在外提的基址上才正确，如上所述（第 2 章，第 2.9.3 节） |
| `kCacheHintEvictNormal` | `0x1000000000000000` | `TMA::CacheHintSm100::EVICT_NORMAL`（`copy_sm90_desc.hpp:193-197`）= `cute::Sm100MemDescDefault`（`copy_sm100_tma.hpp:46`），即每条 `cp.async.bulk.tensor ... .L2::cache_hint` 加载的 64 位 L2 缓存策略操作数；"normal" 优先级，是 CUTLASS 默认为 mainloop 加载传入的值。在 SASS 中它是寄存器对 `{0x0, 0x10000000}`（笔记 D.2） |
| `kWaitTicks` | `0x989680` = 10,000,000 | `mbarrier.try_wait.parity` 的挂起时间提示操作数（PTX ISA 中的 `timeHint`；CUTLASS 的局部变量是 `ticks`）：一个以**纳秒**为单位的时间限制，超过它之后即使 phase 未完成，被挂起的线程也会恢复执行，于是围绕它的循环会重新测试。PTX ISA，`mbarrier.test_wait`/`mbarrier.try_wait`："Suspended thread resumes execution when the specified phase completes OR before the phase completes following a system-dependent time limit. The optional 32-bit unsigned integer operand timeHint specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit"。10 ms 是 CUTLASS 的 "arbitrarily large" 值（`barrier.h:414-415`）。它是一个提示，因此更短的系统限制可能更早唤醒线程。在 SASS 中它成为重试循环中的 `NANOSLEEP.SYNCS 0x989680`（笔记 D.2） |

`:192` 处的 `static_assert` 重新推导了最容易出错的两个计数（事务字节数和那个 800）。

## 5.10 `explicit.cu` 第 1 节，第 5 组：CuTe swizzle 交叉检查

```
$ sed -n 194,204p explicit.cu
// [CHECK] LAYOUT  CuTe cross-check (compile time only, the only use of CuTe in this TU): the epilogue smem byte formula uses
// Swizzle<2,5,2> on byte offsets, bits [5,7) ^= bits [7,9) (Layout_MN_SW128_32B_Atom<float>, Section 4; B.3.3 step 17).
// Example: byte offset 0x1E4 (fp32 element 121 = row 25 of column 3 of a 32 x 4 fp32 atom column block) -> 0x1E4 ^ ((0x1E4 & 0x180) >> 2) = 0x1E4 ^ 0x60 = 0x184.
__host__ __device__ constexpr uint32_t epi_swz(uint32_t o) { return o ^ ((o & 0x180u) >> 2); }
using EpiSwizzle = cute::Swizzle<2, 5, 2>;
static_assert(EpiSwizzle::num_bits == 2 && EpiSwizzle::num_base == 5 && EpiSwizzle::num_shft == 2, "Swizzle<2,5,2>");
static_assert(EpiSwizzle::yyy_msk::value == 0x180 && EpiSwizzle::zzz_msk::value == 0x60, "Swizzle<2,5,2> masks: bits [7,9) fold into bits [5,7)");
static_assert(epi_swz(0x080u) == EpiSwizzle{}(0x080u) && epi_swz(0x1E4u) == EpiSwizzle{}(0x1E4u) && epi_swz(0x7FFCu) == EpiSwizzle{}(0x7FFCu),
              "epi_swz matches cute::Swizzle<2,5,2> (E.6.1) [confirm at first build: apply() is constexpr]");
// Swizzle<3,4,3> (A/B stages) is applied by the TMA unit on write and by the UMMA unit on read (SWIZZLE_128B); the kernel never computes it.
static_assert(cute::Swizzle<3, 4, 3>::yyy_msk::value == 0x380 && cute::Swizzle<3, 4, 3>::num_shft == 3, "Swizzle<3,4,3>: 16-byte chunk ^= row mod 8");
```

`:196` 注释中的元素 121 是 `0x1E4 = 484 = 4 x 121 = 3 x 128 + 25 x 4`：32 x 4 原子的第 3 列第 25 行（第 5.10.2 节演算了这个例子）。

### 5.10.1 一张图看 `cute::Swizzle<B, M, S>`

`cute/swizzle.hpp:42-96` 定义了作用在整数偏移上的这个函子：

```
      0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx          (the header's own picture, swizzle.hpp:43-47)
                                    ^--^  M = num_base : the M least-significant bits are left alone
                       ^-^       ^-^      B = num_bits : the width of the two masks
                         ^---------^      S = num_shft : the distance from the YYY mask down to the ZZZ mask

      bit_msk = (1 << B) - 1
      yyy_msk = bit_msk << (M + max(0, S))          the source bits
      zzz_msk = bit_msk << (M - min(0, S))          the target bits (for S > 0 this is bit_msk << M)
      apply(o) = o ^ ((o & yyy_msk) >> S)           ZZZ ^= YYY   (swizzle.hpp:78: offset ^ shiftr(offset & yyy_msk{}, msk_sft{}))
```

所以，一个 swizzle 把位于第 `M + S` 位处的 B 位字段 XOR 到位于第 `M` 位处的 B 位字段上，其他一切保持不动。为什么它针对 GPU 共享内存要写成这样：低 `M` 位寻址一个访问单元（一个 16 或 32 字节的 chunk）内部的字节，`ZZZ` 位选择一行之内的 chunk，而 `YYY` 位是行（row）索引。把 chunk 索引与行索引做 XOR，会使给定的逻辑 chunk 所落入的 bank 从一行到下一行发生变化，于是一列 chunk 不会堆积到同一个 bank 组上。CUTLASS 按各自的 `(B, M, S)` 为每个 swizzle 命名：

| Swizzle | `bit_msk` | `yyy_msk`（源位） | `zzz_msk`（目标位） | 通俗说法 | TMA 枚举项 |
|---|---|---|---|---|---|
| `Swizzle<3,4,3>` | 7 | `7 << 7 = 0x380`，位 [7,10) | `7 << 4 = 0x70`，位 [4,7) | 16 字节 chunk 索引（3 位，128 字节行的 8 个 chunk）^= 行索引 mod 8 | `SWIZZLE_128B` |
| `Swizzle<2,5,2>` | 3 | `3 << 7 = 0x180`，位 [7,9) | `3 << 5 = 0x60`，位 [5,7) | 32 字节 chunk 索引（2 位，128 字节行的 4 个 chunk）^= 行索引 mod 4 | `SWIZZLE_128B_ATOM_32B` |

二者的行索引都位于第 7 位，因为二者的行都是 128 字节；它们的差别在于 chunk 大小（16 B vs 32 B），因此 `M = 4` vs `M = 5` 以及 `B = 3` vs `B = 2`。二者都有 `S = B`，这正是让两个字段相邻的原因。

为什么硬件把 XOR 建立在*绝对*共享地址位上，而不是建立在相对于缓冲区的行索引上：TMA 单元和 UMMA 单元只看到地址，也就是一个目的地址或起始地址加上一种 swizzle 模式，而从不看到 CuTe layout 或缓冲区基址。二者都从交给它们的地址的位 [7,10)（SW128）或位 [7,9)（SW128_32B）计算物理 chunk。本章中分别陈述的三个事实都源于这一条规则。A/B stage 基址必须按 1024 字节对齐，C/D 基址必须按 512 字节对齐（`explicit.cu:162-163`；5.4.3 (2)、5.7.2），这样硬件用来做 XOR 的地址位才是 layout 所意指的行索引；同样的对齐正是让 UMMA 描述符的 `base_offset` 得以为 0（5.8.1）而不是 `(pattern start >> 7) & 7` 的原因。epilogue 的 `st.shared` 必须对同样的地址位施加同样的 XOR（`epi_swz`、`:197`），这样 TMA 在数据出去时做的反 swizzle 才能恢复每一列内部的逻辑行顺序（5.10.2）。

### 5.10.2 手算这些断言

- `:199` 钉住三个模板参数。
- `:200` 检查掩码：对 `<2,5,2>`，`bit_msk = 3`、`yyy_msk = 3 << (5 + 2) = 3 << 7 = 0x180`、`zzz_msk = 3 << (5 - 0) = 0x60`。正确。
- `:197` 是手写的公式 `epi_swz(o) = o ^ ((o & 0x180) >> 2)`，也就是带这些掩码且 `S = 2` 的 `apply`。`:201` 在三个点上把它与真正的 `cute::Swizzle` 对照检查（这之所以可行，是因为 `Swizzle::apply` 是 `constexpr`，`swizzle.hpp:73-79`）：
  - `0x080`：`0x080 & 0x180 = 0x080`；`>> 2 = 0x20`；`0x080 ^ 0x20 = 0x0A0`。（第 1 行，chunk 0 -> chunk 1。）
  - `0x1E4`（注释中的例子）：`0x1E4 = 0b1_1110_0100`；`& 0x180 = 0x180`（第 7 位和第 8 位置位：4 行周期中的第 3 行）；`>> 2 = 0x60`；`0x1E4 ^ 0x60 = 0b1_1000_0100 = 0x184`。用原子的术语说，`0x1E4 = 484 = 4 x 121`：一个 32 x 4 fp32 原子块的元素 121，即第 `n = 3` 行（字节 384..511），行内字节 100，chunk `100 >> 5 = 3`，chunk 内偏移 4；swizzled 后的 chunk 为 `3 ^ 3 = 0`，所以物理字节是 `384 + 0 + 4 = 388 = 0x184`。
  - `0x7FFC`：`& 0x180 = 0x180`；`^ 0x60` -> `0x7F9C`。（整个 32768 字节、四个 stage 的 C/D 区域的最后一个字，相对于 `OFF_SMEM_CD`：`4 x 8192 - 4 = 32764 = 0x7FFC`；单个 8192 字节 stage 的最后一个字则会是 `0x1FFC`。）
- `:204` 检查 `Swizzle<3,4,3>`：`yyy_msk = 7 << (4 + 3) = 0x380` 且 `S = 3`，这就是第 5.4.3 节和第 5.6 节的 "16-byte chunk ^= row mod 8" 陈述。

为什么 kernel 在软件中需要 `Swizzle<2,5,2>` 却不需要 `Swizzle<3,4,3>`：TMA 写入 A/B stage，MMA 单元读取它们，这两个硬件代理都认识 `SWIZZLE_128B`，所以没有任何线程会碰到那些字节。*线程*写入 C/D stage（epilogue 中的 `st.shared.b32`），而 TMA 用 `SWIZZLE_128B_ATOM_32B` 读取它。TMA 在数据出去时对 32 字节 chunk 做逆置换，所以线程必须把每个值存到被置换后的位置上；否则，在输出列 `j`（stage 的一个 128 字节行）内，box 行 `m` 收到的将是行 `m ^ (8 (j & 3))` 的数据：被置换的是一行之内的各个八行 chunk，从来不是列。除了仅主机端的头文件桥接中的 `cute::get`（`explicit.hpp:107-110`，一个 kernel TU 从不实例化的模板）之外，这是该 TU 使用的唯一一处 CuTe，而且只在编译期使用（`explicit.cu:17-18`）：设备端代码求值的是整数公式，从来不是 `cute::Layout`。

## 5.11 第 2 节简述：仅用于 trace 的类型

```
$ sed -n 206,240p explicit.cu
///////////////////////////////////////////////////////////////////////////////////////////////////
// 2. [TRACE]  Trace-only diagnostics (E.7.2 B1/B2, E.7.3): hang records in a host-mapped pinned buffer, heartbeats, K_TAIL.
//    With the toggle off this section reduces to the two #else lines: EXPLICIT_WAIT is the plain blocking wait and the CLC
//    record kinds are 0 (clc_consume then emits nothing).
///////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
...   // [TRACE] branch elided (compiled out in the performance binary)
#else
// [PROD] with the toggle off: the baseline's mbarrier.try_wait.parity loop with the 0x989680 suspend hint (W6); site/tile/q vanish
#define EXPLICIT_WAIT(addr, parity, site, tile, q) mbar_wait((addr), (parity))
constexpr uint32_t K_CLC_SCHED_KIND = 0u;             // no records with the toggle off
constexpr uint32_t K_CLC_MMA_KIND   = 0u;
#endif
```

对生产 kernel 要紧的是：开关关闭时，`EXPLICIT_WAIT(addr, parity, site, tile, q)` *就是* `mbar_wait(addr, parity)`，即复制了 `ClusterBarrier::wait`（`barrier.h:410-430`）的 W6 wrapper。`site`、`tile`、`q` 在预处理器中消失，而 `K_CLC_SCHED_KIND = K_CLC_MMA_KIND = 0` 使两个带 trace 的调用点（`:853`、`:913`）成为与不带 trace 的那些（`:793`、`:1028`）相同的 `clc_consume<0>`；`clc_consume` 内部的记录块（`:695-704`）处于同一个开关 `#if` 之下，所以没有任何实例化会发出记录（7.4.1）。两个开关关闭的构建的 SASS 逐字节相同（实测，笔记 G.3 D6）。`__device__` 变量 `g_explicit_hang_buf` 不存在于生产 cubin 中：它只在 `#if defined(CUTLASS_DEABSTRACTION_TRACE)` 之下声明（`explicit.cu:211`、`:229`；源自源码；笔记只提到该变量一次，在 F.2 第 2 项，而不在 G.3 的清单中）。与此一致，G.3 的资源行报告 `GLOBAL:10`，即十个一字节的 libcu++ 标签对象，kernel 的任何指令都不引用它们。在你计时的那个二进制文件中，第 2 节里没有任何东西会花费一条指令或一个字节的全局内存。

---

# 第 6 章 词汇：PTX wrapper W1-W37，逐字符解读

第 4 章解释了这些 wrapper 所实现的协议。第 5 章定义了它们所消费的常量（`OFF_*`、`kPeerBitMask`、计数与掩码）。第 7-11 章展示了它们在各角色中的调用点。本章是这些层次之间的字典。

---

## 6.0 为什么需要一个词汇层，以及如何阅读一条内联汇编语句

### 6.0.1 第 3 节的设计规则

explicit.cu:243-254 处的横幅注释陈述了这份契约：

```
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
```

三个后果塑造了下文的全部内容。

1. **每个 wrapper 一条 asm 字符串（W1 和 W37 是内建函数），从 CUTLASS 复制而来。** explicit kernel 的主张是：同样的 PTX，没有抽象层。凡是 CUTLASS 函数所做的不止是发出一条指令之处（一次选举、一个 `if (pred)`、一个 fence、一个地址掩码），这些额外工作就移到 C++ 中的调用点，而 wrapper 保持为单条指令。发生了四处搬迁。
   - W10：`ClusterBarrier::arrive(ptr, cta_id, pred)` 的 `if (pred)`（barrier.h:489）。
   - W21/W22：`SM100_TMA_2SM_LOAD_3D::copy` 与 `SM100_TMA_2SM_LOAD_MULTICAST_3D::copy` 施加在屏障地址上的 `& Sm100MmaPeerBitMask`（copy_sm100_tma.hpp:115 与 :292），被 kernel 外提到了 `full_masked_base`（explicit.cu:770）。
   - W29/W30：`fma` 与 `umma_arrive_multicast_2x1SM` 的 `elect_one_sync()`（mma_sm100_umma.hpp:570 与 barrier.h:851）。
   - W34：`work_tile_info_from_clc_response` 的 `fence_view_async_shared()`（sm100_tile_scheduler.hpp:430）。

   只有 W11 在内部保留了一个非 asm 步骤：`umma_arrive_2x1SM_sm0` 施加 `& cute::Sm100MmaPeerBitMask`（barrier.h:909），W11 把它写作 `& kPeerBitMask`（explicit.cu:371），而 W11 的调用点是一次单独的每线程 arrive，没有什么可外提的。

2. **地址空间规则。** 每一个指名共享内存的 asm 操作数都是一个 32 位的 `shared::cta` 地址 `smem_base + OFF_*`；每一次 C++ 解引用都使用通用指针 `smem + OFF_*`（explicit.cu:22-23）。因此 wrapper 对共享内存接受的是 `uint32_t` 地址，而绝不是指向它的指针（唯一的指针参数，即 W20-W23 的 `desc_ptr`，是参数空间中 tensor map 的通用地址，第 6.6.1 节）。`smem_u32`（第 6.1 节）是唯一的桥梁。

3. **收敛是调用者的问题。** 有几条指令（`elect.sync`、每一条 `.sync.aligned`、`barrier.cluster.*.aligned`、`bar.sync`）要求整个 warp 一起执行它们。wrapper 不检查这一点。角色函数让 wrapper 周围的每一个分支都保持 warp-uniform（`is_leader`、`is_cta0`、`warp_idx == k`、`e == 7`）。唯一随 lane 发散的谓词是选举结果（一次新的 `elect_one_sync()`，或者 prologue 的 `lane_pred` 在 explicit.cu:736 处 W12 周围以及 :1113-1119 处的预取周围被复用为 `is_leader_lane`）以及 W13 在其 asm 内部转为 PTX 谓词的 `lane < 4`。

### 6.0.2 一条 `asm volatile` 语句的剖析

每个 wrapper 都具有如下 GCC/nvcc 形式

```
asm volatile("template string" : output operands : input operands : clobbers);
```

- **模板字符串。** PTX 文本。`%0`、`%1`、……是操作数占位符，按它们在操作数列表中出现的顺序编号，输出在前。字面的 `%`（PTX 特殊寄存器如 `%cluster_ctarank` 以及 PTX 寄存器名如 `%rx` 需要它）写作 `%%`。`\n\t` 分隔各条 PTX 语句。编译器会拼接相邻的字符串字面量（C/C++ 的翻译阶段 6，在预处理之后），这正是一个 PTX 模板可以跨多个源码行拆分的原因。
- **`{ ... }`。** 一个 PTX 语句块。花括号内的寄存器声明（`.reg .pred P1;`）与标签（`LAB_WAIT:`）在该块内是局部的，因此 W6 可以在十几个调用点被内联，而不会让 ptxas 抱怨符号重复（笔记 E.6.2："write the blocking wait and the election with PTX labels/registers inside `{ }` so they can be inlined many times"）。只有在模板声明了局部寄存器或标签之处，花括号才是*必需的*：W2、W6、W7、W8、W10、W13、W29、W34。W3/W4、W16-W26、W28、W31 和 W35/W36 没有花括号（W1 和 W37 是内建函数，不是 asm）。W5、W9、W11、W12、W14、W15、W27、W30、W32 和 W33 在没有局部名字的单条指令周围带有无害的花括号，因为 CUTLASS 源码就是这样写的。
- **约束字母**（笔记 E.6.2，与 nvcc 的内联 PTX 文档一致）：

| 约束 | PTX 类型 | 用途 |
|---|---|---|
| `"r"` | 32 位 `.b32`/`.u32`/`.s32` 寄存器 | 地址、计数、parity、坐标、idesc、TMEM 地址 |
| `"l"` | 64 位 `.b64` 寄存器 | tensor-map 通用地址、UMMA 描述符、缓存提示 |
| `"h"` | 16 位 `.b16` 寄存器 | 三个 multicast 掩码（W21、W30）：ISA 默认的 `ctaMask` 宽度是 16 位（`::16b`；也存在 `::32b` 形式，第 6.6.3 节） |
| `"n"` | 整数立即数 | `cp.async.bulk.wait_group.read N`（W25）：ISA 要求一个常量 |
| `"=r"` | 只写输出 | W3/W4、W7、W8、W31、W34：asm 总是会写入的值（W34 的 `x0/y0/z0` 是唯一的例外，第 6.8.3 节） |
| `"+r"` | 读写输出 | 仅 W2（第 6.2.2 节解释了原因） |

在生成的 PTX 中（实测，`trace_out_v2/ptx_explicit.txt`，笔记 G.3），kernel 声明了四个寄存器类，`.reg .pred %p<92>; .reg .b16 %rs<7>; .reg .b32 %r<1014>; .reg .b64 %rd<29>;`（ptx_explicit.txt:7-10）。`"r"` 操作数变成 `%r`（`.b32`），`"l"` 操作数变成 `%rd`（`.b64`），`"h"` 操作数变成 `%rs`（`.b16`），而 `"n"` 作为字面量被粘贴进指令文本（`cp.async.bulk.wait_group.read 1`）。`.b16` 类很小，而且可以完全对账：`%rs6` 在两处 A 加载点（:556、:658）都是 `a_mcast_mask`；`%rs3` 是常量 15 = `kMainloopReleaseMask`（`mov.b16 %rs3, 15;` 位于 :1384，被 :1387 处的 mainloop commit 操作使用）；`%rs4` 是 `acc_commit_mask`（:1415 处的累加器 commit 操作）；`%rs1`、`%rs2`、`%rs5` 是收窄到 16 位的 rank 簿记：`%rs2` 是 rank，`%rs1 = %rs2 & 1`（`rank & 1`，即 `is_leader` 的否定），而 `%rs5 = %rs1 & 255`；`setp.ne.b16` 把 `%rs1` 或 `%rs5` 与 0 作比较（:34-35、:406-407、:1112、:1430、:1516）。baseline 出于同样的原因、再加上它的 C 路径，声明了 `%rs<16>`。

- **`volatile`。** 告诉编译器该语句具有它看不见的副作用。编译器不得在其输出未被使用时删除该语句，不得把它与一条相同的语句合并（两条连续的 `elect.sync` 仍是两条指令），也不得把它外提出循环。对于一次 wait（`mbarrier.try_wait`）而言这是必不可少的：没有 `volatile`，编译器可能只对这段“纯” asm 求值一次，然后在一个陈旧的 token 上自旋。第 3 节中的每个 wrapper 都是 `volatile` 的，CUTLASS 的原版也是如此。
- **`"memory"` clobber。** 告诉编译器该 asm 可能读或写任何内存：待处理的 C++ 存储必须在它之前发射，而在它之前加载的值必须在它之后重新加载。它是一个*编译器* fence，而不是硬件的。第 3 节完全遵循 CUTLASS 的原版（笔记 E.6.2）：

| 有 `"memory"` | 无 `"memory"` |
|---|---|
| W5-W15（全部 mbarrier 操作以及两个 fence）、W18/W19（`bar.sync`/`bar.arrive`）、W20（预取）、W21-W23（TMA 加载/存储）、W25（`wait_group.read`）、W30（`tcgen05.commit`）、W32（`tcgen05.wait::ld`）、W34（CLC 解码）、W35（`griddepcontrol.wait`） | W2（`elect.sync`）、W3/W4（特殊寄存器）、W16/W17（`barrier.cluster.*`）、W24（`commit_group`）、W26-W28（TMEM alloc/dealloc/relinquish）、W29（`tcgen05.mma`）、W31（`tcgen05.ld`）、W33（`try_cancel`）、W36（`launch_dependents`） |

右列之所以安全，理由如下（我的分析，沿用 CUTLASS 作者们的选择）。W2-W4 只产生寄存器值。W29、W31 和 W33 的操作数全是寄存器（包括 W31 的十六个 `"=r"` 输出）；它们所触及的共享内存（W29、W33）或张量内存（W31）是被异步访问的，并由 mbarrier wait（W29、W33）或 W32 的 `tcgen05.wait::ld`（W31）来同步，而这些确实带有 clobber。W24 只是把先前的 bulk 操作编成一组，而发射这些操作的 asm（W23）已经带有 clobber。W36 是一个调度提示。W16/W17 提供 cluster 会合，但在它们之前的那些 `mbarrier.init` 的内存排序由 W14 承载，它带有 clobber，并且在 W16 之前跟在四个 init 组的每一组之后（例外是 explicit.cu:1165-1167 处的 `tmem_dealloc` init，它位于第四个 W14 之后且没有自己的 fence，正如 kernel :562-580 中那样；第 11.5.4 节）。唯一微妙的情形是 W26：`tcgen05.alloc` 把 TMEM 基地址写入共享内存，随后 kernel 用一次普通的 C++ 加载读取它（explicit.cu:888）。排序之所以成立，是因为 W19（`bar.arrive`，带有 clobber）位于 alloc 与该加载之间。这正是笔记 E.6.4 的论证（`mma_role` 段落：该加载 "is guaranteed by the `"memory"` clobber of the intervening `bar.arrive 6,160` (W19, barrier.h:308), which W19 must keep per E.6.2"）；W37（`__syncwarp()`，同样是一个编译器屏障）也位于那里，并且单凭它自己就足够了（这是我对笔记论证的补充）。在左列中，clobber 对 W25 最为要紧：在 `wait_group.read 1` 之后，epilogue 通过一个 C++ 指针用普通的 `st.shared`（explicit.cu:1047）覆写一个由更早的 `wait_group.read` 所释放的 stage（subtile `e + 1` 写入 stage `(e + 1) % 4`，它最后一次被组 `e - 3` 读取，第 3.5.4 节），而如果没有 clobber，W25 自身中没有任何东西能阻止编译器把这些存储外提到该 wait 之上（在本 kernel 中，explicit.cu:1071 处 W18 的带 clobber 的 `bar.sync 1, 128` 也把该 wait 与下一个 subtile 的存储分隔开，而且只有 warp 4 执行该 wait）。

### 6.0.3 族映射与你将看到的动态计数

下面引用的每个 tile 的计数针对的是固定运行中已执行路径上的一个 CTA（beta = 0，每个 tile 128 个 k-tile）。它们源自角色代码（第 7-10 章），并与笔记的清单相符（笔记 B.5、7.5-7.7）。"Pair leader" 指偶数 rank，只有它发射 MMA 和 `expect_tx`；"rank 0" 指每个 cluster 中运行 scheduler warp 的那一个 CTA。实测的量（SASS 助记符、指令计数）来自笔记 D.2 和 G.3。

---

## 6.1 第 0 族：地址桥梁，`smem_u32`

```
// cute::cast_smem_ptr_to_uint (cute/arch/util.hpp:93-108): the 32-bit shared::cta address of a generic pointer
__device__ __forceinline__ uint32_t smem_u32(void const* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
```
(explicit.cu:257-258)

**目的。** 把动态共享缓冲区的通用指针转换为本 kernel 中每一个 `shared::cta` / `shared::cluster` asm 操作数所接受的 32 位地址（PTX 也接受 64 位共享地址，ISA 6.4.1）。

**分解。** `__cvta_generic_to_shared` 是对应于 PTX `cvta.to.shared` 转换的 CUDA 内建函数：它从通用地址中减去共享窗口的基址，并返回该 CTA 的共享窗口内的偏移。到 `uint32_t` 的强制转换丢弃了高半部分，而对于 `shared::cta` 地址来说高半部分为零。这里没有 `asm`：nvcc 认识这个内建函数，并且在本 kernel 中把它折叠为 `mov.b32 %r, smem` 加上常量偏移（`trace_out_v2/ptx_explicit.txt` 中没有 `cvta.to.shared`，只有四条用于 tensor-map 地址的 `cvta.param`；笔记 B.6 对 CUTLASS kernel 观察到了同样的情况："no `cvta.to.shared` instruction occurs in the compiled kernel"）。

**调用者。** explicit.cu:1098，每线程一次：`uint32_t const smem_base = smem_u32(smem);`。kernel 中的每一个其他共享地址都由它导出：`smem_base + OFF_*`，或者把该地址的第 24 位清零以指名 pair leader 的屏障（`& kPeerBitMask`，explicit.cu:371、770）。

**这个值是什么。** 实测（笔记 D.1 第 D5 项）：`0x400 + (rank << 24)`。低 24 位在每个 CTA 中都相同（系统保留的 1 KiB 共享内存位于动态缓冲区之前）；cluster rank 位于字节 3。kernel 从不假定这一点（explicit.cu:1098 处的注释写着 "computed, never assumed"），但它的*结构*以两种方式被依赖：peer 位掩码清除第 24 位（W11，第 6.3.7 节，以及 producer 的 `full_masked_base`），而 ptxas 把 `mapa`（W10、W13）下沉为一次字节 3 插入（笔记 D.2：`PRMT ..., 0x654`）。

**陷阱。** 在期望 32 位地址之处传入一个通用指针能够编译通过（经强制转换后两者都是整数），而在运行时会引发故障或悄无声息地寻址到垃圾。第 6.0.1 节的地址空间规则就是防线：wrapper 对共享内存接受 `uint32_t`，C++ 解引用接受 `char*`。

---

## 6.2 第 1 族：warp 与 cluster 身份，W1-W4

这四个 wrapper 在 warp 与 cluster 粒度上回答“我是谁？”。W1、W3 和 W4 在 prologue 中每线程运行一次（explicit.cu:1101、1104、1173-1174），而在 tile 循环中完全不运行；W2 也在那里运行（1103），并且还在 `mbarrier.init` 组中、以及 producer、scheduler 和 MMA 循环中的每一个选举点运行（第 6.2.2 节的调用者表）。

### 6.2.1 W1 `warp_idx_sync`：warp-uniform 的 warp 索引

```
// W1 [warp]  cutlass::canonical_warp_idx_sync (cutlass.h:127-133)
__device__ __forceinline__ int warp_idx_sync() { return __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x / 32), 0); }
```
(explicit.cu:260-261)

**目的。** 以一种编译器能够证明在全部 32 个 lane 中都相同的形式，产生 warp 索引 0..7。

**分解。** `threadIdx.x / 32` 在每个 lane 中本来就相同，但编译器并不知道这一点：每个 lane 都是从自己的 `threadIdx.x` 算出它的。`__shfl_sync(0xffffffff, v, 0)` 以完整掩码把 lane 0 的值广播给所有 lane（全部 32 个 lane 都必须参与，所以 warp 必须是收敛的，而在 kernel 入口处它确实是）。ptxas 能够证明该结果是 warp-uniform 的（PTX 不携带这样的标记；SASS 在 `SHFL.IDX` 之后紧接着用 `R2UR` 把它移入 `UR20`），因此下文中每一个普通的 `if (warp_idx == k)` 都变成一个 uniform 分支（同时还测试 `lane_pred` 的那些守卫仍是 per-lane 的，11.3.1），并且该值驻留在一个 uniform 寄存器中。在 SASS 中它是一条 `SHFL.IDX`，而角色分发变成了一个三项跳转表（`BRXU`），每线程执行一次（笔记 G.3）。

该 wrapper 加上了 `static_cast<int>`，把 `NumThreadsPerWarp`（= 32，cutlass.h:96）写成字面量 `32`，并去掉了主机端的 `#else return 0;` 分支；shuffle 本身没有改变。

**调用者。** explicit.cu:1101 `int const warp_idx = warp_idx_sync();`。计数：每线程一次。

### 6.2.2 W2 `elect_one_sync`：一个 lane，确定性地

```
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
```
(explicit.cu:263-278)

**目的。** 在 warp 的恰好一个 lane 中返回 1，在其余 31 个中返回 0，使得一条 warp 范围的指令流能够只发射一次 TMA、MMA、commit 或 CLC 查询。

**分解。**

| 文本 | 含义 |
|---|---|
| `.reg .b32 %%rx; .reg .pred %%px;` | 块局部的 PTX 寄存器；`%%` 转义 PTX 寄存器名中的 `%` |
| `elect.sync %%rx\|%%px, %2;` | `elect.sync d\|p, membermask`：在 `membermask` 所指名且执行该指令的各 lane 中，恰好选出一个；`d` 在每个 lane 中都收到其 lane id，`p` 仅在被选中的 lane 中为真。`.sync` 是强制的：该指令会等待，直到掩码中的所有 lane 都到达它，因此 warp 必须是收敛的，并且每个执行它的 lane 都必须在掩码之内。ISA 保证了确定性："Election of a leader thread happens deterministically, i.e. the same leader thread is elected for the same membermask every time"，以及 "The behavior is undefined if the executing thread is not in membermask"（PTX ISA 9.4，第 9.7.15.15 节）。它没有说*哪一个* lane 会胜出，所以 kernel 不得假定是 lane 0。它可以依赖的是：在给定的 warp 中，每一次以掩码 `0xFFFFFFFF` 进行的 W2 调用都会选出同一个 lane，这正是使 `tcgen05.commit` 之前的重新选举安全的原因（第 6.7.3 节）。 |
| `%2` = `"r"(0xFFFFFFFF)` | 完整的 warp 掩码，以寄存器形式传入：ISA 需要一个 32 位操作数，而 CUTLASS 把它写成一个寄存器，因此才有笔记所观察到的 PTX `mov.b32 %r, -1`（笔记 7.2，"Operand form"） |
| `@%%px mov.s32 %1, 1;` | 带谓词：只有被选中的 lane 向 `pred` 写入 1 |
| `mov.s32 %0, %%rx;` | 每个 lane 都把胜出者的 lane id 复制到 `laneid` 中（kernel 并未使用；为与 PTX 保持一致而保留） |

**为什么对零初始化的局部变量使用 `"+r"`。** `pred` 只在被选中的 lane 中被写入。若使用只写的 `"=r"` 约束，编译器会假定该 asm 总是写这个寄存器，从而可能把 `pred = 0` 这一初始化当作死代码丢弃，让 31 个未被选中的 lane 留下一个不确定的值。`"+r"` 把该寄存器声明为既读*又*写，因此初始的 0 会被实际物化出来，并在 asm 未触及的每个 lane 中保留下来（笔记 E.6.2："the `+r` plus the zero init is what returns 0 in the non-elected lanes; `"=r"` would leave them indeterminate"）。出于与源码对称的考虑，`laneid` 得到同样的处理。

**`volatile`，不带 `"memory"`。** `volatile` 使每一次选举保持各自独立：kernel 在 MMA warp 的 k-tile 循环内连续四次调用 W2（每条 `tcgen05.mma` 一次，explicit.cu:932），若没有 `volatile`，编译器就可能把一个结果复用于全部四次。没有触及任何内存，因此没有 clobber（CUTLASS 也没有）。

**调用者**（explicit.cu，对 `elect_one_sync` 的 grep）：

| 行 | 谁 | 该选举守护的是什么 |
|---|---|---|
| 1103 | 每个线程，prologue | `lane_pred`，被复用于 tensormap 预取 (1113-1119)、producer 的 `is_leader_lane` (1192) 以及 `tmem_dealloc` 的 init (1165) |
| 1125, 1136, 1146, 1156 | warp 0、4、5、3 | 每个 `mbarrier.init` 组一次全新的选举（仿照 `initialize_barrier_array*_aligned`，barrier.h:123/134/146，它们各自都调用 `elect_one_sync()`） |
| 740 | warp 2，每个 k-tile | A + B 这一对 TMA（两次加载共用一次选举，如 `sm100_mma_warpspecialized.hpp:619-622`） |
| 843 | rank 0 的 warp 1，每个 tile | 那条 `clusterlaunchcontrol.try_cancel` |
| 932 | leader 的 warp 0，每个 k-block | 每条 `tcgen05.mma` |
| 942 | leader 的 warp 0，每个 k-tile | mainloop 的 `tcgen05.commit` |
| 944 | leader 的 warp 0，每个 tile | 累加器的 `tcgen05.commit` |

**每个 tile 每个 CTA 的次数。** warp 2：128。pair leader 的 warp 0：4 x 128 + 128 + 1 = 641（笔记 B.3.2 数出的也是同样的 641）。rank 0 的 warp 1：1。peer 的 warp 0 与各 epilogue warp：0。

**ptxas 把它变成了什么（实测）。** explicit kernel 的 PTX 有 14 处 `elect.sync` 站点。SASS 保留了 4 条 `ELECT`（prologue，`tcgen05.alloc` 的下沉结果内两条，退出序列附近一条 (0x55d0)，来源未查明；不是 relinquish，后者是 warp 集体执行的（NVIDIA 内部，第 6.7.1 节）），而在 producer、MMA 或 scheduler 循环中一条也没有：`UTMALDG`、`UTCHMMA`、`UTCBAR` 和 `UGETNEXTWORKID` 是 uniform 数据通路指令，无论 lane 数量多少都是每个 warp 发射一次，因此选举被折叠掉了（笔记 D.6 第 6 项，G.3）。PTX 形式仍然是你所书写的契约；硬件使它成为免费的。

**陷阱。** (1) 绝不要在 lane 发散的分支下调用 W2。ISA 的 `.sync` 规则是：执行线程要等待 "until all threads in the membermask execute the elect instruction"。因此，在掩码中却跳过了该指令的 lane 会让其余的一直等待（挂死），而一个*不*在掩码中却执行了它的 lane 则是未定义行为（两句话都在 9.7.15.15 中）。在本 kernel 中，每个 W2 都只处于 warp-uniform 的条件之下。(2) 不要为了“节省指令”而合并选举：SASS 表明它们已经是免费的了，而合并会改变 PTX 契约（笔记 E.6.7："one per TMA A+B pair, per MMA, per commit, per CLC query"；E.6.2 的 W2 行："one fresh call per TMA pair, per MMA, per commit, per CLC query"）。(3) 反过来，也不要通过在循环中一直携带一个 lane 谓词来“修复”这些重新选举。对于固定的掩码，选举是确定性的，因此每条指令重新选举一次是等价的，而且这正是 baseline 所使用的形式。

### 6.2.3 W3/W4 `cluster_ctarank`、`cluster_ctaid_x/y`：rank 与坐标

```
// W3/W4 [cluster]  cute::block_rank_in_cluster, cute::block_id_in_cluster (cluster_sm90.hpp:154-158, 126-132)
__device__ __forceinline__ uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r) :); return r; }
__device__ __forceinline__ uint32_t cluster_ctaid_x() { uint32_t x; asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) :); return x; }
__device__ __forceinline__ uint32_t cluster_ctaid_y() { uint32_t y; asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) :); return y; }
```
(explicit.cu:280-283)

**目的。** 读取 CTA 在 cluster 中的线性 rank，以及它在 (2,2,1) cluster 中的 (x, y) 坐标。

**分解。** `mov.u32 %0, %%cluster_ctarank;` 把一个 PTX 特殊寄存器复制到一个通用寄存器中。对于 (2,2,1) cluster，`%cluster_ctarank` 是线性 rank `x + 2y`（x 变化最快）；`%cluster_ctaid.x/.y` 是各维度的坐标。`"=r"` 输出，无输入（输出列表之后那个空的 `:`），无 clobber。CuTe 原版中的 `.z` 坐标不需要（cluster z = 1），也不被读取。

**为什么两种形式都要。** rank 驱动 pair 算术（`is_leader = (rank & 1) == 0`、`peer = rank ^ 1`，explicit.cu:1106-1107）以及屏障寻址。坐标驱动 multicast 掩码 `0x5 << x` 和 tile 的两半（`128 tm + 64 y`、`128 tn + 64 x`）。对于这个 cluster shape，`x == rank & 1` 且 `y == rank >> 1`（explicit.cu:1173-1174 处的注释）；kernel 两者都读取，因为 baseline 也是如此。

该 wrapper 把返回 dim3 的原版拆成了两个标量函数。

**调用者。** explicit.cu:1104 (`rank`)、1173 (`x`)、1174 (`y`)。次数：每个线程各一次。

**实测的下沉结果。** `%cluster_ctarank` 是 `S2UR SR_CgaCtaId`；`%cluster_ctaid.x/.y` 不是从特殊寄存器读出，而是用 `I2FP`/`FMUL`/`F2I` 从 `SR_CTAID.X/Y` 和一张驱动常量表计算得到（笔记 D.2，G.3；对该序列的解读为推断；D.6 第 14 项：kernel 中仅有的两条非 epilogue 的 `FMUL`）。

---

## 6.3 第 2 族：mbarrier，W5-W13

### 6.3.1 入门：屏障区、parity 界限与三种寻址形式

kernel 的 344 字节屏障区（偏移 0..343）容纳 43 个八字节槽位；其中 33 个在每个 CTA 中都被初始化并被使用（`full[]` 与 `acc_empty[]` 槽位仅在 pair leader 中，`clc_empty[]` 与 throttle 槽位仅在 rank 0 中：第 4.4.3、4.4.5、7.4.5、8.4.4 节），而位于 `OFF_EPI_LOAD_FULL` 128、`OFF_EPI_LOAD_EMPTY` 160 和 `OFF_LOAD_ORDER` 192 的十个槽位属于 beta != 0 路径，从不被初始化（explicit.cu:141-143；第 5 章）。

parity 的定义（第 2 章，第 2.8.3 节）限定了等待者可以询问什么：phase k 与 k + 2 具有相同的 parity。环的纪律使每个屏障领先其最慢的等待者至多一个 phase，无论环有多深，因此单个 parity 位绝不会指到错误的 phase（我的分析；第 4 章，第 4.2.3 节，“为什么 1 位的 parity 就足够”；第 4.3.3 节带着各 phase 走完 8 个 stage 的环的一整圈绕回）。

各 mbarrier wrapper 中出现了三种寻址形式：

- `.shared::cta`，配一个指向执行 CTA 自身屏障的 32 位地址（init W5、各 wait W6-W8、各 arrive W9 与 W12）；
- `.shared::cluster`，配一个由 `mapa` 产生的地址（W10、W13）："arrive on CTA `b`'s barrier at my offset"；
- `.shared::cluster`，配 peer 位掩码后的地址（W11）："arrive on the pair leader's barrier at my offset"，不使用 `mapa`。

`mbarrier.arrive... _, [addr]` 中的 `_` 是该指令本可返回的状态 token 的弃置目标；远程（`shared::cluster`）arrive 根本无法返回状态，而 kernel 也从不使用它，因此每个 arrive 都把它丢弃。

### 6.3.2 W5 `mbar_init`

```
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
```
(explicit.cu:285-294)

**目的。** 用其期望到达计数初始化一个屏障（phase 0，tx-count 0）。

**分解。** `mbarrier.init` / `.shared::cta`（地址位于执行 CTA 的窗口内）/ `.b64`（对象为 64 位）/ `[%1]` 即地址 / `%0` 即计数。计数之所以是 `%0`，是因为 CUTLASS 把 `"r"(arrive_count)` 列在最前；wrapper 保持这一顺序，使 PTX 文本完全相同。`"memory"`：屏障是后续 asm 语句要读取的内存，而这个 clobber 阻止编译器把普通的 smem 流量重排到 init 的另一侧。

CUTLASS 的 `CUTLASS_ASSERT(arrive_count != 0)` 在 NDEBUG 下是空操作，未被复制。

**调用者与计数**（explicit.cu:1128-1166，仅 prologue，执行初始化的 warp 中被选中的 lane）：

| 行 | warp | 屏障 | 计数 | 为什么是这个计数（第 4 章推导了它们；每个一行） |
|---|---|---|---|---|
| 1128-1129 | 0 | mainloop full[0..7] / empty[0..7] | 1 / 2 | leader lane 的一次 `arrive.expect_tx`；两次 `tcgen05.commit`（每个 pair leader 一次，掩码 0xF） |
| 1139-1140 | 4 | CLC full[0..1] / empty[0..1] | 1 / 800 | 每个 CTA 一次远程 `arrive.expect_tx`；cluster 范围内 32 + 4 x (32 + 32 + 128) 个消费线程 |
| 1149-1150 | 5 | 累加器 full[0..3] / empty[0..3] | 1 / 256 | leader 的 commit；2 x 128 个 epilogue 线程 |
| 1159-1160 | 3 | throttle full[0..1] / empty[0..1] | 32 / 32 | 每一侧各有一个 warp 到达 |
| 1166 | 0 (lane_pred) | tmem_dealloc | 32 | peer 的一个 warp，然后是 leader 的 |

每个 CTA 33 次 init。在 explicit kernel 中实测（笔记 G.3；`trace_out_v2/sass_explicit.txt`）：33 条 `SYNCS.EXCH.64`，带有 init 字 `{(0x100000 - count) << 1, (0x100000 - count) << 11}`，位于字节映射的各偏移处（第 5 章，第 5.7.1 节；笔记第 6.2 节）。其中 32 条是无谓词的 uniform 指令，因为 ptxas 把每个组的选举折叠进了 warp 索引分支。只有 `tmem_dealloc` 的 init 保留了一个谓词，即位于 `0x0790` 的 `@!UP0 SYNCS.EXCH.64 URZ, [UR7+0x150], UR4`（0x150 = 336 = `OFF_TMEM_DEALLOC`）：它是唯一一个由 prologue 的 `lane_pred`（explicit.cu:1165）而非由 warp-uniform 块内的一次全新选举所门控的 init。baseline 有同样的形状，43 条中有 42 条无谓词，dealloc 的 init 位于 `VOTEU.ALL UP0; @!UP0` 之后（笔记 D.2，D.6 第 8 项）。init 字中的 `0x100000` 是 2^20：ISA 把 `.layout::v0` 屏障的期望到达计数限定在 [1, 2^20 - 1] 之内（PTX ISA 9.4，9.7.15.16.12 与表 43），而 32 位的 `count` 操作数必须放得进该界限背后的硬件字段/范围（NVIDIA 内部）；实测的字把每个计数放在一个 20 位字段中这一点属于推断（ISA 称该对象是不透明的，没有给出其位 layout）。它存储的是 `2^20 - count` 而不是 `count`，这是从实测的字中读出的，而非 ISA 所记载。

**陷阱。** 各 init 必须在任何远程 arrive 之前在 cluster 范围内可见：因此四个流水线 init 组的每一个之后都有 W14（`tmem_dealloc` 的 init，explicit.cu:1165-1167，没有自己的 fence），并且在任何角色开始之前有 W16/W17 会合（第 6.4、6.5 节）。

### 6.3.3 W6 `mbar_wait`：带挂起提示的阻塞等待

```
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
```
(explicit.cu:296-311)

**目的。** 阻塞，直到屏障的具有给定 parity 的 phase 完成。

**分解。**

| 文本 | 含义 |
|---|---|
| `mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;` | `try_wait` 是*可能阻塞的*测试：如果 phase 尚未完成，线程可能被挂起，直到它完成或某个时间限制到期，随后该指令在 phase 已完成时返回 `P1 = true`，在时间限制先到期时返回 `P1 = false`。`.parity` 选择 parity 形式（操作数 `%1`）。`%2` 是可选的 **timeHint**（ISA 语法：`mbarrier.try_wait.parity{...}.b64 waitComplete, [addr], phaseParity {, timeHint};`）："The optional 32-bit unsigned integer operand timeHint specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit"（PTX ISA 9.4，9.7.15.16.19）。更早的 ISA 版本把它命名为 `suspendTimeHint`；笔记（第 7.2 节）和 explicit.cu 的注释（explicit.cu:189、296）称它为挂起时间提示。 |
| `kWaitTicks = 0x989680` (explicit.cu:189) | = 10,000,000 ns = 10 ms。CUTLASS 的注释："Arbitrarily large timer value after which try-wait expires and re-tries"（barrier.h:414）。它是一个关于硬件在报告失败之前可以把线程挂起多久的提示；随后循环会重试，因此正确性并不依赖于这个值。 |
| `@P1 bra DONE; bra LAB_WAIT; DONE:` | 重试循环，用 PTX 而非 C++ 写成，使得一条 asm 语句就是一次 wait。各标签是块局部的（第 6.0.2 节），这正是 W6 可以在每个阻塞等待站点被内联的原因。 |
| `uint32_t ticks = kWaitTicks;` | 一个局部变量，与 barrier.h:415 完全一致，使该提示成为一个寄存器操作数（PTX `mov.b32 %r, 10000000`，笔记 7.2）；SASS 把它变成了 `NANOSLEEP.SYNCS 0x989680` 的立即数 |

**为什么重试循环用 PTX 而不用 C++。** 围绕一条一次性 asm 的 C++ `while (!mbar_try_wait(addr, parity)) {}` 在语义上会是等价的。有两个理由让生产版 wrapper 保持为 PTX。(a) 忠实性：`ClusterBarrier::wait` 是一条包含该循环的 asm 语句。(b) 提示：带该提示的 C++ 原语确实存在，即 `cuda_awbarrier_primitives.h` 中的 `__mbarrier_try_wait_parity(__mbarrier_t*, bool phase_parity, uint32_t max_sleep_nanosec)`（CUDA Programming Guide 13.4，5.6.1.2；用于 4.10.3 的 "CUDA C primitives" 示例中），但它接受的是 `__mbarrier_t*` 而不是各 wrapper 所携带的 `uint32_t` 共享地址，而且 CUTLASS 是从 asm 中传递 `timeHint` 的（barrier.h:415-420）；复制那段 asm 使 PTX 文本保持完全相同。

**实测的下沉结果**（笔记 D.2，G.3）：`SYNCS.PHASECHK.TRANS64.TRYWAIT P, [addr], R`，parity 位于 `R` 的第 31 位，而重试被放到了非内联位置：`TRYWAIT; @!P NANOSLEEP.SYNCS 0x989680; @!P SYNCS.PHASECHK.TRANS64; @!P BRA`（睡眠中的 warp 如何被唤醒：第 12 章，第 12.2.2 节）。explicit kernel 总共有 30 个这样的重试循环，26 个放在非内联位置 + 4 个内联，相对于 baseline 的 47 + 5（缺少的那 22 个是 C 路径的站点）。SASS 的计数能对得上：`NANOSLEEP` 31 = 30 个重试循环 + 1 次 alloc 重试，以及 32 条普通的 `PHASECHK.TRANS64`（不带 `.TRYWAIT`）= 30 次重试路径上的阻塞检查（NVIDIA 内部；每个循环一次，在其 `NANOSLEEP` 之后）+ 2 次 `test_wait`（笔记 G.3；`NANOSLEEP` 31 在 `trace_out_v2/sass_explicit.txt` 中重新计数）。

**调用者。** 每一次阻塞等待都经由 `EXPLICIT_WAIT` 宏（explicit.cu:237，trace 开关关闭时：`mbar_wait((addr), (parity))`）；额外的宏参数会消失。站点：

| 行 | 角色 | 屏障（地址） | Parity | 何时执行 |
|---|---|---|---|---|
| 680 | 每个消费的 warp (`clc_consume`) | `OFF_CLC_FULL + 8 clc_idx` | `clc_phase` | 每个 tile 一次 |
| 735 | warp 2 | `OFF_MAIN_EMPTY + 8 prod_idx` | `prod_phase` | 仅当 W7 peek 失败时 |
| 784 | rank 0 的 warp 2 | `OFF_THROTTLE_EMPTY + 8 thr_idx` | `thr_phase` | 每个 tile 一次，无 peek |
| 800 | warp 2 尾段 | `OFF_MAIN_EMPTY + 8 prod_idx` | `prod_phase` | 退出时 8 次 |
| 836, 841 | rank 0 的 warp 1 | throttle full、CLC empty | | 每个 tile 各一次 |
| 861 | rank 0 的 warp 1 尾段 | CLC empty | | 退出时，仅当 W8 说尚未完成 |
| 918 | leader 的 warp 0 | `OFF_ACC_EMPTY + 8 acc_stage` | `accp_phase` | 每个 tile 一次，无 peek |
| 923 | leader 的 warp 0 | `OFF_MAIN_FULL + 8 cons_idx` | `cons_phase` | 仅当 W7 peek 失败时 |
| 955 | leader 的 warp 0 尾段 | 累加器 empty | | 退出时 4 次 |
| 965 | warp 0（两个 CTA） | `OFF_TMEM_DEALLOC` | 0 | 退出时一次 |
| 1034 | warp 4-7 | `OFF_ACC_FULL + 8 acc_stage` | `accc_phase` | 第一个 subtile，仅当 W7 peek 失败时 |

**每个 tile 每个 CTA 的次数。** 无条件的：6 次 CLC 等待（warp 0、2、4-7；rank 0 中为 7 次）+ 1 次累加器 empty 等待（leader）+ 3 次 throttle/CLC-producer 等待（仅 rank 0）。有条件的：mainloop 的 full/empty 与累加器 full，仅当 peek 失败时。退出时的尾段：8 + 4 + 2 + 1，其中 2 次 CLC 的仅当 W8 说尚未完成时。

**为什么先 peek 再 wait。** W6 可能长时间占住线程：它会循环，每次尝试都把线程挂起至多 10 ms 的提示时长。一次性的 W7 只测试一次并返回一个 token（第 6.3.5 节的表）。CUTLASS 在推进状态之后立即 peek *下一个* stage，此时当前 stage 的工作正在发射，并且仅当 peek 的回答是“尚未”时才进入 W6（第 8 章和第 9 章展示了这一模式）。实测：在 MMA warp 的 k-tile 循环中，当 peek 成功时 `@P0 BRA` 跳过阻塞等待；否则该等待的第一条 `TRYWAIT` 内联执行，只有其重试循环被放到非内联位置，位于 `@!P0 BRA` 之后（笔记 G.3；`trace_out_v2/sass_explicit.txt` `0x51f0-0x5230` 与 `0x6350-0x63a0`）。

### 6.3.4 W7 `mbar_try_wait`：一次性 peek

```
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
```
(explicit.cu:313-326)

**目的。** 测试一次该 phase 是否已经完成；返回一个 token（1 = 已完成，0 = 再次等待），不做自旋。

**分解。** 与 W6 相同的 `try_wait.parity`，采用其三操作数形式：没有 `timeHint`，因此硬件在返回 false 之前使用其系统相关的限制，也没有循环。`selp.b32 %0, 1, 0, P1;` 把谓词转换为 C++ 可据以分支的 32 位值（`if (!tok)`）。`"=r"` 是因为该 asm 总是写入 token。

CUTLASS 返回 `static_cast<bool>`；wrapper 则返回 `uint32_t`。

**调用者。**

| 行 | 角色 | 屏障 | 模式 |
|---|---|---|---|
| 732 | warp 2 | mainloop empty | 每次 `load_ktiles` 调用的循环前 peek (`producer_try_acquire`, sm100_mma_warpspecialized.hpp:604) |
| 739 | warp 2 | mainloop empty，下一个 stage | `advance` 之后的无条件 peek (:617) |
| 916 | leader 的 warp 0 | mainloop full | 循环前 peek，若 `k_tiles <= 0` 则跳过（`tok = 1`） |
| 927 | leader 的 warp 0 | mainloop full，下一个 stage | `advance` 之后的 peek，在最后一个 k-tile 上跳过 |
| 1030 | warp 4-7 | 累加器 full | 每个 tile 一次，在 subtile 0 处消费 |

**每个 tile 每个 CTA 的计数。** warp 2：2 + 128 = 130（消费 128 次；每次调用的最后一次 peek 被丢弃，笔记 7.5）。leader 的 warp 0：1 + 127 = 128。epilogue：每个 warp 1 次（4）。实测：`SYNCS.PHASECHK.TRANS64.TRYWAIT`，不带重试循环（baseline 见笔记 D.2，explicit kernel 见 G.3）。

### 6.3.5 W8 `mbar_test_wait`：非阻塞测试

```
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
```
(explicit.cu:328-343)

**目的。** 测试该 phase，且绝无挂起的可能；仅在单一调用点使用，即 CLC producer 尾段。

**分解。** `mbarrier.test_wait` 是 `try_wait` 的*非阻塞*同胞：它立即返回当前的完成状态，绝不挂起线程。`setp.eq.u32 P2, %3, 1; @P2 ...` 让该测试以一个 C++ 参数为谓词条件：CUTLASS 的 `test_wait(ptr, phase, pred)` 允许调用者跳过测试，而本 kernel 总是传入 `1u`。如果 `pred` 为 0，`P1` 将未被写入，`selp` 的结果也就不确定；wrapper 保留了 CUTLASS 的接口而不是修正这一点，并且那唯一的调用者从不会触及它。

**try_wait vs test_wait vs 提示，一表对照。**

| Wrapper | 指令 | 可能挂起线程？ | 循环？ | 使用处 |
|---|---|---|---|---|
| W6 | `try_wait.parity ... , hint` | 是，至多到提示值（10 ms）然后重试 | PTX 循环 | 每一次阻塞等待 |
| W7 | `try_wait.parity`（无提示） | 是，至多到系统限制，仅一次 | 否 | 决定是否调用 W6 的那些 peek |
| W8 | `test_wait.parity` | 从不 | 否 | CLC `producer_tail` (`sm100_pipeline.hpp:1041-1050`) |

实测：这两种形式在 SASS 中仍可区分：W8 对应不带 `.TRYWAIT` 的无谓词 `SYNCS.PHASECHK.TRANS64`（2 处，`0x1d70`/`0x1e50`，笔记 G.3；baseline 见笔记 D.2），而另外 30 条非 `.TRYWAIT` 的 `PHASECHK.TRANS64` 则是 W6 循环内部由 `@!P` 作谓词的重试路径阻塞检查，在 `NANOSLEEP` 退避之后发射（NVIDIA 内部；第 6.3.3 节）。

**调用者。** explicit.cu:860，rank 0 的 warp 1，`mbar_test_wait(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, 1u)`，退出时两次（`kClcStages = 2`）；仅当结果为 0 时才接着执行 W6。每个 tile 的计数：0。

### 6.3.6 W9/W10：本地与远程 arrive

```
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
```
(explicit.cu:345-367)

**目的。** W9：对执行中的 CTA 的一个屏障做一次到达。W10：对 cluster 中 CTA `cta_id` 内相同偏移处的屏障做一次到达。

**W10 的分解。** `mapa.shared::cluster.u32 d, a, b` 把 `shared::cta` 地址 `a` 映射为 rank 为 `b` 的 CTA 中相同偏移处的 `shared::cluster` 地址。随后 `mbarrier.arrive.shared::cluster.b64 _, [d]` 在那个远程屏障上 arrive。`.shared::cluster` 是使跨 CTA 地址合法的状态空间，而 `_` 丢弃状态（远程 arrive 无法返回状态）。以 `cta_id == rank` 调用 W10 是换了个名字的合法本地 arrive（dealloc 握手做的恰恰相反：`peer`）。

**被移出的 `if (pred)`。** CUTLASS 的 `arrive(ptr, cta_id, pred)` 把 asm 包在 `if (pred)` 之中（barrier.h:489）。wrapper 去掉了这个参数。本 kernel 只在 W10 应当触发的地方调用它（`is_cta0`、`!is_leader`、`is_leader`），因此对 dealloc 握手而言，该谓词是调用点处的一个 C++ 分支（`if (!is_leader)` / `if (is_leader)`，explicit.cu:964、970，它们是其 CUTLASS 原版传入非默认 `pred` 的仅有的 W10 调用，kernel :794、796）；CLC 与 throttle 的释放（explicit.cu:684、837）不带谓词，因为 CUTLASS 的 `consumer_release` 使用默认的 `pred = true`（barrier.h:383），而 `is_cta0` 只经由角色分发到达 throttle 释放处（explicit.cu:1194）。

**调用者。**

| W | 行 | 角色 | 屏障 | 目标 CTA | lane |
|---|---|---|---|---|---|
| W9 | 785 | rank 0 的 warp 2 | `OFF_THROTTLE_FULL + 8 thr_idx` | 自身 | 32（计数 32） |
| W10 | 684 | 每个消费 warp（`clc_consume`） | `OFF_CLC_EMPTY + 8 clc_idx` | 0 | 每个消费 warp 的全部 lane：32 + 4 x (32 + 32 + 128) = 800，整个 cluster 范围内 |
| W10 | 837 | rank 0 的 warp 1 | `OFF_THROTTLE_EMPTY + 8 thc_idx` | 0 | 32 |
| W10 | 964 | peer 的 warp 0，退出 | `OFF_TMEM_DEALLOC` | `peer`（= leader） | 32 |
| W10 | 970 | leader 的 warp 0，退出 | `OFF_TMEM_DEALLOC` | `peer`（= peer） | 32 |

CLC 释放总是以 CTA 0 为目标，因为 CLC 流水线的 `consumer_release` 在 `params_.producer_blockid`（sm100_pipeline.hpp:1132-1134）上 arrive，而它为 0：只有 rank 0 的 scheduler 产生响应，也只有它的 `clc_empty[]` 曾被等待（第 7 章）。

**每个 tile 每个 CTA 的计数。** W9：32 次线程到达（一次 warp 范围的调用，仅 rank 0）。W10：每个 CTA 192 次线程到达（6 个 warp x 32），rank 0 中为 224 次（scheduler warp 也参与消费），外加 rank 0 中 throttle 释放的 32 次；退出时每个 pair 32 + 32 次。实测的下沉结果：`mapa` 不产生自己的指令，只有一条做字节 3 插入的 `PRMT ..., 0x654`，这证实了 rank 位于地址位 [24,32)；远程形式的 arrive 是 `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [addr], RZ`，本地形式则是 `SYNCS.ARRIVE.TRANS64.A1T0`（笔记 D.2）。

### 6.3.7 W11 `mbar_arrive_cluster_masked`：peer 位 arrive

```
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
```
(explicit.cu:369-379)

**peer 位机制。** `kPeerBitMask = 0xFEFFFFFF` (explicit.cu:187; `cute::Sm100MmaPeerBitMask`, copy_sm100_tma.hpp:45) 清除第 24 位，它之所以是 rank 的第 0 位，是因为 rank `r` 的基址是 `0x400 + (r << 24)`（实测，笔记 D.1 D5）。清除该位把 rank 1 映射到 rank 0、把 rank 3 映射到 rank 2，并使偶数 rank 保持不变：

```
  rank 0  0x00000400 + off  --&0xFEFFFFFF-->  0x00000400 + off   (itself)
  rank 1  0x01000400 + off  --&0xFEFFFFFF-->  0x00000400 + off   (its leader, rank 0)
  rank 2  0x02000400 + off  --&0xFEFFFFFF-->  0x02000400 + off   (itself)
  rank 3  0x03000400 + off  --&0xFEFFFFFF-->  0x02000400 + off   (its leader, rank 2)
```

**nvcc 发出了什么。** PTX 中带有 `and.b32 %r, %r, -16777224` = `0xFEFFFFF8`，SASS 中则是 `ULOP3.LUT ... 0xfefffff8`（在 baseline 的 epilogue 中是一条向量 `LOP3.LUT`）：8 字节对齐的屏障地址中三个恒为零的对齐位被折入掩码，语义上完全相同（笔记 D.6 第 2 项、G.3）。

**每个 tile 每个 CTA 的计数。** 在 explicit kernel 中实测（`trace_out_v2/sass_explicit.txt`）：`0x44c0` 处的 `ULOP3.LUT UR24, UR24, 0xfefffff8, URZ, 0xc0, !UPT`（唯一幸存的 `0xfefffff8` 字面量，笔记 G.3），随后是 `0x44f0` 处的 `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [UR24], RZ`，位于 subtile 7 的 `LDTM.x16` 与其第一条 `FMUL` 之间。被掩码后的地址是 warp-uniform 的（`acc_empty_base + 8 acc_stage` 不含 lane 项），因此它存放在一个 uniform 寄存器中。baseline 中的对应物是 `0xb090` 处的 `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R62+URZ], RZ`，其 `R62 = (base + 0x110 + 8 acc_stage) & 0xFEFFFFF8`（笔记 D.2；0x110 = 272 = `OFF_ACC_EMPTY`）。

### 6.3.8 W12/W13：带事务字节的 arrive

```
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
```
(explicit.cu:381-405)

**分解。** 在 W12 中字节数是 `%0`，地址是 `%1`（CUTLASS 的操作数顺序，予以保留）。W13 的 `setp.eq.u32 p, %2, 1` 把 C++ 的 `pred` 变成一个 PTX 谓词，因此**全部 32 个 lane 执行同一条 asm 语句**，只有被谓词选中的 lane 才动作，而没有发散的 C++ 分支。

**究竟是谁。** W12：`if (is_leader_lane)`（explicit.cu:736），其中 `is_leader_lane` 是实参 `lane_pred != 0u && is_leader`（explicit.cu:1192）：pair leader 的 warp 2 中在 prologue 被选中的那个 lane，每个 k-tile 一次，在加载发射之前。W13：rank 0 的 warp 1，全部 32 个 lane 都以 `cta_id = lane` 和 `pred = lane < 4` 执行它（explicit.cu:842）：lane 0..3 各以 16 字节装填 CTA 0..3 的 `clc_full[idx]`（计数 1，explicit.cu:183），每个 CTA 一个 lane（笔记 13 第 6 项）。

**每个 tile 每个 CTA 的计数。** 在 explicit kernel 中实测：W12 是 `@!P0 SYNCS.ARRIVE.TRANS64 RZ, [UR10], R5`，其 `R5 = 0xc000`（= 49152），出现在两个 producer 循环中（`0x0ee0`、`0x11b0`；笔记 G.3，`trace_out_v2/sass_explicit.txt`）。baseline 的形式是 `@P0 SYNCS.ARRIVE.TRANS64 RZ, [UR17], R2`（笔记 D.2），对同一个 leader-lane 条件采用相反的谓词极性。W13 是 `0x1b70` 处的 `@!P0 SYNCS.ARRIVE.TRANS64.RED RZ, [R11+URZ], R5`，其 `R5 = 0x10`（`0x1ad0` 处的 `@!P0 MOV R5, 0x10`；`R11` 是经 `PRMT ..., 0x654` 按 lane 映射的地址；`trace_out_v2/sass_explicit.txt`，第 12.2 节）；baseline 的形式是 `0x2cd0` 处的 `@!P2 SYNCS.ARRIVE.TRANS64.RED RZ, [PRMT(lane)], R9`，其 `R9 = 0x10`（笔记 D.2）。

**为什么字节可能在 `expect_tx` 之前落地，以及为什么这没有问题。** tx-count 是有符号的，对 `.layout::v0` 而言范围为 -(2^20 - 1) .. 2^20 - 1（PTX ISA 9.4，表 43），因此早到的 `complete_tx` 操作会把它驱至负值，直到 `expect_tx 49152` 把它拉回零。因而“发射之前先装填”是一条线程内的约定（leader 自己的加载跟在它的 arrive 之后，正如 CUTLASS 的 `producer_acquire` 那样），而不是跨 CTA 的要求。第 4 章（“为什么无论谁先落地都行得通”）算了这些数字，包括一个两次到达的变体。

---

## 6.4 第 3 族：fence，W14-W15

### 6.4.1 W14 `fence_mbarrier_init_release_cluster`

```
// W14 [fence]  cutlass::arch::fence_barrier_init (barrier.h:712-720)
__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
  asm volatile(
    "{\n\t"
    "fence.mbarrier_init.release.cluster; \n"
    "}"
    ::
    : "memory");
}
```
(explicit.cu:407-415)

**分解。** `fence.mbarrier_init` 是一种受限的 fence，只对先前的 `mbarrier.init` 操作排序，效果上比完整的 `fence.acq_rel.cluster` 更窄；`.release.cluster` 是 ISA 允许它使用的唯一语义与作用域。`::`（空的输出与输入）后接 `"memory"` 就是只含 clobber 的操作数列表。CUTLASS 的注释（barrier.h:707-710）说它 "must be composed with an appropriate sync instruction with the right scope ... eg. `__syncthreads()` or a `cluster_arrive()` + `cluster_wait()`"：W14 是 release 的一半，W17 是 acquire 的一半。

**按 ISA 的规定，作用域是每线程的。** 只有那些做初始化的 warp 中被选中的 lane 执行了初始化，因此只有它们的 W14 会对任何东西排序。其他线程的 fence（第一组中 256 个里的 255 个）是无害的空操作，这与 ptxas 不为这条 PTX 语句发出任何指令相一致。

**调用者。** explicit.cu:1133、1144、1154、1164：每个线程，在四个 init 组中每一组之后各一次。CUTLASS 在每个由 mbarrier 支撑的流水线构造函数中调用它一次，因此有 33 次 init 的 explicit kernel 有 4 个 fence，而有 43 次 init 的 baseline 有 6 个（笔记 E.6.3）。计数：每个线程 4 次，仅在 prologue。实测：它不发出自己的 SASS 指令；在 baseline 中有七条裸的 `FENCE.VIEW.ASYNC.S` 位于各 init 组之前，而这一归因是推断的（笔记 D.2、D.6 第 8 项）。

### 6.4.2 W15 `fence_proxy_async_shared_cta`

```
// W15 [fence]  cutlass::arch::fence_view_async_shared (barrier.h:728-736)
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile (
    "{\n\t"
    "fence.proxy.async.shared::cta; \n"
    "}"
    ::
    : "memory");
}
```
(explicit.cu:417-425)

**目的。** 使通用 proxy 对共享内存的访问排在随后异步 proxy 对同一 CTA 共享内存的访问之前（反之亦然）。

**分解。** `fence.proxy.async` 是通用与异步 proxy 之间的双向 proxy fence。`.shared::cta` 把它限制在 CTA 共享内存中的对象上（其他形式是 `.global`、`.shared::cluster`，或不带状态空间以覆盖全部）。`"memory"` 在这里必不可少：这个 fence 的全部用途就是位于 C++ 的 `st.shared` 存储*之后*（explicit.cu:1047），而 clobber 禁止编译器把那些存储下沉到它之下。CUTLASS 有一个不带 clobber 的孪生版本 `cute::tma_store_fence`（copy_sm90_tma.hpp:1214-1221：`asm volatile ("fence.proxy.async.shared::cta;");`）；本 kernel 刻意照搬了带 clobber 的 `barrier.h` 形式（笔记 E.6.2 W15）。

**调用者。**

| 行 | 角色 | 为什么 |
|---|---|---|
| 683 | 每个消费 warp（`clc_consume`），在 W34 之后 | 响应槽位刚被通用 proxy 读取（`ld.shared.b128`），并将在释放之后被异步 proxy 重写（两次查询之后进入此槽位的下一次 `try_cancel`）；CUTLASS 在同一位置设置 fence（sm100_tile_scheduler.hpp:430）。ISA 给出了首要原因：响应 "is asynchronously written using weak async-proxy writes to the corresponding local shared memory addr of each CTA in the requesting cluster"（PTX ISA 9.4，9.7.15.18）。因此，那次较晚的查询对此槽位的写入是一次异步 proxy 写入，必须排在这次通用 proxy 的 `ld.shared.b128` 之后，而该 fence 建立了跨 proxy 的这一读后写次序（第 7 章追踪了该槽位的两轮复用）。 |
| 1050 | warp 4-7，每个 subtile，在 16 次 `st.shared` 之后 | TMA 存储（W23，异步 proxy）必须观察到通用 proxy 写入 C/D stage 的那些存储 |

**每个 tile 每个 CTA 的计数。** 每个消费 warp 一次（6；rank 0 中为 7）+ 每个 epilogue warp 8 次（32）。实测：`MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S` 成对出现（explicit kernel 的 SASS 中有 12 对，笔记 G.3）。

---

## 6.5 第 4 族：cluster 屏障与命名屏障，W16-W19

**两种不是 mbarrier 的屏障。** *cluster 屏障*（`barrier.cluster.*`）是 cluster 中所有 CTA 的所有线程的一次硬件会合：没有地址、没有计数操作数、没有事务字节；每个线程每个 phase 到达一次。*命名屏障*（`bar.sync` / `bar.arrive`，ISA 中的 `barrier{.cta}`）是 16 个 per-CTA 硬件屏障 0..15 之一。它以*线程*计数（32 的倍数），没有事务字节、没有跨 CTA 寻址、也没有需要跟踪的 parity，当规定数量的线程到达后它即完成，然后自行复位。两者都不位于共享内存中，所以两者都不需要 init 或 fence。epilogue 使用命名屏障 1 而不是 mbarrier，因为它需要的正是一个 CTA 的 128 个 epilogue 线程的一次线程计数会合，每个 subtile 两次，没有要计数的异步字节，也没有任何远程的东西。一个命名屏障用一条指令就给出了这一点，smem 中没有状态、没有 phase 簿记、也没有 arrive/wait 的拆分（第 2 章，第 2.10.2 节引用了 ISA 的规则）。

### 6.5.1 W16/W17 `cluster_arrive_relaxed`、`cluster_wait`

```
// W16/W17 [cluster]  cute::cluster_arrive_relaxed, cute::cluster_wait (cluster_sm90.hpp:48-51, 66-69)
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : : ); }
__device__ __forceinline__ void cluster_wait()           { asm volatile("barrier.cluster.wait.aligned;\n" : : ); }
```
(explicit.cu:427-429)

**目的。** prologue 的 cluster 范围的会合，拆成它的两半，以便能在两者之间做工作。

**分解。**

| 限定符 | 含义 |
|---|---|
| `barrier.cluster.arrive` | 标记本线程在 cluster 屏障处的到达，而不等待 |
| `.relaxed` | 该 arrive 不附带任何内存排序（默认会是 `.release`）；各 init 的排序已经由 W14 提供，所以这里的 release 会是多余的工作 |
| `.aligned` | warp 的所有线程执行同一条 `barrier.cluster` 指令；该 warp 必须是收敛的（它确实是：两半都由全部 256 个线程无条件执行，位于 prologue 中被选中的 lane 的 `if` 块之外） |
| `barrier.cluster.wait` | 阻塞，直到 cluster 中每个 CTA 的每个未退出线程都已到达；默认的 `.acquire` 语义使已释放的各 init 可见 |

（PTX ISA 9.4，9.7.15.3，`barrier.cluster`："If the optional .acquire qualifier is absent for barrier.cluster.wait, .acquire is assumed by default"，而对于 arrive，除非写了 `.relaxed`，否则 ".release is assumed by default"。）

**为什么拆分。** 在 explicit.cu:1170（arrive）与 :1179（wait）之间，kernel 计算 `x`、`y`、`tm`、`tn`（W4 和纯算术，不涉及共享内存）。kernel 因此把最慢的那个 CTA 的到达延迟与有用的工作重叠起来，与 baseline 的做法完全一样（kernel :580 与 :615）。任何远程屏障操作都不得先于该 wait（explicit.cu:1178 处的注释），因为某个 peer CTA 可能尚未初始化它的屏障。

**没有 `"memory"` clobber。** CuTe 的原版没有它（`: : );`）。这是安全的：各 init 自带 clobber，W14 的 clobber 位于各流水线 init 与 W16 之间（第 6.0.2 节；只有 tmem_dealloc init，explicit.cu:1166，跟在最后一个 W14 之后，与 baseline 一样，kernel :565 在 :580 之前），而且 W17 之前不会发生任何 C++ 共享内存访问。

**调用者。** explicit.cu:1170 与 1179，每个线程一次。实测：`UCGABAR_ARV` 与 `UCGABAR_WAIT; CCTL.IVALL`，两者都处在同一个由 ptxas 插入的 `c[0x0][0x36c] == 1` 测试之后（把该常量解读为“以 cluster 方式启动”是推断，笔记 D.2），该测试的不跳转路径对 arrive 是一条 `NOP`，对 wait 是 `BAR.SYNC.DEFER_BLOCKING 0x0`（笔记 D.2，G.3）。

### 6.5.2 W18/W19 `named_bar_sync`、`named_bar_arrive`

```
// W18/W19 [named barrier]  NamedBarrier::arrive_and_wait_internal / arrive_internal (barrier.h:285-288, 305-309)
__device__ __forceinline__ void named_bar_sync(uint32_t barrier_id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(uint32_t barrier_id, uint32_t num_threads) {
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads) : "memory");
}
```
(explicit.cu:431-437)

**目的。** 通过 16 个硬件命名屏障之一，同步 CTA 的 warp 的一个*子集*。

**分解。** `bar.sync a, b`：屏障 id `a`（0..15），`b` = 参与线程数（32 的倍数）；该线程到达并等待，直到 `b` 个线程已到达。`bar.arrive a, b`：到达而不等待；当发生了 `b` 次任一种类的到达时，屏障完成。两者都是 warp 范围的。屏障 0 归编译器所有：编译器通常把 `__syncthreads()` 下沉为一个 CTA 屏障，一般是屏障 0，所以手写的 `bar.sync 0` 协议不得与 `__syncthreads()` 在同一区域中混用（NVIDIA 内部）；第 2.10.2 节引用过的 barrier.h:163-164（其他 driver API "may use it"）出于同样的原因警告不要使用 id 0，因此 CUTLASS 保留了 id 1-7（`ReservedNamedBarriers`，barrier.h:169-178）：`EpilogueBarrier = 1`、`TmemAllocBarrier = 6`（explicit.cu:190）。这些 id 和计数是 `"r"` 寄存器，因为 CUTLASS 以运行时值传递它们；ptxas 仍然在 SASS 中把它们折叠为立即数（`BAR.SYNC.DEFER_BLOCKING 0x1, 0x80`）。

**两个群体。**

| 调用 | Id | 线程数 | 谁 | 为什么是这个数 |
|---|---|---|---|---|
| explicit.cu:887 `named_bar_arrive(6, 160)` | 6 | 160 = 32 + 128 | warp 0 在 `tcgen05.alloc` 写入 TMEM 基址之后到达 | 只有需要 TMEM 地址的 warp 参与：warp 0（该值的 producer）+ warp 4-7（consumer）；warp 1-3 从不读取 T |
| explicit.cu:997 `named_bar_sync(6, 160)` | 6 | 160 | warp 4-7 在 `epilogue_role` 的开头等待 | 在它之后，`T = *(smem + OFF_TMEM_BASE_PTR)` 是安全的（explicit.cu:998） |
| explicit.cu:1051, 1071 `named_bar_sync(1, 128)` | 1 | 128 = 4 x 32 | warp 4-7，每个 subtile 两次 | (1) 在 fence 之后：全部 128 个线程对 C/D stage 的存储都已就位，然后 warp 4 才发射 TMA 存储；(2) 在存储/commit/wait 之后：在 warp 4 的 `wait_group.read 1` 返回之前，即在除刚提交的那一组之外的每个存储组都已读完之前，没有人开始下一个 subtile 的 `st.shared`（接下来要写的 stage，`(e + 1) % 4`，上一次是被 subtile e - 3 的那一组读取的，第 3.5.4 节） |

两者都带 `"memory"`（barrier.h:287，:308）：该屏障正是 C++ smem 存储必须完成的位置（在 `bar.sync 1,128` 之前是 epilogue 的 `st.shared`；在 `bar.arrive 6,160` 之后，对 T 的加载不得被外提；笔记 E.6.4，`mma_role` 段落）。

**每个 tile 每个 CTA 的计数。** W18：16 次（`bar.sync 1,128`，每个 subtile 2 次）；外加开始时每个 CTA 1 次 `bar.sync 6,160`。W19：开始时每个 CTA 1 次。实测：16 条 `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80`、1 条 `0x6, 0xa0`、1 条 `BAR.ARV 0x6, 0xa0`（笔记 G.3）。

---

## 6.6 第 5 族：TMA，W20-W25

### 6.6.1 入门：TMA 单元，以及 map 位于何处

TMA 单元异步地完成地址生成、swizzle 和传输。全部四个 map 地址都位于 kernel 参数空间（`__grid_constant__ params`）中，而这些指令接受每个 map 的通用地址（`"l"`）。

### 6.6.2 W20 `prefetch_tensormap`

```
// W20 [TMA]  cute::prefetch_tma_descriptor (copy_sm90_desc.hpp:302-317): generic address of the parameter-space descriptor
__device__ __forceinline__ void prefetch_tensormap(void const* desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
}
```
(explicit.cu:439-447)

**目的。** 把一个 tensor map 的缓存行带入 `cp.async.bulk.tensor` 从中读取它的那个 tensor-map 描述符缓存，而不是 L2 数据缓存（NVIDIA 内部；ISA 没有指名任何缓存），赶在第一条使用它的 `cp.async.bulk.tensor` 之前。

**分解。** `prefetch{.tensormap_space}.tensormap [a]`：带 `.tensormap` 时，该预取把存放某个 tensor map 的缓存行带入 `cp.async.bulk.tensor` 所用的 tensor-map 描述符缓存，而不是 L2 数据缓存（NVIDIA 内部；ISA 没有指名任何缓存）。该预取是每线程的、尽力而为的，并且没有完成机制（NVIDIA 内部）：kernel 发射它，从不等待它。可选的状态空间限定符是 `.const` 或 `.param`；**不带它时地址就是通用地址**（CUTLASS 的注释，copy_sm90_desc.hpp:308："using generic addressing (i.e. no specific state space: const or param)"）。这就是为什么操作数是一个存放 `reinterpret_cast<uint64_t>(&params.tma_a)` 的 64 位 `"l"` 寄存器：nvcc 发出 `cvta.param` 来构成该参数空间对象的通用地址（笔记 G.3："four `cvta.param` for the map addresses"）。对齐要求并不是 ISA 的。PTX 只把 tensor map 描述为 "a 128-byte opaque object either in .const space or .param (kernel function parameter) space or .global space"（PTX ISA 9.4，5.5.8），并把 `prefetch` 的操作数引向通用寻址规则。64 字节对齐来自 CUDA driver API（`cuTensorMapEncodeTiled`："tensorMap address must be aligned to 64 bytes"，CUDA Driver API，Tensor Map Object Management）。CUTLASS 的后备类型把它明写为 `struct alignas(64) { char bytes[128]; }`（copy_sm90_desc.hpp:295），而 CUDA 13.3 的头文件以 `alignof(CUtensorMap) = 128` 超额满足了它（explicit.hpp:18，:66）。`"memory"` 与源码相同。

该 wrapper 接受 `void const*` 而不是 `TmaDescriptor const*`。

**调用者。** explicit.cu:1114-1115（warp 1，`lane_pred`：`&params.tma_a`、`&params.tma_b`）与 1118-1119（warp 3，`lane_pred`：`&params.tma_c`、`&params.tma_d`），在每个 CTA 中，不论 beta 为何（预取 C 是为了与 baseline 的四条 `UTMACCTL.PF` 保持一致，笔记 D.7）。计数：每个 CTA 4 次，仅在 prologue 中。在 explicit kernel 中实测：`UTMACCTL.PF` x4（笔记 G.3）。参数块的地址（其通用地址，按第 12.1.1 节的推断）在每个执行预取的 warp 中加载一次（warp 1 的路径上是 `0x00d0` 处的 `LDCU.64 UR4, c[0x0][0x348]`，warp 3 的路径上是 `0x0130` 处的 `LDCU.64 UR6, c[0x0][0x348]`），四个 map 则在距它 `+0x0`、`+0x80`、`+0x100`、`+0x180` 处被预取（`trace_out_v2/sass_explicit.txt`，`0x00d0-0x0190`）= 字节 map 的 0/128/256/384（explicit.hpp:20-23）。这些描述符留在参数空间中，那里的 640 字节块位于常量 bank 偏移 0x380 处（`CONSTANT[0] = 0x380 + 640`，G.3）。笔记 D.2 中的 `+0x80/+0x180/+0x580/+0x680` 是 baseline 的 map 在其 2048 字节 `Params` 内的偏移。

### 6.6.3 W21 `tma_load_2sm_mcast_3d`：A 加载

```
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
```
(explicit.cu:449-460)

**目的。** 把 A 的一个 64 x 64 fp16 box（8192 B）从全局内存加载到掩码中点名的每个 CTA 的同一共享内存偏移处，并记入目的 pair 的 leader 屏障。

**分解，逐个限定符。**

| 文本 | 在本 kernel 中的含义 |
|---|---|
| `cp.async.bulk.tensor` | 一次由 TMA 单元执行的异步 bulk 张量复制，通过一个 tensor map 寻址 |
| `.3d` | 三个坐标：`{64 q, 128 tm + 64 y, 0}` = (k, row, batch)，对应维度为 (K, M, L) 的 A map（第 5 章）。第三个坐标是字面量 0，因为 L = 1：explicit kernel 按构造就是一个 L = 1 的 kernel（笔记 G.3，"Generality"）。baseline 把它绑定到 `blockIdx.z`，继而绑定到 CLC 响应的 `z0`（笔记 D.2，`UR20` 操作数）；把它变成一个变量是批量版本的第一处差异（第 13 章，第 13.4.2 节） |
| `.cta_group::2` | 该指令参与一个 CTA pair。ISA（PTX ISA 9.4，9.7.10.28.5.3）："When .cta_group::2 is specified, the mbarrier object mbar can be in shared memory of either the same CTA as the shared memory destination dstMem or in its peer-CTA"，而带 multicast 时 "for each destination CTA specified in the ctaMask, the mbarrier signal is sent either to the destination CTA or its peer-CTA based on the %cluster_ctarank parity of the shared memory where the mbarrier object mbar resides"。一个 32 位 `shared::cluster` 地址的 rank parity 是它的 rank 的第 0 位，编译器把它放在 **第 24 位**（实测，笔记 D.1 D5；不是 ISA 原文）。因此，用 `Sm100MmaPeerBitMask` 清除第 24 位 会把每个目的地的信号都发送到 **其 pair 中的偶数 CTA**：配合调用者的掩码，落在一个 pair 的任一 CTA 中的字节总是计入该 pair 的 leader（笔记 B.6；49152 字节的核算，笔记 D.1 D8）。 |
| `.shared::cluster.global` | 目的状态空间 `shared::cluster`（32 位目的地址在 cluster 窗口中解释；带 multicast 时，是每个目的 CTA 中与 `dstMem` 相同的 CTA 相对偏移），源 `global` |
| `.mbarrier::complete_tx::bytes` | 完成机制：当 box 落地后，以所写入的字节数（8192）在该 mbarrier 上执行 `complete_tx` |
| `.multicast::cluster` | 该 box 被写入 `ctaMask`（`%3`、`"h"`）中置位的每个 CTA，"to the same offset as dstMem in the shared memory of each destination CTA"；每个目的地的 complete_tx 发往如上所选的屏障。"By default, ctaMask is assumed to be of 16-bit width"（`::16b`）。ISA 9.4 还提供了带 32 位掩码的 `::32b`，CUTLASS 和一个 4-CTA 的 cluster 都不需要它 |
| `.L2::cache_hint` | 后面跟一个 64 位的缓存策略操作数（`%7`、`"l"`）：`kCacheHintEvictNormal = 0x1000000000000000`（explicit.cu:188）= `TMA::CacheHintSm100::EVICT_NORMAL`（copy_sm90_desc.hpp:193-194）= `cute::Sm100MemDescDefault`（copy_sm100_tma.hpp:46），即默认策略 |
| `[%0]` | 目的 smem 地址：`a_dst_base + 16384 s`，其中 `a_dst_base = smem_base + OFF_SMEM_A + 8192 y`（explicit.cu:768）：本 CTA 在 128 x 64 stage 中的 64 行半块 |
| `[%1, {%4, %5, %6}]` | tensor map 的通用地址以及各坐标（`int32_t`、`"r"`） |
| `[%2]` | 该 mbarrier：`full_masked_base + 8 s`，其中 `full_masked_base = smem_base & kPeerBitMask`（explicit.cu:770）：pair leader 的 `full[s]` |
| `%3` | `a_mcast_mask = 0x5 << x`（explicit.cu:772）：rank x 与 x + 2，即共享 `tm` 的那两个 CTA |

**为什么由调用者预先给屏障加掩码。** CUTLASS 的 `SM100_TMA_2SM_LOAD_MULTICAST_3D::copy` 在 copy 内部加掩码（copy_sm100_tma.hpp:292：`cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask`，附有注释 "Executed by both CTAs. Set peer bit to 0 so that the transaction bytes will update CTA0's barrier"）。kernel 则把 `smem_base & kPeerBitMask` 外提到循环之外，作为 `full_masked_base`（explicit.cu:770，在 :776 处 tile 循环的 `do {` 之前），并按每个 k-tile 加上 `8 s`，所以 wrapper 收到的是已经加过掩码的 32 位地址。PTX 指令完全相同，而掩码是每个 CTA 施加一次而不是每个 tile 128 次；这就是笔记（G.3）所观察到的、在 SASS `0x0c00` 处折叠成 `0xfefffff0` 的那一处，它位于 warp 2 角色分发之后的循环前设置中。

**49152 字节的核算，算例（pair {0, 1}，stage s）。**

```
  issuer  mask       lands in            offset in stage      credited barrier
  rank 0  0x5 {0,2}  rank 0 and rank 2   +0     (y = 0 half)  rank 0 full[s] (own) and rank 2 full[s] (own)
  rank 1  0xA {1,3}  rank 1 and rank 3   +0                   rank 0 full[s] (bit 24 cleared) and rank 2 full[s]
  rank 2  0x5 {0,2}  rank 0 and rank 2   +8192 (y = 1 half)   rank 0 full[s] and rank 2 full[s]
  rank 3  0xA {1,3}  rank 1 and rank 3   +8192                rank 0 full[s] and rank 2 full[s]
  B loads (W22, no multicast): rank 0 -> rank 0 full[s]; rank 1 -> rank 0 full[s]
  rank 0 full[s] receives: 4 x 8192 (A) + 2 x 8192 (B) = 49152 = kMainloopTxBytes
```

每个 CTA 最终都拥有完整的 128 x 64 A stage（它自己的一半，加上由 `x` 相同而 `y` 不同的那个 CTA multicast 过来的一半）以及它自己的 64 x 64 B 半块；2SM MMA（W29）读取两个 CTA 的 stage（第 9 章）。

**约束。** `"h"(multicast_mask)` 接受该 `uint16_t`：每个 `%cluster_ctarank` 一位，最多 16 个 CTA（ISA 的 16 位默认形式与未使用的 `::32b` 形式见上面的 `.multicast::cluster` 行）。在 PTX 中它是一个 `.b16` 寄存器，在 SASS 中则是一个由 `UPRMT ..., 0x5410` 放到位 [16,32) 中的 uniform 寄存器（笔记 D.2）。`"memory"` 与源码相同。

该 wrapper 接受 `uint32_t` 地址（第 6.0.1 节）并按值接受各坐标；唯一的其他改动是屏障加掩码移到了调用者处。

**调用者。** explicit.cu:747-748，warp 2，位于 `if (elect_one_sync())` 内，每个 k-tile 一次：`tma_load_2sm_mcast_3d(&params.tma_a, full_masked_base + 8u * s, a_mcast_mask, kCacheHintEvictNormal, a_dst_base + kStageBytesA * s, c0, a_row, 0)`，其中 `c0 = 64 q`、`a_row = 128 tm + 64 y`。每个 tile 每个 CTA 的计数：128。实测：`UTMALDG.3D.MULTICAST.2CTA [UR16], [URtmap], URmask, desc[UR]`，其中 `desc = {0x0, 0x10000000}`（缓存提示），无谓词（笔记 D.2，G.3）。

### 6.6.4 W22 `tma_load_2sm_3d`：B 加载

```
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
```
(explicit.cu:462-473)

**目的。** 把本 CTA 的 B 的 64 x 64 fp16 半块（8192 B）加载到它自己的 stage 中，并将其计入 pair leader 的 `full[s]`。

**分解。** 与 W21 相同，只是少了 `.multicast::cluster` 与掩码操作数，因此操作数编号发生移位（`{%3, %4, %5}` 为坐标，`%6` 为提示）。B 不做 multicast，因为一个 pair 的两个 CTA 需要的是 B 的*不同的* 64 列半块（`128 tn + 64 x`），而 cluster 的两个 pair 有不同的 `tn`：没有任何东西被共享，所以没有任何东西被 multicast。仍然需要 `.cta_group::2`，以便 peer 的字节计入 leader 的屏障。

与 W21 一样，掩码处理移到了调用者处。

**调用者。** explicit.cu:754-755，与 W21 相同的选举：`tma_load_2sm_3d(&params.tma_b, full_masked_base + 8u * s, kCacheHintEvictNormal, b_dst_base + kStageBytesB * s, c0, b_col, 0)`，其中 `b_dst_base = smem_base + OFF_SMEM_B`（explicit.cu:769），`b_col = 128 tn + 64 x`。每个 tile 每个 CTA 的计数：128。实测：`UTMALDG.3D.2CTA`。

### 6.6.5 W23 `tma_store_3d`：D 存储

```
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
```
(explicit.cu:475-484)

**目的。** 把一个 (32 m, 16 n) 的 fp32 box（2048 B）从 swizzled 的 C/D smem stage 存储到全局内存中的 D。

**分解。** `.global.shared::cta`：目的地为全局内存，源为执行 CTA 的共享内存（一次存储从不跨越 CTA）。`.bulk_group` 是存储的完成机制：该操作保持未提交状态，直到 W24 把它与发射线程的其他未提交存储一起归入一个新的、由 W25 等待的每线程 bulk async-group；这里没有 mbarrier。操作数：`[%0, {%2, %3, %4}]` 是 D map 与坐标 `(128 tm + 32 i, 128 tn + 16 e, 0)` = (m, n, batch)，针对 (M, N, L) map（batch 坐标是字面量 0，因为 L = 1，第 6.6.3 节）；`[%1]` 是源 `src_base + 2048 i`，其中 `src_base = cd_base + 8192 (e & 3)`。这是一条 Hopper 时代的指令（没有 `.cta_group`，也没有传入缓存提示，尽管存储形式接受 `.L2::cache_hint`）：epilogue 是 per-CTA 的。

CUTLASS 的 `synclog_emit_tma_store` 在此构建中是空操作，未被复制。

**调用者。** explicit.cu:1066，仅 warp 4（`issue_tma_store = (w == 0u)`，explicit.cu:1018），**全部 32 个 lane，带 uniform 的操作数且无选举**，每个 subtile 4 个 box（`for i in 0..3`）。每个 tile 每个 CTA 的计数：32 (4 x 8)。

**PTX 语义与下沉的对比，以及为什么不需要选举。** 在 PTX 层面，`cp.async.bulk.tensor` 是一条每线程的指令：warp 4 的 32 个 lane 中的每一个名义上都发射自己的存储，并在自己的 bulk async-group 上维护自己的 `commit_group` / `wait_group.read` 簿记。ptxas 能够证明每个操作数都是 warp-uniform 的（来自参数空间的 map 地址、`src_base + 2048 i`、由 uniform 的 `tm`、`tn`、`e`、`i` 构造的坐标）。因此实测的 SASS 显示每条 PTX 存储对应一条 uniform 的 `UTMASTG.3D`，无论 lane 数多少都每个 warp 发射一次；K4b 在每次发射时都实测到 32 个活跃的 lane（笔记 D.2）。baseline 以同样的方式发射该存储，不做选举（collective epilogue 中的 `issue_tma_store = (warp_idx == 0)`，笔记 13 第 8 项），并依赖完全相同的下沉。我对替代方案的分析：一个 ptxas 无法证明为 warp-uniform 的操作数（比如说，一个经由 lane id 传递、却在每个 lane 上都相等的坐标）会是合法的 PTX 并且仍然正确，但编译器随后 "may insert a peeling loop over all active threads, which results in warp serialization"（CUDA Programming Guide 4.12.2.1，"TMA write and sync" 下的注记）：对同一个 box 多达 32 次单独的存储以及 32x 的 TMA 流量。这就是为什么 kernel 中 W23 的每个操作数都是 warp-uniform 的，也是为什么这里不需要选举，而加载（W21/W22）之所以经过选举，是因为 baseline 对它们做了选举。

### 6.6.6 W24/W25 `tma_store_commit_group`、`tma_store_wait_read<Count>`

```
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
```
(explicit.cu:486-495)

**目的。** W24：把刚刚发射的四个存储作为一个新的 bulk async-group 提交。W25：等待直到本线程至多还有 `Count` 个组仍在*读取*其共享内存源。

**分解。** `cp.async.bulk.commit_group` 把执行线程此前所有未提交的 bulk 异步操作捆成一组。`cp.async.bulk.wait_group N` 等待直到最近的组中至多还有 `N` 个处于待完成状态。`.read` 修饰符把这一条件放宽为“待完成的*读取*”：只要这些组从共享内存读取源的操作已经完成，即便它们的全局写入仍在途，该等待就会返回。这正是 epilogue 所需要的：它只想复用 smem stage，而不是想知道 D 已全局可见（后者留给 kernel 完成，笔记 D.1 D15 与第 10.11 节）。`"n"(Count)`：ISA 要求一个整数常量，所以模板参数变成一个立即数（`cp.async.bulk.wait_group.read 1` 在 PTX 中保留其立即数，笔记 7.2）。

**为什么 `Count = 1`。** 在提交组 `e` 之后，`wait_group.read 1` 保证组 `e - 1` 已经读完 stage `(e - 1) % 4`。接下来要写入的 stage `(e + 1) % 4` 最后一次被读取是由组 `e - 3` 进行的，因此已经空闲。等待 0 会把 TMA 读取与下一个 subtile 的 `tcgen05.ld` 及算术串行化（CUTLASS：`PipelineTmaStore<4,1>` 的 `producer_acquire`，带 `UnacquiredStages = 1`，笔记 13 第 16 项；第 10 章，第 10.10.4 节）。CUTLASS 的 `store_tail` 也有一条 `wait_group.read 0`。beta 是一个运行时值，所以 baseline 无条件地编译了它（SASS `0xb760` 处的 `DEPBAR.LE SB0, 0x0`），但只有当 `is_producer_load_needed()` 为真、即 beta != 0 时才会执行到它（笔记 D.6 第 11 项、D.1 D15、笔记第 7.7 节）。explicit kernel 按构造就是一个 beta = 0 的 kernel，并省略了它（笔记 G.3：`wait_group.read 0` = 0，`DEPBAR.LE SB0, 0x0` = 0），所以 CTA 退出时至多有一个 bulk group 的 smem 读取未被等待，而全局写入随 kernel 一起完成（D15；第 10 章，第 10.11 节）。

**Clobber。** 第 6.0.2 节给出了这两个论证。W24 没有（CuTe 也没有）：它只是把 W23 那条带 clobber 的 asm 已经发射的操作归成一组。W25 有 `"memory"`：复用被释放 stage 的那条 C++ `st.shared`（explicit.cu:1047）出现在该等待之后三个 subtile 处（在读取过该 stage 的那条存储之后则是四个），位于同一个展开体中或下一个 tile 的展开体中，并且必须留在该等待之下。

**调用者。** explicit.cu:1068 与 1069，warp 4，各自每个 subtile 一次。每个 tile 每个 CTA 的计数：8 + 8。实测：`UTMACMDFLUSH` 与 `DEPBAR.LE SB0, 0x1`（笔记 D.2、G.3）。

---

## 6.7 第 6 族：tcgen05，W26-W32

### 6.7.1 W26-W28 `tmem_alloc_2sm`、`tmem_dealloc_2sm`、`tmem_relinquish_2sm`

```
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
```
(explicit.cu:497-514)

**分解。** 第 2 章，第 2.3.2 节解码了 `tcgen05.alloc` 与 `tcgen05.dealloc`；relinquish 是此处剩下唯一需要解读的一条指令：

| 文本 | 含义 |
|---|---|
| `tcgen05.relinquish_alloc_permit` | 声明本 CTA 将不再进行任何分配，ISA 对此仅表述为 "relinquishing the right to allocate Tensor Memory"，此后它自己的 `tcgen05.alloc` 即为非法（9.7.18.7.1）；释放该许可会释放 Virtual Resource Counter（VRC，第 9.6.2 节），因此一个后继 CTA 可以被调度到该 SM 上并进入其分配阶段，而不必等待本 CTA 退出，且没有任何顺序或调度上的保证（NVIDIA 内部）；这是 CUTLASS 在 kernel :783 处所声明的意图，"so that the next CTA can rasterize"（第 9.6.2 节）；在 tile 循环之后、dealloc 之前发射 |

**没有 `"memory"` clobber**（与 CuTe 中一样）。写入 `smem[384]` 的结果由 C++ 在 explicit.cu:888（warp 0）与 :998（warp 4-7）处读取。两种情况下都有一条带 clobber 的语句介于其间（warp 0 是 W37 + W19，epilogue warp 是 W18），所以编译器无法把该加载外提（第 6.0.2 节；笔记 E.6.4 的 `mma_role` 段落，它把这归功于 W19 的 clobber；E.6.2："adding one to `tcgen05.alloc` is a harmless tightening"）。

**调用者。** W26：explicit.cu:885，两个 CTA 的 warp 0，在 `mma_role` 中先于其他一切。实测：alloc 是围绕 `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN` 的一个 69 条指令的序列（一个分配单元 = 32 列，`0x10` 个单元 = 512），其原子操作仅由 leader 的被选中的 lane 执行（该序列本身在两个 CTA 的 warp 0 中都运行；peer 的被选中的 lane 等待、重新缩放并应答，笔记 D.2）。结果被存储在本地（`STS [UR..+0x180]`，偏移 384），并用 `STAS`（`st.async`）转发到 peer 的 `smem+384`。那条 `STAS` 完成 peer 的系统保留共享内存中位于 `(rank << 24) + 0x48` 处的一个握手屏障，簿记字位于 `+0x50/+0x54/+0x58`（笔记 D.2）。explicit kernel 的 69 条指令的序列与 baseline 的有相同的长度与骨架（G.3；八处等价的助记符拼法、一些重排以及寄存器编号有所不同，第 9.4.1 节）。dealloc 在两个 CTA 中都是 `UTCATOMSWS.AND`，其前有两条 ptxas 插入的一致性检查，在不匹配时 `BPT.TRAP`。relinquish 下沉为 `UVIRTCOUNT.DEALLOC.SMPOOL 0x80`（0x80 = 128：编译器为它发出 `UVIRTCOUNT 128` 这一点是 NVIDIA 内部信息）；ptxas 把它放在了 W36 的 `PREEXIT` 之前，调换了 PTX 中的顺序，而那条 `PREEXIT` 周围的 `ELECT`（0x55d0）与受 `ELECT` 保护的 `STS.U8`（0x5630）不属于 relinquish，后者是 warp 集体的，必须位于被选中的单线程区域之外（NVIDIA 内部，撤回笔记 D.2 中推断出的归因；D.6 第 7 项和第 9 项）；它们在源码层面的来源尚未解决。

**陷阱。** (1) 两个 CTA 必须从相同的 warp ID 以相同的操作数发射 alloc 与 dealloc（相同的逻辑 warp ID 与相同的 `dst_ptr` 是 CuTe 的前置条件，tmem_allocator_sm100.hpp:128-132、152-154）；不匹配的后果是未定义的（NVIDIA 内部），ISA 也没有规定它，所以不要指望某种特定的症状，比如 pair 在分配握手中死锁。(2) 忘记 dealloc 违反了 ISA 的规则，即一个 kernel 中分配的所有 TMEM "must be explicitly deallocated using tcgen05.dealloc before the kernel exits"（9.7.18.7.1）；一个泄漏的分配可能在该 CTA 仍驻留期间使之后的分配饥饿，而且不得依赖拆除时的清理（NVIDIA 内部），所以那个 SM 上的下一条 `tcgen05.alloc` 可能阻塞。

### 6.7.2 W29 `umma_f16_2sm`：MMA

```
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
```
(explicit.cu:516-529)

**目的。** 一次跨 CTA pair 的 256 x 128 x 16 的 fp16 -> fp32 矩阵乘累加：D[256 x 128] (+)= A[256 x 16] x B[16 x 128]，A 与 B 位于共享内存（两个 CTA 的）中，D 位于 TMEM（两个 CTA 的）中。

**分解。**

| 文本 | 含义 |
|---|---|
| `tcgen05.mma` | 第 5 代 tensor core MMA；异步 |
| `.cta_group::2` | 该 MMA 横跨 CTA pair，由一个线程发射（ISA 允许 pair 中任一 CTA 的单个线程，表 55；CUTLASS 与本 kernel 使用 leader CTA）。A 的行 [0,128) 与 B 的列 [0,64) 来自 leader 的 smem；行 [128,256) 与列 [64,128) 来自 peer 的 smem 中**相同的 CTA 相对地址**（因此两个 CTA 中的 smem layout 完全相同）。每个 CTA 的 TMEM 接收它的那 128 行 D（笔记 B.6；Colfax："pair-UMMA must be launched from one thread in one of the CTAs that we elect as the leader CTA; in CUTLASS, we will always choose the even CTA as the leader"） |
| `.kind::f16` | fp16/bf16 输入，fp32（或 fp16）累加；元素类型在指令描述符中 |
| `[%0]` | `tmem_c = T + 128 st`（explicit.cu:919）：D 的 TMEM 地址，lane 0，列 128 st；该指令在每个 CTA 中填充 128 个 lane x 128 列 |
| `%1`、`%2`（`"l"`） | A 与 B 的 64 位**共享内存描述符**：`0x4000404000010880 + 1024 s + 2 kb` 与 `0x4000404000012880 + 512 s + 2 kb`（第 5 章解码了这些位域：以 16 字节为单位的 14 位起始地址、LBO、SBO、SWIZZLE_128B） |
| `%3`（`"r"`） | 32 位的**指令描述符** `kInstrDesc = 0x10200010`（explicit.cu:173）：c_format F32，a/b_format F16，K-major 的 A 与 B，N = 128，M = 256 |
| `setp.ne.b32 p, %4, 0;` + `, p` | **enable-input-d** 谓词：`p = (scaleC != 0)`；true -> D = A x B + D，false -> D = A x B（累加器被覆盖）。kernel 为一个 tile 的第一条 MMA（q = 0，kb = 0）传入 `scale_c = 0`，为其余 511 条传入 1（explicit.cu:917、940）。这就是一个 TMEM stage 免费被“清零”的方式。 |
| `{%5, ..., %12}` | **disable-output-lane** 向量：对 `.cta_group::2` 为 8 x 32 位 = 256 位（对 `::1` 为 4 x 32），pair 的每个 TMEM lane 各一位；置位的位表示该 lane 不被写入。全零 = 写入每个 lane。CUTLASS 总是传入零（局部变量 `uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0}`，mma_sm100_umma.hpp:571），而且，尽管 ISA 把该操作数列为可选（其语法中的 `{ disable-output-lane }`；它自己的示例省略了它），CUTLASS 的 asm 字符串把该向量完整拼写出来，所以八个零寄存器在 PTX 中被物化。在 SASS 中它们消失了（`UTCHMMA.2CTA` 没有掩码操作数，笔记 D.2）。 |

**没有 `"memory"` clobber**（与 CuTe 中一样）：每个操作数都是寄存器。它读取的 smem 由 TMA（异步 proxy）写入，而其可用性由 W6/W7 对 `full[s]` 的 wait 建立，后者带有该 clobber。

**调用点处的选举。** CUTLASS 的 `fma` 把该 asm 包裹在 `if (cute::elect_one_sync())` 之中（mma_sm100_umma.hpp:570）。该 wrapper 是无条件的，而 kernel 写的是 `if (elect_one_sync()) { ... umma_f16_2sm(...); }`（explicit.cu:932-939），所以 PTX 保留每条 MMA 一次选举。

CUTLASS 的操作数 `"r"(uint32_t(idescE>>32))` 变成了简单的 `"r"(idesc)`，因为 kernel 直接传入 32 位描述符。

**调用者。** explicit.cu:938，leader warp 0，每个 k-block：`umma_f16_2sm(tmem_c, desc_a, desc_b, kInstrDesc, scale_c)`，其中 `desc_a = desc64(kSmemDescHi, lo_a + 2 kb)`、`desc_b = desc64(kSmemDescHi, lo_b + 2 kb)`（explicit.cu:933-934）。每个 tile 的计数：pair leader 中 512 条（每 k-tile 4 条 x 128），peer 中 0 条。实测：`UTCHMMA.2CTA gdesc[URa], gdesc[URb], tmem[UR], tmem[UR28 = 0], idesc[UR29 = 0x10200010], UP`，其中第一条 MMA 由 scale-c uniform 谓词控制，其余三条由 `UPT` 控制（笔记 D.2、G.3）。

### 6.7.3 W30 `umma_commit_mcast_2sm`：commit 操作

```
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
```
(explicit.cu:531-540)

**分解。**

| 文本 | 含义 |
|---|---|
| `tcgen05.commit` | 异步："Makes the mbarrier object track the completion of all prior async-tcgen05 operations initiated by the executing thread"；"Upon the completion of the tracked asynchronous tcgen05 operations, the signal specified by the .completion_mechanism is triggered by the system on the mbarrier object. This instruction accesses its mbarrier operand using generic-proxy."（PTX ISA 9.4，第 9.7.18.12.1 节） |
| `.cta_group::2` | 被 commit 的操作是 pair MMA；commit 操作由同一个 leader 线程发射 |
| `.mbarrier::arrive::one` | 完成以**一次** arrive-on 操作（计数 1）的形式向每个目的屏障发出信号："upon the completion of the prior asynchronous tcgen05 operation issued by the current thread, an arrive-on operation, with the count argument of 1, is signaled on the mbarrier object. The scope of the arrive-on operation is the cluster scope."（9.7.18.12.1） |
| `.shared::cluster` | 屏障地址在 cluster 窗口中解释：在每个目的 CTA 中都是同一个相对于 CTA 的偏移 |
| `.multicast::cluster` + `%1`（`"h"`） | 该 arrive 被投递到掩码中置位的每个 CTA 内位于该偏移处的屏障；`ctaMask` 是与 W21 相同的 16 位（`::16b`）操作数，带有同样未使用的 `::32b` 形式（第 6.6.3 节） |
| `.b64 [%0]` | 位于执行 CTA 的屏障地址处的 64 位屏障（`empty_base + 8 rs` 或 `acc_full_base + 8 st`），未加掩码：选择目的 CTA 的是 multicast，而不是第 24 位 |

**为什么它不能是一条普通的 `mbarrier.arrive`。** `tcgen05.commit` 把 arrive 推迟到先前每一个 "initiated by the executing thread"（9.7.18.12.1）的 tcgen05 操作全部完成之后，这正是 commit 操作必须由发射了这些 MMA 的**同一个被选中的线程**来发射的原因（第 2 章，第 2.4.3 节；笔记 7.6）。kernel 在 commit 操作之前重新选举（explicit.cu:942、944），而不是沿用 MMA 的 lane 谓词。这是正确的，因为带有相同 `membermask` `0xFFFFFFFF` 的 `elect.sync` 是确定性的（PTX ISA 9.4，9.7.15.15；第 2 章，第 2.10.3 节），所以发射了那四条 MMA 的 lane 就是发射 commit 操作的 lane。来自任何其他 lane 的 commit 操作不会跟踪任何东西，并且会在 tensor core 仍在读取该 stage 或仍在写入 TMEM 时就向 `empty[s]`（或 `acc_full[st]`）发出信号。SASS 把二者都折叠成 uniform 指令（笔记 D.6 第 6 项）是叠加在一份本已正确的 PTX 契约之上的下沉细节。

**为什么没有 fence 伴随 commit 操作。** 三个 ISA 事实承载了这一排序：commit 操作 "accesses its mbarrier operand using generic-proxy"；它的 arrive-on 在 cluster 作用域执行（两者均见 9.7.18.12.1）；而等待一侧的 `mbarrier.try_wait` 默认具有 acquire 语义（"If the .sem qualifier is absent, .acquire is assumed by default"，9.7.15.16.19）。`acc_full[st]` 上借由 commit 的 release，加上 epilogue 中借由 wait 的 acquire，使 MMA 写入的 TMEM 可被 `tcgen05.ld` 读取。`empty[s]` 上借由 commit 的 release，加上 producer 中借由 wait 的 acquire，让 TMA 得以覆写该 stage。因此 k 循环（explicit.cu:921-943，第 9.5.3 节）不包含任何 `tcgen05.fence::*`，也不包含任何 `fence.proxy.async`（`tcgen05.fence` 在整个 PTX 中都不存在，笔记 G.3 的缺失列表；PTX 中的十二条 `fence.proxy.async.shared::cta` 位于 CLC 解码处，explicit.cu:683，以及 epilogue 的存储路径，:1050；笔记 G.3 的清单，第 2.7.6 节）。对这三个事实如何组合的这一解读是我的，而 ISA 自己的跨线程模式更严格：9.7.18.6.4.4（例 2）在 `mbarrier.try_wait` 与 `tcgen05.ld` 之间放了一条 `tcgen05.fence::after_thread_sync`，CUTLASS 和 kernel 都省略了它（第 2.4.4 节；第 13 章，第 13.3 节末尾的列表）。ISA 的 "Memory Consistency Model for 5th generation of TensorCore operations" 一节是权威。

`"h"(cta_mask)`：`uint16_t` 参数，与 W21 相同。`"memory"` 与源码相同（barrier.h:858）。

选举移到了调用者处。

**每个 tile 每个 CTA 的计数。** 在 pair leader 中为 128 + 1，在 peer 中为 0。实测：`UTCBAR.2CTA.MULTICAST [UR], URZ, URmask`，掩码位于 uniform 寄存器中（`0xf`；`0x3`/`0xc`）；mainloop 的 commit 操作是紧跟在第四条 `UTCHMMA` 之后的那条指令（笔记 D.2、G.3）。

### 6.7.4 W31 `tmem_ld_32x32b_x16`：读取累加器

```
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
```
(explicit.cu:542-555)

**目的。** warp 的每个 lane 把自己 TMEM lane 的 16 个连续 fp32 累加器值（16 列）加载到 16 个寄存器中。

**分解。**

| 文本 | 含义 |
|---|---|
| `tcgen05.ld` | 异步的 TMEM -> 寄存器加载 |
| `.sync.aligned` | 全部 32 个 lane，已收敛 |
| `.32x32b` | shape：32 个 lane（数据通路）x 32 位：warp 的 lane `l` 读取 TMEM lane `(base lane) + l` 的 32 位单元 |
| `.x16` | 重复次数：16 个连续的列，每列一个寄存器，于是 `d[j]` = 列 `(base column) + j` |
| `.b32` | 寄存器类型 |
| `{%0 ... %15}` | 16 个 `"=r"` 目的操作数，调用点处为 `r[0..15]` |
| `[%16]` | TMEM 地址 `taddr + 16 e`，其中 `taddr = T + ((32 w) << 16) + 128 st`（explicit.cu:1031、1036）：lane `32 w`，列 `128 st + 16 e` |

**lane 象限规则。** 一个 warp 只能访问 TMEM lane `[32 (warp_id % 4), 32 (warp_id % 4) + 32)`（笔记 B.6；第 2 章，第 2.3.3 节）。epilogue warp 是 4-7，所以 `warp_idx % 4 = w = warp_idx - 4`（explicit.cu:1004），而 `tmem_lane_off = (32 w) << 16`（explicit.cu:1006：`0, 0x200000, 0x400000, 0x600000`）恰好寻址 warp `w` 可以读取的那个象限。因此线程 `(w, l)` 拥有 CTA tile 的累加器第 `32 w + l` 行。源码注释中的例子（explicit.cu:982：`w = 2, st = 1, e = 3` -> `taddr + 16 e = 0x4000B0`，lane 64，列 176，所以 warp 6 的 lane `l` 得到 stage 1 的 `D[64 + l][48..63]`）在第 2 章，第 2.3.4 节中演算过。

**为什么是 `.32x32b`，为什么是 `.x16`。** 存储决定了 shape。`.32x32b` 给 32 个 lane 中的每一个各一个来自*它自己的* TMEM lane 的 32 位单元，也就是每个线程一行累加器。M-major 的 `Swizzle<2,5,2>` 存储模式恰好消费这个：线程 `(w, l)` 写每个 16 列 subtile 的第 `32 w + l` 行（第 10 章）。`.x16` 是 subtile 宽度：epilogue tile 是 (128, 16)，对于 32 位 C/D 且存在残差 C 的情形，`EpilogueTileAuto` -> N = 16（第 3 章，第 3.6 节给出完整条件），所以一次加载交付一个线程在两次 `bar.sync` 之间缩放并存储的 16 个 fp32 值。ISA 的其他 `tcgen05.ld` shape，即 `.16x64b`、`.16x128b`、`.16x256b` 和 `.16x32bx2`（PTX ISA 9.4，9.7.18.2.3，"Data Movement Shape"），把 16 个 TMEM lane 的数据以矩阵片段 layout 分布到 warp 的 32 个线程上。它们不提供存储所需要的每线程一行映射（这是我对 CUTLASS 的 `SM100_TMEM_LOAD_32dp32b16x` 为何是此处正确原子的分析）。

**超出 `.sync.aligned` 之外的收敛规则。** "The behavior of tcgen05.ld is undefined if all threads do not use the same values of taddr, or if any thread in the warp has exited"（PTX ISA 9.4，9.7.18.8.3）。因此 kernel 以 warp-uniform 的方式构建 `taddr`：`T + tmem_lane_off + 128 acc_stage`，每一项在整个 warp 上都相同（explicit.cu:1031），而每次调用之间唯一的变化是 uniform 的 `16 e`（:1036）。lane 索引进入的是*结果*，而从不进入地址。“没有已退出线程”这一条款得到满足，是因为全部 128 个 epilogue 线程都把整个 subtile 循环运行到最后一个 tile 的末尾：tile 循环没有任何每线程的退出（explicit.cu:1024-1075）。

**没有 `"memory"` clobber**（与 CuTe 一致）：16 个输出是带有显式数据依赖的寄存器，而 TMEM 的内容不是编译器所知道的 "memory"。

该 wrapper 接受一个 `uint32_t (&)[16]`，而不是 16 个引用参数。

**调用者。** explicit.cu:1036，warp 4-7，每个 subtile `e` 一次。每个 tile 每个 CTA 的计数：每个 warp 8 次，共 32 次。实测：explicit kernel 中为 `LDTM.x16 R4, tmem[UR29 + 0x10*e]`（`trace_out_v2/sass_explicit.txt`，`0x2290`-`0x44d0`；第 12 章，第 12.2 节），baseline 中为 `tmem[UR4 + 0x10*e]`（笔记 D.2）；在两者中，`16 e` 项都是立即数，目的寄存器都是 `R4..R19`。

### 6.7.5 W32 `tmem_wait_ld`

```
// W32 [tcgen05 ld]  cutlass::arch::fence_view_async_tmem_load (barrier.h:923-931)
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile (
    "{\n\t"
    "tcgen05.wait::ld.sync.aligned; \n"
    "}"
    ::
    : "memory");
}
```
(explicit.cu:557-565)

**分解。** `tcgen05.wait::ld` 阻塞直到该线程未完成的 TMEM 加载全部结束（`::st` 变体对 `tcgen05.st` 做同样的事，此处未使用）；`.sync.aligned` 一如往常。笔记 B.6 把 ISA 解读为要求在一条 `tcgen05.ld` 的寄存器被消费之前执行这个 wait；ISA 本身只为内存排序而要求它，而真正的寄存器依赖即使没有它也会被遵守（第 2 章，第 2.4.4 节）。`"memory"` 与源码一致。

**kernel 实际做了什么，以及为什么这是安全的。** subtile 0..6 消费各自的寄存器时没有显式的 wait。SASS 解释了这为什么行得通（没有 wait 操作码这一点是实测的；scoreboard 的解码是从控制字推断的）：`tcgen05.wait::ld` 根本不产生任何指令，而每条 `LDTM.x16` 之后的第一条 `FMUL` 等待 `LDTM` 所设置的 scoreboard（笔记 D.2、G.3："the dependency is carried by the control-word scoreboards"）。explicit kernel 在这里精确地镜像了 baseline（笔记 B.6："the explicit kernel can mirror the baseline (one `tcgen05.wait::ld` per tile, before the release)"）。这个 wait 在语义上不可或缺的唯一位置是 W11 之前：release 告诉 MMA warp 它可以覆写该 TMEM stage，所以来自它的每一次加载都必须先完成。

**每个 tile 每个 CTA 的计数。** 每个 warp 1 次，共 4 次。实测：没有指令；W11 的 `SYNCS.ARRIVE` 等待最后一条 `LDTM` 的 scoreboard（从控制字推断；笔记 D.2、G.3）。

---

## 6.8 第 7 族：CLC，W33-W34

### 6.8.1 入门：硬件工作窃取

网格启动时带有每个 tile 的 CTA（此处为 64 x 64），但只有 `N_native = 33` 个 cluster 是驻留的（实测，笔记 D.3）。在观察到一次失败的 `try_cancel` 请求之后，一个 CTA 不得再发出另一次请求（PTX ISA 9.4，9.7.15.18，`clusterlaunchcontrol.try_cancel`："If the executing CTA has already observed the completion of a clusterlaunchcontrol.try_cancel instruction as failed, then the behavior of issuing a subsequent clusterlaunchcontrol.try_cancel instruction is undefined"；CUDA Programming Guide，"4.13 Work Stealing with Cluster Launch Control"，4.13.1.2："Submitting another cancellation request after observing a previously failed request is undefined behavior"；Colfax 的 CLC 文章："after such an observance, the CTA can no longer use CLC and should exit after exhausting its current work queue"）。

### 6.8.2 W33 `clc_try_cancel`

```
// W33 [CLC]  PersistentTileSchedulerSm100::issue_clc_query (sm100_tile_scheduler.hpp:392-405); elected lane of the scheduler warp
__device__ __forceinline__ void clc_try_cancel(uint32_t result_addr, uint32_t mbarrier_addr) {
  asm volatile(
    "{\n\t"
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
    "}\n"
    :
    : "r"(result_addr), "r"(mbarrier_addr));
}
```
(explicit.cu:567-575)

**目的。** 为整个 cluster 发出一次取消请求；响应落入每个 CTA 的响应槽位，并完成每个 CTA 的 `clc_full[idx]`。

**分解。**

| 文本 | 含义 |
|---|---|
| `clusterlaunchcontrol.try_cancel` | 请求原子地取消本网格中一个尚未启动的 cluster |
| `.async` | 该请求是异步的；完成通过 mbarrier 报告 |
| `.shared::cta` | **两个**操作数的状态空间（ISA："both operands addr and mbar must be in the .shared::cta state space"）；响应槽位 `[%0]` = `smem_base + OFF_CLC_RESPONSE + 16 idx`（explicit.cu:847；16 字节对齐，explicit.cu:151） |
| `.mbarrier::complete_tx::bytes` | 当响应被写入时，带有其字节数 (16) 的 `complete_tx` 在 mbarrier `[%1]` = `smem_base + OFF_CLC_FULL + 8 idx` 上执行（explicit.cu:840） |
| `.multicast::cluster::all` | 响应被写到发出请求的 cluster 中**每一个** CTA 的相同偏移处，并且每个 CTA 位于相同偏移处的屏障都收到这 16 个事务字节；`::all` 形式不带掩码 |
| `.b128` | 响应是 128 位；"any `try_cancel` operation always transfers 16 bytes"（Colfax 的 CLC 文章） |

因此有 `kClcTxBytes = 16`（explicit.cu:181）以及第 6.3.8 节的 W13 装填：硬件给**每一个** CTA 的 `clc_full[idx]` 计入 16 字节，所以每个 CTA 的屏障事先需要 `expect_tx 16` 外加一次到达，这由 rank 0 的 lane 0..3 远程提供。

**没有 `"memory"` clobber**（与 CUTLASS 一致）：异步 proxy 写入响应，而 consumer 在读取它之前先在屏障上等待（W6，带 clobber）。

**调用者。** explicit.cu:847，rank-0 的 warp 1，位于 `if (elect_one_sync())` 之内，每个 tile 一次，在 throttle 等待和 W13 装填之后：`clc_try_cancel(smem_base + OFF_CLC_RESPONSE + 16u * clp_idx, full_addr)`。每个 cluster 同一时刻只有一个查询在途（实测：从未多于一个未完成，笔记 D.3）。每个 tile 的计数：每个 cluster 1 次。实测：explicit kernel 中为 `UGETNEXTWORKID.BROADCAST [UR8], [UR9]`（baseline 中为 `[UR4], [UR5]`），uniform，无谓词（笔记 G.3、D.2）。

### 6.8.3 W34 `clc_decode`

```
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
```
(explicit.cu:577-595)

**分解。**

| 文本 | 含义 |
|---|---|
| `.reg .b128 clc_result; ld.shared.b128 clc_result, [%4];` | 从 consumer 本轮所拥有的槽位，把不透明的响应加载到一个 128 位的 PTX 寄存器中 |
| `@p1 ... get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result;` | 以成功作为**谓词条件**：提取被取消 cluster 的第一个 CTA 的启动网格坐标 `(x0, y0, z0)`（cluster 对齐，所以 `x0`、`y0` 为偶数）；第四个 `.v4` 元素用 `_` 丢弃 |

这次提取之所以带谓词条件，是因为读取一个失败响应的 `get_first_ctaid` 是未定义行为（PTX ISA 9.4，9.7.15.19，`clusterlaunchcontrol.query_cancel`："If the request fails the behavior of clusterlaunchcontrol.query_cancel.get_first_ctaid is undefined"；Programming Guide 4.13.1.2："Retrieving the thread block index of a failed cancellation request is Undefined Behavior"；Colfax："if `.is_canceled` returns false then `query_cancel` instructions other than `.is_canceled` results in undefined behavior"）。kernel 在调用之前初始化 `x0 = y0 = z0 = 0`（explicit.cu:681），并且只在 `if (valid)` 之下读取它们（explicit.cu:691-694），在那里它们被映射为 `(tm, tn) = (2 (y0 >> 1) + (x0 & 1) + x, 2 (x0 >> 1) + (y0 & 1) + y)`（第 7 章）。

`"memory"` 与源码中相同：这段 asm 读取由异步 proxy 写入的共享内存。CUTLASS 的函数随后调用 `fence_view_async_shared()`（sm100_tile_scheduler.hpp:430）；kernel 在 W34 之后紧接着在 C++ 中调用 W15（explicit.cu:683），所以这个 wrapper 仍然只是一次解码。

这些输出在 CUTLASS 中曾是结构体成员，在这里则是引用参数。

**调用者。** explicit.cu:682，位于 `clc_consume` 中，每个进行消费的 warp（每个 CTA 的 warp 0、2、4-7；rank 0 的 warp 1），每个 tile 一次，在对 `clc_full[clc_idx]` 的 W6 wait 之后。每个 tile 每个 CTA 的计数：6（rank 0 中为 7）。实测：`LDS.128` 加上位字段提取（`is_canceled` = 字 2 的第 0 位，`x0 = word0`，`y0 = word1 & 0xffff`；该 layout 是从提取代码推断的，笔记 D.2；baseline 的 `z0 = word1 >> 16` 在这里没有对应物，因为 explicit kernel 丢弃了 `z0`，笔记 G.3）。explicit kernel 有 4 条 `LDS.128`（`0x1370` producer，`0x1c60` scheduler，`0x20b0` epilogue，`0x5070` MMA warp；笔记 G.3，`trace_out_v2/sass_explicit.txt`）；在 scheduler 与 MMA warp 中只有 valid 位在死代码消除中幸存下来；producer 与 epilogue 的解码则提取 `x0`、`y0`。

---

## 6.9 第 8 族：网格依赖控制，W35-W36，以及 W37

**什么是程序化依赖启动，以及为什么一个从不使用它的 kernel 仍然会发出这些指令。** PDL 允许一个以 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性启动的 kernel（即 CUTLASS 的 `ClusterLauncher` 在 cluster_launch.hpp:199-200 处写入的那个属性）在流中前一个 kernel 完成之前就开始。explicit kernel 的启动不带该属性，由固定运行的普通 `cudaLaunchKernelExC` 完成（explicit.cu:1304-1312：单个属性，`cudaLaunchAttributeClusterDimension`；第 11 章：baseline 的 `ClusterLauncher` 写入了 PDL 属性，但在 `launch_with_pdl = false` 时并不把它计入，cluster_launch.hpp:206）；两条指令在那里都没有运行时效果：`griddepcontrol.wait` 找不到任何在途的前置网格，而 `launch_dependents` 没有被指定的依赖者，因为固定运行的任何一次启动都不带 PDL 属性 [推导：PTX ISA 9.7.15.14 + 启动配置，如第 2.13 节中那样]。kernel 仍然发出它们，因为 baseline 在其 `CUTLASS_ENABLE_GDC_FOR_SM100=1` 构建下发出它们，并且笔记 E.7.4 把 `griddepcontrol.wait` x2 与 `launch_dependents` x1 定为硬性 PTX 行（E.8 D3），而 G.3 证实了这一点。代价是每个 CTA 在 tile 循环之外的两到三条指令（warp 2 中的 W35，rank 0 的 warp 1 中再一次 W35，warp 0 中的 W36）。确实以 PDL 启动该 kernel 的用户可以免费获得 prologue（屏障初始化、TMEM 分配、描述符预取）与前一个 kernel 尾段的重叠；这种重叠是机会性的，而不是有保证的（Programming Guide 4.5）。

### 6.9.1 W35/W36 `griddep_wait`、`griddep_launch_dependents`

```
// W35/W36 [GDC]  cutlass::arch::wait_on_dependent_grids / launch_dependent_grids (grid_dependency_control.h:95-98, 85-88);
// emitted unconditionally: the fixed build defines CUTLASS_ENABLE_GDC_FOR_SM100=1 for sm_100a (Section 11)
__device__ __forceinline__ void griddep_wait()              { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }
```
(explicit.cu:597-600)

**Clobber。** `"memory"` 加在 wait 上而不加在提示上，与 CUTLASS 中一样：全局加载不得被外提到该 wait 之上，而提示不对任何东西排序。

**为什么无条件。** CUTLASS 用 `CUTLASS_GDC_ENABLED` 守护二者，而固定构建通过针对 `sm_100a` 的 `CUTLASS_ENABLE_GDC_FOR_SM100=1` 来定义它（笔记 D.1 D14：`GDC_FOR_SM100 1`、`IsGdcGloballyEnabled 1`）。

**实测。** 两个 kernel 都只有一条 `PREEXIT`。explicit kernel 有 `ACQBULK` x2、`PREEXIT` x1（笔记 G.3）；baseline 执行同样的两条 `ACQBULK`，而其 SASS 在死的 warp-3 C 路径中还带有从未执行的第三条（笔记 D.2；第 2 章，第 2.13 节）。

### 6.9.2 W37 `__syncwarp()`

```
// W37 is the __syncwarp() intrinsic (bar.warp.sync -1), used directly.
```
(explicit.cu:601)

**目的。** 使 warp 重新收敛，并对其各 lane 的内存操作排序（带完整掩码 `-1` = `0xffffffff` 的 `bar.warp.sync`）。

**调用者。** explicit.cu:792（warp 2，在一个 tile 的加载之后、`clc_consume` 之前；kernel :664）：它在随后的 CLC 消费（仅有每线程的等待）以及下一个 tile 的 warp 集体 `elect.sync` 之前，收拢了被选中的 lane 发射 TMA 所造成的发散。explicit.cu:886（warp 0，在 `tcgen05.alloc` 之后、`bar.arrive 6,160` 之前；kernel :729）。计数：每 tile 1 次（warp 2）+ 每 CTA 1 次（warp 0）。在 explicit kernel 中实测（`trace_out_v2/sass_explicit.txt`）：恰好存在两条 `WARPSYNC.ALL`，一条在 `0x4d40`（位于 `tcgen05.alloc` 的下沉实现内部，在其位于 `0x4d60` 的 `ELECT` 之前），一条在 `0x4eb0`，其后跟着一条 `NOP` 和位于 `0x4ed0` 的 `BAR.ARV 0x6, 0xa0`，也就是在 warp 0 的 `__syncwarp()` 所在的位置。warp 2 的每 tile 调用点没有产生任何指令。这就是笔记为 baseline 测得的形态（`0x32d0` 在 alloc 内部，`0x3440` 在 `BAR.ARV` 之前；笔记 D.2，D.6 第 15 项）；把第二条 `WARPSYNC.ALL` 归于 W37 是从其位置推断的，与笔记中一样。

---

## 6.10 汇总表

计数为已执行路径上每 tile 每 CTA 的值（固定运行：beta = 0，128 个 k-tile）。"L" 标记仅适用于 pair leader 的计数（peer 中为 0）；"R0" 标记仅限 rank 0 的计数；"exit"/"start" 是在 tile 循环之外每个 CTA 一次。SASS 助记符是笔记 D.2/G.3 实测的下沉结果；"none" 表示该 PTX 指令没有产生它自己的任何 SASS 指令。

| W | Wrapper | PTX 助记符 | SASS | 调用者 warp | 每 tile 每 CTA 的计数 |
|---|---|---|---|---|---|
| - | `smem_u32` | `cvta.to.shared`（已折叠） | `S2UR SR_CgaCtaId` + `ULEA` | 全部，start | start：每线程 1 |
| W1 | `warp_idx_sync` | `shfl.sync.idx.b32` | `SHFL.IDX` | 全部，start | start：1 |
| W2 | `elect_one_sync` | `elect.sync` | `ELECT`（保留 4 条；在循环中被折叠） | 2；0 (L)；1 (R0)；做初始化的各 warp | 128 (warp 2) + 641 (L) + 1 (R0)；start 1 + 4 组 |
| W3 | `cluster_ctarank` | `mov %cluster_ctarank` | `S2UR SR_CgaCtaId` | 全部，start | start：1 |
| W4 | `cluster_ctaid_x/y` | `mov %cluster_ctaid.x/.y` | `SR_CTAID` + 表 + `I2FP/FMUL/F2I`（推断） | 全部，start | start：各 1 |
| W5 | `mbar_init` | `mbarrier.init.shared::cta.b64` | `SYNCS.EXCH.64` | 0、4、5、3（经选举） | start：每 CTA 33 |
| W6 | `mbar_wait` | `mbarrier.try_wait.parity ..., hint` 循环 | `SYNCS.PHASECHK.TRANS64.TRYWAIT` + `NANOSLEEP.SYNCS 0x989680` 循环 | 所有角色 | 6 (7 R0) CLC + 1 (L) acc-empty + 3 (R0) throttle/CLC；mainloop/acc-full 仅当 peek 失败时；exit 8 + 4 (L) + 2（R0，仅当 `test_wait` 探针失败时）+ 1 |
| W7 | `mbar_try_wait` | `mbarrier.try_wait.parity` | `SYNCS.PHASECHK.TRANS64.TRYWAIT` | 2；0 (L)；4-7 | 130 + 128 (L) + 4 |
| W8 | `mbar_test_wait` | `mbarrier.test_wait.parity` | `SYNCS.PHASECHK.TRANS64` | 1 (R0) | exit：2 |
| W9 | `mbar_arrive_local` | `mbarrier.arrive.shared::cta.b64` | `SYNCS.ARRIVE.TRANS64.A1T0` | 2 (R0) | 1（32 个 lane） |
| W10 | `mbar_arrive_remote` | `mapa` + `mbarrier.arrive.shared::cluster.b64` | `PRMT 0x654` + `SYNCS.ARRIVE.TRANS64.RED.A1T0` | 0、2、4-7 (+1 R0)；0（exit 时） | 192 次到达 (224 R0) + 32 (R0 throttle)；exit 每 pair 32 + 32 |
| W11 | `mbar_arrive_cluster_masked` | 作用于 `& 0xFEFFFFFF` 的 `mbarrier.arrive.shared::cluster.b64` | `ULOP3 0xfefffff8` + `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [UR24], RZ` | 4-7 | 128（每线程一次） |
| W12 | `mbar_arrive_expect_tx_local` | `mbarrier.arrive.expect_tx.shared::cta.b64` | `@!P0 SYNCS.ARRIVE.TRANS64 RZ, [UR10], R5 = 0xc000` | 2（L，leader lane） | 128 (L) |
| W13 | `mbar_arrive_expect_tx_remote_pred` | `setp` + `@p mapa` + `@p mbarrier.arrive.expect_tx.shared::cluster.b64` | `@!P SYNCS.ARRIVE.TRANS64.RED ..., R = 0x10` | 1 (R0)，lane 0..3 执行动作 | 1 (R0) |
| W14 | `fence_mbarrier_init_release_cluster` | `fence.mbarrier_init.release.cluster` | none（`FENCE.VIEW.ASYNC.S`，推断） | 全部，start | start：4 |
| W15 | `fence_proxy_async_shared_cta` | `fence.proxy.async.shared::cta` | `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S` | 0、2、4-7 (+1 R0)；4-7 | 6 (7 R0) + 32 |
| W16 | `cluster_arrive_relaxed` | `barrier.cluster.arrive.relaxed.aligned` | `UCGABAR_ARV` | 全部，start | start：1 |
| W17 | `cluster_wait` | `barrier.cluster.wait.aligned` | `UCGABAR_WAIT; CCTL.IVALL` | 全部，start | start：1 |
| W18 | `named_bar_sync` | `bar.sync 1,128` / `bar.sync 6,160` | `BAR.SYNC.DEFER_BLOCKING 0x1,0x80` / `0x6,0xa0` | 4-7 | 16；start：1 |
| W19 | `named_bar_arrive` | `bar.arrive 6,160` | `BAR.ARV 0x6,0xa0` | 0 | start：1 |
| W20 | `prefetch_tensormap` | `prefetch.tensormap` | `UTMACCTL.PF` | 1、3（经选举） | start：4 |
| W21 | `tma_load_2sm_mcast_3d` | `cp.async.bulk.tensor.3d.cta_group::2...multicast::cluster.L2::cache_hint` | `UTMALDG.3D.MULTICAST.2CTA` | 2（经选举） | 128 |
| W22 | `tma_load_2sm_3d` | `cp.async.bulk.tensor.3d.cta_group::2...L2::cache_hint` | `UTMALDG.3D.2CTA` | 2（经选举） | 128 |
| W23 | `tma_store_3d` | `cp.async.bulk.tensor.3d.global.shared::cta.bulk_group` | `UTMASTG.3D` | 4（所有 lane） | 32 |
| W24 | `tma_store_commit_group` | `cp.async.bulk.commit_group` | `UTMACMDFLUSH` | 4 | 8 |
| W25 | `tma_store_wait_read<1>` | `cp.async.bulk.wait_group.read 1` | `DEPBAR.LE SB0, 0x1` | 4 | 8 |
| W26 | `tmem_alloc_2sm` | `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32` | `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN`（69 条指令的序列） | 0 | start：1 |
| W27 | `tmem_dealloc_2sm` | `tcgen05.dealloc.cta_group::2.sync.aligned.b32` | `UTCATOMSWS.AND`（+ 2 次检查） | 0 | exit：1 |
| W28 | `tmem_relinquish_2sm` | `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned` | `UVIRTCOUNT.DEALLOC.SMPOOL 0x80`（NVIDIA 内部） | 0 | exit：1 |
| W29 | `umma_f16_2sm` | `tcgen05.mma.cta_group::2.kind::f16` | `UTCHMMA.2CTA` | 0（L，经选举） | 512 (L) |
| W30 | `umma_commit_mcast_2sm` | `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64` | `UTCBAR.2CTA.MULTICAST` | 0（L，经选举） | 128 + 1 (L) |
| W31 | `tmem_ld_32x32b_x16` | `tcgen05.ld.sync.aligned.32x32b.x16.b32` | `LDTM.x16` | 4-7 | 32（每 warp 8） |
| W32 | `tmem_wait_ld` | `tcgen05.wait::ld.sync.aligned` | none（scoreboard） | 4-7 | 4（每 warp 1） |
| W33 | `clc_try_cancel` | `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128` | `UGETNEXTWORKID.BROADCAST` | 1（R0，经选举） | 每 cluster 1 |
| W34 | `clc_decode` | `ld.shared.b128` + `query_cancel.is_canceled` + `@p get_first_ctaid` | `LDS.128` + `LOP3` 提取 | 0、2、4-7 (+1 R0) | 6 (7 R0) |
| W35 | `griddep_wait` | `griddepcontrol.wait` | `ACQBULK` | 2；1 (R0) | start：2（rank 1-3 中为 1） |
| W36 | `griddep_launch_dependents` | `griddepcontrol.launch_dependents` | `PREEXIT` | 0 | exit：1 |
| W37 | `__syncwarp()` | `bar.warp.sync -1` | `WARPSYNC.ALL` (warp 0) / none (warp 2) | 2；0 | 1 (warp 2)；start：1 (warp 0) |

与实测的静态清单（笔记 G.3）交叉核对：4 条 `tcgen05.mma`、2 条 `tcgen05.commit`、8 条 `tcgen05.ld`、1 条 `tcgen05.wait::ld`、alloc/dealloc/relinquish 各 1 条、2 条 A 加载 + 2 条 B 加载（两个未展开的 producer 循环）、32 条 D 存储、8 条 commit_group、8 条 `wait_group.read 1`、1 条 `try_cancel`、4 处解码、2 条 `griddepcontrol.wait` + 1 条 `launch_dependents`、4 条预取、33 条初始化、4 条初始化 fence、17 条 `bar.sync` + 1 条 `bar.arrive`、12 条 `fence.proxy.async.shared::cta`、2 条 `test_wait`、14 条 `elect.sync`、8 个 `mapa` 站点。上表的每一行都是这些站点之一，或是它们的动态执行之一。

---

# 第 7 章 分发器：流水线辅助函数、clc_consume 与 scheduler warp

## 7.1 本章在文件中的位置

分发器横跨 explicit.cu 的三节，外加这几节所依赖的常量与 prologue 行：

| 部分 | 行号（用 `grep -n` 验证） | 它是什么 |
|---|---|---|
| 第 4 节横幅注释 + 辅助函数 | explicit.cu:604-616（横幅注释 :604 位于 `////` 分隔线 :603 与 :605 之间，辅助函数 :606-616） | `advance<Stages>`、`desc64` |
| 第 4 节仅用于 trace 的辅助函数 | explicit.cu:618-662 | `globaltimer_ns`、`hang_slot`、`heartbeat`、`mbar_wait_traced`、`trace_tail`（全部位于 `#if defined(CUTLASS_DEABSTRACTION_TRACE)` 之内） |
| `EXPLICIT_WAIT` 与记录种类常量 | explicit.cu:231-240 | 各角色每一次阻塞等待背后的宏 |
| 第 5 节 | explicit.cu:665-706 | `clc_consume<TraceKind>` |
| 第 7 节 | explicit.cu:810-868（横幅注释 :810-818 位于 `////` 分隔线 :809 与 :819 之间，函数 :820-868） | `scheduler_role` |
| CLC/throttle 常量 | explicit.cu:109-110, 144-145, 148-149, 151, 164, 181, 183, 185, 189, 192 | stage 数、偏移、计数、tx 字节数、wait 提示 |
| CLC 与 throttle 屏障初始化 | explicit.cu:1135-1143, 1155-1163 | prologue 中的 warp 4 与 warp 3 |
| 初始 tile | explicit.cu:1175-1176 | 启动坐标映射 |
| scheduler 的分发 | explicit.cu:1194-1196 | `warp_idx == 1 && is_cta0` |
| 四个消费调用点 | explicit.cu:793, 853, 913, 1028 | producer、scheduler、MMA、epilogue |

第 6 章已经逐字符讲解了这几节所调用的 wrapper。本章不再重新引用它们中的任何一个。函数体*本身就是*协议要点的两个是 W13，即 lane 到 CTA 的装填谓词（6.3.8），以及 W34，即本章规定其契约的解码（6.8.3）。其余的，连同它们在第 6 章中的小节：W2（`elect_one_sync`，第 6.2.2 节：`elect.sync` 使谓词恰好在已收敛 warp 的一个 lane 中为真，这一点结合 7.6.1 中仅 rank 0 的分发，正是使“每个 cluster 一次 `try_cancel`”成立的原因）、W6/W8（阻塞等待与非阻塞测试，第 6.3.3 节和第 6.3.5 节；W7，即 6.3.4 的 peek，不被这三节调用）、W10（远程 arrive，6.3.6）、W15（proxy fence，6.4.2）、W33（`try_cancel`，6.8.2）、W35（`griddepcontrol.wait`，6.9.1）。有一个实参以不寻常的方式被复用：`lane = threadIdx.x & 31u`（explicit.cu:1102）被传入 `scheduler_role`，并作为 *CTA id* 交给 W13，于是对 lane 0..3 而言，lane i 装填 CTA i；谓词实参 `lane < 4u` 使其余 28 个 lane 不执行 arrive（第 7.6.5 节）。

本章用到的常量，逐字摘自第 1 节（stage 深度，explicit.cu:109-110，已在第 5.5 节引用；CLC/throttle 屏障偏移 :144-145 与 :148-149，已在第 5.7 节引用）：

```
constexpr uint32_t OFF_CLC_RESPONSE   =    352;   // 2 x 16 B, alignas(16)
```
(explicit.cu:151)

```
constexpr uint32_t kClcTxBytes       = 16;
```
(explicit.cu:181)

```
constexpr uint32_t kClcFullCount     = 1, kClcEmptyCount  = 800;    // 32 + 4 * (32 + 128 + 32); beta = 0
```
(explicit.cu:183)

```
constexpr uint32_t kThrottleCount    = 32, kTmemDeallocCount = 32;
```
(explicit.cu:185)

```
constexpr uint32_t kWaitTicks        = 0x989680u;                    // ClusterBarrier::wait suspend-time hint (barrier.h:415)
```
(explicit.cu:189)

这些计数 (1, 800, 32)、stage 深度 (2) 与 16 个事务字节在第 7.3-7.6 节和第 7.8 节中推导。五个字节偏移 (208, 224, 304, 320, 352) *不*在此推导。它们遵循 CUTLASS 的 `SharedStorage` 成员顺序，由第 5 章的共享内存映射固定（explicit.cu:137-166，其中 :164 处有 `OFF_CLC_RESPONSE % 16 == 0` 断言）；本章只是使用它们。`kWaitTicks` 在 7.2.3 中解释。

---

## 7.2 第 4 节：流水线状态约定、`desc64` 与 wait 形式

### 7.2.1 `{index, phase}` 与 `advance<Stages>`

```
// E.6.5  Pipeline state: {index, phase}; producers start {0, 1}, consumers {0, 0} (sm90_pipeline.hpp:254-260); operator++ (:204-213).
// Why the producer starts at phase 1: a freshly initialised mbarrier is in phase 0, and try_wait.parity with parity 1 returns true at once
// ("the previous phase is complete"), so the first pass over the 8 empty[] barriers never blocks; the consumer waits parity 0 on full[]
// and blocks until the first arrival + transaction bytes complete phase 0. After every wrap of the ring the phase bit flips.
template <uint32_t Stages>
__device__ __forceinline__ void advance(uint32_t& idx, uint32_t& phase) {
  if (++idx == Stages) { idx = 0; phase ^= 1; }
}
```
(explicit.cu:606-613)

这就是 CUTLASS 的 `PipelineState::operator++` 的全部，只是去掉了 `count_` 成员和 `if constexpr (Stages > 0)` 守卫（在这里无关紧要：每个 `Stages` 都是 8、4 或 2）：

```
  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }
```
以及 producer 的起始状态：

```
template<class Pipeline>
CUTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerCount};
}
```
因为落后两个 phase 的 wait 超出了已定义的情形（第 4 章，第 4.2.3 节），这个 kernel 中的每个流水线状态每次使用都恰好推进一次，而一次被跳过的消费（7.4.2）是致命的，而非延迟。

**为什么 explicit kernel 丢弃 `count_`。** CUTLASS 在每个 `PipelineState` 中都携带 `count_`（由 epilogue collective 的加载与存储簿记读取，sm100_epilogue_tma_warpspecialized.hpp:538 与 :791；由 `PipelineTmaStore` 读取，sm90_pipeline.hpp:676-682；以及由 trace 探针读取）。explicit kernel 只保留 `clc_count`，而且只是因为 trace 记录格式（K5b/K5c）需要它。在生产构建中，它被递增后立即被作废（`++clc_count; (void)clc_count;`，explicit.cu:689-690），所以编译器可以把它作为死代码移除（推断，未在 SASS 中核验）。被丢弃的 `count_` 只不过是少了一个 uniform 值，仅此而已。笔记 G.3（"Why 26 and not 68"）把 explicit kernel 的 26 个向量寄存器相对 baseline 的 68 个的差别归因于零填充的 C 片段（16 个向量寄存器）、`epi_load` 流水线状态、为 scheduler 簿记而保存在向量寄存器中的每个 tile 的 CLC 解码，以及 44 对 `MOV.SPILL`/`R2UR.FILL` uniform 到向量的溢出/填回。它并没有把它们归因于 `count_`，而后者既然是 warp-uniform 的，本来就会放在一个 uniform 寄存器中。

### 7.2.2 `desc64`

```
// 64-bit descriptor formed at the asm boundary from the constant high word and the 32-bit low word (D.2: ULEA/UIADD3 on the low word, UMOV of the high word)
__device__ __forceinline__ uint64_t desc64(uint32_t hi, uint32_t lo) { return (uint64_t(hi) << 32) | uint64_t(lo); }
```
(explicit.cu:615-616)

只有 `mma_role` 使用它（explicit.cu:933-934），用来把常量高字 `kSmemDescHi = 0x40004040` 与一个 per-stage、per-k-block 的低字拼接起来。究竟为什么要把描述符拆开：baseline 的 SASS 正是这么做的（笔记 D.2：对低字做 `ULEA`/`UIADD3`，对高字做 `UMOV URn, 0x40004040`）。一次 64 位加法会是*更差*的匹配，而且可能每个 k-block 耗费一次 64 位 uniform 加法。第 9 章推导低字算术；这里要紧的只是 `desc64` 是一个没有内存副作用的纯位运算。

### 7.2.3 `EXPLICIT_WAIT` 与三种 wait 形式

在性能二进制文件中（`CUTLASS_DEABSTRACTION_TRACE` 未定义；开关是 util.hpp 的第 1 行，已注释掉），这个宏（explicit.cu:231-240，引文见第 5.11 节）在文本上就是 `mbar_wait((addr), (parity))`。`site`、`tile` 与 `q` 实参甚至不会被求值，所以 `WAIT_CLC_FULL` 和其他 site 常量不必存在（它们只在开关之下声明，explicit.cu:212-215），而 `tile`/`q` 不耗费任何指令。

第 4 章，第 4.2.4 节以表格列出了三种 wait 形式；本章特有的那一种：

| Wrapper | PTX | 是否阻塞？ | 时间限制 | 用途 |
|---|---|---|---|---|
| W8 `mbar_test_wait` | `@P2 mbarrier.test_wait.parity ... [addr], parity` + `selp` | 否，从不挂起 | 无 | 仅 CLC producer 尾段 (explicit.cu:860) |

为什么 W6 把挂起提示 `kWaitTicks` 放在*寄存器*中而不是立即数中：它是从 barrier.h:415-427 复制来的（asm 文本逐字复制；字面量 `0x989680` 变为 `kWaitTicks`），而笔记第 7.2 节记录了编译器用 `mov.b32 %r, 10000000` 来物化它。为什么是这个值：barrier.h:414 用一行话说明了它，"// Arbitrarily large timer value after which try-wait expires and re-tries." 它是复制来的，不是调优出来的。

**`mbar_test_wait` 的第三个实参 `1u`（explicit.cu:860）。** W8（explicit.cu:329-343，第 6.3.5 节）保留了 CUTLASS 的 `test_wait(phase, pred)` 谓词操作数，即 `setp.eq.u32 P2, %3, 1;` 后接 `@P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;`，使得 PTX 的形态与 baseline 的 `ClusterBarrier::test_wait` 一致。explicit kernel 总是传入 `1u`，所以这个测试是无条件的。ptxas 把该谓词折叠掉，SASS 就是不带 `.TRYWAIT` 的普通 `SYNCS.PHASECHK.TRANS64`（笔记第 7.2 节，总结 D.2："`test_wait.parity` is the plain `SYNCS.PHASECHK.TRANS64`"；D.2 把 baseline 的 CLC 尾段中的两处 `test_wait` 调用点列为仅有的调用点，而 G.3 记录了 explicit kernel 的 "tail `SYNCS.PHASECHK.TRANS64` x2 (`test_wait`) as in the baseline"）。

各角色中的全部十二个 `EXPLICIT_WAIT` 调用点（由 `grep -n EXPLICIT_WAIT` 得出），其中属于分发器的四个 (:680, :836, :841, :861)，都已在第 4 章中逐个屏障地讲解（第 4.3.1、4.4、4.5 和 4.8 节）。

---

## 7.3 硬件契约：CLC 承诺了什么

在阅读 `clc_consume` 之前，你需要了解 kernel 所依赖的语义。第 2.12 节完整引用了硬件契约：ISA 关于 `try_cancel` 与响应的句子、两条未定义行为规则，以及 CUTLASS 的调度模型；W33/W34 本身则在第 6.8.2 节和第 6.8.3 节中被引用。本小节只保留第 7.4-7.6 节直接用到的要点，每一点用一两行陈述，并补充三条第 2.12 节未引用的 ISA 句子：通用 proxy 那一句、退出规则（也在第 2.9.4 节中引用），以及未指定的第四个元素。

1. **`try_cancel` 做什么**（完整引文见第 2.12.1 节）。PTX ISA："This instruction accesses its mbarrier operand using generic-proxy." 网格以每个 256 x 256 输出 tile 一个 cluster 的方式启动（32 x 32 = 1024 个 cluster，第 3 章）。这句 ISA 语句对 7.4.4 很重要：`try_cancel` 的 *mbarrier* 一侧是通用 proxy 的，而*响应写入*是异步 proxy 的（第 2 章，第 2.12.1 节），所以跨 proxy 的位置是槽位，而不是屏障。
2. **响应包含什么。** PTX ISA："On success, the opaque response contains the ctaid of the first CTA of the canceled cluster; no other successful response from other clusterlaunchcontrol.try_cancel operations from the same grid will contain that id." 第二个分句就是整个动态调度所依赖的无重复保证：kernel 永远不需要对 tile 去重。
3. **Multicast、完成与退出规则。** multicast 投递规则与每个 CTA 的完成规则已在第 2 章，第 2.12.1 节引用；ISA 还补充了一条退出规则："The behavior of instruction with .multicast::cluster::all qualifier is undefined if any CTA in the cluster is exited." 只要 rank 0 的 scheduler 仍能发射一条 multicast `try_cancel`，任何 CTA 都不得退出。kernel 满足这一点：每个 CTA 中的每个角色都在离开其循环之前消费最终的（未取消的）响应，而那个响应回答的是 scheduler 所发出的最后一次查询。所以最后一次 `try_cancel` 先于 cluster 中的每一次退出（7.6.8；笔记 7.10 给出了 mbarrier 一侧的原因）。
4. **大小。** CUTLASS 把响应建模为 `struct CLCResponse { uint32_t data[4] = {0}; };`，并附注释 "CLC response is an opaque 16B value"（cutlass/gemm/kernel/sm100_tile_scheduler.hpp:73-74），以 `alignas(16)` 存储（`:95`）；kernel 设置 `transaction_bytes = CLCResponseSize = sizeof(CLCResponse)`（sm100_gemm_tma_warpspecialized.hpp:155, :516）。
5. **解码。** PTX ISA（9.7.15.19）："The contents of the 4th element are unspecified"（这就是 W34 的 `{%0, %1, %2, _}` 中的 `_`），以及 "If the request fails the behavior of clusterlaunchcontrol.query_cancel.get_first_ctaid is undefined." Colfax 的文章由此得出实用规则："if `.is_canceled` returns false then `query_cancel` instructions other than `.is_canceled` results in undefined behavior, so we should always start with `.is_canceled`."
6. **每个 cluster 一个请求者。** Colfax 的文章："with `try_cancel` only one thread per cluster should be used. In particular, multiple threads submitting `try_cancel` will result in multiple clusters being cancelled." Programming Guide（4.13.1.1）："The preferred way to use Cluster Launch Control is from a single thread, i.e., one request at a time." 以及 "Since thread block cancellation is a uniform instruction, it is recommended to submit it inside invoke_one thread selector." 这就是为什么发射放在 `elect_one_sync()`（W2）之下，以及为什么只有 rank 0 运行 scheduler（7.6.1、7.6.6）。
7. **观察到失败之后不再请求。** 这条 ISA 规则已在第 2 章，第 2.12.1 节引用，而第 2.12.3 节展示了 kernel 在何处遵守它。
8. **首 CTA id 按 cluster 对齐。** Programming Guide（4.13.2.2，cluster 情形）："The result received by all thread blocks corresponds to the local block index `{0, 0, 0}` within a cluster." 当 cluster 为 (2,2,1)，且网格在每一维上都是 cluster 的整数倍时（`check_cluster_dims`，explicit.cu 的 run()；util.hpp 中的 `static_assert(kGridX % kClusterX == 0 && ...)`），任何 cluster 的第一个 CTA 都有偶数的 `blockIdx.x` 和偶数的 `blockIdx.y`。

**这 16 个字节，按编译器读取它们的方式。** kernel 只通过两条 `query_cancel` 指令（W34）读取响应，与 CUTLASS 完全一样，所以 ptxas 会吸收未来的 layout 变更（由 SASS 推断出的 layout 见第 2 章，第 2.12.2 节）。

---

## 7.4 第 5 节：逐行解读 `clc_consume`

横幅注释与函数，逐字如下：

```
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
  advance<kClcStages>(clc_idx, clc_phase);
  ++clc_count;
  (void)clc_count;
  if (valid) {
    tm_next = static_cast<int>(2u * (y0 >> 1) + (x0 & 1u) + x);   // swizzle_and_rasterize + possibly_transpose_work_tile (AlongN), B.4
    tn_next = static_cast<int>(2u * (x0 >> 1) + (y0 & 1u) + y);
  }
#if defined(CUTLASS_DEABSTRACTION_TRACE)   // [TRACE]
  ...   // [TRACE] branch elided (compiled out in the performance binary)
#else
  (void)TraceKind;
#endif
  return valid != 0u;
}
```
(explicit.cu:664-706)

它是 CUTLASS 的 `fetch_next_work` 的显式形式：

```
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[scheduler_pipe_consumer_state.index()]);
    auto work_tile = work_tile_info_from_clc_response(smem_addr);
```
```
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);

    work_tile = swizzle_and_rasterize(
      work_tile.M_idx, work_tile.N_idx, work_tile.L_idx, work_tile.is_valid(),
      block_id_in_cluster_.x, block_id_in_cluster_.y);
```
### 7.4.1 签名 (explicit.cu:677-679)

- `template <uint32_t TraceKind>`：生产中为 0。这个参数存在的唯一目的，是让两个带 trace 的调用点（scheduler warp、MMA warp）能从同一个函数体发出带不同标签的记录（K5b vs K5c）。`if constexpr (TraceKind != 0u)` 位于 `#if` 之内，而 `#else` 分支把该参数作废，所以生产实例化 `clc_consume<0u>` 与两个具名种类的实例化（在生产中也是 0，explicit.cu:238-239）是相同的代码。
- `smem_base`：CTA 自己的 32 位 `shared::cta` 基址（E.6.1 地址空间规则：每个 asm 操作数都是 `smem_base + OFF_*`）。
- `x, y`：`%cluster_ctaid`，只为映射所需。
- `clc_idx, clc_phase, clc_count` 按引用传递：调用者的 CLC consumer 状态，在此推进。
- `tm_next, tn_next` 按引用传递：**只**在 `valid` 之下写入。调用者把当前 tile 保存在 `tm, tn` 中，并在其循环体末尾把 `tm_next, tn_next` 复制过去（explicit.cu:794, 854, 947, 1073）。调用者必须初始化它们（`int tm_next = tm, tn_next = tn;`，位于 :773, 828, 906, 1021），这样，未取消响应之后的那次无条件复制所复制的是一个已定义的（而且是死的）值，而不是一个不确定的值（笔记 E.6.4）。
- `tile`：仅用于 trace 的 `EXPLICIT_WAIT` 实参；在生产中消失。
- 返回 `valid != 0`：`true` = 窃取到了一个 tile（继续循环），`false` = 队列已空（每个角色退出其循环）。

### 7.4.2 第 680 行：等待 CTA *自己的* `clc_full[idx]`

`EXPLICIT_WAIT(smem_base + OFF_CLC_FULL + 8u * clc_idx, clc_phase, ...)` 是作用在地址 `smem_base + 208 + 8 idx` 上、parity 为 `clc_phase` 的 W6。

- 为什么是**自己的**：7.3 第 3 项。硬件在发出请求的 cluster 的*每一个* CTA 中的同一偏移处完成 full 屏障。在 full 这一侧，没有任何 CTA 把响应转发给另一个 CTA。CTA 1-3 的 full 屏障完全由远程动作完成（rank 0 的 `arrive.expect_tx` 加上 multicast 的字节，笔记 7.4），consumer 只是在本地等待。
- 为什么该屏障以计数 **1 (+16 tx)** 初始化：`producer_arv_count = 1`（sm100_gemm_tma_warpspecialized.hpp:510），因为每个 CTA 每次查询恰好有一个线程到达（rank-0 warp 1 的 lane i 通过 `mapa` 装填 CTA i，第 7.6.5 节），而事务计数就是响应的 16 个字节。phase 在待到达计数降到零**并且** tx-count 降到零时完成（PTX ISA，mbarrier 一节；Programming Guide 的 "when the countdown reaches zero, the barrier is complete for the current phase" 是其中到达的那一半）。所以在装填到达与硬件的 16 个字节两者都发生之前（不论先后次序），phase 都不可能完成。
- 为什么 parity 是 `clc_phase`：响应 n 位于槽位 n % 2 中，并完成 `full[n % 2]` 的第 n / 2 个 phase；经过 n 次推进后的 consumer 状态是 `{n % 2, (n / 2) & 1}`（7.2.1）。
- 这个 CTA 等待哪些响应：全部，并且按顺序。跳过一个的 consumer warp 在 kernel 的其余部分都会失步，最终要么等待一个永远不会到来的 parity（死锁），要么过早读取一个槽位。笔记 E.6.4："A role that skips one consume deadlocks CTA 0's empty barrier."

### 7.4.3 第 681-682 行：解码（W34，引文见第 6.8.3 节）

- `ld.shared.b128 clc_result, [%4]` 把位于 `smem_base + 352 + 16 idx` 的 16 字节槽位读入一个 128 位寄存器。一次 `.b128` 加载需要 16 字节对齐的地址：CUTLASS 把槽位声明为 `alignas(16)`（sm100_tile_scheduler.hpp:95），explicit kernel 断言 `OFF_CLC_RESPONSE % 16 == 0`（explicit.cu:164），而 352 = 22 x 16。
- `@p1 get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}`：第四个分量被丢弃（`_`）。
- 微妙之处：输出被声明为 `"=r"`（只写）。对于未取消的响应，这条带谓词的指令不会写它们，所以 `x0, y0, z0` 之后持有的是被分配的寄存器此前持有的任何值：explicit.cu:681 处的 `= 0` 初始化**不会**在 asm 之后幸存（一个 `"=r"` 操作数不必是先前持有那个 0 的寄存器）。实测（笔记 D.3）："In all 33 K5b and 132 K5c records with `valid = 0`, the raw M/N/L fields equal the *previous* response's decoded first-CTA id"。这正是第 691 行用 `if (valid)` 守卫映射的原因，也是 trace 记录写入 `valid ? uint32_t(tm_next) : 0u` 的原因。初始化为 0 是卫生做法（在 asm 之前有一个已定义的值，不会有未初始化变量的诊断），仅此而已。
- 为什么没有 `elect`：warp 的每个 lane 都执行加载和解码。结果是 warp-uniform 的（同一个槽位），加载是来自共享内存的一次广播，而且 CUTLASS 也是这样做的。保持 per-lane 形式避免了用一次 shuffle 把 `valid`/`x0`/`y0` 广播给其他 31 个 lane，而它们全都需要这些值：producer warp 的各 lane 从 `tm, tn` 计算 TMA 坐标，epilogue 的各 lane 计算存储坐标。在 *baseline 的* SASS 中，MMA warp 的解码只保留 `valid` 位（笔记 D.2，关于那五处 `LDS.128` 解码："in the MMA warp only the valid bit survives (`get_first_ctaid` eliminated as dead)"）。由于 `mma_role` 把 `tm, tn` 作废（explicit.cu:883），预期 explicit kernel 中会有同样的消除。但笔记 G.3 只报告了清单（PTX 中 `is_canceled` 4 处和 `get_first_ctaid` 4 处，SASS 中 4 条 `LDS.128`），并未对此作任何陈述：是预期，而非报告。如果你需要这个事实，请用 `cuobjdump -sass` 核对 warp 0 的 `LDS.128` 附近。

### 7.4.4 第 683 行：`fence.proxy.async.shared::cta`

W15 `fence_proxy_async_shared_cta()`（explicit.cu:418-425，第 6.4.2 节）是单独一条指令 `fence.proxy.async.shared::cta;`，带有一个 `"memory"` clobber；其 asm 是 `cutlass::arch::fence_view_async_shared`，cutlass/arch/barrier.h:728-736。

CUTLASS 在解码 asm 之后紧接着发射它：

```
      cutlass::arch::fence_view_async_shared();
```
**关于 proxy 的推理。** 两个 proxy 交替访问我们的槽位：

```
   async proxy      generic proxy        async proxy
   try_cancel n  -> ld.shared.b128   ->  try_cancel n+2        (same slot: n % 2 == (n+2) % 2)
   (hw writes)      (800 reads/cluster)  (hw overwrites)
```

该 fence 保护的是读后写这条边（第 2 章，第 2.7.6 节）：执行消费的线程把它的通用读取排在它此后在异步 proxy 视角中所做的任何事情之前，然后通过对 CTA 0 的 empty 屏障的 release arrive 向 producer 发出信号。因此 producer 在该屏障上的 acquire（它的 `wait own empty[idx]`）及其随后的 `try_cancel` 都在每个 consumer 的读取之后。Colfax 的文章把 CUTLASS 中的同一行解释为 "guarding against the next iteration's `try_cancel` from overwriting the SMEM before the current iteration is able to finish reading from that location." 没有这个 fence，内存模型将允许该通用读取在异步 proxy 的覆写之后被满足，即使在程序顺序中 arrive 是在读取之后。

代价：每个 warp 每个 tile 一条 `fence.proxy.async`。baseline 的 SASS 有 13 对 `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`（5 处 CLC 解码点 + 8 个 epilogue subtile，笔记 D.2）；explicit kernel 有 12 对（笔记 G.3），因为它有 4 处消费点。按每 ~28 us 一次的频率，它在计时中不可见。

### 7.4.5 第 684 行：对 CTA 0 的 `clc_empty[idx]` 的远程释放（计数 800）

W10 `mbar_arrive_remote(smem_addr, cta_id)`（explicit.cu:357-367，第 6.3.6 节）是两条指令，先是 `mapa.shared::cluster.u32 remAddr32, %0, %1;`，然后是 `mbarrier.arrive.shared::cluster.b64 _, [remAddr32];`，即 `ClusterBarrier::arrive(smem_ptr, cta_id, pred)` 的 `pred` 分支，cutlass/arch/barrier.h:486-499。

`mapa.shared::cluster.u32 remAddr32, addr, 0` 把这个 CTA 的 `shared::cta` 地址 `smem_base + 224 + 8 idx` 转换为 CTA 0 中同一偏移的 `shared::cluster` 地址；随后 `mbarrier.arrive.shared::cluster` 在那里到达。在 CTA 0 自身中，`mapa(addr, 0)` 是恒等映射。在 SASS 中 `mapa` 不是一条指令：rank 被插入到地址的字节 3 中（`PRMT ..., 0x654`，笔记 D.2），这是 rank 位于地址位 [24,32) 这一事实的编译器可见形式。

**为什么是 CTA 0 的屏障，而且只是 CTA 0 的。** CUTLASS 中这条流水线的 `consumer_release` 是

```
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(params_.producer_blockid);
  }
```
其中 `clc_pipeline_params.producer_blockid = 0;`（sm100_gemm_tma_warpspecialized.hpp:509）。producer（scheduler warp）只存在于 rank 0 中，并在*它自己的* empty 屏障上等待（7.6.5），所以每个 CTA 中的 consumer 都必须把它们的释放发送给 rank 0。rank 1-3 的 empty 屏障像 rank 0 的一样以计数 800 初始化（explicit.cu:1140 在每个 CTA 中运行），但从未被到达过，也从未被等待过（笔记 7.4："their empty barriers are initialized but never used"）。

**为什么是 800。** CUTLASS 的计算是

```
    clc_pipeline_params.producer_arv_count = 1;
    clc_pipeline_params.consumer_arv_count = NumSchedThreads + cluster_size *
                                                 (NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads);
    if (is_epi_load_needed) {
      clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
    }
```
其中 `NumSchedThreads = NumMMAThreads = NumMainloopLoadThreads = NumEpilogueLoadThreads = 32`（各一个 warp，`:137-140`）且 `NumEpilogueThreads = CollectiveEpilogue::ThreadCount = 128`（`:141`；四个 warp）。对于 beta = 0，`is_epi_load_needed` 为 false，所以

```
800 = 32 (rank-0 warp 1) + 4 CTAs x (32 (warp 2) + 128 (warps 4-7) + 32 (warp 0))
```

带 warp 3 的 928 变体是 beta != 0 的路径；baseline 在运行时用一条 `selp.b32 928, 800` 在两者之间选择（笔记 D.2）。explicit kernel 硬编码了 800，并在 explicit.cu:192 处断言该公式。每个响应究竟由谁到达：

| 贡献者 | CTA | 每 CTA 线程数 | 到达次数 | 代码中的位置 |
|---|---|---|---|---|
| scheduler warp（warp 1） | 仅 rank 0 | 32 | 32 | explicit.cu:853 |
| producer warp（warp 2） | 全部四个 | 32 | 128 | explicit.cu:793 |
| MMA warp（warp 0），leader **和** peer | 全部四个 | 32 | 128 | explicit.cu:913 |
| epilogue warp 4-7 | 全部四个 | 128 | 512 | explicit.cu:1028 |
| **总计** | | | **800** | |

**为什么 empty 计数是*线程*计数（800）而不是 warp 计数（25）。** 从逻辑上说，每个 warp 一次被选中的到达就足够了。kernel 使用每一个 lane 有三个原因。(1) CUTLASS 的 `consumer_release`（见上）没有选举：每个 consumer warp 的每个 lane 都执行 `mbarrier.arrive`，所以计数必须是 `consumer_arv_count` 个*线程*，而 explicit kernel 复现的是源码的协议，而不是自己发明一个。(2) 一次被选中的释放只有在 warp 的每个 lane 都读完槽位*之后*才是正确的，所以它需要在那单次 arrive 之前加一个 `__syncwarp()`（或者对一个“完成”标志做一次 shuffle），这会给每个 consumer warp 每个 tile 增加一条指令和一个收敛点。在全 lane 形式中，每个 lane 自己的 arrive 在程序顺序上跟在它自己的读取和 fence 之后，两者都不需要。(3) 它会改变笔记用来与 baseline 核对的 PTX 清单（那 8 处 `mapa` 及其 arrive 会保留，但 4 处消费点中的每一处都会多出一条 `elect.sync` 以及一条 `bar.warp.sync` 或 `shfl.sync`，而物化出来的 CLC empty 计数会读作 25 而不是 800；笔记 G.3）。代价是每个 cluster 每个 tile 800 次远程到达，每 ~28 us 一次：可以忽略。

为什么释放发生在映射*之前*（第 684 行 vs 第 691 行）：映射是对已经从槽位中提取出的值所做的寄存器运算，所以一旦读取和 fence 完成，槽位就可以交还，这已是尽可能早的时机。CUTLASS 出于同样的原因把 `consumer_release` 排在 `swizzle_and_rasterize` 之前（sm100_tile_scheduler.hpp:465-469）。

### 7.4.6 第 688-690 行：状态推进与死计数器

`advance<kClcStages>(clc_idx, clc_phase)` 移到另一个槽位，并每隔一个响应翻转一次 phase（7.2.1）。`++clc_count; (void)clc_count;` 是 trace 的 `state.count`，在生产版本中是死代码（7.2.1）。

### 7.4.7 第 691-694 行：映射

推导见第 7.5 节。它由 `valid` 守卫（7.4.3），并且使用的是 *consumer 自己的* `(x, y)`：同一个响应在四个 CTA 中产生四个不同的 `(tm, tn)`，各为 256 x 256 cluster tile 的一个象限。

---

## 7.5 从 `(x0, y0)` 到 `(tm, tn)`：从 CUTLASS 推导映射

### 7.5.1 CUTLASS 路径

`fetch_next_work` 把解码出的 id 传给 `swizzle_and_rasterize(M_idx = x0, N_idx = y0, L_idx = z0, valid, block_id_in_cluster_.x, block_id_in_cluster_.y)`（sm100_tile_scheduler.hpp:467-469）。在该函数内部（`:695-814`，函数名在 :697 处），第一个块是 swizzle：

```
    // Swizzling is enabled if the swizzle size is greater than 0
    if (params_.divmod_swizzle_size_.divisor > 0) {
```
我们的运行有 `max_swizzle_size = 0` -> 除数 0（主机端守卫 `params.swizzle_divisor != 0` 拒绝任何其他值，explicit.cu run()），所以整个块被跳过，`cta_coord_m = x0`、`cta_coord_n = y0` 保持不变。然后：

```
    // Since we swap the grid x and y modes if raster order is AlongN, swap the M and N tile offsets when
    // raster order is AlongN.
    auto [new_cta_coord_m, new_cta_coord_n] = possibly_transpose_work_tile(
      params_.raster_order_, cta_coord_m, cta_coord_n, params_.divmod_cluster_shape_m_, params_.divmod_cluster_shape_n_);

    new_cta_coord_m += cta_in_cluster_offset_m;
    new_cta_coord_n += cta_in_cluster_offset_n;
```
其中

```
  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t>
  possibly_transpose_work_tile(RasterOrder raster_order, int32_t M_idx, int32_t N_idx, FastDivmod divmod_cluster_shape_m, FastDivmod divmod_cluster_shape_n) {
    if (raster_order == RasterOrder::AlongN) {
      int cluster_m, remainder_m, cluster_n, remainder_n;
      divmod_cluster_shape_m(cluster_m, remainder_m, M_idx);
      divmod_cluster_shape_n(cluster_n, remainder_n, N_idx);
      M_idx = cluster_n * divmod_cluster_shape_m.divisor + remainder_m;
      N_idx = cluster_m * divmod_cluster_shape_n.divisor + remainder_n;
    }
    return cute::make_tuple(M_idx, N_idx);
  }
```
### 7.5.2 推导

两个 cluster 除数都是 2（cluster (2,2,1)；`cluster_divisors 2 2` 为实测，笔记 D.1 D10）。代入 `M_idx = x0`、`N_idx = y0`：

```
cluster_m = x0 / 2,  remainder_m = x0 % 2
cluster_n = y0 / 2,  remainder_n = y0 % 2
M_idx' = cluster_n * 2 + remainder_m = 2 (y0 / 2) + x0 % 2
N_idx' = cluster_m * 2 + remainder_n = 2 (x0 / 2) + y0 % 2
tm = M_idx' + x,   tn = N_idx' + y          (cta_in_cluster_offset = block_id_in_cluster = (x, y))
```

这正是 explicit.cu:692-693，其中以 `>> 1` 代替 `/ 2`、以 `& 1` 代替 `% 2`。由于 `x0` 和 `y0` 是偶数（7.3 第 8 项），余数消失，映射坍缩为

```
(tm, tn) = (y0 + x, x0 + y)
```

这是笔记在 7.4、D.3 和 D.7 中使用的形式（其 B.4 表给出的是一般形式）。explicit kernel 保留一般形式有两个原因：它是源码算术的逐字复现（在 SASS 中，一般形式每个 tile 多花四条 `ULOP3`，位于 0x1490 和 0x14b0-0x14d0，相对于 27.8 us 的 tile 可以忽略），而且万一有非 cluster 对齐的 id 到来，它仍然正确。

**初始 tile** 是把同一个函数施加于启动坐标、且 cluster 内偏移为零的结果（`initial_work_tile_info`，cutlass/gemm/kernel/sm100_tile_scheduler.hpp:368-371；第 11 章，第 11.6.3 节逐环节推导了它），所以 `tm = 2 (blockIdx.y / 2) + blockIdx.x % 2`、`tn = 2 (blockIdx.x / 2) + blockIdx.y % 2`（explicit.cu:1175-1176，引文见第 11.6 节）。

对初始 tile 而言偏移为零，因为 `blockIdx` 已经包含了 CTA 在其 cluster 内的位置（`blockIdx.x % 2 == x`、`blockIdx.y % 2 == y`）。CLC id 是*第一个* CTA 的 id，所以对于 CLC tile，必须把偏移加回去。

### 7.5.3 为什么会有转置

光栅顺序在主机端由以下代码选择

```
    if (raster_order_option == RasterOrderOptions::Heuristic) {
      if (tiles_n > tiles_m) {
        return RasterOrder::AlongM;
      }
      else {
        return RasterOrder::AlongN;
      }
    }
```
参数是 *cluster*-tile 计数，这里是 32 对 32，平局给出 **AlongN**（实测 `raster_order AlongN`，笔记 D.1 D10；explicit kernel 的主机端守卫要求 `raster_order == kRasterOrderAlongN`）。第 11 章，第 11.6.3 节从 `get_tiled_cta_shape_mnl` 推导出未转置的 `(64, 64, 1)` 网格（M 上的 `x 2` 是 2SM 的 `AtomThrShape`）、启发式规则所比较的 cluster-tile 计数，以及 65535 的 grid-Y 守卫。然后启动网格被转置：

```
    if (raster_order == RasterOrder::AlongN) {
      // Swap grid.x and grid.y for AlongN rasterization order, since the CLC scheduler
      // will schedule in AlongM order by default.
```
交换之后 `grid = (64, 64, 1)`，且 **grid.x 索引 N、grid.y 索引 M**。实测（笔记 D.3），硬件大致按启动线性顺序分发 cluster，即 `blockIdx.x` 变化最快。这是一个观察结果，而非有文档记载的保证；7.6.9 给出了数字，以及为什么 kernel 不得依赖它（笔记 D.7）。把 N 放在 x 上使得连续的 cluster 在 M 的一个 256 行条带内沿 N 扫过，这就是 "AlongN" 的含义。代价是每一个 id，不论来自启动还是 CLC，到达时其 cluster 坐标都是 (N, M) 顺序，必须被转置回 (tm, tn)：这就是 `possibly_transpose_work_tile`。

### 7.5.4 启动线性 cluster id，以及 `(tm, tn)` 与它的关系

定义 cluster 坐标 `(CX, CY) = (x0 / 2, y0 / 2)` 以及启动线性 cluster id

```
L = CY * 32 + CX = (y0 / 2) * 32 + x0 / 2
```

（笔记 D.3 使用这个 id；kernel 从不计算它）。这个 32 是 AlongN 转置之后每个网格行中的 cluster 数，`gridDim.x / clusterDim.x = 64 / 2 = 32`（网格 x 索引 N）。因此 L 是该 cluster 在硬件启动 cluster 的顺序中所处的位置，x 最快，然后是 y：即启动线性 id。硬件按 L 递增的顺序行走（近似如此，而且是实测而非保证；7.6.9）。cluster tile 以 256 x 256 为单位是 `(m, n) = (L / 32, L % 32) = (CY, CX)`，而四个 CTA 的 128 x 128 tile 是

| rank r | (x, y) | tm | tn | A / D 的行 | B / D 的列 |
|---|---|---|---|---|---|
| 0 | (0, 0) | y0 = 2 CY | x0 = 2 CX | [256 CY, +128) | [256 CX, +128) |
| 1 | (1, 0) | y0 + 1 | x0 | [256 CY + 128, +128) | [256 CX, +128) |
| 2 | (0, 1) | y0 | x0 + 1 | [256 CY, +128) | [256 CX + 128, +128) |
| 3 | (1, 1) | y0 + 1 | x0 + 1 | [256 CY + 128, +128) | [256 CX + 128, +128) |

对照第 8 章的几何来读这张表。MMA pair {2y, 2y + 1} = {0, 1}（y = 0）共享 `tn = x0`，而 `tm` 相差 1；这就是为什么一条 `tcgen05.mma.cta_group::2` 能计算 256 行（两个 CTA 的 A）x 128 列（两个 CTA 中同一 `tn` 的 64 列 B 半块）。rank 0 和 2（x = 0）共享 `tm`；这就是为什么 A 以掩码 `0x5 << x` 跨 y 被 multicast。所以一个 CTA 所收到的每一个 tile 都满足 `tm % 2 == x`（explicit.cu:777 依赖于此），而两个 pair 的 `tn` 恰好相差 1。

### 7.5.5 算例

**算例 1（注释中的那个，explicit.cu:672）。** 响应 `(x0, y0) = (10, 6)` 在 CTA `(x, y) = (1, 0)`（rank 1）中被消费：

```
tm = 2 * (6 >> 1) + (10 & 1) + 1 = 2 * 3 + 0 + 1 = 7      -> A rows / D rows   [128 * 7, 128 * 8) = [896, 1024)
tn = 2 * (10 >> 1) + (6 & 1) + 0 = 2 * 5 + 0 + 0 = 10     -> D columns         [128 * 10, 128 * 11) = [1280, 1408)
L  = (6 / 2) * 32 + 10 / 2 = 3 * 32 + 5 = 101; cluster tile (m, n) = (3, 5)
```

同一 cluster 的另外三个 CTA 把同一响应映射为 rank 0 中的 (6, 10)、rank 2 中的 (6, 11) 和 rank 3 中的 (7, 11)：即行 [768, 1024) x 列 [1280, 1536) 的四个象限。

**算例 2。** 响应 `(x0, y0) = (62, 30)` 在 CTA `(x, y) = (0, 1)`（rank 2）中被消费：

```
tm = 2 * (30 >> 1) + (62 & 1) + 0 = 30 + 0 + 0 = 30       -> rows    [3840, 3968)
tn = 2 * (62 >> 1) + (30 & 1) + 1 = 62 + 0 + 1 = 63       -> columns [8064, 8192)   (the last 128 columns of D)
L  = 15 * 32 + 31 = 511; cluster tile (15, 31)
```

检验：`tm % 2 == 0 == x`，并且 `tn = 63` 是奇数，正如 y = 1 时所必须的那样。

**算例 3（初始 tile）。** explicit.cu:1093 处的 kernel 横幅注释陈述了它；:1175-1176 是 7.5.2 中推导出的代码（引文见第 11.6 节）。`blockIdx = (5, 2)`：`tm = 2 * (2 >> 1) + (5 & 1) = 2 + 1 = 3`，`tn = 2 * (5 >> 1) + (2 & 1) = 4 + 0 = 4`；这个 CTA 是 `(x, y) = (1, 0)`，rank 1，属于启动 cluster `(CX, CY) = (2, 1)`，`L = 34`。与算例 1 的模式一致：rank 1 的 `tm` 为奇数，`tn` 为偶数。

**覆盖。** L 的 1024 个不同取值（32 x 32），每一个要么被原生启动（实测运行中有 33 个，笔记 D.3），要么恰好由一次成功的 `try_cancel` 分发出去（7.3 第 2 项），绝不会两者皆是（被取消的 cluster 是 "has not started running yet" 的那一种，第 2.12.1 节），每一个都覆盖四个 CTA 中四个不同的 `(tm, tn)`：4096 个 CTA tile = (8192 / 128)^2，恰好一次。实测："the 1024 cluster tiles are covered exactly once"（笔记 D.3；对 explicit kernel 再次验证，笔记 G.1）。

### 7.5.6 这个顺序对数据移动意味着什么（我的分析，基于实测顺序）

L 大致按分发顺序递增，而 `L % 32` 是 N 坐标，所以连续的分发共享 A 的同一个 256 行条带（fp16 下为 4 MiB），并横跨 B 的 32 个列条带行走。笔记 D.3 实测到 "the resident clusters work on two (834 of 991 hand-outs) or three (157) adjacent 256-row bands of A at any moment, never on one"。因此整个 GPU 的 A working set 在任何时刻都是 8-12 MiB，而对 A 的每个条带，B 的全部内容都流经 L2 一次。第 3 章量化了由此产生的 L2/HBM 流量。分发器的唯一贡献是保持硬件的顺序而不依赖它。

---

## 7.6 第 7 节：逐行解读 `scheduler_role`

横幅注释与该函数，逐字照录，只略去了它的五个 `#if defined(CUTLASS_DEABSTRACTION_TRACE)` 块（explicit.cu:825-827、832-834、844-846、850-852、862-865），它们在性能二进制文件中被编译掉，此处省略：

```
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
  int tm_next = tm, tn_next = tn;
  uint32_t tile = 0;
  bool valid;
  do {
    // throttle consumer_wait + consumer_release (all 32 lanes; PipelineAsync, dst_blockid 0) (kernel :693-695)
    EXPLICIT_WAIT(smem_base + OFF_THROTTLE_FULL + 8u * thc_idx, thc_phase, WAIT_THROTTLE_FULL, tile, 0u);
    mbar_arrive_remote(smem_base + OFF_THROTTLE_EMPTY + 8u * thc_idx, 0u);
    advance<kThrottleStages>(thc_idx, thc_phase);
    // advance_to_next_work (sm100_tile_scheduler.hpp:438-451): producer_acquire (sm100_pipeline.hpp:1095-1104) + elected try_cancel
    uint32_t const full_addr = smem_base + OFF_CLC_FULL + 8u * clp_idx;
    EXPLICIT_WAIT(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, WAIT_CLC_EMPTY, tile, 0u);   // wait own empty[idx]
    mbar_arrive_expect_tx_remote_pred(full_addr, lane, lane < 4u ? 1u : 0u, kClcTxBytes);   // lanes 0..3 arm CTAs 0..3's full[idx] with 16 bytes
    if (elect_one_sync()) {
      clc_try_cancel(smem_base + OFF_CLC_RESPONSE + 16u * clp_idx, full_addr);
    }
    advance<kClcStages>(clp_idx, clp_phase);
    valid = clc_consume<K_CLC_SCHED_KIND>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // fetch_next_work (:702-706)
    tm = tm_next; tn = tn_next;
    ++tile;
  } while (valid);
  // CLC producer_tail (sm100_pipeline.hpp:1041-1050): test_wait, then the blocking wait only if not done (Section 7.10 tail 2)
  #pragma unroll
  for (int i = 0; i < kClcStages; ++i) {
    uint32_t const done = mbar_test_wait(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, 1u);
    if (!done) { EXPLICIT_WAIT(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, TAIL_CLC, tile, uint32_t(i)); }
    advance<kClcStages>(clp_idx, clp_phase);
  }
}
```
(explicit.cu:809-868)

它的源头是 CUTLASS kernel 的 `is_participant.sched` 分支：

```
    else if (is_participant.sched) {
      if constexpr (IsSchedDynamicPersistent) {
        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        cutlass::arch::wait_on_dependent_grids();

        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;

            // Query next clcID and update producer state
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }

          // Fetch next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
```
（该循环以 :721-722 处的 `} while (work_tile_info.is_valid()); clc_pipeline.producer_tail(clc_pipe_producer_state);` 结束（在本工作树中为 :749-750）。）

对这个 scheduler 而言 `requires_clc_query` 始终为 true（`fetch_next_work` 返回 `increment_pipe = true`，sm100_tile_scheduler.hpp:472（在本工作树中为 :503）），所以 explicit kernel 去掉了这个标志。

所引代码中有两个名字不是本章的 wrapper：`elect_one_sync()` 是 W2（explicit.cu:263，第 6.2.2 节），而 `lane` 是 kernel 的 `threadIdx.x & 31u`（explicit.cu:1102），被复用为 W13 的 `cta_id` 操作数（7.1、7.6.5）。

### 7.6.1 谁运行它，以及为什么只有 rank 0 (explicit.cu:1194-1196)

（引文见第 11.7 节；`is_cta0 = (rank == 0u)` 位于 :1105，`lane = threadIdx.x & 31u` 位于 :1102）

CUTLASS：`is_participant.sched = (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster`，其中 `is_first_cta_in_cluster = cta_rank_in_cluster == 0`（sm100_gemm_tma_warpspecialized.hpp:426、:451）。这两个理由都来自硬件契约。第一，每个 cluster 只能有一次 `try_cancel`，否则你会取消多个 cluster（7.3 第 6 项）。第二，响应被 multicast 到全部四个 CTA 并完成全部四个 full 屏障（7.3 第 3 项），所以其余三个 CTA 不需要自己的 scheduler：它们的 warp 1 在 prologue 中预取 A/B tensor map（explicit.cu:1113-1116），然后从分发链中贯穿而过并退出（笔记 7.3 步骤 7）。rank 0 只是一个约定（`producer_blockid = 0`）；任何固定的 rank 都可以，只要每一个 consumer 都释放到同一个即可。

### 7.6.2 第 821 行：`griddep_wait()`

`griddepcontrol.wait`（W35，explicit.cu:599）镜像了 kernel 中位于本工作树 `:716` 处的 `cutlass::arch::wait_on_dependent_grids()`（按 explicit.cu 的注释所用的编号则是 `:688`）。它属于程序化依赖启动（第 11 章）：它一直阻塞，直到所有前置网格都已完成且其内存操作均已可见为止。本 kernel 中只有两个 warp 角色执行它：warp 2（在它的第一次 TMA 加载触及 A 和 B 之前）与 rank 0 的 warp 1（此处）；explicit SASS 中 `ACQBULK` x2（笔记 G.3）。源码没有说 scheduler warp 为什么需要它。我的假设：一个依赖网格的 cluster 不得在它赖以启动的依赖关系得到解决之前，就开始取消（窃取）它自己网格的 cluster，从而让自己承诺承担更多工作。请把这一推理标为（假设）；可观察的事实是 CUTLASS 把这个等待放在那里，而 explicit kernel 保留了它。

### 7.6.3 第 822-824 行：三个流水线状态

| 状态 | 此 warp 的角色 | Stage 数 | 起始 | 推进处 | 对端 |
|---|---|---|---|---|---|
| `thc_idx, thc_phase` | throttle **consumer** | 2 | {0, 0} | :838 | rank-0 warp 2（producer，`thr_idx, thr_phase = {0, 1}` 位于 explicit.cu:766） |
| `clp_idx, clp_phase` | CLC **producer** | 2 | {0, 1} | :849 | 800 个 consumer（释放）与硬件（填充） |
| `clc_idx, clc_phase` | CLC **consumer** | 2 | {0, 0} | 在 `clc_consume` 内部 (:688) | 与其他每个 consumer warp 相同 |

三者在每次循环迭代中都恰好推进一次，所以在 tile t 之后它们全都指向槽位 (t + 1) % 2，并带有 7.2.1 的 parity。scheduler 是唯一一个既是同一流水线的 producer *又是*其 consumer 的 warp，这就是它携带两个 CLC 状态的原因；CUTLASS 给它 `ThreadCategory::ProducerConsumer`（kernel :503-505）。

### 7.6.4 第 836-838 行：throttle consumer

```
    EXPLICIT_WAIT(smem_base + OFF_THROTTLE_FULL + 8u * thc_idx, thc_phase, WAIT_THROTTLE_FULL, tile, 0u);
    mbar_arrive_remote(smem_base + OFF_THROTTLE_EMPTY + 8u * thc_idx, 0u);
    advance<kThrottleStages>(thc_idx, thc_phase);
```
(explicit.cu:836-838)

producer 一侧位于 `producer_role` 中，仅限 rank 0，在每个 tile 的开始处、该 tile 的任何 TMA 之前（explicit.cu:783-787，引文见第 8.6.4 节）。

- **计数 32 / 32。** explicit kernel 用 `kThrottleCount = 32` 初始化两个 throttle 屏障（explicit.cu:1159-1160，warp 3，正如 CUTLASS 的 `initializing_warp = 3`，kernel :549）。
- **为什么 consumer 的释放是指向 CTA 0（也就是它自己）的远程形式。** CUTLASS 的 `PipelineAsync::consumer_release` 是 `empty_barrier_ptr_[stage].arrive(params_.dst_blockid);`（cutlass/pipeline/sm90_pipeline.hpp:1236-1238），其中 `clc_throttle_pipeline_params.dst_blockid = 0`（kernel :548）；`arrive(cta_id)` 是 `mapa` + `mbarrier.arrive.shared::cluster` 的形式（barrier.h:383-385 -> :486-499）。explicit kernel 复现了这一形式（W10），以便 PTX 清单相匹配（`mapa` 8 处，笔记 G.3）。如果它是一次本地 arrive，什么也不会出错，但也没有任何收益。
- **为什么 producer 的 acquire 是一次没有 peek 的阻塞等待。** `PipelineAsync::producer_acquire` 在 kernel 中前面没有 `try`（本工作树中 kernel :661 直接调用 `producer_acquire`，然后在 :662 调用 `producer_commit`），而且在实践中它应当从不阻塞（见下；推断，该等待未被计时），所以做一次 peek 只会是浪费。
- **为什么 producer 从不在 throttle 上阻塞。** producer 在生产 token i + 2 之前等待 `throttle_empty[i % 2]`，而在实践中从不需要等待：scheduler 几乎整个 tile 都阻塞在它的 `throttle_full` 等待中（每 tile 28.8-37.7 us，7.6.9），所以 token i 被消费、槽位 i % 2 被释放，都在它被生产出来之后不久，并且比 token i + 2 被生产早大约两个 tile 周期（8.6.4）[由 scheduler 一侧的计时推断；producer 一侧的等待本身未被计时]。

### 7.6.5 第 840-842 行：获取槽位并装填四个 full 屏障

```
    uint32_t const full_addr = smem_base + OFF_CLC_FULL + 8u * clp_idx;
    EXPLICIT_WAIT(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, WAIT_CLC_EMPTY, tile, 0u);   // wait own empty[idx]
    mbar_arrive_expect_tx_remote_pred(full_addr, lane, lane < 4u ? 1u : 0u, kClcTxBytes);   // lanes 0..3 arm CTAs 0..3's full[idx] with 16 bytes
```
(explicit.cu:840-842)

这是 `advance_to_next_work` 的 `producer_acquire`：

```
  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    // 1. Wait for empty barrier to be ready
    // 2. Set the transaction bytes set to occur on the Full barrier for all blocks
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes, lane_idx_, uint32_t(lane_idx_ < cluster_size_));
  }
```
`advance_to_next_work` 调用 `clc_pipeline.producer_acquire(clc_pipe_producer_state)` 时不带 token（sm100_tile_scheduler.hpp:443），即 `barrier_token` 默认为 `{BarrierStatus::WaitAgain}` 的那个 `PipelineState` 重载（sm100_pipeline.hpp:1027）。因此上面的 `if` 总是成立：scheduler 的 empty 屏障等待总是不带前置 peek 的阻塞形式。

- **装填。** W13（explicit.cu:393-405，引文见第 6.3.8 节）。

  全部 32 个 lane 中的每一个都执行这条 asm。为什么是 `lane < 4`：cluster 有 4 个 CTA（源码中是 `lane_idx_ < cluster_size_`；explicit kernel 把它硬编码为 `lane < 4u`，因为它的 cluster 固定为 (2,2,1)）。为什么是每个 CTA 一个 lane，而不是一个 lane 做四次到达：带谓词的形式代价相同（每个 lane 一条带谓词的指令，每个 warp 发射一次），不需要循环，而且在 SASS 中是每个 lane 一条带谓词的 arrive，`@!P0 SYNCS.ARRIVE.TRANS64.RED RZ, [R11+URZ], R5`，其中 `R5 = 0x10`（`0x1b70`，第 12.2 节；baseline 的是 `@!P2 ..., [PRMT(lane)], R9`，笔记 D.2）。
- **mbarrier 的计数范围。** PTX ISA 9.4 表 43 对 `.layout::v0`（默认值，也是本 kernel 唯一使用的 layout）给出：期望到达计数 1 .. 2^20 - 1，待到达计数 0 .. 2^20 - 1，tx-count -(2^20 - 1) .. 2^20 - 1。所以 `mbarrier.init` 的计数 800 和 256 是合法的，而 16 字节（或 49152 字节）的预期值不会溢出。Programming Guide 用它自己的话陈述了双侧完成规则（待到达计数与 tx-count 都为零，第 4 章，第 4.2.1 节）："When a barrier uses a transaction count, it blocks threads at the wait operation until all the producer threads have performed an arrive and the sum of all the transaction counts reaches an expected value"（Asynchronous Barriers，4.10.6）。

### 7.6.6 第 843-849 行：被选中的 `try_cancel` 与 producer 推进

W33 `clc_try_cancel(result_addr, mbarrier_addr)`（explicit.cu:568-575，第 6.8.2 节和第 2.12.2 节）是单条指令 `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];`，不带 `"memory"` clobber；该 asm 是 `issue_clc_query`，cutlass/gemm/kernel/sm100_tile_scheduler.hpp:398-403。

- 操作数：`result_addr = smem_base + 352 + 16 clp_idx`（此 CTA 的槽位）与 `mbarrier_addr = full_addr = smem_base + 208 + 8 clp_idx`（此 CTA 的 full 屏障），二者都是 rank 0 的 `shared::cta` 地址。带 `.multicast::cluster::all` 时，硬件把同样的 16 字节写到**每个 CTA 中**偏移 352 + 16 idx 处，并对**每个 CTA 中**偏移 208 + 8 idx 处的屏障执行 16 字节的 complete-tx（7.3 第 3 项）。这些地址是 rank-0 的地址，但其语义是 ISA 所说的 "corresponding local shared memory addr of each CTA"。这之所以可行，是因为全部四个 CTA 具有完全相同的共享内存 layout（同一个 `SharedStorage`；实测的基址只在第 24-25 位上不同，笔记 D.1 D5）。
- `.async`：该指令立即返回；响应稍后到达（在发射之后中位数 1408 ns 时被 scheduler warp 消费，笔记 D.3）。
- 为什么用 `elect_one_sync()`：每个 cluster 一次请求，从单个线程提交，正如 Programming Guide 对这条 uniform 指令所建议的那样（7.3 第 6 项）。SASS 证实了这个 "uniform"：`UGETNEXTWORKID.BROADCAST [UR8], [UR9]`（baseline 中为 `[UR4], [UR5]`）是一条无谓词的 uniform 数据通路指令，每个 warp 发射一次，所以 `ELECT` 被折叠掉了（笔记 G.3、D.2）。PTX 层面的选举对于正确性的可移植性仍然是必需的，而且它是 kernel 所书写的契约（笔记 E.6.6）。
- W33 上没有 `"memory"` clobber：源码中的 asm 也没有（`:398-403`）。它不读也不写编译器能看见的任何东西；它的效果通过 mbarrier 到达，而随后的每一次等待都以自己的 `"memory"` clobber 观察它。
- `advance<kClcStages>(clp_idx, clp_phase)` (:849)：下一次查询使用另一个槽位，而每两次查询就翻转一次 producer 将要等待的 parity，与 consumer 的释放步调一致。

### 7.6.7 第 853-856 行：消费自己的响应，然后决定

```
    valid = clc_consume<K_CLC_SCHED_KIND>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // fetch_next_work (:702-706)
    tm = tm_next; tn = tn_next;
    ++tile;
  } while (valid);
```
(explicit.cu:853-856)

scheduler warp 是 800 个 consumer 之一（它的 32 次到达计入 `NumSchedThreads`），所以它运行与其他每个 consumer warp 相同的 `clc_consume`，作用于它*刚刚*发出的那次查询的响应（同一次迭代：在 :847 发出 n，在 :853 消费 n）。这里就是该 warp 为查询延迟而阻塞的地方：它对 `clc_full[idx]` 的 `EXPLICIT_WAIT` 在硬件的字节落地时返回（7.6.9 给出数字）。该 warp 不把 `tm, tn` 用于任何用途，但保留它们以便 trace 能报告被映射的 tile；在生产版本中 :854 处的拷贝是死代码并会消失。

循环在第一个未取消的响应（`valid == 0`）处结束，它回答的是该 warp 所发出的最后一次查询。发出查询 t（:847），然后消费响应 t（:853），并且只有当响应 t 成功时才进入迭代 t + 1：正是这个顺序满足了 7.3 第 7 项——在观察到失败之后不再有任何请求。

### 7.6.8 第 857-867 行：producer 尾段

尾段，在此引用是因为状态变量 `clp_idx, clp_phase` 与查询计数 T 都定义在本章中：

```
  // CLC producer_tail (sm100_pipeline.hpp:1041-1050): test_wait, then the blocking wait only if not done (Section 7.10 tail 2)
  #pragma unroll
  for (int i = 0; i < kClcStages; ++i) {
    uint32_t const done = mbar_test_wait(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, 1u);
    if (!done) { EXPLICIT_WAIT(smem_base + OFF_CLC_EMPTY + 8u * clp_idx, clp_phase, TAIL_CLC, tile, uint32_t(i)); }
    advance<kClcStages>(clp_idx, clp_phase);
  }
```
(explicit.cu:857-867)

它的源码：

```
  // Prevents early exit of producer blocks in Cluster.
  // Does NOT reset transaction bytes.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    detail::pipeline_check_is_producer(params_.role);
    for (int count = 0; count < Stages; ++count) {
      bool done = empty_barrier_ptr_[state.index()].test_wait(state.phase());
      if (!done) {
        empty_barrier_ptr_[state.index()].wait(state.phase());
      }
      ++state;
    }
  }
```

**实测 tile 数的 parity**（T = 32、31 与 19）在第 4 章，第 4.8.4 节中算出（笔记 7.4）；explicit 运行的 `K_TAIL` 记录验证了它们（笔记 G.1："CLC `((T+i) % 2, 1 ^ (((T+i) / 2) & 1))`"）。

验算那张表的一行（第 4 章，第 4.8.4 节）：对于 T = 32，响应 30 使用了槽位 0，而它的 800 次释放完成了 `clc_empty[0]` 的 phase 15（phase 按曾占据该槽位的响应编号：0, 2, 4, ..., 30 -> phase 15），parity 15 & 1 = 1。

**为什么尾段存在。** 除了远程到达规则（第 4 章，第 4.8 节）之外，整个退出纪律还有一个 ISA 层面的理由：7.3 第 3 项的 `.multicast::cluster::all` 退出规则，kernel 之所以满足它，是因为最后一次 `try_cancel` 先于 cluster 中的每一次退出（7.6.7）。

**为什么先 `test_wait`。** 如果 phase 已经完成，阻塞循环就被整个跳过；这对步骤 0 成立，但对步骤 1 不成立：warp 2 只有在发射完最后一个 tile 的 128 个 k-tile 的加载之后（7.8）才释放最后一个响应，比这个 warp 消费它晚大约一个 tile 的时间，所以第二次等待在每个 cluster 中都会阻塞。在 explicit kernel 的 trace 运行中实测（对 `trace_out_v2/launch0.csv` 中 `K_TAIL` 时间戳的一次重新统计）：步骤 0 的记录只比 scheduler 的最后一个 K5b 晚 0.7-1.2 us，而步骤 1 的记录在全部 32 个不带探针的 cluster 中比它晚 33.7-34.6 us，在 cluster (0,0) 中晚 58.0 us。从功能上说，单靠阻塞等待就足够了（它在 phase 已完成时立即返回）。CUTLASS 为这条流水线的 `producer_tail` 选择了先探测再等待的形式（这是 sm90_pipeline.hpp 与 sm100_pipeline.hpp 中唯一先探测的 `producer_tail`），而 explicit kernel 镜像了它，使得 baseline 的 PTX 中的两个 `test_wait.parity` 位点（笔记 D.2）在 explicit PTX 中重现（`mbarrier.test_wait.parity 2`，笔记 G.3）。scheduler 在 SASS 中的两个 `EXIT` 位点（是代码地址，不是计时）在 baseline 中为 `0x2f80`/`0x2fe0`（笔记 D.2），在 explicit kernel 中为 `0x1e60`/`0x1ec0`（笔记 G.3）。笔记本身没有报告任何尾段时长。G.1 报告的是 `K_TAIL` 记录的顺序。CLC 尾段记录对每个 cluster 的 T 都遵循流水线公式 `((T+i) % 2, 1 ^ (((T+i) / 2) & 1))`。有一条步骤 0 的记录合理地比其 CTA 的最后一个 K5c 早 32 ns，因为步骤 0 等待的是响应 T - 2 的释放，而不是 T - 1 的。`mbar_test_wait` 的 `1u` 参数在 7.2.3 中有解释。

### 7.6.9 为什么每 tile 一次查询、为什么最多一次在途，以及测得了什么

**每 tile 一次查询。** 每个角色的循环中的每一次迭代都恰好消费一个响应，而 scheduler 每次迭代恰好发出一次查询。因此一个处理 T 个 tile 的 cluster（启动 tile 加上 T - 1 个窃取来的）恰好发出 T 次查询：T - 1 次得到的回答是“已取消”，一次得到的回答是“未取消”（笔记 7.4）。整个启动过程中：1024 次查询，其中 `N_native` 次失败，这里 `N_native` 是原生启动的 cluster 的数目。实测：`N_native = 33`（= `cudaOccupancyMaxActiveClusters`），991 次已取消 + 33 次未取消（笔记 D.3，D9）。是 33 而不是 148 / 4 = 37，因为 "a 2x2 cluster needs two whole TPCs in one GPC"，而 8 个剩余的 TPC（16 个 SM：64, 65, 122-129, 142-147）无法配对（笔记 D.3；在这块 B200 的 skyline 上，这 8 个是每个奇数大小的 GPC 中那一个未配对的 TPC，NVIDIA 内部信息；算术见第 2.1.1 节）。

**最多一次在途。** 把一个 cluster 的依赖关系串起来：

```
producer warp 2 (rank 0):   begin tile t  --throttle_full[t%2]-->  scheduler: query t  --hw-->  response t (= tile t+1 or "none")
                            ... 128 k-tiles of loads for tile t ...                         ^
                            consume response t  <------------------------------------------+
                            begin tile t+1  --throttle_full[(t+1)%2]-->  scheduler: query t+1
```

查询 t + 1 需要 producer 已经开始了 tile t + 1。这又需要 producer 已经消费了响应 t（其迭代 t 的末尾），而后者需要响应 t 已经到达。所以查询 t + 1 绝不会在查询 t 在途时也在途：**每个 cluster 最多一次查询在途**，并且查询 t（它取回 tile t + 1）只在加载者已经开始了 tile t 之后才发出（笔记 D.7）。实测："Never more than one query was outstanding per cluster (0 violations in 1024 queries); at most 25 were outstanding GPU-wide at any instant"（笔记 D.3），以及 "a query is actually in flight only about 4% of the time"（笔记 D.7）。

**为什么 throttle 存在（CUTLASS 的注释与我的分析）。** 源码注释写道 "Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers"（在本工作树中为 kernel :720）。没有它，就只有 CLC empty 屏障（计数 800）来门控 scheduler 的循环，也就是两次查询之前的那个响应的最慢 consumer。这些释放发生在 tile n *期间*（warp 0 与 4-7 在它们的 tile n 开始处消费响应 n，warp 2 在 tile n 的加载之后消费；7.8），因此只需要 `clc_empty[n % 2]` 的查询 n + 2 就可以在 tile n 仍在运行时发出。于是在 tile n 期间，scheduler 本可以已经发出查询 n + 1 和 n + 2。到 tile n 结束时，该 cluster 将知道 tile n + 1（来自响应 n）、n + 2 和 n + 3：多达三个已窃取但尚未开始的 tile id，而有了 throttle 则恰好只有一个（tile n + 1）。与此同时，一个已经没有工作可做的更快的 cluster 却发现队列为空。在网格末尾这就是纯粹的尾段不均衡：最后几个 tile 会躺在忙碌 cluster 的共享内存槽位里，而其他 SM 空闲。把每次查询绑定到 producer *开始*一个 tile 的时刻，意味着一个 cluster 所拥有的绝不会超过它正在加载的 tile 加上一个已认领的 id，因此分发会跟随实际进度。实测的每个 cluster 的 tile 数分布（13 个 cluster 为 32，19 个为 31，被探针拖慢的 cluster (0,0) 为 19，笔记 D.3）正是 throttle 在起作用：慢的 cluster 只是拿了更少的 tile。

**实测动态**（笔记 D.3，CUTLASS trace 构建；explicit kernel 复现了相同的动态，笔记 G.1；第 12 章，第 12.6.3 节有完整的表格）。从查询到消费的延迟：中位数 1408 ns，p90 1536，p99 1792；explicit kernel 自己的 trace 构建测得 1024 ns 的中位数（笔记 G.2），这是一个与 Release 比较无关的探针放置产物。scheduler 从消费响应 n 到发出查询 n + 1 的间隔，即 throttle 等待：在 32 个不带探针的 cluster 中为 28.8-37.7 us，中位数 30.8 us。每个 tile 的时间：trace 构建中中位数 31.90 us，而 Release 中为 27.8 us（0.889 ms / 32 个 tile）。响应：991 次已取消（id 33..1023，每个恰好一次）加上 33 次未取消，大致按递增的启动线性顺序分发；第一个，id 33，在第一次查询之后 1.28 us 被 cluster (0,0) 消费。在途查询：每个 cluster 从未超过一个，GPU 范围内最多 25 个。

因此这一延迟对 scheduler 而言被完全隐藏（它为 throttle 等待 ~30 us，为响应等待 ~1.4 us）。对 MMA warp 来说问题则不同，而且值得把 *warp* 与 *tensor core* 区分开。MMA warp 在其 tile t 的*开始*处消费响应 t（explicit.cu:913 是其循环体的第一条语句；第 9 章），紧接在发射 tile t - 1 的 k-tile 127 之后。它在时间上的位置由 `full[s]`（:923，TMA 数据落地）和 `acc_empty[st]`（:918）门控，而*不是*由 `empty[s]`。`tcgen05.mma` 是异步的，其完成只能通过 `tcgen05.commit`（:942）观察到，所以该 warp 的发射位置可以领先 tensor-core 执行最多达环所容纳的 8 个 stage。因此它到达 `clc_consume` 的时刻，大约在 producer 发射 tile t - 1 的加载 127 之后一个 TMA 延迟（~1 us，未实测；第 3.5 节），这也差不多就是 producer 开始 tile t 且查询 t 被发出的时候。该 warp 实际领先于完成多远，没有实测。唯一实测的数据点是 D.3 的 "K5c of rank 0 follows K5b by 96 ns median"：MMA warp 的消费在 scheduler warp 的消费之后 96 ns 完成。如果 MMA warp 在响应落地时已经在 `clc_full` 上等待，而且是在大多数 tile 上而非偶尔如此，你看到的就会是这样（我对 D.3 的解读）。这对吞吐量无害。tensor core 仍然持有多达 8 个 k-tile 的已发射 MMA（按每 k-tile 27.8 us / 128 计，即 8 x 217 ns = 1.74 us 的工作量，在该 warp 阻塞之前发射）。因此只有当响应延迟超过这个队列时，即高于 ~1.7 us，tensor-core 气泡才可能出现，而这已超出 p90（1536 ns）并接近 p99（1792 ns）（我的分析）。epilogue warp 在它们的 tile 开始处消费，在前一个 tile 的 MMA 完成之后，也就是更晚；它们等待的频率没有实测。

**kernel 不得假设什么。** 分发顺序是这块硬件和这次运行的一个属性；"the kernel must not depend on the order, D.7"（笔记 7.4）。也不得依赖于 `N_native = 33`："the kernel must work for any number of native clusters and any hand-out order"（笔记 D.7）。explicit kernel 从构造上就满足这一点：它只使用 `valid` 和映射，从不使用顺序，从不使用 tile 计数，从不在失败之后发出第二次 `try_cancel`。

为什么该顺序是*大致*递增的（假设）：CLC 单元从网格的启动队列中弹出下一个尚未启动的 cluster，而该队列是按启动线性顺序排列的。这些交换不是时间戳分辨率造成的产物。D.3 以 id 升序打破同一计时器刻度内的并列，因此同一刻度内的一对绝不会被算作一次倒置，并且发现这 25 次倒置中的每一次都是这样一次查询：它在*严格更晚*的 32 ns 计时器刻度中发出，却收到了比更早的某次查询*更低*的 id。也就是说，CLC 单元以不同于其发出顺序的次序服务了两个请求（假设：在通往该单元的路上 SM 或 GPC 之间的仲裁）。25 这个计数只在那种打破并列的约定下成立（其他约定给出 32-62）。这些都未被依赖。

---

## 7.7 一次 CLC 往返：时序

响应 n 使用槽位 `i = n % 2`；parity 就是 7.2.1 中的那些，其中 `k = n / 2`。

```
 rank-0 warp 2            rank-0 warp 1 (scheduler)              CLC hardware                 consumers, every CTA r (0..3)
 (throttle producer)                                                                            w0, w2, w4-7 (+ w1 of rank 0)
 ---------------------    ---------------------------------      ------------------           -------------------------------------
 begin tile n:
  wait  throttle_empty[i]
        (parity 1^(k&1))
  arrive throttle_full[i]
        x32 (local)   --------> wait throttle_full[i] (parity k&1)
                                 arrive throttle_empty[i] x32
                                   (mapa -> CTA 0 == self)
  issue 128 k-tiles of
  TMA for tile n ...             wait own clc_empty[i] (parity 1^(k&1))
                                   <== completed by the 800 releases of response n-2
                                 lanes 0..3: mapa(clc_full[i], lane)
                                   arrive.expect_tx 16  -------------------------------->  CTA r's clc_full[i]: pending arrivals 1 -> 0,
                                                                                            tx-count 0 -> +16
                                 elected lane:
                                   try_cancel [resp[i]], [clc_full[i]] --> cancel an unlaunched
                                                                           cluster (first CTA
                                                                           (x0,y0,z0)) or decline
                                                                           | 16 B async-proxy write to resp[i] of CTA 0,1,2,3
                                                                           | complete_tx 16 on clc_full[i] of CTA 0,1,2,3
                                                                           v                tx-count +16 -> 0  => phase k completes
                                 advance clp state
                                 consume (same code as ->)                                  each warp: wait own clc_full[i] (parity k&1)
                                   wait own clc_full[i]                                       ld.shared.b128 resp[i]
                                   decode, fence.proxy.async                                  is_canceled -> valid; @valid get_first_ctaid
                                   arrive CTA 0's clc_empty[i] x32 ----+                      fence.proxy.async.shared::cta
                                   advance clc state                   |                      arrive CTA 0's clc_empty[i] (x32 per warp,
                                   @valid: (tm,tn) <- map(x0,y0)       |                        x128 for w4-7)  ----------------------+
                                                                       |                      advance clc state                       |
                                                                       |                      @valid: (tm_next,tn_next) <- map(x0,y0) |
  ... last TMA of tile n                                               v                                                              v
  consume response n  (w2 of rank 0 is one of the consumers)       CTA 0's clc_empty[i]: 800 arrivals -> phase k completes
  tm,tn <- tm_next,tn_next                                          (this is what the scheduler's acquire for query n+2 waits for)
 begin tile n+1  ------------->  (next iteration: query n+1)
```

图中的到达簿记：每个 CTA r 向 CTA 0 的 `clc_empty[i]` 贡献 w0 (32) + w2 (32) + w4-7 (128) = 192 次到达；rank 0 再加上它自己的 w1 (32)，所以 rank 0 贡献 224，总计 224 + 3 x 192 = 800（7.4.5）。

MMA warp 在其 tile n 的开始处到达它对响应 n 的消费，通常在响应落地之前，并在 `clc_full[i]` 上阻塞，与此同时 tensor core 排空多达 8 个已排队的 k-tile（在 7.6.9 中由唯一的实测数据点推断得出，即关于 CUTLASS kernel 的 trace 构建的笔记 D.3：按中位数计，其消费比 scheduler 的晚 96 ns）。

同一批往返的尾段的槽位与 parity 簿记，针对一个处理 T = 32 个 tile 的 cluster（第 4 章，第 4.3.5 节把 tile 0..5 列成了表；第 4.8.4 节把这两次尾段等待列成了表）：

| tile t | 查询 / 响应 n | 槽位 | sched acquire：`clc_empty[slot]` parity |
|---|---|---|---|
| 尾段 0 | (32) | 0 | 1（响应 30 的释放） |
| 尾段 1 | (33) | 1 | 1（响应 31 的释放） |

---

## 7.8 设计注记：为什么分发器是这个形状，以及如何复用它

**动态（CLC）而非静态持久化 stride。** 一个静态持久化 kernel 会启动 `N_native` 个 cluster，并让 cluster c 拿 tile c, c + N_native, ...。但 `N_native` 并不由 SM 数给定（这里是 33，而不是算术上的 37；`cudaOccupancyMaxActiveClusters` 返回 33，笔记 D.3，这是一个共驻留最大值而非保证），而且各 cluster 的运行速度并不相等（在 trace 构建中 13 个 cluster 做了 32 个 tile，19 个做了 31 个；被探针拖慢的 cluster (0,0) 做了 19 个，笔记 D.3）。CLC 让硬件充当队列：驻留与速度差异被自动吸收，硬件保证恰好一次的覆盖（7.3 第 2 项以及第 2.12.1 节的调度模型），而 kernel 不需要全局内存中的任何原子计数器。代价是每 tile 一次 16 字节的往返（1.4 us，隐藏在 28 us 背后），以及每 CTA 8 个 mbarrier（`clc_full[2]`、`clc_empty[2]`、`throttle_full[2]`、`throttle_empty[2]`：8 x 8 B = 64 B，位于字节 208..239 和 304..335）加上 2 x 16 B = 32 B 的响应槽位（字节 352..383）。即每 CTA 96 B，cluster 范围内 384 B，其中 throttle 与 `clc_empty` 屏障只在 rank 0 中使用（rank 1-3 也初始化它们自己的，explicit.cu:1135-1143 和 1155-1163，但从不在其上等待或到达；常量位于 :109-110, 144-145, 148-149, 151）。

**两个槽位，而不是一个或八个。** 在 throttle 保证最多一次查询在途的前提下，两个槽位是能让查询 n + 1 在响应 n 的最慢 consumer（epilogue warp，它们在其 tile n 的开始处消费）尚未读取它时就被发出的最小数目。一个槽位会迫使 scheduler 在发出 n + 1 之前等待响应 n 的全部 800 次释放，使往返与最慢 warp 的 tile 边界串行化。这个 2 从哪里来：`kClcStages` 镜像了 `SchedulerPipelineStageCount`，它是分发策略的一个模板参数，而不是硬件常量。collective builder 设置 `SchedulerPipelineStageCount = cute::conditional_return<IsGroupGemm || IsRCGroupGemm>(8, 2)`（cutlass/gemm/collective/builders/sm100_umma_builder.inl:282），即对这个稠密 GEMM 为 2，并把它传入 `MainloopSm100TmaUmmaWarpSpecialized<PipelineStages, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape_MNK, ArchTag>`（`:320-326`）。实测的实例化为 `MainloopSm100TmaUmmaWarpSpecialized<8,2,4,(2,2,1),Sm100>` 和 `PersistentTileSchedulerSm100<(2,2,1),2>`（笔记 D.1）。kernel 以 `DispatchPolicy::Schedule::SchedulerPipelineStageCount`（sm100_gemm_tma_warpspecialized.hpp:118）把它读回来，并用它来确定 `CLCPipeline`、`CLCThrottlePipeline` 和 `clc_response[]` 的大小（`:172`、`:175`、`:200`）。CUTLASS 文档页面泛泛地说 "The CLC pipeline has a depth of 3 to overlap the CLC operations of multiple waves for latency hiding"；那句话描述的是一个不同的配置，并不适用于本构建。

**屏障的不对称性。** 两种不对称（每个 CTA 都有 full 屏障，empty 屏障只在 CTA 0 中；第 4 章，第 4.4.4 节）都直接来自硬件契约与单请求者规则；不要把它们“对称化”。

**每个角色在哪里消费（第 8-10 章给出这些循环）。** warp 2 在 tile t 的加载之后消费响应 t（第 8.6.7 节）；warp 0 与 4-7 在计算/存储 tile t *之前*消费它；MMA warp 根本不需要坐标（`(void)tm; (void)tn;`，explicit.cu:883），而 epilogue 需要的是*当前* tile 的坐标，它已经有了。

**设计时需防范的失效模式（预期的冒险，两者均未被观察到：第一个是笔记 E.6.4 的警告，第二个是我对 ISA 的分析；第一次构建与每次运行都通过了，笔记 G.0）：**

| 错误 | 症状 |
|---|---|
| 某个角色跳过了一次 `clc_consume`（例如一次过早的 `continue`） | CTA 0 的 `clc_empty` 永远到不了 800：scheduler 两次查询之后的 acquire 使整个 cluster 死锁 |
| 一条在 `expect_tx` *之前*发射的单独的 `mbarrier.arrive` | 这次到达使待到达计数 1 -> 0，而此时 tx-count 仍为 0，于是 phase 立即完成（ISA 9.7.15.16.8：两个计数都为零时完成），在任何响应字节存在之前，而 800 个 consumer 读到的是硬件尚未写入的槽位；先 `expect_tx` 再 `arrive` 会是安全的，而融合的 `arrive.expect_tx` 消除了顺序问题并与源码一致 |

**推广到其他 kernel。** 分发器与算子无关：它分发一个被取消的 cluster 的 `(x0, y0, z0)`，并让 `swizzle_and_rasterize` 把它变成工作坐标。要复用它：

1. 装填 `lane < cluster_size` 个 CTA。
2. 保留 throttle，除非你的 producer 有一个不同的、自然的“已开始一个 tile”时间点。
3. 用你自己的光栅替换这个映射。对于 L > 1，像 CUTLASS 那样把 `z0` 绑定到 batch 坐标：baseline 把 `word1 >> 16` 绑定到 `L_idx`（笔记 D.2）；explicit kernel 去掉了它，因为 L = 1 是固定的。
4. 对于 swizzled 光栅，实现完整的 `swizzle_and_rasterize` 块（sm100_tile_scheduler.hpp:711-795；工作树中为 742-826），固定运行跳过了这一块。
5. 对于 stream-K 风格的 scheduler，遵守 `increment_pipe`：并非每次 `fetch_next_work` 都消费一个响应。

---

# 第 8 章 供给者：producer_role 与 load_ktiles（Warp 2）

第 6 章逐字符解释了 W6、W7、W12、W21、W22 和 W35 的 PTX；第 4 章一般性地解释了 parity 和 token。本章讲的是谁调用这些 wrapper、在哪些地址上、用哪些数字，以及为什么。

---

## 8.1 一张图看 Warp 2

producer warp 是唯一搬运 A 和 B 的 warp，而它从不读取其中的任何一个字节。它把坐标交给 TMA 单元，由后者把字节写入 8 个 stage 的共享内存环（对 A 而言，还写入一个同胞 CTA 的环），并且它用两个 mbarrier 数组管理这个环。下游的一切（UMMA、TMEM、epilogue）都是别人的工作。

```
 global A (fp16, K-major)            global B (fp16, K-major)
 rows 128 tm + 64 y .. +64           columns 128 tn + 64 x .. +64
 k = 64 q .. +64                     k = 64 q .. +64
        |                                   |
        | W21: cp.async.bulk.tensor.3d      | W22: cp.async.bulk.tensor.3d
        |      .cta_group::2 .multicast     |      .cta_group::2 (no multicast)
        |      mask 0x5 << x                |
        v                                   v
 +-------------------------------------------------------------+
 | this CTA's smem ring, stage s (and, for A, the ring of the  |
 | sibling CTA that shares tm)                                 |
 |   OFF_SMEM_A + 16384 s + 8192 y  <- 8192 B of A (my half)   |
 |   OFF_SMEM_B +  8192 s           <- 8192 B of B (my half)   |
 +-------------------------------------------------------------+
        |  complete_tx: every box credits the PAIR LEADER's full[s]
        v
 full[s] of the pair leader: count 1 (+49152 tx)   --->  warp 0 of the leader (Chapter 9)
 empty[s] of every CTA:      count 2                <---  tcgen05.commit, mask 0xF, from both leaders
```

每个 CTA 每个 tile，该 warp 发射 256 条 TMA 指令（每个 k-tile 2 条），对 `empty[]` 执行 130 次非阻塞 peek 和至多 128 次阻塞等待，并且——仅在 pair leader 上——执行 128 次 49152 字节的 `mbarrier.arrive.expect_tx`。同一个 warp 还每个 tile 参与一次 CLC 协议（第 7 章），并且在 CTA 0 上驱动约束 scheduler 的 throttle。

---

## 8.2 逐步解读该节的横幅注释 (explicit.cu:708-722)

```
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
```

标题行点明了角色、它所触及的硬件族（`TMA`、`mbarrier`），以及它所镜像的 CUTLASS 函数。该函数是 `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:589-629` 中的 `CollectiveMma::load`。

四个带编号的步骤是 k-tile 循环一次迭代的主体（在代码中，步骤 4 的推进与 peek 位于步骤 3 的 TMA 发射之前：第 737-739 行，然后是第 740-756 行）：

1. **等待 `empty[s]`。** 在 CUTLASS 的计数公式 `cluster_m / atom_m + cluster_n / atom_n - 1`（第 4.4.2 节）中，第一项计数的是沿 M 读取本 CTA 的 B 的 MMA 原子：在 CUTLASS 的一般形式中 B 沿 M multicast，但这里一个 2SM 原子横跨 cluster 的整个 M 范围，因此该项为 1（自身的 leader）。第二项计数的是沿 N 读取本 CTA 的 A 的原子，后者沿 N multicast：为 2（自身的 leader 和另一个 pair 的 leader）。`- 1` 去掉了两项都计入的自身 leader [公式源自源码；对其各项的这一解读是我的分析]。
2. **在 leader 自己的 `full[s]` 上 `arrive.expect_tx` 49152。** 第 8.5.3 节推导出这个数字；第 8.4.4 节解释为什么只有 leader lane 执行 arrive。
3. **由一个被选中的 lane 进行两次 TMA 发射**（第 8.4.6 节）。
4. **推进与 peek**（第 8.4.5 节）。

示例行在第 8.7.1 节中重新计算（其算术是正确的，不过其输入并不是一个可达的状态：rank 1 的 `tn` 总是偶数），字节核算在第 8.5.3 节中推导，尾段则在第 8.6.8 节。

---

## 8.3 ProducerState (explicit.cu:723-726)

```
struct ProducerState {
  uint32_t prod_idx = 0, prod_phase = 1;   // mainloop producer state (8 stages), carried across tiles
  int q = 0;                                // k-tile index inside the current tile
};
```

三个整数，而且每个初始值都是被迫如此的：

- `prod_idx = 0`：环从槽位 0 开始使用（第 4.3.1 节）。
- `prod_phase = 1`（第 4.2.3 节）：从 phase 0 开始的 producer 会等待一个永远不会到来的释放——在 `q = 0` 处死锁，即笔记的失效模式表第一行的症状（E.7.2）。该行列出的原因有字节数、装填 lane、屏障操作数和 A 掩码；错误的起始 phase 不在其中，但产生同样的症状。
- `q = 0`：在每个 tile 开始时重置 (explicit.cu:779)。它放在结构体中而不是 `load_ktiles` 的局部变量中，因为一个 tile 的 128 个 k-tile 是由*两次*调用发射的（先 8 然后 120，第 8.6.5 节），而第二次调用必须从第一次停下的地方接续 k 坐标。

在一个 tile 内部，producer 的 `empty[]` 等待 parity 为：`q = 0..7` parity 1，`q = 8..15` parity 0，`q = 16..23` parity 1，……，`q = 120..127` parity 0（因为 `120 >> 3 = 15` 是奇数）；尾段（第 8.6.8 节）接续这个序列。

---

## 8.4 逐行解读 load_ktiles (explicit.cu:728-760)

```
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
      tma_load_2sm_mcast_3d(&params.tma_a, full_masked_base + 8u * s, a_mcast_mask, kCacheHintEvictNormal,
                            a_dst_base + kStageBytesA * s, c0, a_row, 0);                     // A: (64q, 128 tm + 64 y, 0), mask 0x5 << x
      tma_load_2sm_3d(&params.tma_b, full_masked_base + 8u * s, kCacheHintEvictNormal,
                      b_dst_base + kStageBytesB * s, c0, b_col, 0);                          // B: (64q, 128 tn + 64 x, 0)
    }
    --n;
    ++st.q;
  }
}
```

在开关关闭时（性能二进制文件，笔记的 H.1 部分），`EXPLICIT_WAIT(addr, parity, site, tile, q)` 就是 `mbar_wait(addr, parity)` (explicit.cu:237)，因此 `tile` 参数和 `uint32_t(st.q)` 实参也随之消失；位于 explicit.cu:742-746 和 749-753 的两个 `[TRACE]` 块在同一个二进制文件中被编译掉了，在上面的清单中省略。剩下的是十来条语句。

参数。`params` 是 `__grid_constant__` 参数块（第 5 章），只用于 `&params.tma_a` 和 `&params.tma_b`，即参数空间中两个 128 字节 `CUtensorMap` 的通用地址。`st` 是环状态，`n` 是本次调用要发射的 k-tile 数。`smem_base` 是本 CTA 的 32 位 `shared::cta` 基址。五个不变量 `a_dst_base`、`b_dst_base`、`full_masked_base`、`empty_base`、`a_mcast_mask` 每个角色计算一次（第 8.6.2 节）；谓词 `is_leader_lane` 在角色分发处计算一次（explicit.cu:1192，第 8.4.4 节）；`a_row`/`b_col` 每个 tile 计算一次（第 8.6.3 节）。

### 8.4.1 循环前的 peek 与 token（第 732 行）

```
  uint32_t tok = mbar_try_wait(empty_base + 8u * st.prod_idx, st.prod_phase);            // producer_try_acquire before the loop (:604)
```

token 在循环*之前*计算，这样第一次迭代就和之后的每一次迭代一样，用单个谓词来决定自己是否必须阻塞。

### 8.4.2 `#pragma unroll 1`（第 733-734 行）：为什么 k-tile 循环保持不展开

```
  #pragma unroll 1                                                                          // CUTLASS_PRAGMA_NO_UNROLL (:607)
  while (n > 0) {
```

CUTLASS 把同一个循环标记为 `CUTLASS_PRAGMA_NO_UNROLL` (`sm100_mma_warpspecialized.hpp:607`)。三个理由，按重要性递减排列：

1. **无利可图。** 每次迭代都由两次异步 TMA 发射主导，其代价由 TMA 单元支付，而不是这个 warp。该 warp 自己的工作是每个 k-tile 37 条指令（快速路径上 33 条），其中大多数属于 uniform 数据通路（实测，第 12.5.2 节），而它一生中的大部分时间都阻塞在 `empty[]` 上（第 8.9 节）。展开无法缩短一次等待。
2. **Uniform 寄存器压力。** Uniform 寄存器（`URn`，其中 `URZ` 是 uniform 零寄存器）是 SM 的 *uniform 数据通路* 的每 warp 标量寄存器堆，在 Blackwell 上每个 warp 有 80 x 32-bit 可寻址的 `UR0`-`UR79`，`URZ` 不在其中（NVIDIA 内部），与每 lane 的向量寄存器（`Rn`）并列；第 12 章，第 12.3.1 节定义了它们，而上一条和第 8.9 节使用了这个术语。一个 ptxas 能证明为 warp-uniform 的值（一个地址、一个掩码、一个描述符、一个坐标）不占用任何每 lane 寄存器，而由每个 warp 发射一次的 `U` 前缀指令消费。baseline 的紧缺资源是那个 uniform 寄存器堆，而不是向量寄存器堆（笔记 D.7，由 CUTLASS kernel 中 44 对 uniform 到向量的溢出/填回推断）。一个展开的循环体会让多个 stage 的地址和 token 同时存活；不展开时，循环只需要一个槽位地址、一个 token、一个坐标和那些不变量。
3. **代码大小与性能契约。** baseline 的 SASS 恰好有两个未展开的 producer 循环，分别为 8 次和 120 次迭代，各含一条 `UTMALDG.3D.MULTICAST.2CTA` 和一条 `UTMALDG.3D.2CTA`（笔记 D.2，实测："loop 1 counter runs -7..1 (8 iterations), loop 2 8..k_tiles (120); loop 2 body = 31 instructions"）。explicit kernel 复现了这一形状（笔记 G.3，实测；第 12.5.2 节把指令数与 baseline 的做了对照）。笔记的第 12 节把指令流的形状列为性能等价契约的一部分（"the instruction-stream shape should match"）。

### 8.4.3 仅当 token 说“未完成”时才阻塞等待（第 735 行）

```
    if (!tok) { EXPLICIT_WAIT(empty_base + 8u * st.prod_idx, st.prod_phase, WAIT_MAINLOOP_EMPTY, tile, uint32_t(st.q)); }   // producer_acquire (sm90_pipeline.hpp:531-536)
```

W6 (`mbar_wait`, explicit.cu:296-311) 是 `LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [addr], phase, ticks; @P1 bra DONE; bra LAB_WAIT;`，其中 `ticks = 0x989680` = 10,000,000 ns = 10 ms，即 ISA 的 `timeHint`（第 2 章，第 2.8.4 节）。该提示*可能*被使用，而被挂起的线程可能提前恢复，因此外层的 `bra LAB_WAIT` 才是使等待正确的东西，而不是针对罕见情况的后备。实践中，一个大的提示值让线程睡眠直到被释放，而不是忙轮询（我对其用途的分析，与第 4.5.2 节中相同：ISA 只说线程 "may be suspended"；实测的重试循环在两次重测之间于 `NANOSLEEP.SYNCS 0x989680` 中睡眠，笔记 D.2；睡眠中的 warp 如何被唤醒：第 12 章，第 12.2.2 节）。`if (!tok)` 就是 `sm90_pipeline.hpp:533` 的 `if (barrier_token != BarrierStatus::WaitDone)`。

### 8.4.4 leader lane 的 49152 字节 `arrive.expect_tx`（第 736 行）

```
    if (is_leader_lane) { mbar_arrive_expect_tx_local(smem_base + OFF_MAIN_FULL + 8u * st.prod_idx, kMainloopTxBytes); }   // (:537-539), leader lane, 49152
```

三个“为什么”：

- **为什么 `arrive.expect_tx` 作为一条指令，以及为什么计数为 1。** Warp 0 被选中的 lane 初始化 `full[s]`（计数 1；第 4.2.2、4.4.1 节）：explicit.cu:1124-1125 处的 `if (warp_idx == 0) { if (elect_one_sync()) {`，循环在 :1126-1130。tx-count 是**有符号的**，范围为 `-(2^20 - 1) .. 2^20 - 1`（PTX ISA 9.7.15.16.3，"Contents of the mbarrier object"，两种 mbarrier layout 的范围相同），而 ISA 的 complete-tx 操作只是说 "tx-count is decremented by completeCount"。leader 的那一次 `arrive.expect_tx` 既把 tx-count 提高 49152，又把到达计数降到 0；49152 远在该范围之内（`2^20 - 1 = 1048575`）。[如果第 4.2.2 节的 ISA 规则不是这样，kernel 的那些数值相等的运行就会在 `q = 0` 处死锁——即笔记 E.7.2 失效模式表第一行中的症状。] 一个 lane 的一次到达是让完成保持受门控的最小量；32 次到达（所有 lane）将需要计数 32，却毫无所得。
- **为什么只有 pair leader 的 lane。** `is_leader_lane` 在调用点 (explicit.cu:1192) 是 `lane_pred != 0u && is_leader`：`lane_pred` 是 prologue 的 `elect_one_sync()` 的结果（explicit.cu:1103，每个 warp 一个 lane，在 kernel 的整个生命期内固定），而 `is_leader` 是 `(rank & 1) == 0` (explicit.cu:1106)。
- **为什么 peer 的 `full[]` 屏障存在却从未被使用。** 两个 CTA 运行相同的代码、使用相同的 `SharedStorage` layout（2SM MMA 要求 pair 中的 smem layout 完全相同，笔记 B.6），因此 peer 像其他所有人一样初始化它的八个 `full[]` 屏障（计数 1），并让它们永远停留在 phase 0。它们是 64 字节的 smem，使两个 CTA 的地址映射保持相同。在 SASS 中，这个 arrive 以及为其提供字节数的 `MOV`，除了基于 `try_wait` 结果的两个 `@P1`/`@!P1` 分支之外，是该循环中仅有的两条带 lane 谓词的*向量*指令：`@!P0 MOV R5, 0xc000` 和 `@!P0 SYNCS.ARRIVE.TRANS64 RZ, [UR10], R5`（笔记 G.3，实测）。注意谓词的极性：`P0` 是 `is_leader_lane` 的*补*，因此 `@!P0` 守卫在 `P0` 为假的那一个 lane 中执行这两条指令，而在 peer 的任何 lane 中都不执行。它们不能是 uniform 指令，因为只允许一个 lane 执行它们。第三条指令补全了这幅图景。`VOTEU.ALL UP1, P0` 把 lane 谓词投票成一个 uniform 谓词：当每个 lane 都置位了 `P0`，即没有人 arrive 时，`UP1` 为真。它门控着 arrive 地址的计算 `@!UP1 ULEA UR10`，因此在没有任何 lane arrive 的 peer 中，`ULEA` 被 uniform 地跳过，而 leader 则执行它（指令来自笔记 G.3；对该投票的解读是我的）。

### 8.4.5 捕获 `s`、推进、peek 下一个 stage——以及被丢弃的 peek（第 737-739 行）

```
    uint32_t const s = st.prod_idx;                                                          // write_stage
    advance<kStages>(st.prod_idx, st.prod_phase);
    tok = mbar_try_wait(empty_base + 8u * st.prod_idx, st.prod_phase);                     // peek the next stage, unconditional (:617)
```

`s` 是这个 k-tile 所写入的槽位 (`write_stage`, `sm100_mma_warpspecialized.hpp:615`)。

每次调用的**最后一次**迭代会 peek 该调用将要使用的槽位之后的下一个槽位，并把那个 token 丢掉：`tok` 是 `load_ktiles` 的局部变量，而下一次调用（或下一个 tile 的第一次调用）以它自己的循环前 peek（第 732 行）开始。`try_wait` 对屏障没有副作用，因此每个 tile 的两次浪费掉的 peek（每次调用一次）每个 tile 只花费两条 `SYNCS.PHASECHK.TRANS64.TRYWAIT`，别无其他。

### 8.4.6 一次选举，两次 TMA 发射（第 740-756 行）

```
    if (elect_one_sync()) {                                                                  // one election for the A+B pair (:619-622)
      int32_t const c0 = 64 * st.q;
```

W2 (`elect_one_sync`, explicit.cu:263-278) 是 `elect.sync`：收敛的 warp 中恰好一个 lane 获得该谓词。一次选举覆盖*两次* TMA 发射，这是 CUTLASS 所用的粒度 (`sm100_mma_warpspecialized.hpp:619-622`)，也是笔记定为契约的粒度（E.6.6："one `elect_one_sync()` per TMA A+B pair"）。它是一次*全新的*选举，而不是 `is_leader_lane`：执行 arrive 的 lane 和发射的 lane 可能不同，而且发射的 lane 在每个 CTA 中都存在（peer 也发射它自己的加载），而 `is_leader_lane` 在 peer 中为假。在 SASS 中这里没有任何 `ELECT` 幸存下来。`UTMALDG` 是一条 uniform 数据通路指令，无论活跃 lane 有多少都每个 warp 发射一次，因此 ptxas 把选举折叠掉了（笔记 D.2、D.6 第 6 项）：PTX 形式是你所写的，硬件形式是你所得到的。

`c0 = 64 * st.q` 是最内层的 TMA 坐标：box 第一个元素的 k 索引，以元素为单位。TMA 坐标元组是以**元素**为单位的 `(innermost, outer, batch)`，而每个坐标都是一个*有符号的* 32 位整数（PTX ISA，`cp.async.bulk.tensor`："The individual tensor coordinates in tensorCoords are of type .s32"）。这就是为什么 `c0`、`a_row` 和 `b_col` 是 `int32_t`/`int`，而 W21/W22 用针对 `int32_t` 的 `"r"` 约束来绑定它们。对 A 和 B 的 map 而言，元组是 `(k, row, l)`（第 5 章：这些 map 以 mode 顺序 `(K, M, L)` / `(K, N, L)` 编码，笔记第 5.3 节）。落在张量之外的 box 元素会被零填充：这些 map 以 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` 编码（笔记第 5.3 节），即 ISA 的 "Zero fill mode"（"Elements in the Bounding Box which fall outside of the tensor boundary are set to 0"）。本次运行中没有任何越界（K = 128 x 64 恰好成立，笔记第 10 节）。但这正是为什么一个不是 64 的倍数的 K 用这同一个循环仍会得到正确的 D：`k_tiles = (params.k + 63) >> 6` (explicit.cu:1112) 向上取整，最后一个 k-tile 中 A 和 B 的尾部列读出为零，而零对累加器没有任何贡献 [源自源码；固定运行并未执行到该余数路径]。

A 的发射是 W21（explicit.cu:449-460，在第 6.6.3 节中引用）。

它的调用形式为 `tma_load_2sm_mcast_3d(&params.tma_a, full_masked_base + 8u * s, a_mcast_mask, kCacheHintEvictNormal, a_dst_base + kStageBytesA * s, c0, a_row, 0)`。逐个操作数来看：

| PTX 操作数 | C++ 实参 | 本 CTA 的取值 | 为什么 |
|---|---|---|---|
| `[%0]` 目的地 (`shared::cluster`) | `a_dst_base + 16384 s` | `smem_base + 33792 + 16384 s + 8192 y` | 本 CTA 的 64 行半块进入 stage `s` 的上部（`y = 1`）或下部（`y = 0`）8192 字节。multicast 在每个目的 CTA 中写入*相同的 CTA 相对偏移*（笔记 B.6；Colfax：该操作 "places the loaded data into the SMEMs of all CTAs specified by the bitmask"；gau-nernst 的 tcgen05 帖子（"tcgen05 for dummies"）：smem layout "must be identical across cluster ranks"，正是出于这个原因），因此两个接收方都在正确的位置得到这半块。该地址的 rank 位 [24,26) 指明发射 CTA；TMA 对另一个接收方使用其中的偏移部分。 |
| `[%1, {%4,%5,%6}]` tensor map + 坐标 | `&params.tma_a`, `(c0, a_row, 0)` | `(64 q, 128 tm + 64 y, 0)` | 64 个 k 值（box 的一个 128 字节行），从本 CTA 的半块开始的 64 行，batch 0。该 box 为 `(64, 64, 1)` fp16 = 8192 B（第 5 章，笔记第 5.3 节：每个 CTA 的 A box 是 128 行的 tile *除以 multicast 计数 2*，`copy_traits_sm90_tma.hpp:996-1002`）。 |
| `[%2]` mbarrier | `full_masked_base + 8 s` | `(smem_base & 0xFEFFFFFF) + 8 s` = **pair leader 的** `full[s]` | `.cta_group::2` 允许屏障操作数指向 pair 中的任一 CTA——PTX ISA，`cp.async.bulk.tensor`："When .cta_group::2 is specified, the mbarrier object mbar can be in shared memory of either the same CTA as the shared memory destination dstMem or in its peer-CTA"；清除地址第 24 位选中偶数 rank (`Sm100MmaPeerBitMask = 0xFEFFFFFF`, `include/cute/arch/copy_sm100_tma.hpp:45`)。CUTLASS 在 `:63/:89/:115` 处应用该掩码，并在 `SM100_TMA_2SM_LOAD_MULTICAST_3D`（`:281-305`，其 `copy` 成员即 W21 注释中的 `:283-305`）中于 `:292` 处应用。使用 multicast 时，每个目的 CTA 的信号都送往*它自己的* pair leader 中该偏移处的屏障——ISA："for each destination CTA specified in the ctaMask, the mbarrier signal is sent either to the destination CTA or its peer-CTA based on the %cluster_ctarank parity of the shared memory where the mbarrier object mbar resides"（笔记 B.6；Colfax："a CTA can find the address of its leader CTA's mbarrier by taking its own mbarrier address and setting the 24th bit to 0"）。 |
| `%3` ctaMask (`.b16`, `"h"`) | `a_mcast_mask` | `0x5 << x`：`x = 0` 时为 `0x5`（rank 0、2），`x = 1` 时为 `0xA`（rank 1、3） | 具有相同 `x` 的两个 CTA 共享 `tm`（第 8.5.4 节），因此需要 A 的相同 128 行。掩码的第 `i` 位 = rank `i` 接收该 box。CUTLASS 用 `create_tma_multicast_mask<2>` 推导该掩码（`include/cute/atom/copy_traits_sm90_tma.hpp:1468-1501`，经由 `:1504-1511` 处的投影重载到达，它在除 N 之外的每个 mode 上对 `(v, m, n, k)` cluster layout 做切片）；explicit kernel 则写出该结果 (explicit.cu:772)。 |
| `%7` 缓存策略（`.b64`、`"l"`） | `kCacheHintEvictNormal` | `0x1000000000000000` | `TMA::CacheHintSm100::EVICT_NORMAL`（`include/cute/arch/copy_sm90_desc.hpp:194`），即 CUTLASS 的 `with()` 的默认值（`include/cute/atom/copy_traits_sm100_tma.hpp:182, 193`）。`.L2::cache_hint` 限定符是所拷贝字符串的一部分，所以必须提供一个操作数；EVICT_NORMAL 是中性的策略。在 SASS 中它是 64 位的 `desc[UR32] = {0x0, 0x10000000}`（笔记 G.3）。 |

B 的发射（W22，explicit.cu:462-473，引文见第 6.6.4 节）是同一条指令，只是没有 `.multicast::cluster`，也没有掩码操作数。

它的调用形式是 `tma_load_2sm_3d(&params.tma_b, full_masked_base + 8u * s, kCacheHintEvictNormal, b_dst_base + kStageBytesB * s, c0, b_col, 0)`：目的地 `smem_base + 164864 + 8192 s`（这个 CTA 的整个 64 x 64 B stage，没有 `y` 项），坐标 `(64 q, 128 tn + 64 x, 0)`（64 个 k 值，tile `tn` 的 128 个 N 列中属于这个 CTA 的那 64 列），同一个 pair-leader 屏障，不做 multicast。B 仍然带有 `.cta_group::2`：正是它允许屏障操作数是 leader 的，即使数据纯粹是本地的。

既然只有 leader 做计算，为什么 *peer* 还要发射加载？因为 2SM MMA 直接读取 peer 的 smem（第 9 章，笔记 B.6："A rows [0,128) and B columns [0,64) of the 256x128 instruction come from the leader's smem and the remaining halves from the peer's smem at the same CTA-relative addresses"）。因此 peer 必须填满自己的环，逐个槽位地与 leader 步调一致。两个 producer 都会自动做到这一点，因为它们等待的 `empty[]` 屏障是由相同的 commit 操作释放的。

为什么是被选中的 lane，而不是全部 32 个 lane？这条 PTX 指令是每线程的；32 个 lane 会把每个 box 发射 32 份。（epilogue 的 TMA *存储*是刻意由 warp 4 的全部 32 个 lane 发射的，因为 baseline 就是这么做的；ptxas 反正会把它变成一条 uniform `UTMASTG`，第 10 章。不同的历史，相同的硬件结果。）

### 8.4.7 `--n; ++st.q;`（第 757-758 行）

`n` 对本次调用的 k-tile 倒数计数；`st.q` 则为坐标对 k 索引递增计数。在 CUTLASS 中它们是 `--k_tile_count; ++k_tile_iter;`，其中对于 rank-1 的 K mode，迭代器就是一个普通整数（笔记 B.3.1 步骤 11）。保留两者正是让每个 tile 两次调用变得平凡的原因：第二次调用从 `st.q = 8` 开始，此时 `n = 120`。

---

## 8.5 跨 cluster 装配一个 stage

### 8.5.1 谁加载哪些行与列

设 cluster 的 256 x 256 tile 从 CTA tile `(Tm, Tn)` 开始，其中 `Tm`、`Tn` 为偶数。Rank `r = x + 2y` 处理 CTA tile `(tm, tn) = (Tm + x, Tn + y)`（第 8.5.4 节）。它的加载：

| rank | `(x, y)` | tile `(tm, tn)` | A box 的行（`a_row .. +64`） | A 落在（掩码） | A 在 stage 中的偏移 | B box 的列（`b_col .. +64`） | B 落在 |
|---|---|---|---|---|---|---|---|
| 0 | (0, 0) | `(Tm, Tn)` | `128 Tm + 0` | rank 0、2（`0x5`） | `+0` | `128 Tn + 0` | rank 0 |
| 1 | (1, 0) | `(Tm+1, Tn)` | `128 (Tm+1) + 0` | rank 1、3（`0xA`） | `+0` | `128 Tn + 64` | rank 1 |
| 2 | (0, 1) | `(Tm, Tn+1)` | `128 Tm + 64` | rank 0、2（`0x5`） | `+8192` | `128 (Tn+1) + 0` | rank 2 |
| 3 | (1, 1) | `(Tm+1, Tn+1)` | `128 (Tm+1) + 64` | rank 1、3（`0xA`） | `+8192` | `128 (Tn+1) + 64` | rank 3 |

### 8.5.2 A-stage 装配图

```
 cluster tile (Tm, Tn): 256 rows of A (two tm) x 256 columns of B/D (two tn)

                 y = 0  (tn = Tn)                       y = 1  (tn = Tn + 1)
            +----------------------------+        +----------------------------+
  x = 0     | rank 0                     |        | rank 2                     |
  tm = Tm   | A stage s (16384 B):       |        | A stage s (16384 B):       |
            |  [+0    ] rows 128Tm+0..   | --+--> |  [+0    ] rows 128Tm+0..   |  <- loaded by rank 0, multicast 0x5: lands in ranks 0 and 2
            |  [+8192 ] rows 128Tm+64..  | <--+-- |  [+8192 ] rows 128Tm+64..  |  <- loaded by rank 2, multicast 0x5: lands in ranks 0 and 2
            | B stage s (8192 B):        |        | B stage s (8192 B):        |
            |  cols 128Tn + 0..63        |        |  cols 128(Tn+1) + 0..63    |  <- each CTA loads its own, no multicast
            +----------------------------+        +----------------------------+
  x = 1     | rank 1                     |        | rank 3                     |
  tm = Tm+1 |  [+0    ] rows 128(Tm+1)+0 | --+--> |  [+0    ] rows 128(Tm+1)+0 |  <- loaded by rank 1, multicast 0xA: lands in ranks 1 and 3
            |  [+8192 ] rows 128(Tm+1)+64| <--+-- |  [+8192 ] rows 128(Tm+1)+64|  <- loaded by rank 3, multicast 0xA: lands in ranks 1 and 3
            |  B cols 128Tn + 64..127    |        |  B cols 128(Tn+1)+64..127  |
            +----------------------------+        +----------------------------+
             \________ MMA pair {0,1} _______/     \________ MMA pair {2,3} _______/
              A: 256 rows [128Tm, 128Tm+256)        A: the same 256 rows
              B: 128 cols [128Tn, 128Tn+128)        B: 128 cols [128(Tn+1), +128)

 arrows = the direction of the multicast copy, from the issuing CTA into its sibling (same x, other y):
   the [+0] half is issued by the y = 0 CTA and flows left -> right into rank 2 / rank 3;
   the [+8192] half is issued by the y = 1 CTA and flows right -> left into rank 0 / rank 1
```

当 cluster 对一个 stage 的四次 A 发射和四次 B 发射都已落地之后，每个 CTA 都持有它自己的 `tm` 的*整个* 128 x 64 A stage（自己的那一半，加上由 `x` 相同的同胞 multicast 过来的那一半）以及它自己的 64 x 64 B 半块。随后 pair leader 的 `tcgen05.mma.cta_group::2` 把 A 看作 256 行（leader 的 128 + peer 的 128），把 B 看作 128 列（leader 的 64 + peer 的 64）；每个 CTA 的 TMEM 接收它自己的 `(tm, tn)` 的 128 x 128 累加器（第 9 章）。

[源自源码 + 实测] 这正是第一个 cluster 的 K4a 记录对 `q = 0` 所显示的内容（笔记 D.1 D8；在第 8.7.2 节中重新计算）。

### 8.5.3 一个 leader 屏障的 49152 字节台账

取 pair `{0, 1}` 及其 leader 的 `full[s]`（rank 0 的，位于 `0x00000400 + 8 s`）。哪些 box 计入它？每一个其目的 CTA 以 rank 0 为 pair leader 的 box，也就是每一个落在 rank 0 **或** rank 1 中的 box：

| box | 由谁发射 | 落在 | 是否计入 rank 0 的 `full[s]`？ | 是否计入 rank 2 的 `full[s]`？ |
|---|---|---|---|---|
| A 的行 `128Tm + 0..63` | rank 0 | rank 0、2 | 是（经由 rank 0） | 是（经由 rank 2） |
| A 的行 `128Tm + 64..127` | rank 2 | rank 0、2 | 是（经由 rank 0） | 是（经由 rank 2） |
| A 的行 `128(Tm+1) + 0..63` | rank 1 | rank 1、3 | 是（经由 rank 1，peer 位） | 是（经由 rank 3，peer 位） |
| A 的行 `128(Tm+1) + 64..127` | rank 3 | rank 1、3 | 是（经由 rank 1，peer 位） | 是（经由 rank 3，peer 位） |
| B 的列 `128Tn + 0..63` | rank 0 | rank 0 | 是 | 否 |
| B 的列 `128Tn + 64..127` | rank 1 | rank 1 | 是（peer 位） | 否 |
| B 的列 `128(Tn+1) + 0..63` | rank 2 | rank 2 | 否 | 是 |
| B 的列 `128(Tn+1) + 64..127` | rank 3 | rank 3 | 否 | 是（peer 位） |
| **总计** | | | **6 x 8192 = 49152** | **6 x 8192 = 49152** |

每个 A box 都计入*两个* leader，因为两个 pair 都需要全部 256 个 A 行；每个 B box 计入一个 leader，因为 B 是拆分的，而不是共享的。每个 cluster 每个 k-tile，TMA 单元从 L2 搬运 8 个 box = 65536 字节，并写入 4 x 24576 = 98304 字节的共享内存（A 的各半块被写了两次）。若没有 2SM 和 multicast，四个独立的 128 x 128 CTA tile 每个 k-tile 将从 L2 读取 4 x (16384 + 16384) = 131072 字节——两倍之多 [我的分析]。

右侧各列之所以成立，只是因为每个 box 的屏障操作数都是 leader 的加掩码地址（`full_masked_base`，explicit.cu:770）。若操作数不加掩码，peer 的三个 box 将计入 peer 那个从未装填的屏障，leader 的 `full[s]` 将停顿在 49152 字节中的 24576 字节处，而每次运行都会在 `q = 0` 处死锁，即笔记 E.7.2 失败模式表的第一行（"unmasked TMA barrier operand (peer's boxes credit the peer)"）。相反，每次运行都以数值相等告终 (G.1)，并且 SASS 显示发射的是加了掩码的操作数：baseline 在每个 producer 循环内部对屏障寄存器加掩码（`sass_gemm.txt` 的 `0x1b90`/`0x1e80` 处的 `ULOP3 ..., 0xfefffff8`；笔记第 7.5 节），而 explicit kernel 则在两个循环之前把掩码一次性折叠到已外提的基址上（`0x0c00` 处的 `ULOP3 ..., 0xfefffff0`），并在每个 k-tile 用 `ULEA` 从它构成 `full[s]`（G.3；第 2.9.3 节）。

### 8.5.4 为什么 `tm % 2 == x`，因此 `a_row = 128 tm + 64 y`

explicit.cu:777 上的注释说 "tm % 2 == x for every tile this CTA receives (B.3.1 step 13)"。由两个映射公式给出的证明：

- **启动 tile**（explicit.cu:1175-1176）：`tm = 2 (blockIdx.y >> 1) + (blockIdx.x & 1)`，`tn = 2 (blockIdx.x >> 1) + (blockIdx.y & 1)`。cluster 是 `(2, 2, 1)`，而各 cluster 按 `blockIdx` 顺序以 tile 方式铺满网格，所以 `%cluster_ctaid.x = blockIdx.x & 1 = x` 且 `%cluster_ctaid.y = blockIdx.y & 1 = y`（explicit.cu:1173-1174 的注释：`x == rank & 1`、`y == rank >> 1`）。这一映射是 `possibly_transpose_work_tile` 的 `AlongN` 转置（`include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:668-677`）：网格 x 方向的 cluster 索引（`blockIdx.x >> 1`）索引 N，y 方向的（`blockIdx.y >> 1`）索引 M，而 cluster 内偏移 `blockIdx.x & 1` 和 `blockIdx.y & 1` 仍归 M 和 N。因此 `tm % 2 = blockIdx.x & 1 = x`。
- **CLC tile**（explicit.cu:692-693，第 7 章）：`tm = 2 (y0 >> 1) + (x0 & 1) + x`，`tn = 2 (x0 >> 1) + (y0 & 1) + y`，其中 `(x0, y0)` 是被取消的 cluster 的*第一个* CTA 的启动网格 id。一个 cluster 的第一个 CTA 是 cluster 对齐的，所以 `x0` 和 `y0` 为偶数（对全部 991 个被取消的响应实测，笔记 D.3："every valid first-CTA id is even"），`& 1` 项消失，于是 `tm = y0 + x`、`tn = x0 + y`。因此又一次有 `tm % 2 = x`。

所以该 CTA 处理的每个 tile 都有 `tm = Tm + x`（`Tm` 为偶数）和 `tn = Tn + y`（`Tn` 为偶数）。`x` 相同的两个 CTA（rank `x` 和 `x + 2`，`y = 0` 和 `1`）共享 `tm`，并需要同样的 128 个 A 行 `[128 tm, 128 tm + 128)`。它们按 `y` 拆分工作：rank `(x, y)` 加载行 `[128 tm + 64 y, +64)` 并把它们 multicast 给两者。这就是 `a_row = 128 tm + 64 y`（explicit.cu:777），其目的地为 `a_dst_base = smem_base + OFF_SMEM_A + 8192 y`（explicit.cu:768），从而使 `y = 1` 的那一半在两个接收方中都落在 stage 的上部 8192 字节里。对称地，一个 pair 的两个 CTA（`y` 相同）共享 `tn` 并按 `x` 拆分 B：`b_col = 128 tn + 64 x`（explicit.cu:778），目的地 `OFF_SMEM_B + 8192 s`。其中没有 `x` 项：每个 CTA 只保留自己的那一半，而 MMA 从 peer 读取另一半。

CUTLASS 并不是通过 `%cluster_ctaid.x` 来选择 B 半块的。它的 B 坐标是 `128 tn + 64 V`，其中 `V = blockIdx.x % 2`，即 `TiledMma{}.get_slice(blockIdx.x % size(AtomThrID))` 的切片索引（`sm100_mma_warpspecialized.hpp:516`，笔记 B.3.1 步骤 3；在数值上等于 kernel 的 `cta_coord_v = rank % 2`，kernel :427），也就是该 CTA 在 MMA pair 中的位置（笔记 B.3.1 步骤 13：`tBgB = tBgB_nkl(_, tn, _, 0)`，坐标 `(0, 128*tn + 64V, 0)`；笔记 B.4：`(64q, 128*tn + 64*(tm%2), 0)`）。它的 A 坐标是 `256 (tm/2) + 128 V + 64 y`。explicit kernel 写的是 `64 x` 和 `128 tm + 64 y`，因为 `V = rank % 2 = x = tm % 2` 是两种 tile 映射的不变量（上文已证明；`r = x + 2y` 给出 `rank % 2 = x`）。

笔记第 7.5 节用 cluster 的启动坐标给出了同样的公式：对于位于 `(CX, CY)` 的启动 cluster，rank `r` 从 `256 CY + 128 x + 64 y` 开始加载 A 的行，从 `256 CX + 128 y + 64 x` 开始加载 B 的列。

### 8.5.5 为什么 A 沿 `y` multicast，而 B 完全不 multicast

需要同一个操作数 tile 的两个 CTA，要么各自加载它（两次 L2 读取），要么由一个加载并 multicast（一次 L2 读取，两次 smem 写入）。在这个 cluster 内部：

- 同样的 **A** 行被 `x` 相等的两个 CTA（不同的 pair）所需要。再没有别的东西把这两个 CTA 联系起来，所以不做第二份拷贝而共享这次加载的唯一办法就是 TMA multicast。128 行的 tile 被拆成两个 64 行的半块，使得两个 CTA 都发射工作，而 L2 流量减半（笔记第 5.3 节：`make_tma_atom_A_sm100` 传入 `num_multicast = 2`，而 box 从 `(64, 128, 1)` 截断为 `(64, 64, 1)`）。
- 同样的 **B** 列被一个 pair 的两个 CTA 所需要。但这个 pair 已经由 2SM MMA 连在一起，后者读取两个 CTA 的 smem；所以 B 被*拆分*，每个 CTA 加载自己的 64 列，B 的任何字节都不会被写两次。沿 `x` 对 B 做 multicast 只会白白让 smem 写入翻倍。（Colfax 的 cluster 教程描述的是通用的 1SM 模式——A 沿 cluster 的行 multicast，B 沿 cluster 的列；有了 2SM MMA，B 的共享就被该指令本身吸收了。）[源自源码；设计理由是我的分析]

这个设计*没有*采取的选项：跨 pair 对 A 做 multicast（两个 `tm` 不同，所以没有什么可共享的），或者 4-CTA multicast（除了各 `x`-pair 之外，没有任何两个 CTA 共享 A）。`(2,2,1)` cluster 恰恰是每个操作数在每个 cluster 中只加载一次的那种 shape。

---

## 8.6 逐行解读 producer_role (explicit.cu:762-807)

```
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
    valid = clc_consume<0u>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // consume response t AFTER this tile's loads (:665-670)
    tm = tm_next; tn = tn_next;                                                                // the fetched tile becomes current only here
    ++tile;
  } while (valid);
  // load_tail = producer_tail (sm90_pipeline.hpp:445-454): 8 blocking waits continuing the producer state (Section 7.10 tail 1)
  #pragma unroll
  for (int i = 0; i < kStages; ++i) {
    EXPLICIT_WAIT(empty_base + 8u * st.prod_idx, st.prod_phase, TAIL_MAINLOOP, tile, uint32_t(i));
    advance<kStages>(st.prod_idx, st.prod_phase);
  }
}
```

调用点是 explicit.cu:1192，位于 :1191-1193 的 `if (warp_idx == 2) {` ... `}` 之内：`producer_role(params, smem_base, x, y, is_cta0, lane_pred != 0u && is_leader, k_tiles, tm, tn);`。

### 8.6.1 `griddep_wait()`：PDL 在这里做什么（第 764 行）

这次启动**不**带 PDL 属性（`launch_with_pdl == false`，笔记第 5.4 节），所以不存在前置网格，`griddepcontrol.wait` 立即完成 [推导：PTX 语义 + 启动配置]。保留它是为了语义保持：CUTLASS kernel 执行它，而如果用户日后在 A/B 的某个 producer 之后用 PDL 启动这个 kernel，正是这次 wait 使 TMA 加载读到的是已完成的数据。warp 4-7 在两个 kernel 中都不等待：它们唯一的全局内存流量是对 D 的 TMA 存储，其数据来源于两个 kernel 中都在 warp 2 的 wait 之后加载的 A 和 B（explicit.cu:764；kernel :620）[我的分析]。

### 8.6.2 不变量（第 765-775 行）

| 局部变量 | 值 | 为什么把它外提 / 为什么是这个值 |
|---|---|---|
| `st` | `{0, 1, 0}` | 第 8.3 节。 |
| `thr_idx, thr_phase` | `{0, 1}` | Throttle *producer* 状态（2 个槽位）；producer 从 phase 1 开始，如第 8.3 节所述，所以 CTA 0 的前两个 tile 能通过 throttle 等待（第 8.6.4 节）。 |
| `clc_idx, clc_phase, clc_count` | `{0, 0, 0}` | CLC *consumer* 状态（2 个槽位）：consumer 从 phase 0 开始，并阻塞到第一个响应落地为止（第 7 章）。 |
| `a_dst_base` | `smem_base + 33792 + 8192 y` | `OFF_SMEM_A = 33792`（explicit.cu:154）加上这个 CTA 的半块偏移；`kStageBytesAHalf = 8192`（explicit.cu:117，"equal to kStageBytesB by coincidence of the tile shape"——一个 128 x 64 fp16 tile 的一半恰好等于一个 64 x 64 fp16 tile）。 |
| `b_dst_base` | `smem_base + 164864` | `OFF_SMEM_B`（explicit.cu:155）。 |
| `full_masked_base` | `smem_base & 0xFEFFFFFF` | `kPeerBitMask`（explicit.cu:187）。因为 `OFF_MAIN_FULL = 0`，`full[s]` 就是 `smem_base + 8 s`，而对基址加一次掩码等价于对每一个 `(smem_base + 8 s)` 加掩码：掩码只触及第 24 位，而 `8 s <= 56` 不可能从一个低 24 位为 `0x400` 的基址进位到该位。实测值：rank 0 和 1 为 `0x00000400`，rank 2 和 3 为 `0x02000400`（笔记 D.1 D5：`base & 0xFEFFFFFF` 是偶数伙伴的基址）。nvcc 把已知为零的低位折叠进常量：CUTLASS kernel 中是 `0xFEFFFFF8`（一个 8 字节屏障的三个对齐位，笔记 D.6 第 2 项），这里则是 `0xFEFFFFF0`（`and.b32 ..., -16777232`，因为基址本身已加掩码；笔记 G.3）；对这些地址而言结果完全相同。explicit.cu:187 处的源码注释（"nvcc folds it to 0xFEFFFFF8"）只对 epilogue 的累加器释放位置（`0x44c0`，笔记 G.3）成立，对 producer 的 SASS 则不成立，后者的折叠是上面的 `0xFEFFFFF0`。 |
| `empty_base` | `smem_base + 64` | `OFF_MAIN_EMPTY`（explicit.cu:140）。 |
| `a_mcast_mask` | `0x5 << x`（16 位） | 第 8.4.6 节。`"h"` 约束需要一个 `uint16_t`。 |
| `tm_next, tn_next, tile, valid` | `tm, tn, 0, -` | CLC 消费把*下一个* tile 写入 `tm_next/tn_next`；当前 tile 留在 `tm/tn` 中直到循环体结束（第 8.6.7 节）。`tile` 是 tile 计数器，仅由 trace 使用。 |

把这些外提，并不是一项 nvcc 需要帮助才能完成的优化。这是笔记 D.7 的纪律（"the explicit kernel should keep loop-invariant operands few (descriptor bases, masks, barrier bases) and let ptxas keep them uniform"），并在笔记 E.6.6 中被重述为一条规则。在实测的 SASS 中，四个循环不变量（其中只有一个，即掩码，来自这张表）仍然在每个 k-tile 被重新物化，共五条指令（B 的 tensor-map 地址、掩码插入 `UPRMT ..., 0x5410`、z 坐标、`0xc000`）。它们是相对于 baseline 循环的*净* +6 条指令中最大的一部分（笔记 G.3）。第 8.9 节列出了全部八条，并论证它们没有任何代价；第 12.5.2 节给出了对照 baseline 循环的完整台账。

### 8.6.3 per-tile 设置（第 777-779 行）

`a_row` 与 `b_col` 根据当前的 `(tm, tn)` 重新计算（第 8.5.4 节），`st.q` 被重置为 0——但 `st.prod_idx/prod_phase` 不被触碰：环继续运转。

### 8.6.4 CTA 0 上的 throttle producer（第 783-787 行）

```
    if (is_cta0) {                                                                             // throttle producer_acquire + producer_commit (kernel :632-636)
      EXPLICIT_WAIT(smem_base + OFF_THROTTLE_EMPTY + 8u * thr_idx, thr_phase, WAIT_THROTTLE_EMPTY, tile, 0u);   // blocking, no peek (PipelineAsync)
      mbar_arrive_local(smem_base + OFF_THROTTLE_FULL + 8u * thr_idx);                        // all 32 lanes (count 32)
      advance<kThrottleStages>(thr_idx, thr_phase);
    }
```

屏障：`throttle_full[2]` 位于 `OFF_THROTTLE_FULL = 304`，`throttle_empty[2]` 位于 `OFF_THROTTLE_EMPTY = 320`，二者计数均为 32（explicit.cu:148-149, 185；由 warp 3 初始化，explicit.cu:1155-1162）。这次 arrive 是 `producer_commit`，每个线程一条 `mbarrier.arrive.shared::cta.b64`（`sm90_pipeline.hpp:1193-1196`）；因此有 "all 32 lanes (count 32)" 以及 explicit.cu:345-354 处的 W9。

**为什么是 warp 2 而不是 warp 0，以及为什么是阻塞等待。** 把查询绑定到 warp 2 的进度上，使它在“提前一个”规则（第 4.4.5 节）所允许的范围内尽早发出，这让响应（中位延迟 1.4 us，笔记 D.3）在任何人需要它之前有最多的时间到达 [我的分析]。这个 wait 是阻塞的，不带 peek，因为它几乎总是立即通过，而且每个 tile 只发生一次；一次 peek 及其 token 测试会为一个几乎从不阻塞的 wait 增加指令 [我的分析；CUTLASS 只是以其默认的 `WaitAgain` token 调用 `PipelineAsync::producer_acquire`，`sm90_pipeline.hpp:1110`]。

**释放领先多远。** 对于每个 tile `t >= 2`，warp 2 所依赖的那次释放（scheduler 在 tile `t - 2` 之后对槽位 `t % 2` 的释放，第 4.3.5 节）大约已有两个 tile 周期之久：在 Release 运行中约为 2 x 27.8 us = 55.6 us（32 个 tile 共 0.889354 ms），在 trace 构建中为 2 x 31.9 us（笔记 D.3 的 per-tile 时间）[`t - 2` 这一关系源自代码；per-tile 时间是实测的；其乘积是我的推断]。笔记 D.3 的 28.8-37.7 us（中位数 30.8 us）是一个*不同的*数字：scheduler 从消费响应 `n` 到发出查询 `n + 1` 之间的间隔，即 scheduler 每个 tile 在其 `throttle_full` wait 中停留多久。这一测量表明，等待的是 scheduler，而不是 warp 2。

**为什么是两个槽位而不是一个。** 若只有一个槽位，处于 tile `t + 1` 的 warp 2 将一直阻塞，直到 scheduler 醒来并消费了 tile `t` 的 arrive。有了两个，warp 2 可以在 scheduler 消费 tile `t` 的那次 arrive 之前就为 tile `t + 1` arrive，于是 producer 从不受制于 scheduler 的唤醒延迟，而 scheduler 仍被限制为只提前一个查询 [我的分析；这个 2 是 baseline 的 stage 数，笔记第 7.2 节，`kThrottleStages = 2` 位于 explicit.cu:110]。

### 8.6.5 `k_tile_prologue` 与两次 `load_ktiles` 调用（第 788-791 行）

```
    int const k_tile_prologue = k_tiles < kPrologueKTiles ? k_tiles : kPrologueKTiles;        // min(Stages, k_tile_count) (kernel :629)
```

`kPrologueKTiles = kStages = 8`（explicit.cu:107, 115）；`k_tiles = (params.k + 63) >> 6 = 128`（explicit.cu:1112）。第一次调用发射 `min(8, k_tiles)` 个 k-tile——对环的一整趟遍历，在一个全新的环上它从不阻塞（第 8.3 节）——而第二次调用发射剩余的 `k_tiles - 8 = 120` 个。为什么要两次调用？在 CUTLASS 中，二者之间放着

```
        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }
```

当 `beta != 0` 时，epilogue 的 C 加载 warp（warp 3）只有在 mainloop producer 已经发射了它对环的第一趟加载之后，才可以开始它对 C 的 TMA 加载，这样 mainloop 就在 TMA 单元和 L2 上抢得先机 [我的分析；CUTLASS 没有陈述理由]。对于 `beta = 0`，这次 arrive 被跳过（`is_epi_load_needed` 为 false；explicit kernel 甚至不初始化加载顺序屏障，explicit.cu:141-143，"reserved, not initialised"）。但这两次 `load` 调用在 baseline 中保留了下来，并且每一次在 SASS 中都成为它自己的未展开循环："two rolled k-tile loops (8 and 120 iterations)"（笔记 D.2，实测）。explicit kernel 保留两次调用的理由，与它保留 `#pragma unroll 1` 的理由相同：指令流的形态是契约的一部分，而两种方式下语义完全相同（状态存放在 `st` 中，第二次调用在其循环之前会重新 peek，第 8.4.5 节）。对于 `k_tiles <= 8`，第二次调用运行零次迭代（它循环前的 peek 仍然执行一次，无害）；对于 `k_tiles == 0`，两次调用都是如此。

### 8.6.6 `__syncwarp()`（第 792 行）

W37，`bar.warp.sync -1`。CUTLASS："Sync warp to prevent non-participating threads entering next wave early"（`sm100_gemm_tma_warpspecialized.hpp:663-664`）。在 PTX 模型中，只有被选中的 lane 执行了 TMA 发射；其他 31 个 lane 可能在被选中的 lane 仍在发射其 TMA 加载时，就抢先跑进 CLC 消费。`__syncwarp` 先让 warp 重新收敛。在 SASS 中它不产生任何指令（笔记 D.2："warp 2's per-tile `__syncwarp()` (`ptx.txt:1026`) produced no instruction"），因为 ptxas 把选举折叠进了 uniform 指令，没有留下需要修复的发散。你写下它是为了 PTX 契约。

### 8.6.7 加载之后的 `clc_consume`（第 793-795 行）

```
    valid = clc_consume<0u>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // consume response t AFTER this tile's loads (:665-670)
    tm = tm_next; tn = tn_next;                                                                // the fetched tile becomes current only here
    ++tile;
```

`clc_consume`（第 7 章）在本 CTA 的 `clc_full[idx]` 上等待，解码 16 字节的响应，执行 fence，远程 arrive 到 CTA 0 的 `clc_empty[idx]` 上（这个 warp 的 32 次到达是那 800 次的一部分），推进 2 槽位的状态，并且，如果响应是一个被取消的 cluster，就把下一个 tile 写入 `tm_next/tn_next`。模板实参 `0u` 的意思是 "no record"（explicit.cu:675；开关关闭时，`K_CLC_SCHED_KIND` 与 `K_CLC_MMA_KIND` 也都是 `0u`，:238-239）。

**为什么在加载之后？** 如果 warp 2 在发射 tile `t` 的加载*之前*消费响应 `t`，它会在每个 tile 开始时停顿 ~1.4 us，而环在此期间排空，这是一个 MMA warp 会在八个 k-tile 之后感受到的气泡。在 128 个 k-tile 的加载*之后*消费（大约 28 us，因为环限定着 producer 的节奏），会发现响应早已落地，于是这次消费是免费的。baseline 的注释与代码顺序（`fetch_next_work` 在 `:665` 处，位于加载之后；`:742` 在 `mma()` 之前；`:878` 在 `store()` 之前）被原样保留。[源自源码；延迟方面的推理是我的分析，并得到笔记 D.3 的测量支持]

**为什么 `tm = tm_next` 只在这里。** 当前 tile 的 `a_row`/`b_col` 是在循环体顶部由 `tm/tn` 计算出来的，在加载进行期间不得改变；取回的 tile 在循环体末尾才成为当前的，与 CUTLASS 中一样（`work_tile_info = next_work_tile_info` 位于 `:670`）。循环在某个响应*未被取消*（`valid == false`）时结束：网格中任何地方都不再有未启动的 cluster（第 7 章）。

### 8.6.8 尾段：八次阻塞等待，parity 1（第 797-806 行）

```
  #pragma unroll
  for (int i = 0; i < kStages; ++i) {
    EXPLICIT_WAIT(empty_base + 8u * st.prod_idx, st.prod_phase, TAIL_MAINLOOP, tile, uint32_t(i));
    ...
    advance<kStages>(st.prod_idx, st.prod_phase);
  }
```

这就是 `load_tail` -> `PipelineTmaAsync::producer_tail`（`sm100_mma_warpspecialized.hpp:633-640`，`sm90_pipeline.hpp:448-454`：`for (count < Stages) { empty_barrier_ptr_[state.index()].wait(state.phase()); ++state; }`）。它从循环停下的地方**延续 producer 的状态**（笔记第 7.2 节告诫不要把同一个 parity 应用到全部八个屏障上）。

**证明：对任意 tile 数，每个尾段步骤都等待 parity 1。** 设 `n_tiles` 为本 CTA 处理过的 tile 数（在被追踪的运行中为 32 或 31，带探针的 cluster 为 19，笔记 D.3；之所以写成全名，是因为本讲义的 `T` 是第 9-10 章的 TMEM 基址）。在 `n_tiles` 个 tile 之后，producer 已经走了 `N = 128 n_tiles` 步。尾段的第 `i` 步（`i = 0..7`）是状态的第 `N + i` 步，以 parity `1 ^ (((N + i) / 8) & 1)` 在 `empty[(N + i) % 8]` 上等待。由于 `128 n_tiles % 8 = 0`，对于 `i < 8` 有 `(N + i) % 8 = i` 且 `(N + i) / 8 = 16 n_tiles`；`16 n_tiles` 对每个 `n_tiles` 都是偶数，所以 parity 是 `1 ^ 0 = 1`，槽位是 `i`：尾段依次在 `empty[0..7]` 上等待，每次都以 parity 1。[推导] 实测：被追踪的运行中有 1518 条 `K_TAIL` 记录，每个 CTA 8 条 `TAIL_MAINLOOP`，在全部 132 个 CTA 的每一个中都带 `idx 0..7` 与 parity 1（"every mainloop tail step waited parity 1 (`k_tail_mainloop_parity` PASS)"，笔记 G.1）。

**在那一刻 parity 1 意味着什么。** Stage `i` 被使用了 `16 n_tiles` 次（每个 tile 16 次：128 个 k-tile 分布在 8 个槽位上），每次使用完成 `empty[i]` 的一个 phase（两次 commit 到达）。该屏障处于 phase `16 n_tiles`；最近完成的 phase 是 `16 n_tiles - 1`，为奇数，parity 1。因此 `try_wait.parity(1)` 恰好在 stage `i` 的*最后一次*使用已被两个 pair leader 释放时返回 true——即最后一个 tile 的 k-tile `120 + i` 的那些 commit。在某个 stage 从未被使用的假想 shape 中，它的屏障仍处于 phase 0，parity 1 立即通过；CUTLASS 自己的注释："if the stage was never used then would just be acquired since the phase was still inverted"（`sm100_mma_warpspecialized.hpp:634-639`）。硬编码 parity 1 对这个 shape 是正确的，而在一般情况下是错误的：例如每个 tile 72 个 k-tile 会给出 `9 n_tiles` 趟，在奇数个 tile 之后为奇数，于是尾段那时就必须等待 parity 0。继续沿用状态总是正确的。

**为什么存在尾段。** kernel 退出时没有 cluster 范围的屏障（笔记第 7.10 节）。没有它，两件事会出错：

1. 每个 pair leader 的最后一条 `tcgen05.commit` 把到达（掩码 `0xF`）multicast 到*这个* CTA 的 `empty[]` 中；如果这个 CTA 已经退出，一次远程的 `mbarrier.arrive` 就会指向已释放的共享内存（未定义；笔记 B.6，`mbarrier.*.shared::cluster` 规则）。
2. 更糟的是，本 CTA 的 pair leader 的 2SM MMA 直接读取本 CTA 的 smem stage。（*另一个* pair 的 leader 不从我们的 smem 读取任何东西；它读到的属于我们的内容，是我们的 producer 写进其同胞槽位中的 A 半块。）在 `empty[0..7]` 上等待——它们只有在*两个* leader 的 MMA 都读完每个 stage 之后才被释放——保证了没有人仍在读取本 CTA 即将放弃的共享内存。peer 的 warp 0 在这个环上没有自己的尾段（它从未消费过这个环），所以 warp 2 的八次等待是这个环上唯一的守卫，尽管并不是唯一让 peer 停留到其 leader 的最后一条 MMA 之后的守卫（peer 的 epilogue 在最后一个 tile 的 `acc_full` 上等待，explicit.cu:1030-1034，随后是 dealloc 握手，守卫 (4)）。[源自源码；笔记第 7.10 节 "guard (1)"]

**为什么这里用 `#pragma unroll`（explicit.cu:798），而两个 k-tile 循环都是 `#pragma unroll 1`。** 尾段镜像了 `producer_tail` 在编译期界限上的 `for (count < Stages)`，nvcc 在 baseline 中同样把它展开（"`producer_tail` = 8 unrolled waits"，笔记 D.2），而笔记 E.6.6 把这一形态定为契约的一部分（"tails as unrolled `for` over compile-time `Stages`"）。尾段每个 kernel 只运行一次，所以它的代码大小无关紧要，而展开去掉了计数器和回跳分支。剩下的是八次等待，每次之后跟着 `advance` 的状态更新。在 SASS 中，这是八次等待，其间夹着 `UIADD3 +1; UISETP.NE.U32 ..., 8; USEL; LOP3 ... 0x3c` 序列，最后一次之后跟着该 warp 的 `EXIT`。Baseline：尾段 `0x2660-0x2a60`，最后一次等待带有内联重试循环，`EXIT` 位于 `0x2a90`（笔记 D.2）。Explicit kernel：warp 2 的 `EXIT` 位于 `0x1980`（笔记 G.3）；笔记没有记录 explicit 尾段的起始地址。

---

## 8.7 算例与计数

地址是 32 位的 `shared::cta` 值，其中 `smem_base = 0x400 + (r << 24)`；`OFF_SMEM_A = 33792 = 0x8400`，`OFF_SMEM_B = 164864 = 0x28400`，stage stride 为 `0x4000`（A）与 `0x2000`（B），A 半块偏移为 `0x2000 y`。

### 8.7.1 示例 A：横幅注释里的示例，重新计算，以及它周围的整个 cluster

**公式核验。** 把横幅注释的输入——rank 1（`smem_base = 0x01000400`，`x = 1`，`y = 0`）、tile `(5, 9)`、`q = 7`——代入代码：`s = 7`，`c0 = 448`，`a_row = 128 x 5 + 0 = 640`，`b_col = 128 x 9 + 64 = 1216`；`a_dst_base = 0x01000400 + 0x8400 + 0 = 0x01008800`，A 目的地址 `0x01008800 + 7 x 0x4000 = 0x01024800`，multicast `0xA` -> rank 3 中的同一偏移，`0x03024800`；`b_dst_base = 0x01028800`，B 目的地址 `0x01028800 + 7 x 0x2000 = 0x01036800`；屏障 `(0x01000400 & 0xFEFFFFFF) + 56 = 0x00000438` = rank 0 的 `full[7]`。rank 2 的 A 半块（`a_dst_base = 0x0200A800`）去往它自身中的 `0x0200A800 + 0x1C000 = 0x02026800` 以及 rank 0 中的 `0x00026800`。横幅注释的全部五个数字（explicit.cu:718-719）都被复现。

**关于这些输入的一个注意事项。** 根据第 8.5.4 节，`y = 0` 的 CTA 只接收偶数的 `tn`（`tn = Tn + y`，其中 `Tn` 为偶数），所以 tile `(5, 9)` 在这次运行中绝不可能属于 rank 1；它是 cluster tile `(Tm, Tn) = (4, 8)` 中 rank 3 的 tile（对它而言，`a_row` 将是 `704`，而不是 `640`）。横幅注释里的那一行是*地址公式*在任意输入上的一次正确演练，而不是一个可达的状态；第 8.7.2 节的实测记录才是。[由映射公式推导]

**包含 (5, 9) 的可达 cluster：** `(Tm, Tn) = (4, 8)`，各 tile 为 rank 0 `(4, 8)`、rank 1 `(5, 8)`、rank 2 `(4, 9)`、rank 3 `(5, 9)`；`q = 7`：`s = 7`，tile 内 parity `1 ^ ((7 >> 3) & 1) = 1`，`c0 = 448`。目的地址与屏障不依赖于 tile：

| rank | `(tm, tn)` | `a_row` | A 目的地址（`a_dst_base + 7 x 0x4000`） | A 落在 | `b_col` | B 目的地址（`b_dst_base + 7 x 0x2000`） | 屏障操作数（`full_masked_base + 56`） |
|---|---|---|---|---|---|---|---|
| 0 | (4, 8) | 512 | `0x00008800 + 0x1C000 = 0x00024800` | rank 0、2 | 1024 | `0x00028800 + 0xE000 = 0x00036800` | `0x00000438`（rank 0 的 `full[7]`） |
| 1 | (5, 8) | 640 | `0x01008800 + 0x1C000 = 0x01024800` | rank 1、3 | 1088 | `0x01028800 + 0xE000 = 0x01036800` | `0x00000438`（rank 0 的 `full[7]`） |
| 2 | (4, 9) | 576 | `0x0200A800 + 0x1C000 = 0x02026800` | rank 0、2 | 1152 | `0x02028800 + 0xE000 = 0x02036800` | `0x02000438`（rank 2 的 `full[7]`） |
| 3 | (5, 9) | 704 | `0x0300A800 + 0x1C000 = 0x03026800` | rank 1、3 | 1216 | `0x03028800 + 0xE000 = 0x03036800` | `0x02000438`（rank 2 的 `full[7]`） |

在该 stage 落地之后，rank 0 的 stage 7 在 `0x24800` 处持有 A 的行 `[512, 576)`（它自己的 box），在 `0x26800` 处持有 `[576, 640)`（rank 2 的 box）= `tm = 4` 的那 128 行；rank 1 持有行 `[640, 768)` = `tm = 5`。Pair `{0, 1}` 的 MMA 读取 A 的行 `[512, 768)` 与 B 的列 `[1024, 1152)`（rank 0 的 `[1024, 1088)` + rank 1 的 `[1088, 1152)`）——即两个 CTA tile `(4, 8)` 与 `(5, 8)`；pair `{2, 3}` 读取同样的 A 的行以及 B 的列 `[1152, 1280)`——即 tile `(4, 9)` 与 `(5, 9)`。

### 8.7.2 示例 B：实测的第一个 cluster（启动 tile，`q = 0`）

位于 `blockIdx = (0..1, 0..1)` 的启动 cluster 具有 `(Tm, Tn) = (0, 0)`：rank 0 为 `(0, 0)`，rank 1 为 `(1, 0)`，rank 2 为 `(0, 1)`，rank 3 为 `(1, 1)`。`q = 0`：`s = 0`，`c0 = 0`。

| rank | A 坐标 -> 目的地址 | B 坐标 -> 目的地址 |
|---|---|---|
| 0 | `(0, 0, 0) -> 0x00008800` | `(0, 0, 0) -> 0x00028800` |
| 1 | `(0, 128, 0) -> 0x01008800` | `(0, 64, 0) -> 0x01028800` |
| 2 | `(0, 64, 0) -> 0x0200A800` | `(0, 128, 0) -> 0x02028800` |
| 3 | `(0, 192, 0) -> 0x0300A800` | `(0, 192, 0) -> 0x03028800` |

[实测] 这些正是笔记 D.1 D8 的 K4a 记录（"rank 0 A (0,0,0) -> 0x8800, B (0,0,0) -> 0x28800, barrier 0x400; rank 1 A (0,128,0) -> 0x1008800, B (0,64,0) -> 0x1028800; rank 2 A (0,64,0) -> 0x200a800, B (0,128,0) -> 0x2028800; rank 3 A (0,192,0) -> 0x300a800, B (0,192,0) -> 0x3028800; k advances by 64 per record pair with destination strides 0x4000/0x2000 and barrier +8"），对 CUTLASS kernel 和 explicit kernel 都是如此（笔记 G.1：K4a 多重集完全相同）。全部四个 rank 的每一个坐标和目的地址都在 128 条记录上与第 8.5.1 节的 per-rank 行和列以及第 8.7 节开头的地址公式相符（笔记 D.1 D8："all 128 records match B.4"）；第 8.5.3 节台账中依赖掩码的“落在”列和“计入”列不在 K4a 记录之内，它们依据的是对 SASS 的解读（第 8.5.3 节，笔记 D.6）。

### 8.7.3 示例 C：一个 CLC tile，`q = 77`

第 5 节的示例响应 `(x0, y0) = (10, 6)`（两者皆为偶数）在 CTA `(x, y) = (1, 0)` 中被消费，得到 `(tm, tn) = (2 (6 >> 1) + (10 & 1) + 1, 2 (10 >> 1) + (6 & 1) + 0) = (7, 10)` (explicit.cu:672)。该 cluster：rank 0 为 `(6, 10)`，rank 1 为 `(7, 10)`，rank 2 为 `(6, 11)`，rank 3 为 `(7, 11)`；`tm % 2 == x` 在每一行都成立。

`q = 77`：`s = 77 % 8 = 5`，tile 内 parity 为 `1 ^ ((77 >> 3) & 1) = 1 ^ (9 & 1) = 0`（该 tile 中环的第十趟遍历，`77 >> 3 = 9`：producer 等待本 tile 中槽位 5 的第九次释放，即其 `q = 69` 那次使用的释放），`c0 = 64 x 77 = 4928`。

| rank | `a_row` | A 目的地址（`+ 5 x 0x4000 = 0x14000`） | `b_col` | B 目的地址（`+ 5 x 0x2000 = 0xA000`） | 屏障 |
|---|---|---|---|---|---|
| 0 | 768 | `0x0001C800`（也写入 rank 2 的 `0x0201C800`） | 1280 | `0x00032800` | `0x00000428` |
| 1 | 896 | `0x0101C800`（也写入 rank 3：`0x0301C800`） | 1344 | `0x01032800` | `0x00000428` |
| 2 | 832 | `0x0201E800`（也写入 rank 0：`0x0001E800`） | 1408 | `0x02032800` | `0x02000428` |
| 3 | 960 | `0x0301E800`（也写入 rank 1：`0x0101E800`） | 1472 | `0x03032800` | `0x02000428` |

rank 0 位于 `0x428` 的 `full[5]` 收到：来自 rank 0 和 2 的 A（落在 rank 0），来自 rank 1 和 3 的 A（落在 rank 1，由 peer 位重定向），来自 rank 0 的 B，来自 rank 1 的 B = 49152 字节。

### 8.7.4 per-tile 与 per-stage 计数

| 量 | 每 CTA 每 tile | 推导 |
|---|---|---|
| 发射的 TMA 指令 | 256 | 每 k-tile 2 条（一条 A multicast，一条 B）x 128 |
| 本 CTA 的 TMA 发出的字节数 | 2 MiB | 256 x 8192 |
| 写入本 CTA smem 的字节数 | 3 MiB | 128 x (16384 A + 8192 B)；多出的 1 MiB 是同胞的 multicast 半块 |
| 对 `empty[]` 的 `try_wait` peek | 130 | `(8 + 1) + (120 + 1)`；128 次被消费，2 次被丢弃 |
| 对 `empty[]` 的阻塞等待 | 0..128 | 每一次返回 0 的 peek 对应一次 |
| `arrive.expect_tx` 49152 | pair leader 上 128 次，peer 上 0 次 | 每 k-tile 一次，由 `is_leader_lane` 决定 |
| `elect.sync` | PTX 中 128 条，SASS 中 0 条 | 折叠进 uniform `UTMALDG`（笔记 D.2，D.6 第 6 项；explicit kernel 的 SASS 在 producer 循环中没有 `ELECT`，笔记 G.3，第 6.2.2 节） |
| throttle wait + arrive | 1 + 32 次到达，仅 CTA 0 | 第 8.6.4 节 |
| `__syncwarp` | 1（无 SASS） | 第 8.6.6 节 |
| CLC 消费 | 1 | 1 次 wait，1 条 `ld.shared.b128`，解码，fence，32 次远程 arrive |

| 量 | 每 cluster 每 k-tile | 每 pair 每 k-tile |
|---|---|---|
| TMA box | 8 (4 A + 4 B) | 计入：每个 leader 屏障 6 个 |
| 从 L2 读取的字节数 | 65536 | — |
| 写入 smem 的字节数 | 98304 | 49152 |

[源自源码] 每 tile 每 pair，leader 的八个 `full[]` barrier 完成 128 个 phase，每个 49152 字节 = 6 MiB。

---

## 8.8 TMA 单元在写入时做什么：SWIZZLE_128B

producer 传递的是*线性*的目的地址（`a_dst_base + 16384 s`、`b_dst_base + 8192 s`），从不计算 swizzled 偏移，然而字节并不是线性落地的：两个 tensor map 都以 `CU_TENSOR_MAP_SWIZZLE_128B` 编码（第 5 章；笔记第 5.3 节，实测 D3），而 UMMA 描述符声明 `layout_type = 2` = SWIZZLE_128B (`include/cute/arch/mma_sm100_desc.hpp:114`; explicit.cu:171 `kSmemDescHi`)。写入方与读取方就同一个置换达成一致，因此 kernel 可以对它毫不知情（explicit.cu:203："applied by the TMA unit on write and by the UMMA unit on read (SWIZZLE_128B); the kernel never computes it"）。

### 8.8.1 `Swizzle<3,4,3>`

CuTe 的 `Swizzle<BBits, MBase, SShift>`（`include/cute/swizzle.hpp:54-68`）把一个偏移中从第 `MBase` 位起的 `BBits` 个位（`zzz_msk`），与再往上 `SShift` 位处的那 `BBits` 个位（`yyy_msk`）做 XOR：`apply(o) = o ^ ((o & yyy_msk) >> SShift)`。对于 `<3,4,3>`：`zzz_msk = 0b111 << 4 = 0x70`（位 [4,7)），`yyy_msk = 0b111 << 7 = 0x380`（位 [7,10)）；explicit.cu:204 断言的正是 `yyy_msk == 0x380` 和 `num_shft == 3`。在字节地址上：

```
 physical = pre ^ (((pre >> 7) & 7) << 4)
```

位 [4,7) 是 16 字节 chunk 在 128 字节行内的索引；位 [7,10) 是行索引模 8。于是：**在每个 1024 字节的原子（8 行 x 128 B）内部，第 `r` 行的 chunk `c` 存放在 chunk 位置 `c ^ (r % 8)`。** CUDA Programming Guide 对 128B 模式把同一条规则表述为 `smem[y][((y + offset) % 8) ^ x]`，其中 `x` 为 16 字节单元，`y` 为行，且 `offset = (smem_ptr / 128) % 8`（其表 24，"Swizzle Pattern Pointer Offset Formula and Index Relation"）。`offset` 是缓冲区*起始地址*在原子内所处的行位置；它被加到行索引上，是因为硬件按绝对地址位做 swizzle，而不是按相对于缓冲区起点的偏移。对于 1024 字节对齐的基址，`offset = 0`，Guide 的公式就坍缩为上面的 `chunk ^ (row % 8)`；本 kernel 中的每个基址都是这样的基址（第 8.8.3 节）。该模式每 8 行 = 1024 字节重复一次。

### 8.8.2 一个 stage，画出来

A 元素 `(m, k)` 在 stage `s` 中的朴素（swizzle 前）字节偏移：`16384 s + 128 m + 2 k`（K-major：每行 `m` 为 64 个 fp16 = 128 B；explicit.cu:124-125 处的 CuTe layout，以元素计为 `((_128,_16),_1,_4,(_1,_8)) : ((_64,_1),_0,_16,(_0,_8192))`，说的是同一件事：行 stride 64 个元素，k-block stride 16 个元素 = 32 B，stage stride 8192 个元素）。对于 B 元素 `(n, k)`：`8192 s + 128 n + 2 k`。由于 `2 k < 128`，朴素偏移的位 [7,10) 就是 `m % 8`，因此物理字节为 `16384 s + 128 m + ((2 k) ^ ((m % 8) << 4))`；第 `m` 行的 16 字节 chunk `k >> 3` 位于 chunk `(k >> 3) ^ (m % 8)`。

```
 one 1024-byte atom = rows m..m+7 of a stage (128 B each); entries = logical chunk (8 fp16 = 16 B) stored at that physical position

 physical chunk ->    0   1   2   3   4   5   6   7        <- the 8 x 16 B = 128 B of one row
 row m%8 = 0        [ 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 ]      k 0-7, 8-15, ..., 56-63 in order
 row m%8 = 1        [ 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 ]
 row m%8 = 2        [ 2 | 3 | 0 | 1 | 6 | 7 | 4 | 5 ]
 row m%8 = 3        [ 3 | 2 | 1 | 0 | 7 | 6 | 5 | 4 ]
 row m%8 = 4        [ 4 | 5 | 6 | 7 | 0 | 1 | 2 | 3 ]
 row m%8 = 5        [ 5 | 4 | 7 | 6 | 1 | 0 | 3 | 2 ]
 row m%8 = 6        [ 6 | 7 | 4 | 5 | 2 | 3 | 0 | 1 ]
 row m%8 = 7        [ 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 ]

 A stage s = 16 such atoms (rows 0-7, 8-15, ..., 120-127), 16384 B, at smem_base + 0x8400 + 0x4000 s
   the y = 0 half (rows 0-63) = atoms 0-7 at +0, the y = 1 half (rows 64-127) = atoms 8-15 at +8192: each multicast box is 8 whole atoms
 B stage s = 8 atoms (rows 0-63), 8192 B, at smem_base + 0x28400 + 0x2000 s
 k-block kb of a k-tile = logical chunks 2 kb and 2 kb + 1 of every row (32 B = 16 fp16): the UMMA descriptor's start address advances by 32 B per kb
```

为什么是这个模式？共享内存按 32 x 4 B = 128 B 的宽度划分 bank，因此一行覆盖全部 bank，而一*列* chunk（8 个连续行中相同的 `k` 范围）会八次命中同样的四个 bank。把 chunk 索引与行索引做 XOR，就把这八次访问分散到全部八个 chunk 位置上，也就是全部 32 个 bank——即 CUDA Programming Guide 所说的 "each matrix element is from a different bank for both the rows and columns, and so without any bank conflicts"。UMMA 每次读取 8 行 x 16 B 的核心矩阵；有了 swizzle，它就能无冲突地读取它们 [推导；UMMA 的内部访问模式是硬件自己的事]。

### 8.8.3 为什么要 1024 字节对齐，以及为什么 kernel 从不计算 swizzle

两个单元都把 XOR 施加在**绝对的共享内存地址位**上，而不是施加在 box 内或 tile 内的偏移上。它们使用的行索引是地址位 [7,10)。要让它等于 `m % 8`，stage 基址在位 [7,10) 上必须贡献全零：`base % 1024 == 0`。kernel 以三重方式保证了这一点。`OFF_SMEM_A = 33792 = 33 x 1024` 和 `OFF_SMEM_B = 164864 = 161 x 1024`（explicit.cu:162，`static_assert(OFF_SMEM_A % 1024 == 0 && OFF_SMEM_B % 1024 == 0, "SW128 stage bases need 1 KiB alignment (Section 7.5)")`）。stage stride 16384 和 8192 以及半块偏移 8192 都是 1024 的倍数。动态基址的低 24 位是 `0x400 = 1 x 1024`（实测，笔记 D.1 D5）。kernel 没有声明任何静态 `__shared__`（explicit.cu 第 10 节横幅注释），所以动态缓冲区紧跟在系统保留的 1 KiB 共享内存之后开始。这些当中哪些是硬件强制的，哪些是设计选择？CUDA Programming Guide 对 swizzled TMA 目的地的硬性要求是 **128 字节**对齐（"When using TMA, the shared memory is required to be aligned to 128 bytes"；其表 25，针对 Compute Capability 9 所述，为每种 swizzle 模式列出的 "Shared memory alignment" 都是 128 字节）；1024 只是一个建议（"For simplicity shared memory should be aligned according to the number of bytes after which the swizzle pattern repeats"）。一个 128 字节对齐但不是 1024 字节对齐的目的地对 TMA 来说是合法的：模式只是被平移了 `offset = (addr / 128) % 8` 行（第 8.8.1 节）。使 1024 成为本 kernel 的*硬性*要求的，是读取方。UMMA 描述符带有一个 3 位的 `base_offset` 字段（`mma_sm100_desc.hpp:112-113`），正是为起始地址在位 [7,10) 上非零的 swizzle 模式而设（PTX ISA 9.7.18.4.1：当模式从 1024 字节边界开始时 base offset 为 0，否则为 `(pattern start addr >> 0x7) & 0x7`）。CUTLASS 把它留为 0（`constexpr uint8_t base_offset = 0; desc.base_offset_ = base_offset;`，`mma_sm100_desc.hpp:757-758`），而 explicit kernel 的 `kSmemDescHi = 0x40004040`（explicit.cu:171）在那些位上也是零（第 9 章）。在 `base_offset = 0` 下，UMMA 假定 stage 的 `offset = 0`，所以交给 TMA 一侧的基址也必须是 `offset = 0` 的基址：即 1024 字节对齐。依赖链：TMA 要求 128 B；带 `base_offset = 0` 的 UMMA 要求 1024 B；kernel 像 CUTLASS 一样选择了 `base_offset = 0`，因此在 `OFF_SMEM_A`、`OFF_SMEM_B`、stage stride 和动态基址上付出了 1024 字节对齐的代价 [源自源码；Guide 的两条陈述引自其 async-copies 页面]。

因为这个置换是地址的函数，UMMA 可以从原子*内部*开始一个 k-block：描述符的起始地址 `+ 32 kb` 的位 [4,7) = `2 kb`，硬件把它与所取每一行的行位做 XOR，从而落在正确的 chunk 上。这就是为什么 producer 可以用一个 box 写入 64 宽的 k-tile，而 MMA 以四个 16 宽的 k-block 消费它们，没有任何人需要重排数据。kernel 中唯一的 swizzle 算术是 epilogue 对*它自己的* `st.shared` 存储所做的 `Swizzle<2,5,2>`（第 10 章），那里的写入方是软件；而这里写入方和读取方都是硬件。

### 8.8.4 为什么 k-tile 是 64

对于 `CU_TENSOR_MAP_SWIZZLE_128B`，box 的内维最多只能是 128 字节。CUDA Programming Guide 的表 25 对 128B 模式给出 "Shared box's inner dimension <= 128 bytes"，而驱动程序会强制执行它：`cuTensorMapEncodeTiled` 的 CUDA Driver API 参考文档写明 "CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128"，并把 `CUDA_ERROR_INVALID_VALUE` 列在其返回值之中。CuTe 对这条规则*不做*任何检查：它的编码器（`copy_traits_sm90_tma.hpp:1004-1013`）只断言了通用的 box 限制 `1 <= boxDim[i] <= 256`（Driver API 的 "must be non-zero and less than or equal to 256"），以及在 `:1015-1024` 处断言了 `1..8` 的元素 stride 限制。128 字节的 fp16 是 `128 / 2 = 64` 个元素，所以一个 TMA box 最多覆盖 64 个 k 值，而 UMMA 的 SW128 K-major 规范 layout 恰好是 8 行 x 64 个 fp16。要追踪的名字是 `UMMA::Layout_K_SW128_Atom<ElementType>`，`sm100_smem_selector`（`include/cutlass/gemm/collective/builders/sm100_common.inl:82`，分支在 `:116-117`）选中它，是因为 `BLK_K = 64` 是该原子 K 范围的倍数。这个名字是命名空间 `SM90::GMMA` 中那个定义的别名（`using cute::GMMA::Layout_K_SW128_Atom;`，`include/cute/atom/mma_traits_sm100.hpp:71`），所以搜索 UMMA 名字的读者必须经由这个别名跳转过去。该定义以位计为 `ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>`（`include/cute/atom/mma_traits_sm90_gmma.hpp:84`），并在 `:104` 处被赋予类型。因此，64 的 k-tile 是一个 swizzled TMA box 和一个 SW128 UMMA 原子二者在 K 上都能跨越的最宽单位，也是一个 `mbarrier` phase 所核算的单位；四条 K = 16 的 `tcgen05.mma` 消费它（设计算术见第 3 章，描述符增量见第 9 章）。64 x 64 的 box 满足 256 元素的限制且绰绰有余。

---

## 8.9 warp 2 的占用，以及为什么它的循环长度无关紧要

循环在每个 k-tile 中保持活跃的东西：槽位索引和 phase（`st.prod_idx`、`st.prod_phase`）、token、`n`、`st.q`、第 8.6.2 节的五个外提不变量（两个目的基址、带掩码的 barrier 基址、empty 基址、掩码）、缓存提示（64 位）、两个 tensor-map 地址（常量 bank 地址，`&params.tma_a/b`）、`a_row`、`b_col`，以及两个谓词（`is_leader_lane`、全新的选举）。没有任何东西在运行时被索引，也没有任何东西驻留在局部内存中：整个 kernel 报告 `REG 26, LOCAL 0, STACK 0`（笔记 G.3，实测）。[我的分析：这个 26 是由 epilogue warp 决定的，它们同时让 16 个 `tcgen05.ld` 结果外加存储地址保持活跃（`alpha` 和 `beta` 位于 uniform 寄存器中，第 11.4.1 节），这是 kernel 中最大的每线程 working set；而不是由本 warp 决定的，它的状态几乎完全是 uniform 的。]

producer 循环中相对于 baseline 仅 explicit 一侧才有的指令（按笔记 G.3 的计数是八条；第 12.5.2 节的匹配计为九条，加上了等待块里的 `ULEA UR4`），是对四个循环不变量的五次重新物化（49152 立即数、`tma_b` 参数地址对、multicast 掩码插入和 z 坐标）、warp 投票 `VOTEU.ALL UP1, P0`、带谓词的 arrive 地址 `@!UP1 ULEA UR10` 以及带掩码的 `full[s]` `ULEA UR17`；除了按 lane 加谓词的 `MOV R5` 和 `VOTEU.ALL` 之外，其余都是 uniform 数据通路指令（术语见第 8.4.2 节）。发射顺序 wait -> `expect_tx` -> peek -> TMA A -> TMA B 在 explicit 循环和 baseline 循环中完全相同。

k-tile 预算约为 426 个周期（来自 CUTLASS baseline 的 0.888 ms，即第 3 章，第 3.9.2 节的工作数值；explicit kernel 自己的中位数给出 427，第 9 章，第 9.8 节）。在这个预算之内，tensor core 执行四条 256 x 128 x 16 的 MMA（峰值下 256 个周期，第 3.9.2 节），TMA/L2 交付每 CTA 24576 字节；是哪种资源决定了这个 426，是第 3 章，第 3.9 节的开放问题（共享内存带宽是首要假设）。producer warp 可以领先 MMA 八个 k-tile（约 1.7 us），而在环的第一轮遍历之后它大部分时间是在等待槽位而不是在工作（我的分析：相对该预算，每 k-tile 37 条循环指令；等待占比未经实测）。隐藏 TMA 往返的是环的深度，而不是循环长度：8 个 stage，即 epilogue 和屏障 carveout 之后剩下的全部共享内存（第 3 章，第 3.4.1 节）。

---

## 8.10 把这些带到其他 kernel

- **每个共享资源一个 barrier，归需要整个资源的 consumer 所有。** pair leader 的 `full[s]` 对 pair 的字节求和，因为 leader 的 MMA 是 pair 的 stage 的唯一 consumer。当你设计一个多 CTA 操作数时，问一问 "who issues the instruction that needs all of it?"，然后把每个 `complete_tx` 都路由到那里；于是 `expect_tx = sum of everything that lands in the consumer's domain`，而到达计数是不同*到达者*的数量，不是 box 的数量。
- **在 CTA 共享数据且没有别的东西把它们联结在一起的地方用 multicast；在一条指令已经横跨这些 CTA 的地方做拆分。** A 跨 `y`（共享 `tm`，没有其他联系），B 跨 pair（由 `cta_group::2` 联结）。
- **先 peek 再等待，并让状态跨 tile 延续、一直延续到尾段。** parity 簿记就是环的全部正确性；要从一个持续累计的计数器计算它，绝不要从 tile shape 计算。
- **让 producer 对硬件对称施加的 layout 变换毫不知情。** TMA 以 swizzled 方式写，UMMA 以 swizzled 方式读；你的工作是对齐（1024 B）和 box shape（内维 <= 128 B），而不是 XOR。
- **把慢的消费排在快的发射之后。** 任何延迟可与一次流水线重新填满相比的 per-tile 握手（这里是 CLC 响应），都应放在让流水线保持忙碌的工作之后。
- **在 cluster kernel 中，尾段不是可选项。** 每一个远程 CTA 仍可能在其上 arrive 的 barrier，以及每一个远程指令仍可能读取的 smem 区域，都需要在 `EXIT` 之前等待一次。

---

# 第 9 章 引擎：mma_role（Warp 0）

## 9.1 引擎位于何处

第 8 章结束时，mainloop producer（warp 2）正在共享内存中填充一个由 A 与 B tile 组成的 8-stage 环，并以每个 stage 49152 个事务字节向 pair leader 的 `full[s]` 屏障发出信号。第 10 章将把完成的累加器从 TMEM 中取出。二者之间是 warp 0：唯一向 tensor core 发射 MMA 的 warp。它的工作说起来简单，要做对却极为严苛：

1. 每个 CTA 一次：分配 pair 的整个 TMEM，把基址发布给 epilogue warp，构建两个描述符基址；
2. 每个 tile 一次，仅 leader：等待直到某个累加器 stage 空闲，然后对 128 个 k-tile 中的每一个，等待该 stage 变满，发射四条 2SM MMA，把该 stage 释放给全部四个 producer；最后把该累加器 stage 交给两个 CTA 的 epilogue；
3. 每个 tile 一次，两个 CTA 都执行：消费一个 CLC 响应并推进累加器状态；
4. 最后：释放分配许可，排空累加器流水线，与 peer 握手，解除分配。

整个角色位于 explicit.cu:871-975。我们按这一顺序阅读它。

---

## 9.2 逐句解读横幅注释 (explicit.cu:871-880)

```
871:// 8. [PROD] tcgen05  MMA role: warp 0 of every CTA (Section 7.6, B.3.2; kernel :726-805; CollectiveMma::mma, sm100_mma_warpspecialized.hpp:651-712)
872://   Both CTAs of the pair: tcgen05.alloc 512 TMEM columns (cta_group::2: one allocation for the pair, the result written to smem + 384 of
873://   both CTAs), bar.arrive 6,160 to release the epilogue warps, and the accumulator producer state; at the end relinquish + dealloc.
874://   The PAIR LEADER only, per tile: wait acc_empty[st] (count 256 = 2 x 128 epilogue threads released stage st), then per k-tile:
875://     wait full[s] (count 1 + 49152 tx bytes, section 6) -> 4 x tcgen05.mma.cta_group::2 (one per 16-k block; the first of the tile with
876://     scale_c = 0 overwrites the accumulator, the rest accumulate) -> tcgen05.commit to empty[s] of all four CTAs (mask 0xF, count 2 per CTA)
877://   then tcgen05.commit to acc_full[st] of the pair (mask 0x3 << 2y) and advance the accumulator stage (both CTAs advance the state).
878://   TMEM address = (lane << 16) | column; tmem_c = T + 128 st with T = 0 measured: stage st is columns [128 st, 128 st + 128) of all 128 lanes.
879://   The MMA tile is 256 (M) x 128 (N): rows [0,128) come from the leader's A stage, [128,256) from the peer's; N from both B halves.
880:///////////////////////////////////////////////////////////////////////////////////////////////////
```

| 行 | 句子 | 含义以及出处 |
|---|---|---|
| 871 | `[PROD] tcgen05 MMA role: warp 0 of every CTA` | 图例标签（explicit.cu:26-36）：生产代码，硬件家族 tcgen05。*每个* CTA 的 warp 0 都运行这个函数，而不只是 leader 的；leader/peer 的划分在函数内部（第 915 行）。四处引用分别是笔记的协议节（7.6）、笔记的逐调用 trace（B.3.2），以及该函数所去抽象化的 CUTLASS kernel 分支（`sm100_gemm_tma_warpspecialized.hpp:726-805`）和 collective 的 `mma()`（`sm100_mma_warpspecialized.hpp:651-712`）。 |
| 872-873 | `tcgen05.alloc 512 TMEM columns (cta_group::2: one allocation for the pair, the result written to smem + 384 of both CTAs)` | 每个 SM 的 TMEM 有 512 列（`tmem_allocator_sm100.hpp:120`，`Sm100TmemCapacityColumns = 512`）；kernel 取走全部列，与 CUTLASS kernel 一样（kernel :728 分配 `Sm100TmemCapacityColumns`），而 4 个累加器 stage x 128 列恰好填满它们：512 / 128 = 4（第 9.3 节）。使用 `.cta_group::2` 时，两个 CTA 的 warp 0 一起执行一次分配，基址被写到两个 CTA 中同一个共享内存偏移处（384 = `OFF_TMEM_BASE_PTR`，explicit.cu:152）（第 9.4.1 节）。 |
| 873 | `bar.arrive 6,160 to release the epilogue warps` | 命名屏障 6，有 160 个参与线程：warp 0 的 32 个 lane 到达，128 个 epilogue 线程在第 10 章中对它执行 `bar.sync`（explicit.cu:997）。第 9.4.2 节推导这两个数。 |
| 873 | `and the accumulator producer state; at the end relinquish + dealloc` | 两个 CTA 都让 4-stage 累加器 producer 状态保持同步（第 946 行），并且两者都执行 `tcgen05.relinquish_alloc_permit` 与 `tcgen05.dealloc`（第 951、971 行）。 |
| 874 | `The PAIR LEADER only, per tile: wait acc_empty[st] (count 256 = 2 x 128 epilogue threads released stage st)` | 只有偶数 rank 发射 tensor-core 工作。在复用 TMEM stage `st` 之前，它必须知道*两个* CTA 的 epilogue 都已读完它：128 个线程 x 2 个 CTA 到达 leader 的 `acc_empty[st]`（第 10 章，explicit.cu:1039），因此计数为 256（explicit.cu:184）。 |
| 875 | `wait full[s] (count 1 + 49152 tx bytes, section 6)` | mainloop 的 full 屏障：来自 leader 的 producer lane 的一次 `arrive.expect_tx`，加上落入 pair 的六个 8192 字节 TMA box（第 8 章）。 |
| 875-876 | `4 x tcgen05.mma.cta_group::2 (one per 16-k block; the first of the tile with scale_c = 0 overwrites the accumulator, the rest accumulate)` | 一个 k-tile 在 K 上宽 64；一条 f16 MMA 消费 K = 16（一行的 256 位，`mma_traits_sm100.hpp:1175`），所以每个 k-tile 有 4 条 MMA（`kKBlocks = kTileK / kMmaK`，explicit.cu:106）。`scale_c` 是该指令的 enable-input-D 谓词（第 9.5.3 节）。 |
| 876 | `tcgen05.commit to empty[s] of all four CTAs (mask 0xF, count 2 per CTA)` | 对一个 smem stage 的释放必须到达 MMA 读取过其 smem 的每个 CTA，*以及*其 producer 写入过它的每个 CTA；第 9.5.3 节推导为什么这是全部四个，以及为什么每个 CTA 的 `empty[s]` 因此需要两次到达。 |
| 877 | `tcgen05.commit to acc_full[st] of the pair (mask 0x3 << 2y)` | 对 pair 的两个 CTA 而言，累加器在同一瞬间完成（一条指令写了两个半块），所以一次 multicast commit 操作就通知了两个 epilogue；另一个 pair 有它自己的累加器，不得被触碰（第 9.5.4 节）。 |
| 877 | `advance the accumulator stage (both CTAs advance the state)` | peer 的 warp 0 不发射任何 MMA，但完全像 leader 那样推进 `{accp_idx, accp_phase}`，因为源码就是这样做的（`:773` 在 leader 分支之外）。没有任何东西读取 peer 的状态；这一行是为与 PTX 保持一致而保留的（第 9.5.5 节）。 |
| 878 | `TMEM address = (lane << 16) \| column; tmem_c = T + 128 st with T = 0 measured` | TMEM 以一个 32 位字寻址，其高 16 位选择数据通路 lane（0..127），低 16 位选择列（0..511）（`cute/pointer.hpp:314-316`，笔记 6.3）。MMA 的 D 操作数指定 lane 0、列 `128 st`；该指令从那里起填满全部 128 个 lane（第 9.3 节）。 |
| 879 | `The MMA tile is 256 (M) x 128 (N): rows [0,128) come from the leader's A stage, [128,256) from the peer's; N from both B halves.` | 2SM 指令的操作数划分（第 9.3 节）：每个 CTA 在*相同的 CTA 相对地址*处提供 A 的 128 行和 B 的 64 列，并接收属于它自己的 A 行的那 128 行 D。 |

---

## 9.3 pair、2SM 指令与 TMEM 映射

### 9.3.1 一条 `tcgen05.mma.cta_group::2` 计算什么

CUTLASS 对本 kernel 的 MMA 背后的原子（W29 `umma_f16_2sm`，explicit.cu:517-529）的描述是 `MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS<...>>`（`cute/atom/mma_traits_sm100.hpp:1160-1219`）。它的线程与值 layout（已在源码中核实）是：

```
1177:  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
1178:  using ThrID   = Layout<_2>;
1179:  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
1180:                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
1181:  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
1182:                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
1183:  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
1184:                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
```

请用 `M = 256, N = 128, K = 16`（固定实例化，笔记第 4 节与 explicit.cu:105）来解读这些。`ThrID = 2` 表示有两个“线程”`v = 0, 1` 执行该原子。它们就是 pair 的两个 CTA，`v = x = r % 2`：`sm100_gemm_tma_warpspecialized.hpp:427-428` 写的是 `int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});` 和 `bool is_mma_leader_cta = cta_coord_v == 0;`，其中对于 `ThrID = Layout<_2>`，`size<0>(AtomThrID)` 求值为 2。CTA `v` 提供 A 的行 `[128 v, 128 v + 128)`（ALayout，线程 mode 上的 stride `M/2 = 128`）和 B 的列 `[64 v, 64 v + 64)`（BLayout，stride `N/2 = 64`），并接收 D 的行 `[128 v, 128 v + 128)`（CLayout，stride `M/2 = 128`）。

PTX ISA（`tcgen05.mma`）没有说明 A 与 B 如何在两块共享内存之间划分；它关于双 CTA 组织方式的图是张量内存数据通路 layout 图（9.7.18.10.5 "Data Path Layout Organization"，图 211 "Layout organization for M = 256"），这些图展示的是张量内存区域以及 `tcgen05.ld`/`tcgen05.st` 地址，而不是共享内存操作数。因此下面所用的划分是*源自源码*，而非来自 ISA：即上面读到的 `ALayout`/`BLayout`/`CLayout`（`mma_traits_sm100.hpp:1179-1184`）。

```
            one tcgen05.mma.cta_group::2 of the fixed run (M 256, N 128, K 16), issued by the LEADER's elected lane

   A operand (K-major, from smem stage s, k-block kb)        B operand (K-major, same s, kb)
   +------------------------------+                           +------------------+------------------+
   | rows   0..127  LEADER smem   |  <- A rows 128 tm_L ..    | cols 0..63       | cols 64..127     |
   |        (OFF_SMEM_A + 16384 s |                           | LEADER smem      | PEER smem        |
   |         + 32 kb)             |                           | (OFF_SMEM_B      | (same CTA-local  |
   +------------------------------+                           |  + 8192 s + 32kb)|  offset)         |
   | rows 128..255  PEER smem     |  <- A rows 128 tm_P ..    +------------------+------------------+
   |        (same CTA-local       |     with tm_P = tm_L + 1     = B columns 128 tn .. 128 tn + 127
   |         offset)              |
   +------------------------------+

   D operand (fp32, TMEM column 128 st, lane 0): the LEADER's TMEM receives rows 0..127 (its own 128 x 128 tile (tm_L, tn)),
                                                 the PEER's TMEM receives rows 128..255 (its own tile (tm_P, tn)), both in lanes 0..127.
```

每个 CTA 的 B stage 保存它的 64 列 `128 tn + 64 x .. +64`（第 8 章，explicit.cu:778），而指令的 N = 128 看到的是两个半块。Colfax 的 cluster 教程针对填充各 stage 的 multicast TMA 复制，陈述了两个 CTA 的共享内存 layout 必须一致这一要求（第 2 章，第 2.5.2 节）："using `cute::copy` for this TMA copy requires all CTAs in a cluster to have the same shared storage layout"。

为什么每条指令是 N = 128 而不是 256：CTA tile 是 128 x 128（`CtaShape_MNK (128, 128, 64)`，explicit.cu:132），而 pair 的 tile 是 256 x 128（`TileShape (256, 128, 64) per pair`）。设计上的理由（A 沿 `y` 的 multicast、TMEM 容量、epilogue 宽度）是第 3 章的主题。

### 9.3.2 TMEM 映射

2SM 原子的累加器片段类型是 `UMMA::tmem_frg_2sm<c_type>`（`mma_traits_sm100.hpp:1172`）。笔记 B.3.2 的步骤 3 把它追溯到每个 CTA 的 layout `((128,128),1,1,4):((65536,1),0,0,128)`：stage `st` 的元素 `(m, n)` 位于 `T + (m << 16) + 128 st + n`。代入 `T = 0`（在全部 264 条 K2 记录中实测，笔记 D.1 第 D6 行）：

```
   TMEM of ONE CTA (the leader's and the peer's look identical; each holds its own 128 x 128 tile per stage)

   column:  0        127 128       255 256       383 384       511
            +-----------+-----------+-----------+-----------+
   lane   0 | st = 0    | st = 1    | st = 2    | st = 3    |   lane m  <->  row m of this CTA's 128 x 128 tile
   lane   1 | tile t    | tile t+1  | tile t+2  | tile t+3  |              = global row 128 tm + m
     ...    | (t%4 == 0)| (t%4 == 1)| (t%4 == 2)| (t%4 == 3)|   column 128 st + n  <->  global column 128 tn + n
   lane 127 |           |           |           |           |
            +-----------+-----------+-----------+-----------+
   address of (m, n) in stage st:  (m << 16) | (128 st + n)        e.g. (lane 64, st 1, n 48) -> 0x0040_00B0

   MMA D operand per tile:  tmem_c = T + 128 st  (lane field 0; the instruction writes lanes 0..127)
   Epilogue warp w (Chapter 10) reads lanes [32 w, 32 w + 32) from  T + ((32 w) << 16) + 128 st + 16 e
```

CTA `v` 所拥有的指令 tile 的行 `[128 v, +128)`（第 2 章，第 2.5.1 节）就是*它自己的* CTA tile，所以每个 CTA 的 128 个 TMEM lane 一一映射到它的 128 个输出行，epilogue 不需要任何跨 CTA 的 TMEM 流量（第 10 章只读取它自己的 TMEM）。列的四路划分就是累加器流水线：`kAccStages = 4`（explicit.cu:108），因为 512 / 128 = 4 是 TMEM 对一个 128 列 fp32 tile 所能容纳的最多 stage 数，而四个 stage 让 tile `t` 的 epilogue 与 tile `t+1 .. t+3` 的 mainloop 重叠。

---

## 9.4 设置：分配、发布、描述符基址、掩码、状态 (explicit.cu:881-908)

```
881:__device__ __forceinline__ void mma_role(char* smem, uint32_t smem_base, uint32_t x, uint32_t y, uint32_t rank, bool is_leader,
882:                                         uint32_t peer, uint32_t lane_pred, int k_tiles, int tm, int tn) {
883:  (void)tm; (void)tn;   // the MMA warp needs no tile coordinates (the TMEM stage and the smem stages are tile-independent)
884:  // TMEM allocation: whole warp 0 of both CTAs of the pair, same dst offset (tmem_allocator_sm100.hpp:124-133); result lands in smem
885:  tmem_alloc_2sm(smem_base + OFF_TMEM_BASE_PTR, kTmemColumns);
886:  __syncwarp();                                                                                // (kernel :729)
887:  named_bar_arrive(kBarTmemAlloc, kTmemAllocBarThreads);                                       // bar.arrive 6, 160 (:730)
888:  uint32_t const T = *reinterpret_cast<uint32_t const*>(smem + OFF_TMEM_BASE_PTR);             // tmem_base_ptr (:731); measured 0, never assumed (TMEM address = lane << 16 | column)
```

签名（881-882）接收文件的地址空间规则所要求的共享内存的两种视图（explicit.cu:22-23）：`smem`（通用指针，用于 C++ 解引用）和 `smem_base`（32 位 `shared::cta` 地址，用于每个 asm 操作数）。`x, y, rank, is_leader, peer` 是在 prologue 中计算一次的 cluster 身份（explicit.cu:1104-1107 与 1173-1174；`is_leader = ((rank & 1u) == 0u)`，`peer = rank ^ 1u`）。`lane_pred` 是 prologue 的选举结果，这里只在 trace 开关下使用（893：生产分支中为 `(void)lane_pred; (void)rank;`）。`k_tiles = (params.k + 63) >> 6 = 128`，而 `tm, tn` 是启动 tile。

第 883 行陈述了一个把这个 warp 与 producer 和 epilogue 区分开的事实：**MMA warp 不需要 tile 坐标。** 它的输入是地址只取决于 `s` 和 `kb` 的 smem stage；它的输出是地址只取决于 `st` 的 TMEM stage。tile 位于矩阵中的何处是 producer 的事（TMA 坐标）和 epilogue 的事（存储坐标）。`(void)` 强制转换是说明性的而非必需的：`tm` 和 `tn` 在第 906 行仍被读取（`int tm_next = tm, tn_next = tn;`），所以并不存在未使用参数警告的问题。`tm_next/tn_next` 仍然经由 `clc_consume`（第 913 行）一路传递，因为这个共享的辅助函数会写它们，但这里对它们仅有的读取是第 947 行的死副本 `tm = tm_next; tn = tn_next;`。在 baseline 的 SASS 中，ptxas 相应地消除了这个 warp 中的 `get_first_ctaid` 解码（"in the MMA warp only the valid bit survives"，笔记 D.2，它描述的是 `sass_gemm.txt`）。同样的消除在 explicit kernel 中也成立，因为 `tm_next/tn_next` 在 `mma_role` 中是死的：它位于 `0x5070` 处的解码只保留 valid 位（`trace_out_v2/sass_explicit.txt`；第 6.8.3 节），笔记 G.3 没有单独记录这一点。

### 9.4.1 两个 CTA 的整个 warp 0 执行 `tcgen05.alloc` (885)

wrapper W26（explicit.cu:498-503）是 `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], num_columns;`，复制自 `cute::TMEM::Allocator2Sm::allocate`（`tmem_allocator_sm100.hpp:134-145`）。第 6 章，第 6.7.1 节分解了每一个限定符。对这个角色而言重要的是，为什么其中两个在*这里*是正确的：

- `.shared::cta.b32 [dst]`：结果（一个 32 位 TMEM 地址）被写入共享内存而不是寄存器，因为这个值必须到达其他 warp（epilogue），并且在使用 `.cta_group::2` 时还要到达另一个 CTA。
- `num_columns = kTmemColumns = 512`：512 就是 `Sm100TmemCapacityColumns`（`tmem_allocator_sm100.hpp:120`），而 kernel 恰好用掉这么多：4 个 stage x 128 列。

硬件实际做的事（在 baseline 的 SASS 上实测，笔记 D.2；笔记 G.3 以及对两份转储的比较：explicit kernel 由 69 条指令组成的分配序列，`sass_explicit.txt` 0x4a90-0x4ed0，与 `sass_gemm.txt` 0x3020-0x3460 有相同的长度和骨架，但在八处等价的助记符拼写（例如以 `SHF.L.U32` 代替 `IMAD.SHL.U32`）、若干局部重排以及寄存器编号上有差异）：ptxas 把这一条 PTX 指令下沉为一个由 **leader 的被选中的 lane** 来做这件工作的序列。它执行原子操作 `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN UP0, UR4, UR4`，其中 `UR4 = 0x10`：分配单位是 32 列（`tmem_allocator_sm100.hpp:119` 处的 `ColumnsPerAllocationSlice = 32`），所以 0x10 = 16 个单位 = 512 列。如果 TMEM 不空闲，它用 `NANOSLEEP 0x64` 重试；这个循环就是 ISA 所说的 "The tcgen05.alloc blocks if the requested amount of Tensor Memory is not available and unblocks as soon as the requested amount of Tensor Memory becomes available for allocation"。它以 `IMAD.SHL.U32 R8, R9, 0x20, RZ` 后跟 `STS [UR29+0x180], R8` 把结果存到本地（0x180 = 384；`R9` 保存原子操作的结果；笔记 D.2 列出了同样的一对）。原子操作以 32 列为单位返回分配结果，而 ptxas 把它移位 5 位变成列地址，这是 `T` 为何是一个硬件返回值、不得被硬编码的一个具体原因（第 9.4.3 节）。然后它用一条 `STAS`（st.async，`STAS [R2.64], R9`）把未移位的 `R9` 转发到 peer 的 `smem + 384`，这条指令完成 peer 的系统保留共享内存中的一个屏障；peer 的被选中的 lane 等待它，就地重新缩放它（`IMAD.SHL.U32 R4, R5, 0x20, RZ; STS [UR29+0x180], R4`）并予以确认，然后两个 warp 重新收敛（`WARPSYNC.ALL`）。

为什么 TMEM 非分配不可：TMEM 是 SM 的一种共享资源，由分配器于运行时在驻留 CTA 之间划分，而一个 CTA 只能使用已授予它的那些列（这正是 alloc/dealloc 协议的全部意义所在）。分配还把同一个 SM 上的各 CTA 串行化：前驱尚未解除分配的 CTA 会在 alloc 中自旋（上面的重试循环）。

### 9.4.2 `__syncwarp()` 与 `bar.arrive 6, 160` (886-887)

`__syncwarp()`（W37，即 `bar.warp.sync -1` 内建函数，explicit.cu:601）镜像了 `sm100_gemm_tma_warpspecialized.hpp:729`。我对 `__syncwarp()` 本身的分析：在一条 `.sync.aligned` 指令之后，warp 已经收敛，所以它在这里的目的是排序（把分配的弱存储发布给其他 lane，第 2 章，第 2.3.2 节）而不是收敛；在源码中，它也是在一条命名屏障指令之前重新建立收敛的自然位置。在 baseline 的 SASS 中，ptxas 在其位置上保留了一条 `WARPSYNC.ALL`（笔记 D.2，归因是推断；G.3 没有对 explicit kernel 重新核对这一细节（它的分配序列比较在第 9.4.1 节）；`sass_explicit.txt` 确实在 `0x4eb0` 处有同样的一条 `WARPSYNC.ALL`，位于 `BAR.ARV 0x6, 0xa0` 之前两条指令处）。

wrapper W19 (explicit.cu:435-437) 是 `bar.arrive %0, %1;`，其 `barrier_id = kBarTmemAlloc = 6` 且 `num_threads = kTmemAllocBarThreads = 160` (explicit.cu:190-191)，来自 `NamedBarrier::arrive_internal`（`barrier.h:305-309`）。

- **这 160 个是谁。** CUTLASS 把该屏障构造为 `NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads, ReservedNamedBarriers::TmemAllocBarrier)`（`sm100_gemm_tma_warpspecialized.hpp:558`），其中 `NumMMAThreads = NumThreadsPerWarp = 32`（`:138`）且 `NumEpilogueThreads = CollectiveEpilogue::ThreadCount = 128`（`:141`）。（ISA 补充道："Note that a non-zero thread count is required for barrier{.cta}.arrive"；160 = 5 个 warp。）
- **为什么用 `bar.arrive` 而不是 `bar.sync`。** ISA："barrier{.cta}.arrive does not cause executing thread to wait for threads of other participating warps"，而 `.sync` 则 "causes executing thread to wait for non-exited threads of all other warps participating in the barrier to arrive"（这也是 CUTLASS 把这两种形式命名为 `arrive_internal` 和 `arrive_and_wait_internal` 的原因，`barrier.h:285, 305`）。warp 0 产生这一信息（TMEM 基址），而它接下来的动作，即消费第一个 CLC 响应（`clc_consume`，第 913 行）然后在 `full[0]` 上等待第一批 TMA 数据（第 916 行 peek，第 923 行阻塞等待），不依赖于 epilogue warp。一条 `bar.sync` 会让第一条 MMA 停顿，直到全部四个 epilogue warp 都走完各自的 prologue。
- **是什么发布了 `T`。** ISA 指出，`.sync`、`.red` 或 `.arrive` "instruction guarantees that when the barrier completes, prior memory accesses requested by this thread are performed relative to all threads participating in the barrier"。使第 10 章对 `tmem_base_ptr` 的普通加载正确的，是这一保证，而不是 `bar.sync` 本身。
- **为什么是 id 6。** 两个参与者集合不同的屏障必须有不同的 id。explicit kernel 沿用了 CUTLASS 的保留 id 1 和 6（第 2 章，第 2.10.2 节），以便 PTX 逐行匹配（笔记 G.3）。

### 9.4.3 读取 `T` (888)

`T` 是通过**通用**指针 `smem + OFF_TMEM_BASE_PTR` 读取的（地址空间规则中 C++ 解引用的那一半，explicit.cu:22-23），镜像了 `uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;`（`sm100_gemm_tma_warpspecialized.hpp:731`）。注释说 "measured 0, never assumed"。这两半都是刻意为之：

- *实测 0*：被追踪的那次运行的全部 264 条 K2 记录（两个站点，全部 132 个 CTA）以及 K0 探针 kernel 的 16 个 CTA 都报告 `tmem_base_ptr = 0`（笔记 D.1 第 D6 行）。在每个 SM 一个 CTA 且请求了整个 TMEM 的情况下，没有任何其他分配能先于这一次，所以 0 正是人们所预期的（我的分析）。
- *绝不假定*：基址是一个硬件返回值。如果这个 kernel 曾以一种让两个 CTA 共享一个 SM 的配置运行，或者分配的列数少于 512，那么非零基址将是合法的。Colfax 的 TMEM 教程把分配描述为动态的。"tcgen05 for dummies" 博客（gau-nernst 的 tcgen05 文章）评论说 "From my experience, `taddr` is always zero, at least when doing only 1 allocation in the kernel's lifetime"，这是一个恰恰带着那个要紧的限定语的观察，而不是一个保证。而且第 9.4.1 节展示了 ptxas 把该原子操作的运行时结果从以 32 列为单位重新缩放为 `T`。笔记的目标清单把这一点说得很明确："TMEM base read from tcgen05.alloc's result (0 here; do not hard-code)"（笔记 D.7）。kernel 中的每个 TMEM 地址都是 `T + ...`（第 919 行；第 10 章的第 1031 行）。

这次普通加载在分配之后的排序：在编译器一侧，`tcgen05.alloc` 的 asm 没有 `"memory"` clobber（W26），但介于其间的 `bar.arrive` asm 有（W19，`: "memory"`），并且 `__syncwarp()` 同样是一个编译器屏障，所以该加载不可能被外提到两者中任何一个之上。在硬件一侧，`__syncwarp()` 为 warp 0 的各个 lane 排序了分配的弱存储，如第 9.4.2 节所示（笔记 E.6.4 讨论了这一点；CUTLASS 自己在 `:731` 处做的就是普通加载）。

### 9.4.4 描述符低字 (895-897)

```
895:  // UMMA smem descriptors from this CTA's own stage bases; the 14-bit start-address field drops the rank bits (mma_sm100_desc.hpp:754-755)
896:  uint32_t const desc_lo_a = kSmemDescLo0 | (((smem_base + OFF_SMEM_A) >> 4) & 0x3FFFu);      // expected 0x10880 (D.7)
897:  uint32_t const desc_lo_b = kSmemDescLo0 | (((smem_base + OFF_SMEM_B) >> 4) & 0x3FFFu);      // expected 0x12880 (D.7)
```

一条两个操作数都在共享内存中（"SS"）的 `tcgen05.mma` 接受两个 64 位的*共享内存描述符*（A 和 B）和一个 32 位的*指令描述符*。第 5 章，第 5.8.1 节逐字段给出了共享内存描述符的完整位图（`union SmemDescriptor`，`mma_sm100_desc.hpp:101-127`）以及每个值的理由；第 5.8.2 节对指令描述符做了同样的事。本节只讲引擎拿它们做什么。常量的两半是 `kSmemDescHi` 和 `kSmemDescLo0`（explicit.cu:171-173，引文见第 5.8 节）。

`kSmemDescHi` 是这个 64 位字的第 [32,64) 位。它位于 [32,46) 的 SBO = 64 是相邻两个 8 行 swizzle 组之间的 1024 B 距离；版本 1 位于 [46,48)；而它位于 [61,64) 的 `SWIZZLE_128B` = 2 必须与 TMA 写入的模式一致（explicit.cu:133），这样 UMMA 单元才会按写入时的样子读取该 stage。`kSmemDescLo0` 是第 [0,32) 位。它位于 [16,30) 的 LBO = 1（16 B）是 swizzled K-major layout 的规范占位值，而它位于 [0,14) 的 14 位起始地址字段留空，待运行时做 OR。文件对这两个常量都做了断言（explicit.cu:174-175），而在 trace 中实测的首条 MMA 的描述符 `0x4000404000010880` / `0x4000404000012880`（笔记 D.1 第 D7 行）证实了它们。

**为什么起始地址放在一个 14 位字段里，以及这换来了什么。** `make_umma_desc` 把 `desc.start_address_ = static_cast<uint16_t>(start_address >> 4);` 写入一个 `uint16_t start_address_ : 14` 位域（`mma_sm100_desc.hpp:107, 754-755`）。这个 CTA 的 smem 的 `shared::cta` 地址是 `0x400 + (r << 24) + OFF`（实测，笔记 D.1 第 D5 行）：cluster rank 位于地址字节 3。右移 4 位把 rank 移到第 [20,22) 位；14 位字段只保留第 [0,14) 位。于是 rank 被丢弃，而 `desc_lo_a`、`desc_lo_b` 在全部四个 CTA 中是*相同*的：

```
   rank r:  smem_base + OFF_SMEM_A = 0x0r00_0400 + 0x8400 = 0x0r00_8800     >> 4  = 0x00r0_0880     & 0x3FFF = 0x0880
            smem_base + OFF_SMEM_B = 0x0r00_0400 + 0x28400 = 0x0r02_8800    >> 4  = 0x00r0_2880     & 0x3FFF = 0x2880
   desc_lo_a = 0x10000 | 0x880  = 0x10880          desc_lo_b = 0x10000 | 0x2880 = 0x12880           (for every r = 0..3)
```

文件对实测基址下的这两个值做了断言（explicit.cu:177），而 trace 在两个 leader 的首条 MMA 处证实了它们（`desc_a = 0x4000404000010880`、`desc_b = 0x4000404000012880`，笔记 D.1 第 D7 行）。这种与 rank 无关的性质并不是一种便利；它是 `cta_group::2` 的一项要求。注意这个算例证明了什么、没有证明什么。它证明了该*字段*在每个 CTA 中是同一个数。而被读取的仍然是 peer 的字节，这是该指令的一个性质：在 `.cta_group::2` 下，leader 发射的那一个描述符在每个 CTA 自己的共享内存窗口中、在相同的 CTA 相对偏移处被解析（源自原子 layout 和笔记 B.6，"at the same CTA-relative addresses"；ISA 文本没有把这一点明说，第 9.3.1 节）。这与 ISA 确实为 multicast 投递——其数据和 mbarrier 信号落在每个被掩码的 CTA 中 "the same CTA-relative offset" 处（`cp.async.bulk.tensor`；第 8 章）——以及为第 9.5.3 节的 multicast commit 操作所陈述的约定是同一个约定。所以，一个编码了“rank 0 的缓冲区”的描述符对该指令的 peer 那一半来说将毫无意义。14 位字段还把 CTA 相对地址的上限定在 256 KiB，所以整个缓冲区都必须落在该上限之下：`static_assert(kSmemBaseLow24 + kSmemBytes < (1u << 18), ...)`（explicit.cu:165；`0x400 + 230400 = 231424 < 262144`）。

为什么 explicit kernel 用 `& 0x3FFFu` 来掩码而不是依赖位域：文件用普通整数构建描述符（运行时没有 CuTe），所以位域隐式完成的截断必须明确地写出来。同样的选择让 `desc_lo_*` 可以每个 CTA 只构造一次，然后用 32 位加法推进（第 9.5.3 节）。

**指令描述符** `kInstrDesc = 0x10200010`（explicit.cu:173，引文见第 5.8 节）一次性固定了 256 x 128 x 16 的 shape 和格式：F32 累加 `0x10`、F16 输入、K-major 的 A 和 B、位于 [17,23) 的 `n_dim = 128 >> 3 = 16`、位于 [24,29) 的 `m_dim = 256 >> 4 = 16`。它的位图是 `union InstrDescriptor`（`mma_sm100_desc.hpp:416-443`），在第 5 章，第 5.8.2 节中逐字段推导。文件在 explicit.cu:176 处对它做了断言，并且它在每条 K3 记录中都被实测到（笔记 D.1 第 D7 行）。它在 asm 中是一个运行时的 `"r"` 寄存器（W29 第 526 行），与 CUTLASS 中一样（`mma_sm100_umma.hpp:579`，按 cb424739 的编号，`"r"(uint32_t(idescE>>32))`），所以 PTX 形式一致；ptxas 在循环中用 `UMOV UR29, 0x10200010` 把它物化（笔记 G.3）。

### 9.4.5 屏障基址与累加器 commit 掩码 (898-902)

```
898:  uint32_t const full_base = smem_base + OFF_MAIN_FULL;
899:  uint32_t const empty_base = smem_base + OFF_MAIN_EMPTY;
900:  uint32_t const acc_full_base = smem_base + OFF_ACC_FULL;
901:  uint32_t const acc_empty_base = smem_base + OFF_ACC_EMPTY;
902:  uint16_t const acc_commit_mask = static_cast<uint16_t>(0x3u << (2u * y));                    // the pair: 0x3 for ranks {0,1}, 0xC for {2,3} (calculate_umma_peer_mask)
```

在 `AtomThrShape_MNK = (2,1,1)` 下，`acc_commit_mask` 为该原子的每个 CTA 置一位，即 rank `2y` 和 `2y+1`（列主序的 cluster layout，rank = `x + 2y`），所以对 pair `{0,1}`（y = 0）有 `0x3 << 2y` = `0x3`，对 `{2,3}`（y = 1）则为 `0xC`。掩码是 16 位的，因为 `tcgen05.commit ... multicast::cluster` 接受一个 `.b16` 的 cta 掩码操作数（`"h"` 约束，W30 第 538 行）。一次误落在另一个 pair 的 `acc_full[st]` 上的到达，会在那个 pair 的 MMA 完成之前使它完成一个 phase，而它的 epilogue 就会读到一个只完成了一半的累加器（第 9.5.4 节）。

### 9.4.6 流水线状态 (903-908)

```
903:  uint32_t cons_idx = 0, cons_phase = 0;     // mainloop consumer state (8 stages), leader only
904:  uint32_t accp_idx = 0, accp_phase = 1;     // accumulator producer state (4 stages), both CTAs
905:  uint32_t clc_idx = 0, clc_phase = 0, clc_count = 0;
906:  int tm_next = tm, tn_next = tn;
907:  uint32_t tile = 0;
908:  bool valid;
```

三个流水线状态：

| 状态 | warp 0 的角色 | 起始值 | 深度 | 谁使用它 |
|---|---|---|---|---|
| `cons_idx/cons_phase` | mainloop 环的 consumer（`full[s]`） | `{0, 0}` | 8 | 仅 leader（第 916-927 行） |
| `accp_idx/accp_phase` | 累加器流水线的 producer（`acc_empty[st]` acquire，`acc_full[st]` commit） | `{0, 1}` | 4 | 两个 CTA 都读取并推进它（914、946）；acquire、commit 和尾段仅 leader（918、944、955-960） |
| `clc_idx/clc_phase/clc_count` | CLC 响应槽位的 consumer | `{0, 0, 0}` | 2 | 两个 CTA（913） |

---

## 9.5 tile 循环 (explicit.cu:909-949)

```
909:  do {
913:    valid = clc_consume<K_CLC_MMA_KIND>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // BEFORE computing the current tile (:742-750)
914:    uint32_t const acc_stage = accp_idx;
915:    if (is_leader) {                                                                            // mma() (:762-770)
916:      uint32_t tok = (k_tiles <= 0) ? 1u : mbar_try_wait(full_base + 8u * cons_idx, cons_phase);   // consumer_try_wait(skip_wait) (:670-671)
917:      uint32_t scale_c = 0;                                                                     // ScaleOut::Zero (:676)
918:      EXPLICIT_WAIT(acc_empty_base + 8u * acc_stage, accp_phase, WAIT_ACC_EMPTY, tile, 0u);    // accumulator producer_acquire, blocking (:678)
919:      uint32_t const tmem_c = T + 128u * acc_stage;                                             // lane 0, columns [128 st, 128 st + 128): 0, 128, 256, 384 for T = 0
920:      int k = k_tiles;
921:      #pragma unroll 1                                                                          // CUTLASS_PRAGMA_NO_UNROLL (:680)
922:      while (k > 0) {
923:        if (!tok) { EXPLICIT_WAIT(full_base + 8u * cons_idx, cons_phase, WAIT_MAINLOOP_FULL, tile, uint32_t(k_tiles - k)); }   // consumer_wait (:684)
924:        uint32_t const rs = cons_idx;                                                           // read_stage
925:        advance<kStages>(cons_idx, cons_phase);
926:        --k;
927:        tok = (k <= 0) ? 1u : mbar_try_wait(full_base + 8u * cons_idx, cons_phase);             // peek the next stage; skipped on the last k-tile (:694-696)
928:        uint32_t const lo_a = desc_lo_a + 1024u * rs;                                           // + 16384 B per stage in 16-byte units (32-bit low-word add, mma_sm100_desc.hpp:861-874)
929:        uint32_t const lo_b = desc_lo_b + 512u * rs;                                            // +  8192 B per stage
930:        #pragma unroll                                                                          // (:699)
931:        for (int kb = 0; kb < kKBlocks; ++kb) {
932:          if (elect_one_sync()) {
933:            uint64_t const desc_a = desc64(kSmemDescHi, lo_a + 2u * uint32_t(kb));              // + 32 B per k-block
934:            uint64_t const desc_b = desc64(kSmemDescHi, lo_b + 2u * uint32_t(kb));
938:            umma_f16_2sm(tmem_c, desc_a, desc_b, kInstrDesc, scale_c);
939:          }
940:          scale_c = 1u;                                                                          // ScaleOut::One after every gemm (:706)
941:        }
942:        if (elect_one_sync()) { umma_commit_mcast_2sm(empty_base + 8u * rs, kMainloopReleaseMask); }   // consumer_release, mask 0xF (:708)
943:      }
944:      if (elect_one_sync()) { umma_commit_mcast_2sm(acc_full_base + 8u * acc_stage, acc_commit_mask); }   // accumulator producer_commit (:771)
945:    }
946:    advance<kAccStages>(accp_idx, accp_phase);                                                  // both CTAs (:773)
947:    tm = tm_next; tn = tn_next;                                                                 // (:774-775)
948:    ++tile;
949:  } while (valid);
```

在生产构建中，`[TRACE]` 的第 910-912 行和第 935-937 行被编译掉，`K_CLC_MMA_KIND` 是 `0u`，而 `EXPLICIT_WAIT(addr, parity, ...)` 就是 `mbar_wait(addr, parity)`（explicit.cu:237-239），所以每个 `EXPLICIT_WAIT` 都是阻塞的 W6 循环。

### 9.5.1 先消费 CLC 响应 (913)

`clc_consume`（第 7 章，explicit.cu:677-706）返回 `valid`。

为什么它排在计算当前 tile **之前**：这个循环精确地镜像了 `sm100_gemm_tma_warpspecialized.hpp:738-776`。在那里，`scheduler.fetch_next_work`（`:742-746`）先于 `collective_mainloop.mma`（`:763`），而取到的 tile 只在底部才被装入（`work_tile_info = next_work_tile_info;` `:774`）。由此得出三个后果（笔记 7.4）：

1. 在 CUTLASS kernel 的 trace 中实测（笔记 7.6，cluster (0,0)）：rank 0 的第一次消费（它甚至先于初始 tile）比它的第一条 `tcgen05.mma` 早 1248 ns，rank 2 的早 576 ns（在 explicit kernel 自己的 trace 中，`trace_out_v2/launch0.csv`，两个间隔都是 1056 ns）。
2. 最后一个有效 tile 是在未取消的终止符被消费*之后*才计算的，所以结束循环的那个响应永远不会让最后一个 tile 的计算等待。
3. 循环只有在第 914-948 行已为当时的当前 tile 运行过之后才在 `while (valid)`（949）处退出，这就是 `tm = tm_next`（947）和 `++accp`（946）位于 leader 分支*之下*的原因。

### 9.5.2 leader 进入一个 tile (914-920)

`acc_stage = accp_idx`（914）在 `is_leader` 分支**之前**读取，因为两个 CTA 都将在 946 处推进同一个状态。leader 把它用于自己的屏障地址和 TMEM 地址；peer 用它只是为了保持步调一致。

**跳过等待的 peek（916）。** CUTLASS 的 `consumer_try_wait(state, skip_wait)` 在 `skip_wait` 被置位时不触碰屏障就返回 `WaitDone`，否则执行一次 `mbarrier.try_wait.parity`（`sm90_pipeline.hpp:590-597`）；`skip_wait = k_tile_count <= 0`（`sm100_mma_warpspecialized.hpp:670-671`）。保留 `k_tiles <= 0` 这个守卫，是因为源码这样做，也是因为它保护一个 k-tile 数为零的 kernel 不去 peek 一个它永远不会等待的屏障。对 `full[0]` 的 peek 在阻塞的累加器等待（918）*之前*发射，是因为那是 CUTLASS 的顺序（`:671` 先于 `:678`）。这次 peek 没有副作用，而且一个 `WaitDone` token 不会变陈旧：一个已完成的 phase 保持已完成，直到这个 consumer 释放该 stage，而它还没有释放。这个靠前的位置所*不*做的事，是帮助数据到达。`mbarrier.try_wait.parity` 报告的是它返回那一刻的 phase 状态（ISA 允许它把线程挂起，直到该 phase 完成或一个系统相关的时间限制到期，PTX ISA 9.4，9.7.15.16.19，引文见第 2.8.4 节），而 `tok` 从不被重新求值，所以一次 peek 只能观察到届时已经落地的数据，而一次更晚的 peek 成功的频率至少不会更低。对这第一次 peek 而言，顺序在实践中无关紧要：producer 在 cluster 屏障之后立刻发射了 stage 0，而 warp 0 在到达第 916 行之前花掉了分配序列加上它的第一次 CLC 消费的时间（在 CUTLASS kernel 的 trace 中，rank 0 的第一次消费比它的第一条 MMA 早 1248 ns，笔记 7.6，在 explicit kernel 自己的 trace 中早 1056 ns），所以 stage 0 早已落地（我的分析）。

**`scale_c = 0`（917）** 是 `tiled_mma.accumulate_ = UMMA::ScaleOut::Zero`（`:676`）：该 tile 的第一条 MMA 将*覆写* TMEM stage。第 9.5.3 节解释这个谓词。

**对 `acc_empty[st]` 的阻塞等待（918）** 是 `accumulator_pipeline.producer_acquire(state)`（`:678`）。对于 `PipelineUmmaAsync`，它以默认的 `WaitAgain` token 转发到 `PipelineAsync::producer_acquire`（`sm100_pipeline.hpp:207-209`，`sm90_pipeline.hpp:1110-1112, 1185-1190`）：一次普通的阻塞 `wait(phase)`，没有 peek。这里用阻塞等待而不是第 916 行那种先 peek 再 wait 的做法是对的，因为在 TMEM 满时 leader 没有任何有用的事可做；剩下的唯一工作是这个 tile 的 MMA，而它们需要该 stage。

**`tmem_c = T + 128u * acc_stage`（919）**：D 操作数。列字段是 `128 st`（对 `T = 0` 为 0、128、256、384），lane 字段是 0。该指令在每个 CTA 中从 lane 0 开始写入 128 个 lane（第 9.3.2 节）。它是一个运行时值，因为 `T` 是运行时值。

### 9.5.3 k 循环 (921-943)

`#pragma unroll 1` (921) 即 `CUTLASS_PRAGMA_NO_UNROLL`（`:680`）。循环体每个 tile 执行 128 次；展开会成倍增加代码体积和活跃 uniform 寄存器的数量而毫无收益，因为 tensor core 已经隐藏了该循环的发射成本（第 9.8 节）。ptxas 在两个 kernel 中都把它保持为一个真正的循环，没有剥离出第一次迭代（笔记 D.2、G.3）。

**等待、推进、peek (923-927)。** 在等待（`if (!tok)`，`:684`）之后依次是 `rs = cons_idx`（要读取的 stage，`read_stage`，`:687`）、`advance<8>`（`:692`）、`--k`（`:693`），以及对*下一个* stage 的 peek（`:694-696`），在 `k <= 0` 时跳过。为什么 peek 放在这里、在四条 MMA 之前，而不是放在下一次迭代的顶部（我的分析）：这个测试没有副作用，并且一旦下一个 `{index, phase}` 已知就立即发射（CUTLASS `:694-696`）。如第 9.5.2 节所述，它的结果在 `try_wait` 返回时就已固定（ISA 允许它最多挂起到一个取决于系统的上限，第 2.8.4 节），所以这一放置位置并不能让数据“及时到达”；它换来的是对测试本身的流水化。`TRYWAIT` 的谓词在这里产生，而只由循环顶部的分支消费（笔记 G.3：当 peek 成功时，`@P0 BRA` 跳过阻塞等待的入口 `TRYWAIT`，连同主路径之外的重试循环一起跳过），因此这条指令自身的延迟隐藏在四次 MMA 发射的背后，而不是落在通往下一个 k-tile 的第一条 MMA 的路径上。当它返回 `WaitAgain` 时，唯一的代价是循环顶部多一条 `TRYWAIT`，它随后要么发现数据已经就位，要么进入阻塞循环。

笔记 B.3.2 从源码推导出每个 tile 的 peek 次数（第 916 行 1 次 + 第 927 行 127 次 = 128），而 G.3 确认 SASS 恰好保留了这一谓词化；peek 在运行时*失败*的频率没有实测。

**描述符算术 (928-929, 933-934)。**

```
   lo_a = desc_lo_a + 1024 rs + 2 kb         1024 = 16384 B / 16 B  (one A stage: 128 rows x 64 fp16 x 2 B)
   lo_b = desc_lo_b +  512 rs + 2 kb          512 =  8192 B / 16 B  (one B stage:  64 rows x 64 fp16 x 2 B)
                                                 2 =    32 B / 16 B  (one k-block: 16 fp16 along K)
```

这些是*仅作用于低 32 位字*的 `DescriptorIterator::operator+`（`mma_sm100_desc.hpp:861-874`；对于 CUDA 13.3，编译的是 `#else` 分支 `ret.lo = desc_.lo + uint32_t(offset); ret.hi = desc_.hi;`）。这就是为什么 kernel 把 `desc_lo_*` 保持为 32 位整数，只在 asm 边界处用 `desc64(kSmemDescHi, lo)` 构成 64 位操作数（explicit.cu:616）。对于实测基址 `0x400`（explicit.cu:156），这个加法不可能从 14 位字段进位溢出：最大的低 14 位值是 B 的 `0x2880 + 7 x 512 + 6 = 13958 < 16384`（笔记 B.3.2 步骤 8 把上界表述为 `13894 + (smem >> 4)`，其中 `smem >> 4 = 64`），所以普通加法是精确的。ptxas 把 stage 项下沉为 `ULEA`（按 `rs << 10` 和 `rs << 9` 做移位加），把 k-block 项下沉为 `UIADD3 +0x2/+0x4/+0x6`，并用 `UMOV URn, 0x40004040` 重新物化高字（笔记 D.2 和 G.3，实测）。

为什么 `+32 B per k-block` 是 swizzled stage 内部正确的地址：K-major SW128 规范 layout 以 16 字节为单位是 `Swizzle<3,4,3> o ((8,n),2):((8,SBO),1)`（`mma_sm100_desc.hpp:807`）。一行容纳 64 fp16 = 128 B = 八个 16 字节 chunk，而 k-block `kb` 是 swizzling 之前每一行的 chunk `[2 kb, 2 kb + 2)`。UMMA 单元对地址位 [4,7) 与位 [7,10) 施加与 TMA 写入时相同的 XOR，因此把起始地址前移 2 个 chunk，就在 8 行组的每一行中选中了正确的 swizzle 前的列；SBO (1024 B) 在组与组之间步进。kernel 从不计算 swizzle（explicit.cu:203）。

**四条 MMA (930-941)。** k-block 循环被完全展开（`CUTLASS_PRAGMA_UNROLL`，`:699`）。`:698` 处的源码注释说明了原因："Unroll the K mode manually so we can set scale C to 1"。展开使 `scale_c` 对于 `kb = 1..3` 成为编译期的 1，所以只有 `kb = 0` 那条指令带有活跃的谓词（笔记 D.2："only the k-block-0 MMA has a live scale-c register"），并且它把 `+2 kb` 加法变成了立即数。每次迭代都是 `if (elect_one_sync()) { ... umma_f16_2sm(...); }`：由一个 lane 发射，与 `SM100_MMA_F16BF16_2x1SM_SS::fma`（`mma_sm100_umma.hpp:570`，`if (cute::elect_one_sync())`）中一样，其函数体就是 W29（explicit.cu:517-529，引文见第 6.7.2 节）。

第 6 章，第 6.7.2 节分解了这条指令的每一个限定符。下面按 ISA 自己的名称列出各操作数（`tcgen05.mma`，其语法为 `tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc, { disable-output-lane }, enable-input-d {, scale-input-d};`）。`[%0]` 是 `d-tmem`，即 TMEM 中 D 的地址。`%1, %2` 是 `a-desc`、`b-desc`，即 64 位共享内存描述符（`"l"`）。`%3` 是 `idesc`。`p` 是 **`enable-input-d`**，由 `setp.ne.b32 p, scaleC, 0` 计算得出："The operation of the form D = A*B is issued when the input predicate argument enable-input-d is false"，否则为 `D = A*B + D`。Colfax 的 TMEM 教程把这个参数描述为在 "zeroing out the accumulators before executing MMA (the operation D = A * B) and retaining the accumulators (D = A * B + D)" 之间切换；MLC 一章："false computes D=A x B, while true computes D=A x B+D"。八个为零的 `.b32` 操作数 `{%5, ..., %12}` 是 **`disable-output-lane`** 向量："a mask where each bit corresponds to a lane of the Tensor Memory ... If a bit in the mask is 1, then the corresponding lane in the Tensor Memory for the resultant matrix D will not be updated"。该向量对 `.cta_group::1` 有 4 个元素，对 `.cta_group::2` 有 8 个（8 x 32 = 256 位，pair 的每个 TMEM lane 各一位），所以全零意味着两个 CTA 的每个 lane 都会被写入（CUTLASS 的 `uint32_t mask[8]`，`mma_sm100_umma.hpp:571`）。它们在 PTX 中是寄存器，而在 SASS 中消失了（`UTCHMMA.2CTA` 没有掩码操作数，笔记 D.2）。可选的立即数 `scale-input-d`（`D = A*B + D * 2^-s`，`s` 在 [0, 15] 内，对 `.kind::f16` 和 `.kind::tf32` 有效）未出现，所以 `D` 不经缩放地进入。`.kind::f16` 选择 FP16/BF16 输入族；F16 输入与 F32 累加的选择位于指令描述符中。

**先 `scale_c = 0` 后 `1` 达成了什么 (917, 940)。** 每个 tile 的第一条指令（k-tile `q = 0`，k-block `kb = 0`）把 `A x B` 写入 TMEM stage `st` 而不读取它，所以该 stage 在一个 tile 之前不需要单独的一趟清零，而 tile `t-4` 的陈旧累加器被覆写。第 940 行在*每条* MMA 之后的 `scale_c = 1u` 即 `tiled_mma.accumulate_ = UMMA::ScaleOut::One`（`:706`）。在循环内部设置它，而不是在 `kb = 0` 之后设置一次，保持了源码的结构，并让编译器看到 `kb >= 1` 时的常量。实测：K3 记录到每个 leader 的第一条 MMA 中 `scale_c = 0`，随后 15 条中为 `1`（笔记 7.6）。在 SASS 中，第一条 `UTCHMMA` 把 uniform 谓词 `UP0` 作为它的 scale-c 操作数，该谓词在 k-tile 循环之前每个 tile 清零一次（`0x51c0` 处的 `UPLOP3`），并在每次 k-tile 迭代的底部、commit 操作之后置为真（`0x54b0`；baseline 在第一条 `UTCHMMA` 之后立即设置它的 `UP2`，笔记 D.2）；其余三条使用恒真谓词（笔记 D.2、G.3）。

为什么通过 `elect_one_sync()` 而不是 `if (lane == 0)` 来发射：`tcgen05.mma` "has single thread semantics"，所以一个发射 lane 就启动整条 MMA（第 2 章，第 2.4.2 节）。此时 warp 已收敛，而 `elect.sync` 是 CUTLASS 的惯用写法。ptxas 把 producer、MMA 和 scheduler 循环中的每一条 `ELECT` 都折叠掉了，因为它们所守护的指令（`UTMALDG`、`UTCHMMA.2CTA`、`UTCBAR`、`UGETNEXTWORKID`）都是每个 warp 发射一次的 uniform 数据通路指令（笔记 D.6 第 6 项，G.3；第 12 章，第 12.3.2 节）。explicit kernel 在每个发射点保留一次选举（每个 k-tile 4 条 MMA + 1 次 commit，每个 tile 1 次累加器 commit：PTX 中每个 tile 641 次，笔记 B.3.2），因为这是 baseline 所写出的 PTX 契约；`lane == 0` 测试在语义上等价，但不会复现 baseline 的 PTX。

**以掩码 0xF 向 `empty[rs]` 的 commit 操作 (942)。** 在第四条 MMA 之后，被选中的 lane 发射 W30（explicit.cu:532-540），即 `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [empty_base + 8 rs], 0xF`，它复制自 `umma_arrive_multicast_2x1SM`（`barrier.h:848-861`），也就是 `consumer_release` 在 2SM 流水线中所变成的形式（`sm100_pipeline.hpp:740-745`）。

这些限定符中有一个承担着本章所依赖的工作。`.cta_group::2` 不是逐条指令各自做出的选择："All tcgen05 instructions within a kernel must specify the same value for the .cta_group qualifier"（在 `tcgen05.mma`、`tcgen05.commit` 和 `tcgen05.alloc` 之下都有陈述）。这就是为什么 explicit.cu 中的 alloc、dealloc、relinquish、mma 和 commit 全都带有 `::2`，尽管只有 MMA 需要 pair。由此得出两个设计事实：

- **为什么掩码是 0xF（`kMainloopReleaseMask`，explicit.cu:186）而不是 pair 的 0x3/0xC。** CTA `r` 的 stage `s` 也会被 CTA `r ^ 2` 的 warp 2 写入，即 `x` 相同而 `y` 不同的那个 CTA，也就是与本 CTA 共享 `tm` 并接收相同的 A 的行的那个。那个 producer 以掩码 `0x5 << x` 进行的 A multicast 使它 64 行的那一半落在 CTA `x` 与 `x + 2` 两者的 `OFF_SMEM_A + 16384 s + 8192 y'` 处（第 8 章，explicit.cu:747, 772）。（这里避免使用行和列这两个词，因为 CUTLASS 在 `sm100_pipeline.hpp:60-61` 处的注释把相同的 `x` 称为 "same MMA cluster col"，把相同的 `y` 称为 "same MMA cluster row"。）在 CTA `r ^ 2` 的 producer 可以覆写 stage `s` 之前，它要等待**它自己的** `empty[s]`。
- **掩码 0x3 会破坏什么。** 若用 pair 掩码，CTA `r` 的 `empty[s]` 每次使用只会得到一次到达（来自它自己的 leader），而计数却是 2，所以它的 producer 会在环的第二次使用时永远阻塞：8 个 k-tile 之后死锁。

```
   stage s of CTA r is written by:  r's TMA (B half, A half)  and  (r ^ 2)'s TMA (A half, multicast 0x5 << x)
   stage s of CTA r is read by:     the 2SM MMA issued by leader(r) = r & ~1   (reads both CTAs of the pair)
   so before ANY producer rewrites stage s anywhere, BOTH leaders' MMAs on stage s must be done:
       leader 0 --commit 0xF--> empty[s] of ranks 0,1,2,3        leader 2 --commit 0xF--> empty[s] of ranks 0,1,2,3
       each empty[s]: count 2 = one arrival from each leader  -> producer of that CTA may refill s (and multicast into its partner)
```

### 9.5.4 累加器 commit 与共享状态推进 (944-948)

第 944 行是 `accumulator_pipeline.producer_commit(state)`（`sm100_gemm_tma_warpspecialized.hpp:771` -> `sm100_pipeline.hpp:212-214, 259-265` -> `umma_arrive_multicast_2x1SM(&full_barrier_ptr_[stage], tmem_sync_mask_)`）：还是那条 W30，如今作用于 `acc_full_base + 8 acc_stage`，掩码为 `acc_commit_mask = 0x3 << 2y`。同一个被选中的 lane 每个 tile 发射它一次，在第 512 条 MMA 之后，因此它到达之时，整个 tile 的累加在两个 TMEM 中都已完成。pair 掩码向 leader 的*以及* peer 的 `acc_full[st]` 各递送一次到达，二者的计数都是 1（`kAccFullCount = 1`，explicit.cu:184；`producer_arv_count = 1`，`:529`），而每个 CTA 的 epilogue 等待的是**它自己的** `acc_full[st]`（第 10 章，explicit.cu:1030，未加掩码的 `acc_full_base`）。peer 的 epilogue 正是这样得知其 TMEM 已就绪的：peer 的 warp 0 没有任何指令触及 tensor core，所以只有 leader 的发射线程知道已完成，而 multicast 把这一消息传递过去。

第 946 行在**两个** CTA 中都推进累加器 producer 状态（`:773` 处的 `++accumulator_pipe_producer_state;` 位于 `if (is_mma_leader_cta)` 之外）；947-948 安装取来的 tile 并对 tile 计数。peer 的状态在循环内从未被用于任何屏障；第 9.5.5 节解释为什么保留它。

### 9.5.5 peer 的 warp 0 做什么，以及它为什么必须存在

对于奇数 rank，循环体只有第 913、914、946、947、948 行：每个 tile 一次 CLC 消费、一次状态推进和一次 tile 更新。在 baseline 的 SASS 中，ptxas 为它生成了一条绕过整个 leader 段的分支（笔记 D.2："The peer CTA's warp 0 skips the whole leader section (`BRA.U UP3, 0x3b90`) and executes per tile only the CLC consume and the state increments"）。explicit kernel 的循环具有相同的结构（笔记 G.3），但 G.3 没有单独记录 peer 的分支。尽管如此，peer 的 warp 0 仍不可或缺。PTX ISA 的 Issue Granularity 表（第 9.7.18.5 节，表 55）是第 1 项和第 4 项背后的规范性理由：

1. **分配与解除分配是 pair 操作。** 对于带 `::2` 的 `.alloc`、`.dealloc` 和 `.relinquish_alloc_permit`，ISA 要求 "Issue from two warps, one in each of the current CTA and its Peer CTA, in order to collectively perform the operation, i.e., the first warp to perform the operation could block until the the second warp in the Peer CTA also performs the operation" [sic]。它还补充说 "the issuing warp must make sure that peer CTA is launched and its warps eventually participate in collective operations"。CUTLASS 把同样的内容表述为前置条件（`tmem_allocator_sm100.hpp:129-132, 152-154`：*两个* CTA 中具有相同逻辑 warp id 的一个完整 warp）。没有 peer 的 warp 0，leader 的分配就无法完成（第 9.4.1 节中的 SASS 握手要等待 peer）。
2. **命名屏障。** peer 的 epilogue warp 执行 `bar.sync 6, 160`，需要来自它们自己 CTA 的 warp 0 的 32 次到达（第 9.4.2 节）；`T` 必须存在于 peer 的 smem 中，这由 pair 分配保证。
3. **CLC 计数。** CTA 0 的 `clc_empty` 每个响应期望 800 次到达，其中每个 CTA 的 warp 0 各贡献 32 次（explicit.cu:183）。缺少一个 consumer 会让 scheduler 在它的第三次查询时死锁（即第二次进入槽位 0 的查询：producer 状态从 `{0, 1}` 开始，explicit.cu:823，所以查询 1 和 2 的 acquire 在全新的屏障上通过，而查询 3 是第一个需要响应 1 的 800 次到达的查询）。
4. **循环终止与驻留。** peer 只能从终止符响应（`valid == 0`）得知 kernel 已经结束，而且它必须运行与 leader 完全相同的迭代次数，因为二者消费同一个响应流（第 3 项需要每个响应上每个 warp 0 的 32 次到达）；在终止符之后它继续执行第 950-951 行以及第 9.6.4 节的握手，并在那里阻塞 (965)，直到 leader——它在计算自己最后一个 tile 之前于第 913 行消费了同一个终止符——完成了那个 tile 及其尾段。ISA 还为 MMA 本身增加了一条驻留规则：对于带 `::2` 的 `.mma` 和 `.commit`，"When the current CTA issues the operation, the peer CTA should be active and should not have exited"。因此当 leader 发射它最后的 MMA 和 commit 时，peer *CTA*——而不只是它的 warp 0——必须仍然驻留（该规则说的是发射时刻；直到每次 TMEM 访问都完成为止的驻留，则由第 9.6.4 节的解除分配规则推出）。peer 的 warp 0 跑完整个循环然后在握手中阻塞，正是把它留在那里的原因；它的 epilogue warp 则由累加器流水线留住（第 10 章）。
5. **保留状态推进是为了与源码保持一致，而不是因为有什么东西读取它。** 第 1-4 项是 peer 的 warp 0 存在的运作层面的理由。`:773` 处无条件的 `++accumulator_pipe_producer_state`（第 9.5.4 节）则是另一回事：CUTLASS kernel 的两条 `IsOverlappingAccum` 路径也都不使用 peer 的状态。在 `IsOverlappingAccum` 路径（`:753-756`）中，peer 同样只计算 `acc_stage = phase ^ 1`，而 `mma()` 和 `producer_commit` 仍在 `is_mma_leader_cta` 之下（`:762-772`），重叠的尾段（`:799-801`）只等待解除分配屏障。explicit kernel 保持第 946 行无条件执行，以便 PTX 与源码一致（笔记 G.3）。第 9.6.4 节的握手是一个屏障的一个 phase，用任何 peer 状态都能工作（我的分析）。

---

## 9.6 终止 (explicit.cu:950-975)

```
950:  griddep_launch_dependents();                                                                  // launch_dependent_grids (:781)
951:  tmem_relinquish_2sm();                                                                        // release_allocation_lock (:784)
952:  if (is_leader) {                                                                              // accumulator producer_tail (:789; sm90_pipeline.hpp:1129-1134) (Section 7.10 tail 3)
953:    #pragma unroll
954:    for (int i = 0; i < kAccStages; ++i) {
955:      EXPLICIT_WAIT(acc_empty_base + 8u * accp_idx, accp_phase, TAIL_ACC, tile, uint32_t(i));
960:      advance<kAccStages>(accp_idx, accp_phase);
961:    }
962:  }
963:  // TMEM deallocation handshake (:794-796): the peer arrives on the leader's barrier, both wait phase 0, the leader arrives on the peer's (Section 7.10 tail 4)
964:  if (!is_leader) { mbar_arrive_remote(smem_base + OFF_TMEM_DEALLOC, peer); }
965:  EXPLICIT_WAIT(smem_base + OFF_TMEM_DEALLOC, 0u, WAIT_TMEM_DEALLOC, tile, 0u);
970:  if (is_leader) { mbar_arrive_remote(smem_base + OFF_TMEM_DEALLOC, peer); }
971:  tmem_dealloc_2sm(T, kTmemColumns);                                                            // (:804), both CTAs
975:}
```

### 9.6.1 `griddepcontrol.launch_dependents` (950)

warp 0 的全部 32 个 lane 都执行 `griddep_launch_dependents()`（W36，explicit.cu:600；SASS 中每个 CTA 一条 `PREEXIT`，笔记 D.2）。为什么不更早（我的分析）：依赖网格只有在本网格的*每一个* CTA 都已执行 `griddepcontrol.launch_dependents` 或已退出之后才会被启动（第 2 章，第 2.13 节引用的 PTX ISA 那句话），所以单个 CTA 提前触发本身不会改变任何事情。如果所有 CTA 都在各自开始时触发，依赖网格的 CTA 就可能在本 kernel 仍在计算时被调度，但在每个 SM 一个 230400 字节 CTA 的情况下，若一个依赖网格的 CTA 需要相当份额的 SM 共享内存，它就只能占据本次启动闲置的那 16 个 SM（笔记 D.3），这正是 CUTLASS 只在最后一条 MMA 发射之后才放置触发的原因。

### 9.6.2 `tcgen05.relinquish_alloc_permit` (951)

W28（explicit.cu:512-514）是 `release_allocation_lock()`（`:784`；`tmem_allocator_sm100.hpp:172-179`）。它承诺本 CTA 不会再进行任何 TMEM 分配，这让一个后继 CTA 得以进入该 SM 的 TMEM 的分配阶段（NVIDIA 内部；Colfax 的 TMEM 教程对它的解读相同，并带有一个保留性说法："apparently, this is a guarantee that the CTA will not perform any further TMEM allocation, allowing future CTAs to queue up for the same SM"）。relinquish 所释放的资源是 Virtual Resource Counter（VRC），即与 TMEM 分配相关联的启动/调度资源：CWD（CTA scheduler）在调度一个 CTA 之前会检查它，CTA 从启动起就持有它，而显式的释放让另一个 CTA 得以继续（NVIDIA 内部）。CUTLASS 有意把它放在解除分配*之前*（`:783`："Release the right to allocate before deallocations so that the next CTA can rasterize"）：一旦 VRC 被释放，一个后继 CTA 就可以在本 CTA 排空期间被调度到该 SM 上，开始其分配尝试，并在各列一被释放时即获准进入（NVIDIA 内部：这就是释放许可所换来的，而不是一直串行化到本 CTA 退出；它既不是排序原语，也不是有保证的 scheduler 交接，所以资源可用性和正常的 CTA 调度仍然适用。各列释放后的获准进入是 ISA 的 `tcgen05.alloc` 阻塞规则；至于这条指令本身，ISA 只说 CTA 放弃其分配的权利。在每个 SM 一个 230400 字节 CTA 的情况下，笔记 G.3，本 kernel 没有任何 CTA 会处于那种境地）。它是 `.cta_group::2.sync.aligned` 的，所以两个 CTA 的 warp 0 都执行它。ptxas 把它下沉为 `UVIRTCOUNT.DEALLOC.SMPOOL 0x80`（`UVIRTCOUNT 128`，NVIDIA 内部），并把它放在第 950 行的 `PREEXIT` *之前*，相对于 PTX 交换了二者的次序（笔记 D.2，D.6 第 9 项）。附近那条由 `ELECT` 守卫、写入保留共享内存的 `STS.U8` 不属于这次下沉（NVIDIA 内部：relinquish 是 warp 集体的，位于任何被选中单线程区域之外，而那条 `STS.U8` 是一次普通的共享内存字节存储；其来源未查明）。

### 9.6.3 leader 的累加器尾段 (952-962)

`PipelineAsync` 的 `producer_tail` 是 `for (int count = 0; count < Stages; ++count) { producer_acquire(state); ++state; }`（`sm90_pipeline.hpp:1129-1134`），经由 `PipelineUmmaAsync::producer_tail`（`sm100_pipeline.hpp:219-221`）到达：再做四次阻塞的 acquire，它们从循环停下的地方**继续** producer 状态，每次之后各跟着一次推进。

第 4 章，第 4.8.4 节的 `{idx, phase}` 对与笔记 7.6 步骤 3 相符，而被追踪的那次运行对每个 cluster 的 tile 数 `T` 都在其 `TAIL_ACC` 记录中恰好记录了这些对（笔记 G.1）。按照第 2 章，第 2.8.3 节所引用的 ISA 规则，`try_wait.parity` "valid only for" 当前未完成的 phase 或紧邻其前的那一个，而一个 parity 位不可能指代任何其他 phase；因此对于某些 `T`，固定的 parity 会在某个 `acc_empty[s]` 的两个可测试 phase 中指到错误的那一个，于是那个 wait 要么立刻返回（较早的 phase 已经完成），要么等待一个永远不会完成的 phase（第 4 章，第 4.8.4 节：“挂起或提前退出”）。该循环是 `#pragma unroll` 的（953），使得每一步的地址与 parity 更新都是直线代码（SASS：四个展开的 wait，中间夹着索引回绕/phase-XOR；baseline 的位于 `0x3cd0-0x3f20`，笔记 7.2，explicit kernel 的位于 `trace_out_v2/sass_explicit.txt` 的 `0x5680`、`0x5710`、`0x57a0`、`0x5850`）。

### 9.6.4 解除分配握手 (963-970) 及其证明

该屏障是位于 `OFF_TMEM_DEALLOC = 336`（explicit.cu:150）的 `tmem_dealloc`，每个 CTA 一个，由 warp 0 被选中的 lane 以计数 `kTmemDeallocCount = 32` 初始化（explicit.cu:185、1165-1166；CUTLASS 的 `tmem_deallocation_result_barrier.init(NumMMAThreads)`，`sm100_gemm_tma_warpspecialized.hpp:565`）。它镜像了 `:794-796`，即 `arrive(mma_peer_cta_rank, not is_mma_leader_cta); wait(dealloc_barrier_phase); arrive(mma_peer_cta_rank, is_mma_leader_cta)` 这三条语句，其 `:793` 处的注释写道 "Leader does wait + arrive, follower does arrive + wait"。这次握手满足 PTX ISA 的两条规则（`tcgen05.dealloc`）。第一条："If .cta_group::2 is specified, issuing warp and peer CTA warp must synchronize Tensor Memory accesses before attempting to collectively deallocate the Tensor Memory, and tcgen05.dealloc may block to collectively performs the deallocation with the other peer CTA's warp" [sic]。第二条："All of the Tensor Memory that was allocated using tcgen05.alloc instruction in a kernel, must be explicitly deallocated using tcgen05.dealloc before the kernel exits." 下面的断言 1 和断言 2 确立了第一条；第二条则是两个 CTA 的 warp 0 为何非到达第 971 行不可的原因：

```
   PEER (odd rank), warp 0, all 32 lanes                      LEADER (even rank), warp 0, all 32 lanes
   ----------------------------------------                   ----------------------------------------
   964: mapa + mbarrier.arrive.shared::cluster on               (952-962: 4 tail waits on own acc_empty[])
        the LEADER's tmem_dealloc  (32 arrivals -> completes
        phase 0 of the leader's barrier)
   965: wait parity 0 on OWN tmem_dealloc  ..................  965: wait parity 0 on OWN tmem_dealloc
        (blocks: nobody has arrived yet)                             (passes once the peer's 32 arrivals landed)
                                                               970: mapa + mbarrier.arrive.shared::cluster on the
                                                                    PEER's tmem_dealloc (32 arrivals -> phase 0 completes)
        (unblocks)                                             971: tcgen05.dealloc.cta_group::2  T, 512
   971: tcgen05.dealloc.cta_group::2  T, 512
```

`mbar_arrive_remote`（W10，explicit.cu:357-367）是来自 `ClusterBarrier::arrive(cta_id, pred)`（`barrier.h:486-499`）的 `mapa.shared::cluster.u32 rem, addr, cta_id; mbarrier.arrive.shared::cluster.b64 _, [rem];`，其谓词表达为 C++ 的 `if (!is_leader)` / `if (is_leader)`。`peer = rank ^ 1` 是 `mapa` 的目标。每个 lane 都执行它，这就是计数为 32 且不使用选举的原因。两个 wait 都使用 parity 0，因为每个屏障恰好只使用一次：它从 phase 0 开始，而 `try_wait.parity(0)` 在 32 次到达完成该 phase 时完成。

*断言 1：peer 不可能在 leader 完成其累加器尾段之前解除分配。* peer 的 `tcgen05.dealloc`（971）在程序顺序上位于其 wait（965）之后。该 wait 只有在 peer 屏障的 phase 0 完成时才完成，而这需要 32 次到达。在 peer 的 `tmem_dealloc` 上曾经发生的唯一到达是 leader 在第 970 行的到达（peer 到达的是 *leader 的* 屏障，而不是它自己的），而 leader 的第 970 行在程序顺序上位于其尾段（952-962）之后。因此 peer 的解除分配 happens-after leader 的尾段完成。而尾段又只有在两个 CTA 的 epilogue 为最后四个 tile 的每个 stage 到达了 256 次之后才完成，并且每个 epilogue 线程只在 `tcgen05.wait::ld` 之后才到达（第 10 章，explicit.cu:1038-1039），所以到那时两个 CTA 都没有任何未完成的 TMEM 读取。peer 的 warp 0 自己无法知道这一点：它从不等待 `acc_empty`（只有 leader 等待，explicit.cu:915-918），所以它不知道*它自己的* epilogue warp 是否已经完成了对最后一个 tile 的 `tcgen05.ld`；握手把 leader 的尾段知识借给了它（每个 stage 的 256 次到达包含了 peer 的 128 次）。没有它，peer 就可能在它自己的 TMEM 读取仍在途时执行 `tcgen05.dealloc.cta_group::2`：这违反了上面的第一条 ISA 规则，也是一次 use-after-free，而这是未定义的（NVIDIA 内部；ISA 没有规定其后果），还叠加在已退出 CTA 的冒险之上。更早地解除分配会释放 peer 自己的 epilogue 可能仍在加载的那些列。

*断言 2：leader 不得在 peer 到达之前解除分配。* 被保护的规则，用一行话说：`tcgen05.dealloc.cta_group::2.sync.aligned` 是一条两个 CTA 的 warp 0 都必须执行的 pair 集体指令（"The 2 warps from participating CTAs have the same logical warp ID"，`tmem_allocator_sm100.hpp:154`；ISA：第一个 warp "could block until the the second warp in the Peer CTA also performs the operation"，PTX ISA 9.4，9.7.18.5，表 55，引用于第 9.5.5 节第 1 项）。所以在 peer 的 warp 0 离开 tile 循环并发射它那一半之前，解除分配无法进行。peer 的到达（964）是告诉 leader 这一点的唯一消息；它是 peer 的声明“我已离开 tile 循环并执行了 `relinquish`；我已处于解除分配点”。没有 leader 的 wait，leader 就可能在 peer 的 warp 0 仍在消费 CLC 响应时执行它那一半的集体解除分配，而 ISA 允许这样（leader 的 warp 随后可能在 `tcgen05.dealloc` 中阻塞，直到 peer 发射它那一半："could block"，表 55；是否阻塞未作规定，表 57）；这次 wait 所保证的是：在 leader 能够退出之前，peer 在 leader 屏障上的远程 arrive 已被吸收（笔记 B.6；第 9.6.5 节 (ii)）。在 SASS 中，这条指令在两个 CTA 中都是 `UTCATOMSWS.AND`，其前有两次由 ptxas 插入的、针对保留的那些字的一致性检查，其失败路径会触发自陷（`BPT.TRAP`，笔记 D.2；explicit kernel 带有同样的两个桩，笔记 G.3）：它们检验被释放的单元是否记录在分配所设置的那些字中，即所有权（CUTLASS："Release the TMEM after checking that the CTA issuing the free does indeed own the corresponding slices"，`tmem_allocator_sm100.hpp:149`），而不是 pair 的顺序。这种两阶段的形式（peer 宣告，leader 确认）给了每一方恰好它所缺少的那一个事实：leader 缺少的是“peer 已就绪”，peer 缺少的是“TMEM 已静默”。每个 CTA 一个屏障、各一个 phase，是承载这两个事实的最低限度。

为什么 leader 的 wait 不会使 peer 的 arrive 死锁：peer 的 arrive（964）是无条件的，并且先于它自己的 wait（965），所以 peer 从不会在向 leader 发信号之前就等待它。依赖图是无环的：peer-arrive -> leader-wait -> leader-arrive -> peer-wait。

### 9.6.5 两个 CTA 中的 `tcgen05.dealloc` (971)

W27（explicit.cu:504-511）是来自 `Allocator2Sm::free`（`tmem_allocator_sm100.hpp:157-170`）的 `tcgen05.dealloc.cta_group::2.sync.aligned.b32 T, 512;`，由两个 CTA 的整个 warp 0 以基址 `T` 和与分配时相同的列数（`kTmemColumns`）执行。SASS：`UTCATOMSWS.AND URZ, UR4`（`0x5ae0`；baseline 中为 `UR5`），其中 `UR4 = ~(0xffff << ((T & 0xffff) >> 5))`，即 16 个各 32 列的分配单元被清除（baseline 见笔记 D.2；笔记 G.3 与原始转储：explicit kernel 的 40 条指令的 dealloc 序列与 baseline 的一致，差别仅在寄存器名、指令顺序和两处以 `MOV` 代替 `IMAD.U32` 的替换）。在此之后该 warp 从 `mma_role` 的末尾落出并退出。这次退出是安全的，因为 (i) 9.6.3 的尾段已经吸收了所有可能以 leader 的 `acc_empty` 屏障为目标的远程到达，并且 (ii) 握手已经吸收了每个 `tmem_dealloc` 上的 32 次远程到达（一个 phase）（笔记 7.10，尾段 3 和尾段 4）。该 CTA 的其他尾段（warp 2 的八次 mainloop wait，第 8 章；rank 0 的 warp 1 的两次 CLC wait，第 7 章）独立于 warp 0。

---

## 9.7 算例

### 9.7.1 描述符

*Rank 0，stage s = 3，k-block kb = 2。* `smem_base = 0x0000_0400`。

```
   A start (bytes) = 0x400 + 33792 + 16384 x 3 + 32 x 2 = 1024 + 33792 + 49152 + 64 = 84032 = 0x14840
   >> 4 = 0x1484 ;  & 0x3FFF = 0x1484 ;  lo_a = 0x10000 | 0x1484 = 0x11484          (= 0x10880 + 1024 x 3 + 2 x 2)
   desc_a = 0x40004040_00011484

   B start (bytes) = 0x400 + 164864 + 8192 x 3 + 32 x 2 = 1024 + 164864 + 24576 + 64 = 190528 = 0x2E840
   >> 4 = 0x2E84 ;  lo_b = 0x12E84                                                  (= 0x12880 + 512 x 3 + 4)
   desc_b = 0x40004040_00012E84
```

这两个值正是文件自己的注释所预测的值（explicit.cu:170）。trace 实测到每个 stage 为 `+0x400` (A) / `+0x200` (B)，每个 k-block 为 `+2`，起点为 `0x...10880` / `0x...12880`（笔记 D.1 第 D7 行）。

*Rank 3，stage s = 7，k-block kb = 1。* `smem_base = 0x0300_0400`。

```
   A start (bytes) = 0x03000400 + 0x8400 + 7 x 0x4000 + 0x20 = 0x0302_4820
   >> 4 = 0x0030_2482 ;  & 0x3FFF = 0x2482   <- the rank (0x30_0000 after the shift) is gone
   lo_a = 0x12482 ;  desc_a = 0x40004040_00012482

   B start (bytes) = 0x03000400 + 0x28400 + 7 x 0x2000 + 0x20 = 0x0303_6820
   >> 4 = 0x0030_3682 ;  & 0x3FFF = 0x3682
   lo_b = 0x13682 ;  desc_b = 0x40004040_00013682      (0x3682 = 13954; the largest value for kb = 3 is 0x3686 = 13958 < 16384)
```

rank 3 从不发射 MMA（它是 peer），而这正是要点所在：如果它发射了，或者硬件把 leader 的描述符应用于 rank 3 的 smem，那么 CTA 相对的 14 位起始地址就是 leader 为同一个 `(s, kb)` 所计算出的同一个数。

### 9.7.2 K 的顺序

tile 内编号 `n = 4 q + kb`（0..511）的 MMA 消费 8192 宽归约中的 `k = 64 q + 16 kb .. +16`：q-major、kb-minor，因为 smem 环每个 stage 交付一个 64 宽的 k-tile，而一个 stage 的四条 16 宽指令在该 stage 被释放之前背靠背地发射。整个 K 被覆盖一次：`64 x 128 = 8192`。每个 pair 每个 tile：512 条 MMA = 128 个 k-tile x 4，其中 1 条覆写、511 条累加。每条 MMA 的 FLOP：`2 x 256 x 128 x 16 = 1,048,576 = 2^20`；每个 pair tile：`512 x 2^20 = 2^29 = 536,870,912 = 2 x 256 x 128 x 8192`。一个 256 x 256 的 cluster 工作 tile 容纳*两个* pair tile（每个 MMA pair 一个，第 9.3.1 节），所以每个 cluster 每个 tile 两个 leader 发射 `2 x 512 = 1024` 条 MMA 和 `2 x 129 = 258` 次 commit（各为 128 次 mainloop + 1 次累加器），而一个 cluster tile 是 `2 x 2^29 = 2^30` FLOP。在 1024 个 cluster tile 上：`1024 x 2 x 512 = 1,048,576 = 2^20` 条指令和 `2^20 x 2^20 = 2^40 = 1,099,511,627,776 = 2 x 8192^3` FLOP（笔记第 9 节，操作台账："tcgen05.mma | 512 per pair per tile (1 clear + 511 accumulate) | 1048576"）。若每个 cluster tile 计一个 pair，会得到 2^39，那是 8192^3 而不是这个 GEMM。与实测交叉核对：`2^40 / 0.889354 ms = 1.236 x 10^15 FLOP/s`，这就是 harness 的 `GFLOPS` 行（五次 explicit 运行中为 1.23498e6-1.23687e6，笔记 G.2；笔记 D.4 指出 harness 精确地按 `GFLOPS = 2 x 8192^3 / Avg runtime` 计算）。台账与计时相符。

---

## 9.8 ptxas 生成的热循环（笔记 G.3，实测）

explicit kernel 的 k-tile 循环是 47 条 SASS 指令（`0x51f0-0x54d0`），而 baseline 是 44 条。其中三十七条是 uniform 数据通路（以 `U` 为前缀的）指令：12 条 `UMOV`、9 条 `UIADD3`、5 条 `ULEA`、4 条 `UTCHMMA.2CTA`、2 条 `UISETP.NE`、1 条 `UISETP.GT`、2 条 `USEL`、1 条 `UPLOP3`、1 条 `UTCBAR.2CTA.MULTICAST`。其余 10 条（2 条 `SYNCS.PHASECHK.TRANS64.TRYWAIT`、2 条 `SHF`、1 条 `LOP3`、2 条 `PLOP3`、3 条分支）是向量、谓词或控制指令。它们同样每个 warp 执行一次，因为 warp 是收敛的且每个操作数都是 warp-uniform 的，只不过是在向量 pipe 和谓词 pipe 上执行（`TRYWAIT` 从一个向量寄存器取其 parity，笔记 D.2）。比 baseline 多出的 +3 是：两条只在 baseline 中被 ptxas 外提到循环之外的 `UMOV`、一条用于编译期掩码 0xF 的 `UMOV`（baseline 在运行时把它算到一个寄存器里）和一条每 k-tile 重新计算 `empty[rs]` 地址的 `ULEA`，再减去一个循环计数器（第 12 章，第 12.5.1 节逐字列出了循环体；笔记 G.3）。其形状，按指令类别逐一给出：

```
   top:   SHF (parity -> bit 31) ; TRYWAIT full[rs] (only if the peek failed: @P0 BRA skips this inline wait; @!P0 BRA enters the out-of-line NANOSLEEP loop)
          UIADD3/USEL/UISETP: cons_idx++, wrap, flip bit ; LOP3 (vector): phase ^= flip ; UISETP.GT/UIADD3: k-- and the loop test
          @P2 TRYWAIT full[next]           <- the peek, predicated off on the last k-tile
          ULEA UR10 = UR30 + (UR8 << 10)  (lo_a = desc_lo_a + 1024 rs) ; ULEA UR12 = UR31 + (UR8 << 9) (lo_b)
          UMOV 0x40004040 (x8: the high words) ; UMOV 0x0 / UMOV 0x10200010 (UR28 = 0: tmem[UR28] is the sparse-metadata TMEM operand, zero for a dense MMA (NVIDIA internal), no PTX counterpart in the dense form, Section 12.2; idesc)
          UTCHMMA.2CTA gdesc[A], gdesc[B], tmem[T + 128 st], ..., idesc, UP0      <- kb = 0, UP0 = scale_c
          UIADD3 +0x2 ; UTCHMMA ... UPT                                           <- kb = 1
          UIADD3 +0x4 ; UTCHMMA ... UPT                                           <- kb = 2
          UIADD3 +0x6 ; UTCHMMA ... UPT                                           <- kb = 3
          ULEA empty[rs] ; UTCBAR.2CTA.MULTICAST [empty[rs]], URZ, UR37 (= 0xF)   <- the commit, right after the 4th MMA
          BRA top
```

为什么这一切都不耗费时间：四条 `tcgen05.mma` 是异步的，所以 warp 在 tensor core 仍在执行第 `q` 次迭代的 MMA 时就发射第 `q + 1` 次迭代的 47 条指令。预算至多为每 k-tile 427 个周期、每条 256 x 128 x 16 MMA 约 107 个周期（由实测的 explicit 中位数 0.889354 ms，笔记 G.2，摊到 32-tile 的关键路径上得出，笔记 D.3；笔记 D.4 以同样的方式为 baseline 推导出 426 / 106.5）。两者都是上界：计时区间还包含主机端工作（笔记 D.4），而 1965 MHz 这一最高 SM 时钟是从计时循环前后的时钟采样推断出来的（第 3 章，第 3.9.2 节；第 12 章，第 12.5.5 节）。这些数字针对的是 32-tile 的关键路径 cluster；平均每个 cluster 处理 31.03 个 tile，因而每 k-tile 多出 3.1 % 的时间（第 12.5.5 节）。即使完全暴露、按每周期一次发射计，这 47 条指令也会是 k-tile 预算的 `47 / 427 = 11%`，而且它们与 tensor 工作相重叠，而不是加在其上（我的分析）。

有两件事限定了 warp 能领先 tensor core 跑多远，“被隐藏”正是建立在这一点上：(a) `full[s]` 等待，因为 producer 只在某个 stage 的 `empty[s]` commit 之后才重新填充它，所以 warp 最多只能领先 tensor core 已完成的 k-tile 8 个 k-tile；(b) tensor core 中已发射但尚未执行的 `tcgen05.mma` 的有限队列，当它满时会使 `UTCHMMA` 的发射停顿（假设；未实测）。因此 warp 的时间花在因数据或 MMA 背压而产生的停顿上，而不是花在它自己的 47 条指令上。循环中唯一暴露出来的延迟是一次*失败的* peek（TMA 数据尚未落地，即 producer 落后了，而 8 个 stage 的环的大小正是为避免这种情况而定的，第 3 章）和每个 tile 的 `acc_empty` 等待（epilogue 落后超过了三个 tile）。先 peek 再 wait 的形状之所以重要，是因为常见情况必须连阻塞循环都不执行：第 9.5.3 节的 `@P0 BRA` 在前一次迭代的 peek 成功时总是跳过顶部的 `TRYWAIT`（笔记 G.3；baseline 中等价的分支是 `@P1 BRA 0x38e0`，笔记 D.2）。

k 循环之外的累加器 empty 等待与 commit 块是 15 条指令，而 baseline 是 18 条，因为 `k_tiles` 一直留在一个 uniform 寄存器中，而不是每个 tile 都从参数 bank 重新读取（笔记 G.3）。

---

## 9.9 设计注记：哪些要带到其他 kernel

- **一个发射线程，零寄存器流量。** 引擎 warp 从不触碰操作数数据：输入经由 TMA 到达 smem（第 8 章），指令通过描述符读取它们，输出则落在 TMEM 中。这个 warp 的全部工作就是地址、谓词和屏障，所以它几乎所有的状态都在 uniform 寄存器中（`full[s]` 的 phase 和 parity 位于向量寄存器 `R9`/`R2` 中，第 12.5.1 节），而它的循环是 47 条指令，其中 37 条是 uniform 数据通路指令。（该 kernel 的 `REG 26`，相对于 baseline 的 68，是在别处决定的：由 epilogue 的最高活跃向量寄存器 `R23` 决定，即 16 个 `LDTM` 结果加上四个存储基址寄存器，笔记 G.3。）
- **凡是指令所配对的，都要配对。** `cta_group::2` 把分配、MMA、commit、relinquish 和解除分配绑定到两个 CTA 上，也绑定到完全相同的 smem layout 上。代码中的每一处不对称（leader 发射，peer 跟踪）都由两端的一个对称集体操作（alloc、dealloc）和中间的 multicast（commit）来匹配，使 peer 永远不会缺少它所需要的事实（`T`、`acc_full`、终止符）。
- **向每一个写入者释放，而不只是向读取者释放。** 0xF 掩码与计数 2 来自于问“谁可以覆写这个 stage？”而非“谁读过它？”。同一个问题决定了任何由 multicast 供给的流水线的掩码。
- **就地累加；在第一条指令上覆写。** enable-input-D 谓词取代了一遍清零；TMEM 中深度为 4 的累加器环取代了寄存器累加器，并让 epilogue 可以落后三个 tile 运行。
- **尾段延续状态。** 每条 producer 侧的流水线在末尾都还需要额外的 `Stages` 次 acquire，并带着*延续下来的* `{index, phase}`；每当尾段之前的 acquire 计数不是 `Stages` 的倍数时，固定的 parity 就会挂起或提前通过（第 4 章，第 4.8.4 节：累加器环和 CLC 环每个 tile 一次 acquire，所以四个累加器 parity 只在 `T % 4 == 0` 时才重合；mainloop 环每个 tile 128 次，所以它的八个总是重合）。
- **先 peek，再 wait。** 一次在数据依赖所允许的最早时刻发射的非阻塞 `try_wait`，加上只在 peek 失败时才做的阻塞 wait，使常见路径上没有 `NANOSLEEP` 循环，而又不改变协议。

---

# 第 10 章 收尾者：epilogue_role（Warp 4-7）

## 10.1 epilogue 位于何处

mainloop（第 8-9 章）把每个已完成的 128 x 128 fp32 CTA tile 留在张量内存中：stage `st` 是全部 128 个 lane 的 TMEM 列 `[128 st, 128 st + 128)`，lane = 输出行（笔记，第 6.3 节）。只有 `tcgen05.ld` 能把 TMEM 数据搬入寄存器。TMEM 仅有的其他读取者是 `tcgen05.mma` 自身——它在 `scale_c = 1` 时读取累加器，并且可以从 TMEM 取其 A 操作数——以及 `tcgen05.shift`，它只在 TMEM 内部移动行；不存在 TMEM 到 smem 或 TMEM 到全局内存的路径，而 `tcgen05.cp` 只做 smem -> TMEM 的拷贝。TMA 存储路径是一个设计选择，而不是硬件上的必然。我的分析：寄存器 -> `st.global` 的路径是存在的，但对于 M-major 的 D，每个线程在一个 subtile 中的 16 个值位于 16 个不同的列中，彼此相隔 32 KB。那将是每线程 128 条标量 `st.global.b32`、每 tile 16384 条，每条 warp 指令覆盖一个 128 字节的列段，而且会放弃 TMA 的地址生成与异步完成。baseline 选择了 TMA，本 kernel 镜像了它。因此 epilogue 是一条四个 stage 的传送带：

```
  TMEM stage st                 registers                 smem stage (e % 4)              global D (M-major)
  128 lanes x 128 cols   ---->  16 fp32 per thread  ---->  8192 B, Swizzle<2,5,2>  ---->  four 2048 B boxes per subtile
  (tcgen05.ld x16, per warp     (FMUL + FFMA,            (16 x st.shared.b32,            (cp.async.bulk.tensor.3d,
   32 lanes x 16 cols)           alpha, beta)              fence.proxy.async)              warp 4, bulk_group)
```

一个 tile 按 8 个 128 行 x 16 列的 subtile 处理（`kEpiSubtiles = 8`、`kEpiTileN = 16`，explicit.cu:112-113），因为这正是 CUTLASS builder 选定的 `EpilogueTile (128,16)`（在 10.5.0 中推导）。每个 subtile 是每 warp 一条 `tcgen05.ld`、每线程 16 个值、一个 8192 字节的 smem stage、四个 TMA box。四个 epilogue warp 运行相同的代码；warp 4 还额外发射 TMA 存储。

与 MMA warp 的交接是一个由 mbarrier 构成的 4-stage 环（笔记，第 7.2 节）。`acc_full[st]`（计数 1，由 leader 的带 pair 掩码的 `tcgen05.commit` 完成）告诉 epilogue 该 stage 已就绪。`acc_empty[st]`（计数 256 = 2 个 CTA x 128 个 epilogue 线程，位于 *leader 的* 屏障上）告诉 leader 的 MMA warp 该 stage 可以被覆写。CLC 响应每 tile 消费一次，在当前 tile 被存储*之前*（笔记，第 7.7 节），所以 epilogue 像 MMA warp 一样，总是提前一个 tile 就知道自己的下一个 tile。

---

## 10.2 横幅注释 (explicit.cu:977-994)

```
977: ///////////////////////////////////////////////////////////////////////////////////////////////////
978: // 9. [PROD] tcgen05 TMA LAYOUT  Epilogue consumers: warps 4-7 of every CTA (Section 7.7, B.3.3; kernel :869-955; CollectiveEpilogue::store)
979: //   Warp w = warp_idx - 4 owns TMEM lanes (= accumulator rows) [32 w, 32 w + 32); lane l owns row 32 w + l of the CTA's 128 x 128 tile.
980: //   Per tile: wait acc_full[st] (count 1: the leader's commit); then 8 subtiles e = 0..7 of 16 columns:
981: //     1. tcgen05.ld.32x32b.x16 from T + ((32 w) << 16) + 128 st + 16 e: lane l gets D[row 32w + l][16 e .. 16 e + 15] as 16 fp32 registers
982: //        (example: w = 2, st = 1, e = 3 -> tmem address 0x4000B0 = lane 64, column 176)
983: //     2. d = fmaf(beta, 0, alpha * acc) per register (the fixed run's beta = 0 keeps the C term at literal 0; the FMUL + FFMA pair is the
984: //        baseline's own arithmetic, D.7); after the 8th load: tcgen05.wait::ld and one arrive per thread on the pair leader's acc_empty[st]
985: //     3. 16 x st.shared.b32 into C/D stage (e % 4) with the Swizzle<2,5,2> pattern below; fence.proxy.async; bar.sync 1,128
986: //     4. warp 4 issues four TMA stores of (32 m, 16 n) boxes from stage (e % 4) + 2048 i to D coordinates (128 tm + 32 i, 128 tn + 16 e, 0),
987: //        commit_group, wait_group.read 1 (at most one group still reading smem, so stage (e % 4) is free again 4 subtiles later); bar.sync 1,128
988: //   smem byte offset of value j (column 16 e + j) of lane l in warp w:
989: //        OFF_SMEM_CD + 8192 (e % 4) + 2048 w + 128 j + 4 ((l & 7) + 8 (((l >> 3) & 3) ^ (j & 3)))
990: //     = st_off[j & 3] + 512 (j >> 2) + 8192 (e & 3)    with  st_off[c] = 512 + 2048 w + 4 (l & 7) + 128 c + 32 (((l >> 3) & 3) ^ c)
991: //   Examples: w = 0, l = 0: st_off = {512, 672, 832, 992};  w = 0, l = 9: {548, 644, 868, 964};  w = 3, l = 31: {6780, 6876, 6972, 7068};
992: //             w = 0, l = 9, j = 5, e = 6 -> 644 + 512 + 16384 = 17540 (0x4484) = stage 2, the TMA store box i = 0 of that stage.
993: //   D box example: tile (5, 9), e = 3, i = 2 -> coordinates (704, 1200, 0), source stage 3 + 4096 B.
994: ///////////////////////////////////////////////////////////////////////////////////////////////////
```

把它当作后面那个函数的规范来读。第 978 行点出了所涉及的三个硬件族：`tcgen05` 对应 TMEM 加载以及先于累加器释放的 `tcgen05.wait::ld`，`TMA` 对应存储，`LAYOUT` 对应 swizzle 算术。它还点出了该函数所镜像的两处 CUTLASS 源码：kernel 的 epilogue 分支（`sm100_gemm_tma_warpspecialized.hpp:869-955`，即每 tile 调用一次 `collective_epilogue.store(...)` 的 `else if (is_participant.epilogue)` 块）以及 `CollectiveEpilogue::store` 本身（`sm100_epilogue_tma_warpspecialized.hpp:573-956`）。第 979 行是所有权陈述，其余一切都由它推出：线程 `(w, l)` 拥有 CTA tile 的输出行 `32 w + l`，覆盖全部 128 列。第 981-987 行是一个 subtile 的四个步骤。第 988-993 行以两种等价形式给出 smem 公式和四个算例，外加一个 D-box 坐标示例（第 993 行）；第 10.5 节推导该公式并重新计算这四个算例，10.13 则演算这个 D-box 示例。

有两个细节容易被忽略。epilogue 对 stage `st` 的等待是*每 tile 一次*，而不是每 subtile 一次（第 980 行，"Per tile: wait acc_full[st]"）。步骤 2 中的释放发生在 "after the 8th load"，也就是在 subtile 7 内部、但在该 subtile 的算术运算与存储*之前*。两者都是忠实于 baseline 的事实（笔记，第 7.7 节步骤 1 和 3）；10.8 给出正确性论证。

---

## 10.3 设置：TMEM 地址、warp 与 lane 身份 (explicit.cu:995-1006)

```
995: __device__ __forceinline__ void epilogue_role(ExplicitGemmParams const& params, char* smem, uint32_t smem_base, uint32_t x, uint32_t y,
996:                                               uint32_t rank, bool is_leader, int warp_idx, uint32_t lane, float alpha, float beta, int tm, int tn) {
997:   named_bar_sync(kBarTmemAlloc, kTmemAllocBarThreads);                                          // bar.sync 6, 160: wait for the TMEM address (:871)
998:   uint32_t const T = *reinterpret_cast<uint32_t const*>(smem + OFF_TMEM_BASE_PTR);              // (:872)
```

（第 999-1003 行是一条仅限 `[TRACE]` 的 `T` 记录，在性能二进制中被编译掉；`rank` 与 `is_leader` 在那里被 `(void)` 强制转换消费掉，此外不被该角色使用）

```
1004:   uint32_t const w = static_cast<uint32_t>(warp_idx) - 4u;                                      // epilogue warp 0..3 (thread_idx / 32)
1005:   uint32_t const l = lane;
1006:   uint32_t const tmem_lane_off = (32u * w) << 16;                                               // TMEM lanes [32 w, 32 w + 32): row 32 w + l of the CTA tile (0, 0x200000, 0x400000, 0x600000)
```

**参数列表 (995-996)。** 该角色接收共享内存的两种视图：`smem`（来自 `extern __shared__ char smem[]` 的通用 `char*`）用于 C++ 解引用，以及 `smem_base`（它的 32 位 `shared::cta` 地址）用于 asm 操作数。这就是文件头部的地址空间规则 (explicit.cu:22-23)，10.9 会再次回顾。`x, y` 送入 `clc_consume`（CLC 的 tile 映射需要该 CTA 在 cluster 中的位置，第 7 章）。`alpha, beta` 在 kernel 入口处读取一次（explicit.cu:1110-1111，`float const alpha = params.alpha; float const beta = params.beta;`），然后作为普通的浮点数向下传递。`tm, tn` 是启动 tile。

**`bar.sync 6, 160` (997)。** 屏障 id 6 是 `ReservedNamedBarriers::TmemAllocBarrier`，而 160 = 32 + 128 是 `NumMMAThreads + NumEpilogueThreads` (cutlass/arch/barrier.h:169-178; sm100_gemm_tma_warpspecialized.hpp:558)。producer 一侧在 `mma_role` 中（explicit.cu:885-888，引用于第 9 章，第 9.4 节）。

explicit.cu:884 上的注释写的是 "whole warp 0 of both CTAs of the pair, same dst offset"，而 `Allocator2Sm` 的文档写明 "Both CTAs _must_ provide the exact same dst_ptr for correctness" 以及 "The 2 warps from participating CTAs have the same logical warp ID" (cute/arch/tmem_allocator_sm100.hpp:124-133)。PTX ISA 关于该指令的效果说它 "dynamically allocates the specified number of columns in the Tensor Memory and writes the address of the allocated Tensor Memory into shared memory at the location specified by address operand dst"，关于 `.cta_group::2` 则说 "one warp from each of the peer CTAs must collectively perform the allocation and de-allocation" (PTX ISA 9.4, tcgen05.alloc)；它没有说每个 CTA 自己的 `dst` 是否都被写入，也没有说两个 `dst` 操作数必须一致。该 kernel 所依赖的契约是：在该指令之后，*每个* CTA 自己的槽位都持有该基址。

PTX ISA，`bar`/`barrier`："The barrier{.cta}.sync or barrier{.cta}.red or barrier{.cta}.arrive instruction guarantees that when the barrier completes, prior memory accesses requested by this thread are performed relative to all threads participating in the barrier"。因此 warp 0 的 `bar.arrive` 把槽位写入发布给在 id 6 上 `bar.sync` 的那 128 个线程。W18 的 `"memory"` clobber (explicit.cu:433) 阻止编译器把该加载外提到屏障之上。

为什么用命名屏障而不是 `__syncthreads()`：256 个线程中只有 160 个参与（warp 1、2 和 3 永远不会到达这一点：warp 3 从分发中贯穿而过并退出），而 `bar.sync 0` 则需要全部 256 个。为什么用 id 6 和 1（下文）而不是 0：id 0 归编译器所有，是 `__syncthreads()` 通常下沉到的那个屏障（NVIDIA 内部），正如第 6.5.2 节所述，而且 CUTLASS 保留了 id 1-7（`EpilogueBarrier = 1`、`TmemAllocBarrier = 6`），使得用户屏障——面向用户的 `NamedBarrier` 构造函数会把它们的 id 偏移 `ReservedNamedBarrierCount = FirstUserBarrier` (= 8)——不可能发生碰撞 (barrier.h:169-178, 201-207, 330-332)。

**`w`、`l` 与硬件象限规则 (1004-1006)。** 象限规则（第 2 章，第 2.3.3 节）就是 epilogue *为什么*是覆盖模 4 全部四个余数的四个连续 warp 的原因，也是 CUTLASS 的 epilogue 线程映射为什么是 `thread_idx = threadIdx.x % ThreadCount, warp_idx = thread_idx / NumThreadsPerWarp`、即 `% 128` 与 `/ 32` 的原因（sm100_epilogue_tma_warpspecialized.hpp:603-604；`ThreadCount = 128` 位于 :127）：把行分配给 warp 的任何其他方式都将无法寻址。

---

## 10.4 st_off[4] 表 (explicit.cu:1007-1014) 与四个基址

```
1007:   // Byte offsets (from the generic smem base) of this thread's 4 column classes inside an 8192-byte epilogue stage (B.4, D.6 item 1):
1008:   //   value j of subtile e (column j of the subtile's stage) -> OFF_SMEM_CD + 8192 (e % 4) + 2048 w + 128 j + 4 ((l & 7) + 8 (((l >> 3) & 3) ^ (j & 3)))
1009:   //   = st_off[j & 3] + 512 (j >> 2) + 8192 (e & 3)      (Swizzle<2,5,2> on byte offsets: bits [5,7) ^= bits [7,9))
1010:   uint32_t st_off[4];
1011:   #pragma unroll
1012:   for (uint32_t c = 0; c < 4u; ++c) {
1013:     st_off[c] = OFF_SMEM_CD + 2048u * w + 4u * (l & 7u) + 128u * c + 32u * (((l >> 3) & 3u) ^ c);
1014:   }
```

该线程预先计算四个字节偏移，每个*列类* `c = j & 3` 一个，因为 swizzle 项 `(((l >> 3) & 3) ^ (j & 3))` 只通过 `j & 3` 依赖于 `j`。其余依赖于 `j` 和 `e` 的部分，在循环展开之后都是加性常数：`512 (j >> 2)` 与 `8192 (e & 3)`。SASS 显示的正是这一点（笔记 G.3）：`STS [Rc + UR7 + imm]`，其中 `Rc` 是四个与 lane 相关的寄存器（`R20-R23`）之一，而 `imm = 0x200 + 0x80 c + 0x200 (j >> 2) + 0x2000 (e & 3)`。四个基址 `R20-R23` 位于 16 个 `LDTM` 目的寄存器（`R4-R19`）之上，并设定了 explicit kernel 的寄存器高水位线 `R23`（笔记 G.3）。baseline 使用它自己的一套四寄存器分解，把 `0x80 c` 这一项折叠进基址（笔记 G.3；D.2：`STS [Rbase + imm]`，其中 "Rbase one of four lane-dependent registers"），却需要 68 个寄存器。26 对 68 的差距就是 explicit kernel 丢弃的死状态：零填充的 C 片段（16 个向量寄存器）、`epi_load` 流水线状态、保存在向量寄存器中的 per-tile CLC 解码，以及 44 对 `MOV.SPILL`/`R2UR.FILL`（笔记 G.3，"Why 26 and not 68"）。第 10.5 节推导该公式；先看余下的设置行。

```
1015:   uint32_t const cd_base = smem_base + OFF_SMEM_CD;
1016:   uint32_t const acc_full_base = smem_base + OFF_ACC_FULL;
1017:   uint32_t const acc_empty_base = smem_base + OFF_ACC_EMPTY;
1018:   bool const issue_tma_store = (w == 0u);                                                        // (:740): warp 4, all 32 lanes, uniform operands, no election
1019:   uint32_t accc_idx = 0, accc_phase = 0;     // accumulator consumer state (4 stages)
1020:   uint32_t clc_idx = 0, clc_phase = 0, clc_count = 0;
1021:   int tm_next = tm, tn_next = tn;
1022:   uint32_t tile = 0;
1023:   bool valid;
```

- `cd_base` (1015) 是 C/D stage 的 *shared::cta* 地址（`OFF_SMEM_CD = 512`，explicit.cu:153），即 TMA 存储所需的 32 位形式。`st_off[]` 保存的是*相对通用指针的字节偏移*，因为 `st.shared` 写入是 C++ 解引用。同一块内存，两种地址形式，从不混用（10.9）。
- `acc_full_base = smem_base + 240`、`acc_empty_base = smem_base + 272` (explicit.cu:146-147)：各为四个 8 字节的 mbarrier，所以 stage `st` 位于 `+ 8 st`。计数：full 1，empty 256（explicit.cu:184；由 warp 5 在 prologue 中初始化，第 11 章）。
- `issue_tma_store = (w == 0u)` (1018) 镜像了 `bool issue_tma_store = warp_idx == 0;` (sm100_epilogue_tma_warpspecialized.hpp:740)：一个 *warp-uniform* 谓词，对 warp 4 的全部 32 个 lane 为真，对 warp 5-7 则无一为真。10.10.3 解释了为什么不需要 `elect.sync`。
- `accc_idx = 0, accc_phase = 0` (1019)：累加器 *consumer* 状态从 `{0, 0}` 开始（consumer 先等待 parity 0；producer 从 `{0, 1}` 开始，第 4 章）。它每 tile 推进一次（4 个 stage），所以 tile `t` 使用 stage `t % 4`、parity 为 `(t / 4) & 1`，与笔记第 7.7 节中的 `acc_stage = t % 4`、parity `(t >> 2) & 1` 完全一致。
- `clc_idx, clc_phase, clc_count` (1020)：CLC consumer 状态，两个槽位，在 `clc_consume` 内部推进（第 7 章）。每个 CTA 中，四个 epilogue warp 贡献其 192 次到达中的 4 x 32 = 128 次（32 个 MMA-warp lane + 32 个 producer-warp lane + 128 个 epilogue 线程）；在四个 CTA 上合计，这是 cluster 每个响应 800 = 32 + 4 x (32 + 32 + 128) 次到达中的 512 次（explicit.cu:183；单独的那个 32 是 CTA 0 的 scheduler warp）。
- `tm_next = tm, tn_next = tn` (1021)：如此初始化是为了当最后一个响应为“未取消”时，循环体末尾的无条件拷贝永远不会读到一个不确定的值（`clc_consume` 只在响应有效时才写入 `tm_next/tn_next`）。

---

## 10.5 推导 swizzled smem 公式

本节完成横幅注释第 988-992 行所声称的事情，从 CuTe 源码出发由内向外，并核对每一个数字。这里的内容在运行时没有任何部分被 kernel 执行，除了第 1013 行的四次 `st_off[c]` 计算（SASS 中的一条 `SHF`、一条 `LEA` 和六条 `LOP3`，`0x1f30-0x2000`；`512 + 128 c` 这部分被折叠进 `STS` 的立即数）；这个推导是你*知道*这些常数的方式。

### 10.5.0 为什么 subtile 是 (128, 16)：builder 的 EpilogueTile

八个 subtile、`.x16` 加载 shape、8192 字节的 stage 和 32 KB 的 C/D smem，全都源于一个编译期选择 `EpilogueTile = (128, 16)`（实测 `EpilogueTile (128:1,16:1)`，笔记 D.1 D1）。explicit kernel 把它硬编码为 `kEpiSubtiles = 8`、`kEpiTileN = 16`（explicit.cu:112-113）。它按如下方式从源码推导得出，在 `sm100_dense_compute_tile_shape_or_override` 中（sm100_builder.inl:980-1054，`EpilogueTileAuto` 分支），取 `CtaTileShape_MNK = (128, 128, 64)`、`ElementC = ElementD = float`、M-major 的 C 与 D：

```
  else if constexpr (is_same_v<EpilogueTileType, EpilogueTileAuto>) {
    constexpr int CtaM = size<0>(CtaTileShape_MNK{});
    constexpr int CtaN = size<1>(CtaTileShape_MNK{});
    constexpr int WarpM = size<0>(TmemWarpShape_MN{});
    constexpr int WarpN = size<1>(TmemWarpShape_MN{});
    constexpr int MaxBits = cute::max(sizeof_bits_v<ElementC>, sizeof_bits_v<ElementD>);

    constexpr int DpFull = 32; // tmem datapaths in 1 subpartition
    constexpr int M = cute::min(CtaM, DpFull * WarpM); // target 32dp tmem load
    constexpr int N_perf = [&]() constexpr { // Known subtile sizes tested for perf
```

`WarpM, WarpN` 来自 `sm100_tmem_warps<Is2SmMma, MmaTileShape_MNK>()` (sm100_builder.inl:1074-1081)：

```
sm100_tmem_warps() {
  if constexpr (Is2SmMma && size<0>(MmaTileShape_MNK{}) == 128) {
    return Shape<_2,_2>{};
  }
  else {
    return Shape<_4,_1>{};
  }
}
```

MMA tile 是 256 x 128 (2SM)，所以 `size<0>(MmaTileShape_MNK) == 128` 为假，shape 是 `(4, 1)`：`WarpM = 4`、`WarpN = 1`。因此 `M = min(128, 32 x 4) = 128`，"target 32dp tmem load"：每个 warp 一条 `tcgen05.ld` 覆盖它的 32 个数据通路（lane），四个 warp 覆盖 128 行。对于 N，builder 区分带与不带残差 (C) 加载的 epilogue；这里 `DisableSource = is_void_v<ElementC_>` 为假，因为 C 的类型是 `float`，无论 beta 的运行时取值为何：

```
      // Epilogues w/ residual load are more sensitive to smem allocation
      // Target optimal smem distribution between epilogue+mainloop based on datatype+tilesize
      else {
        if (MaxBits == 32) {
          return (CtaM > 64 && CtaN <= 128) ? 16 : 32;
        }
```

`MaxBits = max(32, 32) = 32`、`CtaM = 128 > 64`、`CtaN = 128 <= 128`，所以 `N_perf = 16`。然后是 `N_min_C` / `N_min_D` 下界与最终选择：

```
    constexpr int N_min_C = (DisableSource || detail::is_m_major<GmemStrideTypeC>()) ? 8 * WarpN
                              : (sizeof_bits_v<ElementC> == 6) ? 128 * WarpN // TMA store only supports SW128B for FP6 data type
                                                              : 128 / sizeof_bits_v<ElementC> * WarpN;
    constexpr int N_min_D = (detail::is_m_major<GmemStrideTypeD>()) ? 8 * WarpN
                              : (sizeof_bits_v<ElementD> == 6) ? 128 * WarpN // TMA store only supports SW128B for FP6 data type
                                                              : 128 / sizeof_bits_v<ElementD> * WarpN;
    constexpr int N_tmp = cute::min(CtaN, cute::max(N_perf, N_min_C, N_min_D));
    constexpr int N = CtaN % N_tmp == 0 ? N_tmp : CtaN;
```

对于 M-major 的 C 与 D，`N_min_C = N_min_D = 8 x WarpN = 8`（builder 没有给出这个 8 的理由，而且它不是 TMA 的规则：对于 M-major 的 map，N 是 `boxDim[1]`，`cuTensorMapEncodeTiled` 对它只要求非零且至多为 256，其余的 box 规则全都落在内层 M 范围 `boxDim[0]` 上；我的解读是，这个下界使 N 保持为 8 的倍数，正如 `sm100_smem_selector` 的 `static_assert(BLK_K0 % 8 == 0)` 所要求的那样，sm100_common.inl:87）。`N_tmp = min(128, max(16, 8, 8)) = 16`，而由于 `128 % 16 == 0`，`N = 16`。于是 tile 为

```
    auto tile_m = Layout<Int<M>>{};
    auto tile_n = Layout<Shape <Int<N / WarpN>,Int<        WarpN>>,
                        Stride<Int<         1>,Int<CtaN / WarpN>>>{};

    return make_tile(tile_m, coalesce(tile_n));
```

即 `tile_n = (16, 1):(1, 128)`，合并为 `16`：`EpilogueTile = (128, 16)`。

其后果在 explicit.cu 的第 9 节中全都可见：每 tile `128 / 16 = 8` 个 subtile；每线程每次加载 16 个 fp32，因此有 `tcgen05.ld.32x32b.x16` 和 `uint32_t r[16]`（第 1035 行）；一个 stage 为 `128 x 16 x 4 = 8192` 字节（`kEpiStageBytes`，explicit.cu:119）；再加上 10.10.4 的 4 个 stage，便是 230400 字节的 `SharedStorage` 中 `4 x 8192 = 32768` 字节的 C/D smem（笔记，第 6.1-6.2 节），其余是 mainloop 的 8 x 24576 = 196608 字节的 A/B stage，加上 512 字节的控制区和 512 字节的填充（4.1）。builder 自己的注释说明了为什么是 16 而不是 32："Epilogues w/ residual load are more sensitive to smem allocation. Target optimal smem distribution between epilogue+mainloop based on datatype+tilesize"（更小的 subtile 使 C/D 占用减半；认为该注释是这个意思，是我的解读）。反事实情形，出自同一段代码：若 C 为 void（`DisableSource`），则 `N_perf = ComputeElts / M = 4096 / 128 = 32`，于是 tile 会是 `(128, 32)`，4 个 subtile，`.x32` 加载且每线程 32 个寄存器，16384 字节的 stage，而且 `DelayTmaStore = is_void_v<ElementC_>` 也会随之开启（10.6）。

### 10.5.1 原子：Layout_MN_SW128_32B_Atom<float>

CUTLASS 的 epilogue builder 用 `sm100_get_epilogue_smem_swizzle_layout_atom` (sm100_builder.inl:70-97) 为 C 和 D 选择 smem layout 原子。对于 M-major (ColumnMajor) 的 D，它调用 `sm100_smem_selector<UMMA::Major::MN, Element, Int<128>, Int<16>>`，最后两个参数是从 `(128, 16)` tile 派生出来的类型 (sm100_builder.inl:83-86)，而该选择器有一个特殊情况 (sm100_common.inl:89-94)：

```
  if constexpr (major == cute::UMMA::Major::MN) {
    // Handle the special case for F32 NT kernels
    if constexpr ((sizeof(ElementType) == 4)) {
       static_assert(BLK_MN0 % size<0>(UMMA::Layout_MN_SW128_32B_Atom<ElementType>{}) == 0, "for mn-major tf32 operands, SW128_32B is the only available smem layout");
      return UMMA::Layout_MN_SW128_32B_Atom<ElementType>{};
    }
```

所以，对于 MN-major 顺序下的 4 字节元素，原子就是 `Layout_MN_SW128_32B_Atom<float>`，定义于 cute/atom/mma_traits_sm100.hpp:73-76：

```
using Layout_MN_SW128_32B_Atom_Bits = ComposedLayout<Swizzle<2,5,2>, smem_ptr_flag, Layout<Shape< _1024,_4>,Stride<_1, _1024>>>;

template <class Type>
using Layout_MN_SW128_32B_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_32B_Atom_Bits{}));
```

按位计，原子是 `(1024, 4):(1, 1024)`：四行各 1024 位 = 128 字节。upcast 到 32 位元素后，它是 `(32, 4):(1, 32)`：32 个 M 元素连续，4 个 N 列以 32 个元素 (128 B) 为 stride。实测的类型打印与之一致：`SmemLayoutAtomC/D = Sw<2,5,2> o smem_ptr[32b] o (32,4):(1,32)`（笔记 D.1 D1）。原子为 512 字节：4 列 x 128 字节。

该 swizzle 是作用于*字节*地址上的 `Swizzle<2,5,2>`，这是 CuTe 对附着在 `smem_ptr_flag` 上的 swizzle 所采用的约定。笔记 B.3.3 步骤 6 指出，移入以 float 为索引的 layout 之后，它变成作用于 float 偏移上的 `Swizzle<2,3,2>`，是同一个置换。`cute::Swizzle<BBits, MBase, SShift>` (cute/swizzle.hpp:55-79) 为：

```
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;
  ...
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
  using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0,num_shft))>;
  using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0,num_shft))>;
  ...
  apply(Offset const& offset)
  {
    return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});   // ZZZ ^= YYY
  }
```

取 `B = 2, M = 5, S = 2`：`bit_msk = 0b11`、`yyy_msk = 0b11 << 7 = 0x180`（第 7-8 位）、`zzz_msk = 0b11 << 5 = 0x60`（第 5-6 位），以及 `apply(o) = o ^ ((o & 0x180) >> 2)`：**字节偏移的位 [5,7) 与位 [7,9) 做 XOR**。该图案每 512 字节（4 行）重复一次，所以 stage 基址必须是 512 字节对齐的（笔记，第 6.2 节 `alignas(512)`；explicit.cu:163 `static_assert(OFF_SMEM_CD % 512 == 0, ...)`）。explicit.cu 在编译期把这一切全部固定下来（第 194-202 行，在第 5.10 节中引用）。

（第 196 行的例子：`0x1E4 = 484 = 4 x 121`，一个 512 字节原子块中的 float 索引 121 即 `m % 32 = 25`、`n % 4 = 3`；`0x1E4 & 0x180 = 0x180`，`>> 2 = 0x60`，`0x1E4 ^ 0x60 = 0x184`。`0x080`，它给出 `epi_swz(0x080) = 0x080 ^ 0x20 = 0xA0 = 160`，是 explicit.cu:201-202 处的 `static_assert` 把 `epi_swz` 与 `cute::Swizzle<2,5,2>{}` 作比较的三个点中的第一个（5.10.2 重新计算了全部三个）。）

为什么是这个原子，而不是 A 和 B 所用的 16 字节 chunk 的 `Swizzle<3,4,3>`：选择器的 `static_assert` 消息说明了这一点，"for mn-major tf32 operands, SW128_32B is the only available smem layout"。该选择器由 UMMA 操作数路径（在那里，32 位的 MN-major 操作数必须使用 128 字节 swizzle 的 32 字节原子变体）和 epilogue（它继承了这一选择）所共享（源自源码）。至于仅就 TMA 存储而言换一种 swizzle 也同样可行，这是我的分析，而不是代码所检验的事情。

### 10.5.2 将原子平铺到 (128, 16) subtile 上：先 N 后 M

对于 M-major 的 D，stage layout 由 `tile_to_shape(SmemLayoutAtomD{}, product_each(shape(EpilogueTile{})), Step<_2,_1>)` 构建 (sm100_epilogue_tma_warpspecialized.hpp:153-154)，然后被赋予一个 stage mode (`:162-163`)。`Step<_2,_1>` 的 tile 顺序是先 mode 1 (N)，后 mode 0 (M)。(128, 16) subtile 沿 N 容纳 16/4 = 4 个原子，沿 M 容纳 128/32 = 4 个原子，排布为 N 方向的原子相邻（stride 为 128 个 float = 512 B），然后是 M 方向的原子（stride 为 4 x 128 = 512 个 float = 2048 B）。由此得到的未 swizzle 的 stage layout，以 float 为单位，如下（笔记 B.3.3 步骤 6；笔记第 5.3 节中的 TMA 描述符构造对它的 `((32,4),(4,4)):((1,512),(32,128))` 形式求逆）：

```
float offset(m, n) = (m % 32) * 1  +  (m / 32) * 512  +  (n % 4) * 32  +  (n / 4) * 128
```

乘以 4 换算成字节，并合并两个 `n` 项（对于 `0 <= n < 16`，`128 (n % 4) + 512 (n / 4) = 128 n`）：

```
pre-swizzle byte(m, n) = 4 (m % 32) + 128 n + 2048 (m / 32)          (0 <= m < 128, 0 <= n < 16)
```

把单位弄清楚：M 块的 stride 是 512 个 *float* = 2048 *字节*。笔记 D.6 第 1 项以 4 字节的字（float 单位）给出推导，`(m%32) + 32n + 512(m/32)`；像 "4 (m % 32) + 128 n + 512 (m / 32)" 这样单位混杂的写法会在最后一项上差 4 倍，并且通不过与代码中 `2048 w` 的对照检查。8192 字节的 stage 是 `4 x 2048`：M 块 `i = m / 32` 占据字节 `[2048 i, 2048 i + 2048)`，而在其内部，列 `n` 占据 128 字节的行 `[128 n, 128 n + 128)`，该块的 32 个 M 值依序排列于其中。图示如下：

```
  one 8192-byte epilogue stage (pre-swizzle), 128 rows (m) x 16 columns (n) of fp32
  ---------------------------------------------------------------------------------
  M-block 0 (rows   0.. 31):  bytes    0..2047   = 16 lines of 128 B, line n = column n, 32 fp32 = rows 0..31
  M-block 1 (rows  32.. 63):  bytes 2048..4095
  M-block 2 (rows  64.. 95):  bytes 4096..6143
  M-block 3 (rows  96..127):  bytes 6144..8191
  within an M-block:   line n (128 B)  = [ chunk0: m%32 in 0..7 | chunk1: 8..15 | chunk2: 16..23 | chunk3: 24..31 ]   (32 B each)
  atom = 4 consecutive lines (n%4 = 0..3) of one M-block = 512 B
```

同一个 stage 以二维方式画出，即 `tile_to_shape` 所产生的 512 字节原子构成的 4 x 4 网格。原子 `(i, g)` = M 块 `i`（行 `32 i .. 32 i + 31`）乘以 N 组 `g`（列 `4 g .. 4 g + 3`）；它的字节范围 `[2048 i + 512 g, +512)` 是 4 行各 128 B，每列一行。10.5.3/10.5.5 的 chunk 置换在这里针对一个原子的四行提前给出：

```
  one 8192-byte epilogue stage as 16 atoms of 512 B (byte ranges relative to the stage base)

                 N-group g = 0        g = 1          g = 2          g = 3
                 (cols 0..3)         (cols 4..7)    (cols 8..11)   (cols 12..15)
  M-block i = 0  [   0,  512)   [ 512, 1024)   [1024, 1536)   [1536, 2048)    <- warp w = 0 writes these 16 lines (2048 B) = TMA box i = 0
  M-block i = 1  [2048, 2560)   [2560, 3072)   [3072, 3584)   [3584, 4096)    <- warp 1 / box 1
  M-block i = 2  [4096, 4608)   [4608, 5120)   [5120, 5632)   [5632, 6144)    <- warp 2 / box 2
  M-block i = 3  [6144, 6656)   [6656, 7168)   [7168, 7680)   [7680, 8192)    <- warp 3 / box 3

  inside one atom (M-block i, N-group g): 4 lines of 128 B; line c holds column 4 g + c, 32 fp32 = rows 32 i .. 32 i + 31 (8 rows per 32-byte chunk)
     line c = 0 (bits [7,9) = 0): [ chunk0: rows 0-7   | chunk1: rows 8-15  | chunk2: rows 16-23 | chunk3: rows 24-31 ]   (swizzle = identity)
     line c = 1 (bits [7,9) = 1): [ rows 8-15  | rows 0-7   | rows 24-31 | rows 16-23 ]   (chunk index ^ 1 after Swizzle<2,5,2>)
     line c = 2 (bits [7,9) = 2): [ rows 16-23 | rows 24-31 | rows 0-7   | rows 8-15  ]   (chunk index ^ 2)
     line c = 3 (bits [7,9) = 3): [ rows 24-31 | rows 16-23 | rows 8-15  | rows 0-7   ]   (chunk index ^ 3)
  thread (w, l) writes one 4-byte word per line, 16 lines per subtile: word (l & 7) of chunk ((l >> 3) & 3) ^ c in line j of M-block w;
  per column j the 32 lanes of warp w fill the whole 128-byte line [2048 w + 128 j, +128) (10.5.5).
```

在应用 swizzle 之前要看到两个推论：(a) M 块 `i` 恰好就是 TMA 存储 box `i` 的 smem 占用（2048 个连续字节，16 行 x 128 B），这就是为什么在 10.10 中 `src_base + 2048 i` 是 box `i` 的源；(b) 拥有行 `[32 w, 32 w + 32)` 的 warp 只写 M 块 `w`：即 `2048 w` 这一项。

### 10.5.3 应用 Swizzle<2,5,2> 得到 (((l >> 3) & 3) ^ (j & 3)) 这一项

把线程的身份 `m = 32 w + l` 与 `n = j`（值 `j` 是 subtile 的列 `j` = tile 的列 `16 e + j`）代入 swizzle 前的字节偏移，并写成二进制。利用 `m % 32 = l`、`m / 32 = w`：

```
 pre-swizzle byte offset inside the stage = 2048 w + 128 j + 4 l

 bit:   12  11 | 10   9 | 8    7 | 6    5 | 4  3  2 | 1 0
 field:   w    | j >> 2 | j & 3  |(l>>3)&3|  l & 7  | 0 0
 weight: 2048  |  512   |  128   |   32   |    4    |
```

`Swizzle<2,5,2>` 把位 [5,7) 与位 [7,9) 做 XOR：字段 `(l >> 3) & 3` 变成 `((l >> 3) & 3) ^ (j & 3)`，而其他任何位都不动。于是

```
 byte(w, l, j) = 2048 w + 128 j + 4 (l & 7) + 32 (((l >> 3) & 3) ^ (j & 3))
               = 2048 w + 128 j + 4 ((l & 7) + 8 (((l >> 3) & 3) ^ (j & 3)))
```

加上 stage 基址 `OFF_SMEM_CD + 8192 (e % 4)`，你就逐字得到了第 989 行。拆开 `128 j = 128 (j & 3) + 512 (j >> 2)`，你就得到了第 990 行：`st_off[c] = 512 + 2048 w + 4 (l & 7) + 128 c + 32 (((l >> 3) & 3) ^ c)`，其中 `c = j & 3`，再加上 `512 (j >> 2) + 8192 (e & 3)`。这就是第 1013 行，以及第 1047 行的地址表达式。

一次完整的枚举（128 个线程 x 16 个值 x 8 个 subtile，在准备本章时用 Python 完成）确认：第 989 行、第 990/1013/1047 行以及由原子推导出的 `OFF + 8192 (e % 4) + swz(4 (m % 32) + 128 n + 2048 (m / 32))` 在全部 16384 种情况下一致，并且每个 subtile 恰好把它所在 stage 的 8192 字节覆盖一次。笔记 G.3 报告了评审在 SASS 层面的对应结果：explicit kernel 与 baseline 的 `STS` 地址形式，对同样的 128 x 8 x 16 种情况重新计算后，到达相同的字节偏移（0 处不匹配，每个 subtile 恰好把它所在 stage 的 `0x200..0x21ff` 覆盖一次）。

### 10.5.4 算例，重新计算

第 1013 行，`st_off[c] = 512 + 2048 w + 4 (l & 7) + 128 c + 32 (((l >> 3) & 3) ^ c)`，对横幅注释中的三个线程求值：

| 线程 | `l & 7` | `(l >> 3) & 3` | c = 0 | c = 1 | c = 2 | c = 3 | 横幅注释所说 |
|---|---|---|---|---|---|---|---|
| w = 0, l = 0 | 0 | 0 | 512 + 0 + 0 + 0 = **512** | 512 + 128 + 32 (0^1 = 1) = **672** | 512 + 256 + 32 (2) = **832** | 512 + 384 + 32 (3) = **992** | {512, 672, 832, 992} |
| w = 0, l = 9 | 1 | 1 | 512 + 4 + 32 (1) = **548** | 512 + 4 + 128 + 32 (1^1 = 0) = **644** | 512 + 4 + 256 + 32 (1^2 = 3) = **868** | 512 + 4 + 384 + 32 (1^3 = 2) = **964** | {548, 644, 868, 964} |
| w = 3, l = 31 | 7 | 3 | 512 + 6144 + 28 + 32 (3) = **6780** | 6684 + 128 + 32 (3^1 = 2) = **6876** | 6684 + 256 + 32 (3^2 = 1) = **6972** | 6684 + 384 + 32 (0) = **7068** | {6780, 6876, 6972, 7068} |

第四个例子：`w = 0, l = 9, j = 5, e = 6`：`j & 3 = 1`，所以 `st_off[1] = 644`；`512 (j >> 2) = 512`；`8192 (e & 3) = 8192 x 2 = 16384`；总计 `644 + 512 + 16384 = 17540 = 0x4484`。stage `e & 3 = 2` 始于 `512 + 16384 = 16896`，所以 stage 内的偏移是 `644`；`644 / 2048 = 0` 给出 box `i = 0`（= `w`，理应如此）。swizzle 前，该值位于 `2048 x 0 + 128 x 5 + 4 x 9 = 676 = 0x2A4`；`0x2A4 & 0x180 = 0x080`，`>> 2 = 0x20`，`0x2A4 ^ 0x20 = 0x284 = 644`。一致。

### 10.5.5 chunk 置换，以及一个 warp 每列写什么

对固定的 `(w, j, e)`，32 个 lane 写入相对于行基址 `2048 w + 128 j` 的 `4 (l & 7) + 32 (((l >> 3) & 3) ^ (j & 3))`。当 `l` 遍历 0..31 时，`l & 7` 取遍 0..7 的每个值，`(l >> 3) & 3` 取遍 0..3 的每个值，而与常量 `j & 3` 做 XOR 是 0..3 上的一个双射。所以这 32 个 lane 写入的是 32 个字，位于**一个 128 字节的行 `[2048 w + 128 j, +128)` 内，连续且每个字恰好一次**；只有四个 32 字节 chunk 的顺序被置换了：

```
 lane group g = (l >> 3) & 3   ->   32-byte chunk  g ^ (j & 3)
   j & 3 = 0 :  g 0 1 2 3  ->  chunk 0 1 2 3   (identity)
   j & 3 = 1 :  g 0 1 2 3  ->  chunk 1 0 3 2
   j & 3 = 2 :  g 0 1 2 3  ->  chunk 2 3 0 1
   j & 3 = 3 :  g 0 1 2 3  ->  chunk 3 2 1 0
```

因此一个 warp 的每条 `st.shared.b32` 触及一个 128 字节跨度内 32 个不同的 4 字节字：全部 32 个 bank 各一次，无冲突（我对地址集合的分析；swizzle 只在行内置换整个 32 字节 chunk，所以对这种写入模式它既不制造也不消除冲突）。每个 subtile 一个 warp 写 16 行 = 2048 B（它在 stage 中的 M 块），四个 warp 写 8192 B，即整个 stage。

### 10.5.6 为什么 D 的 map 写的是 SWIZZLE_128B_ATOM_32B，以及存储如何反 swizzle

TMA 描述符的 swizzle 模式是*从 smem layout 的 swizzle 推导而来的*，这样 TMA 引擎才会从线程所写入的那个字节读取每个元素。这一映射位于 cute/arch/copy_sm90_desc.hpp:252-259：`SmemSwizzleBits::B128` 配上 `SmemSwizzleBase::SWIZZLE_BASE_32B` 返回 `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`（仅对 CUDA > 12.6 可用，由第 257 行守卫）。CUDA Driver API 对这些枚举量的原文记载是：`CU_TENSOR_MAP_SWIZZLE_128B` "Swizzle 16B chunks within 128B span"、`CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` "Swizzle 32B chunks within 128B span"、`CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B` "Swizzle 32B chunks within 128B span, additionally swap lower 8B with upper 8B within each 16B for every alternate row"，以及 `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B` "Swizzle 64B chunks within 128B span"（`CUtensorMapSwizzle`，cuda-driver-api 的 `CUDA_TENSOR_MEMORY` 组）。CuTe 在我们所用那一行的相邻几行（copy_sm90_desc.hpp:255 和 259）把 `SWIZZLE_BASE_16B` 映射到 `CU_TENSOR_MAP_SWIZZLE_128B`，把 `SWIZZLE_BASE_64B` 映射到 `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B`；`_FLIP_8B` 枚举量在那个 switch 中没有对应的 case（`SWIZZLE_BASE_32B_FLIP_8B`，`:145`，只是被声明了）。用 CuTe 的术语说，这就是 `Swizzle<2,5,2>` 与 `Swizzle<3,4,3>` 之别：前者是 32 字节 chunk（第 5-6 位）与 2 个行位（第 7-8 位）做 XOR，后者是 16 字节 chunk（第 4-6 位）与 3 个行位（第 7-9 位）做 XOR。C 与 D 的实测编码器输入是 `FLOAT32`、shape `(8192, 8192, 1)`、字节 stride `(32768, 0)`、box `(32, 16, 1)`、`SWIZZLE_128B_ATOM_32B`（笔记，第 5.3 节以及 D.1 D3）。

box `(32, 16, 1)`：box 的内层维度是 32 个 fp32 = 128 字节，这是驱动对任何 `SWIZZLE_128B*` 模式所允许的最大值（`cuTensorMapEncodeTiled`："the bounding box inner dimension (computed as boxDim[0] multiplied by element size derived from tensorDataType) must be less than or equal to the swizzle size"，以及 "CU_TENSOR_MAP_SWIZZLE_32B requires the bounding box inner dimension to be <= 32"、"... _64B ... <= 64"、"CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128"）。这也是 CuTe 的 `construct_tma_gbasis` 停下的地方：在 32 个连续的 M 元素和以 32 个 float 为 stride 的 16 个 N 列之后，下一个 M 块在 512 个 float 之外，并不连续（笔记，第 5.3 节）。所以一次逻辑上的 (128, 16) subtile 复制是四条 TMA 指令 `i = 0..3`。

TMA 如何读取 box：它把 `src_base + 2048 i` 处 2048 字节的 smem 区域当作 16 行（box 维度 1 = `n'`），每行 128 字节（box 维度 0 = 32 x 4 B）。对元素 `(m', n')`，它把 128B/32B 原子的 swizzle 施加于*共享内存地址位*：它读取字节 `2048 i + 128 n' + swz(4 m')`，其中 `swz` 把地址的位 [5,7) 与位 [7,9) 做 XOR。`smem_base + OFF_SMEM_CD = 0x400 + 0x200 = 0x600`（加上 `r << 24`）是 512 字节对齐的，而 stage 偏移与 box 偏移都是 2048 的倍数，所以绝对地址的位 [7,9) 等于 `n' & 3`。因此 TMA 对 `(m', n')` 的读地址是 `OFF_SMEM_CD + 8192 (e & 3) + 2048 i + 128 n' + 4 ((m' & 7) + 8 (((m' >> 3) & 3) ^ (n' & 3)))`，即取 `w = i`、`l = m'`、`j = n'` 的线程公式。10.5.3 的枚举对全部 8 x 4 x 16 x 32 种情况检验了这一恒等式。XOR 是一个对合：引擎施加写入者所施加的同一个置换，这*就是*反 swizzle。在全局侧，TMA 把 box `i` 的元素 `(m', n')` 写到普通的 M-major 地址 `D + 4 (128 tm + 32 i + m') + 32768 (128 tn + 16 e + n')`：每个 box 16 段各 128 个连续字节，相隔 32768 字节。D 永远看不到 swizzle。

对齐起着承重作用，而且这里涉及两个不同的对齐事实。(1) PTX ISA 用一张 16 字节单元格的表来描述 128 字节 swizzle 模式的 "32-byte atomicity sub-mode"（"In this sub-mode, the 32-byte of data is kept intact while swizzling"），表的四行依次为 `0 1 | 2 3 | 4 5 | 6 7`、`2 3 | 0 1 | 6 7 | 4 5`、`4 5 | 6 7 | 0 1 | 2 3`、`6 7 | 4 5 | 2 3 | 0 1`，"Pattern repeats"（PTX ISA 9.4，tensor swizzling modes）。4 行周期中的第 `r` 行通过与 `r` 做 XOR 来置换四个 32 字节的对，这恰好就是 `Swizzle<2,5,2>`（chunk 位 [5,7) ^= 行位 [7,9)）；这张表是 `Swizzle<2,5,2> == SWIZZLE_128B_ATOM_32B` 的第一手来源。同一段落还给出了最低的*合法*地址对齐："This sub-mode requires 32 byte alignment at shared memory"。

(2) `OFF_SMEM_CD` 的 512 字节对齐（explicit.cu:163；CUTLASS 的 `alignas(512)`，笔记第 6.2 节）是一种类型不同、也更强的要求。引擎把*绝对共享地址*的位 [5,7) 与位 [7,9) 做 XOR；10.5.3 的线程公式则对 *stage 内* 偏移的位做 XOR。两者只有在 stage 基址对位 [7,9) 的贡献为 0 时才一致，也就是基址是 512 = 4 行 x 128 B 的倍数，即该图案的周期（位 [7,9) 是唯一进入这个 XOR 的行位）。`0x600` 的位 [7,9) = 0。CUDA Programming Guide 给出了一般性的警告："When the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself, there is an offset between the swizzle pattern and the shared memory"（Asynchronous Data Copies），但它的对齐表只列出了 16 字节 chunk 的各个模式（`SWIZZLE_32B/64B/128B`：分别在 256/512/1024 字节后重复），而没有列出 `ATOM_32B` 子模式。因此这里所用的 512 字节周期是从 swizzle 的定义推导出来的，而不是从该指南中读到的。把这些 stage 移到偏移 256 处，每一个 box 都会按照练习 M11（第 14 章）所推演出的图案被打乱。

---

## 10.6 per-tile 循环头 (explicit.cu:1024-1034)

```
1024:   do {
1028:     valid = clc_consume<0u>(smem_base, x, y, clc_idx, clc_phase, clc_count, tm_next, tn_next, tile);   // BEFORE storing the current tile (:878-886)
1029:     uint32_t const acc_stage = accc_idx;
1030:     uint32_t tok = mbar_try_wait(acc_full_base + 8u * acc_stage, accc_phase);                   // acc_pipeline.consumer_try_wait, once per tile (:813)
1031:     uint32_t const taddr = T + tmem_lane_off + 128u * acc_stage;                                // warp-uniform TMEM address of this warp's 32 rows
1032:     #pragma unroll                                                                               // 8 subtiles fully unrolled (:818-820); DelayTmaStore = false
1033:     for (int e = 0; e < kEpiSubtiles; ++e) {
1034:       if (e == 0 && !tok) { EXPLICIT_WAIT(acc_full_base + 8u * acc_stage, accc_phase, WAIT_ACC_FULL, tile, 0u); }   // consumer_wait on the first subtile (:864-867)
```

**存储之前先做 CLC 消费（1028）。** CUTLASS kernel 在 epilogue 循环体的顶部调用 `scheduler.fetch_next_work(...)`（sm100_gemm_tma_warpspecialized.hpp:878-886），然后才以*仍是当前的* `cta_coord_mnkl` 调用 `collective_epilogue.store(...)`；`work_tile_info = next_work_tile_info; cta_coord_mnkl = ...` 在 `store` 返回之后运行（`:932-933`）。因此即使是最初那个 tile 的存储之前也先有一次消费，而最后一个 tile 是在未被取消的终止符被消费之后才存储的（笔记，第 7.7 节；在 baseline 运行中实测：rank 0 的第一条 `UTMASTG` 携带的是启动 tile 的坐标，且在 warp 0 的第二次消费之后 32 ns 发射）。`clc_consume<0u>`（第 7 章）等待本 CTA 的 `clc_full[clc_idx]`，解码响应，执行 fence，在 CTA 0 的 `clc_empty[clc_idx]` 上远程 arrive（那 800 次到达的一部分），推进双槽位状态，并且如果响应是一个被取消的 cluster，则写入 `tm_next, tn_next`。模板实参 `0u` 对这些 warp 而言意味着“无 trace 记录种类”。

explicit kernel 在每个角色中都保持了 baseline 的顺序（先消费，再做该角色自己的工作）（4.7.2）。

**每个 tile 一次 peek（1029-1030）。** `acc_stage = accc_idx` 就是 `t % 4`。先 peek、再 wait 的拆分让编译器可以把第 1031 行的地址运算放在两者之间。这个角色没有更重的工作可以用来重叠，所以保留这一形状是为了忠实性（每个协议步骤一个 PTX `try_wait` 站点，笔记 E.7.4），代价是多一条 PTX `try_wait`，即那次 peek（W7，explicit.cu:314-326；在 SASS 中是 `0x2180` 处的 `TRYWAIT`，以及 `0x2260` 处跳过 `0x2270` 处那条阻塞 `TRYWAIT` 的 `@P0 BRA`）。

**`taddr`（1031）。** `T + ((32 w) << 16) + 128 st`：TMEM 基址，加上本 warp 的 lane 象限，再加上 stage `st` 的第一列。每一项都是 warp-uniform 的，所以 `taddr` 也是，正如 `tcgen05.ld.sync.aligned` 所要求的："All the threads in the warp must specify the same value of taddr, which must be the base address of the collective load operation. Otherwise, the behavior is undefined"（PTX ISA 9.4，tcgen05.ld）。在 baseline 的 SASS 中它是一条写入 uniform 寄存器的 `R2UR`，而每个 subtile 的 `16 e` 变成一个立即数（`LDTM.x16 R4, tmem[UR4 + 0x10*e]`，笔记 D.2，*baseline 的* SASS 词典）。explicit kernel 有同样的八条 `LDTM.x16` 和同样的 16 个目标寄存器 `R4-R19`（笔记 G.3），而转储（`trace_out_v2/sass_explicit.txt`）显示出同样的 uniform 寄存器形式：`LDTM.x16 R4, tmem[UR29 + 0x10*e]` 位于 `0x2290..0x44d0`，其中 `UR29` 在 uniform 数据通路中构建（`R2UR UR17, R2` 位于 `0x2030`，`ULEA` 位于 `0x2050` 和 `0x2140`）。

**为什么这 8 个 subtile 被完全展开（1032）。** baseline 将它们展开（`#pragma unroll(UnrollEpiLoop ? NumEpiSubtilesN : 1)`，sm100_epilogue_tma_warpspecialized.hpp:818-820）。当 `e` 是编译期常量时，TMEM 列立即数 `16 e`、smem 立即数 `0x200 + 0x80 (j & 3) + 0x200 (j >> 2) + 0x2000 (e & 3)`、TMA 源偏移 `0x200 + 0x2000 (e & 3) + 0x800 i`（stage-0 的各 subtile 为 `0x200/0xA00/0x1200/0x1A00`）以及八个存储坐标元组 `(128 tm + 32 i, 128 tn + 16 e)` 要么是常量要么被外提（G.3 指出 "the eight D-store coordinate tuples are computed once"），而两个特例 `e == 0` 和 `e == 7` 也被静态地解决。因此快速路径上没有针对 `e` 的分支。代价是代码体积（8 份副本），在这里无关紧要。

**`DelayTmaStore = false`（1032）。** builder 设置了 `DelayTmaStore = is_void_v<ElementC_>`（sm100_builder.inl:1111），并附有注释 "TMA store delay performs worse with residual loads"。`ElementC = float` 不是 void，所以 subtile `e` 的存储在 subtile `e` 中发射（`if constexpr (not DelayTmaStore) { tma_store_fn(epi_m, epi_n); }`，头文件 `:928-931`）。若开启延迟，subtile `e - 1` 的存储会在 subtile `e` 的算术运算之后才发射，使 FP 工作与 TMA 发射交错；头文件的 `static_assert(not (DelayTmaStore and ReuseSmemC and StagesC <= StagesD), ...)`（`:758`）记录了那种配置可能产生的死锁。explicit kernel 根本没有延迟路径。

**`e == 0` 处的 wait（1034）。** `WAIT_ACC_FULL` 是 trace 站点 id（explicit.cu:213），而末尾的 `tile, 0u` 是记录字段；这一切在性能构建中全部消失。

---

## 10.7 一个 subtile 的步骤 1：tcgen05.ld（explicit.cu:1035-1036，W31 位于 542-555）

```
1035:       uint32_t r[16];
1036:       tmem_ld_32x32b_x16(taddr + 16u * uint32_t(e), r);                                        // columns 16e..16e+15 of row 32 w + l (:881-883)
```

W31 的 asm 字符串（第 6.7.4 节）逐字就是 `SM100_TMEM_LOAD_32dp32b16x::copy`（copy_sm100.hpp:3576-3604；其上方的 CUTLASS 注释写着 "32 data path lanes, 32-bit pattern, repeated 16 times"）。解读这个助记符：

- `.sync.aligned`：warp 集体操作；全部 32 个 lane 以同一地址一起执行它。
- `.32x32b`：shape 是 32 个 TMEM lane x 32 位：warp 的 lane `l` 接收位于 TMEM lane `(address lane) + l`、列 `(address column)` 处的那个 32 位字。
- `.x16`：该 shape 沿列方向重复 16 次，所以 lane `l` 接收 16 个字，即列 `c, c + 1, ..., c + 15`（其中 `c` 是地址中的列），按此顺序放入 `d[0..15]`。
- `.b32`：32 位数据；这些寄存器是原始位，在 10.9 中由 `__uint_as_float` 重新解释为 fp32。

当 `src_addr = T + ((32 w) << 16) + 128 st + 16 e` 时，lane 部分是 `32 w`，列部分是 `128 st + 16 e`：lane `l` 读取 TMEM lane `32 w + l` = CTA tile 的累加器行 `32 w + l`，而 `d[j]` 是 stage `st` 的列 `16 e + j`。线程 `(w, l)` 在该指令结束时持有 `j = 0..15` 的 `acc[32 w + l][16 e + j]`，即它那一行的 16 个连续输出列 `16 e .. 16 e + 15`。实测（而不只是阅读源码）证实了这一映射：explicit kernel 在每次运行中都产生了数值相等的 D，其中一次写入一个用 `0xFF` 字节作哨兵填充的缓冲区（开关开启的那次运行，笔记 G.1），五次写入裸的 `cudaMalloc` 缓冲区（开关关闭）。任何其他的 lane/寄存器分配都会使这一结果几乎不可能出现。

横幅注释中的例子（第 982 行）：`w = 2, st = 1, e = 3` 在 `T = 0` 时给出 `T + 0x400000 + 128 + 48 = 0x4000B0`：lane `0x40 = 64`，列 `0xB0 = 176 = 128 x 1 + 16 x 3`。lane 64 是 tile 的第 64 行，即 warp 6 的象限的第一行；而 TMEM 的第 176 列是 stage 1 的第 48 列，即 subtile 3 的第一列。warp 6 的 lane `l` 的 16 个寄存器持有 `acc[64 + l][48 .. 63]`。

`tcgen05.ld` 是异步的，其完成机制是 `tcgen05.wait::ld`（PTX ISA 9.4，9.7.18.8.3 与 9.7.18.8.5）。kernel 每个 tile 恰好发射一条，在 `e == 7` 处（下一节）；每个 subtile 的 FMUL/FFMA 都在没有它的情况下消费 `r[]`，和 baseline 一样（`:881-896`：加载在 `:882`，唯一的 wait 在 `:887`（位于 `if (do_acc_release)` 内），进行消费的 `visit` 循环在 `:893-896`）。这在 ISA 上是合法的，而不是一条捷径：真正的寄存器依赖是被遵守的，"regardless of the presence or absence of other forms of synchronization"（第 9.7.18.6.4.5 节）。这次 wait 的存在是为了关闭*内存*反依赖，在这里即 leader 的下一条带 `scale_c = 0` 的 `tcgen05.mma` 在某次加载可能仍在读取 TMEM stage `st` 时覆写它。第 1039 行的 arrive 解锁了那次覆写，而第 1038 行那唯一的一次 wait 则为它充当 fence，与 ISA 自己的示例完全一致（两者都在第 2 章，第 2.4.4 节中引用）。另一个独立的实测事实：在 wait 站点处 SASS 没有 wait 操作码。由控制字 scoreboard 承载这一顺序：每条 `LDTM` 之后的第一条 `FMUL` 在该 `LDTM` 所设置的 scoreboard 上等待，而 subtile 7 的 `SYNCS.ARRIVE` 在最后一条 `LDTM` 的 scoreboard 上等待（baseline 见笔记 D.2，explicit kernel 见 G.3；scoreboard 的解码是从编码推断出来的）。第 2 章，第 2.4.4 节还讨论了笔记 B.6 对这条规则更强的解读。

---

## 10.8 e == 7 时的步骤 2：wait::ld、释放 TMEM stage、推进（explicit.cu:1037-1041；W32 位于 557-565；W11 位于 369-379）

```
1037:       if (e == kEpiSubtiles - 1) {                                                               // after the last tmem load of the tile (:886-890)
1038:         tmem_wait_ld();                                                                          // fence_view_async_tmem_load
1039:         mbar_arrive_cluster_masked(acc_empty_base + 8u * acc_stage);                             // consumer_release: 128 threads of both CTAs -> 256 on the leader's empty[st]
1040:         advance<kAccStages>(accc_idx, accc_phase);
1041:       }
```

这一块就是 sm100_epilogue_tma_warpspecialized.hpp:886-890 处的 CUTLASS 序列 `if (do_acc_release) { cutlass::arch::fence_view_async_tmem_load(); acc_pipeline.consumer_release(acc_pipe_consumer_state); ++acc_pipe_consumer_state; }`，其中 `do_acc_release = is_last_iteration`（`:825`）；`fence_view_async_tmem_load` 就是 W32（第 6.7.5 节）。对 2SM MMA 而言，`consumer_release` 是 `consumer_release_2x1SM`（sm100_pipeline.hpp:241-249，271-277），它调用 `umma_arrive_2x1SM_sm0`（barrier.h:905-921）：即在地址 `& Sm100MmaPeerBitMask` 上的一条普通 `mbarrier.arrive.shared::cluster.b64`（W11，第 6.3.7 节）。

**带掩码的地址。** `kPeerBitMask = 0xFEFFFFFF`（explicit.cu:187，`cute::Sm100MmaPeerBitMask`，cute/arch/copy_sm100_tma.hpp:45）清除实测基址 `0x400 + (r << 24)` 的第 24 位，即 rank 的第 0 位。

为什么不对称（wait 自己的 full，arrive 到 leader 的 empty 上）：基于源码的原因在 4.4.3 中（`mbarrier.try_wait` 只接受 `.shared::cta`，所以每个 epilogue 都在自己的 `acc_full` 上等待；只有 leader 的 warp 0 会询问该 stage 是否空闲）；此外，一个计数为 256 的屏障比两个计数为 128 的屏障外加一次跨 CTA 转发更便宜（我的分析）。

**为什么在 subtile 7 的算术运算之前释放是正确的。** `tcgen05.wait::ld`（1038）才是在释放之前完成 TMEM 读取的那一步（4.4.3）；subtile 7 以及之前所有 subtile 的寄存器 `r[0..15]` 从来就不成问题，因为 ISA 无需 wait 就会遵守寄存器依赖（10.7）。顺序方面：W32 与 W11 的 `"memory"` clobber 以及 `asm volatile` 使 arrive 在程序顺序上保持在 wait 之后；而在 SASS 中，`tcgen05.wait::ld` 不产生任何指令，位于第 1039 行位置的 `SYNCS.ARRIVE` 在最后一条 `LDTM` 的 scoreboard 上等待（笔记 G.3，"the subtile-7 accumulator release waits on the `LDTM` scoreboard"；D.2 中有 baseline 的对应物）。

MMA warp 在稳态下实际领先 epilogue 多远并未实测（trace 构建对 CLC 消费与 TMA/MMA 发射打时间戳，而不对 epilogue 的完成打时间戳，笔记 D.1/D.3），所以能说的只有结构性界限（4.7.1 中三个 tile 的 TMEM 余量，以及 4.9 中更紧的两个 tile 的 CLC 界限）。

**`advance<kAccStages>`（1040）。** 之所以在这里而不是在 tile 末尾做，是因为 baseline 在 `:889` 处执行 `++acc_pipe_consumer_state`；`acc_stage`（第 1029 行）在该 subtile 的其余部分保留旧索引，所以 tile 中再没有别的东西读取推进后的状态。

---

## 10.9 算术运算与 16 次标量存储 (explicit.cu:1042-1048)

```
1042:       // Section 8: t = alpha * acc; d = fmaf(beta, c, t) with the zero C fragment; run-time alpha/beta keep FMUL + FFMA (D.7)
1043:       #pragma unroll
1044:       for (int j = 0; j < 16; ++j) {
1045:         float const t = alpha * __uint_as_float(r[j]);
1046:         float const d = fmaf(beta, 0.0f, t);
1047:         *reinterpret_cast<uint32_t*>(smem + st_off[j & 3] + 512u * uint32_t(j >> 2) + kEpiStageBytes * uint32_t(e & 3)) = __float_as_uint(d);   // 16 scalar st.shared.b32 (B.3.3 step 17)
1048:       }
```

**baseline 计算什么（笔记，第 8 节）。** 该 fusion 是 `Sm90LinearCombination`，它是一棵树，其 `visit` 首先逐元素计算 `frg_added = alpha * acc`（`cutlass::multiplies<float>`）。因为 C 是非 void 类型，它随后*总是*计算 `frg_I = multiply_add(beta_frag, C_frag, frg_added)`（sm90_visitor_compute_tma_warpspecialized.hpp:298-322）；`cutlass::multiply_add` 就是 `C(a) * C(b) + c`（cutlass/functional.h:584-590）。当 `beta == 0` 时，不加载 C tile（`is_C_load_needed()` 为 false），而 C 寄存器片段每个 tile 零填充一次：`if (not is_C_load_needed) { cute::clear(args.tCrC); }`（sm90_visitor_compute_tma_warpspecialized.hpp:336-338）。因此对 `beta = 0`，baseline 逐元素计算 `d = beta * 0 + (alpha * acc)`，而 nvcc 的前端在默认的 `-fmad=true` 下把这个乘加收缩为 `fma.rn.f32`。PTX 中每个 tile 有 128 条 `mul.f32` 和 128 条 `fma.rn.f32`，没有 `add.f32`；SASS epilogue 有 128 条 `FMUL` + 128 条 `FFMA` 和 0 条 `FADD`（笔记 D.2；D.6 第 10 项；该 kernel 另外两条 `FMUL` 属于 `%cluster_ctaid` 的下沉结果）。第 1045-1046 行写的恰好就是这两个 IEEE 运算，并使用显式的 `fmaf`，使结果不依赖于 `-fmad` 标志。

**为什么对 `beta = 0` 这两条指令都存在，以及为什么 alpha/beta 是运行时参数。** `alpha` 和 `beta` 来自 `params.alpha/params.beta`（explicit.cu:1110-1111；`ld.param.v2.b32 [+528]`，在 epilogue 入口处由 `LDCU.64 UR26, c[0x0][0x590]` 一次性加载到 uniform 寄存器 `UR26/UR27` 中，笔记 G.3）。编译器无法知道它们是 1 和 0，所以必须发出乘法和 fma。若 alpha 和 beta 是编译期常量 `1.0f` 和 `0.0f`，`1.0f * acc` 会被折叠为 `acc`（一个精确的 IEEE 恒等式），而 `fmaf(0.0f, 0.0f, t)` 最多只会被折叠为 `0.0f + t`，即单条 `FADD`（推断，笔记 D.7；它不能折叠为 `t`，因为 `+0.0 + (-0.0) = +0.0`；见下一段）。这 128 条 FMUL + 128 条 FFMA 将会消失，kernel 也就不再具有 baseline 的指令流（笔记 D.7 "Arithmetic"）。运行时参数同时也是语义上忠实的选择：baseline 在运行时从它的 `Params` 中读取它们。baseline 的 SASS 形式是 `FMUL Rk, R2, Rk`（alpha 在一个向量寄存器中），然后是 `FFMA Rk, R(17+k), UR21, Rk`，其中 `R(17+k)` 是零填充的 C 片段；explicit kernel 的则是 `FMUL Rk, Rk, UR26`，然后是 `FFMA Rk, RZ, UR27, Rk`（这里的零是硬连线的 `RZ`）。两者是对相同操作数值的相同 IEEE 运算，对每一组 `alpha, beta` 都给出逐位相同的结果（笔记 G.3），而活跃的向量寄存器少了 16 个（这是 `REG` 为 26 而非 68 的部分原因）。

**零的符号这一微妙之处。** 当 `beta = +0.0f` 时，`fmaf(beta, 0.0f, t) = (+0.0) + t`，经一次舍入：对每个 `t != 0`，和恰好是 `t`；对 `t = +0.0`，结果是 `+0.0`；对 `t = -0.0`，结果是 **`+0.0`**（就近舍入给出 `(+0) + (-0) = +0`）。因此 D 与 `alpha * acc` 逐位相等，只是负零变成了正零（笔记，第 8 节）。单独一个 `alpha * acc` 会保留 `-0.0`，并恰好在这些元素上与 baseline 不同；这正是 E.6.7 禁止这条捷径的原因（笔记 D.7："never `alpha * acc` alone (sign of zero), never a separate add"）。累加器可能是 `-0.0` 吗？在 IEEE 算术下，只有当一次点积中全部 8192 个乘积都是 `-0.0` 时才可能（例如一个全为 `+0` 的 A 的行乘以严格为负的 B 值）。非零项的精确抵消 `x + (-x)` 在就近舍入下是 `+0.0`，绝不会是 `-0.0`（IEEE 754-2019 第 6.3 节），而一个含有任何 `+0` 乘积的零值和是 `+0`。tensor core 内部的求和顺序与舍入是由硬件定义的（笔记，第 8 节），所以这一结论对 IEEE 算术成立，而对 MMA 并未得到证明。（为完整起见：`beta = -0.0f` 能通过主机端守卫 "like `is_zero()`"（explicit.cu:1272），并给出 `(-0) + t`，这会保留 `-0.0`；守卫拒绝非有限的 `beta`，因为那样 baseline 就会加载 C。）

**为什么先算 `alpha * acc`，再算带 `beta` 的 fma。** 这是 fusion 树的顺序（op_2 = alpha 子树先被访问，然后是 `multiply_add(beta, C, .)`）。对 `beta != 0`，另一种写法 `fmaf(alpha, acc, beta * C)` 的舍入不同；对 `beta = 0`，两者只在零的符号上有差别。对这棵树的忠实才是对*任意* alpha 与 beta 都保证 D 逐位相同的原因。

**十六条 `st.shared.b32` 与地址空间规则。** 第 1047 行解引用 `smem + offset`，其中 `smem` 是动态共享数组的通用 `char*`。它的来源（`extern __shared__`）对 nvcc 可见，所以该存储被发出为 `st.shared.b32`（PTX 清单中有 128 条 `st.shared.b32` 和 0 条通用的 `st.b32`，笔记 G.3），而 SASS 是 10.4 中的 `STS [Rc + UR7 + imm]` 形式。文件头部的规则（explicit.cu:22-23）：*每一次 C++ 解引用都使用通用指针 `smem + OFF_*`；每一个指名共享内存的 asm 操作数都使用 32 位 `shared::cta` 地址 `smem_base + OFF_*`。* 把两者混用会以两种方式失败。为 `"r"` 操作数把 `smem + off` 强制转换为 `uint32_t`，会把一个*通用*地址的低 32 位喂给一条期望 `shared::cta` 地址的指令，于是 TMA 或 mbarrier 指令就在一个垃圾共享地址上操作。（nvcc 会拒绝把原始 64 位指针放进 32 位 `"r"` 约束，所以只有在你做了强制转换时才会发生截断。）把 `smem_base + off` 作为 C++ 指针解引用，会让编译器发出到通用地址 `0x400 + ...` 的*通用*加载/存储，而那不是共享窗口，会引发故障。这些存储是标量的，因为 baseline 的 `AutoVectorizingCopyWithAssumedAlignment<128>` 在寄存器片段与 swizzled smem 视图之间找到 `max_common_vector = 1`（笔记 B.3.3 步骤 17；第 13 节第 9 项）：连续的 `j` 在 unswizzled 的 layout 中相距 128 字节，而经过 `Swizzle<2,5,2>` XOR 之后在 smem 中相距 32 到 224 字节，从不相邻，所以不可能有向量存储。每个线程每个 subtile 十六次 4 字节存储，每个 tile 128 次，每次都无冲突（10.5.5）。

值索引 `j` 遍历 subtile 的 16 列；`(j & 3, j >> 2)` 这一拆分就是 10.4 的 `st_off[]` 因式分解。`kEpiStageBytes = 8192`（explicit.cu:119）乘以 `e & 3` 选出 stage。stage 索引是 `e & 3`，而不是一个持续累计的计数器。CUTLASS 的 `store_pipe_producer_state.index()` 是跨 tile 延续的，但每个 tile 8 个 subtile 是 4 个 stage 的整数倍，所以它在每个 tile 上都等于 `e % 4`（笔记 B.3.3 步骤 17）。实测的 `STS` 立即数带有 stage 项 `0x2000 (e % 4)`（baseline 中为 `0x200 + 0x200 (j >> 2) + 0x2000 (e % 4)`，笔记 D.2），而每一条记录到的 TMA 存储源地址都是 `smem + 512 + 8192 (e % 4) + 2048 i`（笔记 D.1 D8）。这一相等是本配置的性质，而非协议的性质：倘若 EpilogueTile 给出的 subtile 数不是 4 的倍数（比如 5），baseline 的累计索引从第二个 tile 起就会偏离 `e % 4`，而第 1047 行的编译期 `e & 3` 就会出错。如果你更改 tile shape，就要像 baseline 那样把索引跨 tile 延续下去。

---

## 10.10 fence、第一个屏障、warp 4 的四次存储、commit、wait (explicit.cu:1049-1070)

```
1049:       // tma_store_fn (:766-800)
1050:       fence_proxy_async_shared_cta();                                                            // smem writes visible to the async proxy
1051:       named_bar_sync(kBarEpilogue, kEpilogueThreads);                                           // bar.sync 1, 128
1052:       if (issue_tma_store) {
1053:         uint32_t const src_base = cd_base + kEpiStageBytes * uint32_t(e & 3);
1054:         #pragma unroll
1055:         for (int i = 0; i < 4; ++i) {                                                            // four (32 M, 16 N) boxes per subtile (Section 5.3)
1056:           int32_t const c0 = 128 * tm + 32 * i;
1057:           int32_t const c1 = 128 * tn + 16 * e;
```

（第 1058-1065 行是 `[TRACE]` K4b 记录）

```
1066:           tma_store_3d(&params.tma_d, src_base + kStoreBoxBytes * uint32_t(i), c0, c1, 0);
1067:         }
1068:         tma_store_commit_group();                                                                // store_pipeline.producer_commit
1069:         tma_store_wait_read<1>();                                                                // producer_acquire (always_wait): at most 1 group pending on its smem read
1070:       }
```

这就是 CUTLASS 的 `tma_store_fn` lambda（sm100_epilogue_tma_warpspecialized.hpp:766-800）：

```
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        synchronize(); // ensure all threads have issued their async fence
        if (issue_tma_store) {
          copy(params.tma_store_d, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
        }
        ...
        if (issue_tma_store) {
          store_pipeline.producer_commit(store_pipe_producer_state);
        }
        ++store_pipe_producer_state;
        // Wait for the next smem buffer to be available
        if (issue_tma_store) {
          store_pipeline.producer_acquire(store_pipe_producer_state);
        }
        synchronize();
```

其中 `synchronize = NamedBarrier::sync(ThreadCount, EpilogueBarrier)`（`:723`），`ThreadCount = 128`。

### 10.10.1 fence.proxy.async.shared::cta（1050；W15 位于 417-425，第 6.4.2 节）

PTX 默认并不在通用与异步 proxy 之间排序：一次通用写入之后对同一字节的一次异步 proxy 读取需要 `fence.proxy.async`（CUDA Programming Guide 的 TMA 存储序列是 fence、`__syncthreads()`、`cp.async.bulk.tensor`、`commit_group`、`wait_group.read`；Asynchronous Data Copies）。`.shared::cta` 把这个 fence 限制在本 CTA 的共享内存上，这也是唯一涉及的状态。128 个线程中的每一个都执行它，因为一个 proxy fence 排序的是*执行线程自身的*先前访问；若只由一个线程来执行它，那对其他 127 个线程的存储就什么也说明不了。

### 10.10.2 第一个 bar.sync 1,128（1051；W18 位于 431-434，第 6.5.2 节）

屏障 id 1 是 `EpilogueBarrier`（barrier.h:170），128 是四个 epilogue warp（`kBarEpilogue = 1, kEpilogueThreads = 128`，explicit.cu:190-191）。只有当全部四个 warp 都已写入该 stage 中各自的 M 块*并已对其执行了 fence*之后，warp 4 才可以发射 TMA 存储；`bar.sync` 是 "all 128 threads passed line 1050" 开始成立的那个 CTA 作用域的点。SASS 为 `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80`（笔记 D.2）。

### 10.10.3 来自 warp 4 的四次 TMA 存储，无选举（1052-1067；W23 位于 475-484，第 6.6.5 节）

解读这条存储的助记符。`.3d`：这个 kernel 使用的每个 tensor map 都是 rank 3。参数块中的四个 map，即位于偏移 0/128/256/384 的 `tma_a/b/c/d`（explicit.hpp:50-53），都保留了大小为 1 的 L mode（笔记第 5.3 节）。`.global.shared::cta`：目的地在全局内存，源是本 CTA 的共享内存（存储形式没有 cluster 变体，也没有 mbarrier）。`.bulk_group`：完成情况由发射线程的 *bulk async-group* 跟踪，因此用的是 `commit_group`/`wait_group` 而不是 mbarrier。操作数：`[%0, {%2,%3,%4}]` 是 tensor map 和三个坐标。该 tensor map 是 `params.tma_d` 在参数空间中的一个 64 位通用地址；在 SASS 中它是一个 uniform 的 64 位对 `c[0x0][0x348] + 0x180`，即位于参数偏移 384 处的 D map（笔记 G.3）。坐标*以元素为单位*：`c0 = 128 tm + 32 i` 沿 M（内层的、连续的维度），`c1 = 128 tn + 16 e` 沿 N，`0` 对应批次。`[%1]` 是 32 位的 `shared::cta` 源地址 `cd_base + 8192 (e & 3) + 2048 i`（`kStoreBoxBytes = 2048`，explicit.cu:120）。Box `i` 把 D 的行 `[128 tm + 32 i, +32)` x 列 `[128 tn + 16 e, +16)` 从该 stage 的 M 块 `i` 搬出（10.5.2）。

横幅注释中的例子（第 993 行）：tile `(5, 9)`、`e = 3`、`i = 2` 给出 `c0 = 640 + 64 = 704`、`c1 = 1152 + 48 = 1200`，源地址 `OFF_SMEM_CD + 8192 x 3 + 4096 = 512 + 24576 + 4096 = 29184`，即 "stage 3 + 4096 B"。

**全部 32 个 lane、uniform 操作数、无选举。** `issue_tma_store = (w == 0u)` 是 warp-uniform 的，所以 warp 4 的全部 32 个 lane 都以完全相同的操作数执行这四条 `cp.async.bulk.tensor`，正如 baseline 的每个 lane 都在 `if (issue_tma_store)` 之下执行 `copy(params.tma_store_d, ...)`（`:740, :771`）一样；存储路径中任何地方都没有 `elect.sync`（笔记，第 13 节第 8 项）。实测：每条 PTX 存储都下沉为每个 warp **一条** uniform 数据通路的 `UTMASTG.3D`，无谓词，无论 lane 数多少都只发射一次；K4b 探针在它全部 256 条记录（记录上限；笔记 D.2、D.1 D8）中的每一条都记录到 32 个活跃 lane。也就是每个 tile 每个 CTA 32 条 `UTMASTG.3D`。一次选举只会多出一条 `ELECT` 和一个谓词而毫无收益；explicit kernel 保持了 baseline 的粒度（笔记 E.6.6）。

### 10.10.4 commit_group 与 wait_group.read 1（1068-1069；W24/W25 位于 486-495，第 6.6.6 节）

`cp.async.bulk.commit_group` 把该线程尚未提交的存储打包成一个新的 bulk async-group：刚刚发射的四条存储成为“组 `e`”。PTX ISA（cp.async.bulk.wait_group，Description）："A bulk async operation includes the following: Optionally, reading from the tensormap. Reading from the source locations. Writing to their respective destination locations. Writes being made visible to the executing thread. The optional .read modifier indicates that the waiting has to be done until all the bulk async operations in the specified bulk async-group have completed: reading from the tensormap; the reading from their source locations."（CUDA Programming Guide 对 `cp_async_bulk_wait_group_read` 的一行注解是 "Wait for the group to have completed reading from shared memory."）全局写入可能仍在途。这正是 smem 复用所需要的性质，仅此而已：这个 kernel 中没有人把 D 读回来，所以在 tile 循环内部等待全局完成只会增加延迟。

为什么 `N = 1`：`tma_store_wait_read<1>` 就是 `store_pipeline.producer_acquire`，其中 `StorePipeline = PipelineTmaStore<StagesC, StagesD-1>`，原因是 `ReuseSmemC`（sm100_epilogue_tma_warpspecialized.hpp:202-204）。它的 `producer_acquire(stage, count)` 是 `if (params_.always_wait || count > UnacquiredStages) { tma_store_wait<UnacquiredStages>(); }`（sm90_pipeline.hpp:697-700），其中 `UnacquiredStages = StagesD - 1`。builder 给出 `StagesD = min(EpiTiles, 2) = 2` 以及 `StagesC = ReuseSmem ? max(min(EpiTiles, 4), StagesD + 1) : min(EpiTiles, 4) = 4`，其中 `ReuseSmem = sizeof_bits_v<ElementC_> > 8` 对 fp32 为真（sm100_builder.inl:1109-1115；实测的策略是 `Sm100TmaWarpSpecialized<4,2,16,true,false>`，笔记 D.1 D1）。因此 `UnacquiredStages = 1`，并且因为 kernel 设置了 `epi_store_pipeline_params.always_wait = true`（sm100_gemm_tma_warpspecialized.hpp:491），kernel 在*每一个* subtile 上都执行这次等待。SASS：先 `UTMACMDFLUSH` 再 `DEPBAR.LE SB0, 0x1`（笔记 D.2；explicit kernel 有同样的 8 条 `DEPBAR.LE SB0, 0x1`，笔记 G.3）。

`always_wait` 标志改变了实际执行的指令流。`producer_acquire(state)` 传入 `state.count()`（sm90_pipeline.hpp:675-677），即存储状态自 kernel 进入以来所做的推进次数（`make_producer_start_state` 把它置为 0，sm90_pipeline.hpp:254-260；`operator++` 使其递增，`:203-212`）。通用的判断 `count > UnacquiredStages` 会在 `count <= 1` 期间*跳过*等待。acquire 在 `++store_pipe_producer_state` 之后运行（头文件 `:781-785`），所以 CTA 生命周期中的第一次 acquire 看到 `count = 1`，会跳过等待；第二次看到 2，于是等待。因此，explicit kernel 在每个 subtile 上无条件的 `tma_store_wait_read<1>()` 之所以等价于 baseline，*仅仅是因为* `always_wait = true`。没有这个标志，baseline 在每个 CTA 生命周期中会少执行一条 `DEPBAR.LE SB0, 0x1`（源自源码）；8 处 `wait_group.read 1` 位置的静态计数（笔记 E.7.4）仍然会通过，但实际执行的指令流会不同。当你镜像一个流水线类时，请阅读它的 `Params`。

**这次等待保证了什么，以及它保护的是哪个 stage。** 提交组 `e` 之后，`wait_group.read 1` 在所有组（最新的那一组，即组 `e`，可能除外）都已完成读取 smem 时返回。ISA 对 `cp.async.bulk.wait_group` 定义的第二个分句（引述于第 2 章，第 2.7.5 节）"and all the prior bulk async-groups ... are complete"，正是它使这次等待覆盖比最新一组更老的*每一个*组，而不只是四个 subtile 之前的那一组；没有它，这次等待对最近 N 组之外的其他组就什么也说明不了。最新的那个*更老的*组是组 `e - 1`，它读取的是 stage `(e - 1) & 3`；所以在 subtile `e` 结束时，stage `(e - 1) & 3` 已知是空闲的。下面是一个 tile 的八个 subtile 上的环（也跨 tile，因为 `e & 3` 从 0 重新开始，但各组持续流动）：

```
 subtile e :      0      1      2      3      4      5      6      7   | next tile 0 ...
 writes stage :   0      1      2      3      0      1      2      3   |     0
 commits group:   0      1      2      3      4      5      6      7   |     8
 wait.read 1  : groups <= -1 (none) ; <= 0 ; <= 1 ; <= 2 ; <= 3 ; <= 4 ; <= 5 ; <= 6   |  <= 7
 so free after:   -    stage 0  st 1  st 2   st 3   st 0   st 1   st 2  |  st 3
 next write to:  st 1   st 2    st 3  st 0   st 1   st 2   st 3   st 0  |  st 1
 that stage was last read by group: (none) (none) (none)  0      1      2      3      4   |     5
```

在 subtile `e + 1` 写入的 stage 是 `(e + 1) & 3`，它上一次被组 `e - 3` 读取。subtile `e` 处的等待保证了 `<= e - 1` 的各组，其中包括 `e - 3`：安全，并有两个 subtile 的余量。若改为等待到 0，就会在刚刚发射的存储上停顿，并使 TMA 读取与下一个 subtile 的 `tcgen05.ld` 及算术运算串行化；1 则让它们重叠。事实上，subtile `e - 2` 处的等待已经保证了组 `e - 3`，所以在四个 stage 下这个环应当基本上永远不会在 TMA 读取上停顿（我的假设，未实测：当 warp 4 到达这次等待时，它所等待的那一组是在一个 subtile、约 80 条 SASS 指令之前提交的，而且只需读取 8 KB 的 smem）。尽管如此，该协议对两个 stage 的环是*恰好*足够的（stage `(e + 1) & 1 = (e - 1) & 1` 正是被 `wait.read 1` 释放的那一个），即该类默认的 `PipelineTmaStore<Stages, Stages - 1>` 的 `Stages = 2` 实例，其 `wait.read Stages - 1` 对任何 `Stages` 都恰好释放接下来要写入的那个 stage。

四个 stage 是 C 的*类型*在编译期决定的结果，而不是 beta 运行时取值的结果：builder 在 `float` 上求值 `ReuseSmem = sizeof_bits_v<ElementC_> > 8` 和 `StagesC = ReuseSmem ? max(min(EpiTiles, 4), StagesD + 1) : min(EpiTiles, 4)`（sm100_builder.inl:1109-1115），而 D 环取 `ReuseSmemC ? StagesC : StagesD` 个 stage（sm100_epilogue_tma_warpspecialized.hpp:163），所以这个实例化对 `beta = 0` 也有 4 个 stage。这一深度服务的是同一实例化的 `beta != 0` *路径*。在那条路径上，warp 3 把 C 的 tile 预取到同一个联合体中（`CollectiveStorageReuseC`，头文件 `:190-193`），并且 stage 只在其 D 存储被提交一个 subtile 之后才释放给 C 的加载者：这就是 `UnacquiredStages` 的“滞后 1”（笔记第 13 节第 16 项；`store_finished = count > UnacquiredStages` 位于 `:791-797`）。对于 `beta = 0`，这种释放簿记是死状态，explicit kernel 省略了它，但保留了 4 x 8192 字节的 layout，使得 smem 映射、SASS 立即数和 TMA 源地址都与 baseline 的相同（笔记，第 6.2 节）。

### 10.10.5 第二个 bar.sync 1,128 (1071) 与 tile 更新 (1073-1075)

```
1071:       named_bar_sync(kBarEpilogue, kEpilogueThreads);                                           // bar.sync 1, 128
1072:     }
1073:     tm = tm_next; tn = tn_next;                                                                  // (:932-933)
1074:     ++tile;
1075:   } while (valid);
```

`cp.async.bulk.wait_group` 是*线程作用域的*：它只告诉 warp 4 该组已完成读取，不告诉其他任何人。warp 5-7 即将在下一个 subtile 中写入 stage `(e + 1) & 3`（第 1047 行），所以 warp 4 所知道的必须在任何人写入之前传达到全部 128 个线程：这就是第二个 `bar.sync 1, 128`，即 baseline 的第二个 `synchronize()`（`:787`）。在四个 stage 下，接下来要写入的 stage 早在两个 subtile 之前就已被保证空闲，由 subtile `e - 2` 的等待以及随后 subtile `e - 2` 和 `e - 1` 的屏障来保证（10.10.4），所以对本配置而言，第二个屏障对 smem 冒险来说是冗余的（我的分析）。它在该类的默认配置（`UnacquiredStages = Stages - 1`，sm90_pipeline.hpp:653）中起着承重作用，例如非 `ReuseSmemC` epilogue 中两个 stage 的 `PipelineTmaStore<StagesD = 2>`（sm100_epilogue_tma_warpspecialized.hpp:204），其中接下来要写入的 stage 正是本 subtile 的等待刚刚释放的那一个；而它在这里被保留，是因为每个 tile 16 条 `bar.sync 1,128` 是 explicit kernel 所复现的实测指令流的一部分（笔记 D.7、G.3）。一般规则才是要带到其他 kernel 中去的：*TMA 存储的 `wait_group` 之后必须跟一个 CTA 级屏障，然后其他线程才能复用缓冲区。* 一个恰好具有这一模式的生产 kernel 曾在 NVIDIA/TransformerEngine issue #3474 中被报告，该 issue 把 PR #3417 的缺陷类别描述为 "calling `cp_async_bulk_wait_group_read()` from only the issuing thread without a subsequent CTA-wide `__syncthreads()` handoff lets cooperative writers begin overwriting a shared-memory output ring buffer while the TMA engine is still reading it"，并引述了该 PR 预期的症状："rare, timing-dependent output corruption, particularly under global-memory backpressure"。该 issue 自身的症状是 GB200 (sm_100) 上一部分 rank 出现 `cudaErrorIllegalAddress` 故障；讨论串把这些故障追溯到了另一个 bug（动态共享内存中的 per-expert 元数据被 TMA 加载覆写，已在 PR #3483 中修复），所以在那里，缺失的交接只是报告者的假设，而不是已确认的原因（引自该 issue 及其讨论串）。

第 1073-1075 行把在第 1028 行取回的 tile 变为当前的那一个（`:932-933`），对 tile 计数，并在该 tile 有效时继续循环。当 `clc_consume` 返回 false 时，`tm_next/tn_next` 没有被写入，这些拷贝是死的；循环在当前 tile 被完全存储之后退出。

---

## 10.11 循环之后：没有 store_tail (explicit.cu:1076-1080)

```
1076:   // store_tail executes nothing for beta = 0 (is_producer_load_needed() is false): no cp.async.bulk.wait_group.read 0 (Section 7.7, D15, E.6.7)
1080: }
```

baseline 的 `store_tail` (sm100_epilogue_tma_warpspecialized.hpp:1269-1291) 是

```
    if constexpr (ReuseSmemC) {
      if (fusion_callbacks.is_producer_load_needed()) {
        // wait for all TMA stores to complete
        store_pipeline.producer_tail(store_pipe_producer_state);
        ...
```

而 `producer_tail` 是 `tma_store_wait<0>()` (sm90_pipeline.hpp:687-689)。对于 `beta = 0`，`is_producer_load_needed()` 为假（笔记，第 8 节），所以什么都不执行：`cp.async.bulk.wait_group.read 0` 存在于二进制中（`0xb760` 处的 `DEPBAR.LE SB0, 0x0`），处在那个测试之后，而 epilogue 各 warp 在到达它之前就走了 `EXIT`（笔记 D.1 D15、D.6 第 11 项）。explicit kernel 根本没有任何尾段代码；它的 PTX 有 8 条 `wait_group.read 1` 和 0 条 `wait_group.read 0`（笔记 G.3）。

**"exit with at most one bulk group whose smem read is not awaited" 是什么意思。** 在最后一个 tile 的 subtile 7 之后，warp 4 执行了 `wait_group.read 1`（组 6 已经读完 smem）并退出。组 7（最后一个 subtile 的四次存储，来自 stage 3 的 8 KB）可能仍在读取共享内存，而它的全局写入可能仍在途。在那一刻 CTA 的任何其他 warp 是否仍然驻留，源码并未确立：warp 2 的 8 次等待的尾段由每 k-tile 发往 `empty[s]` 的各条 `tcgen05.commit` 释放 (explicit.cu:942)，它们全都在让这最后一次 epilogue 得以开始的 `acc_full` commit 操作 (explicit.cu:944) 之前发射（第 4.8.1 节），而 warp 0 的解除分配握手跟在 epilogue 的 subtile-7 释放之后（第 4.8.3 节、第 10.8 节），所以当 warp 4 退出时，两者可能都已经退出（推导，未实测）；全局写入作为 kernel 完成的一部分而完成（笔记对硬件语义的解读，第 7.7 节与 B.7 D15；未实测）。

PTX 是否在形式上保证一个 CTA 的共享内存比一个已退出线程的挂起 bulk group 存活得更久，这一点未验证：PTX ISA 9.4 的文本中没有任何一句话谈及发射线程或 CTA 退出时仍然挂起的 bulk async-group（"bulk async-group" 的每一处出现都涉及 `.bulk_group` 完成机制、`commit_group` 或 `wait_group`）。tcgen05 这一部分中关于 kernel 退出的唯一规则是 "All of the Tensor Memory that was allocated in a kernel, must be explicitly deallocated before the kernel exits"。CUDA Programming Guide 的示例以带 `N = 0` 的 `cp_async_bulk_wait_group_read` 结束，在终止之前等待每个 bulk group 的 smem 读取（而不是等待全局写入）。baseline 依赖于此，而 explicit kernel 精确地镜像了 baseline。

**为什么不能放松它。** “放松”指的是允许不止一个组未被等待，例如去掉或推迟每个 subtile 的 `wait_group.read 1`，或者以 `N = 2` 等待。在循环内部，这会破坏环 (10.10.4)；在退出时，它会扩大 TMA 在其所有者已经离去之后读取 smem 的窗口。笔记的契约是 "preserve or tighten, never loosen"（第 7.7 节、D.7）。*收紧*，即在 epilogue 各 warp 退出之前加一条 `cp.async.bulk.wait_group 0`（或 `.read 0`），是安全的，而且每个 CTA 花费一条指令（在 warp 4 的路径上），但 explicit kernel 没有这样做：验收标准把 `wait_group.read 0 = 0` 和 `DEPBAR.LE SB0, 0x0 = 0` 算作硬性要求，以使被执行的指令流就是 baseline 的指令流（笔记 E.6.7、E.7.4）。如果你从这个模板出发写自己的 kernel，请在退出时加上完整的等待。

---

## 10.12 per-tile 核算、带宽，以及为什么 epilogue 不在关键路径上

每个 CTA tile（128 x 128 fp32 = 65536 B = 64 KB 的 D），根据第 9 节计数并由 SASS 清单确认（笔记 G.3、D.7）：

| 条目 | 一个 epilogue warp 的程序中的静态计数，每 tile | 每个 CTA tile 的动态 warp 指令数（4 个 warp） | 出处 |
|---|---|---|---|
| `tcgen05.ld.32x32b.x16` (`LDTM.x16`) | 8（每个 subtile 一条，每条 2 KB） | 32，从 TMEM 读取 64 KB | 10.7 |
| `tcgen05.wait::ld` | 1（在此处没有 SASS 指令） | 4 | 10.8 |
| 在 leader 的 `acc_empty[st]` 上的 `mbarrier.arrive` | 1 条指令（32 个 lane -> 32 次到达） | 4 条指令 = 128 次到达（x 2 个 CTA = 256 = 该计数） | 10.8 |
| `FMUL` + `FFMA` | 128 + 128 | 512 + 512; 0 `FADD` | 10.9 |
| `st.shared.b32` (`STS`) | 128（每个 subtile 16 条；每条 warp 指令 128 B） | 512，64 KB | 10.9 |
| `fence.proxy.async.shared::cta` | 8 | 32（`MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S` 对） | 10.10.1 |
| `bar.sync 1,128` | 16（每个 subtile 两条） | 执行 64 次（每个 warp 16 次）= 16 次屏障完成 | 10.10.2、10.10.5 |
| `cp.async.bulk.tensor` 存储 (`UTMASTG.3D`) | warp 4 的路径上 32 条（每个 subtile 4 条，每条 2048 B）；warp 5-7 上 0 条 | 32，向 D 写入 64 KB | 10.10.3 |
| `commit_group` / `wait_group.read 1` | warp 4 的路径上 8 / 8 | 8 / 8 | 10.10.4 |
| CLC 消费（在 CTA 0 的 `clc_empty` 上的远程 `mbarrier.arrive`） | 1 条指令（32 次到达） | 本 CTA 的 192 次到达中的 4 x 32 = 128 次；cluster 的 800 = 32 + 4 x (32 + 32 + 128) 次中的 512 次 | 10.6 |
| per-tile 路径上的 SASS 指令 | 696 = 设置 56 + 各 subtile 77/78/78/78/78/78/81/92（baseline：1265，含其死 C 路径） | - | 笔记 G.3 |

整个问题：`8192^2 x 4 B = 268435456 B = 256 MiB` 的 D；1024 个 cluster tile x 4 个 CTA tile x 64 KB。在实测的 0.889354 ms 内，这相当于全 GPU 约 302 GB/s 的平均写入流，每个活跃 SM 2.3-2.4 GB/s（302 GB/s 分摊到 132 个活跃 SM 上；关键路径 cluster 上每 ~27.8 us 64 KB）：是一块 B200 的 HBM3e 带宽的 3.8 %（NVIDIA 的 DGX B200 产品页面：其八个 GPU 的 "64 TB/s HBM3e bandwidth"，即每 GPU 8 TB/s）。内存带宽不是 epilogue 所关乎的；延迟与重叠才是。

**为什么它不在关键路径上（分析，带有实测的锚点）。** per-tile 周期*至多*约 27.8 us：即 explicit kernel 的 0.889354 ms 分摊到一个关键路径 cluster 的 32 个 tile 上。它是一个上界，因为 `Avg runtime` 区间还包含每次迭代的主机端编码与属性调用（笔记 D.4；第 3 章，第 3.9.2 节）。这在 1965 MHz 下约为 54,600 个周期，并且是 mainloop 决定了它：128 个 k-tile x 4 条 256 x 128 x 16 的 `tcgen05.mma`。tile `t` 的 epilogue 在 MMA warp 计算 tile `t + 1 ..` 的同时运行：4-stage TMEM 环让 MMA warp 在会因 `acc_empty` 阻塞之前最多领先三个 tile（至多约 83 us 的余量），而 epilogue 只需要*平均*每 ~27.8 us 完成一个 tile。

它的实际工作是每个 warp 696 条指令，外加八条短的依赖链（`LDTM` -> 32 次 FP 运算 -> 16 条 `STS` -> fence -> 屏障 -> 4 条 `UTMASTG` -> commit -> 等待前一组的 8 KB smem 读取 -> 屏障）。即使按每条指令 20 个周期的悲观平均值计，指令流也是 `696 x 20 = 13,920` 个周期，在 1965 MHz 下为 7.1 us。再为八次 TMA 往返给每个 subtile 慷慨地加上 1,000-2,000 个周期（8,000-16,000 个周期，4.1-8.1 us），在这些刻意悲观的假设下得到每个 tile 约 11-15 us（我的估计，未实测：trace 构建给 CLC 与 TMA 的*发射*打了时间戳，而不是 epilogue 的结束）。这远低于 27.8 us；这段差额，epilogue 各 warp 是阻塞在 10.6 的 `acc_full` 等待中度过的。

epilogue 每个 CTA 暴露一次：在最后一个 tile 的最后一次 MMA 之后，它的 8 个 subtile 在没有任何东西可重叠的情况下运行，合理估计为几微秒，至多为上面估计的 11-15 us，每个 kernel 一次。（在开始时，epilogue 各 warp 在第一个 tile 的 mainloop 运行期间空闲于 `acc_full` 等待中，但没有任何东西在等它们，所以那段空闲时间不是代价。）相对于 CUTLASS kernel 的实测比值（0.99937，笔记 G.2）与 explicit epilogue 的开销和 baseline 的相同这一点一致，而后者有着同样的 8 x（1 条 `LDTM`、16 条 `FMUL`、16 条 `FFMA`、16 条 `STS`、fence、2 条 `bar.sync`、4 条 `UTMASTG`、commit、wait）骨架（笔记 G.3）。

两个值得带到其他 kernel 中去的设计后果（我的分析）：累加器环的深度用 TMEM（4 x 128 = 512 列，即其全部）换取对 epilogue 抖动的容忍，而在最后一个 subtile 的算术*之前*进行的释放 (10.8) 对 producer 而言是免费的延迟。一旦 epilogue 本身变得比 mainloop 更长（K 很小，或者 fusion 很重），两者都无济于事；那时 epilogue 就是 kernel。

---

## 10.13 一个 subtile 的完整数据路径：tile (5, 9)，e = 3

假设这个 tile 落在了位于 `(x, y)`、rank 为 `r` 的 CTA 中的累加器 stage `st` 上（该 CTA 的 tile 计数器 mod 4 是多少，它就是多少；下面的例子在需要数字的地方使用 `st = 1`）。CTA tile 的各行是 `m = 128 x 5 + (32 w + l) = 640 + 32 w + l`；该 subtile 的各列是 `n = 128 x 9 + 48 + j = 1200 + j`，`j = 0..15`。

```
 (1) TMEM, stage st = 1, subtile e = 3: columns 128*1 + 48 .. +63 = 176..191, all 128 lanes
     warp w reads lanes [32 w, 32 w + 32) at address T + ((32 w) << 16) + 128 + 48:
        w = 0: 0x0000B0   w = 1: 0x2000B0   w = 2: 0x4000B0   w = 3: 0x6000B0
     tcgen05.ld.sync.aligned.32x32b.x16.b32 {r0..r15}, [addr]
        lane l of warp w:  r[j] = acc[640 + 32 w + l][1200 + j]      (j = 0..15)

 (2) registers: t = alpha * r[j]; d = fmaf(beta, 0.0f, t)          (16 FMUL + 16 FFMA per thread)

 (3) smem stage e & 3 = 3, bytes [OFF_SMEM_CD + 24576, +8192) = [25088, 33280) = [0x6200, 0x8200)
     thread (w, l), value j  ->  25088 + 2048 w + 128 j + 4 ((l & 7) + 8 (((l >> 3) & 3) ^ (j & 3)))
     e.g. (w = 1, l = 5, j = 2): 25088 + 2048 + 256 + 4 (5 + 8 (0 ^ 2)) = 27476 = 0x6B54
          M-block 1, line 2, chunk 2 (lanes 0..7 of warp 1 sit in chunk 0 ^ 2 = 2 for j & 3 = 2), word 5
     fence.proxy.async.shared::cta ; bar.sync 1,128

 (4) warp 4: four TMA stores, source smem_base + 25088 + 2048 i, coordinates (640 + 32 i, 1200, 0), i = 0..3
     box 0: rows 640..671   box 1: rows 672..703   box 2: rows 704..735   box 3: rows 736..767   x columns 1200..1215
     the engine reads box i as 16 lines (n' = 0..15) x 128 B, un-swizzling chunk c of line n' to m' in [8 (c ^ (n' & 3)), +8)
     cp.async.bulk.commit_group ; cp.async.bulk.wait_group.read 1 ; bar.sync 1,128

 (5) global D (M-major, element (m, n) at D + 4 (m + 8192 n) = D + 4 m + 32768 n):
     box i, line n'  ->  128 contiguous bytes at  D + 32768 (1200 + n') + 4 (640 + 32 i)
                                               =  D + 0x2580A00 + 0x8000 n' + 0x80 i
       box 0, n' =  0 : D + 0x2580A00    (m 640..671, n 1200)
       box 1, n' =  0 : D + 0x2580A80    (m 672..703, n 1200)
       box 2, n' =  0 : D + 0x2580B00    (m 704..735, n 1200)
       box 3, n' =  0 : D + 0x2580B80    (m 736..767, n 1200)
       box 0, n' =  1 : D + 0x2588A00    (m 640..671, n 1201)
       ...
       box 3, n' = 15 : D + 0x25F8B80    (m 736..767, n 1215)
     per output column n the four boxes together write the 512 contiguous bytes [D + 32768 n + 0xA00, +0x200): rows 640..767.
     the element of step (3), (w = 1, l = 5, j = 2) = (m, n) = (677, 1202): D + 4*677 + 32768*1202 = D + 0x2590A94, inside box 1, line 2.
```

核对步骤 (5) 的算术：`32768 x 1200 = 39321600`，`4 x 640 = 2560`，和为 `39324160 = 0x2580A00`；行与行之间的 stride 是一个 D 列 = `4 x 8192 = 32768 = 0x8000` 字节；各 box 之间的 stride 是 32 行 = 128 字节 = `0x80`。整个 128 x 16 的 subtile 是 16 个各为 512 个连续字节的列段，这对内存系统的友好程度，已经是一次 128 行的 M-major fp32 存储所能达到的极限。

---

## 10.14 如果你自己写这个角色

必须做对的八条不变量，按你将遇到它们的顺序排列：

1. 四个连续的 warp，其索引覆盖 mod 4 的全部余数，每个只读取自己的 TMEM 象限：`lane_off = (32 (warp % 4)) << 16`。任何其他映射都无法寻址。
2. 用一个包含负责分配的 warp 在内的命名屏障等待 TMEM 基址（`bar.sync id, 32 + 128` 对应它的 `bar.arrive`），然后再读取基址；绝不要硬编码 0。
3. 每个 tile 在你自己的 `acc_full[st]` 上做一次 `try_wait` peek，阻塞 wait 放在第一次加载之前；`tcgen05.ld` 的地址是 warp-uniform 的，`16 e` 通过展开成为立即数。
4. 在该 tile 的最后一次加载之后：`tcgen05.wait::ld`（ISA 所要求的反依赖 fence：这些加载的寄存器 consumer 不需要 wait，而 producer 的下一条 MMA 对 TMEM 的覆写需要），然后每个线程都在 *producer* 的 `acc_empty[st]` 上 arrive（针对该 pair 的、经 peer 位掩码的地址，计数 = 两个 CTA 的全部 consumer 线程），然后推进状态。在剩余的算术之前释放。
5. 把 alpha 和 beta 保留为 kernel 参数；用 `c = 0` 计算 `fmaf(beta, c, alpha * acc)`，绝不要只算 `alpha * acc`。
6. 从你交给 TMA 描述符的 layout（原子、平铺顺序、作用于绝对地址位的 swizzle）推导出 smem 字节公式，在编译期用 `cute::Swizzle` 核对它，并把 stage 基址对齐到 swizzle 周期（`SWIZZLE_128B_ATOM_32B` 为 512 B，`SWIZZLE_128B` 为 1024 B）。C++ 通过通用指针解引用；asm 操作数则通过 32 位的 `shared::cta` 地址。
7. 每个 subtile：所有线程执行 `fence.proxy.async.shared::cta`；`bar.sync`；由一个 warp 发射这些 box（uniform 操作数，不需要选举）、`commit_group`、`wait_group.read (Stages_D - 1)`；在任何人写入下一个 stage 之前再次 `bar.sync`。这个 wait 是线程范围的；发布它的是屏障。
8. 在退出时，baseline 留下一个组未被等待；在你自己的 kernel 中，请在负责存储的 warp 退出之前加上 `cp.async.bulk.wait_group 0`。

---

# 第 11 章 框架：kernel prologue、分发与主机端启动镜像

第 7-10 章讲完了四个角色函数；本章是围绕它们的框架：在角色开始之前运行的一切，以及主机端上让 kernel 得以启动的一切。

---

## 11.1 我们在文件中的位置

```
explicit.cu
  ...
  1082  } // namespace (anonymous)              <- roles (Chapters 7-10) end here
  1085  // 10. [PROD] The kernel                 <- THIS CHAPTER, 11.2-11.7
  1095  __global__ void __launch_bounds__(256, 1)
  1096  explicit_blackwell_fp16_gemm_kernel(const __grid_constant__ ExplicitGemmParams params) {
  ...
  1203  }
  1206  // 11. [HOST] [PROD] run()               <- 11.8
  1212  #if defined(CUTLASS_DEABSTRACTION_TRACE)  (watchdog, 1212-1263)
  1265  cutlass::Status run(ExplicitGemmParams const& params) {
  1329  }
  1332  // 12. [HOST] [TRACE] trace_begin / trace_end / mismatch dump
  1337  #if defined(CUTLASS_DEABSTRACTION_TRACE)
  1449  #endif
  1451  } // namespace explicit_gemm
```

该 kernel 位于 `namespace explicit_gemm` 中，在 explicit.cu:1082 处闭合的匿名命名空间之外。有两处从该命名空间之外引用它：位于 1275 的主机端函数 `run()` 取其地址，而 trace 构建打印其名字（`TRACE_HOST explicit_kernel_name explicit_blackwell_fp16_gemm_kernel`，explicit.cu:1352）并为运行后的属性探针取其地址 (explicit.cu:1443)。kernel 所调用的一切（W1-W36 wrapper、辅助函数和角色函数；W37 是 `__syncwarp()` 内建函数）都位于匿名命名空间中并且是 `__forceinline__` 的，所以编译后的 kernel 是单个函数，不再残留任何源码级调用（笔记 G.3：1640 条 SASS 指令，`STACK 0`）；其 SASS 中仅有的两条 `CALL`，位于 0x5b70 和 0x5ba0，是 ptxas 插入的 `tcgen05.dealloc` 一致性检查的失败路径，其目标是位于 0x6530 和 0x6560 的 `BPT.TRAP` 桩（第 12.2 节）。

---

## 11.2 签名 (explicit.cu:1095-1098)

```cpp
// explicit.cu:1095-1098
__global__ void __launch_bounds__(256, 1)
explicit_blackwell_fp16_gemm_kernel(const __grid_constant__ ExplicitGemmParams params) {
  extern __shared__ char smem[];
  uint32_t const smem_base = smem_u32(smem);                                                   // 0x400 + (rank << 24) measured (D.1 D5); computed, never assumed
```

这四行里装进了四个决策。前三个照搬了 CUTLASS 的 `device_kernel` 所做的事（`include/cutlass/device_kernel.h:112-126`）；第四个，即 11.2.4 中的 `smem_u32` 转换，在那里没有对应物：`device_kernel` 把通用的 `smem` 指针传下去，而 CUTLASS 在使用点通过 `cute::cast_smem_ptr_to_uint` 进行转换（`cute/arch/util.hpp:93-108`；`barrier.h`、`sm100_pipeline.hpp`）：

```cpp
// device_kernel.h:112-126 (verbatim)
template <typename Operator>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
{
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();

}
```

（`CUTLASS_GLOBAL` 是 `__global__ static`，`include/cutlass/detail/helper_macros.hpp:67`；explicit kernel 省略了 `static`，所以它的符号具有外部链接；这并不是 `run()` 或 trace 构建所强制的，它们从同一个翻译单元 (11.1) 引用它，在内部链接下同样可以这样做。）

### 11.2.1 `__launch_bounds__(256, 1)`

`MaxThreadsPerBlock = 256` 和 `MinBlocksPerMultiprocessor = 1` 是 baseline kernel 的值（`sm100_gemm_tma_warpspecialized.hpp:144-147`：`32 + 32 + 32 + 32 + 128 = 256` 个线程，每个 SM 一个块）。CUDA Programming Guide（附录 "Launch Bounds"，v13 版指南的第 5.4.3.2 节）陈述了该属性的作用："If launch bounds are specified, the compiler first derives the upper limit, `L`, on the number of registers that the kernel should use. This ensures that `minBlocksPerMultiprocessor` blocks (or a single block, if `minBlocksPerMultiprocessor` is not specified) of `maxThreadsPerBlock` threads can reside on the multiprocessor"。多于 `maxThreadsPerBlock` 个线程的启动会失败。

这个 kernel 的算术：SM 有 64 K = 65536 个 32 位寄存器（Blackwell Tuning Guide；Programming Guide 中针对 CC 10.x 的 "Compute Capabilities" 表）。因此一个 256 线程的块每线程可以使用 `65536 / (256 x 1) = 256` 个寄存器，而硬件把它封顶在每线程最大值 255（同一张表；Blackwell Tuning Guide："The maximum number of registers per thread is 255"）。推导出的上限 `L` 就是架构最大值，所以该属性没有施加任何硬件本来不施加的约束。实测 kernel 使用 26 个寄存器（笔记 G.3，`REG:26`），baseline 为 68。

值 1 是精确的，而不是一个保守的下界。230400 字节的动态共享内存加上系统为每个块保留的 1 KB，在 SM 的 233472 字节（228 KB；每块的可选启用最大值是 227 KB = 232448 字节：笔记 D.5 `device_max_smem_per_block_optin 232448`、`device_reserved_smem_per_block 1024`；Blackwell Tuning Guide）中消耗了 231424 字节。所以无论寄存器如何，至多只能容纳一个 CTA（笔记，第 6.2 节）：`2 x 231424 = 462848 > 233472`。

### 11.2.2 `const __grid_constant__ ExplicitGemmParams params`

参数块是**按值**传递的（在 CUDA 13.3 上为 640 字节，explicit.hpp:74；笔记 G.1 `sizeof_ExplicitGemmParams 640`；G.3 `CONSTANT[0]` = 0x380 + 640）。三类设备端 consumer（总共七个取地址表达式）取它的**地址**：

| 位置 | 表达式 | 指令 |
|---|---|---|
| explicit.cu:1114-1115, 1118-1119 | `&params.tma_a` ... `&params.tma_d` | `prefetch.tensormap [addr]` (W20) |
| explicit.cu:747, 754（producer_role，第 8 章） | `&params.tma_a`、`&params.tma_b` | `cp.async.bulk.tensor.3d ... [addr, {...}]` (W21, W22) |
| explicit.cu:1066（epilogue_role，第 10 章） | `&params.tma_d` | `cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [addr, ...]` (W23) |

TMA 单元通过该地址读取 128 字节的 `CUtensorMap`。PTX ISA 把 tensor map 定义为一个 128 字节的不透明对象，位于 `.const`、`.param`（kernel 参数）或 `.global` 空间中，而 `prefetch.tensormap` 从 `.const` 或 `.param` 空间读取它（PTX ISA，`prefetch`；`cp.async.bulk.tensor`；该 ISA 语句在 11.4.3 中被引用）。因此该地址必须指名**原始的**参数空间对象。

`__grid_constant__` 使之成立。CUDA Programming Guide（附录 5.4.1.5，"`__grid_constant__` Parameters"）允许在 `const` 限定的非引用 kernel 参数上使用该注解。有了它，编译器不会创建每线程的局部副本，kernel 中的所有线程看到同一个地址，而修改该对象是未定义行为。hpp 的字节映射 (explicit.hpp:18-29) 记录了该块落在 SASS 中的位置：它从 `c[0x0][0x380]` 开始，所以 `k` 被读作 `ld.param.b32 [+520]` -> `c[0x0][0x588]`，`alpha, beta` 被读作 `ld.param.v2.b32 [+528]` -> `c[0x0][0x590]`/`[0x594]`，而 D 描述符的地址是 `c[0x0][0x348] + 0x180`（笔记 G.3；baseline 的是 `c[0x0][0x348] + 0x680`，笔记 D.2）。

**没有它会发生什么（我的分析，源自该指南的措辞）。** 对一个普通的按值参数取地址会强制在局部内存中产生一份每线程的副本，而 TMA 会被交给一个局部内存的通用地址，它不在 ISA 允许 tensor map 所处的三个状态空间中的任何一个里（第 5 章，第 5.3.2 节）。实测的 `LOCAL:0 STACK:0`（笔记 G.3）就是不存在副本的证据。

`const` 不是装饰：该指南只允许在 `const` 限定的参数上使用 `__grid_constant__`。`run()` 在主机端剥去常量性，仅仅是为了满足 `cudaLaunchKernelExC` 的 `void**` 签名（explicit.cu:1311，`const_cast<void*>(static_cast<void const*>(&params))`）。

### 11.2.3 `extern __shared__ char smem[]` 与缺失的对齐属性

整个 230400 字节的 working set 都是**动态**共享内存，在启动时请求（`launch_config.dynamicSmemBytes = kSmemBytes`，explicit.cu:1302）。翻译单元中不存在其他 `__shared__` 声明：`grep -n __shared__` 找到 explicit.cu:1097 处的声明和 1085-1086 处的横幅注释，别无其他。CUTLASS 也是这样做的（`device_kernel.h:121`；笔记，第 6.2 节：`cudaFuncGetAttributes` 报告 `sharedSizeBytes 0`）。

为什么没有静态共享内存，声明上也没有 `__align__`：

1. **系统保留的那个千字节决定了基地址。** 动态共享内存第一个字节的 32 位 `shared::cta` 地址在每个 CTA 中都是 `0x400 + (rank << 24)`（`0x00000400`、`0x01000400`、`0x02000400`、`0x03000400`；实测，笔记 D.1 第 D5 项，K0 和 K1 一致；ptxas 从 `SR_CgaCtaId` 构造它，第 2 章，第 2.9.2 节）。它下方的那个千字节是系统保留的共享内存，`cuobjdump` 把它报告为 `SHARED:1024`，`tcgen05.alloc` 的下沉用它来做其 pair 握手（笔记 D.2，"Resource facts"），而它使每块的 opt-in 最大值是 227 KB 而不是 SM 的 228 KB（"CUDA reserves 1 KB of shared memory per thread block"，Blackwell Tuning Guide；第 5 章，第 5.7.1 节）：`0x400` 基址、`SHARED:1024` 资源行和 util.hpp:46 处的 `static_assert(kSmemBytes <= 232448)` 是同一个事实的三种视图。rank 位于地址的字节 3，也就是位 [24,32)（编译器下沉，笔记 D.2；第 2.9.2 节；对一个 4-CTA cluster 只有位 24-25 变化，笔记 D.1 D5），而清除位 24 的 peer 位掩码 `0xFEFFFFFF` (explicit.cu:187) 依赖于这一放置（第 4 章，第 8 章）。

2. **每个 `OFF_*` 都保持其对齐。** 以 `0x400`（1024 对齐）为基址，`smem_base + OFF_SMEM_A = 0x400 + 33792 = 0x8800` 和 `smem_base + OFF_SMEM_B = 0x400 + 164864 = 0x28800` 都是 1024 对齐的，正如 SWIZZLE_128B 的 TMA 写入和 UMMA 描述符的 SBO/LBO 算术所要求的那样（笔记第 7.5 节），而 `smem_base + OFF_SMEM_CD = 0x600` 是 512 对齐的（SWIZZLE_128B_ATOM_32B，笔记第 7.7 节）。explicit.cu:162-163 处的那些 `static_assert` 检查这些偏移。基址的对齐是一个这些断言看不到的启动属性。这就是为什么 1098 处的注释说 "computed, never assumed"：`smem_base` 总是在运行时从 `__cvta_generic_to_shared` 获得。

3. **14 位的 UMMA 起始地址字段不可能溢出。** smem 描述符存储 `(addr >> 4) & 0x3FFF`，也就是说它可寻址 2^18 = 262144 字节（第 9 章）。explicit.cu:165 处的 `static_assert(kSmemBaseLow24 + kSmemBytes < (1u << 18))` 检查 `0x400 + 230400 = 231424 < 262144`。静态共享内存相对于动态缓冲区会放在哪里不是源码级的契约（NVIDIA 内部），所以任何静态声明都可能把基址从实测的 `0x400` 挪开（我的分析；见第 4 项）。

4. **静态 `__shared__` 会造成什么。** 假设 kernel 声明了 `__shared__ uint32_t flag;`。`sharedSizeBytes` 将不再是 0，而动态基址可能从实测的 `0x400` 挪开：静态 `__shared__` 对象彼此之间以及相对于动态缓冲区的放置不是源码级的契约（NVIDIA 内部），所以这个移动量无法从声明中读出。（该指南只说了所有 `extern __shared__` 数组都从同一个地址开始。）`OFF_SMEM_A` 随之可能失去其 1024 字节对齐，TMA 的 SW128 写入和 UMMA 的读取会在 swizzle 相位上不一致，而 `D` 就会出错（我对后果的分析）。这就是为什么 kernel 不声明任何静态 `__shared__`，而是在运行时读取基址（第 2 项）。此外 `230400 + static` 不得超过 232448 字节的 opt-in 上限，所以至多也只有 2048 字节的静态数据能放得下。CUTLASS 通过把每个屏障、响应槽位和指针都放进同一个动态 `SharedStorage` 结构体来避免这一切（笔记，第 6.2 节的表）；explicit kernel 保留了那些完全相同的偏移 (explicit.cu:139-155)。

该声明不带任何对齐属性，正如 `device_kernel.h:121` 也不带，也正如笔记 E.6.7 所要求的那样（"no `__align__` on the dynamic buffer"）。一个 `extern __shared__` 声明上的对齐属性是否会移动动态缓冲区，在这里所用的资料中没有记载（未验证）。这对本 kernel 无关紧要：在 `sharedSizeBytes 0` 下基址就是实测的 `0x400`，kernel 交给硬件的每一个地址都是从 `smem_base` 计算出来的整数，而编译器从不需要对 `smem` 作对齐假设。

### 11.2.4 `smem_u32` 与地址空间规则

```cpp
// explicit.cu:258
__device__ __forceinline__ uint32_t smem_u32(void const* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
```

`smem` 是一个**通用**指针（64 位，可供 C++ 加载和存储使用）；`smem_base` 是每个 `.shared::cta`/`.shared::cluster` PTX 操作数都需要的 **32 位共享窗口地址**。在文件头部 (explicit.cu:22-23) 和笔记 E.6.1 中陈述过一次的那条规则：

| 用途 | 形式 | 示例 |
|---|---|---|
| 任何指名共享内存的 asm 操作数（`mbarrier.*`、`cp.async.bulk.tensor.*`、`tcgen05.*`、`clusterlaunchcontrol.*`、`mapa`） | `smem_base + OFF_*`（`"r"` 约束） | 第 7-10 章的全部 |
| 任何对共享内存的 C++ 解引用 | `smem + OFF_*`（通用指针） | TMEM 基址读取 `*reinterpret_cast<uint32_t const*>(smem + OFF_TMEM_BASE_PTR)` (explicit.cu:888, 998)；epilogue 的 16 条 `st.shared.b32` 写作 `*reinterpret_cast<uint32_t*>(smem + ...)` (explicit.cu:1047) |

把两者混用不是一个细微的 bug：用一个已经包含共享窗口偏移的值去索引通用的 `smem`，在 rank 0 中会相差 `0x400`，在 rank 1-3 中会相差 `0x01000400`、`0x02000400`、`0x03000400` (16-48 MiB)（笔记 E.6.1）。kernel 把 `smem` 和 `smem_base` 两者都传给需要 C++ 解引用的那两个角色（`mma_role`、`epilogue_role`，explicit.cu:1198, 1201），而只把 `smem_base` 传给另外两个角色。

阅读 PTX 时的一个编译器细节（笔记 B.6、D.2）：nvcc 不为这次转换发出任何 `cvta.to.shared` 指令。它将其折叠进 `mov.b32 %r, smem` + `add.s32 offset`，而在 SASS 中基址是字面量 `0x400 | (SR_CgaCtaId << 24)`。

---

## 11.3 步骤 1：身份 (explicit.cu:1100-1107)

```cpp
// explicit.cu:1100-1107
  // 7.3 step 1: warp index, election, cluster identity (kernel :417-430)
  int const warp_idx = warp_idx_sync();
  uint32_t const lane = threadIdx.x & 31u;
  uint32_t const lane_pred = elect_one_sync();
  uint32_t const rank = cluster_ctarank();
  bool const is_cta0 = (rank == 0u);
  bool const is_leader = ((rank & 1u) == 0u);                                                   // cta_coord_v = rank % 2 == 0
  uint32_t const peer = rank ^ 1u;
```

"kernel :417-430" 即 `sm100_gemm_tma_warpspecialized.hpp:417-430`：

```cpp
// sm100_gemm_tma_warpspecialized.hpp:418-430 (verbatim)
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;

    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;
```

| 变量 | Wrapper / PTX | 本次运行中的值 | 为什么是这种形式 |
|---|---|---|---|
| `warp_idx` | W1 `__shfl_sync(0xffffffff, static_cast<int>(threadIdx.x / 32), 0)` (explicit.cu:261; `cutlass.h:127-133`) | 0..7 | 见下文 |
| `lane` | `threadIdx.x & 31` | 0..31 | 只有 scheduler（lane 0..3 装填 CLC 屏障，第 7 章）和 epilogue（行的所有权，第 10 章）需要它 |
| `lane_pred` | W2 `elect.sync` (explicit.cu:264-278; `cluster_sm90.hpp:180-197`) | 每个 warp 恰好有一个 lane 为 1 | prologue 中的那一次选举，在 baseline 复用它的地方被复用（11.3.2） |
| `rank` | W3 `mov.u32 %0, %cluster_ctarank` (explicit.cu:281) | 0..3 | 给出 CTA 在 cluster 中的线性 rank 的 PTX 特殊寄存器；实测 `rank = x + 2y`，其中 `(x, y) = %cluster_ctaid`（笔记第 7.1 节；K0 探针在 `launch0.csv` 中的 `cluster_ctaid` 字段以及 D.1 D8 的 per-rank TMA 坐标证实了这一点；第 2.1.2 节） |
| `is_cta0` | `rank == 0` | 每个 cluster 中有一个 CTA 为 true | CLC scheduler warp 以及 CLC/throttle `empty` 屏障的所有者位于 CTA 0 中（第 7 章） |
| `is_leader` | `(rank & 1) == 0` | rank 0、2 | baseline 中的 `cta_coord_v = rank % size<0>(AtomThrID) = rank % 2`：pair 中偶数号的 CTA 发射 2SM MMA（第 9 章） |
| `peer` | `rank ^ 1` | 1、0、3、2 | MMA pair 中的另一个 CTA；只用于 TMEM dealloc 握手（`mbar_arrive_remote(..., peer)`，explicit.cu:964、970） |

### 11.3.1 为什么 warp 索引要经过一次 shuffle

`threadIdx.x / 32` 在一个 warp 的每个 lane 中本来就相同，但编译器无法证明这一点：它看到的是一个每线程的表达式。`__shfl_sync(0xffffffff, v, 0)` 在每个 lane 中都返回 lane 0 的值，所以结果**从构造上就是 warp-uniform 的**，而全掩码则在该点强制 warp 收敛。CUTLASS 自己的注释就是这么说的（`cutlass.h:124-125`："Returns a warp-uniform value indicating the canonical warp index of the calling threads. Threads within the warp must be converged."）。回报体现在 SASS 中：prologue（四个 init 组）中和分发处的每一个普通的 `if (warp_idx == N)` 都变成一条 *uniform* 分支（`UISETP`/`BRA.U`；同时还测试 `lane_pred` 的那三个守卫，explicit.cu:1113、1117、1165，保留一条 per-lane 的 `ISETP ... !P0`），而角色选择则变成一个 uniform 寄存器上的三入口跳转表（`BRXU UR10`，第 11.7.2 节）。explicit kernel 为此发出**一条** `SHFL.IDX`。baseline 的 PTX 有九处 `shfl.sync.idx.b32`：kernel 主体中一处，每个流水线构造函数一到两处，因为 `PipelineTmaAsync` 基类构造函数运行 `canonical_warp_idx_sync()`，而 `init_barriers` 又运行了一次（笔记 B.5 和 B.2 步骤 12）。九处中有七处在 SASS 中以 `SHFL.IDX` 的形式留存下来（笔记 D.2 "7 of 9 PTX sites"；G.3 "one `SHFL.IDX` for the warp index instead of seven"）。

### 11.3.2 prologue 中的那一次选举，以及新的选举在哪里进行

`elect.sync d|p, membermask` 确定性地从 `membermask` 中选举出一个 leader lane（"the same leader thread is elected for the same membermask every time"），而它强制性的 `.sync` 使它成为一个 warp 收敛点（PTX ISA 9.7.15.15，在第 2 章，第 2.10.3 节中引用）。*究竟是哪个* lane 并未规定：实测，在选举之下写出的全部 1184 条 trace 记录中都是 lane 0（笔记 B.2 步骤 4，其措辞 "may differ between calls" 已被 ISA 原文取代；实践规则不变：永远不要把 lane 0 硬编码）。因此 kernel 中的每一个 `elect_one_sync()`（W2，explicit.cu:264-278，掩码始终为 `0xFFFFFFFF`）也都是一个收敛点，这是角色函数可以依赖整个 warp 在每次选举时都聚在一起的原因之一，而 `lane_pred`（explicit.cu:1103）之所以保持有效，是因为没有任何角色函数有 lane 发散的出口（笔记 E.6.3 "warp-convergence contract"）。它恰好在 baseline 复用 `lane_predicate` 的地方被**复用**，而**新的**选举则恰好在 baseline 的辅助函数自行选举的地方进行：

| 位置 | 选举 | Baseline 锚点 | 为什么 |
|---|---|---|---|
| explicit.cu:1113、1117 处对 A、B（warp 1）和 C、D（warp 3）的预取 | 复用 `lane_pred` | kernel `:440, :443` `&& lane_predicate` | 每个 CTA 每个描述符一条 `prefetch.tensormap`；任何 lane 都可以 |
| explicit.cu:1165 `tmem_dealloc` 的 init（warp 0） | 复用 `lane_pred` | kernel `:564` `if (has_mma_peer_cta && lane_predicate)` | 一条 `mbarrier.init` |
| explicit.cu:1192 producer 的 `is_leader_lane` = `lane_pred != 0u && is_leader` | 复用 `lane_pred` | kernel `:465` `is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load` | 每 k-tile 一次的 `arrive.expect_tx`（第 8 章）在整个 kernel 期间都由 pair leader 的 warp 2 的这一个 lane 发射 |
| explicit.cu:1125、1136、1146、1156 处的四个 `mbarrier.init` 组 | 每组各一次**新的** `elect_one_sync()` | `initialize_barrier_array_pair_aligned` 自己做 `if(cute::elect_one_sync())`（`barrier.h:134`、`:146`） | PTX 层面的 parity：baseline 的构造函数看不到 `lane_predicate` |
| explicit.cu:740 处 TMA 的 A+B 对（第 8 章） | 新的，每对一次 | `sm100_mma_warpspecialized.hpp:619-622` | 一次选举守卫两次加载 |
| explicit.cu:932 处的每一条 `tcgen05.mma`（第 9 章） | 新的 | `mma_sm100_umma.hpp:563-586`（选举在 CUTLASS 的 `fma` 内部） | 每条 MMA 一次选举 |
| explicit.cu:942、944 处的每一条 `tcgen05.commit` | 新的 | `barrier.h:848-861` | 每次 commit 一次选举 |
| explicit.cu:843 处的每一次 CLC 查询（第 7 章） | 新的 | cb424739 处的 `sm100_tile_scheduler.hpp:445-446`（工作树中为 `:445` 和 `:452`） | 每次 `try_cancel` 一次选举 |

在功能上，为 init 组复用 `lane_pred` 会是等价的（笔记 E.6.3 步骤 1 这么说），而 ISA 的确定性保证使这一点严格成立：这个 kernel 中的每一次选举都使用同一个 `membermask` `0xFFFFFFFF`，所以来自 prologue 的 `lane_pred` 与某个角色中任何一次新的 `elect_one_sync()` 都会选中**同一个** lane。“复用”与“重新选举”只在指令流上有差别，在由哪个 lane 执行这项工作上从无差别。新的选举之所以存在，是为了让 PTX 清单与 baseline 的契约相匹配（explicit kernel 中有 14 处 `elect.sync`，笔记 G.3）。循环内的选举没有一处在 SASS 中留存下来：ptxas 把它们折叠掉了，因为被守卫的指令（`UTMALDG`、`UTCHMMA`、`UTCBAR`、`UGETNEXTWORKID`）是 uniform 数据通路指令，本来就每个 warp 只发射一次（笔记 D.6 第 6 项，G.3：`ELECT` 4 处，全部在循环之外）。同一个确定性为这种折叠提供了依据（我的分析）。无论如何都要写出 PTX 形式：它是文档化的契约，而且在 ptxas 之后它不花费任何代价。

---

## 11.4 步骤 2：标量、k-tile 计数、描述符预取 (explicit.cu:1109-1120)

```cpp
// explicit.cu:1109-1120
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
```

### 11.4.1 `alpha`, `beta`

每个线程都从参数空间把两者各读一次（`ld.param.v2.b32 [+528]`，一条指令读两个；笔记 G.3），而只有 epilogue 使用它们（`t = alpha * acc; d = fmaf(beta, 0.0f, t)`，explicit.cu:1045-1046，第 10 章）。它们是有意设为**运行时**值的（笔记 D.7 "Arithmetic"）。若用编译期的 `1.0f`/`0.0f`，编译器会把 `1.0f * acc` 折叠为 `acc`（一个精确的恒等式），并把 `fmaf(0.0f, 0.0f, t)` 折叠为至多一条计算 `+0.0f + t` 的 `FADD`（而不是折叠为 `t`：`+0.0 + (-0.0) = +0.0`，所以没有 fast-math 时 nvcc 不能去掉这次加法）；这一折叠是推断，而非实测（笔记第 0 节和 D.7）。与 baseline 相匹配的每 tile 128 条 `FMUL` + 128 条 `FFMA` 将从 explicit kernel 的指令流中消失。在那种折叠下 `D` 会**保持逐位相同**（baseline 自己的 `FFMA` 计算的是同一个 `+0.0 + t`）；失去的将是笔记 D.7 所要求的指令流 parity（在 G.3 中验证），以及随之而去的、把两个 epilogue 逐操作码进行比较的能力。第 10 章讲了 `-0.0` 的算术以及 C 项用运行时 `beta` 保留为一条 FFMA 的原因。baseline 在 epilogue collective 的构造函数中读取同样的两个值（kernel `:437`）。笔记 B.2 步骤 9 列出了 baseline 首先执行的 stride 与指针测试；explicit 结构体没有这些，因为它没有指针路径（explicit.cu:1271 处的主机端守卫拒绝非空的 `alpha_ptr`/`beta_ptr`）。

**ptxas 如何处理这两次加载（实测，笔记 G.3 和 D.2）。** 尽管 PTX 在 prologue 中读取 `alpha, beta`，ptxas 却把 explicit kernel 中的这次加载下沉到 epilogue 入口：一条 `LDCU.64 UR26, c[0x0][0x590]`（`params + 528`），`alpha` 在 `UR26` 中，`beta` 在 `UR27` 中，以 `FMUL Rk, Rk, UR26` 和 `FFMA Rk, RZ, UR27, Rk` 的形式使用。baseline 则在 kernel 入口处加载：`alpha` 进入一个向量寄存器（`LDC R2, c[0x0][0x8a0]`，以 `FMUL Rk, R2, Rk` 的形式使用），`beta` 进入一个 uniform 寄存器（`LDCU UR21, c[0x0][0x880]`）。当你阅读自己改写版本的 SASS 时，要预期这次加载出现在 ptxas 找到最短活跃区间的地方，而不是 C++ 放置它的地方；两种放置方式都对相同的操作数计算相同的 IEEE 运算。

### 11.4.2 `k_tiles = (params.k + 63) >> 6`

每个输出 tile 中 64 宽的 k-tile 的数量，即 `ceil(K / kTileK)`，其中 `kTileK = 64`（util.hpp:30）：`(8192 + 63) >> 6 = 8255 >> 6 = 128`。它是向上取整的除法，因为不是 64 的倍数的 K 仍然需要它最后一个不完整的 k-tile（TMA 单元对越界元素零填充，编码中的 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`，笔记第 5.3 节）。baseline 在 `PersistentTileSchedulerSm100::get_work_k_tile_count` 中把它计算为 `ceil_div(get<2>(problem_shape), get<2>(tile_shape))`（cb424739 处的 `sm100_tile_scheduler.hpp:502-506`，工作树中为 `:533-537`），而它的 SASS 在 `0x1640-0x1700` 处计算一次 `(K + 63) >> 6`（笔记 D.2）。对于这次固定的运行，主机端守卫把 `k == 8192` 作为前提条件（explicit.cu:1268），所以 `k_tiles` 始终是 128。设备端代码保留了通用公式，以便其指令流与 baseline 完全一样地读取 `k`（`ld.param.b32 [+520]`，设备端读取的问题 shape 的唯一字段；笔记 G.3：`m`、`n`、`l` 从未被读取）。一处实测的差别：baseline 为了它的 `k_tile_count >= 1` 测试在**每个 tile** 上都从 `c[0x0][0x38c]` 重新读取 `K`（笔记 D.2，`0x3770` 处的重新加载），而 explicit kernel 把 `k_tiles` 保留在 uniform 寄存器 `UR14` 中，在 `0x0a70` 处加载一次，供整个 producer 和 MMA 角色使用（从不需要它的 epilogue 代码复用了 `UR14`）。这就是为什么它的累加器 empty 等待与 commit 块是 15 条指令，而 baseline 是 18 条（笔记 G.3）。笔记 E.6.6 两者都允许（"`k_tiles` may be held in a register (the baseline re-reads `K` per tile; either is fine, D.2)"）；这是少数几个被允许的指令流差异之一。

### 11.4.3 warp 1 和 3 执行的 `prefetch.tensormap`

- **为什么要预取。** 其 wrapper 是 W20（第 6.6.2 节）。第一条指名某个描述符的 `cp.async.bulk.tensor` 在能够计算哪怕一个地址之前，必须先把 128 字节的 tensor map 从参数空间取入 tensor-map 描述符缓存（ISA 没有给它命名），而不是 L2 数据缓存（NVIDIA 内部）。`prefetch.tensormap` 在 prologue 中就启动了这次取回，远早于 producer 发射它的第一次加载。175 条指令的 prologue 的其余部分（笔记 G.3）、cluster 会合、`griddepcontrol.wait` 以及第一次 `empty[]` 等待全都位于两者之间。（这段距离本身没有实测；没有任何 trace 记录覆盖 `prefetch.tensormap`，也没有任何记录能观察到它的完成：这条指令是每线程的、尽力而为的，并且没有完成机制（NVIDIA 内部）。）这是 PTX 文档所记载的该指令的用途（PTX ISA，`prefetch{.tensormap_space}.tensormap [a]`，`.tensormap_space = {.const, .param}`："brings the cache line containing the specified address in the .const or .param memory state space for subsequent use by the cp.async.bulk.tensor instruction"）。kernel 使用的是通用地址形式（没有 `.const`/`.param` 后缀；"If no state space is given, the prefetch uses Generic Addressing"），与 CUTLASS 的做法完全一致，因此该通用地址必须解析到参数空间中的对象。nvcc 通过在取 `&params.tma_*` 地址的四个代码位置（warp 1 和 warp 3 的预取块、producer、epilogue；位于非零偏移处的 map 随后以 `add.s64` +128/+256/+384 的形式跟上；笔记 G.3："four `cvta.param` for the map addresses"）各发出一条针对参数块的 `cvta.param`，把这一点明确化：这是 PTX 层面的证据，表明第 11.2.2 节中 `__grid_constant__` 的地址就是参数空间中的对象而不是一份副本。
- **为什么是 warp 1 和 3。** 从 kernel `:439-445` 复制到 explicit.cu:1113-1120：`Sched` warp 预取 mainloop 的描述符，`EpilogueLoad` warp 预取 epilogue 的描述符，在**每个** CTA 中都如此（warp 1 不只在 rank 0 中，warp 3 也不只在 `beta != 0` 时）。warp 1 在 prologue 中本来就空闲，warp 3 只发射 throttle 的 `mbarrier.init` 组（explicit.cu:1155-1163），而 rank 1-3 的 warp 1 以及各处的 warp 3 都在分发处退出，所以这次预取对于本来什么都不做的线程而言是免费的工作。（动机是我的分析；事实源自源码。）
- **为什么 `beta = 0` 从不读取 C 却仍要预取它。** 为了与 baseline 的四条 `UTMACCTL.PF` 保持一致（笔记 D.2：warp 1 为 A、B 执行的 `0x0310`/`0x0320` 和 warp 3 为 C、D 执行的 `0x03d0`/`0x03e0`；笔记 D.7 和 E.10 第 7 项使四次预取的形式成为默认）。explicit kernel 的 SASS 有同样的四条（笔记 G.3，`UTMACCTL.PF 4`），其 PTX 清单也保持 `prefetch.tensormap` 为 4。去掉 `tma_c` 对正确性无害；保留它是为了让编译产物的比较是同类相比。这次预取的代价是在 prologue 中对 tensor-map 描述符缓存里 tensor map 所在的那一行做一次尽力而为的填充，而不是一条 L2 数据缓存行（NVIDIA 内部），并且让 C map 为未来的 `beta != 0` 路径保持可用。一个从零开始的 `beta = 0` kernel 可以去掉 C map 及其预取，并把参数块缩减到 512 字节（第 5.3.2 节）。
- **为什么用 `lane_pred`。** 每个 CTA 每个描述符预取一次就足够了；由被选中的 lane 发射它，正如第 11.3.2 节选举表的第一行那样（kernel `:440, :443`）。

描述符在设备端从不被修改（笔记 7.3 步骤 2："descriptors are never modified on the device"；笔记 D.2 和 G.3 的 PTX 清单中没有 `tensormap.replace`），所以在入口处每个描述符预取一次就是 kernel 所需的全部：持久化循环中不出现任何重新预取，也不出现任何 `fence.proxy.tensormap`（四条 `prefetch.tensormap` 全部位于 prologue 中，explicit.cu:1114-1119）。

---

## 11.5 步骤 3：33 次 `mbarrier.init` 调用 (explicit.cu:1122-1167)

```cpp
// explicit.cu:1122-1167
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
```

`mbar_init` 是 W5 (explicit.cu:286-294)，即 `ClusterBarrier::init`（`barrier.h:391-403`）的 asm 字符串：`mbarrier.init.shared::cta.b64 [addr], count;`。一个刚初始化的 mbarrier 处于 phase 0，有 `count` 个待到达，tx-count 为零，后者稍后由 `arrive.expect_tx` 在事务屏障上提高；第 4 章解释了为什么面对这样的屏障，producer 从 parity 1 开始而 consumer 从 parity 0 开始。ISA 对操作数施加两条约束（PTX ISA，`mbarrier.init` 与 "Size and alignment of mbarrier object"）。对于默认的 `.layout::v0` 对象，`count` 必须落在 `[1, 2^20 - 1]` 之内，所以 800 和 256 远未触及该上限。（`count` 操作数是 32 位的，但其值必须放得进这一界限背后的硬件字段/范围（NVIDIA 内部）。笔记 D.2 的 SASS 初始化字 `{(0x100000 - count) << 1, (0x100000 - count) << 11}` 与这个 20 位界限相一致：它保存的是 `2^20 - count`，可读作该计数在一个 20 位字段中的补码；这一解读是从 SASS 推断出来的，与 4.2.1 和 12.2 中一样，而 ISA 只陈述了范围。）并且该对象是一个不透明的 `.b64`，必须在 `.shared` 中按 8 字节对齐，下面的每个 `OFF_*` 都满足这一点（全部是 8 的倍数）。

### 11.5.1 512 字节控制区的映射

每次 init 都在 `smem_base + OFF_* + 8 i` 处写入 8 字节。该区域（笔记，第 6.2 节；常量位于 explicit.cu:139-152）：

```
offset  bytes  object                         count     init warp   this kernel
   0     64    main full[0..7]                1 (+tx)   0           yes
  64     64    main empty[0..7]               2         0           yes
 128     32    epi_load full[0..3]            32 (+tx)  1           NO  (beta != 0 only)
 160     32    epi_load empty[0..3]           128       1           NO
 192     16    load_order[0..1]               32        3           NO
 208     16    clc full[0..1]                 1 (+tx)   4           yes
 224     16    clc empty[0..1]                800       4           yes
 240     32    acc full[0..3]                 1         5           yes
 272     32    acc empty[0..3]                256       5           yes
 304     16    throttle full[0..1]            32        3           yes
 320     16    throttle empty[0..1]           32        3           yes
 336      8    tmem_dealloc                   32        0           yes
 344      8    (pad: PipelineStorage is aligned_struct<16>)
 352     32    clc_response[0..1]             16-byte CLC responses (written by hardware)
 384      4    tmem_base_ptr                  written by tcgen05.alloc
 388    124    (pad to 512: tensors need 512-byte alignment)
 512 ...       smem_C/D, smem_A, smem_B       (Chapters 8-10)
```

这 43 个 mbarrier 槽位占据字节 0..343（344 字节）。CUTLASS 的 `PipelineStorage` 连同其填充是 352 字节（笔记 D.1 D2，`PipelineStorage 352`）。控制对象在 388 处结束（`OFF_TMEM_BASE_PTR + 4`），而填充到 512 则为第一个张量 stage 提供了其 512 字节对齐。（explicit.cu:59 处的文件头部写的是 "barriers 0..391"：那是 388 向上舍入到下一个 8 的倍数，这是我的解读；这个数字在笔记中的任何地方都没有推导。）已初始化的屏障：`8 + 8 + 2 + 2 + 4 + 4 + 2 + 2 + 1 = 33`。baseline 初始化 `33 + 4 + 4 + 2 = 43` 个（笔记 D.2：43 条 `SYNCS.EXCH.64`）。

### 11.5.2 每个计数从何推导

这些常量及其单行推导位于 explicit.cu:180-185（第 4 章，第 4.4 节逐一解读了它们）；explicit.cu:192 处的 `static_assert(kMainloopTxBytes == 49152 && kClcEmptyCount == 32 + 4 * (32 + 128 + 32))` 钉住了最容易弄错的那两个。

### 11.5.3 为什么是 33 而不是 43

baseline 的 `epi_load` 流水线（`PipelineTransactionAsync<4>`，计数 32/128，初始化 warp 为 1，kernel `:474-487`）和 `load_order` 屏障（`OrderedSequenceBarrier<1,2>`，计数 32，warp 3，kernel `:494-499`）服务于 C 的加载路径（`beta != 0`），而本 kernel 省略了这条路径（第 3 章）。当 `beta = 0` 时，它们的十个屏障**被初始化了但从未在其上到达或等待过**（笔记 7.2 的表，"initialized but never used" 各行）。explicit kernel 把那十个 8 字节槽位留作保留的、未初始化的空间（explicit.cu:1134 处的注释）。没有任何东西读取未初始化的槽位。fence 的数目随之而定：这里 4 组，baseline 中 6 组（笔记 E.6.3 步骤 3）。笔记 E.10 第 6 项把这记录为推荐的变体；D.7 两者都接受。

错误的 `clc_empty` 计数的失效模式是不对称的（我的分析）。计数偏少会在每个 consumer 都释放该槽位之前就完成 phase，于是多出的到达溢入下一个 phase，而一个响应槽位可能在某个 warp 仍在解码它时就被重新装填。计数偏多则永远完不成 phase，并使 CTA 0 的 scheduler warp 挂起在它下一次 `clc_empty` 等待处（第 7 章）。

### 11.5.4 为什么 fence 由全部 256 个线程执行，以及它对什么排序

这个 fence 是 W14（第 6.4.1 节），复制自 `cutlass::arch::fence_barrier_init`，`barrier.h:711-724`，其位于 `barrier.h:707-710` 的注释说："Helps with visibility of barrier init operations across warps / cta / cluster ... It must be composed with an appropriate sync instruction with the right scope to ensure visibility eg. `__syncthreads()` or a `cluster_arrive()` + `cluster_wait()`"。

**它对什么排序。** `fence.mbarrier_init` 是一种受限的 fence，其同步效果只适用于执行线程先前在 `.shared::cta` 屏障上的 `mbarrier.init` 操作，并以 release 语义使之在 cluster 作用域可见（PTX ISA `.op_restrict`，引文见第 2 章，第 2.10.1 节）。它与步骤 6 中 `barrier.cluster.wait` 的 acquire 配对：init (release.cluster) -> arrive -> 每个 CTA 等待 (acquire) -> 每个 CTA 的线程现在可以在任何 CTA 的屏障上 arrive。这就是 ISA 自己针对 relaxed cluster 屏障给出的示例的形状（两者都在第 2.10.1 节中引用），先用一个 fence 发布、然后 relaxed 地 arrive，而 `fence.mbarrier_init.release.cluster` 是就 `mbarrier.init` 存储而言满足这一点的最窄 fence：它不覆盖任何其他东西，并且如下所示，不发出任何 SASS 指令。

**为什么尽管每组只有一个 lane 做了初始化，所有线程仍都执行它（源自源码 + 我的分析）。** 在每一个由 mbarrier 支撑的 CUTLASS 流水线构造函数中，fence 都位于 `if (warp_idx == initializing_warp)` 测试**之外**：`sm100_pipeline.hpp:559-573`（mainloop）、`:139-149`（累加器，`PipelineUmmaAsync`）、`:963-979` / `:981-1000`（CLC）、`sm90_pipeline.hpp:1040-1055`（throttle，`PipelineAsync`）。因此每个 CTA 的全部 256 个线程在每个构造函数中都执行它一次，而 explicit kernel 复制了这一形状（explicit.cu:1133, 1144, 1154, 1164 处的四次无条件调用）。三个原因：

1. 这个 fence 是一个按线程进行的操作，对没有发射过任何 init 的线程来说是空操作。因此把它设为无条件的不花费任何代价，还避免了第二次选举或一条发散的分支。
2. *确实*做了 init 的线程必须在其 `barrier.cluster.arrive` **之前** fence。无论被选中的是哪个 lane，紧跟在每个 init 组之后的一个 fence 都保证了这一点。
3. 在 SASS 中这条指令根本不发出任何东西（笔记 D.2 和 7.3 步骤 3："`fence.mbarrier_init.release.cluster` emits no instruction"；在各 init 组之前看到的裸 `FENCE.VIEW.ASYNC.S` 被归属于 init 的下沉，推断）。baseline 在其各 init 组之前有七条这样的裸 `FENCE.VIEW.ASYNC.S`（笔记 D.2）；explicit kernel 的 prologue 有五条：四个 init 组每组之前各一条，以及 `tmem_dealloc` init 之前一条，后者没有自己的 fence（`trace_out_v2/sass_explicit.txt` 0x0290, 0x0460, 0x0570, 0x0680, 0x0780；笔记 G.3 数出同样的五条，比 baseline 的七条少两条）。

explicit.cu:1165-1167 处的 `tmem_dealloc` init 没有自己的 fence，因为 baseline 在那里也没有（kernel `:562-576` 之后直接就是 `:580` 处的 `pipeline_init_arrive_relaxed`）。这个 init 位于第四个 fence *之后*，所以没有任何 `fence.mbarrier_init` 覆盖它，而那个 arrive 是 `.relaxed` 的。它是唯一一个**没有**遵循 ISA 的 "fence prior to arrive" 模式的 init，所以就我对 PTX 内存模型所能读出的而言，prologue 并没有在形式上确立它对 peer 在 explicit.cu:964 处的远程 arrive 的可见性（未验证）。可以确定的是：这一排序是 baseline 自己的（源自源码），explicit kernel 复制了它，在这个屏障上的第一次远程 arrive 发生在 kernel 退出时，大约在 init 之后 0.9 ms，而两个 kernel 的任何已记录的运行都没有在那里挂起过（实测，笔记 G.1）。如果你在重写 kernel 时想要一个可证明有序的变体，把 `tmem_dealloc` init 移到第四个 fence 之前；协议不变。

### 11.5.5 为什么是这个顺序和这些 warp

这些组按 baseline 构造函数的顺序出现：mainloop（kernel `:457-472`）、[epi_load `:474-487`、load_order `:494-499`，已省略]、CLC（`:501-518`）、累加器（`:520-536`）、throttle（`:538-550`）、tmem_dealloc（`:562-565`）。每条流水线的初始化 warp 是 baseline 的 `initializing_warp` 参数：0（`:467`）、4（`:517`）、5（`:531`）、3（`:549`），以及 `tmem_dealloc` 的 warp 0 的 `lane_predicate`（`:564`）。

**从功能上说这些都无关紧要。** 一条 `mbarrier.init` 就是执行它的任一线程对编码后初始化字的一次 8 字节写入。在 SASS 中它是 `SYNCS.EXCH.64 URZ, [UR29 + off], R`（`UR29` 保存 baseline 的 smem 基址；explicit kernel 的是 `UR7`，第 12 章），其中 `R = {(0x100000 - count) << 1, (0x100000 - count) << 11}`，即 11.5 的初始化字（笔记 D.2）：这是对不透明 mbarrier 对象的一次同步单元交换（在 CUDA Binary Utilities 手册的 Blackwell 指令集表中，`SYNCS` 是 "Sync Unit"），而不是一条 `STS`。该交换是原子的：它读出旧字、写入所提供的字并返回旧字（NVIDIA 内部），这里返回到 `URZ` 中，所以旧字被丢弃（`trace_out_v2/sass_explicit.txt` 0x02a0）。ISA 要求该对象由 `mbarrier.init` 创建，而不能由其他任何方式创建。这个操作没有跨线程依赖，所以任何线程都可以初始化任何屏障，只要 (a) 每个屏障在其第一次 arrive 之前恰好被初始化一次，并且 (b) 在任何远程 CTA 能在其上 arrive 之前，init 已被设为可见，而 fence + cluster 会合对它们全体一并保证了这一点。baseline 把 43 次 init 分摊到 warp 0、1、3、4、5 上，使它们并行进行，而不是在一个 warp 中串行化 43 次存储（动机方面是我的分析；映射本身源自源码）。explicit kernel 为保持一致而保留了这一映射，使 prologue 的 SASS 可以逐行比较（笔记 G.3："33 `SYNCS.EXCH.64` inits with the same init words at the same Section 6.2 offsets as the baseline's corresponding 33 of 43"）。如果你重写 kernel，可以把全部 33 次 init 放在 warp 0 的一个被选中的 lane 中，后面跟一个 fence；协议不变。

---

## 11.6 步骤 4-6：拆分的 cluster 会合与初始 tile (explicit.cu:1169-1179)

```cpp
// explicit.cu:1169-1179
  // 7.3 step 4: cluster rendezvous, part 1 (kernel :580)
  cluster_arrive_relaxed();

  // 7.3 step 5: warp-uniform setup; the initial tile from the launch coordinate (swizzle_and_rasterize + AlongN transpose, sm100_tile_scheduler.hpp:365-371, 666-677)
  uint32_t const x = cluster_ctaid_x();                                                        // == rank & 1
  uint32_t const y = cluster_ctaid_y();                                                        // == rank >> 1
  int tm = static_cast<int>(2u * (blockIdx.y >> 1) + (blockIdx.x & 1u));
  int tn = static_cast<int>(2u * (blockIdx.x >> 1) + (blockIdx.y & 1u));

  // 7.3 step 6: cluster rendezvous, part 2 (kernel :615); no remote barrier operation before this point
  cluster_wait();
```

W16/W17 (explicit.cu:428-429) 是 `barrier.cluster.arrive.relaxed.aligned;` 和 `barrier.cluster.wait.aligned;`，复制自 `cute::cluster_arrive_relaxed` / `cute::cluster_wait`（`cluster_sm90.hpp:48-55, 66-73`）。baseline 通过 `pipeline_init_arrive_relaxed(cluster_size)` 和 `pipeline_init_wait(cluster_size)`（`sm90_pipeline.hpp:1377-1384`、`:1364-1371`）到达它们，这两者在 cluster 大小为 1 时回退为 `__syncthreads()`；explicit kernel 知道自己的 cluster 是 4，于是直接调用 cluster 形式。（ptxas 仍然在两个 kernel 中都把两者包进一个运行时测试 `c[0x0][0x36c] == 1`，并带有 `NOP` / `BAR.SYNC.DEFER_BLOCKING 0x0` 回退：笔记 D.2 和 G.3；PTX 中没有任何守卫。）

### 11.6.1 `barrier.cluster` 做什么，以及为什么用 `.relaxed` + 拆分

- **为什么用 `.relaxed`。** 一个 `.release` 的 arrive 会对该线程先前的其他所有存储排序（它们都无关紧要）并付出一个更重的 fence 的代价。CUTLASS 选择了 `.relaxed`，并相应地给辅助函数命名（`pipeline_init_arrive_relaxed`）。（源自源码和 PTX 语义；代价判断是我的分析。）
- **为什么把 arrive 与 wait 拆开。** 在 arrive 与 wait 之间，baseline 中有：`load_init`（TMA 坐标张量）、流水线起始状态、`%cluster_ctaid`、两个 multicast 掩码、带有初始 tile 的 scheduler 对象，以及 TMEM 累加器张量（kernel `:582-613`；笔记 B.2 步骤 21-25）。explicit kernel 中有：两次 `%cluster_ctaid` 读取和两行 tile 坐标（其余的变成了编译期常量或每个角色的局部变量）。cluster 中最慢的 CTA 还在完成它的各次 init 时，其他的就在计算这些，所以会合的延迟与设置重叠，而不是叠加其上。对 explicit kernel 来说，这个窗口只容纳寥寥几条指令，但这一形状被保留下来，因为它是正确的、免费的，并使 SASS 可比（笔记 G.3：`UCGABAR_ARV` ... `UCGABAR_WAIT; CCTL.IVALL` 处于与 baseline 相同的、由 ptxas 插入的 `c[0x0][0x36c] == 1` 守卫之下）。
- **为什么用 `.aligned`。** 实现从 `.aligned` 这一承诺中得到什么（第 2.10.1 节）没有文档记载；此时到达按 warp 而不是按线程处理，这是我的推断，而非 ISA 的陈述。kernel 所复制的 cute 辅助函数（11.6）使用 `.aligned` 形式，所以 explicit kernel 也这样做。
- **`CCTL.IVALL` 是什么（NVIDIA 内部信息和我的分析，已如此标注；第 12 章有该清单）。** SASS 把 `UCGABAR_WAIT` 与一条 `CCTL.IVALL` 配对（笔记 G.3）。它是由编译器的下沉所发出的一个全部失效的缓存控制操作，这里位于 cluster 屏障的 wait 之后；不应把它读作普通的程序数据流或一个通用的 CTA 屏障，而它使哪个缓存域失效并不是一个稳定的 SASS 契约（NVIDIA 内部）。PTX 只请求了该 wait 的默认 `.acquire`（PTX ISA 9.4，9.7.15.3）；这个失效是实现那个 acquire 的方式，这是我的分析，而非 ISA 的陈述。

### 11.6.2 从 `%cluster_ctaid` 得到 `(x, y)`

W4 (explicit.cu:282-283) 读取 `%cluster_ctaid.x` 和 `%cluster_ctaid.y`（`cute::block_id_in_cluster`，`cluster_sm90.hpp:126-137`）。PTX ISA 把 `%cluster_ctaid` 定义为 "the CTA identifier in a cluster in each dimension"，每个 CTA 唯一，且以 `0 <= %cluster_ctaid.x < %cluster_nctaid.x` 为界（y、z 同理），并把 `%cluster_ctarank` 定义为 "unique CTA identifier across all dimensions"。对于 (2,2,1) cluster，`x, y` 各为 0 或 1，且 `rank = x + 2y`（笔记第 7.1 节；K0 探针在 `launch0.csv` 中的 `cluster_ctaid` 字段以及 D.1 D8 的每个 rank 的 TMA 坐标证实了这一点，第 2.1.2 节）。

本章所依赖的第二个恒等式是 `x = blockIdx.x & 1`、`y = blockIdx.y & 1`，即 `%cluster_ctaid == blockIdx mod clusterDim`。它由两个事实推出。CUDA Programming Guide 描述了 cluster 是如何形成的："Specifying clusters groups adjacent thread blocks into clusters"、"Thread blocks within a cluster are always adjacent to each other within the grid"、"Specifying clusters does not change the grid dimensions or the indices of a thread block within a grid"（Programming Guide 1.2.2.1.1 "Thread Block Clusters"）。而 `check_cluster_dims` 强制要求每个 cluster 维度都整除网格（11.8.3）。所以网格被相邻 `blockIdx` 组成的 2 x 2 块平铺，而在一个 tile 内的位置就是 `blockIdx mod 2`。该指南和 ISA 都没有把这个模关系写成公式；实测证实了它（K0 探针在全部 16 个 CTA 中记录到 `cluster_ctaid.x/.y == (blockIdx.x & 1, blockIdx.y & 1)` 和 `rank = x + 2y`，而且 `launch0.csv` 中的每条记录都有 `rank == (bx & 1) + 2 (by & 1)`；笔记 D.3：全部 991 个有效的 CLC 首 CTA id 在两个坐标上都是偶数，即每个 cluster 都从一个偶数 `blockIdx` 开始；1024 个响应中的其余 33 个是未取消的，且不携带有效 id）。注释 `// == rank & 1` 和 `// == rank >> 1` 记录了这一恒等式。kernel 仍然读取特殊寄存器，而不是从 `rank` 推导 `x, y`，baseline 也是如此。（在 SASS 中，特殊寄存器读取变成了对 `SR_CTAID.X/Y` 的算术运算并配合一个常量 bank 表，归因属推断，笔记 D.2 以及 D.6 第 14 项 —— "irrelevant to the protocol"。）`x` 和 `y` 供给 A 的 multicast 掩码 `0x5 << x` 以及 A/B 的坐标与目的地（第 8 章）、累加器 commit 掩码 `0x3 << 2y`（第 9 章），以及 CLC 的 tile 映射（第 7 章，explicit.cu:692-693）。

### 11.6.3 从启动坐标推导 `tm`、`tn`

explicit.cu:1175-1176 处的两个公式是一条始于主机端的链的终点。每个环节都在源码中。

**环节 1：逻辑 CTA 网格是 (M 方向的 tile 数, N 方向的 tile 数, L)。** `GemmUniversalAdapter::run` 调用 `get_grid_shape(params)`（`gemm_universal_adapter.h:381`）-> `GemmUniversal::get_grid_shape`（`sm100_gemm_tma_warpspecialized.hpp:383-396`）-> `TileScheduler::get_grid_shape`（`sm100_tile_scheduler.hpp:220-232`）-> `get_tiled_cta_shape_mnl`（`:593-605`）：

```cpp
// sm100_tile_scheduler.hpp:597-600 (verbatim)
    auto [tiles_m, tiles_n, tiles_l] = product_each(ceil_div(select<0,1,3>(problem_shape_mnkl), take<0,2>(tile_shape_mnk)));
    auto ctas_m = round_nearest(tiles_m * size<0>(atom_thr_shape_mnk), size<0>(cluster_shape_mnk));
    auto ctas_n = round_nearest(tiles_n * size<1>(atom_thr_shape_mnk), size<1>(cluster_shape_mnk));
    auto ctas_l = tiles_l;
```

kernel 以 `TileShape{} = (256, 128, 64)` 和 `AtomThrShapeMNK{} = (2, 1, 1)` 调用它（`sm100_gemm_tma_warpspecialized.hpp:389-393`）。在 `tile_shape_mnk = (256, 128, 64)`（**pair** 的 MMA tile，第 3 章）、`atom_thr_shape = (2, 1, 1)` 和 `cluster = (2, 2, 1)` 下：`tiles_m = ceil(8192/256) = 32`，`ctas_m = round_nearest(32 * 2, 2) = 64`；`tiles_n = ceil(8192/128) = 64`，`ctas_n = round_nearest(64 * 1, 2) = 64`；`ctas_l = 1`。逻辑网格 `(64, 64, 1)`，按 (M, N, L) 索引。M 上的 `x 2` 与 N 上的 `x 1` 就是 `AtomThrShape`：一个 2SM MMA 原子沿 M 跨越两个 CTA，所以 256 行的 MMA tile 是两个 128 行的 CTA tile，而沿 N 一个 CTA 拥有全部 128 列。笔记 7.4 陈述了同样的结果，"ctas_m = round_up(ceil_div(8192, 256) * 2, 2) = 64, ctas_n = round_up(ceil_div(8192, 128), 2) = 64"，而笔记 13 第 5 项记录了这是 "`ceil_div(M, 256) * 2` (MMA tile, then times AtomThrShape), not `ceil_div(M, 128)`"；这里完全相同。有了这两个公式，你就能为任何其他的问题 shape 重现出网格。

**环节 2：所启动的网格针对 AlongN 被转置。** `possibly_transpose_grid`（`:235-250`）：

```cpp
// sm100_tile_scheduler.hpp:238-248 (verbatim)
    if (raster_order == RasterOrder::AlongN) {
      // Swap grid.x and grid.y for AlongN rasterization order, since the CLC scheduler
      // will schedule in AlongM order by default.
      //
      // Each grid dimension must also be a multiple of the corresponding cluster dimension,
      // so we convert the untransposed x into the number of clusters along the M mode,
      // and multiply this by cluster.n (and vice-versa for y).
      auto tmp = grid.x;
      grid.x = divmod_cluster_shape_n.divide(grid.y) * divmod_cluster_shape_m;
      grid.y = divmod_cluster_shape_m.divide(tmp) * divmod_cluster_shape_n;
    }
```

`grid.x = (64 / 2) * 2 = 64`，`grid.y = (64 / 2) * 2 = 64`。对于方形问题，数字不变，但**含义**变了：此后，`blockIdx.x` 行走于 N 方向的 tile，而 `blockIdx.y` 行走于 M 方向的 tile。这里 `raster_order` 是 `AlongN`，因为启发式规则 `tiles_n > tiles_m ? AlongM : AlongN`（`tile_scheduler_params.h:338-343`，在 `RasterOrderOptions::Heuristic` 之下）在数量相等时给出 `AlongN`（笔记 13 第 4 项；实测 `raster_order AlongN 1`，笔记 D.1 D10、D.5 H4）。它所比较的数量是 *cluster*-tile 的数量，而不是 CTA-tile 的数量：`PersistentTileSchedulerSm100Params::initialize` 设置 `problem_tiles_m_ = problem_blocks.x / cluster_shape.m()` 和 `problem_tiles_n_ = problem_blocks.y / cluster_shape.n()`（`tile_scheduler_params.h:1920-1921`），而 `initialize_swizzle` 调用 `get_rasterization_order(problem_tiles_m_, problem_tiles_n_, raster_order_option)`（`:1881`）：这里是 64 / 2 = 32 对 64 / 2 = 32。紧接着在 `:1882-1896` 有第二道守卫。AlongN 转置把 M 方向的 cluster 放到网格 y 上，其上限是 65535，所以 `new_grid_y = problem_tiles_m_ * cluster_shape.n()` 会与 `(1 << 16) - 1` 比对，若会溢出则顺序回退为 AlongM。这里 32 x 2 = 64 不会触发它（笔记 7.4："the 65535 grid-Y guard not triggering"）。

**为什么 AlongN 对这个 GEMM 是正确答案（我的分析，建立在实测的 D.3 之上）。** CLC 大致按递增的启动线性顺序分发 cluster id（笔记 D.3）。网格被转置后，连续的 id 在一行 cluster tile 之内沿 **N** 行走，所以在任一时刻，33 个驻留 cluster 在处理两个或三个相邻的 256 x 8192 fp16 A 面板（每个 4 MiB；实测，笔记 D.3：从不只在一个上），而 A 大约每一行 32 个 cluster tile 才从 HBM 重新读取一次，而不是每个 tile 一次：这就是第 3 章，第 3.8.2 节的 L2 复用论证（第 7 章，第 7.5.6 节给出了实测的条带 occupancy，而第 7 章把同一个 `(tm, tn)` 映射施加于 CLC 响应）。该启发式规则沿较短的 tile 轴光栅化，并在方形问题中取 `AlongN` 作为打破平局的选择（第 3 章，第 3.8.2 节）。`enum class RasterOrder { AlongM, AlongN }`（`tile_scheduler_detail.hpp:38-41`）正是主机端守卫与字面量 1 比对的原因（`kRasterOrderAlongN = 1`，util.hpp:36），而 explicit 主机端函数把结果硬编码为 `dim3 const grid(kGridX, kGridY, kGridZ)`，即 (64, 64, 1)（explicit.cu:1282；util.hpp:32 `kGridX = kFixedN / kCtaTileN, kGridY = kFixedM / kCtaTileM`：x 由 N 算出而 y 由 M 算出，即转置后的顺序）。

**环节 3：在设备端，`initial_work_tile_info` 把 `blockIdx` 送入 `swizzle_and_rasterize`。**

```cpp
// sm100_tile_scheduler.hpp:366-371 (verbatim)
  template <class ClusterShape>
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    return swizzle_and_rasterize(blockIdx.x, blockIdx.y, blockIdx.z, /*valid=*/true, /*cluster_offset_m=*/0, /*cluster_offset_n=*/0);
  }
```

在 `swizzle_and_rasterize`（`:697-814`）中，整个 swizzle 块（`:711-795`）都被跳过，因为 `divmod_swizzle_size_.divisor == 0`（"Swizzling is enabled if the swizzle size is greater than 0"，`:710`；实测 `swizzle_divisor 0`，笔记 D.1 D10；explicit.cu:1269 处的主机端守卫强制了这一点）。剩下的就是转置（`:796-802`，引用于第 7.5.1 节）。

对于初始 tile，两个 cluster 内偏移都是 0，而 `possibly_transpose_work_tile` 位于 `:668-677`（引用于第 7.5.1 节）。

代入 `M_idx = blockIdx.x`、`N_idx = blockIdx.y`，两个 cluster 除数都为 2：

```
cluster_m   = blockIdx.x / 2      remainder_m = blockIdx.x % 2
cluster_n   = blockIdx.y / 2      remainder_n = blockIdx.y % 2
tm = M_idx' = cluster_n * 2 + remainder_m = 2 (blockIdx.y >> 1) + (blockIdx.x & 1)
tn = N_idx' = cluster_m * 2 + remainder_n = 2 (blockIdx.x >> 1) + (blockIdx.y & 1)
```

这正是 explicit.cu:1175-1176（笔记 B.4，第一行）。可以这样读它：**cluster 索引**来自转置后的网格轴（`blockIdx.y / 2` 选出 M 方向的 cluster 行，`blockIdx.x / 2` 选出 N 方向的 cluster 列），而 **cluster 内的位置**仍跟随原来的轴（`blockIdx.x % 2 = x` 选出 M 方向的那一半，`blockIdx.y % 2 = y` 选出 N 方向的那一半）。因此，对于一个 CTA 收到的每一个 tile，都有 `tm % 2 == x`（对 CLC tile 也是如此，第 7 章），这正是使 "CTAs x and x+2 share tm, hence the A rows" 成立的原因（第 8 章）。

### 11.6.4 算例

| `blockIdx` (x, y) | `x = bx & 1`、`y = by & 1` -> rank | `by >> 1`、`bx >> 1` | `tm = 2 (by >> 1) + (bx & 1)` | `tn = 2 (bx >> 1) + (by & 1)` | D 的行 / 列 |
|---|---|---|---|---|---|
| (5, 2) | x = 1, y = 0 -> rank 1 | 1, 2 | 2 * 1 + 1 = **3** | 2 * 2 + 0 = **4** | 行 [384, 512)、列 [512, 640) |
| (4, 2) | x = 0, y = 0 -> rank 0 | 1, 2 | 2 | 4 | 行 [256, 384)、列 [512, 640) |
| (4, 3) | x = 0, y = 1 -> rank 2 | 1, 2 | 2 | 5 | 行 [256, 384)、列 [640, 768) |
| (5, 3) | x = 1, y = 1 -> rank 3 | 1, 2 | 3 | 5 | 行 [384, 512)、列 [640, 768) |
| (0, 0) | rank 0 | 0, 0 | 0 | 0 | 行 [0, 128)、列 [0, 128) |
| (63, 63) | x = 1, y = 1 -> rank 3 | 31, 31 | 63 | 63 | 行 [8064, 8192)、列 [8064, 8192) |
| (1, 0) | rank 1 | 0, 0 | 1 | 0 | 行 [128, 256)、列 [0, 128) |
| (0, 1) | rank 2 | 0, 0 | 0 | 1 | 行 [0, 128)、列 [128, 256) |

前四行是一个 cluster 的四个 CTA（包含 `blockIdx` (4..5, 2..3) 的那个 cluster）。它们合起来覆盖位于 cluster 坐标 (M 方向 cluster 1, N 方向 cluster 2) 处的 256 x 256 cluster tile：

```
                 N tiles:   tn = 4            tn = 5
                          cols 512..639     cols 640..767
              +------------------+------------------+
 tm = 2       |  rank 0 (x0,y0)  |  rank 2 (x0,y1)  |   rows 256..383
 rows 256..   |  blockIdx (4,2)  |  blockIdx (4,3)  |
              +------------------+------------------+
 tm = 3       |  rank 1 (x1,y0)  |  rank 3 (x1,y1)  |   rows 384..511
 rows 384..   |  blockIdx (5,2)  |  blockIdx (5,3)  |
              +------------------+------------------+
                 pair {0,1}: tn = 4         pair {2,3}: tn = 5
                 (same y, adjacent tm)      (same y, adjacent tm)
```

MMA pair `{0,1}` 和 `{2,3}` 是这幅图的两**列**（相同的 y，因而相同的 tn，相邻的 tm）：一条 `tcgen05.mma.cta_group::2` 覆盖该 pair 的 256 行 x 128 列（第 9 章）。rank 0 和 rank 2 有相同的 A 行（tm = 2），这就是 multicast 关系 `0x5 << x`（第 8 章）。

### 11.6.5 4096 个启动坐标恰好覆盖 4096 个 CTA tile 各一次

下面两个映射在 `{0..63} x {0..63}` 上都是良定义的，因为只要 `v, w` 落在 `{0..63}` 中，`2 (v >> 1) + (w & 1)` 就也落在其中。定义

```
F : (bx, by) -> (tm, tn) = (2 (by >> 1) + (bx & 1),  2 (bx >> 1) + (by & 1))
G : (tm, tn) -> (bx, by) = (2 (tn >> 1) + (tm & 1),  2 (tm >> 1) + (tn & 1))
```

在 F 之后施加 G：`bx' = 2 (tn >> 1) + (tm & 1) = 2 (bx >> 1) + (bx & 1) = bx`（因为 `tn >> 1 = bx >> 1` 且 `tm & 1 = bx & 1`），以及 `by' = 2 (tm >> 1) + (tn & 1) = 2 (by >> 1) + (by & 1) = by`。所以 `G o F = id`；由同样的计算可得 `F o G = id`。因此 F 是从 64 x 64 网格到 64 x 64 tile 空间的一个双射：4096 个 CTA tile 中的每一个都恰好是一个 CTA 的**初始** tile。（CLC 的分发随后在被取消的 cluster 的首 CTA id 上复用同一个映射，第 7 章；它对 1024 个 cluster tile 的恰好一次覆盖已被实测，笔记 D.7 "CLC"。）

### 11.6.6 不变量：`cluster_wait` 之前没有远程操作

在一个 CTA 自己通过 `barrier.cluster.wait` 之前（它只有在 cluster 的每个线程都执行了 `barrier.cluster.arrive` 之后才完成，PTX ISA 9.7.15.3，也就是在每个 CTA 的各个 `mbarrier.init` 和 fence 之后），它不得触碰另一个 CTA 的共享内存或屏障：另一个 CTA 可能还没有执行它的各个 `mbarrier.init`（它的 warp 甚至可能还没有在运行），而对一个未初始化的 mbarrier 的 arrive 是未定义的。explicit kernel 的远程操作全部位于角色函数之内（在 explicit.cu:1179 之后）：

| 操作 | Wrapper | 位置 | 目标 |
|---|---|---|---|
| 经由 `mapa` 的远程 `mbarrier.arrive` | W10 | explicit.cu:684（CLC 释放）、837（throttle 释放）、964/970（dealloc 握手） | CTA 0 的 `clc_empty`、CTA 0 的 `throttle_empty`、peer 的 `tmem_dealloc` |
| 经由 peer 位掩码对 pair leader 的屏障的 arrive | W11 | explicit.cu:1039 | leader 的 `acc_empty[st]` |
| 经由 `mapa` 的远程 `arrive.expect_tx` | W13 | explicit.cu:842 | CTA 0..3 的 `clc_full[i]` |
| 2SM TMA 加载（A 数据进入 rank `x` 和 `x+2`，B 数据进入本 CTA 自己的 smem；tx 字节计入 leader 的 `full[s]`） | W21, W22 | explicit.cu:747, 754 | rank `x`、`x+2`（A 数据）；pair leader 的 `full[s]` |
| `tcgen05.commit ... multicast::cluster` | W30 | explicit.cu:942, 944 | 全部四个 CTA 的 `empty[s]`；该 pair 的 `acc_full[st]` |
| `clusterlaunchcontrol.try_cancel ... multicast::cluster::all` | W33 | explicit.cu:847 | 每个 CTA 的 `clc_response[i]` 和 `clc_full[i]` |
| `tcgen05.alloc/dealloc.cta_group::2` | W26, W27 | explicit.cu:885, 971 | 该 pair 的 TMEM（SASS 握手使用 peer 的保留 smem，笔记 D.2） |

第 1179 行之前的一切都是 CTA 本地的：参数读取、特殊寄存器、33 次本地初始化、各个 fence、relaxed arrive。这就是 explicit.cu:1178 处的那句话（"no remote barrier operation before this point"）以及笔记 B.2 步骤 26。

---

## 11.7 步骤 7：分发与贯穿而过的 warp (explicit.cu:1190-1203)

```cpp
// explicit.cu:1190-1203
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
```

baseline 的链是 `if (is_participant.main_load) ... else if (is_participant.sched) ... else if (is_participant.mma) ... else if (is_participant.epi_load) ... else if (is_participant.epilogue)`，其中 `is_participant = {mma: warp 0, sched: warp 1 && rank 0, main_load: warp 2, epi_load: warp 3 && is_epi_load_needed, epilogue: warps 4-7}`（kernel `:449-455`）。explicit kernel 去掉了 `epi_load` 分支（对 `beta = 0` 永远不会走到），并以同样的顺序保留了其他四个。

| 分支 | Warp | 角色（章） | 值得注意的参数 |
|---|---|---|---|
| `warp_idx == 2` | 每个 CTA 的 2 号 | `producer_role`（8） | `is_cta0`（throttle producer 仅在 CTA 0 中）；`lane_pred != 0u && is_leader` = pair leader 的 warp 中发射 `arrive.expect_tx 49152` 的那一个 lane |
| `warp_idx == 1 && is_cta0` | rank 0 的 1 号 | `scheduler_role`（7） | `lane`（lane 0..3 装填四个 CTA 的 `clc_full`） |
| `warp_idx == 0` | 每个 CTA 的 0 号 | `mma_role`（9） | `smem`（generic，用于 TMEM 基址的读取）、`is_leader`、`peer`、`lane_pred`（仅 trace） |
| `warp_idx >= 4` | 每个 CTA 的 4-7 号 | `epilogue_role`（10） | `smem`、`warp_idx`（-> `w = warp_idx - 4`）、`lane`、`alpha`、`beta` |
| （无） | 每个 CTA 的 3 号；rank 1-3 的 1 号 | 贯穿而过到 1203 处的 `}` 并退出 | |

除 `params`（kernel 参数块，按 `const&` 传递）和 `smem`（动态共享内存数组）之外的每个参数都是在 prologue 中计算出的一个值；各角色不接收任何指向 prologue 状态的指针，除共享内存外不共享任何东西。`tm, tn` 是初始 tile；每个角色在 CLC 循环中各自向前携带自己的副本（第 7 章）。

### 11.7.1 为什么贯穿而过的 warp 可以提前退出

每个 CTA 中有一个 warp（warp 3），rank 1-3 中还有第二个（warp 1），它们在第 1203 行离开，而其余的则运行 ~0.9 ms。rank 0 失去一个 warp，rank 1-3 失去两个，所以 cluster 的 32 个 warp 中有 `7 + 3 x 6 = 25` 个留下，这恰好就是 CLC empty 计数的 `25 x 32 = 800`（11.5.2）：每个留下的 warp 都消费 CLC 响应，而没有任何消费的 warp 离开。这一退出是安全的，因为**没有任何同步对象计入离开者**：

| 对象 | 成员 | 是否包含 warp 3？rank 1-3 的 warp 1？ |
|---|---|---|
| `bar.sync 6, 160` / `bar.arrive 6, 160`（TMEM alloc 交接，explicit.cu:887, 997；`kTmemAllocBarThreads = 160` 位于 explicit.cu:191） | 32（warp 0 执行 arrive）+ 128（warp 4-7 执行 sync）= 160 = `NumMMAThreads + NumEpilogueThreads`（kernel `:558`） | 否 |
| `bar.sync 1, 128`（epilogue subtile 屏障，explicit.cu:1051, 1071） | 128 个 epilogue 线程 | 否 |
| `barrier.cluster` | cluster 的全部 1024 个线程，它们在任何人退出之前就在 1179 处**已经通过**（而且 ISA 反正只计入未退出的线程，第 2.10.1 节） | （已通过） |
| `clc_empty` 800 | `32 + 4 * (32 + 32 + 128)`：rank 0 的 warp 1，每个 CTA 的 warp 0、2、4-7 | 否：`NumSchedThreads` 只被计一次，而不是 `cluster_size` 次，并且 `NumEpilogueLoadThreads` 只在 `is_epi_load_needed` 时才计入 |
| `main full/empty`、`acc full/empty`、`throttle`、`tmem_dealloc` | 见 11.5.1 以及第 4 章，第 4.4 节 | 否 |

一个退出的 warp 并不结束 CTA（CTA 保持驻留，直到它的最后一个 warp 退出），不释放共享内存，也不干扰命名屏障，后者只计入那些以 `bar.sync`/`bar.arrive` 指名了它们的线程。除了最后一个 bulk 存储组的退出条件（第 4.8.5 节；笔记 D.1 D15）之外，这个 kernel 中唯一的退出冒险是笔记第 7.10 节的那个：一个 CTA 不得在远程 CTA 仍可能到达**它的**屏障时退出。处理这一点的是角色内部的四个尾段（第 7-9 章），而不是框架。rank 1-3 的 warp 1 在离开之前，于 1113-1116 处完成它唯一有用的工作，即 A/B 预取；warp 3 同样预取 C/D (1117-1120) 并初始化 throttle 屏障 (1155-1163)。

### 11.7.2 ptxas 将分发编译成了什么

笔记 G.3（实测）：explicit 二进制文件在常量 bank 2 中带有一个三项跳转表（`CONSTANT[2]:12`）：`UISETP.LT.U32 UP0, UR20, 0x2; USEL UR4, UR20, 0x2, UP0; USHF.L.U32 UR4, UR4, 0x2; LDCU UR10, c[0x2][UR4]; BRXU UR10` 位于 `0x0a60-0x0ae0`。索引 `min(warp_idx, 2) * 4` 为 warp 0、warp 1 或 warp >= 2 选出目标。warp >= 2 的目标又被 `UISETP.NE.U32.AND UP0, UPT, UR20, 0x2; BRA.U` 再拆分一次（warp 2 对其余），而 `BRA.U` 的目标 `0x1ee0` 处放着 `warp_idx >= 4` 测试（`UISETP.GE.AND UP0, UPT, UR20, 0x4, UPT; PLOP3.LUT P0, PT, PT, PT, UP0, 0x80, 0x8`；实测，`trace_out_v2/sass_explicit.txt`；源码是 explicit.cu:1200），它把 epilogue warp 与 `0x1f00` 处的 `EXIT` 分开，G.3 把后者归于 rank 1-3 的 warp 1 和 warp 3。（G.3 没有为 explicit kernel 引述那条比较指令，而笔记 D.2 引述了 baseline 的那条：rank 1-3 的 warp 1 和 warp 3 "traverse the dispatch chain to `UISETP.GE.AND UP0, UR18, 0x4; @!P0 EXIT`"，位于 `0x6580`/`0x65a0`。跳转表第 1 项通向 `0x19a0` 这一点是推断的（表的内容不在转储中）；`warp_idx == 1 && is_cta0` 测试中的 `is_cta0` 那一半在那里可见（`UISETP.NE.U32.AND UP0, UPT, UR15, URZ, UPT; BRA.U UP0, 0x1ee0` 位于 `0x19a0`/`0x19b0`，`UR15` = `SR_CgaCtaId`，即 rank）。）CUTLASS kernel 把同一条链编译成比较并分支。两者都在 prologue 中每线程执行一次，所以它们的代价属于同一量级：寥寥几条指令，每线程一次。第 12 章有该清单。

---

## 11.8 第 11 节：主机端启动镜像 `run()` (explicit.cu:1265-1329)

下面是这个非模板函数，分段引述；只省略了它的 `[TRACE]` 块（explicit.cu:1317-1324，在性能二进制文件中被编译掉）。

### 11.8.1 守卫 (explicit.cu:1265-1272)

```cpp
// explicit.cu:1265-1272
cutlass::Status run(ExplicitGemmParams const& params) {
  // E.5.1 guards: the kernel hard-codes the Section 0 configuration; refuse anything else loudly (device code has no checks)
  if (params.mode != kModeGemm) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.m != kFixedM || params.n != kFixedN || params.k != kFixedK || params.l != kFixedL) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.raster_order != kRasterOrderAlongN || params.swizzle_divisor != 0) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.tiles_m != kProblemTilesM || params.tiles_n != kProblemTilesN || params.tiles_l != kProblemTilesL) { return cutlass::Status::kErrorInvalidProblem; }
  if (params.alpha_ptr != nullptr || params.beta_ptr != nullptr) { return cutlass::Status::kErrorNotSupported; }   // no pointer path (D.6 item 13)
  if (params.beta != 0.0f) { return cutlass::Status::kErrorNotSupported; }   // the C path is omitted (Section 0; E.10 item 5); -0.0f passes like is_zero()
```

| 守卫 | 拒绝什么 | 为什么 kernel 无法处理它 | Status（由 `CUTLASS_CHECK` 打印的字符串，`examples/common/helper.h:43-52`；`cutlass.h:74-77`） |
|---|---|---|---|
| `mode != 0` | 除 `GemmUniversalMode::kGemm` 之外的任何值 | batched/array 模式会改变参数的语义 | `kErrorInvalidProblem`（"Error Invalid Problem"） |
| shape != (8192, 8192, 8192, 1) | 任何其他的 M、N、K、L | 网格被硬编码为 (64,64,1)；设备端从不读取 M、N、L；TMA 存储带有一个字面量 0 的 batch 坐标；不处理不完整的 tile | `kErrorInvalidProblem` |
| `raster_order != 1` 或 `swizzle_divisor != 0` | AlongM 光栅、任何 swizzle | 11.6.3 的 `tm, tn` 公式和 CLC 映射（第 7 章）是 AlongN、swizzle-0 的特化 | `kErrorInvalidProblem` |
| tiles != (32, 32, 1) | 不一致的 scheduler 状态 | 对仅主机端字段的一致性检查 | `kErrorInvalidProblem` |
| `alpha_ptr` 或 `beta_ptr` 非空 | 经由设备端指针传递的标量 | explicit kernel 没有用于它们的 `ld.global` 路径（笔记 D.6 第 13 项：在 baseline 中被编译出来但从未走过） | `kErrorNotSupported`（"Error Not Supported"） |
| `beta != 0.0f` | 一个真实的 C 项 | C 加载路径（`epi_load` 流水线、每个 tile 32 次 C TMA 加载、`load_order`）被省略；epilogue 用一个字面量为零的 C 片段计算 `fmaf(beta, 0.0f, t)` | `kErrorNotSupported` |

`alpha` 刻意**不**加守卫：任何值都以与 baseline 相同的方式处理（`alpha * acc`）。`-0.0f` 能通过 `beta` 测试，因为在 IEEE 比较下 `-0.0f != 0.0f` 为假，这与 baseline 的 `is_zero()` 测试一致（`is_C_load_needed = !beta.is_zero()`，笔记 B.2 步骤 11，第 8 节）。在 `beta = -0.0f` 下，两个 kernel 随后都用同一个运行时 `beta` 计算 `fmaf(-0.0f, 0.0f, t)`，所以 `D` 与 baseline 的保持逐位相同。唯一能观察到 `beta` 符号的输入是 `t = -0.0f`（`-0.0 + -0.0 = -0.0` 相对于 `+0.0 + -0.0 = +0.0`），这正是 C 项被保留为带运行时 `beta` 的 FFMA 而不是被丢弃的原因（第 10 章，笔记 D.7）。baseline 的 `run()` 完全没有守卫（它们位于 `can_implement` 中，harness 在 CUTLASS 一侧仍在 `70_blackwell_fp16_gemm.cu:407` 处调用它）；这六个就是 11.8.6 中有文档记载的偏离 (b)。

### 11.8.2 kernel 指针与步骤 1：动态 smem 的选择启用 (explicit.cu:1274-1279)

```cpp
// explicit.cu:1274-1279
  // E.5.2: the run() mirror (gemm_universal_adapter.h:375-575 -> cluster_launch.hpp:260-296)
  void const* kernel = reinterpret_cast<void const*>(&explicit_blackwell_fp16_gemm_kernel);
  cutlass::Status launch_result = cutlass::Status::kSuccess;

  // 1. MaxDynamicSharedMemorySize opt-in on the explicit kernel (initialize() does it for the CUTLASS kernel, :342-352); every call, not hoisted (E.5.4)
  launch_result = status_from_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes)));
```

- **为什么需要。** 默认情况下，一个 kernel 最多只能以 48 KB 的动态共享内存启动；"kernels relying on shared memory allocations over 48 KB per block are architecture-specific ... and require an explicit opt-in using `cudaFuncSetAttribute`"（CUDA Programming Guide，v13 版指南的第 3.2.6 节）。Runtime API 参考文档补充道，该值加上静态的 `sharedSizeBytes` 不能超过 `cudaDevAttrMaxSharedMemoryPerBlockOptin`：即 CC 10.0 的 227 KB = 232448 字节（11.2.1），由 util.hpp:46 处的 `static_assert` 检查（11.2.3）。没有这个调用，带 `dynamicSmemBytes = 230400` 的 `cudaLaunchKernelExC` 会被拒绝，什么都不会运行（确切的错误码未实测）。
- **谁为 CUTLASS kernel 做这件事。** `GemmUniversalAdapter::initialize()`（`gemm_universal_adapter.h:342-352`）：`if (smem_size >= (48 << 10)) cudaFuncSetAttribute(device_kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)`。explicit kernel 是另一个函数，需要它自己的选择启用。
- **为什么每次调用都做，而不是外提到一个 `static bool` 后面。** 笔记 E.5.4：baseline 在每次迭代中对它所启动的 kernel 做两次属性调用（`initialize()` 内的选择启用和 `run()` 内的 `NonPortableClusterSizeAllowed`）。在每次调用时都做选择启用，使 explicit 路径保持为每次迭代对所启动的 kernel 做两次属性调用，所以每次迭代的主机端序列**在种类上**保持相同（23 次非同步的主机端 API 调用对 22 次）。一次 `cudaFuncSetAttribute` 是一个只在主机端执行、代价为几微秒量级的调用，在迭代 1-9 中隐藏在前一个 ~0.889 ms 的 kernel 之后，只在迭代 0 中暴露出来（代价和隐藏均为推断，未实测；笔记 E.5.4）。把它外提不会节省任何可测量的东西，并且会让这一对比更难论证。

### 11.8.3 步骤 2-4：几何、`check_cluster_dims`、non-portable-cluster 属性 (explicit.cu:1281-1295)

```cpp
// explicit.cu:1281-1295
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
```

- **几何。** `grid = (64, 64, 1)`（util.hpp:32，11.6.3 的转置顺序）、`block = (256, 1, 1)`（`GemmKernel::get_block_shape()` = `dim3(MaxThreadsPerBlock, 1, 1)`，kernel `:399-401`）、`cluster = (2, 2, 1)`（util.hpp:31）。在 baseline 中 `grid` 在运行时来自 `get_grid_shape(params)`（`gemm_universal_adapter.h:381`）；这里它是一个常量，因为守卫已经把 shape 固定了（偏离 (b)）。
- **`check_cluster_dims`** 是 `cluster_launch.hpp:103-112` 的一份副本：cluster 大小 `<= MaxClusterSize = 32`（`:82`），且每个网格维度都能被 cluster 维度整除。对于这次运行，`4 <= 32`、`64 % 2 == 0`、`64 % 2 == 0`、`1 % 1 == 0`；util.hpp:42-43 也以 `static_assert` 断言了同样的条件，所以这里的运行时测试永远不可能失败。保留它是因为 `launch_with_fallback_cluster` 在每次启动时都执行它（`:274-278`），并在它失败时返回 `kInvalid`。
- **`NonPortableClusterSizeAllowed`。** `ClusterLauncher::init`（`:141-144`）在每次启动时都设置 `cudaFuncAttributeNonPortableClusterSizeAllowed = 1`（`:280`）。它只对大于可移植最大值 8 的 cluster 才是*必需的*（"the maximum portable cluster size supported is 8; however, NVIDIA Blackwell B200 GPU allows for a nonportable cluster size of 16 by opting in. Launching a kernel with a nonportable cluster size requires setting the `cudaFuncAttributeNonPortableClusterSizeAllowed` function attribute"，Blackwell Tuning Guide）。大小为 4 的 cluster 不需要它；保留它是为了与 baseline 的调用序列保持一致（笔记 E.5.2 第 4 项把 "functionally irrelevant" 这一判断标为推断）。

### 11.8.4 步骤 5-6：启动配置与 `cudaLaunchKernelExC` (explicit.cu:1297-1313)

```cpp
// explicit.cu:1297-1313
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
```

这就是针对这次构建和这次调用求值后的 `ClusterLauncher::make_cluster_launch_config`（`cluster_launch.hpp:150-213`）：

| `make_cluster_launch_config` | 此处的值 | 为什么 |
|---|---|---|
| `launch_attribute[0] = {cudaLaunchAttributeClusterDimension, {2,2,1}}`（`:166, :177`） | 被保留的那一个属性 | 没有回退 cluster（`have_fallback = false`，`:168`，因为来自 `gemm_universal_adapter.h:400` 的 `fallback_cluster_dims = {0,0,0}`） |
| `numAttrs` 从 3 开始（对于 nvcc >= 12.8 定义了 `CUDA_ENABLE_PREFERRED_CLUSTER`，`cluster_launch.hpp:55-57`；`:89`） | 3 -> 2：由于没有回退，`numAttrs--`（`:192-194`） | preferred-cluster 属性被丢弃 |
| `launch_attribute[numAttrs - 1] = PDL`（`:199-200`） | 写入了但不计入：`numAttrs = launch_with_pdl ? numAttrs : numAttrs - 1 = 1`（`:206`） | `launch_with_pdl = false`（harness 所调用的 `run()` 重载的默认实参，`gemm_universal_adapter.h:614`，在 `:615` 处转发给静态的 `run()`；harness 从不设置它）；所以 kernel 的 `griddepcontrol.wait` 立即返回，而 `launch_dependents` 没有依赖者（第 8 章和第 9 章；笔记 E.5.4 "absence of PDL"，推断） |
| `gridDim, blockDim, dynamicSmemBytes` (`:202-204`) | (64,64,1), (256,1,1), 230400 | |
| `stream`（`:205`） | `nullptr` = 传统默认流 | `gemm.run()` 以其默认的 `stream = nullptr` 被调用（`gemm_universal_adapter.h:612`） |

`cudaLaunchKernelExC(&config, func, args)`（CUDA Runtime API，"Execution Control"）带着 `config` 中的属性把 kernel 异步入队。`args` 是一个指向 kernel 各参数的指针数组，这里是一个指向 640 字节 `ExplicitGemmParams` 的指针，运行时把它复制到本次启动的参数缓冲区中（baseline 复制的是它 2048 字节的 `Params`，笔记 D.1 D2）。这次调用是 `cluster_launch.hpp:293-294`（没有 `CUTLASS_ENABLE_SYNCLOG` 时 `synclog_setup()` 是一个空操作，笔记 D.1 D14）。

**参数大小限制从何而来。** kernel 参数 "are passed to the device through constant memory and have been limited to 4,096 bytes"；"CUDA 12.1 increases this parameter limit from 4,096 bytes to 32,764 bytes on all device architectures including NVIDIA Volta and above"（NVIDIA 开发者博客，"CUDA 12.1 Supports Large Kernel Parameters"；需要 R530 或更新的驱动，而在 Volta 之前的架构上该限制仍为 4,096 字节）。这种常量内存放置正是 11.2.2 所展示的：该参数块落在常量 bank 0 的 `c[0x0][0x380]` 处，所以资源行读作 `CONSTANT[0]:1536 = 0x380 + 640`（笔记 G.3）。explicit.hpp:76 把经典的界限保留为一个守卫，`static_assert(sizeof(ExplicitGemmParams) <= 4096, "kernel parameter block limit");`。640 字节的参数块（以及 baseline 的 2048）远低于任一限制，所以该断言是在记录这一约束，而不是在防范一个真实存在的风险。

### 11.8.5 步骤 7：`cudaGetLastError` 与 Status 映射 (explicit.cu:1315-1329)

```cpp
// explicit.cu:1315-1329; the [TRACE] block at 1317-1324 is elided (compiled out in the performance binary)
  // 7. cudaGetLastError -> kSuccess / kErrorInternal (gemm_universal_adapter.h:564-574)
  cudaError_t const result = cudaGetLastError();
  ...   // [TRACE] branch elided (explicit.cu:1317-1324, compiled out in the performance binary)
  if (cudaSuccess == result && cutlass::Status::kSuccess == launch_result) {
    return cutlass::Status::kSuccess;
  }
  return cutlass::Status::kErrorInternal;
}
```

最后四行在逻辑上与 `GemmUniversalAdapter::run` `:564-574` 逐字一致。`cudaGetLastError()` "returns the last error that has been produced by any of the runtime calls in the same instance of the CUDA Runtime library in the host thread and resets it to `cudaSuccess`"（CUDA Runtime API，Error Handling）。这次重置清除了每线程的错误状态（CUDA Programming Guide 2.1.7.1，"Error State"）：一次启动拒绝在这里被清除，因此之后某个不相关的 `cudaGetLastError` 不会拾取到一个陈旧的错误（笔记 E.5.2 第 7 项："clears the per-thread error variable so that a stale non-sticky error is not picked up"）。一次设备端故障（非法地址）的行为则不同：它 "leaves the process in an inconsistent state and any further CUDA work will return the same error"（CUDA Runtime API，`cudaErrorIllegalAddress`），非正式地称为 "sticky" 错误。每一条失败路径（任何步骤返回的 `kInvalid`，或非成功的 `result`）都归并为 `kErrorInternal`，与适配器的做法完全一致。因此在 `[PROD]` 路径上，该函数以两种方式之一返回：`kSuccess`，此时 kernel **已入队，但未完成**；或者一个错误状态，通常此时什么也没有启动（如果 `cudaLaunchKernelExC` 成功，但随后 `cudaGetLastError` 报告了一个遗留在每线程错误状态中的错误，那么返回的是 `kErrorInternal` 而 kernel 已入队；CUDA Programming Guide 2.1.7.1-2.1.7.2）。

### 11.8.6 与 `GemmUniversalAdapter::run()` 和 `ClusterLauncher` 并排对照

| # | `explicit_gemm::run` (explicit.cu) | CUTLASS | 相同？ |
|---|---|---|---|
| 0 | 六个守卫 (1267-1272) | `run()` 中没有；`can_implement()` 在主机端 | **偏离 (b)**：有文档记载，只影响固定配置之外的参数 |
| 1 | `cudaFuncSetAttribute(kernel, MaxDynamicSharedMemorySize, 230400)` (1279) | 在 `initialize()` `:342-352` 中，作用于 `device_kernel<GemmKernel>` | **偏离 (a)**：同一属性，作用于实际被启动的 kernel；baseline 的调用仍然在那个从未被启动的 kernel 上运行 |
| 2 | `grid(64,64,1)`、`block(256,1,1)`、`cluster(2,2,1)` (1282-1284) | `block = get_block_shape()`、`grid = get_grid_shape(params)`（`:380-381`），`cluster` 来自 `ClusterShape`（`:395-397`） | 值相同；网格为硬编码（偏离 (b) 的一部分） |
| 3 | `check_cluster_dims` (1287-1290) | `cluster_launch.hpp:103-112`，经由 `:274-278` | 相同 |
| 4 | `cudaFuncSetAttribute(kernel, NonPortableClusterSizeAllowed, 1)` (1294) | `ClusterLauncher::init` `:141-144`，经由 `:280` | 相同 |
| 5 | 带有一个 `ClusterDimension` 属性的 `cudaLaunchConfig_t`，流为 `nullptr` (1299-1310) | `make_cluster_launch_config` `:151-213`（3 -> 2 -> 1 个属性） | 配置相同 |
| 6 | `cudaLaunchKernelExC(&cfg, kernel, {&params})` (1311-1312) | `:293-294` | API 相同；640 字节对 2048 字节的参数拷贝 |
| 7 | `cudaGetLastError` -> `kSuccess` / `kErrorInternal` (1316, 1325-1328) | `:564-574` | 相同 |

**为什么没有任何东西同步。** `cudaFuncSetAttribute` 和 `cudaGetLastError` 是主机端的；`cudaLaunchKernelExC` 入队并返回。`[PROD]` 路径上没有 `cudaDeviceSynchronize`、`cudaStreamSynchronize`、`cudaMemcpy`、`cudaFuncGetAttributes`、occupancy 查询或 `printf`（笔记 E.5.2 "Not in the function"）。

这一切的结果：explicit 路径的中位数 `Avg runtime` 为 0.889354 ms，而同一天的未改动二进制文件为 0.889917 ms，比值 0.99937（实测，笔记 G.2）；每次迭代 23 次主机端 API 调用，对比 22 次（从代码中数出，而非追踪得到：笔记 G.2 C4 和 E.5.4）。

---

## 11.9 作为时间线的框架，以及一份重写检查清单

```
 host (one call of explicit_gemm::run)           device (every CTA, 256 threads; t ->)
 ------------------------------------------      --------------------------------------------------------
 guards (6 checks)                               entry: smem_base = cvta(smem)            [all threads]
 FuncSetAttribute MaxDynSmem 230400              step 1  shfl -> warp_idx; elect -> lane_pred;
 grid/block/cluster constants                            %cluster_ctarank -> rank, is_cta0, is_leader, peer
 check_cluster_dims                              step 2  alpha, beta, k_tiles = (k+63)>>6
 FuncSetAttribute NonPortableCluster 1                   warp 1 lane: prefetch A, B; warp 3 lane: prefetch C, D
 cudaLaunchConfig_t {64x64x1, 256, 230400,       step 3  w0: 16 inits | fence | w4: 4 inits | fence |
   stream 0, [ClusterDimension 2x2x1]}                   w5: 8 inits | fence | w3: 4 inits | fence | w0 lane: 1 init
 cudaLaunchKernelExC  ---- enqueue ---->         step 4  barrier.cluster.arrive.relaxed.aligned
 cudaGetLastError -> Status                      step 5  x, y = %cluster_ctaid; tm, tn from blockIdx
 return (kernel not finished)                    step 6  barrier.cluster.wait.aligned   <- first point at which
                                                                                            remote smem may be touched
                                                 step 7  w2 -> producer | w1&rank0 -> scheduler | w0 -> mma |
                                                         w4-7 -> epilogue | w3, w1 of ranks 1-3 -> exit
```

如果你从一个空白文件出发重写这个框架，决策的顺序是：

1. **参数块。** 一个 POD 结构体，`CUtensorMap` 在前（对齐 64/128），标量在后；以 `const __grid_constant__` 传递它；只对这些 map 取地址。
2. **共享内存。** 一个 `extern __shared__ char smem[]`，没有静态 `__shared__`，每个对象都位于一个固定的 `OFF_*`；用 `static_assert` 断言各对齐（SW128 stage 为 1024，SW128_32B epilogue stage 为 512，CLC 响应为 16）以及总量；只计算一次 `smem_base`；绝不把它与通用指针混用。
3. **身份。** 通过 `shfl` 得到 warp 索引，一次可供一次性工作复用的选举，`%cluster_ctarank` 和 `%cluster_ctaid`；从 rank 推导出 leader/peer/cta0。
4. **预取**你将会用到的每一个描述符，由一个空闲的 warp 在会合之前完成。
5. **屏障。** 用每个 phase 的精确到达计数初始化你将会在其上执行 arrive 的每一个屏障（每个屏障一行理由），执行 fence，然后才进行 cluster 会合；让未使用的屏障保持未初始化，并把它们的计数排除在其他屏障之外（800，而不是 928）。
6. **会合。** `arrive.relaxed` -> 本地设置 -> `wait`；在该 wait 之前不做任何远程操作；通过对主机端所施加的任何光栅/转置求逆，从 `blockIdx` 推导出初始 tile。
7. 按 warp-uniform 的索引**分发**；让不参与的 warp 退出；确保没有任何 `bar.sync` 的参与者集合或 mbarrier 计数把它们包括在内。
8. **主机端。** 选择启用动态 smem，设置 cluster 属性，用 `cudaLaunchKernelExC` 在计时器所监视的那条流上启动，映射错误，绝不同步。

第 13 章把这份清单扩展为一个构建计划 (13.2)，并把它泛化到其他 GEMM 配置 (13.4)。

---

# 第 12 章 产物：ptxas 生成了什么，以及测得了什么

本章使用的证据标签：**[M]** 实测（笔记的 D 部分和 G 部分，或我本人对 `trace_out_v2/` 中已提交转储的解读），**[S]** 源自源码（explicit.cu、CUTLASS 头文件或行内引用的文档），**[H]** 我的分析或假设；**（NVIDIA 内部）** 标记未公开记载的 NVIDIA 工程信息。下面引用的每一行 SASS 都复制自 `trace_out_v2/sass_explicit.txt`，并去掉了末尾的 `/* 0x... */` 控制字注释。

---

## 12.1 三个产物，一条工具链

你在第 5-11 章中研读的 kernel 是一个 C++ 文件。在 B200 上运行的是一个 cubin：`nvcc` 的前端把该文件变成 PTX，`ptxas` 再把 PTX 变成 SASS。各 wrapper 的 asm 字符串（第 6 章；除 W1 和 W37 之外的全部，这两个是 `__shfl_sync` 和 `__syncwarp()` 内建函数）逐字进入 PTX。此后的一切，包括每条 `tcgen05.*`、`cp.async.bulk.tensor.*` 和 `mbarrier.*` 指令如何变成机器指令，都是 `ptxas` 的决定。本章讲的就是这些决定，以及最终的二进制文件在运行时做了什么。

```
 explicit.cu ──nvcc front end──> PTX (compute_100a, embedded in the fatbin)
                                   │  1552 instruction lines, 92 predicates, 1014 .b32 regs       [M]
                                   └──ptxas──> SASS (sm_100a cubin)
                                                 1640 instructions, REG 26, LOCAL 0, STACK 0     [M]
```

构建用的是笔记中固定的第 0 节命令。它的两个 `--generate-code` 子句，`arch=compute_100a,code=[sm_100a]` 和 `arch=compute_100a,code=[compute_100a]`，使 fatbin 同时携带 cubin 和 PTX（笔记 D.5）。本章读取的每一份转储都来自对保存下来的开关关闭的二进制文件 `trace_out_v2/70_blackwell_fp16_gemm.explicit` 运行 `cuobjdump` 的结果（笔记的 H.2.2 部分）。

### 12.1.1 PTX 入口头 [M]

```
.visible .entry _ZN13explicit_gemm35explicit_blackwell_fp16_gemm_kernelENS_18ExplicitGemmParamsE(
.param .align 128 .b8 ..._param_0[640]
.maxntid 256
.minnctapersm 1
.reg .pred %p<92>;
.reg .b16 %rs<7>;
.reg .b32 %r<1014>;
.reg .b64 %rd<29>;
```

每一行为什么是这个样子：

| PTX 行 | 来源 | 为什么 |
|---|---|---|
| `.param .align 128 .b8 [640]` | `const __grid_constant__ ExplicitGemmParams params` (explicit.cu:1096) | `sizeof(ExplicitGemmParams)` 为 640，因为在 CUDA 13.3 的头文件中 `alignof(CUtensorMap)` 为 128：四个 128 字节的 map (512) + 12 个 int/float (48) + 两个指针 (16) = 576，再填充到 128 的下一个倍数（explicit.hpp 字节映射；笔记 G.3）。 |
| `.maxntid 256` / `.minnctapersm 1` | `__launch_bounds__(256, 1)` (explicit.cu:1095) | 256 个线程 = 8 个 warp（第 11 章）。230400 B 的动态共享内存无论如何都迫使每个 SM 只能有 1 个 CTA。 |
| `%p<92> %rs<7> %r<1014> %rd<29>` | 分配前的虚拟寄存器 | CUTLASS kernel 的入口声明了 `%p<248> %rs<16> %r<1635> %rd<55>`（笔记 D.2）；explicit kernel 携带的活跃值更少（没有 C 路径，没有 scheduler 簿记）。七个 `.b16` 寄存器中有六个被使用。三个承载掩码操作数：`%rs3`（`mov.b16 %rs3, 15`，`kMainloopReleaseMask`，W30 的 `"h"` 操作数）、`%rs4`（`3 << 2y` 的 `cvt.u16.u32`，累加器 commit 掩码，W30）和 `%rs6`（`5 << x` 的 `cvt.u16.u32`，A 的 multicast 掩码，W21）。另外三个是 rank 簿记，前端把它们收窄到 16 位，并在角色测试中用 `setp.ne.b16` 重新测试：`%rs2 = cvt.u16.u32 %cluster_ctarank`、`%rs1 = %rs2 & 1`（`rank & 1`：在 peer 上为 1，在 pair leader 上为 0）和 `%rs5 = %rs1 & 255`（`trace_out_v2/ptx_explicit.txt` 第 34-35、406-407、419、1104、1112、1384-1387、1415、1430、1516 行）[M]。 |

设备端只执行两次参数读取：`ld.param.v2.b32 {%r5, %r6}, [..._param_0+528]`（alpha 和 beta）和 `ld.param.b32 %r36, [..._param_0+520]` (k) [M]。四个 tensor-map 地址用 `cvta.param` 形成，并传给 `prefetch.tensormap` 和各条 TMA 指令；`m`、`n`、`l` 从未被读取（笔记 G.3）。

在 SASS 中，参数块从常量 bank 偏移 `c[0x0][0x380]` 开始。所以 k 由 `LDCU UR14, c[0x0][0x588]` 读取 (0x380 + 520 = 0x588)，alpha/beta 由 `LDCU.64 UR26, c[0x0][0x590]` 读取 (0x380 + 528 = 0x590) [M]。`c[0x0][0x348]` 在四处被作为 64 位值读取：用于预取的、位于 0x00d0 的 `@!UP1 LDCU.64 UR4` 和位于 0x0130 的 `LDCU.64 UR6`，producer 中位于 0x0c10 的 `LDCU.64 UR22`，epilogue 中位于 0x1f50 的 `LDCU.64 UR30`。该值随后被加上偏移 `+0x0`、`+0x80`、`+0x100`、`+0x180`，恰好是 explicit.hpp 中四个 map 的字节偏移：`+0x100`/`+0x180` 送入 C 和 D 的 `UTMACCTL.PF`，`+0x80` 是 producer 中的 `tma_b`，`+0x180` 是 epilogue 中的 `tma_d` [M]。由于 PTX 用 `cvta.param` 形成这些操作数，我把这个槽位解读为参数块的通用地址，也就是笔记 D.2 所称的 "descriptor base" [H：常量 bank 的 layout 本身在我所用的任何资料中都没有文档记载]。

### 12.1.2 资源行 [M]

```
Function _ZN13explicit_gemm35explicit_blackwell_fp16_gemm_kernelENS_18ExplicitGemmParamsE:
  REG:26 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[2]:12 CONSTANT[0]:1536 TEXTURE:0 SURFACE:0 SAMPLER:0
```

（`trace_out_v2/resources.txt`；CUDA Binary Utilities 手册指出，`REG`、`TEXTURE`、`SURFACE`、`SAMPLER` 是计数，其他字段是字节数。）

| 字段 | 值 | 为什么 |
|---|---|---|
| `REG` | 26 | SASS 命名了 24 个不同的向量寄存器 `R0..R23`；epilogue 的活跃集决定了最大值（第 12.4.1 节中逐个寄存器给出）。ptxas 报告 26，比命名的数目多两个；CUTLASS kernel 命名了 `R0..R65`（66 个）并报告 68 [M]。这个 +2 的原因在我所用的资料中没有文档记载（未验证）。重新构建后，`nvcc -Xptxas -v`（或 `--resource-usage`）在编译时打印同样的数字，比 `cuobjdump --dump-resource-usage` 更快。 |
| `STACK` / `LOCAL` | 0 / 0 | 没有溢出（`LDL`/`STL` 计数为 0 [M]）。没有任何局部数组残留：W29 的 `mask[8]` 和 epilogue 的 `r[16]` 都被完全提升到寄存器中。 |
| `SHARED` | 1024 | **不是**这个 kernel 的静态 `__shared__`。它是动态缓冲区之下系统保留的那个千字节，对二进制文件中的每个 kernel 都会报告：`cudaFuncGetAttributes` 报告 `sharedSizeBytes 0`（`trace_out_v2/host.txt` 中的 `TRACE_HOST kernel_sharedSizeBytes 0`），而设备属性 `cudaDevAttrReservedSharedMemoryPerBlock` 报告同样的 1024（`TRACE_HOST device_reserved_smem_per_block 1024`）[M]（笔记 D.6 第 4 项，G.3）。这个千字节就是 `smem_base` 为 `0x400 + (rank << 24)` 而非 0 的原因（第 5 章）。 |
| `CONSTANT[0]` | 1536 | 0x380 + 640：kernel 的参数块位于常量 bank 0 中的 `c[0x0][0x380]`，所以该 bank 装有 896 字节的驱动/启动数据外加那 640 字节的块（explicit.hpp 头文件；笔记 G.3）。CUTLASS kernel 的是 2944 = 0x380 + 2048。 |
| `CONSTANT[2]` | 12 | ptxas 为角色分发构建的一张三项跳转表（第 12.2 节，“角色分发”一行）。CUTLASS kernel 则没有；它使用一条比较并分支的链 [M]。 |

### 12.1.3 三个解码前提：控制字、真值表立即数、字节选择器 [S，标注处为 H]

第 12.2 节的词典和第 12.5 节的清单用到三项任何 CUDA 课程都不教的 SASS 识读知识。在这里把它们读一遍。

**控制字与 scoreboard。** `cuobjdump` 把每条指令打印为一个 128 位编码，分在两条 `/* 0x... */` 注释中：第一条在指令所在行，第二条单独占下一行（`sass_explicit.txt` 中的 `/*5410*/ UTCHMMA.2CTA ... ; /* 0x00ff1c0c0a0075ea */` 后面跟着 `/* 0x0009e20008200024 */`）。该编码的一部分是由编译器填入的**控制信息**，而不是硬件的依赖逻辑：一个停顿计数（下一条指令发射之前要等待的周期数）、一个让出标志、一个写 scoreboard 索引和一个读 scoreboard 索引（这条指令在其结果写入时、或其操作数读取完毕时，设置六个依赖屏障中的哪一个），以及一个等待掩码（这条指令可以发射之前哪些屏障必须已清除）。

逆向工程文献记载了这一字段列表。对于 Maxwell，maxas wiki（"Control Codes"）陈述 "Maxwell has 6 barrier resources for this purpose"；写屏障服务于 "variable latency instructions"；读屏障防止后一条指令在操作数被读取之前 "overwrite the operand values"；等待掩码使一条指令 "wait on a previously set barrier of either type"；一个 64 位控制字覆盖三条指令。对于 Volta，Jia 等人 (2018) 表明控制字移入了每条 128 位指令之中。`sm_100a` 保留了这类方案，这一点由 NVIDIA 内部信息证实：producer 的 `&wr=x` 修饰读作 "this instruction produces/completes scoreboard token x"，consumer 的 `&req={x}` 读作 "the consumer requires token x to be released"，而确切的调度/控制码字段属于实现细节（`trace_out_v2/` 中的 `cuobjdump --dump-sass` 清单不打印这类修饰）。清单中 `DEPBAR.LE` 的 `SB0`/`SB1` 操作数（没有出现 `SB2..SB5` 操作数）以及笔记 D.2 对 baseline 控制字的解码（`SB1`、`SB4`）与之一致；上面的字段列表和六屏障计数，以及对本清单的字段级解码，对 `sm_100a` 而言仍属推断 [H]。

固定延迟的 ALU 指令只需要停顿计数。**可变延迟指令**（`LDTM`、`LDS`、`LDCU`、`S2UR`、`SYNCS.*`、`UTMA*` 以及 TC-pipe 操作）设置一个 scoreboard，而它们的 consumer 携带相匹配的等待位。这就是 `tcgen05.wait::ld` 的去向（第 12.2 节，`tcgen05.wait::ld` 一行）。你将遇到的唯一显式形式是 `DEPBAR.LE SBn, k`（CUDA Binary Utilities 手册的 Blackwell 指令表中的 "Dependency Barrier"）：等待直到 scoreboard bank n 至多还有 k 个未完成的已提交异步组；`SBn` 来自前面的 commit/flush 结果，而不是一个屏障 id（NVIDIA 内部）。该手册只给出了名字；`.LE` 以及与 `wait_group.read 1` 的 8 对 8 匹配（第 12.2 节）印证了这一解读。所以 `DEPBAR.LE SB0, 0x1` 就是 `cp.async.bulk.wait_group.read 1`（至多还有一个 bulk 存储组仍在读取共享内存），而 `DEPBAR.LE SB0, 0x0` 将是 `wait_group.read 0`（此处不存在）。alloc 序列内部位于 0x4b20/0x4b80 的两条 `DEPBAR.LE SB1, 0x36` 在 TMEM 分配器原子操作之前把 scoreboard 1 排空到 54：这是编译器在 `tcgen05.alloc` 的下沉内部插入的一次排序，保护同一个分配循环的两次尝试，一条紧接在第一条 `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN` (0x4b30) 之前，另一条在 `NANOSLEEP 0x64` 之后、重试之前 (0x4b90)；它们不是两次独立的分配，而每一条都确保在那次尝试之前，先前的异步/scoreboard 工作已充分退役（NVIDIA 内部；位置 [M]）。

**真值表立即数。** `LOP3.LUT d, a, b, c, LUT` 逐位计算以这个 8 位立即数为真值表的三输入布尔函数：`LUT` 的第 `(a << 2) | (b << 1) | c` 位就是输入为 `(a, b, c)` 时的结果。这是 PTX `lop3.b32 d, a, b, c, immLut` 的约定（`immLut = F(0xF0, 0xCC, 0xAA)`），所以 `0xc0` = `a & b`、`0x3c` = `a ^ b`、`0x33` = `~b`、`0xf6` = `a | (b ^ c)`、`0xf8` = `a | (b & c)`、`0xf4` = `a | (b & ~c)` [S]。

来自这个 kernel 的算例。`LOP3.LUT R9, R9, UR11, RZ, 0x3c, !PT` (0x52f0) 是 `cons_phase ^= flip`。`LOP3.LUT P2, RZ, R6, 0x1, RZ, 0xc0, !PT` (0x1ce0) 写出谓词 `P2 = (R6 & 1) != 0`，并把寄存器结果丢弃到 `RZ` 中。在 epilogue 入口 (0x1f30-0x2000) 处，`R20 = threadIdx.x << 2`（下面的掩码只保留第 2-6 位，即 `l << 2`）、`R0 = ((w << 11) | ((l & 7) << 2))` 和 `R21 = R20 & 0x60 = ((l >> 3) & 3) << 5`；于是四个存储基址为 `LOP3.LUT R23, R0, 0x60, R20, 0xf8` = `R0 | R21`、`LOP3.LUT R22, R0, 0x20, R21, 0xf6` = `R0 | (0x20 ^ R21)`、`LOP3.LUT R21, R0, 0x40, R21, 0xf6` = `R0 | (0x40 ^ R21)`、`LOP3.LUT R20, R0, 0x60, R20, 0xf4` = `R0 | (0x60 ^ R21)`。这就是把列类 `c = 0..3` 以 `Swizzle<2,5,2>` XOR 到 lane 项的第 5-6 位中（第 10 章），每个类别一条 `LOP3` [M 对应清单，S 对应解码]。

`PLOP3.LUT` 和 `UPLOP3.LUT` 把同样的思路用于谓词。当三个输入恒为真时索引为 7，所以 LUT `0x80`（第 7 位置位）写入 **true**，而 LUT `0x40`（第 6 位置位）写入 **false**。`PLOP3.LUT P2, PT, PT, PT, UP2, 0x80, 0x8` 把 uniform 谓词 `UP2` 复制到向量谓词 `P2` 中（输入 `11x` 选择 `0x80` 的第 7 位或第 6 位）。末尾的第二个立即数（与 `0x80` 搭配的 `0x8`，与 `0x40` 搭配的 `0x4`）是另一个我尚未解码的编码字段（未验证）。这就是第 12.5.1 节中的标注 `P0 = true`、`P2 = UP2`、`UP0 = false` / `UP0 = true` 的含义。

**字节选择器。** `PRMT d, a, sel, b`（以及 `UPRMT`）从这一对 `{b, a}` 的八个字节中组装出四个结果字节中的每一个："The bytes in the two source registers are numbered from 0 to 7: {b, a} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}" 以及 "the 3 lsbs of the selection value specify which of the 8 source bytes should be moved into the target position"（PTX ISA，`prmt`）。选择器的半字节 k（从最低有效位算起）指名结果字节 k 的来源。SASS 的操作数顺序是 `a, sel, b`，这是我对清单的解读 [H]，但只有在这一解读下，下面三种用法才说得通。`PRMT R4, RZ, 0x654, R4` = `{R4.b0, R4.b1, R4.b2, RZ.b0}`：把字节 3 换成 0 的 shared::cta 地址，即 `mapa(addr, 0)`。`UPRMT UR6, UR9, 0x654, UR8` = `UR8` 中的地址，其字节 3 = `UR9` = 目标 rank。`UPRMT UR34, URZ, 0x5410, UR25` = `{0, 0, UR25.b0, UR25.b1}`：16 位 multicast 掩码被移入 `UTMALDG` 掩码操作数的第 [16, 32) 位。

---

## 12.2 PTX 到 SASS 词典

该表涵盖了 kernel 发出的每一个指令族。“站点”是 explicit kernel 的 SASS 中的计数（我对 `trace_out_v2/sass_explicit.txt` 的重新计数，与笔记 G.3 一致）；CUTLASS kernel 的计数在有差别之处列于笔记 D.2。"Uniform" 指 uniform 数据通路指令（第 12.3 节）：`U*` 助记符、`UR`/`UP` 操作数、每个 warp 发射一次。

| PTX（wrapper，explicit.cu 行） | explicit kernel 中的 SASS（站点） | 下沉教给我们什么 |
|---|---|---|
| `tcgen05.mma.cta_group::2.kind::f16 [tmem], descA, descB, idesc, {8 masks}, p` (W29, :517) | `UTCHMMA.2CTA gdesc[UR10], gdesc[UR12], tmem[UR36], tmem[UR28], idesc[UR29], UP0` (4: 0x5410, 0x5440, 0x5460, 0x5470) | Uniform、无谓词：围绕它的 `elect.sync` 被折叠掉了。64 位描述符是 UR 寄存器对，其高字为 `UMOV URn, 0x40004040`（每 k-tile 8 条）。scale-c 谓词 `p` 在第一条 MMA 上成为 uniform 谓词 `UP0`，在其余三条上成为恒真的 `UPT`（k-block 0 之后 `scale_c = 1u` 对 kb 1..3 而言是编译期事实）。八个零掩码寄存器没有出现。`UR28` 由 `UMOV UR28, 0x0` (0x5390) 保持为 0，而 `tmem[UR28]`，即第二个 `tmem[]` 槽位，是稀疏元数据 TMEM 操作数 (E)，对稠密 MMA 为零，这里经由 `UR28 = 0` 实现；第一个槽位 `tmem[UR36]` 是累加器（NVIDIA 内部）；`tmem[URx]` 操作数是一个地址，并非本质上为零，而寄存器编号是分配器的产物（NVIDIA 内部）。该槽位在稠密形式中没有 PTX 对应物：`[sp-meta-tmem]` 仅是稀疏形式 `tcgen05.mma.sp` 和 `tcgen05.mma.ws.sp` 的操作数（PTX ISA 9.7.18.10.10.2 和 9.7.18.10.10.4，`[d-tmem], a-desc, b-desc, [sp-meta-tmem], idesc, ...`），而 9.7.18.10.10.1 的稠密 `tcgen05.mma` 语法是 `[d-tmem], a-desc, b-desc, idesc, ...`。 |
| `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [bar], mask` (W30, :532) | `UTCBAR.2CTA.MULTICAST [UR32], URZ, UR37`（2：0x5480，带 `UR37 = 0xf`；累加器 commit 带 `UR33 = 0x3 << 2y`，笔记 G.3） | Uniform；commit 操作是一条 TC-pipe 屏障指令，而不是内存操作。mainloop commit 是第四条 `UTCHMMA` 之后的那条指令。编译期掩码 0xF 在循环内成为 `UMOV UR37, 0xf`；运行时的 pair 掩码留在一个只计算一次的 UR 中。 |
| `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [smem+384], 512` (W26, :498) | `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN UP0, UR6, UR6`，其中 `UR6 = 0x10`（2 份：0x4b30 以及 `NANOSLEEP 0x64` 之后的重试 0x4b90），位于一个 69 条指令的序列 0x4a90-0x4ed0 之内 [M] | 一个分配单元 = 32 个 TMEM 列，所以 512 列 = `0x10` 个单元。ptxas 把这条“整个 warp、两个 CTA”的 PTX 操作当作一次握手来执行。**leader 的**被选中的 lane 执行该原子操作，存储结果（`STS [UR7+0x180], R6`，其中 `R6 = R2 << 5`，即单元 -> 列），用 `STAS [R4.64], R2`（st.async）把它转发给 peer，这会完成 peer 的保留共享内存中位于 `(rank << 24) + 0x48` 的一个屏障，并用 `ATOMS.OR/XOR` 维护位于 `+0x50/+0x54/+0x58` 的簿记字。peer 的被选中的 lane 等待（`SYNCS...TRYWAIT [UR4+0x8]`）、重新缩放并应答。`WARPSYNC.ALL` 运行两次；然后是 `BAR.ARV 0x6, 0xa0`。保留的千字节（`SHARED:1024`）就是这次握手所在之处。该序列与 CUTLASS kernel 的序列有相同的长度和骨架（笔记 G.3；第 9.4.1 节列出了八处助记符差异）。 |
| `tcgen05.dealloc.cta_group::2.sync.aligned.b32 T, 512` (W27, :504) | `UTCATOMSWS.AND URZ, UR4` (1: 0x5ae0) + 对保留的那些字的 `ATOMS.AND` 清零 + 两次 ptxas 插入的一致性检查，其失败路径 `CALL` 含有 `BPT.TRAP 0x1` 的桩 (0x6530, 0x6560) | 两个 CTA 都执行它。`UR4 = ~(0xffff << ((T & 0xffff) >> 5))` 清除该分配的 16 个单元位。PTX 中没有 `trap`；这两条 `BPT.TRAP` 是 ptxas 自己对保留的那些字的合理性检查 [M]。 |
| `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned` (W28, :512) | 没有 `UTC*` 助记符；`UVIRTCOUNT.DEALLOC.SMPOOL 0x80` (0x55f0)，紧接在 `PREEXIT` (0x5600) 之前；相邻的 `ELECT` (0x55d0) 和由 `ELECT` 守卫的 `STS.U8 [UR6+0x1c], R2` (0x5630) 不属于该下沉（NVIDIA 内部） | `UVIRTCOUNT` 就是 relinquish 的下沉（`UVIRTCOUNT 128`，即 `0x80`）；一个较旧的编译器映射曾把 relinquish 变成空操作，于是后继 CTA 一直串行化到 CTA 退出为止（NVIDIA 内部）。`ELECT` 为 mbarrier/TMA/commit 侧的工作选举出那一个线程，而 `STS.U8` 是周围一条普通的共享内存字节存储；relinquish 是 warp 集体的，并且位于被选中线程的区域之外，所以仅凭位置无法把二者中的任何一个归于它（NVIDIA 内部；笔记 D.2 的推断已撤回，而发出 `STS.U8` 的是什么尚未解决）。ptxas 把 `UVIRTCOUNT` 放在 `PREEXIT` **之前**发出，与 PTX 顺序相反（:950 处的 `griddep_launch_dependents()` 先于 :951 处的 `tmem_relinquish_2sm()`）。 |
| `tcgen05.ld.sync.aligned.32x32b.x16.b32 {16 regs}, [addr]` (W31, :543) | `LDTM.x16 R4, tmem[UR29+0x10*e]` (8: 0x2290, 0x2760, 0x2c40, 0x3120, 0x3600, 0x3ae0, 0x3fc0, 0x44d0) | TMEM 地址 `T + ((32w) << 16) + 128 st` 是 warp-uniform 的（`UR29`，由 `ULEA UR17, UR16, UR17, 0x15` = `w << 21` 加上基址构成）；subtile 项 `16e` 被折叠进立即数 `0x10*e`。每个 lane 十六个向量目的寄存器 `R4..R19`。 |
| `tcgen05.wait::ld.sync.aligned` (W32, :558) | **没有指令** | 控制字中的 scoreboard 跟踪 `LDTM` 的异步寄存器写入：一条 `LDTM` 之后的第一条 `FMUL R4` 等待该 `LDTM` 所设置的 scoreboard，下一条 `LDTM` 等待 `STS` 的读 scoreboard（WAR），而 subtile 7 在 0x44f0 处的 `SYNCS.ARRIVE` 等待第八条 `LDTM` (0x44d0)。不存在 wait 操作码这一点是实测的；scoreboard 的解码是推断的（笔记 D.2、G.3；第 12.1.3 节）。PTX 仍然需要恰好一个 `wait::ld`，置于 release 之前。ISA（9.7.18.6.4.5，"Register dependencies, same thread"）保证 "an intra-thread ordering through true register dependency will be respected regardless of the presence or absence of other forms of synchronization"，所以每个 subtile 的 `FMUL` 不需要任何 wait 就是合法的。但该依赖 "does not imply any other form of ordering"，并且 "to enforce such memory orderings and avoiding anti-dependency hazards around tcgen05.ld, tcgen05.wait::ld must be used"。释放累加器 stage 的 `mbarrier.arrive` (explicit.cu:1039) 不是一个寄存器 consumer，所以若没有 explicit.cu:1038 处的 `wait::ld`，该 release 可能先于第八次加载的完成。 |
| `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint` (W21, :450) | `UTMALDG.3D.MULTICAST.2CTA [UR16], [UR22], UR34, desc[UR32]`（2：prologue 循环中的 0x0fc0，主循环中的 0x12a0） | Uniform、无谓词（选举被折叠）。`UR34` 是由 `UPRMT UR34, URZ, 0x5410, UR25` 插入到第 [16,32) 位的 multicast 掩码（`UR25 = 0x5 << x`）。`desc[UR32]` = `{UR32, UR33} = {0x0, 0x10000000}` = 缓存提示 `0x1000000000000000`（`kCacheHintEvictNormal`）。`[UR22]` 是 64 位的 tensormap 地址。`[UR16]` 指名目的地 `base + 0x8400 + (s << 14) + 8192 y`。mbarrier 与三个坐标放在连续的 uniform 寄存器 `UR17`（带掩码的 `full[s]`）、`UR18`（`64 q`）、`UR19`（`128 tm + 64 y`）、`UR20`（z = 0）中，循环恰在该指令之前填充它们 [H，源自配套的 `ULEA`/`USHF`/`UMOV`]。 |
| `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` (W22, :463) | `UTMALDG.3D.2CTA [UR8], [UR30], desc[UR32]` (2: 0x1000, 0x12e0) | 同一形式，但没有掩码操作数；`UR8 = base + 0x28400 + (s << 13)`，`UR9..UR12` = `{UR17, UR18, b_col, UR20}`，由 `UMOV UR9, UR17; UMOV UR10, UR18; UMOV UR12, UR20` 复制而来 [复制为 M，块的解读为 H]。两个 producer 循环都未展开，各有一份 A 拷贝和一份 B 拷贝。 |
| `cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [tmap, {c0,c1,c2}], [smem]` (W23, :476) | `UTMASTG.3D [UR20], [UR36]`（32：每个 subtile 4 条，例如 0x26b0、0x2700、0x2710、0x2720） | 每条 PTX 存储对应一条 uniform 指令，无论 lane 数多少都每个 warp 发射一次（K4b 实测每次发射都有 32 个活跃 lane，笔记 D.1 D8）。`[UR36]` 是 D 的 tensormap（`{UR30, UR31} + 0x180`，其中 `UR30:UR31` = `LDCU.64 c[0x0][0x348]`，第 12.1.1 节）。`[UR20]` 指名源 `UR7 + 0x200 + 0x800 i`，随后的寄存器 `UR21..UR23` 保存 `{128 tm + 32 i, 128 tn + 16 e, 0}`（`UIADD3 UR13, UR21, 0x40` 构造 box 2 的 c0 = box 0 的 + 64）[H，源自寄存器填充代码；完整的四个 box 的构成见第 12.5.4 节]。 |
| `cp.async.bulk.commit_group` (W24, :487) | `UTMACMDFLUSH` (8) | 每个 subtile 一条，紧接在第四次存储之后。 |
| `cp.async.bulk.wait_group.read 1` (W25, :489) | `DEPBAR.LE SB0, 0x1` (8: 0x2740 ... 0x49b0) | Bulk-async 存储组在 scoreboard bank `SB0` 上被跟踪（第 12.1.3 节；`DEPBAR.LE` 基于计数的等待语义是 NVIDIA 内部信息）："at most one group still reading smem" 是一条依赖屏障指令，而不是自旋循环。位于 0x4b20/0x4b80 的两条 `DEPBAR.LE SB1, 0x36` 属于 alloc 序列，而不属于 TMA。`wait_group.read 0` 不存在（beta = 0 时没有 `store_tail`；第 10 章）。 |
| `prefetch.tensormap [generic addr]` (W20, :440) | `UTMACCTL.PF [UR4]`（4：A、B 的 0x00f0、0x0110 由 warp 1 发出；C、D 的 0x0180、0x0190 由 warp 3 发出） | 描述符留在参数空间中；该预取是一条 TMA 控制指令，它预热的是 tensor-map 描述符缓存，而不是 L2 数据缓存（NVIDIA 内部）。`lane_pred` 守卫被保留为一个 uniform 谓词（通过 `VOTEU.ALL UP1, P1`，在第一对上为 `@!UP1`）。 |
| `mbarrier.init.shared::cta.b64 [addr], count` (W5, :286) | `SYNCS.EXCH.64 URZ, [UR7+off], URn` (33: 0x02a0 ...) | 屏障初始化是对一个常量字的 64 位原子交换：读旧值、写新值、返回旧值；`URZ` 目的操作数丢弃返回的旧字（NVIDIA 内部，第 12.7.4 节）。该字是 `{(0x100000 - count) << 1, (0x100000 - count) << 11}`，由 `UIADD3 UR8, -UR8, 0x100000; USHF.L.U32 UR9, UR8, 0xb; USHF.L.U32 UR8, UR8, 0x1` 构造 [M]；字段含义（待到达计数与期望计数以 2^20 - count 的形式存储）是推断的。这 33 个偏移恰好就是笔记第 6.2 节的映射，计数恰好就是笔记第 7.2 节的表格（第 4 章）。按初始化顺序：`[UR7]`、`+0x40`、`+0x8`、`+0x48`、... `+0x38`、`+0x78`（`full[s]`/`empty[s]` 交错，16 个）、`+0xd0`、`+0xe0`、`+0xd8`、`+0xe8`（CLC full/empty，4 个）、`+0xf0`、`+0x110`、... `+0x108`、`+0x128`（累加器 full/empty，8 个）、`+0x130`、`+0x140`、`+0x138`、`+0x148`（throttle，4 个）、`+0x150`（`tmem_dealloc`）[M]。未被触及的范围 `0x80-0xcf` 是 CUTLASS layout 存放其 `epi_load` 和 `load_order` 屏障的地方：这是 33 对 43 初始化差异的可见证明（第 12.8.2 节）。被选中 lane 的守卫被折叠掉了；33 次初始化中有 32 次是位于 warp 索引分支之后的无谓词 uniform 指令；只有 `tmem_dealloc` 的初始化——它在 explicit.cu:1165 处由 kernel 入口的 `lane_pred`（= :1103 处的 `elect_one_sync()`）守卫，而不是由一次新的 `elect_one_sync()` 守卫——保留了一个 uniform 谓词（0x06e0 处的 `VOTEU.ALL UP0, P2`，0x0790 处的 `@!UP0 SYNCS.EXCH.64`）。 |
| `fence.mbarrier_init.release.cluster` (W14, :408) | **没有自己的指令**；每个初始化组之前都有一条裸的 `FENCE.VIEW.ASYNC.S`（第一条 `EXCH` 之前的 0x0290） | 归因系推断（笔记 D.2）。在 PTX 中保留它：它就是让这些初始化在 `barrier.cluster.arrive` 之前对 cluster 可见的那个 release。 |
| `mbarrier.try_wait.parity.shared::cta.b64 P1, [addr], parity, 0x989680`，位于一个 `bra` 循环中 (W6, :297) | `SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4], R2`，parity 位于 `R2` 的第 31 位（`SHF.L.U32 R2, R9, 0x1f`），重试循环为非内联：`TRYWAIT; @!P NANOSLEEP.SYNCS 0x989680; @!P SYNCS.PHASECHK.TRANS64; @!P BRA`（65 条 `TRYWAIT`；30 个重试循环，每个各一条 `NANOSLEEP.SYNCS`：26 个非内联，位于 0x5be0-0x64f0，在 warp 0 位于 0x5bc0 的 `EXIT` 之后，另有 4 个内联于角色尾段（producer 的 0x1960、scheduler 的 0x1ea0、warp 0 的 0x5860 和 0x58a0）；31 条 `NANOSLEEP` = 那 30 条 + alloc 重试位于 0x4b60 的 `NANOSLEEP 0x64`） | ptxas 把每个阻塞等待拆成一次内联的首次尝试和一个非内联的重试循环，所以当 peek 成功时各热循环保持为直线代码。`0x989680` = 10,000,000 ns = 10 ms 是 `timeHint` 操作数（含义与 ISA 措辞见第 12.2.2 节）。 |
| `mbarrier.try_wait.parity` peek (W7, :314) | 同一条 `TRYWAIT`，其结果谓词馈入一条跳过阻塞块的 `@P BRA` | 第 4 章“先 peek 再 wait”的惯用写法给代码增加了一条 `TRYWAIT` 和一条 `BRA`；在快速路径上它执行两条指令，正如一次阻塞等待的 `TRYWAIT; @!P BRA` 所执行的那样（`0x2180`、`0x2260-0x2280`）。 |
| `mbarrier.test_wait.parity` (W8, :329) | 不带 `.TRYWAIT` 的 `SYNCS.PHASECHK.TRANS64 P0, [UR8+0xe0], R0`（2：0x1d70，以及地址在 `[R3+UR7+0xe0]` 中的 0x1e50） | 非阻塞的 PTX test 下沉为裸形式，无谓词且没有重试循环；同一助记符也是 W6 重试循环中带 `@!P` 谓词的再检查（第 12.2.2 节）。两种形式并不等价：`.TRYWAIT` 是最初的非阻塞探针，裸形式是重试路径上的检查（NVIDIA 内部）。只有 CLC `producer_tail` 使用 `test_wait`（第 7 章）。 |
| `mapa.shared::cluster.u32 r, addr, cta` (W10/W13, :357, :393) | **没有指令**：`PRMT R4, RZ, 0x654, R4` / `UPRMT UR6, UR9, 0x654, UR8`（11 处站点） | 一次字节置换，把地址的字节 3 替换为 rank（第 12.1.3 节）。这是编译器侧的证明，表明 cluster 范围的共享地址是 `(rank << 24) + shared::cta address`，这也是让 `& 0xFEFFFFFF` 这一 peer 技巧合法的原因（第 5 章）。以 `RZ` 作为 rank 操作数 = “映射到 rank 0”（CLC 的各次释放）。 |
| `mbarrier.arrive.shared::cluster.b64 _, [remote]` (W10) | `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R4+URZ], RZ`（9 次远程 arrive，笔记 G.3） | `.RED` = 归约式的 arrive，远程（peer CTA）屏障目标必须使用它，在那里普通的 `SYNCS.ARRIVE` 是非法的，而它在本地也有效；`A1T0` = 到达 +1，事务增量 0（NVIDIA 内部；CUDA Binary Utilities 表只记录了操作码，`SYNCS` = "Sync Unit"，而没有记录修饰符）。 |
| 经由带掩码地址的 pair-leader arrive (W11, :370) | `ULOP3.LUT UR24, UR24, 0xfefffff8, URZ, 0xc0, !UPT`，然后是 `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [UR24], RZ` (0x44c0, 0x44f0) | kernel 中唯一的 `0xfefffff8`：nvcc 把 8 字节屏障地址的三个对齐位折叠进了 `0xFEFFFFFF`（笔记 D.6 第 2 项）。在 producer 中，该掩码以 `ULOP3.LUT UR35, UR7, 0xfefffff0, ...` (0x0c00) 的形式被折叠到外提的基址上，这是等价的，因为基址是 1 KiB 对齐的且 `8 s <= 56`（笔记 G.3）。两个 CTA 的全部 128 个 epilogue 线程都执行该 arrive。操作数是 uniform 的（`UR24`），但 arrive 是每线程的，因此 leader 的 `acc_empty[st]` 上的计数为 256（第 4 章）。 |
| `mbarrier.arrive.shared::cta.b64` 本地 (W9, :346) | `SYNCS.ARRIVE.TRANS64.A1T0`（1：throttle `full`，rank 0） | 本地 arrive，没有 `.RED`。`.RED` arrive 在本地屏障上也有效；只有远程目标才要求它（NVIDIA 内部）。 |
| `mbarrier.arrive.expect_tx.shared::cta.b64 _, [full[s]], 49152` (W12, :382) | `@!P0 MOV R5, 0xc000; ... @!P0 SYNCS.ARRIVE.TRANS64 RZ, [UR10], R5`（2 份：prologue 循环中的 0x0e40/0x0ee0，主循环中的 0x1110/0x11b0） | 事务计数是一个向量寄存器（`R5 = 0xc000 = 49152`），因为 PTX 通过 `mov.b32` 送入它；每 lane 的谓词（pair leader 的 leader lane）以 `@!P0` 的形式幸存下来。地址是**未加掩码的**本地 `full[s]`。 |
| `@p mapa; @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote], 16` (W13, :393) | `@!P0 MOV R5, 0x10; ... @!P0 SYNCS.ARRIVE.TRANS64.RED RZ, [R11+URZ], R5` (0x1ad0, 0x1b70)，其中 `R11 = PRMT(lane)` | lane 0..3 用 16 字节装填 CTA 0..3 的 `clc_full[idx]`；这是一个每 lane 的远程地址，因此需要一个向量寄存器和一个 `.RED` arrive，远程目标要求如此（NVIDIA 内部）。 |
| `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [resp], [bar]` (W33, :568) | `UGETNEXTWORKID.BROADCAST [UR8], [UR9]` (1: 0x1bc0; `UR8 = base + 0x160 + 16 idx`, `UR9 = base + 0xd0 + 8 idx`) | Uniform、无谓词（选举被折叠）。助记符说明了 CLC 是什么："get the next work id and broadcast it to the cluster"。没有任何 SASS 助记符包含 "CLC"（一个 grep 陷阱，笔记 G.6 第 2 项）。两条 ISA 规则，kernel 的结构正是为了遵守它们而存在：在使用 `.multicast::cluster::all` 时，"the behavior of instruction ... is undefined if any CTA in the cluster is exited"，以及 "if the executing CTA has already observed the completion of a clusterlaunchcontrol.try_cancel instruction as failed, then the behavior of issuing a subsequent clusterlaunchcontrol.try_cancel instruction is undefined"（PTX ISA，`clusterlaunchcontrol.try_cancel`）。它们是第 4 章的退出尾段以及第 7 章的 "no query after a not-cancelled response" 的又一个理由。 |
| `ld.shared.b128; query_cancel.is_canceled.pred; @p1 get_first_ctaid.v4.b32` (W34, :579) | `LDS.128 R4, [R4+0x160]` (4) + 位域提取（`LOP3.LUT P2, RZ, R6, 0x1` = 字 2 的第 0 位是有效位；`x0 = word0`、`y0 = word1 & 0xffff`、`z0 = word1 >> 16` 见于 CUTLASS kernel 的 SASS，笔记 D.2；explicit kernel 的 `z0` 在 trace 构建之外是死的，所以其 SASS 只提取 `x0` 和 `y0`，笔记 G.3） | 响应 layout 是从提取代码推断出来的（笔记 D.2）。在 scheduler 与 MMA warp 中只有有效位幸存下来（tile 坐标在那里是死的；第 6.8.3 节）。 |
| `fence.proxy.async.shared::cta` (W15, :418) | `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`（12 对：4 次 CLC 解码 + 8 个 epilogue subtile） | 一个从通用 proxy 到异步 proxy 的 fence 是一个 CTA 作用域的内存屏障外加一个异步视图 fence。 |
| `barrier.cluster.arrive.relaxed.aligned` / `barrier.cluster.wait.aligned` (W16/W17, :428-429) | `UCGABAR_ARV` (0x07c0) / `UCGABAR_WAIT; CCTL.IVALL` (0x0a20, 0x0a30)，两者都位于一个由 ptxas 插入的 `c[0x0][0x36c] == 1` 测试之后，带有 `NOP` / `BAR.SYNC.DEFER_BLOCKING 0x0` 回退 (0x07e0 / 0x0a50) | "CGA" = cooperative group array = cluster。该 wait 之后跟着 `CCTL.IVALL`（`CCTL` = CUDA Binary Utilities 操作码表中的 "Cache Control"；`.IVALL` = 由下沉发出的一个全部失效的缓存控制操作，其缓存域并不是一个稳定的 SASS 契约（NVIDIA 内部），第 11.6.1 节）。这个守卫（“以 cluster 方式启动？”）不在 PTX 中；对该常量的解读是推断的（笔记 D.2）。 |
| `bar.sync 1, 128` / `bar.sync 6, 160` / `bar.arrive 6, 160` (W18/W19, :432, :435) | `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` (16) / `0x6, 0xa0` (1: 0x1f10) / `BAR.ARV 0x6, 0xa0` (1: 0x4ed0) | 命名屏障把它们的 id 和线程计数保留为立即数（PTX 通过 `mov.b32` 传入它们，ptxas 对它们做了常量折叠）。 |
| `bar.warp.sync -1` (`__syncwarp()`, W37) | `WARPSYNC.ALL`（2：0x4d40、0x4eb0，两者都在 alloc 序列内） | Warp 0 在 `tcgen05.alloc` 之后的 `__syncwarp()` (:886) 位于 0x4eb0（归因是推断的，笔记 D.6 第 15 项）。Warp 2 的每个 tile 的 `__syncwarp()` (:792) **没有**产生指令（笔记 D.6 第 15 项）；自然的解读是 ptxas 证明了该 warp 在那一点上是收敛的 [H]。 |
| `elect.sync` (W2, :264) | `ELECT P0, URZ, PT`（14 个 PTX 站点中的 4 个：0x0020 prologue，0x4a90 和 0x4d60 alloc/握手，0x55d0 靠近退出序列，来源未查明；不是 relinquish，后者是 warp 集体的（NVIDIA 内部）） | **producer、MMA 或 scheduler 循环内一个都没有**（第 12.3 节）。 |
| `griddepcontrol.wait` / `griddepcontrol.launch_dependents` (W35/W36, :599-600) | `ACQBULK`（2：0x0bb0 在 warp 2，0x19f0 在 rank-0 的 warp 1）/ `PREEXIT`（1：0x5600，warp 0） | 程序化依赖启动："acquire bulk" 等待前一个网格的内存变得可见；"pre-exit" 发出依赖者可以启动的信号。 |
| `mov.u32 %cluster_ctarank` (W3, :281) | `S2UR UR15, SR_CgaCtaId`（4 条 `S2UR` 中的 2 条） | rank 是一个被读入 UR 两次的特殊寄存器（0x0030 处读入 `UR15`，MMA warp 稍后在 0x4ff0 处将其复用为 `T`；0x55c0 处读入 `UR6`，用于 warp 0 的退出尾段；笔记 G.3）；CUTLASS kernel 重新读取它 16 次（笔记 D.2）。 |
| `mov.u32 %cluster_ctaid.x/.y` (W4, :282-283) | 用 `I2FP`、`FMUL`、`F2I` 从 `SR_CTAID.X/Y` 和一张常量 bank 表计算得出（0x0920/0x0960 处的两条非 epilogue `FMUL`） | 该特殊寄存器没有被直接读取；对该序列的解读是推断的（笔记 D.2）。这与协议无关，但它正是 `FMUL` 计数为 130 而不是 128 的原因。 |
| `shfl.sync.idx.b32` (W1, :261) | `SHFL.IDX PT, R0, R0, RZ, 0x1f` (1: 0x0060) | warp 索引广播；CUTLASS kernel 有 7 条（笔记 D.2）。 |
| `__cvta_generic_to_shared(smem)` (:258) | `S2UR UR15, SR_CgaCtaId` + `0x400 + (rank << 24)` 送入 `UR7`；每个屏障地址都是 `ULEA URn, URidx, UR7, 0x3` | `smem_base` 是一个形状在编译期确定的常量：ptxas 自己把基址构造为 `0x400 + (rank << 24)`（笔记 D.1 D5），而 `UR7` 在整个 kernel 中都是基址寄存器（`[UR7+0x180]` 是 `tmem_base_ptr`，`UR4 = ULEA(s, UR7, 3)` 时的 `[UR4+0x40]` 是 `empty[s]`）[M]。 |
| `mul.f32` 然后 `fma.rn.f32` (:1045-1046) | `FMUL Rk, Rk, UR26` 然后 `FFMA Rk, RZ, UR27, Rk`（128 + 128；`FADD` 0） | alpha 在 `UR26` 中，beta 在 `UR27` 中，由 `LDCU.64 UR26, c[0x0][0x590]` (0x1f80) 加载一次；字面量 C = 0 是 `RZ`。`fma.rn.f32` 已经作为一个显式的 `fmaf` 调用 (:1046) 存在于 PTX 中，与 `-fmad` 无关（baseline 的那个是 `-fmad=true` 下的前端收缩，第 10 章）；ptxas 没有改变算术运算。与 CUTLASS kernel 相同的 IEEE 运算和操作数值（`FMUL Rk, R2, Rk; FFMA Rk, R(17+k), UR21, Rk`，带一个零填充的 C 片段），因此 D 逐位相同（第 10 章，笔记 G.3）。 |
| `st.shared.b32 [smem + off], v` (:1047) | `STS [Rc+UR7+imm], Rk`，其中 `imm = 0x200 + 0x80 (j & 3) + 0x200 (j >> 2) + 0x2000 (e & 3)`，且对于 `j & 3 = 0..3`，`Rc` 取自 `{R23, R22, R21, R20}`（128 + 3 条簿记） | lane 相关的 `Swizzle<2,5,2>` 项存放在四个只计算一次的基址寄存器中；列类和 subtile stage 是立即数。评审对照 CUTLASS kernel 的地址重新计算了全部 128 x 8 x 16 个地址：0 处不匹配（笔记 G.3）。 |
| `ld.shared.b32 tmem_base_ptr` | `LDS R2, [UR7+0x180]`（0x2020 epilogue；以 `LDS R0, [UR7+0x180]` 的形式出现在 alloc 握手内的 0x4df0 处，以及用于 MMA warp 的 `T` 的 0x4f00 处） | `T` 在每个角色中都从共享内存读取一次，从不假定。 |
| 角色分发 (:1191-1200) | `UISETP.LT.U32 UP0, UR20, 0x2; USEL UR4, UR20, 0x2, UP0; USHF.L.U32 UR4, UR4, 0x2; LDCU UR10, c[0x2][UR4]; BRXU UR10 -0xaf0` (0x0a60-0x0ae0)，然后 `UISETP.NE.U32 UP0, UR20, 0x2; BRA.U UP0, 0x1ee0` | ptxas 把 `if/else if` 链变成了一个由 `min(warp_idx, 2)` 索引的 3 项跳转表（`CONSTANT[2]:12`），外加对 warp 2 与 warp >= 3 的第二次测试。它每线程执行一次。`BRXU` = 基于一个 uniform 寄存器的间接分支。 |

有四行值得用长形式来展开。

### 12.2.1 `UTCHMMA` 及其操作数 [M]

```
        /*5410*/                   UTCHMMA.2CTA gdesc[UR10], gdesc[UR12], tmem[UR36], tmem[UR28], idesc[UR29], UP0 ;
        /*5440*/                   UTCHMMA.2CTA gdesc[UR16], gdesc[UR18], tmem[UR36], tmem[UR28], idesc[UR29], UPT ;
        /*5460*/                   UTCHMMA.2CTA gdesc[UR20], gdesc[UR22], tmem[UR36], tmem[UR28], idesc[UR29], UPT ;
        /*5470*/                   UTCHMMA.2CTA gdesc[UR24], gdesc[UR26], tmem[UR36], tmem[UR28], idesc[UR29], UPT ;
        /*5480*/                   UTCBAR.2CTA.MULTICAST [UR32], URZ, UR37 ;
```

请与该 wrapper 的 PTX 源码对照（W29，explicit.cu:517-529，引于第 6.7.2 节）。

- `gdesc[UR10]` 是位于 `{UR10, UR11}` 中的 64 位 A 描述符：`UR10 = ULEA UR8, UR30, 0xa`（= `desc_lo_a + 1024 s`，即以 16 字节为单位的、explicit.cu:928 的 `+16384 B per stage`），`UR11 = 0x40004040`（`kSmemDescHi`）。后三个 k-block 使用 `UR16/18/20/22/24/26 = UR10/12 + 0x2/0x4/0x6`（`+ 2 kb`，explicit.cu:933-934），并带有它们自己的 `0x40004040` 高字 `UR17/19/21/23/25/27`。这就是为什么每个 k-tile 会出现八条 `UMOV URn, 0x40004040`。
- `tmem[UR36]` = `tmem_c = T + 128 st` (explicit.cu:919)，每个 tile 计算一次：即累加器，PTX 中的 `[d-tmem]`。第二个槽位 `tmem[UR28]` 保持为 0，是稀疏元数据 TMEM 操作数，对稠密 MMA 为零（NVIDIA 内部）；稠密的 PTX 形式没有对应它的操作数（第 12.2 节）。
- `idesc[UR29]` = `0x10200010`（`kInstrDesc`），每个 k-tile 由 `UMOV UR29, 0x10200010` 重新物化。
- `UP0` / `UPT`：即该 wrapper 的 `setp.ne.b32 p, scaleC, 0`。在循环中，`scale_c` 只对一个 tile 的第一个 k-tile 的 `kb = 0` 为 0（explicit.cu:917、:940），所以 ptxas 保留了一个 uniform 谓词 `UP0`。它由位于 0x51c0 的 `UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x40, 0x4` 在 k-tile 循环之前每个 tile 清零（置为假）一次。它由位于 0x54b0 的 `UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x80, 0x8` 在第一个 k-tile 的四条 MMA 发射之后置为真，因此后面每个 k-tile 的第一条 `UTCHMMA` 都以 `scale_c = 1` 运行（LUT 解码见第 12.1.3 节；入口序列与循环清单见第 12.5.1 节）。另外三条 MMA 被硬连线到 `UPT`。CUTLASS kernel 在 `UP2` 上于 0x3860 / 0x3a90 处显示出相同的两站点模式（笔记 D.2）。
- 该调用周围的 `if (elect_one_sync())` (explicit.cu:932) 不见了。整个 warp 都到达这条指令；硬件执行它一次（第 12.3 节）。

### 12.2.2 屏障等待，内联与非内联 [M]

该 wrapper（W6，explicit.cu:297-311，引于第 6.3.3 节）是一个 `bra` 循环。

ptxas 把第一次尝试内联发出，并把循环移出热路径。在 MMA 循环中：

```
        /*51f0*/               @P0 BRA 0x5240 ;                                   <- peek succeeded: skip the wait
        /*5200*/                   ULEA UR4, UR8, UR7, 0x3 ;                       <- full[s] = smem_base + 8 s
        /*5210*/                   SHF.L.U32 R2, R9, 0x1f, RZ ;                    <- parity -> bit 31
        /*5220*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4], R2 ;
        /*5230*/              @!P0 BRA 0x6350 ;                                    <- not yet: go to the out-of-line retry loop
```

以及重试循环的形状（这一份副本是第 3 条经验的四个内联循环之一，是 producer 尾段的最后一次等待，以 `EXIT` 结尾）：

```
        /*1950*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R3+URZ], R0 ;
        /*1960*/              @!P0 NANOSLEEP.SYNCS 0x989680 ;
        /*1970*/              @!P0 SYNCS.PHASECHK.TRANS64 P0, [R3+URZ], R0 ;
        /*1980*/               @P0 EXIT ;
        /*1990*/                   BRA 0x1940 ;
```

三条经验。

1. parity 操作数是一个 32 位寄存器，其第 31 位是 parity。所以 C++ 中的 `phase ^= 1` 花费一条 `LOP3 ... 0x3c`（XOR）外加那条 `SHF`。
2. `NANOSLEEP.SYNCS 0x989680` 携带着 10 ms 的 `timeHint`。ISA 说被挂起的线程 "resumes execution when the specified phase completes OR before the phase completes following a system-dependent time limit"，并说该操作数 "specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit"（PTX ISA，`mbarrier.test_wait`/`mbarrier.try_wait`）。它是对一次挂起的时间限制的提示，而不是一个有保证的上界。在 SASS 序列中，`NANOSLEEP` 是未命中时的退避，其后裸的 `PHASECHK.TRANS64` 是跳回之前重试路径上的阻塞检查（NVIDIA 内部）；ISA 所说的在 phase 完成时唤醒是通过一个唤醒触发器实现的：一次失败的 `PHASECHK.TRANS64.TRYWAIT` 装填它，而屏障的 phase 完成会唤醒正在睡眠的 warp（唤醒条件是一次 phase 翻转，或一次锁断言），所以 `NANOSLEEP` 退避不是普通的周期性轮询（NVIDIA 内部）。
3. 在第 12.2 节 W6 行所计数的 30 个重试循环中，4 个内联的位于各角色的尾段：producer 尾段的最后一次等待（0x1960，以 `EXIT` 结尾）、scheduler 尾段的那次（0x1ea0，同样以 `EXIT` 结尾），以及 warp 0 的累加器尾段的最后一次等待，ptxas 把它的重试发出了两次（0x5860，一个被剥离出来的、没有回跳分支的首次重试，以及 0x58a0，即循环）。这与笔记 G.3 针对 CUTLASS kernel 的 "26 out-of-line + 4 inline loops against 47 + 5" 相符，后者的计数是 116 条 `TRYWAIT` + 54 条 `PHASECHK`，相对于 explicit kernel 的 65 + 32（笔记 D.2、G.3）。把 2 条 `test_wait` 加到这 30 上得到 32 条不带 `.TRYWAIT` 的 `SYNCS.PHASECHK.TRANS64`。这些计数是 ptxas 的，不属于契约的一部分（笔记 D.7）。

### 12.2.3 epilogue 元素：`LDTM`、`FMUL`、`FFMA`、`STS` [M]

```
        /*2290*/                   LDTM.x16 R4, tmem[UR29] ;
        /*22a0*/                   UISETP.NE.U32.AND UP1, UPT, UR16, URZ, UPT ;
        /*22b0*/                   FMUL R4, R4, UR26 ;
        /*22c0*/                   FMUL R5, R5, UR26 ;
        ...
        /*2300*/                   FFMA R4, RZ, UR27, R4 ;
        ...
        /*23a0*/                   STS [R23+UR7+0x200], R4 ;
        ...
        /*23e0*/                   STS [R22+UR7+0x280], R5 ;
        ...
        /*2420*/                   STS [R21+UR7+0x300], R6 ;
        ...
        /*2460*/                   STS [R20+UR7+0x380], R7 ;
        ...
        /*24a0*/                   STS [R23+UR7+0x400], R8 ;
```

`LDTM` 的源码是 explicit.cu:1036，算术运算和存储的源码是 explicit.cu:1043-1048（引于第 10.7 节和第 10.9 节）。

把 `STS` 的立即数对照 explicit.cu:1047 的存储地址公式 `st_off[j & 3] + 512 (j >> 2) + 8192 (e & 3)` 来读。`st_off[c]` 贡献 `OFF_SMEM_CD (0x200) + 128 c (0x80 c)`，加上存放在 `R20..R23` 中的 warp 与 lane 项 `2048 w + 4 (l & 7) + 32 (((l >> 3) & 3) ^ c)`；`512 (j >> 2)` 就是 `0x200 (j >> 2)`；`8192 (e & 3)` 就是 `0x2000 (e & 3)`。对于 `j = 0..4`，立即数是 `0x200, 0x280, 0x300, 0x380, 0x400`，基址寄存器是 `R23, R22, R21, R20, R23`：恰好是 `0x200 + 0x80 (j & 3) + 0x200 (j >> 2)`。ptxas 把 `FMUL`、`FFMA` 和 `STS` 交错排列，使存储与剩余的算术运算重叠：`R4..R8` 的 `STS` 在其 `FFMA` 之后 10、12、14、16 和 18 条指令处发射，每四个槽位一次存储。第一条 `FMUL R4` 是通过 scoreboard 等待异步 `LDTM` 的那条指令（从控制字推断，第 12.1.3 节）。

### 12.2.4 CLC 查询行 [M]

```
        /*1b70*/              @!P0 SYNCS.ARRIVE.TRANS64.RED RZ, [R11+URZ], R5 ;     <- lanes 0..3: arm CTA (lane)'s clc_full[idx] with R5 = 0x10 bytes
        ...
        /*1bc0*/                   UGETNEXTWORKID.BROADCAST [UR8], [UR9] ;         <- the try_cancel: response slot UR8, barrier UR9
        ...
        /*1c60*/                   LDS.128 R4, [R4+0x160] ;                        <- the 16-byte response (OFF_CLC_RESPONSE = 352 = 0x160)
        /*1c90*/                   MEMBAR.ALL.CTA ;
        /*1ca0*/                   FENCE.VIEW.ASYNC.S ;                            <- with the MEMBAR.ALL.CTA above: fence.proxy.async.shared::cta
        /*1cc0*/                   SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R2+URZ], RZ ; <- consumer release on CTA 0's clc_empty[idx] (R2 = PRMT(.., RZ, 0x654, ..))
        /*1ce0*/                   LOP3.LUT P2, RZ, R6, 0x1, RZ, 0xc0, !PT ;        <- valid = bit 0 of response word 2
```

这就是第 7 章的 scheduler 循环：上面的七条机器指令，加上该清单省略的三个阻塞等待和 throttle 释放，以及簿记；整个循环是 50 条指令（第 12.5.3 节）。

---

## 12.3 uniform 数据通路，或者说为什么选举消失了

### 12.3.1 `UR`、`UP`、`URZ`、`UPT` 是什么

自 Turing 起，每个 SM 子分区在 32-lane 的向量数据通路之外都有一条标量的 **uniform 数据通路**（Turing 上仅限整数；Blackwell 的 uniform 指令集还有 FP32 算术，例如 `UFADD`、`UFMUL`、`UFFMA`）："this scalar unit executes instructions where all threads use the same input and generate the same output"（Nsight Compute Profiling Guide，流水线描述）。它的状态是一个单独的 **uniform 寄存器**堆 `URn`（每个 warp 一个 32 位值，而不是每个 lane 一个），Turing 上有 64 个，其中包括一个 uniform 零寄存器，按 SASS 记法为 `UR0`-`UR62` 外加 `URZ`；baseline 的 Blackwell SASS（`trace_out_v2/sass_gemm.txt`）最高用到 `UR77`，而 Blackwell 的 uniform 寄存器堆每个 warp 容纳 80 x 32-bit 可寻址的 uniform 寄存器 `UR0`-`UR79`，每个 warp 320 B，在每个子分区 16 个 warp 时每个 SM 子分区 5 KB，其中 `URZ` 是一个不计入这 80 个之内的单独的恒零寄存器，所以 Turing 的 64 这一计数包括 `URZ`，而 Blackwell 的 80 不包括（NVIDIA 内部；Turing 的计数转述自 Jia、Maggioni、Smith、Scarpazza，*Dissecting the NVidia Turing T4 GPU via Microbenchmarking*，arXiv:1903.07486，第 3.5.2 节 "Uniform Registers"，其中写道 "we were able to enumerate the 64 uniform registers supported by Turing (including a Uniform Zero Register URZ and 63 general-purpose uniform registers UR0–UR62)"），以及 **uniform 谓词** `UPn`。`URZ` 是 uniform 零寄存器，`UPT` 是恒真的 uniform 谓词，它们是每 lane 的 `RZ` 与 `PT` 的对应物；这四者中的任何一个被用作目的操作数时都会丢弃该次写入（NVIDIA 内部）。助记符以 `U` 开头的指令（`UMOV`、`UIADD3`、`ULEA`、`USHF`、`ULOP3`、`UISETP`、`USEL`、`UPRMT`、`UPLOP3`）以及该寄存器堆的加载与搬移指令（`LDCU`、`S2UR`、`R2UR`；NVIDIA 的指令表把这些全部归在 Uniform Datapath Instructions 之下）读写该寄存器堆；uniform 分支 `BRA.U` 和 `BRXU` 依据它来分支。PTX 对此没有任何控制权：ptxas 通过证明一个值在整个 warp 内都相同来决定它存放在 `UR` 还是 `R` 中。

这对 GEMM kernel 很重要，因为一次 TMA 加载的操作数（目的地址、屏障、坐标、掩码）、一条 `tcgen05.mma` 的操作数（描述符、TMEM 地址、指令描述符）和一条 `tcgen05.commit` 的操作数（屏障地址、掩码）在构造上就是 warp-uniform 的。它们取决于 cluster rank、tile 坐标、stage 索引和编译期常量，从不取决于 lane。因此 ptxas 把它们保存在 `UR` 中，并把指令本身作为一次**单一的 warp 级操作**发射：`UTMALDG`、`UTMASTG`、`UTCHMMA`、`UTCBAR`、`UGETNEXTWORKID`、`UTMACCTL` 都只接受 `UR`/`UP` 操作数并且每个 warp 发射一次，这就是本讲义所说的 uniform 数据通路指令的含义（NVIDIA 的 Blackwell 指令表只把 `UGETNEXTWORKID` 归在 Uniform Datapath Instructions 之下；`UTMALDG`/`UTMASTG`/`UTMACCTL` 是 Tensor Memory Access 指令，`UTCHMMA`/`UTCBAR` 是 Tensor Core Memory 指令，后者按 Nsight Compute 指南的说法由 `tc` pipe 执行）；即便是 warp 4 的 32-lane TMA 存储，每个 box 也只花费一条 `UTMASTG`（K4b：每次发射时 32 个活跃 lane）。

### 12.3.2 为什么 `elect.sync` 从循环中消失了 [M]

CUTLASS 的 PTX 契约（也是 explicit.cu 的，第 8-9 章）是 "one elected lane issues the TMA/MMA/commit/query"：在 W21/W22 (explicit.cu:740)、W29 (:932)、W30 (:942, :944) 和 W33 (:843) 周围的 `if (elect_one_sync()) { ... }`。PTX 中有 14 个 `elect.sync` 位点，SASS 中有 4 条 `ELECT` 指令，全部位于角色循环之外（地址见第 12.2 节的 `elect.sync` 行）；CUTLASS kernel 的 19 个 PTX 位点以 5 条 SASS `ELECT` 留存下来（笔记 D.2、G.3）。在 producer、MMA 和 scheduler 循环内部**一个也没有**：uniform 指令反正每个 warp 只执行一次，所以把它谓词化到一个 lane 上不会改变任何东西，于是 ptxas 移除了选举（笔记 D.6 第 6 项）。

选举只在发射确实是按 lane 进行的地方、或者某个向量副作用依赖于它的地方留存下来：prologue 的 `lane_pred`（它通过 `VOTEU.ALL UP1, P1`，一次向量到 uniform 的谓词转换，对 `UTMACCTL.PF` 做谓词化）、alloc 原子操作（由一个 lane 执行 `STS`/`STAS`/`ATOMS` 握手）以及 0x5630 处由 `ELECT` 守卫的 `STS.U8`，其来源未查明（不是 relinquish，后者是 warp 集体执行的；NVIDIA 内部）。

`elect.sync` 是确定性的（"the same leader thread is elected for the same membermask every time"，PTX ISA），这就是为什么 "leader lane" 在 producer 的 `expect_tx` 谓词与其 TMA 发射之间是一个稳定的身份（第 8 章）。

教训**不是**“去掉 `elect.sync`”。PTX ISA 赋予 `tcgen05.mma` "single thread semantics, unlike the collective instructions mma.sync or wgmma.mma_async. So, a single thread issuing the tcgen05.mma will result in the initiation of the whole matrix multiply and accumulate operation"。这并不禁止更多的发射线程；它的意思是每一个发射线程都会启动一次完整的 MMA。因此需要一次选举，或者像这里这样一条每个 warp 执行一次的 uniform 指令，以避免把它发射 32 次。CUTLASS 的契约是选举；SASS 表明保留它不花费任何代价。教训是其逆命题：**一条操作数无法被证明为 uniform 的发射指令不会变成 `U*` 指令**。那时选举就有影响了，lane 相关的地址算术会出现在向量寄存器中，并且在该指令之前需要 `R2UR` 搬移（向量到 uniform）。explicit kernel 有 13 条 `R2UR`，对比 CUTLASS kernel 的 150 条（106 条普通的 `R2UR` 加上接下来计入的 44 条 `R2UR.FILL`），以及 0 条 `MOV.SPILL`/`R2UR.FILL`，对比 44 + 44（笔记 G.3）。uniform 寄存器堆而非向量寄存器堆才是 CUTLASS kernel 的紧缺资源（笔记 D.7，推断）。

### 12.3.3 这对你如何编写发射代码意味着什么 [H，基于上文的实测]

1. 只从 warp-uniform 的来源计算每一个 TMA/MMA/屏障操作数：`%cluster_ctarank`/`%cluster_ctaid`、`blockIdx`、流水线的 `{index, phase}` 状态、循环计数器、编译期常量，以及由每个 lane 从共享内存读取的值（例如 `smem + 384` 处的 `T`）。绝不从 `threadIdx.x & 31` 或 lane 相关的加载中计算。
2. 让循环不变的 uniform 值的数量保持很少（描述符基址、屏障基址、掩码、map 地址）。explicit kernel 在 MMA 循环中每 k-tile 多出的三条 `UMOV` 和一条 `ULEA`，以及它在 producer 中每 k-tile 对四个循环不变量的五次重新物化，都是 ptxas 选择了重新计算而不是占住一个 `UR`（笔记 G.3；计数见第 12.5.1-12.5.2 节）。在这里无害，因为这些循环不受发射限制（第 12.5.5 节），但对于一个受发射限制的循环来说，这一机制值得了解。
3. lane 相关的工作属于向量 lane：epilogue 的 `STS` 基址（`R20..R23`）、每 lane 的 `expect_tx` 谓词、由 lane 0..3 进行的 CLC 装填。只要代码把两条数据通路分得干净，ptxas 也会把它们分得干净。
4. 一条 warp-uniform 的分支变成 `BRA.U UPn`（0x12f0 和 0x54d0 处的循环回边是 `BRA.U UP0`/`BRA.U UP1`）：uniform 谓词使这条分支可证明是不发散的 [H：对 `.U` 后缀的解读]。

---

## 12.4 寄存器：26 对 68

### 12.4.1 26 从何而来 [M]

| 寄存器范围 | 保存的内容 | 为什么需要它 |
|---|---|---|
| 低编号寄存器，按角色 | 每 lane 的流水线状态和暂存值，由每个角色复用。MMA warp 把 `cons_phase` 保存在 `R9` 中（0x52f0 处的 `LOP3.LUT R9, R9, UR11, RZ, 0x3c`），把移位后的 parity 保存在 `R2` 中（`SHF.L.U32 R2, R9, 0x1f`），并且在其 k-tile 循环 (0x51f0-0x54d0) 内只触及 `R2` 和 `R9`；每 tile 一次的入口 0x5100-0x5180 增加了 `R0`（累加器 phase，在 0x5130 处移位到 `R2` 中）和 `R4`（0x5160 处的 peek parity）。producer 把 `prod_phase` 保存在 `R0` 中（0x1190 处的 `LOP3.LUT R0, R0, UR8, ...`）并使用 `R4`、`R5`。scheduler 循环使用 `R0` 和 `R2..R12`（响应字 `R4..R7`、装填地址 `R11`、屏障基址 `R12`）。epilogue 除下面两个范围之外还使用 `R0`、`R2`、`R3`、`R8` 和 `R9`（它的 `acc_full` parity：0x2180 处的 `TRYWAIT P0, [UR24+0xf0], R9`）[M] | phase 位是向量值，因为 wrapper 把它们作为 `"r"` 操作数接受。由于每个角色都复用低编号寄存器，仅 epilogue 的活跃集合就决定了 kernel 的最大值 |
| `R4..R19` | 16 个 `LDTM.x16` 目的寄存器 = 一个 lane 在一个 subtile 上的 16 个 fp32 累加器值；也是就地存放的 `FMUL`/`FFMA` 结果 | `tcgen05.ld.32x32b.x16` 每 lane 写 16 个寄存器 (W31) |
| `R20..R23` | 四个 lane 相关的 `STS` 基址 `st_off[c]` (explicit.cu:1011-1014)：`R23` 对应 `j & 3 = 0`，`R22` 对应 1，`R21` 对应 2，`R20` 对应 3 | lane 项 `4 (l & 7)` 和 `Swizzle<2,5,2>` 的 XOR 项 `32 (((l >> 3) & 3) ^ c)` 都依赖于 lane（第 10 章） |
| `R1` | 按代码生成约定，编译器的每线程局部内存栈基址（NVIDIA 内部）；0x0000 处的 `LDC R1, c[0x0][0x37c]` 是对 `R1` 的唯一引用 | ABI/代码生成约定，而非架构上的承诺（NVIDIA 内部）；这里未使用，没有任何 `STL`/`LDL` 寻址它，并且 `STACK` 为 0 |

其余一切（屏障地址、描述符、坐标、alpha、beta、`k_tiles`、流水线状态）都存放在 uniform 寄存器中。CUTLASS kernel 的最高寄存器是 `R65`。它额外携带了零填充的 C 片段（`R21..R36`，16 个寄存器：`beta != 0` 路径的操作数，在 beta = 0 时是死的但仍被分配）、`epi_load` 流水线状态、为 scheduler 簿记而保存在向量寄存器中的每个 tile 的 CLC 解码，以及 44 对 `MOV.SPILL`/`R2UR.FILL`（一个 uniform 值被停放在向量寄存器中再搬回来，88 条指令），其中 40 对位于从不执行的 C 加载路径中（笔记 D.2、G.3）。

### 12.4.2 为什么在这里无关紧要，以及何时会有影响 [S, H]

- 共享内存把 occupancy 固定在**每个 SM 1 个 CTA**：可选启用的 232448 B 中的 230400 B 动态 smem（util.hpp 的 `static_assert`），而 `__launch_bounds__(256, 1)` 只要求至少那一个（第 11.2.1 节）。256 个线程 x 68 个寄存器 = 17408，而 SM 共有 65536 个 32 位寄存器；即使 255 x 256 = 65280 也放得下。寄存器数量不可能改变 occupancy，而且它也没有改变发射速率（第 12.5 节）。寄存器以 warp/线程块粒度按舍入后的块分配，即 Nsight Compute 的 occupancy 数据中的 "register allocation unit size"（NVIDIA 内部：这是一个 occupancy 核算量子，而不是 SASS 寄存器宽度或指令编码；Nsight Compute 指南说寄存器是 "allocated by the HW in fixed-size chunks"），所以几个寄存器的差别在跨过一个块边界之前不会改变任何东西；这一粒度随计算能力而变，所以请从该数据（`reg_allocation_unit_size`、`warp_reg_allocation_granularities`）中读取 CC 10.0 的值，而不是从反汇编中的寄存器编号范围推导它（NVIDIA 内部）；它的值在此未经验证。
- 它会在三种情况下有影响。(a) 一个为每个 SM 2 个或更多 CTA 而设计的 kernel（更小的 smem 占用），其中 256 个线程 x >128 个寄存器会使 occupancy 减半。(b) 一个需要大量寄存器的 epilogue（例如 `tcgen05.ld.32x32b.x32` = 每次加载 32 个寄存器，再加上转换临时量和多个在途的 subtile），其中 ptxas 会开始溢出；每 subtile 循环内部的一次溢出（`STL`/`LDL`）会落在 epilogue 的关键路径上。(c) Hopper 风格的 `setmaxnreg` 在 producer 与 consumer warp 之间的寄存器重新分配，本 kernel 不使用它，因为 1 CTA/SM 加上 8 个 warp 反正已让寄存器堆约 90 % 为空（256 x 26 = 6656，而寄存器共有 65536 个；即使按 CUTLASS kernel 的 68 个算也有 73 % 为空）。
- 因此要盯住的数字不是 `REG`，而是 `LOCAL` 和 `STACK`（两者都为 0），以及热循环中没有 `LDL`/`STL`。

---

## 12.5 四个热循环

本节中的所有地址和计数都是我对 `trace_out_v2/sass_explicit.txt` 的重新清点；除另有说明之处外，它们与笔记 G.3 一致。

### 12.5.1 MMA warp：每 k-tile 47 条指令 (0x51f0-0x54d0) [M]

warp 0 如何到达该循环，每 tile 一次（0x5180-0x51e0，逐字照录；注释是我的）：

```
/*5180*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR35+0x110], R2 ;   acc_empty[st] blocking wait (UR35 = base + 8 st, +0x110 = OFF_ACC_EMPTY)
/*5190*/              @!P2 BRA 0x62f0 ;                                             out-of-line retry loop
/*51a0*/                   BRA.U !UP0, 0x54e0 ;                                     UP0 = (k_tiles >= 1), set at 0x5100: skip the k-tile loop (not the commit) when there is none; the non-leader CTA already left at 0x50f0
/*51b0*/                   ULEA UR36, UR5, UR15, 0x7 ;                              tmem_c = T + 128 st   (UR5 = st, UR15 = T)
/*51c0*/                   UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x40, 0x4 ;          UP0 = false: scale_c = 0 for the first MMA of the tile
/*51d0*/                   UMOV UR6, UR14 ;                                         k = k_tiles
/*51e0*/                   UMOV UR8, UR4 ;                                          s = cons_idx
```

这里有三点需要读出。第一，`UP0` 被重载了。在 0x51a0 处它保存 `k_tiles >= 1`（0x5100 处的 `UISETP.GE.AND UP0, UPT, UR14, 0x1, UPT`），即 explicit.cu:922 处的 `while (k > 0)` 守卫：没有 k-tile 时循环被跳过，只运行 0x54e0 处的 commit 操作。非 leader CTA 的 warp 0 从不到达这个块：0x50f0 处的 `BRA.U UP0, 0x5510`（`UP0 = UR34 != 0` 来自 0x5020，`UR34 = rank & 1` 来自 0x0730，即 `!is_leader`，explicit.cu:1106）越过 `acc_empty` 等待、四条 MMA 和 commit 操作，跳到 0x5510 处的累加器状态推进，所以每个 tile 它只执行 CLC 消费和那次推进（第 9 章）。从 0x51c0 起，`UP0` 是 scale-c 谓词：在此处每 tile 清零一次，并在循环内部的 0x54b0 处置位（第 12.2.1 节）。第二，`UR36`（`tmem_c`）、循环计数器 `UR6 = k_tiles`（`UR14`，在 0x0a70-0x0ac0 处由从 `c[0x0][0x588]` 加载的 `k` 一次性算得 `(k + 63) >> 6`）和 stage `UR8` 每 tile 形成一次，这就是为什么下面的循环体中不包含它们中的任何一个。第三，`acc_empty[st]` 等待是在 leader 自己的屏障上（0x5120 处的 `UR35 = ULEA(st, base, 3)`，未加掩码），即全部 256 个 epilogue 线程在其上到达的那个屏障（第 4 章）。循环之后，0x54e0 处由同一个寄存器形成 `acc_full[st]`（`UIADD3 UR6, UR35, 0xf0`）以用于累加器的 commit 操作。

循环体，完整且按原有顺序，并按用途分组（SASS 是逐字照录的；右侧的注释是我的）：

```
--- blocking wait on full[s], skipped when the peek succeeded (4 instructions, off the fast path) ---
/*51f0*/               @P0 BRA 0x5240 ;
/*5200*/                   ULEA UR4, UR8, UR7, 0x3 ;              full[s] = base + 8 s
/*5210*/                   SHF.L.U32 R2, R9, 0x1f, RZ ;           parity -> bit 31
/*5220*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4], R2 ;
/*5230*/              @!P0 BRA 0x6350 ;                           out-of-line retry loop
--- pipeline state: rs = s; advance {s, phase}; decide whether to peek (C++: k > 0 after --k; SASS: UR6 != 1 before the decrement at 0x54a0) ---
/*5240*/                   UISETP.NE.U32.AND UP2, UPT, UR6, 0x1, UPT ;   UP2 = (k != 1): peek the next stage
/*5250*/                   PLOP3.LUT P0, PT, PT, PT, PT, 0x80, 0x8 ;      P0 = true (tok = 1 when no peek)
/*5260*/                   UIADD3 UR4, UPT, UPT, UR8, 0x1, URZ ;          s + 1
/*5270*/                   ULEA UR10, UR8, UR30, 0xa ;                    lo_a = desc_lo_a + 1024 s
/*5280*/                   UISETP.NE.U32.AND UP1, UPT, UR4, 0x8, UPT ;    wrap test (s + 1 == 8)
/*5290*/                   PLOP3.LUT P2, PT, PT, PT, UP2, 0x80, 0x8 ;     P2 = UP2 as a vector predicate
/*52a0*/                   ULEA UR12, UR8, UR31, 0x9 ;                    lo_b = desc_lo_b + 512 s
/*52b0*/                   USEL UR11, URZ, 0x1, UP1 ;                     phase flip bit
/*52c0*/                   USEL UR4, UR4, URZ, UP1 ;                      s = wrap ? 0 : s + 1
/*52d0*/                   ULEA UR32, UR8, UR7, 0x3 ;                     (base + 8 rs) -> empty[rs] after + 0x40
/*52e0*/               @UP2 ULEA UR8, UR4, UR7, 0x3 ;                     full[s] address for the peek (s already advanced and wrapped at 0x52c0)
/*52f0*/                   LOP3.LUT R9, R9, UR11, RZ, 0x3c, !PT ;         cons_phase ^= flip
--- descriptor low words for k-blocks 1..3 (+2, +4, +6 = +32, +64, +96 bytes) ---
/*5300*/                   UIADD3 UR16, UPT, UPT, UR10, 0x2, URZ ;
/*5310*/                   UIADD3 UR18, UPT, UPT, UR12, 0x2, URZ ;
/*5320*/               @P2 SHF.L.U32 R2, R9, 0x1f, RZ ;                   peek parity
/*5330*/                   UIADD3 UR20, UPT, UPT, UR10, 0x4, URZ ;
/*5340*/                   UIADD3 UR22, UPT, UPT, UR12, 0x4, URZ ;
/*5350*/               @P2 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR8], R2 ; the peek of the next stage (predicated off on the last k-tile)
/*5360*/                   UIADD3 UR24, UPT, UPT, UR10, 0x6, URZ ;
/*5370*/                   UIADD3 UR26, UPT, UPT, UR12, 0x6, URZ ;
/*5380*/                   UIADD3 UR32, UPT, UPT, UR32, 0x40, URZ ;        empty[rs] = base + 64 + 8 rs
--- constants the PTX materialises per k-block (mov.b32 for UR29, the idesc; UR28 = 0 fills the sparse-metadata slot, which the dense PTX form has no operand for, Section 12.2; the 0x40004040 high words come from the PTX or.b64 immediate 0x4000404000000000, one per descriptor) ---
/*5390*/                   UMOV UR28, 0x0 ;                               tmem[UR28]: sparse-metadata TMEM operand, zero for a dense MMA (NVIDIA internal)
/*53a0*/                   UMOV UR29, 0x10200010 ;                        idesc
/*53b0*/                   UMOV UR11, 0x40004040 ;                        descriptor high words x 8
/*53c0*/                   UMOV UR13, 0x40004040 ;
/*53d0*/                   UMOV UR17, 0x40004040 ;
/*53e0*/                   UMOV UR19, 0x40004040 ;
/*53f0*/                   UMOV UR21, 0x40004040 ;
/*5400*/                   UMOV UR23, 0x40004040 ;
--- the four MMAs and the release ---
/*5410*/                   UTCHMMA.2CTA gdesc[UR10], gdesc[UR12], tmem[UR36], tmem[UR28], idesc[UR29], UP0 ;
/*5420*/                   UMOV UR25, 0x40004040 ;
/*5430*/                   UMOV UR27, 0x40004040 ;
/*5440*/                   UTCHMMA.2CTA gdesc[UR16], gdesc[UR18], tmem[UR36], tmem[UR28], idesc[UR29], UPT ;
/*5450*/                   UMOV UR37, 0xf ;                               kMainloopReleaseMask
/*5460*/                   UTCHMMA.2CTA gdesc[UR20], gdesc[UR22], tmem[UR36], tmem[UR28], idesc[UR29], UPT ;
/*5470*/                   UTCHMMA.2CTA gdesc[UR24], gdesc[UR26], tmem[UR36], tmem[UR28], idesc[UR29], UPT ;
/*5480*/                   UTCBAR.2CTA.MULTICAST [UR32], URZ, UR37 ;      tcgen05.commit -> empty[rs] of all four CTAs
--- loop counter, scale-c, back-edge ---
/*5490*/                   UISETP.GT.U32.AND UP1, UPT, UR6, 0x1, UPT ;    k > 1 ?
/*54a0*/                   UIADD3 UR6, UPT, UPT, UR6, -0x1, URZ ;         --k
/*54b0*/                   UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x80, 0x8 ; UP0 = true: scale_c = 1 from now on
/*54c0*/                   UMOV UR8, UR4 ;                                cons_idx = s
/*54d0*/                   BRA.U UP1, 0x51f0 ;
```

构成，按数据通路分组：32 条 uniform 数据通路 ALU 操作（12 条 `UMOV`、9 条 `UIADD3`、5 条 `ULEA`、2 条 `UISETP.NE`、1 条 `UISETP.GT`、2 条 `USEL`、1 条 `UPLOP3`）、5 条 TC-pipe 指令（4 条 `UTCHMMA`、1 条 `UTCBAR`）、2 条屏障操作（2 条 `TRYWAIT`）、5 条向量数据通路指令（2 条 `SHF`、1 条 `LOP3`、2 条 `PLOP3` 谓词搬移）和 3 条分支；5 + 2 + 5 + 3 + 32 = 47。除了两条 `TRYWAIT`（parity 来自每 lane 的 `R2`，结果放在向量谓词 `P0` 中）和分支之外，warp 0 的向量 lane 每 k-tile 至多发射那 5 条指令（快速路径上为 4 条：0x5210 处的 `SHF` 位于被跳过的阻塞等待中）：它们基本上是空闲的。CUTLASS kernel 的循环（`sass_gemm.txt` 中的 0x3890-0x3b40）是 44 条 [M]。这 +3 是 `UMOV UR28/UR29`（只在 CUTLASS kernel 中被外提到循环之外）、`UMOV UR37, 0xf`（CUTLASS kernel 在 tile 循环之前于运行时构造该掩码）和重新计算 `empty[rs]` 的那条 `ULEA`，再减去一条循环计数器指令（笔记 G.3）。有两点值得注意。第一次迭代**没有被剥离**：explicit.cu:916 处的 C++ `tok` 变成了 `P0`，跨回边传递。并且 0x5350 处的 peek 在最后一个 k-tile 上被谓词关闭（explicit.cu:927 处的 `(k <= 0) ? 1u : mbar_try_wait(...)`），给出每 tile 恰好 1 + 127 = 128 次 peek（第 9 章）。

### 12.5.2 Producer warp 2：每 k-tile 37 条指令 (0x10b0-0x12f0)，快速路径上为 33 条 [M]

```
/*10b0*/               @P1 BRA 0x1100 ;                                  peek succeeded: skip the blocking wait
/*10c0*/                   ULEA UR4, UR28, UR7, 0x3 ;
/*10d0*/                   SHF.L.U32 R4, R0, 0x1f, RZ ;
/*10e0*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [UR4+0x40], R4 ;      empty[s] = base + 64 + 8 s
/*10f0*/              @!P1 BRA 0x5c80 ;
/*1100*/                   UIADD3 UR4, UPT, UPT, UR28, 0x1, URZ ;                   s + 1
/*1110*/              @!P0 MOV R5, 0xc000 ;                                         49152 (leader lane only)
/*1120*/                   VOTEU.ALL UP1, P0 ;
/*1130*/                   USHF.L.U32 UR18, UR13, 0x6, URZ ;                        c0 = 64 q
/*1140*/                   UISETP.NE.U32.AND UP0, UPT, UR4, 0x8, UPT ;              wrap
/*1150*/              @!UP1 ULEA UR10, UR28, UR7, 0x3 ;                             full[s] (unmasked, local) for the expect_tx
/*1160*/                   USEL UR8, URZ, 0x1, UP0 ;
/*1170*/                   USEL UR4, UR4, URZ, UP0 ;
/*1180*/                   UIADD3 UR30, UP0, UPT, UR22, 0x80, URZ ;                 &params.tma_b = param base + 0x80 (low word)
/*1190*/                   LOP3.LUT R0, R0, UR8, RZ, 0x3c, !PT ;                    prod_phase ^= flip
/*11a0*/                   ULEA UR8, UR28, UR7, 0xd ;                               base + (s << 13)
/*11b0*/              @!P0 SYNCS.ARRIVE.TRANS64 RZ, [UR10], R5 ;                    mbarrier.arrive.expect_tx full[s], 49152
/*11c0*/                   ULEA UR9, UR4, UR7, 0x3 ;                                empty[next] for the peek (UR4 = s + 1, wrapped at 0x1170)
/*11d0*/                   SHF.L.U32 R4, R0, 0x1f, RZ ;
/*11e0*/                   ULEA UR17, UR28, UR35, 0x3 ;                             masked full[s] = (base & 0xfefffff0) + 8 s
/*11f0*/                   ULEA UR16, UR28, UR24, 0xe ;                             A dst = (base + 0x8400 + 8192 y) + (s << 14)
/*1200*/                   UPRMT UR34, URZ, 0x5410, UR25 ;                          multicast mask (0x5 << x) into bits [16,32)
/*1210*/                   UIADD3 UR8, UPT, UPT, UR8, 0x28400, URZ ;                B dst = base + 0x28400 + (s << 13)
/*1220*/                   UIADD3.X UR31, UPT, UPT, URZ, UR23, URZ, UP0, !UPT ;     &params.tma_b high word
/*1230*/                   UMOV UR20, URZ ;                                         z coordinate = 0
/*1240*/                   UMOV UR32, 0x0 ;                                         cache hint low
/*1250*/                   UMOV UR33, 0x10000000 ;                                  cache hint high -> 0x1000000000000000
/*1260*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [UR9+0x40], R4 ;      peek empty[next]
/*1270*/                   UMOV UR9, UR17 ;                                         B's barrier
/*1280*/                   UMOV UR10, UR18 ;                                        B's c0
/*1290*/                   UMOV UR12, UR20 ;                                        B's z
/*12a0*/                   UTMALDG.3D.MULTICAST.2CTA [UR16], [UR22], UR34, desc[UR32] ;   A
/*12b0*/                   UIADD3 UR13, UPT, UPT, UR13, 0x1, URZ ;                  ++q
/*12c0*/                   UMOV UR28, UR4 ;                                         prod_idx = s + 1 (wrapped)
/*12d0*/                   UISETP.NE.U32.AND UP0, UPT, UR13, UR39, UPT ;            q != end ?
/*12e0*/                   UTMALDG.3D.2CTA [UR8], [UR30], desc[UR32] ;              B
/*12f0*/                   BRA.U UP0, 0x10b0 ;
```

发射顺序正是源码所规定的那一个 (explicit.cu:732-755)：等待、`expect_tx`、推进、peek、A 加载、B 加载。CUTLASS kernel 的循环（`sass_gemm.txt` 中的 0x1d60-0x1f40）是 31 条（快速路径 28 条）。把两个循环体逐条指令对应起来，剩下 28 条共有、9 条 explicit 独有、3 条 baseline 独有，净 +6 [M]。这九条 explicit 独有的指令分为三组。(1) 五条重新物化的不变量：`@!P0 MOV R5, 0xc000`（baseline 把 `@P0 MOV R2, 0xc000` 外提到每个循环的前面，0x1a10 和 0x1cf0）、`UIADD3 UR30` + `UIADD3.X UR31`（`tma_b` 地址对）、`UPRMT UR34`（掩码插入）以及 `UMOV UR20, URZ`（z = 0）。(2) `VOTEU.ALL UP1, P0` 以及由它作谓词控制的 `@!UP1 ULEA UR10`。(3) 另外两次 `full[s]` 地址构造：用于阻塞等待的 `ULEA UR4` 和用于带掩码 TMA 屏障的 `ULEA UR17`。baseline 只构造一次 `full[s]`（0x1d60 处的 `ULEA UR17, UR4, UR29, 0x3`，位于其 `@P1 BRA` 之前，同时服务于等待和 `expect_tx`），并用 `ULOP3.LUT UR17, UR17, 0xfefffff8` (0x1e80) 原地给它加掩码。三条 baseline 独有的指令是那条 `ULEA`、那条 `ULOP3`，以及 `UIADD3 UR16, UR16, 0x8400` (0x1e90)，explicit kernel 把后者折叠进了外提的 `UR24 = base + 0x8400 + 8192 y`。笔记 G.3 把同一个净值表述为 "eight explicit-only in-loop instructions minus two baseline-only ones"；它的八和二正是它所点名的那些指令，包括 `@!UP1 ULEA UR10` 和 baseline 的无条件 `ULEA`，因此它隐含地把等待块的 `ULEA UR4` 与那个 `0x8400` 加法配成了一对，而这两条它都没有提及。8-k-tile 的 prologue 循环 (0x0de0-0x1020) 同样是 37 条，而 baseline 为 32 条，并带有它自己的两条 `UTMALDG`，位于 0x0fc0/0x1000。

### 12.5.3 CTA 0 的 scheduler warp 1：每 tile 50 条指令 (0x1a20-0x1d30) [M]

头部位于 0x1a20（0x1d30 处 `@P2 BRA 0x1a20` 的目标），因此循环体是 `(0x1d30 - 0x1a20) / 16 + 1 = 50`，与 CUTLASS kernel 的 CLC 循环（`sass_gemm.txt` 中的 0x2b60-0x2e70）计数和结构相同；八个整数加法/搬移槽位的拼写不同（这里是 `IADD3`/`MOV`/`LEA`，那里是 `VIADD`/`IMAD.U32`/`IMAD.MOV.U32`/`IMAD`；笔记 G.3 说法相同）。按顺序：throttle 等待（`TRYWAIT [R5+URZ+0x130]`）和远程释放（`PRMT R4, RZ, 0x654, R4; SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R4+URZ], RZ`）、CLC empty 等待（`TRYWAIT P1, [UR6+0xe0], R6`）、由 lane 0..3 装填（`@!P0 SYNCS.ARRIVE.TRANS64.RED RZ, [R11+URZ], R5`，`R5 = 0x10`）、`UGETNEXTWORKID.BROADCAST [UR8], [UR9]`、响应等待（`TRYWAIT P2, [R12+URZ+0xd0], R5`）、`LDS.128`、`MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`、向 CTA 0 释放、解码。尾段 (0x1d40-0x1ed0) 是两个 `SYNCS.PHASECHK.TRANS64` 测试等待及其阻塞回退；该 warp 通过 0x1e60 或 0x1ec0 处的 `@P0 EXIT` 退出，而 0x1ed0 处的 `BRA 0x1e80` 是第二个回退的重试回边。

### 12.5.4 Epilogue warp 4-7：循环内每 tile 674 条，函数内 696 条 [M]

epilogue 角色从 0x1f10（`BAR.SYNC.DEFER_BLOCKING 0x6, 0xa0`，即 explicit.cu:997 的 `named_bar_sync(kBarTmemAlloc, ...)`）跨越到 0x4a80（`EXIT`）：696 条指令，即笔记 G.3 报告为 "per tile" 的那个数字。我的重新计数把它拆成三部分。一次性的入口 0x1f10-0x2050 是 21 条指令：alpha/beta 和参数基址的 `LDCU.64`、四个 `st_off` 基址 `R20..R23`、来自 `[UR7+0x180]` 的 `T`、`UR17 = T + ((32 w) << 16)`（0x2050 处的 `ULEA UR17, UR16, UR17, 0x15`）。tile 循环 0x2060-0x4a70 是 674 条指令（回边 `BRA.U UP0, 0x2060` 位于 0x4a70）。然后是 `EXIT`。循环内部：

| 段 | 地址 | 指令数 | 内容 |
|---|---|---|---|
| per-tile 设置 | 0x2060-0x2280 | 35 | `clc_consume`（CLC 等待、`LDS.128`、fence、向 CTA 0 释放、解码为 `tm/tn`）、`acc_full[st]` 的 peek、TMEM 地址 `UR29 = T + ((32w) << 16) + 128 st` |
| subtile 0 | 0x2290-0x2750 | 77 | `LDTM.x16`、16 条 `FMUL`、16 条 `FFMA`、16 条 `STS`、`MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S`、`BAR.SYNC 0x1, 0x80`、仅 warp 4 的存储块（有 `BRA.U UP1` 绕过它，其 `UP1 = (w != 0)` 在 0x22a0 处每 tile 设置一次：为四个 `{src, c0, c1, c2}` 元组做的 `UIADD3`/`UMOV` 簿记、4 条 `UTMASTG.3D`、`UTMACMDFLUSH`、`DEPBAR.LE SB0, 0x1`）、`BAR.SYNC 0x1, 0x80` |
| subtile 1-5 | 0x2760-0x3fb0 | 每个 78 | 内容相同；与 subtile 0 的净一条指令之差，是多出两条簿记指令（用于 `c1` 的 `UIADD3 UR10, UR22, 16 e` 和用于 box 0 的 `c0` 的 `UMOV UR9, UR21`，这两个值 subtile 0 直接取自 `UR22`/`UR21`）减去只有 subtile 0 才带有的每 tile 一次的 `UISETP.NE.U32.AND UP1` [M] |
| subtile 6 | 0x3fc0-0x44c0 | 81 | 相同，外加为 subtile 7 构建带掩码的 `acc_empty[st]` 地址的 `ULOP3 ... 0xfefffff8` 和 `UIADD3 ... 0x110`，以及累加器状态推进（`st + 1`，explicit.cu:1040）的 `UIADD3 UR4, UR4, 0x1`，ptxas 把它调度到了这里 |
| subtile 7 | 0x44d0-0x4a70 | 91 | `LDTM.x16 R4, tmem[UR29+0x70]; NOP; SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [UR24], RZ`（explicit.cu:1038-1039 的 `tcgen05.wait::ld` + 释放，该 arrive 等待的是 `LDTM` 的 scoreboard）、算术运算与存储、累加器状态推进、`tm = tm_next` 等、循环测试 |

subtile 0 的存储块（0x25f0-0x2740，逐字照录）展示了 warp 4 如何构造 `UTMASTG` 操作数块所指名的四个 `{src, c0, c1, c2}` 元组：

```
        /*25f0*/                   UIADD3 UR36, UP2, UPT, UR30, 0x180, URZ ;
        /*2600*/                   UIADD3 UR20, UPT, UPT, UR7, 0x200, URZ ;
        /*2610*/                   UIADD3.X UR37, UPT, UPT, URZ, UR31, URZ, UP2, !UPT ;
        /*2620*/                   UIADD3 UR9, UPT, UPT, UR21, 0x20, URZ ;
        /*2630*/                   UIADD3 UR8, UPT, UPT, UR7, 0xa00, URZ ;
        /*2640*/                   UIADD3 UR13, UPT, UPT, UR21, 0x40, URZ ;
        /*2650*/                   UIADD3 UR12, UPT, UPT, UR7, 0x1200, URZ ;
        /*2660*/                   UIADD3 UR33, UPT, UPT, UR21, 0x60, URZ ;
        /*2670*/                   UIADD3 UR32, UPT, UPT, UR7, 0x1a00, URZ ;
        /*2680*/                   UMOV UR23, URZ ;
        /*2690*/                   UMOV UR10, UR22 ;
        /*26a0*/                   UMOV UR11, UR23 ;
        /*26b0*/                   UTMASTG.3D [UR20], [UR36] ;
        /*26c0*/                   UMOV UR14, UR22 ;
        /*26d0*/                   UMOV UR15, UR23 ;
        /*26e0*/                   UMOV UR34, UR22 ;
        /*26f0*/                   UMOV UR35, UR23 ;
        /*2700*/                   UTMASTG.3D [UR8], [UR36] ;
        /*2710*/                   UTMASTG.3D [UR12], [UR36] ;
        /*2720*/                   UTMASTG.3D [UR32], [UR36] ;
        /*2730*/                   UTMACMDFLUSH ;
        /*2740*/                   DEPBAR.LE SB0, 0x1 ;
```

`UR30:UR31` 是参数块地址（0x1f50 处的 `LDCU.64 UR30, c[0x0][0x348]`），所以 `UR36:UR37 = +0x180` 是 `&params.tma_d`。`UR7` 是 `smem_base`，所以四个源是 `OFF_SMEM_CD + 2048 i` = `0x200, 0xa00, 0x1200, 0x1a00`（一个 32 行 x 16 列的 fp32 box 是 2048 B）。`UR21 = 128 tm` 和 `UR22 = 128 tn` 每 tile 构造一次（0x2120-0x2130 处的 `USHF.L.U32 UR21, UR21, 0x7` 和 `USHF.L.U32 UR22, UR29, 0x7`）。于是 `c0 = 128 tm + 32 i` 是 `UR21`、`UR9 = UR21 + 0x20`、`UR13 = UR21 + 0x40`、`UR33 = UR21 + 0x60`。`c1 = 128 tn + 16 e` 对 subtile 0 是 `UR22`，为 box 1-3 复制到 `UR10`、`UR14`、`UR34`。`c2 = 0` 是 `UR23 = URZ`，复制到 `UR11`、`UR15`、`UR35`。因此这些元组是 `{UR20, UR21, UR22, UR23}`、`{UR8, UR9, UR10, UR11}`、`{UR12, UR13, UR14, UR15}`、`{UR32, UR33, UR34, UR35}` [块的解读为 H，算术为 M]。对 subtile e 只有两个值改变。`c1` 变为 `UR22 + 16 e`（subtile 1：0x2ac0 处的 `UIADD3 UR10, UR22, 0x10`，然后是用于 box 0 的 `c0` 的 `UMOV UR9, UR21`）。源移到 epilogue 的 smem stage `0x200 + 0x2000 (e & 3) + 0x800 i`（subtile 1：`UR7 + 0x2200, 0x2a00, 0x3200, 0x3a00`）。D-map 地址保持其值不变，但 ptxas 为每个展开的 subtile 重新构造它，在 subtile 0 中构造进 `UR36`，在 subtile 1-6 中进 `UR40`，在 subtile 7 中进 `UR8`（0x2ab0 处的 `UIADD3 UR40, UP2, UPT, UR30, 0x180`，所以 subtile 1 存储的是 `UTMASTG.3D [UR8], [UR40]`）。把 subtile 0 中的 `[UR36]` 与 subtile 1 中的 `[UR40]` 作比较的读者，看到的是同一个 64 位值位于两个寄存器中，而不是一个错误。

每个 subtile，两个 kernel 都有 1 条 `LDTM.x16`、16 条 `FMUL`、16 条 `FFMA`、16 条 `STS`、1 对 fence、2 条 `BAR.SYNC`、4 条 `UTMASTG`、1 条 `UTMACMDFLUSH`、1 条 `DEPBAR`。CUTLASS kernel 每 tile 的 1265 条则在每个 subtile 上多出它那 16 条位于一个永不成立的谓词之后的对 C 的死 `LDS`、死的 `epi_load` 流水线的三条 `SYNCS`、一条 `R2UR`、一条 `NOP` 和谓词簿记，外加一个 309 条指令的设置（笔记 G.3）。

### 12.5.5 为什么这些都不在关键路径上 [输入为 M，模型为 H]

每 k-tile 的时间预算由实测运行时间推出（笔记 D.4）。关键路径上的一个 cluster 在 0.888 ms 内处理 32 个 tile，因此一个 tile 最多耗时 27.75 us，一个 k-tile 则是 27.75 us / 128 = 216.8 ns = 1965 MHz 下的 **426 个周期**。因此一次 256 x 128 x 16 的 MMA 有 106.5 个周期。如果这 1024 个 tile 按平均每 cluster 31.03 个来分摊，这些数字会大 3.1 % (32 / 31.03 = 1.031)：每 tile 28.6 us，每 k-tile 439 个周期，每次 MMA 109.8 个周期，即笔记 D.4 引用的第二个数字。

发射模型：一个 SM 有四个子分区，每个子分区有一个 warp scheduler（Nsight Compute Profiling Guide，Hardware Model）。该指南 Scheduler Statistics 一节的描述说："From the set of eligible warps the scheduler selects a single warp from which to issue one or more instructions." 我把**每个 scheduler 每周期一条指令**作为发射界限 [H：该指南没有陈述每周期的速率；如果每周期能发射多于一条指令，下面的利用率数字就是上界]。四个 scheduler 上的八个 warp 意味着每个 scheduler 两个 warp；是哪两个，在我所知的架构上是 `warp_id % 4`，对这块芯片未验证。

| Warp | 每 k-tile 的指令数（静态，快速路径） | 占一个 scheduler 的 426 个槽位的份额 |
|---|---|---|
| 0（MMA，leader CTA） | 43（47 减去被跳过的 4 条等待指令） | 10.1 % |
| 2（producer） | 33 | 7.7 % |
| 4-7（epilogue），每个 | 每 tile 674 条 / 128 个 k-tile = 5.3 | 1.2 % |
| 1（scheduler，仅 CTA 0） | 每 tile 50 条 / 128 = 0.4 | 0.1 % |
| 1（rank 1-3）、3 | 已退出 | 0 |

即使 warp 0 和 4 共用一个 scheduler，它们合计的需求也约为其发射容量的 11 %：最忙的 scheduler 每十个周期有九个是空闲的。此外，warp 0 的 uniform 数据通路指令并不与 epilogue 的 `FMUL`/`FFMA`/`STS` 竞争，后者使用的是向量 pipe。只有当 tensor core 在少于 ~43 个周期内完成四次 MMA 时，MMA warp 才可能成为瓶颈，也就是说硬件得比实际快上大约十倍。只要下一个 k-tile 的 `UTCHMMA` 在 tensor core 排空之前发射，循环长度就无关紧要。8-stage 环和先 peek 再 wait 的惯用写法（`full[s]` 等待几乎总是在内联尝试时就成功）使这成为常见情形。同样的推理适用于 producer：33 个发射槽使每 k-tile 每 pair 有 49152 字节在途，最多提前 8 个 stage 运行（第 8 章）。

**确实**在关键路径上的，是这些循环所编码的依赖链：`full[s]` 完成（TMA 字节已落地） -> `UTCHMMA` 发射 -> tensor-core 执行 -> `UTCBAR` -> `empty[s]` -> 下一条进入 stage s 的 `UTMALDG`；以及每 tile 的 `acc_full[st]` -> `LDTM` -> `STS` -> `UTMASTG` -> `acc_empty[st]`。第 3 章推导了那条链的 speed-of-light 界；第 12.7.5 节列出了测量它实际在哪里等待的计数器。

---

## 12.6 测得了什么

### 12.6.1 性能：五个交错的配对 [M]

`trace_out_v2/baseline.txt` 和 `timing_summary.txt`（原始示例未改动的开关关闭二进制文件的五次运行与 explicit 二进制文件的五次运行交错进行，同一天、同一块 GPU，`nvidia-smi` 每 100 ms 采样一次时钟）：

| 运行（交错顺序） | 未改动版 `Avg runtime` (ms) | explicit 版 `Avg runtime` (ms) | explicit / 未改动版 |
|---|---|---|---|
| 1 | 0.888653 | 0.889261 | 1.00068 |
| 2 | 0.890288 | 0.890307 | 1.00002 |
| 3 | 0.889411 | 0.888947 | 0.99948 |
| 4 | 0.889990 | 0.889517 | 0.99947 |
| 5 | 0.889917 | 0.889354 | 0.99937 |
| 中位数 | **0.889917** | **0.889354** | **0.99937** |
| 均值 / 标准差 | 0.889652 / 0.000641 | 0.889477 / 0.000508 | 0.99980 |
| 最小 / 最大 | 0.888653 / 0.890288 | 0.888947 / 0.890307 | |

- 按中位数计的 0.06 % 优势落在各次运行之间的离散范围之内（(max - min) / median：explicit 二进制文件为 0.15 %，未改动的那个为 0.18 %）。它除了“无法区分”之外不说明任何问题。
- `GFLOPS` 恰好就是 `2 m n k / Avg runtime`（`Options::gflops`，70_blackwell_fp16_gemm.cu:256-262）：2 x 8192^3 = 1.0995 x 10^12 FLOP；/ 0.889354 ms = 1.2363 x 10^15 FLOP/s，打印为 `1.2363e+06`。
- 纯 kernel 时间需要 `nsys`（笔记，第 12 节第 2 项），并且没有测量。
- 时钟：101 个稳态样本中的 101 个处于 1965 MHz SM / 3996 MHz 内存；在这 101 个样本上功率为 196.5-306.1 W（中位数 300.0 W）；全部 108 个样本上温度为 31-37 C。更早的七个样本（三个在 120 MHz，四个在 787 MHz）是第一个 kernel 之前的空闲爬升段；笔记引用的 188.6 W 下限和 299.9 W 中位数把它们包含在内（我对 `clocks.csv` 的重新计数）。那 9 ms 的计时循环本身运行在 1965 MHz，这一点是从周围的样本推断出来的（笔记 G.2 C1，D.4）。

### 12.6.2 正确性 [M]

| 证据 | 来自何处 | 证明了什么 |
|---|---|---|
| 1 次开关开启 + 5 次开关关闭运行中的 `Disposition: Passed` | `verify()` 用相同的 alpha/beta/C 运行设备端参考 GEMM，并用 `cutlass::reference::device::BlockCompareEqual` 比较全部 M x N 个输出 (70_blackwell_fp16_gemm.cu:351-378)：**精确**相等，无容差 | D 与参考结果在值上相等 |
| 哨兵填充 | `trace_begin()` 在被追踪的那次启动之前用 `0xFF` 字节填充 D（explicit.cu:1349：`cudaMemset(d_ptr, 0xFF, d_bytes)`），所以任何从未被存储的元素读出来都是 NaN，并通不过 `==` | 2^26 个元素中的每一个都被 kernel 写入了；被调度 bug 跳过的 tile 不可能通过 |
| `check_trace.py`：4116 PASS，0 FAIL（`trace_out_v2/checklist.txt`，于 2026-10-16 用已提交的检查器在记录下来的 `launch0.csv` 和 `host.txt` 上重新生成；2026-09-11 的文件写的是来自第 1-7 节的 4112 PASS，那是在第 8 节的四个 G.6 自检行加入之前） | 检查器的第 1-8 节：共享内存地址、TMEM 基址、每个 MMA 操作数、每个 TMA 加载/存储操作数、CLC 槽位/parity/覆盖、主机端事实、explicit kernel 的尾段记录（CUTLASS 运行的 4105 项检查加上七个 E 部分的行，笔记 G.1）。 | 笔记中预测的每个地址、偏移、描述符、坐标、槽位、parity 和计数都与 kernel 实际所做的相符 |
| 1518 条 `K_TAIL` 记录 = 每 cluster 46 条 x 33 | rank 0：8 条 mainloop + 2 条 CLC + 4 条累加器 + 1 条 dealloc；rank 1：8 + 1；rank 2：8 + 4 + 1；rank 3：8 + 1（`check_trace.py` 的 `TAIL_EXPECT`） | 第 4 章的全部四个退出尾段在每个 CTA 中都运行到了结束，其 `{index, phase}` 与流水线公式所预测的一致（mainloop 尾段全都等待 parity 1；累加器为 `((T + i) % 4, 1 ^ (((T + i) / 4) & 1))`；CLC 为 `((T + i) % 2, 1 ^ (((T + i) / 2) & 1))`，对于有 T 个 tile 的 cluster） |
| 没有 `TRACE_HANG` / `TRACE_HANG_HB` 行 | trace 构建的 4 s 有界等待和 20 s 看门狗从未触发 | 没有任何屏障等待阻塞达 4 s 或更久，因此没有等待停在一个从未完成的 phase 上；过早通过的等待（另一个 parity 指名的是紧邻的前一个、已经完成的 phase，并立即返回 true，PTX ISA `mbarrier.try_wait.parity`；过小的到达计数会使 phase 提前完成）并不会挂起，而上面的尾段记录和精确相等的结果（而不是这一行）才是反驳它的证据 |

### 12.6.3 CLC 动态 [M，推断处已标注]

**驻留。** 33 个 cluster 被原生启动（启动线性 id 0..32），等于此配置下的 `cudaOccupancyMaxActiveClusters`（`host.txt` 中的 `max_active_clusters 33`）。148 个 SM 中有 132 个承载着 CTA；SM 64、65、122-129 和 142-147 在整个启动期间保持空闲（我对 `smid` 列的重新计数与笔记 D.3 一致）。朴素的 148 / 4 = 37 并未达到。

为什么是 33（笔记 D.3；第 2.1.1 节补充了 GPC skyline，NVIDIA 内部信息）：每个 MMA pair 位于一个 SM pair `(2j, 2j + 1)` 上，即一个 TPC，而一个 cluster 的两个 pair 位于同一个 SM-id 组中：

| Cluster id | pair {0, 1} 所在的 SM | pair {2, 3} 所在的 SM |
|---|---|---|
| 0-7 | `2k`, `2k + 1` | `+16` |
| 8-15 | `2k + 16`, `2k + 17` | `+16` |
| 16-22 | `66 + 2(k - 16)`, `+1` | `+14` |
| 23-29 | `94 + 2(k - 23)`, `+1` | `+14` |
| 30-31 | `130 + 2(k - 30)`, `+1` | `+6` |
| 32 | 134, 135 | 140, 141 |

这一推断（笔记 D.3）：一个 2 x 2 cluster 需要同一个 GPC 内的两个完整 TPC，上面的各组拥有偶数个可用的 TPC，而剩下的 8 个 TPC（16 个 SM）无法配对，因此是 33 而不是 37。第 2 章，第 2.1.1 节用 GPC skyline，即每个 GPC 启用的 TPC 数（NVIDIA 内部）把它补全：每个奇数大小的 GPC 都会空闲一个 TPC，而这块 B200 的 skyline，10/10/10/9/9/9/9/5/1/1/1（NVIDIA 内部），有八个奇数大小的 GPC，恰好空闲 8 个 TPC = 16 个 SM，即实测的计数，而另一种 B200 skyline，10/10/9/9/9/9/9/9（8 个 GPC；同样是 NVIDIA 内部），有六个奇数大小的 GPC，则不会如此（算术在那一节，并附有关于 SM-id 组和 GPC 归属的两条注意事项）。对设计者而言：cluster shape 在第一条指令运行之前就花掉了机器的 10.8 %，而这一损失取决于芯片的 skyline，而不取决于 kernel。一个 (2,1,1) cluster 或一个不带 cluster 的 kernel 会填满更多的 SM，但会失去 A 的 multicast；第 3 章权衡了这一点。

**每个 cluster 的 tile 数。** 13 个 cluster 各 32 个 tile，19 个各 31 个，而 cluster (0, 0)，即带探针的 cluster，为 19 个，它在 trace 构建中被其 K3/K4 探针拖慢到每个 tile 59.7 us（归因在笔记中为推断；第 3.8.1 节），是其他 cluster 的 35.46 us 的 1.68x（`trace_out_v2/launch0.csv` 中 scheduler warp 的 K5a 查询发出间隔的中位数，一次重新计数；在 CUTLASS kernel 的 trace 中为 1.70x 和 54.35 us，笔记 D.3）；总和 13 x 32 + 19 x 31 + 19 = 416 + 589 + 19 = 1024（我的重新计数）。动态调度自动吸收了探针引起的变慢：慢的 cluster 只是收到了更少的 tile。[H] Release 二进制文件中没有带探针的 cluster。1024 个 tile 分给 33 个 cluster（33 x 31 = 1023，多出一个 tile），预期的分配是每个 cluster 31 或 32 个，更快的 cluster 拿走多出的 tile；一次均衡的分发会迫使一个 cluster 拿 32 个 tile、三十二个拿 31 个（32 + 32 x 31 = 1024；推导所得，非实测）。Release 运行只记录 `Disposition`、`Problem Size`、`Avg runtime` 和 `GFLOPS`，所以这一分配没有被测量。在该预期之下，尾段不均衡至多是 31 个中的一个 tile，约为运行时间的 3 %：这是 256 x 256 的 cluster tile 落在 33 个 cluster 上的代价。

**分发顺序**（CUTLASS kernel 的 trace，笔记 D.3）。991 个被取消的 id 是 33..1023，每个恰好一次。在每个 cluster 内部，相继的 id 严格递增。按查询发出时间排序，这些 id 是升序的，只有 990 对中的 25 次相邻交换例外（991 个中有 933 个处于其排序后的位置）。没有任何 id 错位超过 4。25 这个计数在同一 tick 的平局按 id 升序打破时成立：991 个发出时间戳落在 950 个不同的 32 ns `%globaltimer` tick 上，而其他平局排序方式给出 32-62 次倒置。第一次分发是把 id 33 分给 cluster (0, 0)，在第一次查询之后 1.28 us；最后一次是 id 1023，在 959.2 us 时。在 `AlongN` 且没有 swizzle 的情况下，id L 是 cluster tile `(L / 32, L % 32)`，所以一个 cluster 相继的 tile 之间相隔约 33 个 id：`m` 每个 tile 增加一，而 `n` 漂移。在任一时刻，这 33 个 cluster 工作在 A 的两个（991 次分发中的 834 次）或三个（157 次）相邻的 256 行条带上。这是从启动顺序免费获得的良好 L2 行为。

**查询延迟**（从 scheduler 的被选中的 lane 发出，到同一 warp 消费）：

| Trace | 最小 | p10 | 中位数 | p90 | 最大 |
|---|---|---|---|---|---|
| CUTLASS kernel（笔记 D.3，1024 次查询） | 736 ns | 800 | 1408 | 1536 | 2336 |
| explicit kernel（我的重新计数，1024 次查询） | 928 ns | 960 | 1024 | 1184 | 1408 |

CUTLASS trace 的 p75 是 1472 ns，其 p99 是 1792 ns；该分布按 cluster 放置呈双峰（14 个 cluster 为 768-928 ns，19 个为 1408-1504 ns；这一划分在放置表每个组内部的相同 SM 偏移处重复出现，而笔记 D.3 推断原因是在 GPC 内部的位置）；33 个未取消的响应并不更慢（中位数 1376，最大 1504 ns）；并且 rank 0 的 MMA warp 的消费记录 K5c 比 scheduler warp 的 K5b 晚 96 ns（中位数），这是关于 MMA warp 在 `clc_full` 上的等待的唯一实测数据，而且是间接的：两条记录都跟在各自 warp 的等待之后，所以这 96 ns 标定的是 MMA warp 的消费相对于 scheduler 的消费的时刻，而不是等待的持续时间（均为笔记 D.3；第 7 章，第 7.6.9 节在其基础上展开）。两行之间的差别在于 scheduler warp 中探针的放置位置，而不是硬件（笔记 G.2 的解释，"Trace build, for the record only"）。无论哪种情况，这一延迟都无关紧要。在 CUTLASS kernel 的 trace 中，从消费响应 n 到发出查询 n + 1 的 throttle 间隔为 28.8-37.7 us（32 个不带探针的 cluster 上的中位数 30.8 us；cluster (0, 0) 被排除，笔记 D.3）；按其 1.41 us 的中位延迟，一次查询处于在途状态的时间占 4.6 %。在 explicit kernel 的 trace 中，该间隔为 33.7-41.0 us（32 个不带探针的 cluster 的 973 个消费到发出对上的中位数 34.4 us，我的重新计数），对比 1.02 us 的中位数：约 3 %。每个 tile 的时间在 CUTLASS kernel 的 trace 中的中位数为 31.90 us（笔记 D.3；trace 构建在探针开启且 108/72 个寄存器下运行），在 explicit kernel 的 trace 中为 35.46 us（笔记 G.2，它丢弃每个 cluster 的第一个间隔，并在此基础上给出 CUTLASS trace 的 31.84 us；explicit 一侧的数字在两种基础上都是 35.46 us，一次重新计数），对比 Release 构建中的 0.889 ms / 32 个 tile = 每个 tile 27.8 us（至多 27.75 us：笔记 D.4 的 0.888 ms 除以一个关键路径 cluster 的 32 个 tile；explicit kernel 的 0.889354 ms 给出 27.8 us）。每个 cluster 从未有超过一次查询处于未决状态，GPU 范围内至多 25 次（笔记 D.3）。

设计者得到三点启示。CLC 往返约为 1 us，所以每个 tile 一次查询（约 28 us）就把它完全隐藏，而一条深于 2 个槽位的 CLC 流水线不会带来任何好处。throttle（第 7 章）正是让“每个 cluster 一次未决查询”成立的东西。而 1024 个 tile 的恰好一次覆盖必须由检查器来验证，因为不存在针对它的运行时断言。

---

## 12.7 如何复现与检视

### 12.7.1 全速运行（笔记第 0 节，H.1 部分）

```
./build/examples/70_blackwell_gemm/70_blackwell_fp16_gemm --m=8192 --n=8192 --k=8192
```

预期：`Disposition: Passed`、`Problem Size: 8192x8192x8192`、`Avg runtime: 0.889 ms`（在 1965 MHz 的 B200 上为 0.888-0.891）、`GFLOPS: 1.236e+06`（五次 explicit 运行中为 1.235e+06-1.237e+06；harness 以六位有效数字打印这两个值，例如 `Avg runtime: 0.889354 ms` 和 `GFLOPS: 1.2363e+06`）。

### 12.7.2 自己转储二进制文件（笔记 H.2.2）

```
BIN=./build/examples/70_blackwell_gemm/70_blackwell_fp16_gemm
cuobjdump --dump-sass "$BIN" > sass.txt
cuobjdump --dump-ptx  "$BIN" > ptx.txt
cuobjdump --dump-resource-usage "$BIN"
cuobjdump --list-elf "$BIN"            # two sm_100a cubins expected, one per translation unit
awk '/^Fatbin /{p=0} /Function :/{p=index($0, "explicit_blackwell_fp16_gemm_kernel") > 0} p' sass.txt > sass_explicit.txt
awk '/^Fatbin /{p=0} /^\.visible \.entry|^\.entry/{p=index($0, "explicit_blackwell_fp16_gemm_kernel") > 0} p' ptx.txt > ptx_explicit.txt
```

`--dump-sass` "dumps CUDA assembly for a single cubin file or all cubin files embedded in the binary"，`--dump-resource-usage` "dumps resource usage for each ELF"，`--list-elf` "lists all the ELF files available in the fatbin"（CUDA Binary Utilities 手册）。这段 `awk` 切分之所以奏效，是因为 `cuobjdump` 为每个 kernel 打印一个 `Function : <mangled name>` 头，并为每个嵌入的镜像打印一个 `Fatbin ...` 头。

### 12.7.3 检查清单：要 grep 什么 [M]

你应当看到的已执行路径清单，附有我重新计数的数目：

| 族 | explicit kernel 中的预期值 | 说明 |
|---|---|---|
| `UTCHMMA.2CTA` | 4 | 四个 k-block |
| `UTCBAR.2CTA.MULTICAST` | 2 | mainloop 释放、累加器 commit |
| `UTMALDG.3D.MULTICAST.2CTA` / `UTMALDG.3D.2CTA` / 普通的 `UTMALDG.3D` | 2 / 2 / 0 | 每个未展开的 producer 循环中一个 A 和一个 B；没有 C 加载 |
| `UTMASTG.3D` / `UTMACMDFLUSH` / `DEPBAR.LE SB0, 0x1` / `DEPBAR.LE SB0, 0x0` | 32 / 8 / 8 / 0 | 4 个 box x 8 个 subtile；没有 `store_tail` |
| `LDTM.x16` | 8 | |
| `UGETNEXTWORKID.BROADCAST` | 1 | CLC 查询（grep 这个，而不是 "CLC"） |
| `ACQBULK` / `PREEXIT` | 2 / 1 | GDC |
| `UTMACCTL.PF` | 4 | A、B、C、D 预取 |
| `UTCATOMSWS.2CTA.FIND_AND_SET` / `UTCATOMSWS.AND` | 2 / 1 | alloc（+ 重试副本）、dealloc |
| `SYNCS.EXCH.64` | 33 | 这 33 个 `mbarrier.init` |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` / 不带 `.TRYWAIT` 的 `SYNCS.PHASECHK.TRANS64` / `NANOSLEEP` | 65 / 32 / 31 | ptxas 的重试结构；不是契约 |
| `SYNCS.ARRIVE*` | 14 | 9 条远程 `.RED.A1T0`、2 条 `.RED`（远程 expect_tx 与 alloc 握手）、1 条 `.A1T0`、2 条普通的（本地 expect_tx） |
| `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` / `0x6, 0xa0` / `BAR.ARV 0x6, 0xa0` | 16 / 1 / 1 | 外加一条 `BAR.SYNC.DEFER_BLOCKING 0x0`，即非 cluster 回退 |
| `FMUL` / `FFMA` / `FADD` | 130 / 128 / 0 | 128 + `%cluster_ctaid` 下沉的那两条（归因为推断，笔记 D.2） |
| `STS` / `LDS` | 131 / 11 | 128 + 3 条簿记；4 条 `LDS.128` + 7 次保留区/tmem_base 读取 |
| `LDL` / `STL` / `MOV.SPILL` / `R2UR.FILL` | 0 / 0 / 0 / 0 | |
| `ELECT` | 4 | 各 `UTCHMMA` 之间没有 |
| `0xfefffff8` / `UPRMT ..., 0x5410` / `PRMT` 或 `UPRMT ..., 0x654` | 1 / 2 / 11 | peer 掩码、掩码插入、`mapa` 下沉 |
| `UMOV URn, 0x40004040` | 8 | 描述符的高位字，全部在 MMA 循环中 |
| 资源 | `REG` 26 (<= 68)、`STACK` 0、`LOCAL` 0、`SHARED` 1024、`CONSTANT[0]` 1536 | |

### 12.7.4 读一行 SASS [M，操作数块的解读为 H]

```
        /*5220*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4], R2 ;
        ^addr   ^predicate slot (empty = always; "@!P0" = if not P0; "@UP2" = uniform predicate)
                                   ^mnemonic.modifiers           ^dst   ^address  ^source
```

- `SYNCS.*`：屏障单元。`TRANS64` = 该指令作用于共享内存中的一个 64 位事务屏障状态，它跟踪一个到达计数和一个事务计数，并且只有在两者都满足时才推进其 phase（64 是屏障状态的宽度，不是线程数）；`PHASECHK.TRANS64` = 测试该 phase；`.TRYWAIT` = 对同一 phase 的初始非阻塞探测，不等价于裸形式：未命中时，标准序列用 `NANOSLEEP` 退避，然后在跳转回去之前用裸的 `PHASECHK.TRANS64` 作为重试路径上的阻塞检查（NVIDIA 内部；实测的循环在第 12.2.2 节，其 `NANOSLEEP.SYNCS 0x989680` 立即数就是 PTX `try_wait` 的 `timeHint` 值）；`EXCH.64` = 原子交换（读出旧字、写入所提供的字、返回旧字；uniform 形式使用 uniform 寄存器寻址和数据），即 `mbarrier.init` 的下沉；`ARRIVE.TRANS64[.RED][.A1T0|.A0TR]` = arrive 或事务计数更新（`.RED` = 归约式 arrive，对远程 peer-CTA 目标是必需的，在本地也有效，其目的写回被抑制；`A1T0` = 到达 +1，事务增量 0；`A0TR` = 无到达，事务计数由源寄存器 `Rb` 更新（NVIDIA 内部）；实测：baseline 的 8 处裸 `mbarrier.expect_tx` 位点下沉为 `.RED.A0TR`，字节数放在源寄存器中（`R3 = 0x2000`；在 beta = 0 时不执行，笔记 D.2；explicit kernel 中没有），在 epilogue 的 TMA 拷贝之后发射，所以字节可能先落地（第 2.8.2 节和第 14.2.4 节）；该更新如何改变事务计数（其符号，以及它是 expect 一侧还是完成一侧的更新）尚未解决，因为没有可用的佐证材料；不带任何一个后缀时，源寄存器携带事务计数，如同带 `R5 = 0xc000` 的 `expect_tx`）（这些后缀定义是没有任何公开文档给出的 NVIDIA 内部信息；对不带后缀时源寄存器的解读是从笔记 D.2 的 PTX 到 SASS 映射推断出来的 [H]）。
- `UTC*`：Tensor Core（TC）pipe，Profiling Guide 说它 "is different from the Tensor pipeline"（这两个 pipe 的指标：第 12.7.5 节）：`UTCHMMA` = MMA，`UTCBAR` = commit 操作，`UTCATOMSWS` = TMEM 分配器原子操作。Nsight Compute Profiling Guide："The TC pipeline executes UTCBAR, UTCCP, UTC*MMA, UTCSHIFT and UTC*SWS instructions"；`UTCCP`（`tcgen05.cp`）和 `UTCSHIFT`（`tcgen05.shift`）不出现在这个 kernel 中。
- `UTMA*`：TMA 单元：`UTMALDG` 从全局到共享的加载，`UTMASTG` 从共享到全局的存储，`UTMACMDFLUSH` commit group，`UTMACCTL.PF` 描述符预取。第一个方括号中的操作数命名一块连续 uniform 寄存器中的第一个寄存器（加载为 `{dst, mbarrier, c0, c1, c2}`，存储为 `{src, c0, c1, c2}`）；第二个方括号是 64 位 tensormap 指针；`desc[URn]` 是 64 位缓存提示描述符 [H]。
- `LDTM.x16 R4, tmem[UR29+0x70]`：一次 TMEM 加载，每个 lane 加载 16 x 32 位，来自 `UR29` 中的 lane 组和列再加上立即数列偏移。
- `UR`、`UP`、`URZ`、`UPT`、`RZ`、`PT`：uniform 寄存器、uniform 谓词；`RZ`/`URZ` 是每 lane（`Rn`）与 uniform（`URn`）寄存器堆的常量零（`URZ` 是一个单独的寄存器，不是 Blackwell 的 80 个可寻址 `UR0`-`UR79` 之一），`PT`/`UPT` 是每 lane（`Pn`）与 uniform（`UPn`）谓词寄存器堆的恒真谓词，而这四个中的任何一个用作目的操作数时都会丢弃写入（NVIDIA 内部）。CUDA Binary Utilities 手册命名了 `RX`/`URX`/`PX`/`UPX` 寄存器类，但没有定义这四个。
- 尾随的 `/* 0x... */` 注释（上面已去掉）是 128 位编码，包括第 12.1.3 节的控制字段。那些字段就是 `tcgen05.wait::ld` 去往的地方。解码它们需要第三方工具，而本课程不需要这样做。

### 12.7.5 将剩余差距归因到 speed of light [指标定义为 S，数字为 H]

实测的 1.2363 x 10^15 FLOP/s 按 148 个 SM、1965 MHz 计是每周期每 SM 4.25 kFLOP，按 132 个活跃 SM 计则是 4.77 k（运行时间来自笔记 G.2，132 个活跃 SM 来自笔记 D.3；笔记 D.4 为其 1.237e15 推导出了同样的两个数字）。厂商给出的 1000 W B200 稠密 FP16/BF16 Tensor Core 数字是 2.25 PFLOPS（面向 HGX B200 180 GB 1000 W GPU 的 Lenovo ThinkSystem 产品指南，"2.25 / 4.5 petaFLOPS"，不含/含稀疏性）。比值是 55 %，但数据手册的条件并不是本次运行的条件：在 1965 MHz 下实测功耗约为 300 W，而数据手册的时钟未知。如果 2.25 PFLOPS 对应于 8192 稠密 FP16 FLOP/cycle/SM，那么隐含的时钟是 1.856 GHz，一次 256 x 128 x 16 的 pair-MMA 将耗时 64 个周期，相对于由实测 0.888 ms 运行时间推导出的 106.5 周期上界（笔记 D.4；第 12.5.5 节）[H]。第 3 章从第一性原理推导出这个界；本节给你能把它敲定的 `ncu` 指标。

先说三点实际事项。(a) `ncu` 中的 "Speed Of Light" 是 `GPU Speed Of Light Throughput` 节："For each unit, the throughput reports the achieved percentage of utilization with respect to the theoretical maximum"（Profiling Guide），即最忙单元占其峰值持续速率的份额，也就是第 3 章以分析方式运用的同一个概念。(b) `--launch-skip 1 --launch-count 1` 只计入与 `--kernel-name` 匹配的启动（"The number takes into account only launches that match the kernel filters"；"The count is only incremented for launches that match the kernel filters"，Nsight Compute CLI 手册）。因此 `ncu` 跳过 explicit kernel 的第 0 次启动，即 harness 的预热，并剖析它的第一次计时启动；它不计入参考 GEMM 和比较 kernel。(c) `ncu` 在剖析时锁定 GPU 时钟（"Nsight Compute locks SM clocks by default"，Profiling Guide，Range and Precision，"Workload Durations"）。`--clock-control base` 把 GPC 与内存时钟锁定到基础频率；当前的 CLI 手册（v2026.3.1）把 `boost` 记载为默认值，在不支持 boost 锁定之处回退到 `base`（较旧的版本把 `base` 记载为默认值；此处未验证）。无论哪种情况，时钟都可能不同于计时运行的 1965 MHz。因此，基于周期的数字不会与第 12.5.5 节的 426 周期预算对上，除非你传入 `--clock-control none`，并用 `sm__cycles_elapsed` 对照 `gpu__time_duration` 确认时钟；对于与未锁频测量的比较，该指南的建议是用 `nvidia-smi` 在外部锁定时钟并使用 `--clock-control none`。剖析一次计时启动：

```
ncu --kernel-name regex:explicit_blackwell --launch-skip 1 --launch-count 1 \
    --section SpeedOfLight --section ComputeWorkloadAnalysis --section WarpStateStats --section MemoryWorkloadAnalysis \
    ./build/examples/70_blackwell_gemm/70_blackwell_fp16_gemm --m=8192 --n=8192 --k=8192
```

然后按以下顺序读取：

| 问题 | 指标或节 | 设计者得出的结论 |
|---|---|---|
| tensor pipe 忙吗？ | `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`（更宽泛的 tcgen05 TC pipe：MMA 运算加上 `UTCCP`、`UTCBAR` 和 TMEM 分配原子操作，NVIDIA 内部）以及与之不同的 `sm__pipe_tensor_cycles_active`（UMMA 运算，即 tensor pipe，NVIDIA 内部）；在其他 `sm__pipe_tensor_*` 指标中，`sm__pipe_tensor_op_hmma_cycles_active` 是一个应为利用率而请求的 tensor-pipe 活跃指标，其本身并不是 `UTCHMMA` 使其递增的证明（NVIDIA 内部）；操作码/模式的证明是以 `...op_utchmma_utcqmma_utcomma_scope_{1cta,2cta}.sum` 结尾的那个计数器（NVIDIA 内部；对本 kernel 的 `cta_group::2` 应请求的是 `2cta` 形式，我的解读；被省略的前缀请从所安装版本的 `ncu --query-metrics` 获取）；关于 TC/tensor 的区分，NVIDIA 工作人员在开发者论坛（thread 355164）上表示 "A tcgen05 instruction will result in both sm__pipe_tc_cycles_active (singleton per SM) and sm[sp]__pipe_tensor_cycles_active (per SMSP) counters being updated"，并表示 "The TC unit is also responsible for some of the other tcgen05 (not .mma) instructions so activity of TC could exceed Tensor Cycles Active"；Profiling Guide 本身只列出了 TC pipe 的指令，并说它 "is different from the Tensor pipeline" | 如果 tensor-pipe 指标在活跃 SM 上接近 100 %，那么该 kernel 在其时钟下就处于 tensor-core speed of light，剩下的唯一杠杆就是那 16 个空闲 SM 和尾段 |
| 那 16 个空闲 SM 看得见吗？ | `sm__cycles_active.avg` vs `.max` 以及 `sm__cycles_elapsed`；`LaunchStats` 节（cluster 大小、wave） | 平均活跃周期与最大活跃周期之间 10.8 % 的差距是 cluster 打包损失，而不是 kernel 的 bug |
| warp 在哪里等待？ | `WarpStateStats`：`smsp__average_warp_latency_issue_stalled_*`（barrier、long scoreboard、wait、membar、...；在一个经检视的 SM100 warp 特化 GEMM 中，mbarrier `TRYWAIT` 的等待路径被采样为 `long scoreboard`（该指标族的 `_long_scoreboard` 后缀），这是对等待路径的归因，而非 `TRYWAIT` 本身就是原因的证明，NVIDIA 内部，第 3.5.1 节；`NANOSLEEP` 重试则记为 `sleeping`，Profiling Guide） | leader warp 0 在 `full[s]` 屏障上停顿，意味着 TMA 来迟了（受 L2/DRAM 限制或 multicast 不均衡）；epilogue warp 在 `acc_full` 上停顿，意味着 mainloop 是限制因素，正如设计的那样 |
| 发射速率是否接近极限？ | `smsp__issue_active.avg.pct_of_peak_sustained_active`、`smsp__inst_executed.sum` | 第 12.5.5 节预测在最忙的子分区上低于 15 % |
| 流量有多少？ | `lts__t_bytes.sum`（按 Profiling Guide，是 L2 切片 tag-stage 字节；不是 DRAM 字节，也不一定是有效载荷字节，NVIDIA 内部信息；第 3.9.4 节）、TMA 字节族 `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_tma*`（一个 L2 tag-stage 请求/扇区指标：其预期值是 L2 侧流量，在 multicast 下它随唯一请求流量一起下降；请求所安装版本提供的 `.sum` 形式；NVIDIA 内部）、`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum`（交付进共享内存的字节数，不因 multicast 而改变，因为每个接收 CTA 仍然得到一个完整的 tile，所以不要从这个聚合值推断 multicast；要看 multicast 的拆分，请比较 `..._dest_multicast.sum` 和 `..._dest_self.sum` 计数器，并读取 `lts__t_sectors_srcunit_tex_op_read.sum` 来看 L2 读压力的降低；NVIDIA 内部；第 3.9.3 节）、`smsp__inst_executed_op_tma_ld.sum`（TMA 加载指令计数，不是字节数，NVIDIA 内部；这里每个 CTA 每个 k-tile 2 条，W21 和 W22）、`dram__bytes.sum`、`dram__throughput.avg.pct_of_peak_sustained_elapsed` | 操作数需要 8 GiB 的 L2 到 SM 流量（1024 个 cluster tile x 128 个 k-tile x 64 KiB：每个 cluster 每个 k-tile 由 multicast 交付的 32 KiB 的 A 以及 32 KiB 的 B），在 0.889 ms 内 = 来自 L2 约 9.7 TB/s [H]（`lts__t` TMA 族预期跟踪的是那 8 GiB 的 L2 侧流量，而落地字节指标 `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` 预期读数为 12 GiB，即 4096 个 CTA x 128 个 k-tile x 24576 B，第 3.9.3 节，无论有无 multicast；NVIDIA 内部），但如果 L2 复用是完美的，从/到 DRAM 的就只有 128 + 128 MiB 唯一的 A 和 B 加上 256 MiB 的 D（0.6 TB/s 相对于 7.7 TB/s 的 HBM3e）；实测的 `dram__bytes` 告诉你离完美有多远 |
| 不含主机端编码的纯 kernel 时间 | 对同一条命令运行 `nsys profile --stats=true`，或 `ncu` 中的 `gpu__time_duration.sum` | 把每次迭代的 23 次主机端 API 调用与 kernel 分开 |

除引自 Profiling Guide 或作为 NVIDIA 内部信息提供的那些指标名（`lts__t_bytes.sum`、TMA 字节族 `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_tma*`（其可用的 `.sum` 形式必须在所安装的 Nsight Compute 上请求）、`smsp__inst_executed_op_tma_ld.sum`、`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` 以及 `..._dest_multicast.sum` 和 `..._dest_self.sum` 计数器、`lts__t_sectors_srcunit_tex_op_read.sum`、`sm__pipe_tensor_op_hmma_cycles_active`、`sm__pipe_tensor_cycles_active`、以 `...op_utchmma_utcqmma_utcomma_scope_{1cta,2cta}.sum` 结尾的操作码/模式计数器（前缀已省略；完整名称请从 `ncu --query-metrics` 获取），以及第 3.9.3 节的 bank 冲突与共享内存 wavefront 计数器）之外，其余指标名遵循其 `unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)` 约定，并应在所安装的版本上用 `ncu --query-metrics` 核对（对 CUDA 13.3 的 Nsight Compute 未验证）。

---

## 12.8 经验：“性能等价”要求了什么，以及哪些是自由的

### 12.8.1 必需项（结构契约）[S, M]

| 要求 | explicit kernel 在何处做到 | CUTLASS 在何处做到 | 为什么重要 |
|---|---|---|---|
| producer 与 MMA warp 中保持不展开的 k-tile 循环 | explicit.cu:733 和 :921 处的 `#pragma unroll 1` | sm100_mma_warpspecialized.hpp:607 和 :680 处的 `CUTLASS_PRAGMA_NO_UNROLL` | 完全展开是不可能的（运行时迭代次数；第 13.3 节，陷阱 18），而部分展开只会加长直线循环体，极限情况下趋向 128 x 47 条指令，很可能带来指令缓存未命中（我的分析：指令缓存的大小没有文档记载，也没有实测过任何展开的变体），且没有收益，因为该循环并不受发射限制（第 12.5.5 节） |
| 展开的 k-block | explicit.cu:930 处的 `#pragma unroll` | :699 处的 `CUTLASS_PRAGMA_UNROLL` | 让 ptxas 把 k-block 1..3 的 `scale_c` 硬连线为 `UPT`，并在一段直线代码中发射这四条 `UTCHMMA`（0x5410-0x5470，中间只隔着三条 `UMOV`） |
| 完全展开的 epilogue subtile | explicit.cu:1032 处的 `#pragma unroll` | sm100_epilogue_tma_warpspecialized.hpp:818 处的 `#pragma unroll(UnrollEpiLoop ? NumEpiSubtilesN : 1)` | `16 e` 项变成 `LDTM` 的立即数，`8192 (e & 3)` 变成 `STS` 的立即数，存储坐标变成编译期偏移；那唯一的一条 `tcgen05.wait::ld` 和 release 按构造落在 subtile 7 中 |
| warp-uniform 的发射操作数 | 每个 TMA/MMA/commit/query 操作数都来自 rank、tile、stage 和常量 | 相同 | 否则就没有 `U*` 下沉，循环中会有向量地址算术和 `R2UR` 流量（第 12.3 节） |
| 相同的屏障协议：计数、parity、掩码、peer 位寻址、先 peek 再 wait、四个退出尾段 | 第 4 章、第 7-11 章 | 第 4 章中内联引用的 CUTLASS 头文件 | 协议**就是**流水线；不同的计数或多出一次 wait 都会改变重叠，而缺少一个尾段就是一个退出后使用的冒险 |
| 运行时的 alpha 和 beta，`t = alpha * acc; d = fmaf(beta, 0, t)` | explicit.cu:1045-1046，`ld.param +528` | fusion 树的 `FMUL`/`FFMA` | 逐位相同的 D，包括零的符号；常量 1.0f/0.0f 会把这一对折叠掉，最多只剩一条 `FADD`（笔记 D.7，推断） |
| TMA 发射顺序 wait -> expect_tx -> peek -> A -> B；MMA 顺序 wait -> 4 条 MMA -> commit；存储顺序 fence -> bar.sync -> 4 次存储 -> commit_group -> wait_group.read 1 -> bar.sync | 第 12.5.1-12.5.4 节 | 相同 | 正是这个顺序隐藏了延迟：peek 在各次加载之前发射，使得下一次迭代的 wait 通常已被满足 |

### 12.8.2 可自由更改（且已更改）[M]

| 更改项 | Explicit | CUTLASS | 后果 |
|---|---|---|---|
| 向量寄存器 | 26 | 68 | 无（1 CTA/SM） |
| 参数块 | 640 B，`CONSTANT[0]` 1536 | 2048 B，2944 | 无 |
| 角色分发 | `BRXU` 跳转表（`CONSTANT[2]:12`） | 比较并分支的链 | 每线程一次；同一开销等级 |
| 屏障初始化 | 33（省略了 `epi_load`、`load_order`） | 43 | prologue 中少 10 条 `SYNCS.EXCH.64` |
| prologue 长度 | 到分发为止 175 条指令 | 359 | 一条 `SHFL.IDX` 而非 7 条，初始 tile 用四条 `ULOP3` 而非光栅化算术 |
| 死 C 路径 | 不存在 | 已编译但从不执行（32 条 `UTMALDG.3D`、128 条 `LDS`、8 条 `expect_tx`、`DEPBAR.LE SB0, 0x0`、第三条 `ACQBULK`） | SASS 1640 条指令 vs 3224 条 |
| 循环不变量的处理 | 一部分每 k-tile 重新物化（`UMOV`、掩码插入、`tma_b` 地址） | 外提，推断的代价（笔记 D.2、D.7）是别处 44 对 uniform 到向量的溢出/填回，热循环中没有 | 每 k-tile +3 / +6 条指令，不可见 |
| `k_tiles` | 保存在 `UR14` 中 | 每 tile 从 `c[0x0][0x38c]` 重新读取 | 累加器的 acquire/commit 块是 15 条指令 vs 18 条 |
| CLC 的 z 坐标 | 常量 0（按构造就是 L = 1 的 kernel；主机端守卫拒绝 `l != 1`） | 绑定到响应的 `L_idx` | L > 1 的 kernel 会把它恢复回来 |
| multicast 掩码 | 来自常量的 `0x5 << x` 和 `0xf` | 运行时由 cluster shape 计算 | 相同的值 |
| PTX 中 `elect.sync` 的数量 | 14 | 19 | 两者在 SASS 中都变成 4-5 条 `ELECT` |

---

# 第 13 章 自己动手写：构建顺序、陷阱、泛化与一份 SOL 配方

“实测”指笔记的 D 部分或 G 部分；“推导”指在所引行处从 explicit.cu 或 CUTLASS 源码中读出；“我的分析”指一项你应当在自己的机器上重新核对的推断。

---

## 13.1 一页纸的方法

第 3 到 12 章把这个 kernel 拆开了。本章按一位工程师能够构建它的顺序把它重新组装起来。顺序之所以重要，是因为 warp 特化的 cluster kernel 中的大多数错误恰好只有一种可观察的失效模式：什么都不发生。错误的到达计数、错误的 parity、缺失的 peer 位掩码、错误的 multicast 掩码和错误的尾段 parity，都可能表现为 "nothing printed ..., process alive at full utilisation"（笔记 E.7.2，"Hang" 行）。`printf` 无法对此做二分定位。只有两样东西可以：(a) 一份在编码之前写下的、记录每个屏障应当看到什么的书面记录，以及 (b) 分层构建 kernel，使每一层只增加很少几种新的挂起方式。

你最先写下的五件产物，正是笔记从 CUTLASS 中提取出来的那五件（笔记第 6.2、7.1、7.2、9 和 4 节），也是第 3 章、第 4 章和第 5 章所呈现的那五件：

| 产物 | 对于本 kernel | 本讲义在何处推导出它 |
|---|---|---|
| tile 表 | 8192^3；MMA 256 x 128 x 16 (`cta_group::2`)；CTA tile 128 x 128 x 64；cluster (2,2,1)；cluster tile 256 x 256；32 x 32 = 1024 个 cluster tile；网格 (64,64,1)；每个 tile 128 个 k-tile，每个 k-tile 4 条 MMA | 第 3 章 |
| smem 映射 | 控制区域 0..387（屏障 0..343，`clc_response` 352..383，`tmem_base_ptr` 384..387，然后填充到 512），C/D 512 + 4 x 8192，A 33792 + 8 x 16384，B 164864 + 8 x 8192，总计 230400 (explicit.cu:139-155) | 第 5 章 |
| 屏障表 | mainloop full 1 + 49152 tx / empty 2；CLC full 1 + 16 tx / empty 800；acc full 1 / empty 256；throttle 32 / 32；tmem_dealloc 32 (explicit.cu:180-192) | 第 4 章 |
| 角色表 | warp 0 MMA（leader 发射），warp 1 scheduler（仅 CTA 0），warp 2 producer，warp 3 退出（仅当 beta != 0 时为 C producer），warp 4-7 epilogue | 第 7-10 章 |
| 操作台账 | 每个 CTA tile 128 x (1 A + 1 B) 次 TMA 加载，每个 pair tile 512 条 MMA，32 条 `tcgen05.ld`，32 次 TMA 存储；总计 524288 + 524288 次加载，1048576 条 MMA，131072 次存储（笔记第 9 节） | 第 3 章 |

下文的一切都假定你已经把*你自己的* kernel 的这五张表写在了纸上。如果你重建的正是这个 kernel，就从所提到的那些章中把它们抄过来。

---

## 13.2 构建计划：从空白文件开始的八个步骤

每一步都说明要构建什么、要验证什么以及如何验证，还有它消除了哪一类 bug。这个顺序使每一步所引入的*新的*挂起方式的数量既少又已知。

### 步骤 1. 设计产物（无代码）

写出第 13.1 节的五张表。屏障表使用第 13.5 节步骤 6 的那些列：流水线、stage 数、producer（谁，用哪条指令）、consumer（谁，用哪条指令）、full 计数与事务字节数、empty 计数、parity 规则、*哪个 CTA 的屏障*、初始化 warp。人们会忘记*哪个 CTA 的屏障*这一列。在本 kernel 中，五条流水线里有四条的到达落在*另一个* CTA 的屏障上：mainloop full 落在 pair leader 的屏障上，acc empty 落在 pair leader 的屏障上，CLC empty 落在 CTA 0 的屏障上（而 CLC full 落在 CTA 1-3 的屏障上，由 CTA 0 装填，explicit.cu:842），`tmem_dealloc` 握手则在两个方向上都落在 pair 伙伴的屏障上（explicit.cu:964, 970；笔记第 7.2 节）；只有 throttle 的到达留在 CTA 0 内部。累加器的 *full* 屏障是每个 CTA 自己的（explicit.cu:902, 944）；第 13.5 节步骤 6 的表说明了为什么。

仅用算术来验证：smem 总量 230400 <= 232448（可选启用上限，util.hpp:46）；每个 SW128 stage 基址都是 1024 对齐的，每个 SW128_32B stage 基址都是 512 对齐的（explicit.cu:162-163）；描述符起始字段不会溢出（explicit.cu:165）；事务字节数等于*落在 pair 内*的那些 box 之和（2 x (16384 + 8192) = 49152，explicit.cu:180）；empty 计数等于不同到达方的数量（两个 leader 的 commit = 2；2 x 128 个 epilogue 线程 = 256；32 + 4 x (32 + 128 + 32) = 800）。

消除的 bug 类别：预算与对齐错误，以及“谁到达哪里”的混淆，否则它们会在步骤 4 中以挂起的形式浮现。

### 步骤 2. util/params/主机端启动，配一个空 kernel

写出 util.hpp（常量与各条 `static_assert`）、explicit.hpp（`ExplicitGemmParams` 块，先是四个 `CUtensorMap`，然后是 `m, n, k, l, alpha, beta`；explicit.hpp:49-63）以及主机端函数 `run()`（explicit.cu:1265-1329）。kernel 体只记录自己的身份：对每个 CTA，它把 `smem_u32(smem)`、`%cluster_ctarank`、`%cluster_ctaid.x/.y`、`%cluster_nctaid` 和 `blockIdx` 写入一个全局缓冲区，然后退出。

验证：

1. `cudaLaunchKernelExC` 在 cluster 属性为 `{2,2,1}`、`dynamicSmemBytes = 230400` 且 `numAttrs = 1` 的情况下返回 `cudaSuccess`（explicit.cu:1298-1313）。被拒绝的启动会通过 harness 打印 `Error Internal`（笔记 E.7.2，"Launch rejected" 行）。常见原因：缺少 `MaxDynamicSharedMemorySize` 的 opt-in（explicit.cu:1279）、网格不能被 cluster 整除（explicit.cu:1287-1290），或者 `numAttrs` 错误。
2. `cudaFuncGetAttributes` 报告 `sharedSizeBytes 0`（没有静态 `__shared__`）和 `maxDynamicSharedSizeBytes 230400`（在笔记 D.1 D4 中针对 CUTLASS kernel 实测）。
3. `cudaOccupancyMaxActiveClusters` 对此启动配置返回 33（实测，笔记 D.1 D4；explicit kernel 自己的 `H3` 探针报告了 `max_active_clusters 33`，笔记 G.1）。这就是你的 CLC scheduler 将会看到的原生启动的 cluster 数量，而不是 37 = 148 / 4。实测：132 个 SM 忙碌而 16 个空闲（SM id 为 64, 65, 122-129, 142-147；笔记 D.1 D4 与 D.3）。*为什么*有 16 个保持空闲并不是一项测量：一个 2 x 2 cluster 需要同一个 GPC 内的两个完整 TPC（SM pair）（笔记根据其 SM 放置表做出的推断，D.3，标为 "Inferred"），而芯片 skyline（NVIDIA 内部）中每个奇数大小的 GPC 都留下一个 TPC 未配对（算术见第 2.1.1 节）。
4. 记录下来的 `smem_u32(smem)` 在每个 CTA 中都是 `0x400 + (rank << 24)`，并且 `mapa(base_r, t) == base_t`（实测，笔记 D.1 D5）。如果低 24 位不是 0x400，就是有某个静态 `__shared__` 混了进来（第 13.3 节，陷阱 12）。
5. cluster 属性已经生效：在记录下来的身份中，`%cluster_nctaid` 读出 (2, 2, 1)，`%cluster_ctarank == x + 2 y`，其中 `(x, y) = %cluster_ctaid`，并且对全部 4096 个 CTA 都有 `(x, y) == (blockIdx.x & 1, blockIdx.y & 1)`（这些就是笔记 B.4 的 tile 公式所假定的恒等式）。没有该属性的启动是一次隐式的 1 x 1 x 1 cluster 启动：`%cluster_nctaid` 会读出 (1, 1, 1)，而每个 CTA 都会读出 rank 0。于是每个 CTA 都会认为自己是 CTA 0 和 pair leader，而从步骤 3 起，kernel 会在所有角色上同时出错（这一后果出自我的分析；prologue 从这些寄存器推导出每一个角色标志）。

消除的 bug 类别：启动配置、参数块 ABI（在 CUDA 13 的头文件下 `sizeof(ExplicitGemmParams)` 是 640，因为 `alignof(CUtensorMap)` 是 128；explicit.hpp:65-76 与笔记 G.3）、地址空间假设。

### 步骤 3. Prologue：身份、初始化、cluster 屏障、分发骨架

加入真正的 prologue（explicit.cu:1098-1179）：warp 索引与 lane 选举、rank/leader/peer 标志、来自参数块的 `alpha`/`beta`/`k_tiles`、描述符预取，以及 33 次 `mbarrier.init`（四组，分别为 8 + 8、2 + 2、4 + 4 和 2 + 2，每组之后各跟一条 `fence.mbarrier_init.release.cluster`，然后在最后一条 fence 之后是单独的 `tmem_dealloc` 初始化，explicit.cu:1123-1167）。然后是 `barrier.cluster.arrive.relaxed` -> 初始 tile -> `barrier.cluster.wait`，以及分发链（explicit.cu:1191-1202），其中每个角色体都为空，于是每个 warp 都贯穿而过并退出。

验证：kernel 退出；记录下来的地址 `smem_base + OFF_*` 与你的 smem 映射相符（这正是 trace 构建的 K1 记录所做的事，explicit.cu:1181-1188）；初始 tile 公式对全部 4096 个 CTA 给出互不相同的 `(tm, tn)`（explicit.cu:1175-1176，引述于第 11.6 节）。

转储各个 `(tm, tn)` 对，并对每个 CTA 检查 `tm % 2 == x` 与 `tn % 2 == y`。步骤 4 的 A multicast 依赖第一个恒等式：x 相同的两个 CTA 共享 `tm`，因而共享 A 的行（`a_row = 128 * tm + 64 * y`，explicit.cu:777）。第二个恒等式并*不*选择一个 CTA 的 B 半块；选择它的是 x（`b_col = 128 * tn + 64 * x`，explicit.cu:778，而且一个 pair 的两个 CTA 共享 `tn`）。第二个恒等式所做的是让一个 cluster 的两个 pair 持有相邻的 `tn`；CLC 解码在 explicit.cu:693 处加上 `y` 时假定了这个恒等式（对一个 cluster 的第一个 CTA，`y0` 是偶数，所以 `tn_next % 2 == y`）。

消除的 bug 类别：偏移、初始化计数（把它们读回来：一个屏障字编码了它的计数，笔记 D.2 的 `mbarrier.init` 行）、cluster 会合。从这里起必须始终遵守一条顺序规则：在 `cluster_wait()` 之前不得有任何远程屏障操作（explicit.cu:1178-1179；笔记第 7.3 节步骤 6）。

### 步骤 4. 一个静态的 one-tile-per-CTA 调度，配一个朴素的 epilogue

这一步得到第一批正确的数值，而且它刻意*不是*最终结构。每个 CTA 恰好只计算它的启动 tile：`producer_role` 和 `mma_role` 与 explicit.cu 中相同，但把 `do { } while (valid)` 循环缩减为一遍，并去掉 `clc_consume` 调用；累加器 stage 始终为 0；epilogue 读取 TMEM 并用普通的 `st.global` 存储 D，而不是经由 swizzled smem 与 TMA。

朴素的 epilogue：warp w、lane l 在 `T + ((32 w) << 16) + 16 e` 处执行 explicit.cu:1036 的八条 `tcgen05.ld.32x32b.x16`，在这个调通版本中*每次*加载之后都带一条显式的 `tcgen05.wait::ld`，并对 j = 0..15 把 `fmaf(beta, 0.0f, alpha * acc)` 写到 `D[(128 tn + 16 e + j) * M + 128 tm + 32 w + l]`（D 是 M-major 的，元素 (m, n) 位于 n * M + m）。最终的 kernel 每个 tile 发射一条 `tcgen05.wait::ld`，并对 subtile 0..6 依赖 ISA 的寄存器依赖规则（第 2.4.4 节），ptxas 用 scoreboard 实现该规则（wait 操作码的缺失是实测的，scoreboard 的解码是推断的；笔记 D.2 的 `tcgen05.wait::ld` 行，G.3）。在调通阶段，这些额外的 wait 只付出性能代价，而性能此时尚无关紧要，并且它们排除了一个嫌疑对象。

退出脚手架：在这一步，用一次粗暴的 cluster 范围会合取代笔记第 7.10 节的四个尾段。所有 CTA 的全部 256 个线程都在 `tcgen05.dealloc` 之前执行 `barrier.cluster.arrive` / `barrier.cluster.wait`；原本会提前退出的那些 warp（1 和 3）也参与。对于持久化 kernel 来说这既慢又错误，但它是*安全的*：没有任何 CTA 能在某个 peer 仍可能到达其屏障时退出，而 peer 也不能在 leader 的 MMA 被消费之前解除分配 TMEM。步骤 7 会取代它。

按照此顺序验证：

1. 在固定命令上得到 `Disposition: Passed`。
2. 如果失败，per-tile 不匹配图会对错误归类（下面的症状类别就是笔记 E.7.2 的失效模式表；原因则针对这一步的代码做了调整）：整个 tile 错误且 "tile (tm, tn) holds the reference of tile (tm', tn')" -> tile 公式或 A/B 坐标（`128 tm + 64 y`、`128 tn + 64 x`，explicit.cu:777-778）；64 行的半块错误 -> A 的目的地 `+ 8192 y`（explicit.cu:768）或 multicast 掩码 `0x5 << x`（explicit.cu:772）；64 列的半块错误 -> B 的坐标 `+ 64 x`；行 mod 32 错误 -> TMEM lane 项 `(32 w) << 16`（explicit.cu:1006）；每个元素都按某种因子模式出错 -> 描述符的 K 顺序（`+ 2 kb`，explicit.cu:933-934）或 `scale_c`（explicit.cu:917, 940）。
3. 此处的挂起来自三个地方之一，因为别的东西都还不存在。(a) TMEM 分配握手，它运行在任何屏障协议之前。`tcgen05.alloc.cta_group::2` 必须由 pair 的*两个* CTA 中完整的 warp 0 发射（"Must be issued by a single fully active warp"，tmem_allocator_sm100.hpp:124-133），然后 warp 0 必须执行 `bar.arrive 6, 160`（explicit.cu:885-887）。否则 epilogue 的各个 warp 会永远停在 `bar.sync 6, 160`（explicit.cu:997）中。(b) mainloop 协议。leader 的 warp 0 中 `WAIT_MAINLOOP_FULL` 在 q = 0 处的原因列表，以及 `WAIT_MAINLOOP_EMPTY` 在 q = 8 处的原因列表，都在第 13.3 节的陷阱检查清单中。那份检查清单没有解释的是，为什么在 q = 8 处最先挂起的 warp 是 producer。在 q = 8 处，也就是环的第一次回绕，warp 2 中 producer 的 `WAIT_MAINLOOP_EMPTY`（explicit.cu:735）是唯一依赖于 commit 计数的等待。一个到达不了这个 CTA 的 commit 掩码，或者一个大于实际送达的到达数的 empty 计数，会让 `empty[0]` 的到达数不足，于是环无法被重新填充；leader 在 q = 8 处的 `WAIT_MAINLOOP_FULL`（explicit.cu:923）这时只是后果，因为没有任何加载被发射。一个太*小*的 empty 计数根本不会挂起：它会提前重新填充某个 stage（W，第 4.9 节冒险 1）。(c) 在每个 CTA 一个 tile 的情况下，累加器的 *empty* 侧不可能挂起：`WAIT_ACC_EMPTY`（explicit.cu:918）在每个 tile 上都运行，但在一个全新的屏障上直接通过（producer parity 1，笔记 7.2），最早要到 tile 4，即 stage 0 的第一次复用，才可能阻塞。*full* 侧则可能：如果 commit 掩码 `0x3 << 2y`（explicit.cu:902, 944）到达不了这个 CTA，`WAIT_ACC_FULL`（explicit.cu:1034）就会在第一个 tile 上挂起（第 4.9 节冒险 3）。

消除的 bug 类别：描述符（`0x4000404000010880 + 1024 s + 2 kb`、`0x4000404000012880 + 512 s + 2 kb`、`idesc 0x10200010`；实测，笔记 D.1 D7）、TMA 坐标与目的地（实测，笔记 D.1 D8）、mainloop 屏障协议、TMEM 寻址、算术形式。

### 步骤 5. 真正的 epilogue：swizzled smem 与 TMA 存储

用 explicit.cu:1033-1072 替换朴素的 epilogue。每个 subtile：16 条 `st.shared.b32`，地址为 `st_off[j & 3] + 512 (j >> 2) + 8192 (e & 3)`，其中 `st_off[c] = OFF_SMEM_CD + 2048 w + 4 (l & 7) + 128 c + 32 (((l >> 3) & 3) ^ c)` (explicit.cu:1010-1014)；然后是 `fence.proxy.async.shared::cta` 和 `bar.sync 1, 128`；然后 warp 4 发射四次从 `cd_base + 8192 (e & 3) + 2048 i` 到 `(128 tm + 32 i, 128 tn + 16 e, 0)` 的 (32, 16) box TMA 存储，以及 `commit_group` 和 `wait_group.read 1`；然后是 `bar.sync 1, 128`。

验证：断言 `epi_swz` 在样本偏移上等于 `cute::Swizzle<2,5,2>` 的 `static_assert` (explicit.cu:197-202) 在编译期检查该公式。在运行时，不匹配图的列类会指出出错的项：`col % 4` 的模式 -> XOR 项或存储源 `2048 i`；16 列的组互换 -> `tcgen05.ld` 地址中的 `16 e` 对比存储坐标；某个 stage 被过早复用（subtile e + 4 的值出现在 subtile e 中）-> `wait_group.read 1` 或缺少第二个 `bar.sync` (explicit.cu:1069, 1071)。

消除的 bug 类别：swizzle 公式、从通用 proxy 到异步 proxy 的 fence、存储流水线的复用距离、box 几何。

### 步骤 6. CLC scheduler 与 throttle

把单遍执行改成持久化的 `do { } while (valid)` 循环。在四个消费点加入 `clc_consume` (explicit.cu:677-706)：warp 2 在其加载*之后* (explicit.cu:793)；warp 0 和 warp 4-7 在计算/存储当前 tile *之前*（explicit.cu:913 和 1028）；scheduler warp 在其查询之后 (explicit.cu:853)。加入 `scheduler_role` (explicit.cu:820-868) 和 producer 中的 throttle (explicit.cu:783-787)。让每个流水线状态跨 tile 延续；绝不重置它（笔记 E.6.5）。

验证：

1. 恰好一次的覆盖：在启动之前用 0xFF 字节哨兵填充 D (explicit.cu:1349)，使得一个从未被存储的 tile 读出来是 NaN 并使比对失败。于是通过比对就证明了每个 tile 都被存储过。trace 构建的 K5 记录证明 991 个被取消的 id 33..1023 各被消费了一次（实测，笔记 D.3 和 G.1）。
2. 在 33 个原生 cluster 下，每个 cluster 的 tile 数直方图是 32/31（带探针的 cluster 则为 19）（实测，笔记 D.3）。不要把其中任何一项硬编码；kernel 必须对任意数量的原生 cluster 和任意分发顺序都能工作（笔记 D.7）。
3. 当 `valid == 0` 时没有任何 warp 读取 `x0`/`y0`：那些寄存器保存的是*上一个*响应（在全部 165 次未取消的解码中实测，笔记 D.3），而 explicit.cu 只在 `if (valid)` 之下读取它们 (explicit.cu:691-694)。

消除的 bug 类别：800 这个计数（scheduler warp 自己的 32 也是其中一部分，explicit.cu:183）、由 lane 0..3 对全部四个 CTA 的 `clc_full` 进行的远程装填 (explicit.cu:842)、消费的放置位置（取回的 tile 只有在循环体末尾才成为当前的那一个，explicit.cu:794, 947, 1073）、由 throttle 强制执行的单个未完成查询规则（笔记第 7.4 节）。

### 步骤 7. 尾段与解除分配握手

移除步骤 4 的脚手架，放入笔记第 7.10 节的四个退出守卫：producer 的 8 次延续其状态的阻塞等待 (explicit.cu:799-806)、scheduler 的 2 对 `test_wait`/`wait` (explicit.cu:858-867)、leader 的 4 次累加器等待 (explicit.cu:952-962)，以及 `tmem_dealloc` 握手（explicit.cu:964-971，引述于第 9.6 节）。

为什么尾段延续状态，而不是在每个屏障上都等待同一个 parity：在 N 个 producer 步之后，尾段步 i 以 parity `1 ^ (((N + i) / Stages) & 1)` 等待 `empty[(N + i) % Stages]`，而除非 `N % Stages == 0`，这些 parity 在各屏障之间是不同的（笔记第 7.2 节）。对于 mainloop，N = 128 T（T = 该 cluster 的 tile 数，这是笔记的符号；第 8 章写作 `n_tiles`）总是 8 的倍数，所以每个 *mainloop* 尾段等待都恰好使用 parity 1。实测：全部 1056 条 `TAIL_MAINLOOP` 记录，即每 CTA 8 条 x 4 个 CTA x 33 个 cluster，都通过了笔记 G.1 的 `k_tail_mainloop_parity` 检查。总计 1518 条的 `K_TAIL` 还包含 264 条 `TAIL_ACC`、66 条 `TAIL_CLC` 和 132 条 `WAIT_TMEM_DEALLOC` 记录，它们的 parity 遵循各自的 `1 ^ (((T + i) / Stages) & 1)`，其中 `Stages` = 4 和 2（`WAIT_TMEM_DEALLOC` 的等待使用固定的 parity 0，explicit.cu:965），并且*不*全是 1。T = 31 个 tile 的累加器尾段等待 stage 3、0、1、2，parity 为 0、1、1、1（笔记第 7.6 节步骤 3）。对全部四个都施加同一个 parity 的代码会在某些 cluster 上挂起而在另一些上不会。

ISA 规则规定，在观察到一次失败之后发射的 `try_cancel` 是未定义的（第 2 章，第 2.12.1 节），这就是 scheduler 循环在某个响应解码为未取消的那一刻就停止查询（`} while (valid);`，explicit.cu:856）而不是运行固定次数的原因。

验证：连续五次运行都通过（退出竞态是间歇性的）。"Passed, then hang or error at the timed loop"（笔记 E.7.2，最后一行）意味着 TMEM 没有被解除分配（一次泄漏的分配可能使之后的分配陷入饥饿，而且不得依赖拆除时的清理（NVIDIA 内部），所以下一次启动在该 SM 上的 `tcgen05.alloc` 可能阻塞；症状的对应关系是笔记的推断，E.7.2），或者某个 CTA 在到达仍在途时就退出了。trace 构建数出每个 cluster 46 个尾段步（rank 0 上 8 + 2 + 4 + 1，rank 1 和 3 上 8 + 1，rank 2 上 8 + 4 + 1；实测，笔记 G.1）。

消除的 bug 类别：退出竞态、TMEM 泄漏、ptxas 插入的 dealloc 一致性自陷（`BPT.TRAP`，笔记 D.2 的 `tcgen05.dealloc` 行）。

### 步骤 8. 对照设计做结构验证

结构检查把编译出的产物与操作台账相比较：用 `cuobjdump --dump-resource-usage`、`--dump-ptx` 和 `--dump-sass` 提取，按你的 kernel 的 `.entry` / `Function :` 行把它切出来，然后计数。这个 kernel 的契约（实测，笔记 G.3，全部为精确值）：

| 层级 | 项目 | 预期 |
|---|---|---|
| PTX | `tcgen05.mma.cta_group::2.kind::f16` / `tcgen05.commit ... multicast::cluster` / `tcgen05.ld ... 32x32b.x16` / `tcgen05.wait::ld` | 4 / 2 / 8 / 1 |
| PTX | `tcgen05.alloc`、`dealloc`、`relinquish_alloc_permit` | 各 1 |
| PTX | A multicast 加载 / B 加载 / D 存储 / `commit_group` / `wait_group.read 1` / `wait_group.read 0` | 2 / 2 / 32 / 8 / 8 / 0 |
| PTX | `clusterlaunchcontrol.try_cancel` / `mbarrier.init` / `fence.mbarrier_init` / `bar.sync` / `bar.arrive` | 1 / 33 / 4 / 17 / 1 |
| PTX | `mul.f32` / `fma.rn.f32` / `add.f32` / `st.shared.b32` / `fence.proxy.async.shared::cta` | 128 / 128 / 0 / 128 / 12 |
| PTX | `griddepcontrol.wait` / `launch_dependents` / `prefetch.tensormap` | 2 / 1 / 4 |
| SASS | `UTCHMMA.2CTA` / `UTCBAR.2CTA.MULTICAST` / `UTMALDG.3D.MULTICAST.2CTA` / `UTMALDG.3D.2CTA` / `UTMASTG.3D` | 4 / 2 / 2 / 2 / 32 |
| SASS | `LDTM.x16` / `UGETNEXTWORKID.BROADCAST` / `PREEXIT` / `ACQBULK` / `SYNCS.EXCH.64` | 8 / 1 / 1 / 2 / 33 |
| SASS | `FMUL` / `FFMA` / `FADD` / `STS` / `LDL`+`STL` | 130（128 + 2 来自 `%cluster_ctaid` 的下沉；归因是推断，笔记 D.2 和 G.3）/ 128 / 0 / 131 (128 + 3) / 0 |
| 资源 | `REG` / `STACK` / `LOCAL` / `SHARED` | 26 / 0 / 0 / 1024（系统保留的千字节，每个 kernel 都有） |
| 热循环 | MMA k-tile 循环 / producer k-tile 循环 / 每个 tile 的 epilogue / 每个 tile 的 scheduler | 47 / 37 / 696 / 50 条指令（CUTLASS kernel：44 / 31 / 1265 / 50） |

为什么每个计数是这个值（源自 explicit.cu；SASS 一侧源自笔记 G.3）：

- `tcgen05.mma` 4 = 完全展开的 kb 循环中每个 k-block 一条（描述符在 :933-934 处递增 `+ 2 kb`）。`tcgen05.commit` 2 = mainloop 的释放 (:942) + 累加器 commit (:944)。
- `tcgen05.ld` 8 = 完全展开的 epilogue 循环中每个 subtile 一条；`tcgen05.wait::ld` 1 = 每个 tile 在 subtile 7 之后一次 (:1037-1038)。
- A multicast 加载 2 和 B 加载 2 = `load_ktiles` 的每个内联副本各一条，它被调用两次（:789 用于 8 个 k-tile 的 prologue，:791 用于其余 120 个）。D 存储 32 = 4 个 box x 8 个 subtile；`commit_group` 8 和 `wait_group.read 1` 8 = 每个 subtile 一条。
- `clusterlaunchcontrol.try_cancel` 1 = scheduler 循环体。
- `mbarrier.init` 33 = 8 + 8（mainloop）+ 2 + 2（CLC）+ 4 + 4（累加器）+ 2 + 2（throttle）+ 1（dealloc）(:1124-1167)。`fence.mbarrier_init` 4 = 每个 init 组一条（:1133、:1144、:1154、:1164；dealloc 的 init 没有自己的那一条）。
- `bar.sync` 17 = 每个 subtile 2 条 x 8 (:1051, :1071) + 1 条 `bar.sync 6, 160` (:997)。`bar.arrive` 1 = :887。
- `mul.f32` 128 和 `fma.rn.f32` 128 = 16 个值 x 8 个 subtile (:1045-1046)；`st.shared.b32` 128 = 同上。
- `fence.proxy.async.shared::cta` 12 = 8 条存储侧 fence（每个 subtile 一条，:1050）+ 4 条 CLC 解码 fence（每个内联的 `clc_consume` 一条，:683，从 :793、:853、:913、:1028 处调用）。
- `griddepcontrol.wait` 2 = producer (:764) + scheduler (:821)。`prefetch.tensormap` 4 = warp 1 中的 A 和 B，warp 3 中的 C 和 D (:1114-1119)。
- SASS：`SYNCS.EXCH.64` 33 = 33 条 init 一对一地下沉而来。`FMUL` 130 = 128 + 2 来自 `%cluster_ctaid` 的下沉。`STS` 131 = 128 + 两条 `tmem_base_ptr`（alloc 的结果）的 `STS` + 一条在 dealloc 附近存入系统保留千字节的 `STS.U8`（在笔记 D.2 和 G.3 中计入；来源未解决：它不是 relinquish 下沉的一部分，NVIDIA 内部）。

计数时的注意事项（笔记 E.7.4）：用字面意义的点来匹配 `mbarrier\.init`（裸的 `.` 也会匹配 `fence.mbarrier_init`）；在计数 `EXIT` 时排除 `PREEXIT`；`bar.sync`、`expect_tx`、`mbarrier.init` 的操作数以及指令描述符都是由 `mov.b32` 物化出来的寄存器，所以 `bar.sync 1, 128` 在 PTX 中匹配不到任何东西。要通过供值的 `mov.b32` 或 SASS 立即数（`BAR.SYNC.DEFER_BLOCKING 0x1, 0x80`、`R5 = 0xc000`）来验证它们。还要检查缺失列表：没有 `cp.async.ca/cg`、`wgmma`、`st.global`、`atom`、`trap`、`.local`、`ld.global`、`stmatrix`/`ldmatrix`、`tcgen05.fence`。

这能抓住 harness 抓不住的东西：被折叠掉的 `alpha * acc`（没有 `FMUL`）、被丢掉的 `fmaf`（出现了一条 `FADD`）、被展开的 k-tile 循环（热循环长出好几倍）、收紧退出条件的 `wait_group.read 0`（只有在你接受这一偏离时才允许）、溢出（`LDL`/`STL`），以及任何多余的屏障或选举。

---

## 13.3 按症状排列的陷阱检查清单

症状代码：**H** 挂起（什么都不打印，100 % 利用率）、**T** 自陷或设备端故障（`cudaDeviceSynchronize` 错误）、**W** 错误的值（`Disposition: Failed`）、**S** 静默变慢或结构上不同（通过，但清单或计时发生变化）。“避免于何处”引用的是 explicit.cu。

| # | 陷阱 | 症状 | 为什么会发生 | 避免于何处 |
|---|---|---|---|---|
| 1 | 混用通用地址和 `shared::cta` 地址：一个 64 位通用指针被截断进一个 asm `"r"` 操作数，或者一个 32 位共享地址被当作 C++ 指针解引用 | T，或在错误的屏障上 H | 指名共享内存的 asm 操作数必须是 32 位的 `shared::cta` 窗口地址（`smem_u32`，即 `cvta.to.shared` 的结果）；C++ 解引用必须使用通用指针 | explicit.cu:21-23 处的文件级规则；:258 处的 `smem_u32`；MMA 和 epilogue 角色——即在 C++ 中解引用共享内存的那两个——同时接受 `smem` 和 `smem_base` (:881, :995)；producer 和 scheduler 角色只接受 `smem_base` (:762, :820) |
| 2 | 错误的起始 phase | 在最开始的那次 producer 等待处 H（处于 phase 0 的 producer 等待一个需要它自己的数据才能发生的完成），或 W（处于 phase 1 的 consumer 立刻通过一个全新的屏障并读取一个空的 stage） | 刚初始化的 mbarrier 处于 phase 0；`try_wait.parity 1` 立即返回 true（ISA 所说的 "immediately preceding phase, for which waitComplete returns True"，第 2.8.3 节）；producer 以 `{0, 1}` 起始，consumer 以 `{0, 0}` 起始（笔记第 7.2 节） | explicit.cu:723-726 处的 `ProducerState`（`prod_phase = 1`）以及 :766（throttle）、:823（CLC 查询）、:904（累加器）处的其他 producer 状态 `{0, 1}`；:903（mainloop）、:1019（累加器）、:822（throttle）以及 :767、:824、:905、:1020（CLC）处的 consumer 状态 `{0, 0}` |
| 3a | `acc_empty` 计数为 128 而不是 256 | W，间歇性（leader 的 MMA 覆写 peer 的 epilogue 仍在读取的 TMEM stage），随后 phase 漂移 -> H | 两个 CTA 的 128 个 epilogue 线程都在 *leader 的*屏障上到达（笔记第 7.2 节） | explicit.cu:184 处的 `kAccEmptyCount = 256` |
| 3c | CLC `empty` 计数差了一个 warp：768（忘了 scheduler warp 自己的 32）、896 或 928（把空闲 warp 算了进去） | 太小：W（一个槽位在某个迟到的 warp 解码它之前就被重新装填并覆写：tile 重复或丢失）；太大：在对该槽位的第二次查询处 H | 800 = 32 + 4 x (32 + 128 + 32)：cluster 中每一个消费线程，在 CTA 0 的屏障上（kernel :511-514） | :183 处的 `kClcEmptyCount = 800`，连同 :192 处的 `static_assert` |
| 5a | TMA 屏障操作数上没有 peer 位掩码 | 在 leader 中 q = 0 处 H（`WAIT_MAINLOOP_FULL`）：peer 的 box 计入 peer 自己的、从未被装填的 `full[s]` | 在 `.cta_group::2` 下，硬件按*持有所指名屏障的 smem 所属 CTA 的 parity*把每个 box 的完成路由到 leader 或 peer（表格下方引用的 PTX ISA 规则）；清除地址第 24 位指名的是偶数 CTA 的屏障，所以全部六个 box 都计入 pair leader（笔记 B.6；Colfax 的 TMEM 教程没有涉及这一点） | :770 处的 `full_masked_base = smem_base & kPeerBitMask` |
| 5b | 累加器释放上没有 peer 位掩码 | 在 tile 4 处 H（stage 0 的首次复用）：leader 看到 256 中的 128 | 同一机制；`umma_arrive_2x1SM_sm0` (barrier.h:905-921) | :370-372 处的 `mbar_arrive_cluster_masked`，在 :1039 处使用 |
| 6a | 错误的 A multicast 掩码（例如 0x3 而不是 0x5 << x） | W（各半部分落到错误的 CTA 中）和/或 H（字节计数永远达不到） | 共享 `tm` 的两个 CTA 是 rank x 和 x + 2 | :772 处的 `a_mcast_mask = 0x5u << x` |
| 6b | 把 commit 掩码 0x3（字面量）或 `0x3 << 2y`（累加器的 pair 掩码）用于 mainloop 释放 | 在 q = 8 处 H（`WAIT_MAINLOOP_EMPTY`）。来自两个 leader 的字面量 0x3：rank 0 和 1 每次使用收到*两次*到达（两个 leader 都 multicast 到 {0, 1}），而 rank 2 和 3 一次也*没有*收到，所以 rank 2-3 的 producer 挂起。pair 掩码 `0x3 << 2y`：每个 CTA 每次使用收到一次到达而不是两次，所以每个 producer 都挂起 | `calculate_multicast_mask<kRowCol>` 给出全部四个 CTA (sm100_pipeline.hpp:57-91) | :186 处的 `kMainloopReleaseMask = 0xF`；pair 掩码 `0x3 << 2y` 只用于累加器 commit，:902 |
| 7a | TMA 存储之前缺少 `fence.proxy.async.shared::cta` | W，间歇性：异步 proxy 在通用 proxy 的 `st.shared` 对其可见之前就读取 smem | TMA 位于异步 proxy 中；通用写入需要 proxy fence（笔记第 7.7 节步骤 6） | :1050 |
| 7b | CLC 解码之后缺少 fence | W，罕见：对该槽位的下一次 `try_cancel`（异步 proxy）在通用的 `ld.shared.b128` 仍在途时覆写了它 | 同一 proxy 规则，方向相反（sm100_tile_scheduler.hpp:430，两棵树中是同一行；explicit.cu 在 :683 处的注释写的是 :433，差了三行） | :683 |
| 8a | 缺少尾段 | 在*之后的*启动中出现间歇性的 W 或 T：某个 CTA 在指向其 smem 的远程到达仍在途时退出，而这块内存此时已归该 SM 上的下一个 CTA 所有 | 退出时没有 cluster 范围的同步（笔记第 7.10 节） | :799-806, 858-867, 952-962 |
| 8b | 缺少 dealloc 握手，或在所有 TMEM 读取之前就 `dealloc` | 结束时 T（`unspecified launch failure` 是笔记为 ptxas 的 `BPT.TRAP` 检查*推断*出的故障字符串，E.7.2）或在*下一次*启动中 H（它的 `tcgen05.alloc` 阻塞在泄漏的 TMEM 上；在 CTA 仍驻留期间一次泄漏可能使之后的分配陷入饥饿，而且不得依赖拆除时的清理（NVIDIA 内部）；下一次启动的对应关系是推断，E.7.2；本项目中不存在这样的运行）；在所有 TMEM 读取之前 `dealloc` 是一次释放后使用，未定义（NVIDIA 内部） | peer 不得在 leader 排空累加器流水线之前解除分配（笔记第 7.6 节步骤 4） | :964-971 |
| 9 | `tcgen05.ld` 从 warp 的象限之外的 lane 读取 | 未定义：垃圾值或故障（未验证是哪一种） | "A lane of the Tensor Memory can be accessed by a single warp in the warpgroup"：一个 warpgroup 的 warp w 只能访问 lane [32 (w % 4), +32)（PTX ISA 9.4，第 9.7.18.8.1 节 "Access restrictions"，其表格给出 warp 0 -> lane 0-31、1 -> 32-63、2 -> 64-95、3 -> 96-127；Colfax 的 TMEM 教程；笔记 B.6） | :1006 处的 `tmem_lane_off = (32u * w) << 16`；epilogue warp 是 4-7，且 `w = warp_idx - 4` (:1004)，所以该 warp 在其 warpgroup 内的索引 `warp_idx % 4` 等于 `w` |
| 10 | `alloc`/`dealloc` 的列数或基址不匹配，或由不同的 warp 执行 `dealloc` | T（`BPT.TRAP` 一致性检查，笔记 D.2）或下一次启动时 H（推断，E.7.2）；不匹配是未定义的，而一次泄漏可以在该 CTA 仍驻留期间使之后的分配饥饿，且没有可依赖的拆除时清理（NVIDIA 内部） | 二者都必须由单个完全活跃的 warp 发射，并在两个 CTA 中使用相同的逻辑 warp ID（tmem_allocator_sm100.hpp:124-133 `allocate`、:147-156 `free`）；两者都使用*同一个* warp 是 Colfax 的 TMEM 教程的规则（"the same warp should both allocate and deallocate"），并且与相等的计数一样，也是 CUTLASS 的实践（kernel :728、:804）。ISA 的两种形式：不带 `.exclusive` 时，`nCols` "must be a power of 2, in the range [32, 512]"；带 `.exclusive` 时，在 sm_100f 上为 "a multiple of 32, in the range [32, 512]"（PTX ISA 9.4，tcgen05.alloc）。本 kernel 使用非独占形式，取 512 | 位于 :885 和 :971 的 `kTmemColumns = 512`；两者都在 warp 0 中 |
| 11 | 发散代码中的 `elect.sync` 或任何 `.sync.aligned` 指令 | H 或未定义 | 这些都要求一个收敛的 warp（笔记 B.6）；`tcgen05.alloc/dealloc/relinquish`、`tcgen05.ld`、`tcgen05.wait::ld`、`barrier.cluster.*.aligned` 全都是 `.aligned` | 在各角色中，每一次选举都在 warp 层级；CLC 装填的 `if (lane < 4)` 是 W13 `mbar_arrive_expect_tx_remote_pred`（:393-404）的 *asm 内部的谓词*，而不是一个分支 |
| 12 | TU 中任何位置的静态 `__shared__` | W（预期如此；本项目中不存在基址未对齐的运行）：动态基址不再必然是 1024 对齐的（NVIDIA 内部），于是 TMA 在写入时施加的 SW128 XOR 不再与 UMMA 描述符的假设 `base_offset = 0` 相符（mma_sm100_desc.hpp:757-758）。那个 XOR 是 `Swizzle<3,4,3>`：地址位 [4,7)，即 16 字节 chunk，与位 [7,10)，即行号 mod 8，做 XOR（explicit.cu:126-127，`static_assert` 位于 :204）。关于该模式的假设：偏移了 k x 128 B 的基址会使每一行读到第 (r + k) mod 8 行的 chunk 置换，这是一种行 mod 8 的模式；不是 128 B 整数倍的偏移还会打乱行内的 16 字节 chunk，而且 TMA 可能会在目的地址不是 128 字节对齐时触发故障 | 静态 `__shared__` 对象相对于动态缓冲区的放置位置不是源码层面的契约（NVIDIA 内部），所以任何静态声明都可能移动动态基址；在静态 smem 为 0 时，1 KiB 的系统保留区域使实测基址位于 0x400（笔记第 6.2 节） | explicit.cu 中没有静态 `__shared__`：`cuobjdump` 中实测 `SHARED:1024`（系统保留的那个千字节，笔记 G.3），且 K1 记录中的基址为 `0x400 + (rank << 24)`（笔记 G.1）；`sharedSizeBytes 0` 是 CUTLASS kernel 的实测属性（笔记 D.1 D4） |
| 13 | Stage 基址未按 1024 对齐（SW128）或未按 512 对齐（SW128_32B） | W，与第 12 行相同的模式 | swizzle 周期为 1024 B（8 行 x 128 B）和 512 B（4 x 128 B） | `static_assert` 位于 :162-163 |
| 14 | 14 位的描述符起始字段：传入通用地址、忘记 `>> 4`，或超过 256 KiB | T（`illegal instruction` 是笔记*推断*的故障字符串，E.7.2；本项目中不存在描述符畸形的运行）或 W | `start_address = (addr >> 4) & 0x3FFF`（mma_sm100_desc.hpp:754-755）；该字段丢弃了 rank 位，这正是 leader 与 peer 持有完全相同描述符的原因 | :896-897 以及位于 :165 的 `static_assert` |
| 15 | 读取一个未被取消的响应的 `x0`/`y0`/`z0` | W：一个重复的 tile（不匹配图中的 `(tm, tn) holds (tm', tn')`） | 带谓词的 `get_first_ctaid` 让这些寄存器保持未写入；它们保留着上一个响应（实测，笔记 D.3） | :691-694 处的 `if (valid)` |
| 16 | 在错误的时点消费 CLC 响应（warp 2 在其加载之前；warp 0 或 4-7 在其 tile 之后；在循环体中途就把取回的 tile 置为当前） | W：整个 tile 在另一个 tile 的坐标下被计算或存储；不会挂起，因为 800 这一计数仍然得到满足 | 每个角色在 baseline 的时点上每个 tile 恰好消费一个响应，并且只在循环体末尾切换 tile（笔记第 7.4、7.9 节） | :793-794、913 和 947、1028 和 1073 |
| 17 | `scale_c` 没有在每个 tile 开始时重置为 0 | W：一个 TMEM stage 上第一个之后的每个 tile 都包含两个 tile 之和 | 一个 tile 的第一条 MMA 必须覆写该 stage（`ScaleOut::Zero`：CUTLASS 对 `tcgen05.mma` 的 `scale_c = 0` 谓词操作数的命名，它使得 D = A x B 而不是 D = D + A x B；第 9 章） | :917（每个 tile 重置）、:940（每条 MMA 之后设置） |
| 18 | k-tile 循环上缺少 `#pragma unroll 1` | S：迭代次数 `k_tiles - 8` 是一个运行时值（这里为 120；`k_tiles = (params.k + 63) >> 6` 位于 explicit.cu:1112），所以完全展开是不可能的，但编译器可能按某个因子部分展开并附带一个余数循环：代码膨胀、uniform 寄存器压力，可能还有溢出（假设：本项目中未实测；CUTLASS kernel 的未展开循环是实测的参照，笔记 D.2） | k-tile 数是一个运行时值；baseline 把两个循环都标为 `CUTLASS_PRAGMA_NO_UNROLL` | :733、:921 |
| 19 | 编译期的 `alpha = 1.0f`、`beta = 0.0f` | S：`1.0f * acc` 会折叠为 `acc`，`fmaf(0, 0, t)` 会折叠为 `+0.0f + t`（推断的编译器行为，笔记第 0 节和 D.7；本项目中不存在常量 alpha/beta 的构建），所以那 128 条 `FMUL` + 128 条 `FFMA` 会消失。如果你随后“简化”为只剩 `alpha * acc`，那么一个 `-0.0` 的累加器会保持 `-0.0`，而 baseline 的 `fmaf(0, c, -0.0)` 给出 `+0.0`：这仍然是 **S** 而不是 W，因为 harness 的比较是数值相等（`-0.0 == +0.0` 通过，笔记 E.7.2 的 "Pass" 行）。只有 FFMA 计数（预期 128，实得 0）和按位比较才能捕获它（第 13.5 节步骤 8） | baseline 从参数块读取二者，并在 c = 0 下计算 `t = alpha * acc; d = fmaf(beta, c, t)`（笔记第 8 节、D.7） | 位于 :1110-1111 的运行时加载；位于 :1045-1046 的两条语句 |
| 20 | 忘记 `__launch_bounds__(256, 1)` | S：PTX 失去了 `.maxntid 256` / `.minnctapersm 1`（与笔记 G.3 的清单不匹配）以及线程数多于 256 的启动会被拒绝这一编译期保证。由此并不会带来溢出风险：在 256 个线程且每个 SM 1 个 CTA 下，隐含的上限是 65536 / 256 = 256 个寄存器，亦即硬件最大值 255，而这也是没有该属性时的上限，因为那时 nvcc 根本不假设任何寄存器上限（`--maxrregcount`："Default: No maximum is assumed"），而该 kernel 使用 26 个（笔记 G.3；笔记 E.8 D2：任何带 `LOCAL:0` 的 `REG <= 255` 在这里都没有 occupancy 影响）。无论如何都要检查 `REG`，因为 ptxas 可能分配得不同 | `device_kernel` 带有它（device_kernel.h:114-127） | :1095 |
| 21 | 描述符中错误的 K 顺序：每个 kb 用 `+1` 或 `+4`，或在 stage 之外迭代 kb | 每个元素都是 W | 一个 k-block 是 16 个 fp16 = 32 B = 2 个 16 B 的单位；A 与 B 必须步调一致地推进 | :933-934 处的 `+ 2u * uint32_t(kb)` |
| 22 | 尾段中的所有屏障只用一个 parity | H（累加器与 CLC 尾段），发生在 tile 数 T 不是 stage 数倍数的 cluster 上（这里 33 个 cluster 中有 19 个 T = 31） | 尾段延续 producer 的状态（笔记第 7.2 节） | 每个尾段循环内部的 `advance<>`，:805、866、960 |
| 23 | 角色内部的 `__syncthreads()` / `bar.sync 0`，或参与者数量与其到达数不匹配的命名屏障 | H | 存活的角色 warp 运行着互不相关的循环（warp 0 和 2：每个 tile 128 个 k-tile；warp 4-7：每个 tile 8 个 subtile；warp 1：每个 tile 一次迭代），所以放在某一个角色内部的 CTA 范围的 `bar.sync 0` 永远不会被其他角色匹配。被错误数量的参与者到达的命名屏障（`bar.sync 1, 128` 被 96 个线程到达）以同样的方式挂起。已退出的 warp 1（rank 1-3）和 3 *不是*其机制："barrier{.cta}.sync ... causes executing thread to wait for non-exited threads of all other warps participating in the barrier" 以及 "All threads in warp (except for those have exited) must execute barrier{.cta}"（PTX ISA 9.4，barrier{.cta}） | 命名屏障是按角色参与者划分的：:997（`bar.sync 6, 160`）、:887（`bar.arrive 6, 160`，warp 0）、:1051 和 :1071（`bar.sync 1, 128`） |
| 24 | 在 cluster 会合之前进行远程屏障操作，或没有 `fence.mbarrier_init.release.cluster` | H 或 W：一个 peer 到达一个尚未初始化的屏障 | 初始化必须在 cluster 范围内释放，且会合必须先完成（笔记第 7.3 节步骤 3-6） | :1133、1144、1154、1164（fence）；:1170、1179 |
| 25 | 为 16 位的 `multicast::cluster` / `tcgen05.commit` 掩码操作数使用 32 位寄存器 | 来自 ptxas 的构建错误，引用的是 PTX 文本而非源码行 | 该掩码是一个 `"h"` 操作数 | 位于 :772、902 的 `uint16_t` 掩码；位于 :457、538 的 `"h"` 约束 |

第 5a 行和第 6a 行背后的硬件规则，出自 PTX ISA（带 `.cta_group::2` 与 `.multicast::cluster` 的 cp.async.bulk.tensor）："The mbarrier signal is multicast either to all the odd-numbered CTAs or even-numbered CTAs for each possible CTA-Pair within a cluster. In particular, for each destination CTA specified in the ctaMask, the mbarrier signal is sent either to the destination CTA or its peer-CTA based on the %cluster_ctarank parity of the shared memory where the mbarrier object mbar resides"（PTX ISA 9.4，cp.async.bulk.tensor）。关于第 8a 行和第 8b 行，见第 2 章，第 2.9.4 节（DSMEM 退出规则）和第 2.3.2 节（`tcgen05` 解除分配规则）中引用的退出规则。

从该表得出的一条经验法则：**挂起几乎总是一个计数、一个掩码、一个 parity 或一个地址位；错误的值几乎总是一个坐标、一个偏移或一次重置。**

### 在另一套工具链、驱动或 GPU 上需重新验证的假设

该 kernel 在一台机器上经实测是正确的且性能等价的（B200、CUDA 13.3、驱动 610.57.04，笔记第 0 节）。它的工作假设中有七条是*实测的或继承而来的实践*，而不是 ISA 保证。移植到另一套工具链、驱动或器件时，应当在信任该协议之前，用第 13.2 节步骤 8 的检查重新核对它们：

1. **地址约定 `smem_base = 0x400 + (r << 24)` 以及位于地址第 24 位的 peer 位**（实测，笔记 D.1 D5）。ISA 只规定了 `cta_group::2` 屏障的 parity 规则，而没有规定 rank 在 `shared::cta` 地址中的位置（第 2 章，第 2.9.3 节）。重新运行 K0 探针，并检查 `base & 0xFEFFFFFF == base of the even rank` 和 `mapa(base_r, t) == base_t`。
2. **被省略的 `tcgen05.fence::before_thread_sync` / `::after_thread_sync`**，位于 TMEM 读取的前后。ISA 的规范跨线程模式（9.7.18.6.4.4）把它们放在 mbarrier wait 和起释放作用的 arrive 旁边；CUTLASS 和 explicit.cu 都没有发出它们（笔记 7.2；G.3 清单中 `tcgen05.fence` 为 0），并且在这里是数值相等的（G.1）。这是一条 CUTLASS 实践，而不是 ISA 保证（第 2 章，第 2.4.4 节）。
3. **每条 `mbarrier.try_wait` 默认的 `.cta` 作用域的 `.acquire`**，针对从 peer CTA 到达的释放（第 2 章，第 2.8.4 节；关于它为何足够的解释是分析）。
4. **TMEM 基址 0**（实测 D6；kernel 读取它，从不假定它）以及 pair 分配被转发到 peer 的 `smem + 384` 槽位（对 baseline 的一项 SASS 层面的观察，笔记 D.2、D.6 第 7 项；第 9 章，第 9.4.1 节）。
5. **把 `elect.sync` 折叠为每 warp 一次发射的 uniform 数据通路下沉**（第 12 章，第 12.3 节）：这是一种 ptxas 行为，而不是 PTX 保证；源码必须保留每一次选举。
6. **编译器对 peer 位掩码的折叠**（累加器释放处的 `0xFEFFFFF8`，producer 外提基址上的 `0xFEFFFFF0`，笔记 G.3）。
7. **当 box 的一部分越界时，TMA 计入完整 box 的字节数**（CUTLASS 编译期的 `TmaTransactionBytes` 依赖于此；ISA 只说了 "completeCount argument equal to amount of data copied"，第 13.4.1 节）：对固定的 8192^3 运行无关紧要，对余数则是决定性的。

---

## 13.4 泛化：每种情形都作为相对于本 kernel 的差异

下面的每种情形都以“变了什么，以及在哪些行”的形式陈述。协议的*形状*（五条流水线、每个引擎一个发射线程、四个尾段）从不改变；改变的是计数、掩码、字节总数和坐标。

### 13.4.1 M、N、K 上的余数

这次固定的运行没有余数：全部 4096 个 tile 都是完整的，且恰好 K = 128 x 64（笔记第 10 节）。要接纳任意 shape：

- **Tile 数与网格。** 用 scheduler 的公式（`get_tiled_cta_shape_mnl`，sm100_tile_scheduler.hpp:597-600）替换 explicit.cu:1267-1270 的相等性守卫：`ctas_m = round_nearest(ceil_div(M, 256) x 2, 2)`、`ctas_n = round_nearest(ceil_div(N, 128) x 1, 2)`、`ctas_l = L`。因子 2 和 1 是 2SM 原子的线程 shape，除数 2 是 cluster shape；`2 ceil(M / 256)` 等于 `round_up(ceil(M / 128), 2)`。经 AlongN 转置后的网格是 `(ctas_n, ctas_m, 1)`（`possibly_transpose_grid`，由 `get_grid_shape` 调用，:223-232；在这里，一个 `(x / 2) * 2` 会是一个空操作，因为两个计数都已经是 cluster shape 的倍数）。tile 公式本身（explicit.cu:691-694、1175-1176）不变。由于这一舍入，一个 cluster 可能拥有*完全*位于 M 或 N 之外的 CTA tile。对于 M = 8193，`ctas_m = 2 x 33 = 66`：tile 64 持有一个有效行，而 tile 65 完全越过了 M。它们的 TMA 加载完全零填充，它们的 TMA 存储完全被裁剪，而 CLC id -> (tm, tn) 映射不变。kernel 不得对它们做特殊处理；下面的余数处理覆盖了它们。
- **K 余数。** `k_tiles = (params.k + 63) >> 6`（explicit.cu:1112）已经是向上取整的除法。对于最后一个 k-tile，A 和 B 的 box 越过了 K。TMA 对越界元素零填充（`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` 选择的是零填充，笔记第 5.3 节），并且全部四条 MMA 仍然被发射（笔记第 10 节）。事务计数保持为 49152：CUTLASS 的 `TmaTransactionBytes` 是一个由 smem layout 的大小计算出的编译期常量（sm100_mma_warpspecialized.hpp:238-240），与余数无关，所以 CUTLASS（以及基于它构建的每一个 SM90/SM100 余数 kernel）依赖于硬件在 box 的一部分越界并被零填充时计入*完整 box 的*字节数。PTX ISA 只说 complete-tx 是以 "with completeCount argument equal to amount of data copied in bytes" 执行的（PTX ISA 9.4，cp.async.bulk.tensor），并没有说零填充的越界元素是否计入该数量（它的 5.5.3.3 定义的是填充模式，而不是计入的字节数），所以要把“完整的 box 被计入”当作（此处未验证；CUTLASS 依赖于此）来对待。一次 K = 8192 - 32 的运行，在 trace 构建的 4 s 挂起上界之下，一次启动就能解决它：在 `WAIT_MAINLOOP_FULL` 处 q = 127 时的挂起将意味着部分计入。
- **加载时的 M/N 余数。** 相同的机制：越过 M 或 N 的行或列以零到达，所以 MMA 在那里算出零。
- **存储时的 M/N 余数。** 使用 TMA 存储时，该单元会裁剪张量之外的元素，所以 explicit.cu 的存储路径不需要谓词化。（CUTLASS 位于 sm100_epilogue_tma_warpspecialized.hpp:692-716 的 "residue" 张量供给寄存器级的 fusion 回调，而线性组合从不读取这些回调：笔记第 7.7 节。）如果你在任何时候用 `st.global` 存储（如步骤 4 的调通 epilogue 那样），你必须逐元素加谓词：`128 tm + 32 w + l < M` 和 `128 tn + 16 e + j < N`。
- **对齐前提条件。** TMA 需要 16 字节对齐的全局 stride 和基址：对 fp16 的 A/B 而言 K % 8 == 0，对 fp32 的 D 而言 M % 4 == 0（这正是 `can_implement` 所强制的，笔记第 13 节第 18 项）。

### 13.4.2 L > 1（批量 GEMM）

tensor map 已经是 rank 3 的，批量 stride 作为第二个全局 stride（笔记第 5.3 节）。启动网格的 z 范围是 L：`get_grid_shape`（sm100_tile_scheduler.hpp:223-232，即从 kernel :389-395 调用的那个重载）调用 `get_tiled_cta_shape_mnl`（:593-605），后者在 :600 处的 `ctas_l = tiles_l` 就是未被 cluster 取整触及的 L，然后调用 `possibly_transpose_grid`。kernel 在 explicit.cu:747-748、754-755 和 1066 处把常量 0 作为第三个 TMA 坐标传入，而 `clc_decode` 返回一个被 `clc_consume` 丢弃的 `z0`（explicit.cu:681-682）。差异：

1. 初始 tile：`l = blockIdx.z`（CUTLASS：`initial_work_tile_info` 传入 `blockIdx.z`，sm100_tile_scheduler.hpp:370）。
2. CLC tile：`l_next = z0`。`get_first_ctaid` 的第三个输出被绑定到 `L_idx`（sm100_tile_scheduler.hpp:423-425）；在 SASS 中，这一提取写作 `x0 = word0`、`y0 = word1 & 0xffff`、`z0 = word1 >> 16`，这是一种从提取代码*推断*出来而非实测的 layout（笔记 D.2 的 `query_cancel` 一行）。L 不会被 AlongN 规则转置。
3. 把 `l` 作为 `crd2` 传给两个加载和那个存储，并在每个角色中把 `l_next` 与 `tm_next`/`tn_next` 一起携带。
4. explicit.cu:1268 处的主机端守卫（`params.l != kFixedL`）去掉；各 map 的 L stride 必须是 16 字节的倍数且小于 2^40（`cuTensorMapEncodeTiled`；零是被接受的：L = 1 的 map 传入字面量 0，笔记第 5.3 节）。

### 13.4.3 beta != 0：C 路径

这是最大的一处差异，因为它增加了一个 producer warp 和一条流水线。根据笔记第 7.2、7.8 节和 B.3.5 以及 CUTLASS 的 epilogue：

| 要素 | 新增内容 | 出处 |
|---|---|---|
| Warp 3 成为 C 的 producer | `griddepcontrol.wait`；每 tile 一次 `fetch_next_work`（它的 32 个线程现在参与 CLC empty 计数：**928** = 800 + 4 x 32，kernel :511-514）；在它的第一个 tile 上 `load_order_barrier.wait()`；然后 `load()`；循环之后 `load_tail` = 4 次 empty 等待 | kernel :807-867；sm100_epilogue_tma_warpspecialized.hpp:461-550 |
| `epi_load` 流水线（`PipelineTransactionAsync<4>`） | full 屏障位于 `OFF_EPI_LOAD_FULL`（128 + 8 idx），计数 **32 + 8192 tx**：被选中的 lane 发射四个 C box，然后 `mbarrier.expect_tx 8192`（不带 arrive），而全部 32 个 lane 执行 `arrive`；empty 屏障位于 `OFF_EPI_LOAD_EMPTY`（160 + 8 idx），计数 **128**（epilogue 的线程）；由 warp 1 初始化 | 笔记 7.2 的表、B.3.5；explicit.cu 在 :141-142 处预留了这些偏移 |
| `load_order` 屏障（`OrderedSequenceBarrier<1,2>`，位于 `OFF_LOAD_ORDER` 192 处的两个计数为 32 的屏障） | warp 2 到达一次，在它第一次（8 个 k-tile 的）`load_ktiles` 调用之后；warp 3 等待一次，在它第一次加载 C 之前，这样 A/B 的流量就在 C 的流量来争夺 TMA 单元之前开始（这一目的是我的分析） | kernel :649-652（`do_load_order_arrive`）、:831-834（`load_order_barrier.wait()` 在 :833 处）；被跳过的位置在 explicit.cu:790 处有标记 |
| C 的加载 | 每个 subtile e，加载到 D 存储将要使用的*同一个* smem stage 中（`ReuseSmemC`，即 builder 在 sm100_builder.inl:1109 处的 `ReuseSmem = sizeof_bits(C) > 8`：`smem_C`/`smem_D` 是一个联合体，sm100_epilogue_tma_warpspecialized.hpp:190-193；第 10 章）：四条 `cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`（`SM90_TMA_LOAD_3D`，copy_sm90_tma.hpp:181），box 为 (32, 16)，送往 `OFF_SMEM_CD + 8192 idx + 2048 i`，坐标为 `(128 tm + 32 i, 128 tn + 16 e, 0)`，在 `epi_load full[idx]` 上完成 | 笔记 B.3.5 |
| Epilogue consumer | 每个 subtile：等待 `epi_load full[idx]`；用 16 条 `ld.shared.b32` 从与 D 存储*相同的 swizzled 地址*（`st_off[...]`，explicit.cu:1010-1014）读取 C 片段；用真实的 c 计算 `d = fmaf(beta, c, alpha * acc)`（explicit.cu:1046 失去它的字面量 `0.0f`）；把 D 存入该 stage；晚一个 subtile 释放 `epi_load empty`，即在 `wait_group.read 1` 已经证明 D 存储读完该 stage 之后。这一滞后是 `UnacquiredStages = StagesD - 1 = 1`：在 `ReuseSmemC` 下，存储流水线是 `PipelineTmaStore<StagesC, StagesD - 1>`（sm100_epilogue_tma_warpspecialized.hpp:202-204、sm90_pipeline.hpp:653-660），它允许一个已提交的存储组在下一次 acquire 之前保持未被等待。这就是为什么 acquire 是 `wait_group.read 1` 而不是 `read 0`（第 10 章） | sm100_epilogue_tma_warpspecialized.hpp:791-795、839-858 |
| `store_tail` | `cp.async.bulk.wait_group.read 0`（所有存储都已读完 smem）和最后一次 `epi_load` 释放（`release_stages = min(UnacquiredStages, 8) = 1`，:1283） | sm100_epilogue_tma_warpspecialized.hpp:1269-1291；在 beta = 0 的路径中不存在（笔记 D.1 D15） |
| Prologue | 43 次初始化和 6 个 fence，而不是 33 和 4；`kClcEmptyCount` 928 | 笔记 E.10 第 6 项 |
| 主机端 | explicit.cu:1272 处的守卫去掉；`tma_c` 已经被携带并预取 | |

算术不变：baseline *总是*计算 `beta * C + t`；对于 beta = 0，C 片段只不过为零（笔记第 8 节）。这就是为什么 explicit kernel 即使从不加载 C 也保留了 `fmaf`。

当你添加 C 路径时，有一个后果很容易被忽略。在 CUTLASS 中，这一决定是在*运行时*、每个 CTA 一次、根据 beta 做出的（`is_producer_load_needed() = is_C_load_needed() = beta != 0`，笔记第 8 节）。它改变了一个 mbarrier 的*初始化计数*（CLC empty 计数，800 vs 928，kernel :511-514），也改变了 warp 3 的参与。因此，一个同时接纳两种 beta 的 explicit kernel 必须在 prologue 中、在 `fence.mbarrier_init.release.cluster` 和 cluster 会合之前，根据 `params.beta` 选择 `kClcEmptyCount`，以及采用 33 次初始化还是 43 次初始化的那一组。CUTLASS 的 PTX 正是这样做的，用一条基于 C 加载谓词的 `selp.b32 928, 800`（笔记 D.2）。

### 13.4.4 其他数据类型

**FP8（示例 70_blackwell_fp8_gemm.cu，同一目录）。** 该示例对 A、B、C 和 D 使用 `float_e4m3_t`，使用 `MmaTileShape_MNK = Shape<_256,_128,_64>`、`ClusterShape_MNK = Shape<_2,_2,_1>` 以及一个 `ScaledLinCombPerRowBiasEltActAmaxAux` fusion（70_blackwell_fp8_gemm.cu:101-156）。与本 kernel 相比：

| 条目 | FP16 kernel | FP8 示例 | 出处 |
|---|---|---|---|
| 指令 | `tcgen05.mma.cta_group::2.kind::f16` | `tcgen05.mma.cta_group::2.kind::f8f6f4`（`SM100_MMA_F8F6F4_2x1SM_SS`） | mma_sm100_umma.hpp:1463-1492 |
| 每条指令的 K | 16（每个操作数行的 32 B） | 32（仍是 32 B："K is expected to be 32 bytes wide for dense GEMM"，Colfax 的 TMEM 教程） | 推导 |
| 每个 64 宽 k-tile 的 MMA 数 | 4（`kKBlocks`） | 2；描述符低字中的 k-block stride 保持为 **+2**（32 B / 16 B），因为无论类型如何，一个 k-block 都是 32 字节 | explicit.cu:933-934 |
| 指令描述符 | `a_format = b_format = F16 = 0`，值为 0x10200010 | `MXF8F6F4Format` 枚举中 `E4M3 = 0`，所以*值*又是 0x10200010，在 `kind::f8f6f4` 下解释 | mma_sm100_desc.hpp:420-443 |
| Stage 字节数 | A 128 x 64 x 2 = 16384，B 64 x 64 x 2 = 8192 | A 8192，B 4096；每个 pair 的 tx 为 2 x (8192 + 4096) = 24576 | 推导 |
| Smem swizzle 原子 | `Layout_K_SW128_Atom<half_t>` = 8 行 x 64 个 fp16（128 B）：`BLK_K = 64` 是 64 的倍数 | `Layout_K_SW128_Atom<e4m3>` 是 8 行 x 128 个元素；64 % 128 != 0，所以选择器回退到 `Layout_K_SW64_Atom`（8 行 x 64 B）：描述符 `layout_type = SWIZZLE_64B (4)`、`SBO = 8 x 64 B = 512 B`（字段 32），stage 基址按 512 对齐，TMA map 为 `CU_TENSOR_MAP_SWIZZLE_64B`，box 为 (64, 64, 1) = 4096 B | sm100_common.inl:115-121；mma_traits_sm90_gmma.hpp:81-84；mma_sm100_desc.hpp:82-88 |
| Stage 数 | 8 | 用第 13.5 节的步骤 2 公式重新计算：`floor((232448 - carveout - epi_smem) / (8192 + 4096 + 16))`；epilogue 的 smem 有所不同，因为 8 位的 C 禁用了 `ReuseSmemC`（"8b residuals load fast and consume little smem"，sm100_builder.inl:1108-1109），所以 C 和 D 的 stage 是分开的 | sm100_umma_builder.inl:84-99 |
| Epilogue | fp32 的 D，每个 subtile 16 条 `FMUL` + 16 条 `FFMA`，(32, 16) 的 fp32 box | fp32 -> e4m3 转换，带 alpha/beta 缩放、逐行偏置、ReLU、一个 aux 输出和一次全局绝对值最大值归约：一种不同的设计，其 smem 原子、box 大小和屏障计数你用同样的方法推导 | 70_blackwell_fp8_gemm.cu:135-136 |

**TF32。** `tcgen05.mma.cta_group::2.kind::tf32`（`SM100_MMA_TF32_2x1SM_SS`，mma_sm100_umma.hpp:512-544）；`a_format = b_format = TF32 = 2` -> 位 [7,10) 和 [10,13) -> `idesc = 0x10200010 | (2 << 7) | (2 << 10) = 0x10200910`（根据 mma_sm100_desc.hpp:420-443 处的位图推导）；每条指令的 K 为 8（32 B / 4 B）。SW128 原子是 8 行 x 32 个 tf32，所以自然的 k-tile 是 32 个元素（128 B 的行），再次给出 128 x 32 x 4 = 16384 B 的 A stage。

**块缩放的 MXFP8/MXFP6/MXFP4 与 NVFP4。** 该指令为缩放因子增加了两个 TMEM 操作数：`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;`（mma_sm100_umma.hpp:1277），描述符变为带有 `scale_format`（第 23 位）、`a_sf_id`/`b_sf_id` 和 `k_size` 字段的 `InstrDescriptorBlockScaled`（mma_sm100_desc.hpp:446-470），而缩放因子在每条 MMA 之前经全局 -> smem (TMA) -> TMEM 传送。示例 72a/72b/72c（`72_blackwell_narrow_precision_gemm`）是 CUTLASS 的参考；用同样的十个步骤对它们去抽象化，并为缩放因子的拷贝在操作台账中增加第六行。

### 13.4.5 其他 tile shape

**每个 pair 256 x 256（CTA tile 128 x 256 x 64）。** 在第 13.6 节中有完整演算。一行概括：N = 256 -> `n_dim = 32` -> `idesc = 0x10400010`；每个 stage 256 个 TMEM 列 -> `min(4, 65536 / (128 x 256)) = 2` 个累加器 stage（sm100_umma_builder.inl:263-267）；B stage 128 x 64 x 2 = 16384 B；每个 pair 的 tx 为 65536；smem 预算给出 5 个 stage。

**128 x 64 的 CTA tile（MMA 256 x 64 x 16）。** `n_dim = 8` -> `idesc = 0x10100010`；每个 CTA 的 B stage 为 32 x 64 x 2 = 4096 B；每个 stage 64 个 TMEM 列，`min(4, 65536 / (128 x 64) = 8) = 4` 个 stage；epilogue tile (128, 16) -> 4 个 subtile。每条 MMA 每 smem 字节的 FLOP = 2 x 256 x 64 x 16 / (8192 + 2048) = 51.2，相对于 256 x 128 的 85.3（第 13.5 节步骤 2）：你用操作数带宽换取窄 N 上更好的 wave 量化。

### 13.4.6 1SM MMA（`cta_group::1`，cluster 1 x 1 x 1）

该指令是 `tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;`，带四个（而不是八个）禁用掩码寄存器，且 M 取自 {64, 128}（`SM100_MMA_F16BF16_SS`，mma_sm100_umma.hpp:86-120）。所有因 pair 而存在的东西都消失了：

| 条目 | 2SM kernel | 1SM，cluster (1,1,1)，CTA tile 128 x 128 x 64 |
|---|---|---|
| 由谁发射 MMA | pair leader 为两个 CTA 发射 | 每个 CTA 各自发射；`is_leader` 恒为真，`peer == rank`（kernel :427-430） |
| 每个 CTA 的 A/B | A 128 x 64（一半来自 multicast），B 64 x 64 | A 128 x 64 和 B 128 x 64 都由该 CTA 自己加载：每个 stage 的 tx = 16384 + 16384 = 32768，一个计数为 1 的屏障 |
| TMA 操作 | `SM100_TMA_2SM_LOAD[_MULTICAST]_3D`，带 `cta_group::2` 和经 peer 位掩码的屏障 | `SM90_TMA_LOAD_3D`：`cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`（copy_sm90_tma.hpp:181），本地屏障，无掩码 |
| Mainloop 的 empty | 计数 2，`tcgen05.commit ... cta_group::2 ... multicast::cluster` 掩码 0xF | 计数 1（1/1 + 1/1 - 1），`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];`（barrier.h:797-801），无 multicast |
| 累加器 | full 1（pair 掩码的 commit），empty 256（peer 位 arrive） | full 1（本地 commit），empty 128（普通的本地 arrive） |
| TMEM | 两个 CTA 中都有 `tcgen05.alloc.cta_group::2`，dealloc 握手；4 个累加器 stage | `tcgen05.alloc.cta_group::1`（tmem_allocator_sm100.hpp:80），之后仍跟着 `.cta_group::1` 形式的 `relinquish_alloc_permit` 和 `dealloc`；没有握手：在非重叠累加器策略下，当 `has_mma_peer_cta` 为假时，`tmem_dealloc` 屏障甚至不会被初始化（kernel :563-565）。Stage 数不变：对 MMA M = 128，builder 走同一个分支，`min(4, 65536 / (128 x 128)) = 4`；只有对 MMA M = 64，它才使用 `(512 / CtaN) x 2`（sm100_umma_builder.inl:263-267） |
| CLC | empty 800 | 32 + 1 x (32 + 128 + 32) = 224（cluster 大小为 1） |
| Smem 寻址 | `smem_base & kPeerBitMask`、到 rank 0 的 `mapa` | 全部本地；`x = y = 0` |
| 每条 MMA 中，TMA 送入 smem 的每个操作数字节对应的 FLOP（tensor core 的读取：第 3.3.1 节，未验证） | 85.3（256 x 128 x 16） | 64（128 x 128 x 16）：每个 SM 现在为自己的 128 行持有全部的 B |

最后一行就是 2SM *为什么*存在的原因：pair 共享 B（每个 CTA 加载一半），并且借助 2 x 2 的 cluster 共享 A（每个 CTA 加载 pair 的 A 行的四分之一并 multicast）。你在简单性上有所失，便在操作数带宽上有所得（第 13.5 节步骤 2）。

来自源材料的两点注意事项。(1) "All tcgen05 instructions within a kernel must specify the same value for the .cta_group qualifier"（PTX ISA 9.4，在 tcgen05.alloc 和 tcgen05.commit 之下陈述），所以一个 1SM kernel 不能保留任何 `::2` 形式，包括 commit 操作、alloc 和 dealloc。(2) 当选用 CUTLASS 的 `IsOverlappingAccum` 策略时（不是本 kernel 的策略），`tmem_dealloc` 屏障*会*在没有 peer 的情况下被初始化，计数为 `NumEpilogueThreads`（kernel :567-573），因为那时由 epilogue 的 warp 释放该分配。如果你移植那个策略，握手会以另一种形态回来。

### 13.4.7 其他 cluster shape（2SM 原子，cluster (X, Y, 1)）

三个 CUTLASS 函数产生这三个掩码，两处屏障初始化产生这两个计数；请针对你的 shape 手工实例化它们：

- **Mainloop release 掩码** = 每一个满足 `x' / 2 == x / 2` *或* `y' == y` 的 CTA (x', y')（`calculate_multicast_mask<kRowCol>`，sm100_pipeline.hpp:57-91）。对于 (2, 2)：全部四个，0xF。对于 (2, 4)：0xFF。
- **Mainloop empty 计数** = `X / 2 + Y - 1`（sm100_pipeline.hpp:565-566）：其掩码包含本 CTA 的 leader 的数目。(2, 2) -> 2；(2, 4) -> 4；(4, 2) -> 3。
- **累加器 commit 掩码** = 该 pair，`0x3 << (rank & ~1)`（`calculate_umma_peer_mask`，sm100_pipeline.hpp:96-110）。
- **A multicast 掩码** = 具有相同 x 的那些 CTA：对 y' = 0..Y-1 的位 `x + X y'`（`create_tma_multicast_mask<2>`）；(2, 2) -> `0x5 << x`；(2, 4) -> `0x55 << x`。于是每个 CTA 加载 128 x 64 的 A stage 中的 128 / Y 行（TMA box 除以 multicast 计数，笔记第 5.3 节），所以 (2, 4) 的 A box 是 (64, 32, 1) = 4096 B。每个 CTA 最终仍得到一个完整的 16384 字节的 A stage，并且该 pair 的事务计数保持为 2 x (16384 + 8192) = 49152（推导所得；此处未演练）。
- **B multicast** 在 `X / 2 > 1` 时出现：具有相同 y 的各 pair 共享 `tn`（由同一规则推导；box 与掩码遵循上面的模式，此处未演练）。
- **CLC empty 计数** = `32 + X Y (32 + 128 + 32)`，其中 warp 3 空闲，当 beta != 0 时再加 `32 X Y`（C-producer warp 也消费；(2, 2) 时为 928，kernel :511-514）；响应被 multicast 到所有 CTA，并由 scheduler warp 的 lane `0..XY-1` 装填它们。

在动代码之前先把这些行写进屏障表；它们当中的每一行一旦写错都是一次挂起（第 13.3 节，第 3a-6b 行）。

### 13.4.8 Stream-K 或 split-K

示例 74（`74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu:167`）把 scheduler 标签换成了 `cutlass::gemm::StreamKScheduler`。用本 kernel 的术语来说，差异是：

1. 一个工作单元是 `(tm, tn, k_begin, k_count)`，而不是 `(tm, tn)`：producer 的 `st.q = 0` (explicit.cu:779) 和 `k_tiles` (explicit.cu:1112) 变成由 scheduler 给出的每单元的值（`get_k_tile_iterator`、`get_work_k_tile_count`，kernel :627-628）；在每个单元的开始处 `scale_c = 0`，与今天一样。
2. **fixup 放在哪里：** 在累加器等待与 epilogue 的第一条 `tcgen05.ld` 之间 (explicit.cu:1030-1036)。对于 TMEM 累加器，CUTLASS 的 `fixup`（sm100_tile_scheduler_stream_k.hpp:648-728）调用 `tmem_fixup`：加载 TMEM -> 寄存器；通过一个由 CTA 间屏障守卫的全局工作区，跨各个分片归约部分和（`reduction_mode` 为 Deterministic 或 Nondeterministic）；然后要么写 D（拥有 epilogue 的那个分片），要么存储部分和并跳过 epilogue（`compute_epilogue == false`）。工作区指针进入 `ExplicitGemmParams`，而 `requires_fixup` 给 epilogue 角色增加一个分支。
3. 对于完整的 tile，CLC 消费不变。一个 stream-K 单元可以跨越多个 tile 而不消费新的响应（`increment_pipe == false`，kernel :708-714 处的注释），所以 `clc_consume` 变为有条件的，而 800 这一计数按*响应*计，而不是按 tile 计。

### 13.4.9 Swizzled 光栅化 (`max_swizzle_size` = S > 0)

当 S = 0（本次运行）时，一个 CLC id L 就是 cluster tile `(L / 32, L % 32)`，并且连续的 id 沿 N 行进（笔记 D.3）。当 S > 0 时，`swizzle_and_rasterize`（sm100_tile_scheduler.hpp:697-814）在 AlongN 转置之前，在每个 M x S 的 cluster 块内重映射 *cluster* 坐标：

```
new_major = m / S + (M / S) * (n % S)
new_minor = m % S + (n / S) * S
```
（第 788-789 行，其中 `major` 是网格 x 方向的 cluster 坐标，`M` 是沿该方向的 cluster 数目）。M 或 N 不能被 S 整除的余数区域中的 cluster 不被重映射（:783 处对 :775-779 处算出的首个余数 id 所做的测试）。源码注释中描述的蛇形翻转没有实现（笔记第 13 节第 12 项）。效果：以近似启动线性的顺序发放的 id（实测，笔记 D.3）现在填满一条 S 宽的条带：在 AlongN 转置之后，S 个连续的 id 共享相同的 B 列并分布在 S 个相邻的 A 行条带上，所以一行 M 个 id 在 L2 中触及 S 个 A 行条带和 M / S 个 B 列条带，而不是一个和 M 个（由上面的公式推导；此处未演练）。在 explicit.cu 中，两处 tile 公式（:691-694 和 :1175-1176）加上这一重映射，`swizzle_divisor` 来自参数块（今天它必须为 0，explicit.cu:1269），其余一切不变。

---

## 13.5 SOL 设计配方

这份配方是九个编号的步骤。每一步都有其公式和本 kernel 的数字，以便你能看到这一方法产生出你刚刚学习过的设计。

### 步骤 1. 在层次结构的每一级对算子做 roofline

公式：FLOP = 2 M N K；唯一 HBM 字节数 = bytes(A) + bytes(B) + bytes(C) + bytes(D)；强度 I_level = FLOP / 在该级搬运的字节数；时间界限 = max(FLOP / peak, bytes / bandwidth)。

| 层级 | 固定运行所搬运的字节数 | FLOP / 字节 | 备注 |
|---|---|---|---|
| HBM（唯一） | A 128 MiB + B 128 MiB + D 256 MiB = 536,870,912 | 2048 | 按 7.7 TB/s（Lenovo B200 180 GB 产品指南；其他 NVIDIA 资料中为 8 TB/s）计，访存时间约为 70 us；按 2.25 稠密 PFLOPS（同一来源）计，tensor 时间为 489 us：计算受限达 7x |
| L2 -> SM | 每 cluster 每 k-tile：A 4 x 8192 + B 4 x 8192 = 65536 B 的*读取*（得益于 multicast 和 2SM 拆分，每个唯一字节只读一次），对应 2 x 256 x 256 x 64 FLOP | 128 | 每个 tile 的时间至多 27.75 us（CUTLASS baseline 实测的 0.888 ms / 32 个 tile，笔记 D.4；第 3.9.2 节）-> 每 k-tile 216.8 ns -> 聚合 L2 读取需求 65536 x 33 / 216.8 ns = 10 TB/s（我的分析；B200 的 L2 带宽不在 NVIDIA 的公开文档中：NVIDIA 内部信息给出 GPU 范围约 23.9 TB/s 的读带宽、每分区 12.0 TB/s，以及 GPU 范围约 10.8 TB/s 的写带宽，所以这一需求约为读带宽的 42 %） |
| smem -> tensor core | 每 CTA 每 k-tile 由 TMA 送达 16384 (A) + 8192 (B) = 24576 B（tensor core 至少读取这些，第 3.3.1 节），对应 2 x 128 x 128 x 64 FLOP | 85.3 | 等于 256 x 128 x 16 指令的每 smem 字节 FLOP 数（步骤 2） |
| TMEM -> 寄存器 | 每 CTA tile 128 x 128 x 4 = 65536 B，由 32 条 `tcgen05.ld` 读取一次 | 4096 | 累加器由 512 条 MMA 就地写入并读取一次 |
| 寄存器 -> smem -> HBM | 每 CTA tile 65536 B，经由每线程 128 条 `st.shared` 和 32 次 TMA 存储 | 4096 | |

用于对比的实测结果：explicit kernel 的中位数 0.889354 ms 即 2 x 8192^3 / 0.889354 ms = 1.2363e15 FLOP/s，在 1965 MHz 下为 148-SM 峰值的 51.9 % 和 132-SM 峰值的 58.2 %。在 mainloop 期间，一条 256 x 128 x 16 MMA 至多有 106.5 个周期（426 周期 k-tile 预算的四分之一，笔记 D.4；按 explicit kernel 自己的中位数约为 107），相对于每 SM 每时钟 8192 FLOP 稠密峰值下 64 的理想值，即在它所占用的那些 SM 上约为 tensor pipe 峰值的 60 %（第 3 章，第 3.9.1-3.9.2 节）。无论是 CUTLASS 的 kernel 还是这一个，都没有达到 SOL；步骤 9 告诉你该往哪里看。

### 步骤 2. 根据 TMEM 与 smem 预算以及操作数带宽约束，选择 MMA 指令与 CTA/cluster tile

公式：

- 操作数为 b 字节的 M x N x K 指令的每 smem 字节 FLOP 数 = 2 M N K / ((M + N) K b) = 2 M N / ((M + N) b)，其中保存在 smem 中的每个操作数字节只计一次（这是 tensor core 读取量的一个下界，第 2.5.3 节）。
- 对于 fp32 累加器，TMEM 累加器 stage 数 = min(4, 128 x 512 / (CtaM x CtaN))（sm100_umma_builder.inl:263-267）。
- Smem stage 数 = floor((232448 - carveout - epilogue_smem) / (bytes(A stage) + bytes(B stage) + 16))（sm100_umma_builder.inl:84-99；笔记第 6.1 节）。

| 指令（fp16，fp32 累加器） | FLOP | 每条指令的 smem 字节数（保存的操作数字节；这是 tensor core 读取量的一个下界，因为 cta_group::2 并不意味着 B 只读一次，NVIDIA 内部；第 2.5.3 节） | FLOP / smem 字节 | 理想速率下每 SM 每时钟的 smem 字节数（8192 FLOP/clk/SM，第 3 章） |
|---|---|---|---|---|
| 1SM 128 x 128 x 16 | 524,288 | 4096 + 4096 = 8192 | 64 | 128 B / clk |
| 1SM 128 x 256 x 16 | 1,048,576 | 4096 + 8192 = 12288 | 85.3 | 128 个周期内 96 B / clk |
| 2SM 256 x 128 x 16（本 kernel） | 1,048,576 | 8192 + 4096 = 12288，即每 SM 6144 | 85.3 | 96 B / clk |
| 2SM 256 x 256 x 16 | 2,097,152 | 8192 + 8192 = 16384，即每 SM 8192 | 128 | 128 个周期内 64 B / clk |

这些周期数假定步骤 1 的 8192 FLOP/clk/SM 峰值（第 3 章），并显示出这一取舍：256 x 128 指令每 SM 每时钟需要来自共享内存的 96 字节操作数，256 x 256 的那条则为 64。相对于 SemiAnalysis 所报告的每 SM 128 B/clk 的共享内存带宽（第 3 章；并非 NVIDIA 的规格），96 B/clk 的操作数读取在计入 TMA 对同一内存的写入之前就已经是该端口的四分之三。更宽的 N 把每 FLOP 的操作数压力削减了三分之一（96 -> 64 B/clk）。代价是 TMEM 和 smem：在 TMEM 中，每 stage 256 列只留下 2 个累加器 stage 而不是 4 个；在 smem 中，一个 16 KiB 的 B stage 和更大的 (128, 32) epilogue 缓冲区只留下 5 个 mainloop stage 而不是 8 个（第 13.6 节）。

对本 kernel，预算给出：8 个 mainloop stage = floor((232448 - 192 - 33792) / 24592)（笔记第 6.1 节）；4 个累加器 stage = min(4, 65536 / 16384)；epilogue tile (128, 16) 来自 builder 针对 CtaM > 64 且 CtaN <= 128 的 32 位 C/D 的规则 `N_perf = 16`（sm100_builder.inl:1025）。

### 步骤 3. 选择 cluster shape 与 multicast，以达到 L2 -> SM 带宽目标

公式：每 SM 每 k-tile 的 L2 读取字节数 = (每 CTA 的 A 字节数) / (共享相同 A 行的 CTA 数) + (每 CTA 的 B 字节数) / (共享相同 B 列的 CTA 数)。目标：在理想 k-tile 时间 t_ideal（这里为 4 条 MMA x 64 周期 = 256 周期）下，每 SM 的 L2 读取需求 = 字节数 / t_ideal 必须远低于 L2 的每 SM 份额。

| 设计 | 每 SM 每 k-tile 的 L2 读取 | 在 256 周期下 | 1965 MHz 下 148 个 SM 的聚合 |
|---|---|---|---|
| 1SM，无 cluster | 16384 + 16384 = 32768 B | 128 B/clk | 37 TB/s |
| 2SM pair，无 multicast | 16384 + 8192 = 24576 B | 96 B/clk | 28 TB/s |
| 2SM pair + 跨 y 的 A multicast（本 kernel） | 8192 + 8192 = 16384 B | 64 B/clk | 18.6 TB/s |
| 2SM 256 x 256 pair + A multicast（第 13.6 节） | 8192 + 16384 = 24576 B，对应两倍的 FLOP | 512 周期下 48 B/clk | 14 TB/s |

（全部由字节数推导；聚合数字是 L2 在 SOL 下必须维持的量，我的分析。）相对于 GPU 范围约 23.9 TB/s 的 L2 读带宽（NVIDIA 内部），前两行超出了它，而本 kernel 这一行需要它的 78 %。multicast 在指令数上是免费的（无论如何每 k-tile 每操作数一条 TMA），只花费第 4 章，第 4.4.2 节的屏障协议复杂性：mainloop empty 屏障的计数 2 和 `tcgen05.commit` 的 release 掩码 0xF（按 peer 位加掩码的 full 屏障和 pair 范围的事务计数随 2SM pair 本身而来，无论是否 multicast）。

### 步骤 4. 用 Little 定律选择流水线深度

公式：在途字节数 = 送达带宽 x 延迟，所以 stage 数 >= ceil(latency / t_ktile)（如果你希望 producer 永远不在 peek 上阻塞，则 +1）。对于累加器流水线：stage 数 >= ceil(t_epilogue / t_mainloop_tile) + 1，受 TMEM 上限约束。

对本 kernel，mainloop 深度就是简单的 "all the smem that is left"：8 个 stage x 24576 B = 192 KiB 在途。这在实测的 k-tile 预算下隐藏了至多 8 x 426 = 3408 个周期（1.73 us）的 TMA 往返，或者 8 x 256 = 2048 个理想周期（1.04 us）（第 3 章，第 3.5.1 节）。只要共享内存没有其他用途，这一策略就是对的：更深的环除了通过 carveout 之外绝不会有害，而且它吸收一个有 33 个 cluster 同时全都从 L2 加载的持久化 kernel 将会看到的延迟*方差*。累加器深度 4 是 TMEM 的上限。一个 tile 的 epilogue（8 个 subtile，每个一条 `tcgen05.ld`、32 次 FP 运算、16 条 `st.shared`、4 次 TMA 存储）远短于一个 27.75 us 的 mainloop tile，所以即使 2 个 stage 也不会使 MMA warp 停顿（第 3 章，第 3.5.2 节；我的分析；第 13.6 节依赖于它）。

### 步骤 5. 为每个异步引擎分配一个 warp，并为每个配一个单一的发射线程

| 引擎 | Warp | 发射线程 | 指令 | 为什么只用一个线程 |
|---|---|---|---|---|
| TMA 加载单元 | 2 | 每个 A + B 对一次 `elect_one_sync()` (explicit.cu:740) | `cp.async.bulk.tensor ... cta_group::2` | 每个操作数一次描述符驱动的拷贝；搬运字节的是该单元，而不是线程 |
| Tensor core（UMMA） | pair leader 的 0 号 | 每条 MMA 和每次 commit 一次 `elect_one_sync()` (explicit.cu:932, 942, 944) | `tcgen05.mma.cta_group::2`、`tcgen05.commit` | MMA 是单线程指令；commit 必须来自发射这些 MMA 的同一线程（笔记第 7.6 节） |
| TMEM 读取 + TMA 存储 | 4-7（全部 128 个线程读取；warp 4 存储） | `tcgen05.ld` 是 warp 范围的（`.sync.aligned`）；TMA 存储由 warp 4 的全部 32 个 lane 以 uniform 操作数发射，ptxas 把它变成每 box 一条 `UTMASTG`（笔记第 13 节第 8 项） | `tcgen05.ld.32x32b.x16`、`cp.async.bulk.tensor ... bulk_group` | 数据通路是每条指令 32 个 lane x 16 列；4 个 warp 覆盖 128 行 |
| CLC（硬件 scheduler） | CTA 0 的 1 号 | 每次查询一次 `elect_one_sync()` (explicit.cu:843-847) | `clusterlaunchcontrol.try_cancel ... multicast::cluster::all` | "only one thread per cluster should be used"（Colfax CLC 文章）；多于一个线程会取消多于一个 cluster |

空闲的 warp（rank 1-3 上是 1 号，各处都是 3 号）除了 32 个（rank 0）或 64 个（rank 1-3）线程的寄存器堆之外不花费任何东西，并且使 warp 索引与 CUTLASS 的保持一致，从而使 prologue 和命名屏障的参与者集合（160 = warp 0 + warp 4-7）得以匹配。

### 步骤 6. 在写任何代码之前先写出屏障协议表

各列：流水线 | stage 数 | producer（谁，指令） | consumer（谁，指令） | full 计数（+ tx） | empty 计数 | parity 规则 | 屏障归谁 | 初始化 warp。本 kernel 的实例：

| 流水线 | St. | Producer | Consumer | Full | Empty | Parity 规则（第 n 次使用） | 屏障归谁 | 初始化 |
|---|---|---|---|---|---|---|---|---|
| Mainloop A/B | 8 | warp 2（`expect_tx` 由 leader lane 执行；TMA 由被选中的 lane 执行） | leader 的 warp 0（`try_wait`；由掩码为 0xF 的 `tcgen05.commit` 释放） | 1 + 49152 | 2 | consumer 为 `(n / 8) & 1`，producer 为 `1 ^` 该值 | **leader 的** full（peer 位掩码）；自己的 empty | 0 |
| 累加器 | 4 | leader 的 warp 0（`tcgen05.commit`，掩码 `0x3 << 2y`） | 两个 CTA 的 warp 4-7（在加了掩码的地址上执行 `mbarrier.arrive.shared::cluster`） | 1 | 256 | n = tile | **自己的** full（leader 的单次 commit 操作带着掩码 `0x3 << 2y` 被 multicast，落在 pair 中两个 CTA 的同一偏移上；每个 CTA 的 epilogue 在它自己的 `acc_full[st]` 上等待：explicit.cu:902, 944, 1016, 1030）；**leader 的** empty（两个 CTA 的每个 epilogue 线程都通过 peer 位掩码地址到达，:1039） | 5 |
| CLC | 2 | CTA 0 的 warp 1（`try_cancel`；lane 0..3 以 `expect_tx 16` 装填） | 所有 CTA 的 warp 0、2、4-7 + CTA 0 的 warp 1 | 1 + 16 | 800 | n = tile | 自己的 full；**CTA 0 的** empty | 4 |
| Throttle | 2 | CTA 0 的 warp 2（`arrive`，32 个 lane） | CTA 0 的 warp 1（远程向 CTA 0 执行 `arrive`） | 32 | 32 | n = tile | CTA 0 | 3 |
| TMEM dealloc | 1 | peer 的 warp 0 | leader 的 warp 0，然后反过来 | 32 | - | parity 0，一次 | 两者 | 0 |

这张表强制执行的两个习惯：(a) 每个计数都是*到达方*的数目，一条 `tcgen05.commit` 对每个发射线程、每个被掩码覆盖的 CTA 计为一个到达方，而逐线程的 `mbarrier.arrive` 对每个线程计为一个；(b) 一个会接收远程到达的屏障必须由一个尾段（步骤 7）保持存活。

### 步骤 7. 写出尾段与退出条件

列出每一个可能接收远程到达的屏障，并写出等待最后一次到达的尾段：这里是笔记第 7.10 节的那四个。显式地决定异步*存储*的退出条件。本 kernel 退出时至多有一个 bulk 存储组的 smem 读取尚未被等待（最后一个 subtile 之后的 `wait_group.read 1`，没有 `read 0`）；这是 baseline 自己的行为，并且从未观察到失败（笔记 D.1 D15）。替代实现可以收紧这一点，但绝不能放松它（笔记第 7.7 节与 D.7）。

### 步骤 8. 做结构验证与数值验证

结构上：第 13.2 节步骤 8。数值上：harness 中取值在 [-8, 8] 内的整数数据使每一种求和顺序都是精确的（笔记第 8 节），所以还要运行非整数数据（具有完整尾数的随机 fp16），并与参考 kernel 的输出做*逐位*比较。最后一位的差异意味着不同的 MMA 序列、不同的 `scale_c` 放置位置或改变了的 epilogue 算术。比较位模式而不是数值，以捕捉零的符号。

### 步骤 9. 测量，用 profiler 指标归因差距，在设计表上迭代

归因：用 Nsight Compute 读取 tensor-pipe 活跃比例（`sm__pipe_tensor_cycles_active`，即 UMMA 数学的度量，旁边还有范围更广的 TC pipe 的 `sm__pipe_tc_cycles_active`，以及 `sm__pipe_tensor_op_hmma_cycles_active`，一个 tensor-pipe 活跃指标；NVIDIA 内部，第 12.7.5 节）、L2 吞吐量以及*按角色 warp*的 warp 状态采样（指标名称随版本变化；从你的 `ncu --query-metrics` 中获取它们）。决策表：

| 观察 | 含义 | 设计表的改动 |
|---|---|---|
| MMA warp 大多在 `full[s]` 上等待（`SYNCS.PHASECHK ... TRYWAIT` 停顿），tensor pipe 空闲 | 受操作数投递限制 | 更宽的指令 N（步骤 2）、更多 multicast（步骤 3）、更深的环（步骤 4） |
| Tensor pipe 接近峰值，MMA warp 很少等待 | 在被占用的 SM 上受计算限制 | 修复 occupancy：这里有 16 个空闲 SM（33 个 cluster x 4 = 132）；对余下部分采用不同的 cluster shape 或 1SM tile |
| Epilogue warp 在 `acc_full` 上等待，MMA warp 在 `acc_empty` 上等待 | 相对累加器深度而言 epilogue 太慢 | 更多 TMEM stage（更窄的 N）或更廉价的 epilogue |
| 每个 tile 的尾段很长，每个 cluster 的 tile 数不均 | wave 量化（这里是 32 对 31 个 tile，3 %） | 末尾用更小的 tile，stream-K（第 13.4.8 节） |
| 主机端可见的 kernel 之间的间隙 | 启动与编码开销 | 经由 `griddepcontrol` 的 PDL 已经发出 |

在*表*上迭代，而不是在代码上：指令 shape 或 cluster shape 的一次改动会一次性重写每一个计数、掩码、偏移和坐标，而第 13.2 节的八个步骤在一天内就能重新跑完（我的估计），因为每一步都是可测试的。

---

## 13.6 算例：256 x 256 pair-tile FP16 GEMM

在设计表层面把这套配方应用到同一个 8192^3 问题上，采用 `MmaTileShape (256, 256, 64)` 和 cluster (2,2,1)：CTA tile 为 128 (M) x 256 (N) x 64 (K)。除非另有标注，下面的一切都是用上面的公式推导出来的。

### Tile 表

| 量 | 值 | 推导 |
|---|---|---|
| MMA 指令 | `tcgen05.mma.cta_group::2.kind::f16`，256 x 256 x 16 | N <= 256，N % 16 == 0 (mma_sm100_umma.hpp:554-555) |
| 指令描述符 | `0x10400010` | `c_format 1 << 4`、`n_dim (256 >> 3) = 32 << 17 = 0x400000`、`m_dim 16 << 24` (mma_sm100_desc.hpp:420-443) |
| 每个 k-tile 的 MMA 数 | 4 (K = 64 / 16) | 不变 |
| CTA tile / cluster tile | 128 x 256 x 64 / 256 x 512 | cluster (2,2,1) |
| Cluster tile 数 | 32 x 16 = 512 | 8192 / 256 乘以 8192 / 512 |
| 网格 | (32, 64, 1) | `ctas_n = round_up(ceil_div(8192, 256), 2) = 32`、`ctas_m = round_up(ceil_div(8192, 256) * 2, 2) = 64`，AlongN 转置（笔记第 7.4 节） |
| Tile 公式 | 不变：`tm = 2 (by >> 1) + (bx & 1)`、`tn = 2 (bx >> 1) + (by & 1)`；CLC `tm = 2 (y0 >> 1) + (x0 & 1) + x`、`tn = 2 (x0 >> 1) + (y0 & 1) + y` | tn 现在以 256 列为单位 |
| A 的坐标 / box / 掩码 | `(64 q, 128 tm + 64 y, 0)`、(64, 64, 1) = 8192 B、`0x5 << x`，目的地 `OFF_SMEM_A + 16384 s + 8192 y` | 不变 |
| B 的坐标 / box | `(64 q, 256 tn + 128 x, 0)`、一个 (64, 128, 1) box = 16384 B（或两个 (64, 64, 1) box），目的地 `OFF_SMEM_B + 16384 s`，无 multicast | 每个 CTA 加载 pair 的 256 个 N 列中属于它的 128 个 |
| 每个 cluster 的 tile 数 | 512 / 33 = 15.5 -> 16 或 15 | 与 32 / 31 相同的 3 % 量化 |

### Smem 映射

| 偏移 | 字节数 | 内容 | 为什么在这里 |
|---|---|---|---|
| 0 | 40 | mainloop full[5] | 5 个 stage（见下） |
| 40 | 40 | mainloop empty[5] | |
| 80 | 16 + 16 | CLC full[2]、empty[2] | |
| 112 | 16 + 16 | acc full[2]、empty[2] | 2 个 TMEM stage |
| 144 | 16 + 16 | throttle full[2]、empty[2] | |
| 176 | 8 | tmem_dealloc | |
| 192 | 32 | clc_response[2]（按 16 对齐） | |
| 224 | 4 | tmem_base_ptr | |
| 512 | 4 x 16384 = 65536 | C/D stage，(128, 32) fp32 subtile，`Swizzle<2,5,2>` | 按 512 对齐；StagesC = 4（32 位 C 的 `ReuseSmem`：`max(min(8, 4), 3)`，sm100_builder.inl:1113-1115） |
| 66560 | 5 x 16384 = 81920 | A stage | 位于 66048 处或之后的第一个按 1024 对齐的偏移：描述符的 `base_offset` 被硬编码为 0 (mma_sm100_desc.hpp:757-758)，所以 SW128 的基址*必须*按 1 KiB 对齐（我的设计选择；CUTLASS 自己针对这一 shape 的结构体填充未经验证） |
| 148480 | 5 x 16384 = 81920 | B stage | 148480 = 145 x 1024 |
| 230400 | | 末尾 | <= 232448；巧合的是，与 128 x 128 kernel 的总量相同 |

用 builder 的公式检查 stage 数：carveout = 32（acc 流水线，2 个 stage）+ 32（CLC）+ 16（load order）+ 8（dealloc）+ 32（throttle）+ 32（响应）+ 8（TMEM 基址指针）= 160 (sm100_pipeline_carveout.inl:45-83)；epilogue smem = 65536 + 512（fusion 填充）+ 64（加载流水线）按 512 取整 = 66560；stage 数 = floor((232448 - 160 - 66560) / (16384 + 16384 + 16)) = floor(165728 / 32784) = 5。第六个 stage 大约差 31 KB：按 builder 的公式，6 x 32784 = 196704 对比可用的 165728（30976 B）；或者按手工 layout，66560 + 6 x 32768 = 263168 对比 232448（30720 B）；五个 stage 之后，按 builder 的公式还剩 1808 B（手工 layout 中为 2048 B，232448 - 230400）。

关于这一 carveout 和映射，有三件事容易弄错。(a) `SchedulerPipelineStageCount` 对稠密 GEMM 是编译期常量 2（只有分组与 ragged-contiguous 分组 GEMM 才是 8，sm100_umma_builder.inl:282），所以 CLC 流水线（32 B）、throttle（32 B）、响应（2 x 16）和 TMEM 基址指针（2 x 4）都与 tile shape 无关；只有 `AccumulatorPipelineStorage`（每个 stage 16 B）随累加器 stage 数变动，这里是 64 -> 32。(b) 上面的映射去掉了 explicit.cu 保留着的 beta != 0 槽位（`OFF_EPI_LOAD_FULL`/`EMPTY` 128/160 和 `OFF_LOAD_ORDER` 192，explicit.cu:141-143）。它们随第 13.4.3 节的增量（2 x 32 + 16 字节）一起回来，而 160 这个 carveout 数字为了与 CUTLASS 保持一致仍然计入 16 字节的 load-order 屏障。stage 数对此并不在意：165728 和 165744 都给出 5。(c) 新的 B stage 必须通过 TMA 编码器的限制。(64, 128, 1) 的 fp16 box 沿内维是 128 B，共 128 行。这 128 B 等于 SW128 的跨度，并且是一个 128B swizzle 的 map 所允许的最大值（CUDA 驱动 API 的 `cuTensorMapEncodeTiled` 契约："CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128" 字节，在第 3 章中引用过）。这 128 行处于 CUTLASS 所断言的每维 256 的 box 限制之内（copy_traits_sm90_tma.hpp:1005-1013，`smem_box_shape[i] <= 256`）。UMMA 描述符的 SBO 保持为 1024 B，因为无论 stage 容纳多少行，SW128 原子都是 8 行 x 128 B，所以高字 `0x40004040` 不变。

针对实测基址 0x400 的描述符：`desc_lo_a = 0x10000 | ((0x400 + 66560) >> 4) = 0x10000 | 0x1080 = 0x11080`、`desc_lo_b = 0x10000 | ((0x400 + 148480) >> 4) = 0x10000 | 0x2480 = 0x12480`；两者每个 stage（16384 B）推进 +1024，每个 k-block 推进 +2；高字保持为 `0x40004040`（SBO 1024 B，SWIZZLE_128B）。14 位字段是安全的：`(0x400 + 230400) >> 4 = 0x3880 < 0x4000`。

### TMEM 映射

| 项目 | 值 |
|---|---|
| 分配 | 512 列（`tcgen05.alloc.cta_group::2`），基址 T |
| 累加器 stage st | 列 `[256 st, 256 st + 256)`，lane 0..127；2 个 stage |
| MMA 的 D 操作数 | `T + 256 st` |
| Epilogue 加载 | 在 `T + ((32 w) << 16) + 256 st + 32 e` 处执行 `tcgen05.ld.sync.aligned.32x32b.x32.b32`，e = 0..7，每个线程每个 subtile 32 个寄存器（或两次 `.x16` 加载） |

### 屏障表

| 流水线 | St. | Full | Empty | Parity 规则（第 n 次使用） | 屏障归谁 | 初始化 warp | 相对 128 x 128 kernel 的变化 |
|---|---|---|---|---|---|---|---|
| Mainloop | 5 | 1 + **65536** tx（2 x (16384 + 16384)：落在 pair 中的四个 8192 的 A 半块和两个 16384 的 B 半块） | 2，释放掩码 0xF | consumer 为 `(n / 5) & 1`，producer 为 `1 ^` 该值 | leader 的 full（peer 位掩码）；自己的 empty | 0 | tx 与 stage 数；128 % 10 != 0，所以各 tile 不再从同一个 stage 和 phase 开始：状态*必须*跨 tile 携带（它已经如此） |
| 累加器 | **2** | 1，commit 掩码 `0x3 << 2y` | 256 | n = tile：`(n / 2) & 1` | 自己的 full（向 pair multicast 的 commit 操作）；leader 的 empty（peer 位掩码） | 5 | 深度；tile t + 2 的 MMA 要等待 tile t 的 epilogue |
| CLC | 2 | 1 + 16 | 800 | n = tile：`(n / 2) & 1` | 自己的 full；CTA 0 的 empty | 4 | 无 |
| Throttle | 2 | 32 | 32 | n = tile：`(n / 2) & 1` | CTA 0（两者） | 3 | 无 |
| TMEM dealloc | 1 | 32 | - | parity 0，一次 | 两者（每个 CTA 自己的，由对方装填） | 0 | 无 |

每个 subtile e（0..7，32 列）的 epilogue：`tcgen05.ld` x32 -> 32 条 `FMUL` + 32 条 `FFMA` -> 32 条 `st.shared.b32`，地址为 `OFF_SMEM_CD + 16384 (e % 4) + 4096 w + 128 j + 4 ((l & 7) + 8 (((l >> 3) & 3) ^ (j & 3)))`（字节偏移上的同一个 `Swizzle<2,5,2>`：位 [7,9) 是 `j & 3`，因为 `128 j` 保留了它们，位 [5,7) 是 `(l >> 3) & 3`）-> `fence.proxy.async` -> `bar.sync 1, 128` -> 四次 TMA 存储，(32, 32, 1) = 4096 B 的 box，从 `+ 4096 i` 到 `(128 tm + 32 i, 256 tn + 32 e, 0)` -> `commit_group` -> `wait_group.read 1` -> `bar.sync 1, 128`。尾段：5 + 2 + 2 + 1 次等待。

### 预期的 SOL 比例

| 指标 | 128 x 128 kernel（实际构建的版本；只有最后两行建立在实测之上） | 256 x 256 设计（推导） |
|---|---|---|
| TMA 每向一个 SM 的 smem 投递一个操作数字节所对应的 FLOP 数；这是 smem 读取的一个下界，因为 `cta_group::2` 并不意味着 B 只读一次（NVIDIA 内部），第 2.5.3 节 | 85.3 | 128（每 FLOP 的操作数流量少 1.5x） |
| 每个 SM 每个 k-tile 的 L2 读取量 / 每 FLOP | 16384 B / 2.1 MFLOP = 128 FLOP/B | 24576 B / 4.2 MFLOP = 171 FLOP/B |
| 理想 k-tile 时间（8192 FLOP/clk/SM，第 3 章） | 256 个周期 | 512 个周期 |
| 环的覆盖 | 8 x 256 = 2048 个理想周期 | 5 x 512 = 2560 个理想周期（尽管 stage 更少，但在时间上更深） |
| 累加器余量 | 3 个 tile | 1 个 tile（epilogue 必须在一个 mainloop tile 之内完成；按第 3.5.2 节未经实测的估计，0.52 us 到悲观的 11-15 us，对比 27.75 us，它做得到，而且该设计把每个 CTA tile 的 epilogue 字节数和理想 tile 时间都加倍了） |
| Wave 量化 | 32 / 31.03 | 16 / 15.52，两者都是 97 % |
| 实测 / 预期 | 每个活跃 SM 的 tensor 峰值的 60 % | 如果这 40 % 的差距是操作数投递（MMA warp 在 `full[s]` 上等待），这个设计能收回其中一部分；如果它是 tensor pipe 自身在存在依赖关系的累加上的发射速率，它什么也收不回。步骤 9 的归因在你写代码之前就做出判定（假设；CUTLASS profiler 今天就能运行这一配置）。 |

---

# 第 14 章 练习与术语表

**你将学到什么**

- 一张数字卡片：没有任何正文章节以表格列出的字节与操作总数以及实测结果；常量、偏移、掩码、描述符和坐标则留在它们各自所属的章节中（第 5 章、第 8 章和第 10 章）。

本章的每一个答案中所使用的出处标签：

| 标签 | 含义 |
|---|---|
| **[S]** | 源自源码：`explicit.cu`、`explicit.hpp`、`util.hpp`，或提交 cb424739 处的 CUTLASS 头文件（以 file:line 形式引用） |
| **[M]** | 在 B200 上实测，取自笔记的 D 部分和 G 部分（引用所在节） |
| **（NVIDIA 内部）** | 未公开记载的 NVIDIA 工程信息，仅在不存在公开来源之处使用 |
| **[A]** | 我的分析或假设：由 [S]/[M] 事实经推理得出，其本身未经运行验证 |
| **[S/A]** | 一个源码事实加上我对其后果的推断（句子会说明哪一半是哪一半） |
| **[M/A]** | 一个实测数字加上我对它的算术运算（例如由 `Avg runtime` 导出的速率） |

---

## 14.1 如何使用本章

在每个等级内部按顺序做练习；各等级之间可以自由交错。RECALL 练习检查 kernel 的常量及其理由是否已在你的脑中（第 4 章、第 5 章）。DERIVE 练习让你针对具体的 `(rank, tile, q, s, kb, w, l, j, e)` 值手工运行 kernel 的地址与 parity 算术（第 6-10 章）。MODIFY 练习问的是当某一行改变时，什么会出错、挂起，或仅仅变慢（第 4 章、第 7-11 章、第 13 章）。DESIGN 练习在不同的 shape、cluster 或 epilogue 上重新运行第 3 章 / 第 13 章的方法。我用一个脚本按从 `explicit.cu` 引述的公式重新计算了每个数值答案；凡存在公式行之处，答案都按行号引用它，并且要么引述它，要么点名引述它的那一节。

如果你能不打开文件就做完所有 DERIVE 练习，并且在做所有 MODIFY 练习时对*哪一个*屏障计数会改变毫不犹豫，那么你就已准备好从零开始编写这个 kernel（第 13 章）。

---

## 14.2 练习

### 14.2.1 RECALL

**R2. 说出 kernel 所初始化的每一个屏障的到达计数以及（如适用）事务字节计数，并说明每个计数为什么是那个值。**

<details><summary>答案</summary>

常量位于 `explicit.cu:180-185` [S]，引述于第 5.9 节。

| 屏障（x 槽位数） | 计数 | tx 字节 | 为什么 [S，笔记 7.2] |
|---|---|---|---|
| mainloop full[8] | 1 | 49152 | 由 pair leader 的 warp 2 中被选中的 lane 执行一次 `mbarrier.arrive.expect_tx`（`explicit.cu:736`）。这些字节是每个 stage 落入该 pair 的两个 CTA 的六个 8192 字节 TMA box：四个 A box（4 个 CTA 各发射一个，multicast 到 2 个 CTA，所以每个 pair 收到 4 个）和两个 B box；2 x (16384 + 8192) = 49152。peer 的 box 通过经 peer 位掩码处理的地址计入 leader 的屏障。 |
| mainloop empty[8] | 2 | - | 每个 k-tile 有两条掩码为 0xF 的 `tcgen05.commit ... multicast::cluster` 到达每个 CTA，每个 pair leader 各一条。`explicit.cu:182` 处的注释把该计数写成 `2/2 + 2/1 - 1` = (cluster M / 原子 M) + (cluster N / 原子 N) - 1；CUTLASS 源码是 `PipelineTmaUmmaAsync::init_barriers`（`include/cutlass/pipeline/sm100_pipeline.hpp:559-573`，公式位于 `:565-566`：`multicast_consumer_arrival_count = (size<0>(cluster_shape) / size<0>(atom_thr_shape)) + (size<1>(cluster_shape) / size<1>(atom_thr_shape)) - 1`，其中 cluster 为 (2,2,1)，原子为 (2,1,1)）。两个 leader 都必须释放，因为 CTA x 把它的 A 半块 multicast 到 CTA x+2 的 smem 中，而它被*另一个* pair 的 MMA 读取。 |
| CLC full[2] | 1 | 16 | 每个 CTA 一次 `arrive.expect_tx`（rank 0 的 warp 1 的 lane 0..3 远程装填 CTA 0..3，`explicit.cu:842`）；16 字节的 `try_cancel` 响应被 multicast 到所有 CTA，并完成每个 CTA 自己的 full 屏障。 |
| CLC empty[2] | 800 | - | 每个消费线程都远程到达 CTA 0 的屏障：32（rank 0 warp 1）+ 4 CTA x (32 warp 0 + 32 warp 2 + 128 warp 4-7) = 800。当 beta != 0 时，warp 3 加入，CUTLASS kernel 的计数变为 928（kernel :511-514）；explicit kernel 硬编码了 800（`explicit.cu:183`）并拒绝 beta != 0（`explicit.cu:1272`）。 |
| 累加器 full[4] | 1 | - | 由 leader 以 pair 掩码 `0x3 << 2y` 执行一次 `tcgen05.commit`（`explicit.cu:944`）；它到达该 pair 的两个 CTA。 |
| 累加器 empty[4] | 256 | - | 两个 CTA 的每个 epilogue 线程（2 x 128）都经由 peer 位掩码到达 LEADER 的屏障（`explicit.cu:1039`）；peer 的 acc_empty 屏障被初始化但从未使用。 |
| throttle full[2] / empty[2] | 32 / 32 | - | 在 tile 开始时由 rank 0 的 warp 2 执行整个 warp 的本地 arrive / 在消费完 throttle 之后由 rank 0 的 warp 1 执行整个 warp 的远程 arrive；仅限 rank 0。 |
| tmem_dealloc | 32 | - | 一次整个 warp 的远程 arrive（32 个 lane），先由 peer 的 warp 0 执行，然后由 leader 的 warp 0 执行，发生在退出握手中（`explicit.cu:964, 970`）。 |

总计：8+8+2+2+4+4+2+2+1 = 33 次 `mbarrier.init` [M，笔记 G.3：33 条 `SYNCS.EXCH.64`]。

为什么事务屏障能容忍字节在 `expect_tx` *之前*到达 [S, PTX ISA 9.4]：producer 的顺序是等待 `empty[s]` -> 由 leader lane 执行 `mbarrier.arrive.expect_tx 49152` -> 两次 TMA 发射（`explicit.cu:735-755`）。但 peer 运行的是一个独立的 warp，所以落入 peer 的三个 box，甚至 leader 自己的那些，都可能在 leader lane 的 `expect_tx` 执行之前完成。三条 ISA 规则使之安全。(1) tx-count 是一个有符号计数器，范围为 `-(2^20 - 1) .. 2^20 - 1`（第 9.7.15.16.3 节，表 43），所以 `complete-tx` 操作可以在匹配的 `expect-tx` 把期望的字节加回来之前把它驱动为负（此处最低到 -49152，远在范围之内）（9.7.15.16.6.1 "expect-tx operation ... increases the tx-count"，9.7.15.16.6.2 "complete-tx ... tx-count is decremented"）。(2) 一个 phase 只有在 "the count of the pending arrivals has reached zero" 并且 "the tx-count has reached zero" 时才完成（9.7.15.16.8）。(3) 那唯一一次到达（计数 1）与字节期望融合在 `arrive.expect_tx` 中，所以在 `expect_tx` 执行之前待到达计数不可能降到零，无论那 48 KiB 的完成以何种顺序到来。同样的论证也覆盖 CLC 的 `full[]`（计数 1 + 16 字节）：来自 rank 0 的 lane 0..3 的远程 `arrive.expect_tx` 与 multicast 响应的 16 字节 `complete_tx` 可以以任一顺序到达。当到达与期望是两个分开的操作时，如在 C 加载流水线中那样（G2：一次单独的 `expect_tx`，32 次普通 arrive），同样的规则只要求最后一次到达不先于 `expect_tx`。CUTLASS 把它们排成 TMA 发射 -> `expect_tx`（一个线程）-> 在那一个发射 warp 内的 32 次 arrive（`sm100_epilogue_tma_warpspecialized.hpp:528-541`），所以字节可以提前落地（tx-count 为负），但 phase 不可能在期望被登记之前关闭。
</details>

**R7. 给出 UMMA 共享内存描述符和指令描述符的常量，并解码它们的字段。**

<details><summary>答案</summary>

`explicit.cu:171-173` [S]，引述于第 5.8 节。

smem 描述符遵循 `include/cute/arch/mma_sm100_desc.hpp:101-127` 处的位图 [S]。位 [0,14)：以 16 字节为单位的起始地址，`(addr >> 4) & 0x3FFF`，在 `:755` 处由 `static_cast<uint16_t>(start_address >> 4)` 赋值；该转换丢弃了 rank 位，所以 leader 和 peer 构成相同的描述符。位 [16,30)：前导字节偏移 = 1（16 B）。对于 K-major 的 swizzled layout，ISA 说前导维度的 stride 是 "not used, assumed to be 1"，而它的示例表给出 "Encoding of LBO in descriptor: 1 (assumed)"（PTX ISA 9.4 第 9.7.18.3.1.1 节 "Relative offset mode" 以及第 9.7.18.3.3 节 "Canonical Layouts" 的示例表；字段位置为表 49 中的位 16-30，第 9.7.18.4.1 节）。CUTLASS 对每一种 swizzled K-major 规范 layout 都把它固定为 1（`mma_sm100_desc.hpp:822-832`：`static_assert(stride_10 == expected_stride_10)`，其中 `expected_stride_10 = 1`，然后 `desc.leading_byte_offset_ = stride_10`）。位 [32,46)：stride 字节偏移 = 64（1024 B，一个 128 字节行的 SW128 原子中相邻两个 8 行组之间的距离）。位 [46,48)：版本 = 1。位 [61,64)：layout 类型 = 2 = SWIZZLE_128B。高字：`64 | (1 << 14) | (2 << 29)` = 0x40004040（`static_assert` 位于 `explicit.cu:175`）。指令描述符（`mma_sm100_desc.hpp:416-443` [S]）：位 [4,6) c_format = 1（F32）-> 0x10；a/b 格式为 0（F16）；a/b major 为 0（K-major）；[17,23) n_dim = 128 >> 3 = 16 -> 0x200000；[24,29) m_dim = 256 >> 4 = 16 -> 0x10000000；总计 0x10200010（`static_assert` 位于 `explicit.cu:176`）。实测的首条 MMA 的值为 `desc_a = 0x4000404000010880`、`desc_b = 0x4000404000012880`、`idesc = 0x10200010` [M, D.1 D7]。
</details>

**R11. 一位同事对整个二进制文件运行 `cuobjdump --dump-sass`，并报告说“它有 40 条 `UTMALDG` 和 151 条 `LDS`，explicit kernel 太臃肿了”。哪里出了错，正确的数字是什么？**

<details><summary>答案</summary>

这位同事统计的是整个二进制文件，其中仍然包含从未被启动的 CUTLASS kernel（未改动的 `gemm.initialize()` 对它构成 odr-use）：40 = 36 + 4，151 = 140 + 11（我对 `trace_out_v2/sass.txt`、`sass_gemm.txt`、`sass_explicit.txt` 的重新计数）。在 CUTLASS kernel 的 36 条 `UTMALDG` 中，32 条是其已编译但死的 C 路径的普通 `UTMALDG.3D`；另外 4 条是它自己在 prologue 和主循环中的 A/B 对。在它的 140 条 `LDS` 中，128 条是死的 C 读取。按 `Function :` 头把 explicit kernel 切出来（第 12.7.2 节中的那条 `awk`）：2 条 `UTMALDG.3D.MULTICAST.2CTA` + 2 条 `UTMALDG.3D.2CTA`，11 条 `LDS`（4 条 `LDS.128` CLC 解码 + 7 次对 `tmem_base_ptr` 和保留的 alloc 字的读取），总共 1640 条指令，相对于 CUTLASS kernel 的 3224 条。
</details>

### 14.2.2 DERIVE

**D1. 计算 rank 2、stage s = 5、k-block kb = 3 时的 `desc_a` 和 `desc_b`。**

<details><summary>答案</summary>

公式 [S]，`explicit.cu:896-897`（引述于第 9.4.4 节）、`928-929`、`933-934`（引述于第 9.5 节）。

rank 2：`smem_base = 0x02000400`。A：`0x02000400 + 33792 = 0x02008800`；`>> 4` = 0x200880；`& 0x3FFF` = 0x0880（rank 位从 14 位字段中掉出）；`| 0x10000` = 0x10880。stage 5：+ 5 x 1024 = 0x11C80；kb 3：+ 6 = 0x11C86。**`desc_a = 0x4000404000011C86`**。B：`0x02000400 + 164864 = 0x02028800`；`>> 4 & 0x3FFF` = 0x2880；`| 0x10000` = 0x12880；+ 5 x 512 = 0x13280；+ 6 = 0x13286。**`desc_b = 0x4000404000013286`**。与文件自己的例子（`explicit.cu:170`）交叉核对：rank 0、s = 3、kb = 2 给出 0x10880 + 3072 + 4 = 0x11484 和 0x12880 + 1536 + 4 = 0x12E84，与 `0x4000404000011484` / `0x4000404000012E84` 相符。对于相同的 (s, kb)，rank 2 的描述符等于 rank 0 的 [M，D.1 D7：rank 0 与 rank 2 相同]，正如 2SM MMA 所要求的：它在相同的 CTA 相对地址处读取 peer 的半块。最大的低字是 `0x10880 + 7 x 1024 + 6 = 0x12486`（A）和 `0x12880 + 7 x 512 + 6 = 0x13686`（B）。这两次加法是对一个字的普通 32 位加法，该字的位 [0,14) 是起始地址字段，[14,16) 未使用（第 15 位在 ISA 中是保留的；第 5.8.1 节），而 [16,30) 保存 LBO = 1。从该字段进位出去会破坏位 [14,16) 或 LBO，所以这种构造之所以合法，只是因为该字段不可能溢出：最大的字段值是 `0x13686 & 0x3FFF = 0x3686 < 0x3FFF`，所以永远不会产生进位。`explicit.cu:165` 处的 `static_assert`（`kSmemBaseLow24 + kSmemBytes < (1u << 18)`）就是对整个缓冲区的这一保证：`0x400 + 230400 = 0x38800 < 2^18` 字节，所以动态 smem 内每一个以 16 字节为单位的地址都放得进 14 位字段。`:896-897` 处的 `& 0x3FFF` 剥去了 `smem_base` 的 rank 位 [24,26)（以及第 17 位以上的一切），所以 per-stage 加法从一个与 rank 无关的字开始。
</details>

**D2. warp 2 在其第 4 个 tile（tile 计数器 t = 3）的 q = 20 处等待的是哪个屏障地址和 parity，而 MMA warp 对同一个 k-tile 等待的又是哪个？**

<details><summary>答案</summary>

[S] producer 状态跨 tile 传递（`explicit.cu:724`），而每个 tile 128 个 k-tile 是 8 的倍数，所以全局使用计数是 `n = 128 t + q = 404`。Stage `s = n % 8 = 4`；stage 4 的更早使用次数：`n / 8 = 50`；producer parity = `1 ^ (50 & 1) = 1`（第 4.2.3 节）。地址：`empty_base + 8 s = smem_base + OFF_MAIN_EMPTY + 32 = smem_base + 96`，即 rank 0 为 `0x00000460`，rank 1 为 `0x01000460`，rank 2 为 `0x02000460`，rank 3 为 `0x03000460`。每个 CTA 等待自己的 empty 屏障；两个 leader 的掩码为 0xF 的 commit 完成全部四个。在代码中，这次等待是提前一个 k-tile 发射的 peek `mbar_try_wait(empty_base + 8u * st.prod_idx, st.prod_phase)`（`explicit.cu:739`），加上仅当 peek 失败时才对相同操作数执行的阻塞 `EXPLICIT_WAIT`（`explicit.cu:735`）。含义：parity 1 在 phase 49（第 50 次完成，亦即对使用 49 = 本 tile 的 q = 12 的释放）完成后成功，也就是在两个 leader 的那些为 q = 12 读取 stage 4 的 MMA 都已完成之后（每个 leader 的 `tcgen05.commit` 在那时到达，而不是在发射时；第 4.2.2 节）。

leader 的 MMA warp 以 consumer parity `50 & 1 = 0` 等待位于 `smem_base + 32` 的 `full[4]`：phase 50 必须完成，即第 51 次填充（这个 k-tile 的六个 box 加上 leader lane 的 `expect_tx`）必须已经落地。阻塞等待是 `explicit.cu:923`（`if (!tok) { EXPLICIT_WAIT(full_base + 8u * cons_idx, cons_phase, WAIT_MAINLOOP_FULL, ...) }`）。它通常已由 `:916`（tile 的第一个 k-tile）或 `:927`（之后的每个 k-tile）处的 peek 提前满足；`:924` 把 stage 索引捕获为 `rs`。
</details>

**D3. 对于包含 CTA tile (12, 7) 的 cluster tile，在 q = 100 处，给出每个 rank 的 TMA 坐标、目的地、multicast 掩码和屏障操作数。**

<details><summary>答案</summary>

CTA tile (12, 7) 属于一个特定的 CTA，因为对一个 CTA 收到的每个 tile 都有 `tm % 2 == x` 和 `tn % 2 == y`（`explicit.cu:777` 处的注释陈述了前者；第 8.5.4 节从 `explicit.cu:1175-1176` 和 `:692-693` 证明了两者）。所以 (12, 7) 由处理 cluster tile (6, 3) 的那个 cluster 中 x = 0、y = 1 的 CTA（rank 2）计算。该 cluster 的四个 CTA 持有 (12 + x, 6 + y)。公式 [S]，`explicit.cu:777-778`、`768`（引述于第 8.6 节）、`747-748`、`754-755`（引述于第 8.4 节）。

`c0 = 64 x 100 = 6400`；stage `s = 100 % 8 = 4`（tile 的起点是 stage 对齐的）；A 的目的偏移 `33792 + 16384 x 4 + 8192 y = 99328 + 8192 y`；B 的目的偏移 `164864 + 8192 x 4 = 197632`；屏障 `(smem_base + 8 x 4) & 0xFEFFFFFF`。

| rank (x,y) | CTA tile | A box (k, row, 0) = 行 | 落地偏移，所在 rank（地址） | 掩码 | B box (k, col, 0) = 列 | B 位于 | 屏障 |
|---|---|---|---|---|---|---|---|
| 0 (0,0) | (12,6) | (6400, 1536, 0) = 1536..1599 | 99328，在 rank 0、2 中 (0x00018800, 0x02018800) | 0x5 | (6400, 768, 0) = 768..831 | 0x00030800 | 0x00000420 |
| 1 (1,0) | (13,6) | (6400, 1664, 0) = 1664..1727 | 99328，在 rank 1、3 中 (0x01018800, 0x03018800) | 0xA | (6400, 832, 0) = 832..895 | 0x01030800 | 0x00000420 |
| 2 (0,1) | (12,7) | (6400, 1600, 0) = 1600..1663 | 107520，在 rank 0、2 中 (0x0001A800, 0x0201A800) | 0x5 | (6400, 896, 0) = 896..959 | 0x02030800 | 0x02000420 |
| 3 (1,1) | (13,7) | (6400, 1728, 0) = 1728..1791 | 107520，在 rank 1、3 中 (0x0101A800, 0x0301A800) | 0xA | (6400, 960, 0) = 960..1023 | 0x03030800 | 0x02000420 |

解读该表：rank 0 和 2 一起在它们两者的 A stage 中都装配出 tm = 12 的 128 行 1536..1663（行 1536..1599 位于 +0，行 1600..1663 位于 +8192）；rank 1 和 3 对 tm = 13 做同样的事。pair {0,1} 持有 B 的列 768..895 = tn = 6 的 128 列（每个 CTA 64 列）；pair {2,3} 持有 tn = 7。屏障操作数是 pair leader 的 `full[4]`（rank 0/1 -> rank 0 的，rank 2/3 -> rank 2 的），所以每个 leader 的屏障收到落在 leader 中的 2 个 A box + 落在 peer 中的 2 个 + 2 个 B box = 6 x 8192 = 49152 字节。文件在 `explicit.cu:718-719` 处的例子具有相同的结构（tile (5, 9)，q = 7，rank 1：A (448, 640, 0) -> 0x01024800 和 0x03024800；B (448, 1216, 0) -> 0x01036800），已重新计算并确认（指地址；输入本身并不是一个可达的状态，第 8.7.1 节）。
</details>

**D4. 计算 epilogue 线程 (w = 2, l = 17) 的 `st_off[0..3]`，以及 subtile e = 5 的值 j = 11 所写入的 smem 字节。它落入哪个 stage 和哪个存储 box？**

<details><summary>答案</summary>

[S] `explicit.cu:1013` 和 `1047`：

```
    st_off[c] = OFF_SMEM_CD + 2048u * w + 4u * (l & 7u) + 128u * c + 32u * (((l >> 3) & 3u) ^ c);
```
```
        *reinterpret_cast<uint32_t*>(smem + st_off[j & 3] + 512u * uint32_t(j >> 2) + kEpiStageBytes * uint32_t(e & 3)) = __float_as_uint(d);   // 16 scalar st.shared.b32 (B.3.3 step 17)
```

w = 2，l = 17：`l & 7 = 1`，`(l >> 3) & 3 = 2`。`st_off[c] = 512 + 4096 + 4 + 128 c + 32 (2 ^ c)`：c = 0：4612 + 64 = **4676**；c = 1：4612 + 128 + 96 = **4836**；c = 2：4612 + 256 + 0 = **4868**；c = 3：4612 + 384 + 32 = **5028**。对于 j = 11，e = 5：`j & 3 = 3`，`j >> 2 = 2`，`e & 3 = 1`：字节 = 5028 + 1024 + 8192 = **14244 = 0x37A4**。Stage `e % 4 = 1`（字节 8704..16895）；在该 stage 内偏移为 14244 - 8704 = 5540，box `i = 5540 / 2048 = 2 = w`（box i 总是等于 w，因为 subtile 的第 32 w .. 32 w + 31 行就是 box w）。用 unswizzled 形式交叉核对：swizzle 前的字节 = `512 + 8192 x 1 + 2048 x 2 + 128 x 11 + 4 x 17 = 14276 = 0x37C4`；`epi_swz`（`explicit.cu:197`：`o ^ ((o & 0x180u) >> 2)`）：`0x37C4 & 0x180 = 0x180`，`>> 2 = 0x60`，`0x37C4 ^ 0x60 = 0x37A4`。两种形式一致，正如 `explicit.cu:199-202` 处的各条 `static_assert` 对照 `cute::Swizzle<2,5,2>` 所检查的那样。文件自己的算例（`explicit.cu:992`：w = 0，l = 9，j = 5，e = 6 -> 17540）重新计算为 `st_off[1] = 644`，`+ 512 + 16384 = 17540`。所写入的行是 CTA tile 的第 32 w + l = 81 行，列为 16 e + j = 91。这个 XOR 究竟为什么会在这里 [S/A]：对于一条 `st.shared.b32` 指令（j 和 e 固定），32 个 lane 写入 `4 (l & 7) + 32 (((l >> 3) & 3) ^ (j & 3))` 加上一个 warp-uniform 的基址，也就是说，以一种依赖于 lane 但为双射的顺序写入一个 128 字节行的四个 32 字节 chunk：全部 32 个 bank 恰好各一次。因此无论有没有这个 XOR——它只是在同一个 128 字节行内置换各个 chunk（`j & 3` 是 warp-uniform 的）——该存储都是无冲突的。kernel 必须计算这个 XOR，是因为 D 的 tensor map 以 `SWIZZLE_128B_ATOM_32B` 编码 [M, D.1 D3]：TMA 单元在读取时做反 swizzle，所以 smem 镜像必须处于 swizzled layout，和 A/B 的情况完全一样。区别在于，对 A/B 而言是 TMA 单元在写入时做 swizzle、UMMA 单元在读取时做反 swizzle，所以那里 kernel 什么也不计算（`explicit.cu:203`）。swizzle 由 tensor map 决定；而无冲突的 STS 模式是 (32, 16) subtile 映射的一个性质，碰巧在两种情况下都成立。
</details>

**D5. warp 6 在累加器 stage st = 2、subtile e = 4 处从哪个 TMEM 地址加载，lane 3 又收到什么？**

<details><summary>答案</summary>

[S] `explicit.cu:1006`、`1031`、`1036`：

```
  uint32_t const tmem_lane_off = (32u * w) << 16;                                               // TMEM lanes [32 w, 32 w + 32): row 32 w + l of the CTA tile (0, 0x200000, 0x400000, 0x600000)
```
```
    uint32_t const taddr = T + tmem_lane_off + 128u * acc_stage;                                // warp-uniform TMEM address of this warp's 32 rows
```
```
      tmem_ld_32x32b_x16(taddr + 16u * uint32_t(e), r);                                        // columns 16e..16e+15 of row 32 w + l (:881-883)
```

warp 6 是 epilogue warp w = 2；T = 0 [M]。`tmem_lane_off = 64 << 16 = 0x400000`；`128 x 2 = 256 = 0x100`；`16 x 4 = 64 = 0x40`；地址 **0x00400140** = lane 64，列 320。`32x32b.x16` 这一 shape 使 lane l 得到 TMEM lane（行）`64 + l` 的 16 个连续 32 位列：lane 3 以 `r[0..15]` 的形式收到 CTA tile 的第 67 行、累加器列 64..79（stage 2 的绝对 TMEM 列 320..335）。warp 6 只能寻址 lane 64..95（warp id % 4 = 2）[B.6；Colfax TMEM 教程]。文件自己的例子（`explicit.cu:982`：w = 2，st = 1，e = 3 -> 0x4000B0）重新计算为 0x400000 + 128 + 48 = 0x4000B0。
</details>

**D6. 计算 CTA tile (5, 9) 中由线程 (w = 1, l = 20)、寄存器 j = 9、subtile e = 2 写入的 D 元素的全局字节地址；然后给出它经过的 smem 字节以及承载它的 TMA 存储 box。**

<details><summary>答案</summary>

[S] 行 `m = 128 tm + 32 w + l = 640 + 32 + 20 = 692`；列 `n = 128 tn + 16 e + j = 1152 + 32 + 9 = 1193`。D 是 fp32、M-major（列主序）的，前导维度为 8192（每列 stride 32768 B，D.1 D3），所以相对 D 基址的字节偏移是 `(n x 8192 + m) x 4 = (9773056 + 692) x 4 = 39094992 = 0x2548AD0`。Smem：(w = 1, l = 20) 的 `st_off[1]` = `512 + 2048 + 16 + 128 + 32 x (2 ^ 1 = 3) = 2800`；字节 = `2800 + 512 x 2 + 8192 x 2 = 20208 = 0x4EF0`（stage 2，box 1）。核对：swizzle 前 `512 + 16384 + 2048 + 128 x 9 + 80 = 20176 = 0x4ED0`，`epi_swz` 与 0x20 做 XOR -> 0x4EF0。box 坐标在 `explicit.cu:1056-1057`，存储本身在 `:1066`（两者都引述于第 10.10 节）。

box i = 1 的坐标为 (672, 1184, 0)，源为 `cd_base + 8192 x 2 + 2048 x 1`；该元素位于 32 x 16 box 内的 (20, 9) 处；TMA 单元在读取时做反 swizzle（SWIZZLE_128B_ATOM_32B），并向 D 写入 16 段各 128 个连续字节（每列一段）。
</details>

**D7. 一个 cluster 处理 T = 29 个 tile。给出三个尾段中每一次等待的索引与 parity。**

<details><summary>答案</summary>

[S] 每个尾段都在 N 个 producer 步骤之后继续 producer 的状态。一个尾段步骤所执行的两行代码，对 mainloop 而言是（`explicit.cu:800` 与 `:805`）：

```
    EXPLICIT_WAIT(empty_base + 8u * st.prod_idx, st.prod_phase, TAIL_MAINLOOP, tile, uint32_t(i));
```
```
    advance<kStages>(st.prod_idx, st.prod_phase);
```

CLC 尾段也是同样的两行（`:860-861` 以 `clp_phase` 在 `smem_base + OFF_CLC_EMPTY + 8u * clp_idx` 上先测试再等待，然后 `:866` `advance<kClcStages>(clp_idx, clp_phase);`），累加器尾段亦然（`:955` 以 `accp_phase` 等待 `acc_empty_base + 8u * accp_idx`，然后 `:960` `advance<kAccStages>(accp_idx, accp_phase);`）。`advance`（`:612`）递增索引并在每次回绕时翻转 phase。producer 从 `{0, 1}` 出发，已经走了 N 步。因此尾段的第 i 步以 parity `1 ^ (((N + i) / Stages) & 1)` 等待 `empty[(N + i) % Stages]`（这是笔记 7.2 为 `producer_tail` 推导出的公式；尾段位于 `explicit.cu:799-806`、`859-867`、`954-961`）。Mainloop：N = 128 x 29 = 3712，Stages 为 8：索引 0..7，八次全部 parity 为 `1 ^ ((464 + 0..0) & 1) = 1`（3712 / 8 = 464 是偶数；一般地，128 T / 8 = 16 T 总是偶数，所以对每个 T，mainloop 尾段的 parity 都是 1 [M, G.1：每个 K_TAIL mainloop 步等待的都是 parity 1]）。累加器：N = 29，Stages 为 4：i = 0..3 -> 索引 (29, 30, 31, 32) % 4 = **1, 2, 3, 0**，parity 为 `1 ^ ((29/4 = 7) & 1) = 0`、`1 ^ (7 & 1) = 0`、`1 ^ (7 & 1) = 0`、`1 ^ ((32/4 = 8) & 1) = 1` -> **0, 0, 0, 1**。CLC（仅 rank 0）：N = 29，Stages 为 2：索引 (29, 30) % 2 = **1, 0**，parity 为 `1 ^ ((14) & 1) = 1`、`1 ^ ((15) & 1) = 0` -> **1, 0**。对照实测的 tile 数做合理性检查 [M, G.1]：T = 32 给出累加器索引 0,1,2,3 全部 parity 1，CLC 0,1 的 parity 为 1,1；T = 31 给出 3,0,1,2 配 0,1,1,1，CLC 1,0 配 0,1 —— 恰好就是记录下来的那些 `{idx, phase}` 对。这就是为什么尾段必须继续状态，而不是对所有屏障套用同一个 parity。
</details>

**D9. 首 CTA id 为 (x0, y0) = (18, 44) 的 CLC 响应把哪个 CTA tile 交给 rank 3，它的 A 行和 D 列又是什么？**

<details><summary>答案</summary>

[S] 该映射是 `explicit.cu:692-693`（引述于第 7.4 节，推导于第 7.5 节）。

首 CTA id 总是偶数（cluster 对齐）[M, D.3]，所以 `x0 & 1 = y0 & 1 = 0`。rank 3 的 (x, y) = (1, 1)：`tm = 2 x 22 + 0 + 1 = 45`，`tn = 2 x 9 + 0 + 1 = 19`。被取消的 cluster 的启动线性 id 是 `(y0 / 2) x 32 + x0 / 2 = 22 x 32 + 9 = 713` = 以 256 为单位的坐标中的 cluster tile (m, n) = (22, 9) [M, D.3]。A 行 `[128 x 45, +128) = [5760, 5888)`；D 列 `[128 x 19, +128) = [2432, 2560)`。这个转置（网格 x 索引 N，网格 y 索引 M）就是 AlongN 光栅顺序。文件的例子（`explicit.cu:672`：CTA (1, 0) 中的 (10, 6) -> (7, 10)）重新计算为 `2 x 3 + 0 + 1 = 7`、`2 x 5 + 0 + 0 = 10`。
</details>

**D10. 启动坐标是 blockIdx = (41, 26)。该 CTA 是哪个 rank，它的 pair leader 和 A-multicast 伙伴是谁，它的初始 tile 是什么？**

<details><summary>答案</summary>

[S] `x = blockIdx.x & 1 = 1`，`y = blockIdx.y & 1 = 0`：cluster 在网格坐标中是 2 x 2 的，所以 `%cluster_ctaid = (blockIdx.x % 2, blockIdx.y % 2)`（`explicit.cu:1173-1174` 读取它们并注释为 `== rank & 1`、`== rank >> 1`）。Rank `r = x + 2y = 1` 是奇数，所以这个 CTA 是 peer；它的 pair leader 是 rank 0（`peer = rank ^ 1`，`explicit.cu:1107`）；它的 A-multicast 伙伴是 rank x + 2 = 3（掩码 0x5 << 1 = 0xA）。初始 tile（`explicit.cu:1175-1176`，引述于第 11.6 节）：

`tm = 2 x 13 + 1 = 27`，`tn = 2 x 20 + 0 = 40`：D 的行 [3456, 3584) 与列 [5120, 5248)。它所在的 cluster (CX, CY) = (20, 13) 的启动线性 id 为 13 x 32 + 20 = 436。在实测运行中只有 id 为 0..32 的 33 个 cluster 是原生驻留的 [M, D.3]；id 436 不在其中，所以这个 CTA 从不带着这个 tile 运行：CLC 转而把该 tile 交给一个驻留的 cluster。文件的例子（`explicit.cu:1093`：blockIdx (5, 2) -> (3, 4)）重新计算为 `2 x 1 + 1`、`2 x 2 + 0`。
</details>

**D11. 解释共享内存中未使用的 2048 字节，以及为什么让 kernel 使用它们所需要的不只是改一个常量。**

<details><summary>答案</summary>

232448 - 230400 = 2048 = 1728（198464 / 24592 向下取整除法的余数）+ 320。这 320 是 builder 的估计与真实 layout 之间的差距。builder 计入 192（carveout）+ 33792（epilogue）+ 8 x 24592（各 stage，每个带 16 B 的屏障存储）= 230720。真实的 `SharedStorage` 是 512（头部：43 个屏障槽位、CLC 槽位、TMEM 指针，用了 388 B）+ 33280（D stage + FusionStorage 填充）+ 8 x 24576 = 230400。头部加上 D stage 与填充（512 + 33280 = 33792）等于 builder 的 epilogue 估计这一项本身，所以那 192 和 8 x 16 = 128 都是高估（第 3.4.1 节）。第九个 A/B stage 需要 24576 字节，所以这点余量放不下一个；使用它的办法只有不同的 stage shape 或更小的 epilogue 缓冲区，两者都会改变 smem 映射、各处的对齐以及屏障计数。
</details>

**D12. CUDA 12.x 的头文件给出 `alignof(CUtensorMap) = 64`。那么 `sizeof(ExplicitGemmParams)` 是多少，`explicit.hpp` 中哪些 `static_assert` 的判定会改变，设备端代码需要改吗？**

<details><summary>答案</summary>

576 字节：各字段在 576 处结束，而 576 已经是 64 的倍数，所以没有尾部填充。没有任何 assert 的判定会改变：`alignof >= 64` 仍然成立，无论对齐如何，map 的偏移都是 `sizeof(CUtensorMap) = 128` 的倍数，标量偏移相对于 `4 * 128` 固定，而大小 assert 同时接受 576 和 640。设备端在固定偏移 520/528/532 处读取 `k`、`alpha`、`beta`，在 0/128/256/384 处读取四个 map，全都不变，所以设备端代码没有任何改动。只有该块大小的两处记录会改变：PTX 的 `.param` 声明会读作 `.align 64 .b8 [576]` 而不是 `.align 128 .b8 [640]`，SASS 资源行中的 `CONSTANT[0]` 会是 `0x380 + 576` 而不是 `0x380 + 640`。
</details>

**D13. 对 rank 1、`s = 5`、`kb = 1` 重新计算 `desc_b`，并解释为什么结果与 rank 0 的相同。**

<details><summary>答案</summary>

rank 1 的基址是 `0x01000400`。共享地址 `0x01000400 + 164864 + 5 x 8192 + 32 = 0x01000400 + 0x28400 + 0xA000 + 0x20 = 0x01032820`；`>> 4 = 0x103282`；`& 0x3FFF = 0x3282`（位于地址第 24 位的 rank 位在移位后变成第 20 位，并被 14 位掩码丢弃）。`lo = 0x10000 | 0x3282 = 0x13282`；`desc_b = 0x4000404000013282`。增量形式：`0x12880 + 5 x 512 + 2 = 0x12880 + 0xA00 + 0x2 = 0x13282`。rank 0 给出完全相同的值，因为被掩码的字段只看到地址的低 18 位，而这些位在每个 CTA 中都相同；MMA 单元在 pair 的两个 CTA 中都把起始地址解释为 CTA 相对的。
</details>

**D14. 手工把 `Swizzle<2,5,2>` 应用到字节偏移 `0x2A8`，并说明它在应用前后分别是原子的哪一行和哪个 chunk。**

<details><summary>答案</summary>

`0x2A8 = 680`。行索引 = 第 [7,9) 位 = `(0x2A8 >> 7) & 3 = 5 & 3 = 1`（在 512 字节的原子周期内：`680 - 512 = 168`，第二个原子的第 1 行）。Chunk = 第 [5,7) 位 = `(0x2A8 >> 5) & 3 = 21 & 3 = 1`。`yyy = 0x2A8 & 0x180 = 0x080`；`>> 2 = 0x20`；`0x2A8 ^ 0x20 = 0x288`。之后：chunk `1 ^ 1 = 0`，同一行，chunk 内偏移同为 8。所以第 1 行的逻辑 chunk 1 被存放在 chunk 0 处。
</details>

**D15. W2 用 `"+r"` 声明 `pred` 并把它初始化为 0。改用 `"=r"` 且不做初始化会出什么问题，出在哪些 lane 里？**

<details><summary>答案</summary>

只有被选中的 lane 执行 `@%%px mov.s32 %1, 1`。用 `"=r"` 时，编译器假定该 asm 总是会写 `pred`，于是前面的 `pred = 0` 是死代码、可能被删除；在 31 个未被选中的 lane 中，`pred` 随后保持着该寄存器原本的任意内容，而 `if (elect_one_sync())` 可能在若干个 lane 中为真（重复发射 TMA/MMA），也可能一个都没有。`"+r"` 把该寄存器也声明为输入，所以这个 0 会被物化，并在 asm 不写入的那些 lane 中保留下来（explicit.cu:265-277；笔记 E.6.2 W2）。
</details>

**D16. Rank 3 的 epilogue 线程以 `acc_empty_base + 8 st` 调用 W11。给出 `st = 1` 时的地址算术，说出它到达的是哪个屏障，并解释为什么 W21/W22 接收的是已经掩码过的地址，而 W11 则在内部做掩码。**

<details><summary>答案</summary>

`smem_base(3) = 0x03000400`；`acc_empty_base + 8 = 0x03000400 + 272 + 8 = 0x03000518`；`& 0xFEFFFFFF` 清除第 24 位 -> `0x02000518` = rank 2 的 `acc_empty[1]`（pair leader）。W11 在内部做掩码，是因为 CUTLASS 函数 `umma_arrive_2x1SM_sm0` 就是这样做的（barrier.h:909），而 wrapper 照抄了它。CUTLASS 的 TMA 复制函数也在内部做掩码（copy_sm100_tma.hpp:292），但 kernel 把 `smem_base & kPeerBitMask` 外提到 k-tile 循环之外作为 `full_masked_base`（explicit.cu:770），并在每个 k-tile 加上 `8 s`，所以 W21/W22 接收的是掩码后的地址。PTX 指令完全相同，而掩码每个 CTA 只施加一次、在 tile 循环之前，而不是每个 tile 128 次（nvcc 把它折叠为 SASS `0x0c00` 处的 `0xfefffff0`，位于分发之后 producer 的设置代码中，笔记 G.3）。
</details>

**D17. warp 6 可以用 W31 读取哪些 TMEM lane，对 stage 2、subtile 5（T = 0）它传入的地址是什么？**

<details><summary>答案</summary>

`w = 6 - 4 = 2 = 6 % 4`，lane [64, 96)。`taddr = ((32 x 2) << 16) + 128 x 2 = 0x400000 + 256 = 0x400100`；加上 `16 x 5 = 80` -> `0x400150`：lane 64，列 336。Lane `l` 收到 stage 2 的 `D[64 + l][80..95]`（explicit.cu:1006, 1031, 1036）。
</details>

**D18. 推导 producer 在 tile `t = 5` 的 k-tile `q = 19` 处在 `empty[3]` 上等待的 parity，以及在 `n_tiles = 31` 个 tile 之后尾段步骤 `i = 3` 的 parity。**

<details><summary>答案</summary>

步骤编号 <code>n = 128 x 5 + 19 = 659</code>；槽位 <code>659 % 8 = 3</code>（一致）；parity <code>1 ^ ((659 / 8) &amp; 1) = 1 ^ (82 &amp; 1) = 1</code>。尾段：<code>n = 128 x 31 + 3 = 3971</code>；槽位 <code>3</code>；<code>3971 / 8 = 496</code>，为偶数，parity <code>1 ^ 0 = 1</code>。对每个 <code>n_tiles</code>，每个尾段步骤的 parity 都是 1，因为 <code>128 n_tiles / 8 = 16 n_tiles</code> 是偶数（第 8.6.8 节；在 1518 条 <code>K_TAIL</code> 记录中的全部 1056 条 mainloop 尾段记录上实测，笔记 G.1）。
</details>

**D19. explicit kernel 把常量 `0` 作为第三个 TMA 坐标传入，并把 `kCacheHintEvictNormal` 作为缓存策略传入。对于这个固定的运行，两者之中你可以改变哪一个而不改变 D，而你必须先证明什么？**

<details><summary>答案</summary>

缓存提示：EVICT_NORMAL 是 CUTLASS 的 <code>with()</code> 的默认策略，只引导 L2 的替换；EVICT_FIRST/EVICT_LAST（<code>copy_sm90_desc.hpp:193-197</code>）会改变时序，但绝不会改变数值。它仍然是对 baseline 的一处偏离，所以你必须重新测量（笔记第 12 节）并证明没有回归，然后才能在性能层面称其为语义保持的。batch 坐标必须保持为 0：这些 map 的 rank 为 3，L 范围为 1（笔记第 5.3 节），而 explicit kernel 按构造就是一个 <code>L = 1</code> 的 kernel（它的 <code>run()</code> 拒绝 <code>l != 1</code>；笔记 G.3 "Generality"）。改变它会去寻址越界坐标（由 TMA 零填充）并破坏 D。
</details>

**D20. Rank 2 (x = 0, y = 1) 是一个 pair leader。给出它的 `tcgen05.commit` 在读完 stage s = 5 之后所释放的屏障的 32 位 `shared::cta` 地址、该 commit 操作的 16 位掩码，以及它针对 st = 1 的累加器 commit 的地址与掩码。**

<details><summary>答案</summary>

`smem_base = 0x400 + (2 << 24) = 0x02000400`。mainloop 释放：`empty[5] = smem_base + OFF_MAIN_EMPTY + 8 x 5 = 0x02000400 + 64 + 40 = 0x02000468`，掩码 `kMainloopReleaseMask = 0xF`（explicit.cu:942, 186）。该 multicast 向每个 CTA 的 smem 的偏移 `64 + 40` 处递送一次到达，也就是每个 CTA 自己的 `empty[5]`。累加器 commit：`acc_full[1] = smem_base + 240 + 8 = 0x020004F8`（文件自己的例子在 explicit.cu:138），掩码 `0x3 << (2 x 1) = 0xC`（explicit.cu:902），只到达 rank 2 和 3。
</details>

**D21. 一个 cluster 处理了 T = 30 个 tile。列出 leader 的累加器尾段的四个 `(index, parity)` 对，并说明每一个等待的是哪个 tile 的释放。**

<details><summary>答案</summary>

30 个 tile 之后的状态：`idx = 30 % 4 = 2`，`phase = 1 ^ ((30 >> 2) & 1) = 1 ^ (7 & 1) = 0`。各步：i = 0：`acc_empty[2]`，parity 0（tile 30 的 acquire，等待 tile 26 的释放）；i = 1：`acc_empty[3]`，parity 0（tile 31 -> tile 27 的释放）；i = 2：`acc_empty[0]`，parity `1 ^ ((32 >> 2) & 1) = 1`（tile 32 -> tile 28 的释放）；i = 3：`acc_empty[1]`，parity 1（tile 33 -> tile 29 的释放）。规则：尾段步骤 i 以 parity `1 ^ (((T + i) >> 2) & 1)` 等待 tile `T + i - 4` 的释放（第 9.6.3 节）。
</details>

**D22. 线程 `(w = 2, l = 20)` 存储 subtile `e = 5` 的值 `j = 11`。计算相对于通用基址的 smem 字节偏移，说出它所落入的 stage、TMA box、行和 chunk，以及 TMA 将在那里读取的元素 `(m', n')`。**

<details><summary>答案</summary>

`explicit.cu:989`：`512 + 8192 (5 % 4 = 1) + 2048 x 2 + 128 x 11 + 4 ((20 & 7 = 4) + 8 (((20 >> 3) & 3 = 2) ^ (11 & 3 = 3) = 1)) = 512 + 8192 + 4096 + 1408 + 4 x 12 = 14256 = 0x37B0`。经由第 1013 行：`st_off[3] = 512 + 4096 + 16 + 384 + 32 x 1 = 5040`，加上 `512 (11 >> 2 = 2) = 1024`，加上 `8192 (5 & 3 = 1) = 8192`：14256。stage 1（字节 8704..16895），stage 内偏移 5552：box `i = 5552 / 2048 = 2`（= `w`），行 `n' = (5552 - 4096) / 128 = 11`（= `j`），行内字节 48：chunk 1，字 4。TMA 把第 11 行的 chunk 1 反 swizzle 到位于 `[8 (1 ^ (11 & 3)) , +8) = [16, 24)` 内的 `m'`，字 4 给出 `m' = 20 = l`。该元素是 box 2 的 `(m', n') = (20, 11)`：D 的第 `128 tm + 64 + 20` 行、第 `128 tn + 80 + 11` 列。
</details>

**D23. 计算 rank 2 的 `acc_full[1]` 及其 A stage 3 的 `shared::cta` 地址，并说明为什么两者都正确对齐。**

<details><summary>答案</summary>

`smem_base = 0x400 + (2 << 24) = 0x02000400`。`acc_full[1] = smem_base + OFF_ACC_FULL + 8 = 0x02000400 + 240 + 8 = 0x020004F8`（8 字节对齐，正如每个 mbarrier 都必须的那样）。A stage 3 = `smem_base + OFF_SMEM_A + 3 * 16384 = 0x02000400 + 33792 + 49152 = 0x02000400 + 82944 = 0x02014800`；`0x14800 % 1024 = 0`，因为基址 `0x400` 和 `OFF_SMEM_A = 33792 = 33 * 1024` 都是 1024 的倍数，而 stage stride 16384 也是。基址的对齐是启动的一种属性（没有静态 `__shared__`），实测为 `0x400`，不是声明所能强制的东西。
</details>

**D24. 推导 `blockIdx = (6, 1)` 的 `(tm, tn)`，说出其 rank，并说明它所在 cluster 中的哪一个其他 CTA 与它共享 A 的行。**

<details><summary>答案</summary>

`x = 6 & 1 = 0`，`y = 1 & 1 = 1`，rank `= x + 2y = 2`。`tm = 2 * (1 >> 1) + (6 & 1) = 0 + 0 = 0`；`tn = 2 * (6 >> 1) + (1 & 1) = 6 + 1 = 7`。tile (0, 7)：D 的行 [0, 128)，列 [896, 1024)。与它共享 A 的行的 CTA，是 tm 相同而 y 不同的那一个，即 rank `x + 2(1-y) = 0`，`blockIdx = (6, 0)` -> `(tm, tn) = (0, 6)`。两者的 A multicast 掩码都是 `0x5 << x = 0x5`（rank 0 和 2）。
</details>

**D25. explicit kernel 使用 26 个寄存器，而 CUTLASS kernel 使用 68 个。给出 26 这个数的逐寄存器理由，并精确说明为什么这一差别在这里值零纳秒。**

<details><summary>答案</summary>

`R4..R19` 是 16 个 `LDTM.x16` 目的寄存器（一个 lane 的一个 subtile 的 16 个 fp32 值）；`R20..R23` 是四个随 lane 而异的 `STS` 基址，携带着 `Swizzle<2,5,2>` 项；低编号寄存器是每个角色的暂存值和每个 lane 的 phase 位（`R9` 是 MMA warp 的 `cons_phase` 和 epilogue 的 `acc_full` parity，`R0` 是 producer 的 `prod_phase`）；`R1` 按代码生成约定是编译器的每线程局部内存栈基址（NVIDIA 内部），加载一次（0x0000 处的 `LDC R1, c[0x0][0x37c]`）且从未被读取（没有 `STL`/`LDL` 对它寻址，`STACK 0`）。SASS 命名了 24 个寄存器 `R0..R23`，而 ptxas 报告 26，比命名的数目多两个（对 CUTLASS kernel 是 `R0..R65` = 66 -> 68）。CUTLASS kernel 多出了用于死的 beta != 0 路径的 16 寄存器零 C 片段、`epi_load` 状态、向量寄存器中的 scheduler 簿记，以及 44 对 uniform 到向量的 spill/fill（`R65` -> 68）。occupancy 为 1 CTA/SM，因为有 230400 B 的共享内存（`__launch_bounds__(256, 1)` 只要求至少一个驻留的 CTA，第 11.2.1 节），所以寄存器数目无法改变驻留。两个 kernel 都不溢出（`LOCAL 0`）。热循环不受发射限制（第 12.5.5 节），所以 CUTLASS kernel 的那些额外搬移同样不花费任何代价。这个数目在多 CTA/SM 的 kernel 中，或在溢出落在热循环内时，就会变得重要。
</details>

**D26. 一个 kernel 在固定命令上通过了 `Disposition: Passed`。说出三种它无法检测出的缺陷，以及能捕获每一种缺陷的检查。**

<details><summary>答案</summary>

(1) 不同的算术（折叠的 `alpha * acc`、丢掉的 `fmaf`、不同的 MMA 序列）：数据是 [-8, 8] 内的整数，所以任何求和顺序都是精确的；用 PTX/SASS 清单（epilogue 中 128 条 `FMUL` + 128 条 `FFMA`、0 条 `FADD`；SASS 总数是 130 条 `FMUL`，多出的两条来自 `%cluster_ctaid` 的下沉结果）和对非整数数据的逐位比较（第 13.5 节步骤 8）来捕获它。(2) 一个覆盖缺陷，它把一个 tile 用相同的值写两次，或者把上一次运行留下的陈旧的正确值留在 D 中：用哨兵填充（0xFF -> NaN）和 K5 恰好一次检查来捕获它。(3) 丢掉的尾段或缺失的 dealloc 握手，这是一种间歇性的退出竞态：用连续五次运行、"Passed then hang in iteration 2" 症状（笔记 E.7.2，失效模式表）以及每个 cluster 46 条 `K_TAIL` 记录来捕获它。
</details>

### 14.2.3 MODIFY

**M1. 累加器的 empty 屏障用计数 128 而不是 256 初始化。会发生什么？**

<details><summary>答案</summary>

[A，根据 [S] 协议] 每次使用时恰好有 256 次到达落在 leader 的 `acc_empty[st]` 上（来自每个 CTA 的 epilogue 各 128 次，`explicit.cu:1039`），所以在计数为 128 时，屏障每次使用会完成两个 phase。两种失效模式，取决于时序。(1) 仅 leader 的 128 次到达就完成了该 phase，MMA warp 针对 tile t + 4 的 `producer_acquire` 得以通过，而 2SM MMA（它写入的是两个 CTA 的 TMEM）在 peer 的 epilogue 可能仍在对 stage st 发射 `tcgen05.ld` 时就覆写了它：peer 中的 D 悄无声息地出错。(2) parity 簿记发生漂移：在两个 CTA 都到达之后，屏障领先了两个 phase。producer 的下一次 parity 等待于是要么在没有任何新到达的情况下通过（又一次覆写），要么，如果交错让屏障停在 phase 中途，就阻塞等待 128 次到达。这些到达只能来自一个自身正在等待 `acc_full` 的 epilogue：挂起。epilogue 远短于 27.8 us 的 per-tile 周期（推导得出，0.889354 ms / 32 个 tile；epilogue 自身的墙钟时间在笔记中没有测量，笔记只记录了整个 tile 的周期和每个 warp 每个 subtile 的 77-92 条 SASS 指令，G.3；因此“几微秒”是（未验证）的）。有了 4 个累加器 stage 和那样短的 epilogue，竞态窗口很小，这个 bug 在本次运行中可能保持潜伏。它会在 stage 更少或 epilogue 更慢时浮现。计数 256 是唯一等于每个 phase 到达次数的值。
</details>

**M4. cluster 变为 (2, 1, 1)。哪些常量、掩码、计数和公式会改变？**

<details><summary>答案</summary>

[A，用 [S] 的公式] rank 只有 {0, 1}（始终 y = 0），一个 pair。A 没有 multicast 伙伴：每个 CTA 自己加载它完整的 128 x 64 A stage。单个 16384 B 的 (64, 128, 1) fp16 box 是可编码的，因为 `cuTensorMapEncodeTiled` 把每个 `boxDim` 项限制为 256 个元素（"must be non-zero and less than or equal to 256"），而 `SWIZZLE_128B` 只约束内层范围（"CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128" 字节；64 fp16 = 128 B）[S, CUDA Driver API `cuTensorMapEncodeTiled`]。leader 的 `full[s]` 于是和以前一样期望 2 x (16384 + 8192) = 49152 字节，但来自 4 个 box 而不是 6 个（或者保留两个 64 行的 box 和 6 次字节到达；两者都能配平）。TMA A wrapper 不再需要 `.multicast::cluster`（或者平凡地使用掩码 0x1 << x）。mainloop empty 计数变为 `2/2 + 1/1 - 1 = 1`，release 掩码为 0x3（只有那一个 leader 读取任一 CTA 的 smem）。累加器 commit 掩码 0x3。CLC：由 lane 0..1 做 full 装填；empty 计数 `32 + 2 x (32 + 32 + 128) = 416`。从启动坐标和从首 CTA id 到 (tm, tn) 的映射必须用 cluster 除数 (2, 1) 从 `swizzle_and_rasterize` / `possibly_transpose_work_tile` 重新推导（`sm100_tile_scheduler.hpp:666-677, 695-814`）；它不是 D9/D10 的 2 x 2 公式。cluster tile 为 256 x 128，所以共有 32 x 64 = 2048 个。代价：每个输出的 L2 到 SM 流量上升 1.5x（每个 k-tile 32768 A + 16384 B = 49152 B 对应 32768 个输出，而 65536 B 对应 65536 个输出）。驻留可能上升到每个 TPC 一个 cluster（74 个 cluster，全部 148 个 SM），而不是 132 个 SM 上的 33 个 cluster [A；2-CTA cluster 的硬件放置没有实测]。那是否是净收益，是一项第 3 章的测量，而不是推导。
</details>

**M5. `alpha` 和 `beta` 变为 `constexpr float alpha = 1.0f, beta = 0.0f`。D 会改变吗？还有别的会改变吗？**

<details><summary>答案</summary>

[A，根据笔记第 0/8 节] D 不会改变：`1.0f * acc` 是精确恒等，而 `fmaf(0.0f, 0.0f, t)` 折叠为 `+0.0f + t`，这对每个 t（包括 `t = -0.0f`）都等于 FFMA 的结果（两者都给出 `+0.0f`）。一个 `-0.0` 只能在 alpha < 0 或 acc = -0.0 时从 `alpha * acc` 产生，而 acc = -0.0 需要 8192 个乘积中的每一个在 IEEE 算术下都是 -0.0（对 harness 在 [-8, 8] 内的随机整数而言实际上永远不会发生，harness:307-312；对 MMA 未经证明，第 10.9 节）；无论哪种情况 D 都不受影响。改变的是指令流：每个 epilogue warp 每个 tile 的 128 条 FMUL + 128 条 FFMA 消失（至多一条 FADD，或者什么都没有），`ld.param.v2.b32 [+528]` 及其 `LDCU.64 UR26, c[0x0][0x590]` 消失，释放出保存 alpha/beta 的两个 uniform 寄存器（笔记 G.3；每线程寄存器数目是否改变未验证），并且该 kernel 再也不能运行任何其他的 alpha/beta。主机端守卫 `explicit.cu:1272`（`if (params.beta != 0.0f) { return cutlass::Status::kErrorNotSupported; }`）仍会拒绝 beta != 0，但 alpha != 1 现在会悄无声息地算出错误的结果。性能在噪声范围内不会改变，因为 epilogue 不在关键路径上 [A]。
</details>

**M7. warp 2 在发射其 tile 的加载之前而不是之后消费 CLC 响应。**

<details><summary>答案</summary>

[A，用 [M] 的延迟] 不会死锁：rank 0 的 warp 2 在 tile 开始时、在 `load_ktiles` 之前（`explicit.cu:783-786`）到达 `throttle_full[t]`。正是这次到达允许 scheduler 发出查询 t，所以一次先于加载的消费仍然会得到满足。但它暴露了 CLC 延迟：warp 2 会在发射 tile t 的加载之前等待响应 t（从发出到消费最小 736、中位数 1408、最大 2336 ns [M, D.3, trace 构建]）。环限制了这种暴露。在 tile 边界处 producer 已经领先运行了至多 8 个 stage，所以 MMA warp 持有至多 8 个 k-tile = 8 x 217 ns = 1.74 us 的工作，而第一个新 stage 在第一次新发射之后一个 TMA 延迟就变得可消费。因此每个 tile 的气泡是 `max(0, response latency + first-load TMA latency - 1.74 us)`：在中位数延迟下约为 `max(0, first-load TMA latency - 0.33 us)`（只有当该延迟低于约 0.33 us 时才为零；在第 7.6.9 节的 ~1 us 下约为 0.7 us），在 2.34 us 的最大值下则至多约 0.6 us 加上重新填充的延迟，即**一个 27.8 us 的 tile 的 0 到大约 2-5%** [A]。一个笼统的 "1.4 us = 5%" 会是假定在等待响应时环已经排空的上界。baseline 顺序让响应在 tile t 的 256 次加载期间到达。在 trace 构建的未加探针的 cluster 中，scheduler 从消费响应 n 到发出查询 n+1 的实测间隔（它的 throttle 等待）是 28.8-37.7 us（中位数 30.8）[M, D.3 和第 7.4 节]，所以该延迟被完全隐藏。取回的 tile 只在循环体末尾才取代当前这个（`explicit.cu:794`）。为什么 CUTLASS kernel 这样安排顺序 [S，cb424739 `sm100_gemm_tma_warpspecialized.hpp:617-679`]：在 producer 分支中，两次 `collective_mainloop.load` 调用（`:640-647`、`:654-661`）在 `scheduler.fetch_next_work`（`:665-669`）之前返回，所以 tile t 的 TMA 发射与 CLC 往返重叠。而且因为 throttle 的 `producer_acquire` + `producer_commit`（`:632-636`）每次循环迭代恰好推进一次，先消费后加载还会把查询 t+1 推迟 tile t 的整个加载阶段那么久。先加载后消费的顺序也与 800 计数相符：每个响应的到达次数不依赖于顺序，只有它们的时序依赖于顺序。
</details>

**M9. epilogue 的 `cp.async.bulk.wait_group.read 1` 被改为 `.read 0`；然后改为 `.read 3`。**

<details><summary>答案</summary>

[S/A] `explicit.cu:1069` `tma_store_wait_read<1>();` 由 warp 4 在提交 subtile e 的组之后立即发射。`.read N` 保证除最近的 N 个之外的所有 bulk group 都已完成对其 smem 源的读取。stage `(e + 1) % 4` 是下一个被写入的，而它最后一次被组 e - 3 读取，所以最宽松的合法值是 N = 3；`read 1` 是保守的。这个 1 从哪里来 [S]：CUTLASS 存储流水线是 `PipelineTmaStore<4,1>` [M, D.1 D1]，即 `Stages = 4`、`UnacquiredStages = 1`（"the number of committed TMA store batches that can be in flight upon return of producer acquire"，`include/cutlass/pipeline/sm90_pipeline.hpp:650-700`）。`producer_acquire` 在 `params_.always_wait || count > UnacquiredStages` 时运行 `tma_store_wait<UnacquiredStages>()`（`:698-699`），而 kernel 设置了 `epi_store_pipeline_params.always_wait = true`（`sm100_gemm_tma_warpspecialized.hpp:491`）。这就是为什么每个 subtile（包括第一个）都执行这条 `.read 1`，而不是在待完成的组少于两个时跳过它。`.read 0` 是合法的，但它让每个 subtile 都串行排在其自身存储的 smem 读取之后，给 epilogue 每个 subtile 增加一次 TMA 读取延迟；在本 kernel 中这仍然不在关键路径上，所以在这里大概不可见，但在一个受 epilogue 限制的 kernel 中会可见 [A]。`.read 3` 在 4 个 stage 下是合法的，并允许三个组在途；它是否有帮助未经测量。无论你选择哪个 N，退出条件都不变：对 beta = 0，`store_tail` 什么都不执行，所以 CTA 在最后一个组的 smem 读取尚未被等待的情况下退出 [M, D15]；实测的那些运行在该退出条件下通过了，而全局写入作为 kernel 完成的一部分而完成这一说法，是笔记对硬件语义的解读（笔记 B.7 D15），不是一项测量。
</details>

**M10. 一位同事提议通过让 peer CTA 跳过它的 `arrive.expect_tx`，*并且*让 leader 只期望它自己的 24576 字节来节省工作。什么会出错？**

<details><summary>答案</summary>

peer 已经跳过了 arrive（只有 <code>is_leader_lane</code> 到达，explicit.cu:736）。字节计数不能减半：peer 的 A box 和 B box 通过 peer 位掩码（<code>full_masked_base + 8 s</code>，explicit.cu:747/754）把 leader 的 <code>full[s]</code> 作为它们的屏障，所以无论 leader 期望多少，每个 stage 都会有 49152 字节在它上面完成。在 <code>expect_tx = 24576</code> 下，结果取决于六个 box 中有多少个在 leader 的 <code>arrive.expect_tx</code> 之前落地：tx-count 从 0 开始，每次提前的 <code>complete_tx</code> 都把它驱动到 <code>-8192 k</code>（第 8.4.4 节）。如果先落地了 0..3 个 box，expect 把 tx-count 带到 <code>24576 - 8192 k &gt;= 0</code>，而 phase 在总共三个 box 落地后就立即完成——可能在 peer 的那半个 stage 进入 smem 之前——所以 leader 的 <code>tcgen05.mma.cta_group::2</code> 读到陈旧的 peer smem：悄无声息的错误结果。如果先落地了 4 个或更多，expect 让 tx-count 停在 <code>24576 - 32768 = -8192</code> 或更低，而此时待到达计数已经是 0，其余的 box 只会把它驱动得更负，它永远不会回到 0：leader 的 warp 0 挂在 <code>full[s]</code> 上——即笔记失效模式表（E.7.2）中的 "mainloop deadlock at <code>q = 0</code> -&gt; <code>expect_tx != 49152</code>" 行。所以：悄无声息的损坏或挂起，取决于落地顺序。49152 是 pair 内落地的一切之和（第 8.5.3 节）。
</details>

**M11. `OFF_SMEM_CD` 是 512，stage stride 是 8192。假设一次重构把这些 stage 移到了偏移 256（保持 8192 字节的 stage）。什么会出错，它会如何表现出来？**

<details><summary>答案</summary>

TMA 把 `SWIZZLE_128B_ATOM_32B` 置换施加于绝对共享内存地址的位 [5,9)，而线程公式把 `Swizzle<2,5,2>` 施加于相对 stage 基址的偏移，并假设该基址是 512 字节周期的倍数。在偏移 256 处（绝对地址 `0x500 + (r << 24)`），stage 基址对地址位 [7,9) 贡献了 `(0x500 >> 7) & 3 = 2`，因此对于行 `n'`，引擎的相位为 `(n' + 2) & 3 = (n' & 3) ^ 2`：对于行组 `g`，它读取的是 chunk `g ^ (n' & 3) ^ 2`，而线程写入的却是 chunk `g ^ (n' & 3)`。在每个 128 字节行之内，32 字节的 chunk `0 <-> 2` 与 `1 <-> 3` 被互换，也就是说行 `[8a, 8a + 8)` 落到了行 `[8 (a ^ 2), +8)` 应在的位置（行 0-7 <-> 16-23，8-15 <-> 24-31），于是 D 会在这一有规律的 8 行模式中数值出错，而每列的 512 字节段仍留在原位（在准备本章时，针对基址 `0x500` 枚举了全部 8 x 4 x 16 x 32 种情形加以核对）。explicit.cu:163 处的 `static_assert(OFF_SMEM_CD % 512 == 0, ...)` 正是为了让这种情形转而成为一个编译错误。
</details>

### 14.2.4 DESIGN

**G1. 为一个 256 x 256 的 pair tile（CTA tile 为 128 x 256 x 64）、同样的 fp16 输入与 fp32 TMEM 累加器，确定各流水线的规模。**

<details><summary>答案</summary>

[A，使用笔记第 6.1 节的 builder 算术与 [S] 常量] 每个 CTA 每个 stage：A 为 128 x 64 x 2 = 16384 B（不变），B 现在是 128 列 x 64 x 2 = 16384 B（每个 CTA 持有该 pair 的 256 个 N 列的一半），再加上 16 B 的屏障：共 32784 B。采用同一 (128, 16) subtile 的 epilogue 存储：4 x 8192 + 填充 = 33792 B。Carveout 为 160 B（192 - 32：累加器流水线的屏障存储在 2 个 stage 下减半，第 3.9.7 节）。stage 数 = floor((232448 - 160 - 33792) / 32784) = floor(6.05) = **6**，共使用 160 + 33792 + 6 x 32784 = 230656 B。TMEM：一个 stage 为 128 个 lane x 256 个 fp32 列，因此 512 列中只放得下 **2 个累加器 stage**（MMA 与 epilogue 之间仍有整整一个 tile 的余量）。屏障：mainloop 的 full tx = 2 x (16384 + 16384) = **65536 B**。A 保留其 4 x 8192 的 multicast box。对于 B，每个 CTA 要么加载一个 16384 B 的 (64, 128, 1) fp16 box，要么加载两个 64 列的 box；单个 box 是可编码的，理由见 M4（每个 `boxDim` 项至多 256 个元素，swizzle 限制只针对 128 字节的内层范围 [S, CUDA Driver API `cuTensorMapEncodeTiled`]）。Empty 计数仍为 2 / 掩码 0xF；acc full 1 / empty 256；CLC 与 throttle 不变。MMA：`tcgen05.mma.cta_group::2` 256 x 256 x 16（每条 2 MFLOP），每个 k-tile 仍为 4 条；指令描述符的 n_dim = 256 >> 3 = 32 -> **idesc = 0x10400010**；B 描述符的起始地址像 A 的一样，每个 stage 步进 16384 B（以 16 字节为单位则为 1024）。Epilogue：每个 tile 16 个 128 x 16 的 subtile，TMEM 地址为 `T + ((32 w) << 16) + 256 st + 16 e`。每个 k-tile 的时间加倍（按实测 MMA 速率约为 434 ns），因此 6 个 stage 缓冲了约 2.6 us 的 MMA 工作，而之前是 1.7 us：按时间更深，按数量更浅。这笔权衡是：每 L2 字节 1.33x 的 FLOP（128 -> 171，第 3.9.7 节和第 13.6 节：A 每个 k-tile 的流量不变，所以每个输出的 A 字节数减半；B 的流量随 tile 加倍，所以每个输出的 B 字节数不变），对应的代价是 2 个 TMEM stage、6 个 smem stage 以及每个 tile 长 2x 的 epilogue。Colfax 的 cluster 教程就 256 x 256 的 pair MMA 提出了一个相关的观点，把它与两条独立的单 CTA 128 x 256 MMA 相比较（"the same number of FLOPs but transfers half the operand data"）。注意这里的假设：builder 针对 `CtaN > 128` 的规则给出 `N_perf = 32`、一个 (128, 32) 的 subtile、66560 B 的 epilogue smem（4 x 16384 加上同样的填充）以及 **5** 个 mainloop stage（第 3.9.7 节和第 13.6 节）；上面的 6 只在保留 (128, 16) subtile 的前提下才成立。
</details>

**G2. 为 beta != 0（必须读取 C）设计屏障表。**

<details><summary>答案</summary>

[S，笔记 7.2 的表以及 7.8/B.3.5；初始化已在 smem 映射中布置好] 增加 warp 3 作为 C producer，并初始化那三个保留区域：

| 屏障 | 计数 / tx | 谁 |
|---|---|---|
| epi_load full[4] 位于 128 | 32 次到达 + 8192 个 tx 字节 | `PipelineTransactionAsync<4>` 把两者分开：单单一个线程的 `mbarrier.expect_tx.shared::cta.b64 [full], 8192`（`producer_expect_transaction`，"Must be called by 1 thread"，`sm90_pipeline.hpp:941-946` -> `barrier.h:626-637`，无到达）加上来自 warp 3 各 lane、经 `mapa` 送到 `dst_blockid` 的 32 次普通 `mbarrier.arrive`（`producer_commit`，`:948-951`）。如果全部 32 个 lane 都执行 `arrive.expect_tx(8192)`，tx-count 将为 32 x 8192 = 262144，而该 phase 将永远不会完成。这 8192 字节是一个 128 x 16 的 fp32 C subtile = 四个 2048 B 的 (32, 16) box，进入 C/D stage。CUTLASS epilogue 中的顺序（`sm100_epilogue_tma_warpspecialized.hpp:528-541`）：`producer_acquire` -> TMA 复制 -> `producer_expect_transaction` -> `producer_commit`（字节可能先于该期望落地，R2）。SASS：裸的 `expect_tx` 是 `@!P2 SYNCS.ARRIVE.TRANS64.RED.A0TR RZ, [UR29+0x80..0x98], R3`，其中 `R3 = 0x2000` = 8192（8 处，位于 `sass_gemm.txt` 的 `0x5240-0x6460`，在 beta = 0 时不执行；`A0TR` = 无到达，事务计数由源寄存器更新，NVIDIA 内部；第 12.7.4 节）[M, D.2] |
| epi_load empty[4] 位于 160 | 128 | 128 个 epilogue 线程在从 smem 读取 C 之后释放 |
| load_order[2] 位于 192 | 各 32 | warp 2 到达一次，在其第一次 `load()` 调用（8 个 k-tile 的 prologue）之后，使 C 的加载不与 A/B 的 prologue 竞争（我对其目的的分析）；warp 3 等待一次，在其第一次 C 加载之前 |
| CLC empty[2] | **928** = 32 + 4 x (32 + 32 + 32 + 128) | warp 3 现在也消费 CLC 响应 |

由谁初始化它们以及它们以什么顺序被使用 [S，笔记 7.2 的 "Init warp" 列；cb424739 `sm100_gemm_tma_warpspecialized.hpp`]：warp 1 初始化 8 个 `epi_load` 屏障，warp 3 初始化 2 个 `load_order` 屏障（`OrderedSequenceBarrier<1,2>`，计数各 32），因此初始化总数变为 43（baseline 中 `SYNCS.EXCH.64` 为 43 [M, D.2]）。warp 2 每个 kernel 在 `load_order` 上到达一次，介于其第一次 `load()` 调用（8 个 k-tile 的 prologue，`:640-647`）与第二次（`:654-661`）之间：`:649-652` `if (do_load_order_arrive) { load_order_barrier.arrive(); do_load_order_arrive = false; }`。因此第一个 tile 的 C 加载不可能与 A/B 的 prologue 争夺 TMA 带宽。warp 3 在它上面等待一次，在其第一次 C 加载之前（`:832-835` `if (do_load_order_wait) { load_order_barrier.wait(); do_load_order_wait = false; }`），之后再也不等。epilogue 以每个线程每个 subtile 16 条 `ld.shared.b32`（每个寄存器一条；每个线程每个 tile 128 条，即 PTX 中那 128 条静态的 `ld.shared.b32` C 读取 [M, D.2]）把 C 读入 C 片段，并用真实的 c 计算 `d = fmaf(beta, c, t)`。`store_tail` 现在必须执行 `cp.async.bulk.wait_group.read 0`，因为 C stage 会被复用。C 路径上受 `elect.sync` 守卫的 TMA 加载是唯一保留选举谓词的 TMA 指令 [M, D.2]。其他一切（mainloop、累加器、throttle、尾段 1-4）均不变。explicit kernel 的主机端守卫拒绝 beta != 0，正是因为这些都没有实现（`explicit.cu:1272`）。
</details>

**G3. 估计本 kernel 在 4096^3 上（同样的 tile、同样的 cluster）能达到的 speed-of-light 比例。**

<details><summary>答案</summary>

[A，锚定于 [M]] Cluster tile 数：(4096 / 256)^2 = 256；每个 tile 的 k-tile 数：4096 / 64 = 64。在 33 个驻留 cluster 下 [M, D.1 D4]，256 / 33 = 7.76 -> 8 个 wave：25 个 cluster 做 8 个 tile，8 个做 7 个，wave 量化效率为 7.76 / 8 = 0.97。按实测的每 k-tile 速率（0.889354 ms / 32 个 tile / 128 = 每个 k-tile 217 ns [M/A]；这是 kernel 时间的上界，见第 3.9.2 节的注意事项），每个 tile 的时间为 64 x 217 ns = 13.9 us，因此 mainloop 的关键路径为 8 x 13.9 = 111 us。关于这一比较有两点。第一，wave 量化*不是*一个差异项。8192^3 的参照有同样的 0.97（1024 / 33 = 31.03 -> 32 个 wave），而 217 ns 的速率是从那次运行的 32-tile 关键路径推导出来的。因此在 111 us 下，4096^3 的运行将恰好复现 8192^3 的吞吐量（2 x 4096^3 = 1.374e11 FLOP / 111 us = 1.238 PFLOP/s，对比 1.236 PFLOP/s [M, G.2]）。第二，唯一的差异项是固定成本 F（prologue、TMEM 分配、第一个 tile 的流水线填充、最后一个 tile 的 epilogue、各尾段：第 12 章），它被摊销到 8 个 tile 而不是 32 个上。笔记中没有实测 F（未验证）。如果它是 2-6 us，运行耗时 113-117 us，即 1.17-1.22 PFLOP/s，**为 8192^3 吞吐量的 0.95-0.98**；在 F = 9 us（120 us）时则为 0.93（1.145 PFLOP/s）。上限本身由 148 个 SM 中有 132 个可被 2 x 2 的 cluster 使用这一点所设定 [M, D.3]，任何 tile 的改变都无法挽回它；一个 (2,1,1) 的 cluster 或许可以（M4），代价是每个输出 1.5x 的 L2 流量。本章没有去获取 B200 的稠密 FP16 峰值规格数字，因此也未使用它（未验证）。
</details>

**G4. 为 M = 1024、N = 65536、K = 8192（同样的数据类型）选择 cluster shape 与光栅顺序。**

<details><summary>答案</summary>

[A，结合 [S] 的 scheduler 启发式规则] FLOP = 2 x 1024 x 65536 x 8192 = 1.0995e12，与 8192^3 相同。A 为 16 MiB（假设保持驻留于 L2；未实测），B 为 1 GiB，D 为 256 MiB。CTA tile 数为 8 x 512 = 4096（数量相同），cluster tile 数在 (2,2,1) 下为 4 x 256 = 1024。CUTLASS 的启发式规则 `tiles_n > tiles_m ? AlongM : AlongN`（笔记 7.4）在这里选择 **AlongM**（cluster-tile 数为 256 > 4），这是正确的选择。CLC 大致按递增顺序分发 id [M, D.3]，而在 AlongM 下，四个连续的 id 共享 B 的同一个 256 列条带，所以每个 B 条带从 HBM 大约读取一次（总计约 1 GiB）。在 AlongN 下，33 个 cluster 将在一个 m 条带内沿 n 行进，而每个 B 条带都会为 4 个 m 条带各重新取一次：约 4 GiB。按 7.7 TB/s（Lenovo LP2226，第 2.2 节；其他 NVIDIA 材料中为 8 TB/s）计算，这大约是 0.56 ms 的 HBM 时间，对比 0.89 ms 的计算时间：很紧。cluster shape：每个 k-tile 每 1K 个输出的 L2 到 SM 字节数，(2,1,1) 为 1536，**(2,2,1) 为 1024**，(4,1,1) 为 1280，(4,2,1) 与 (2,4,1) 为 768（A 每个 x 加载一次，B 每个 y 加载一次；见第 3 章的模型）。所以 (2,2,1) 是最好的 4-CTA shape，而 (4,2,1) 会把流量再削减 25%，代价是 8-CTA 的 cluster（驻留的 cluster 更少：8 个 CTA 需要同一个 GPC 内的 4 个 TPC [A]），以及 M = 1024 = 仅 2 个 512 的 cluster 行。建议：保留 (2,2,1) + AlongM（启发式规则自己的答案），进行实测，只有在 L2 到 SM 带宽被证明是限制因素时才尝试 (4,2,1)。
</details>

**G5. 用一个静态持久化调度取代 CLC（如同在 sm_90 上会做的那样）。什么会改变，以及就这个问题本身而言，CLC 换来了什么？**

<details><summary>答案</summary>

[A，结合 [M] 的驻留情况] 静态：恰好启动 N_resident 个 cluster，并让 cluster c 处理 tile id c, c + N, c + 2N, ...；每个 warp 在本地计算下一个 id。删除 CLC 流水线（2 + 2 个屏障、800 计数、装填 lane、`try_cancel`、解码、proxy fence、尾段 2）、throttle（4 个屏障）以及 scheduler warp；保留其他每一个屏障。微妙之处在于 N_resident：朴素的 148 / 4 = 37 在这台机器上是错的。实测：33 个原生 cluster，132 个 SM 忙碌，16 个空闲 (64, 65, 122-129, 142-147)，每个 MMA pair 都位于一个 SM pair (2j, 2j+1) 上，且 `cudaOccupancyMaxActiveClusters` = 33 [M, D.3, D.1 D4]。其解释——一个 2 x 2 的 cluster 需要同一个 GPC 内的两个完整 TPC，而芯片 skyline 中每个奇数大小的 GPC 都会留下一个未配对的 TPC，总共 8 个——是笔记的推断经 skyline 补全后的结果 [A，D.3 "Inferred"；skyline（NVIDIA 内部），算术见第 2.1.1 节]。静态地启动 37 个会使最后 4 个 cluster 在前 33 个完成之后作为第二个 wave 运行（每个约有 28 个 tile 的工作量）：运行时间将几乎翻倍。在 N = 33 下，静态轮转给出 1 个 cluster x 32 个 tile + 32 个 cluster x 31（1024 = 32 + 992）。TRACE 构建中记录到的 CLC 分发结果是 13 个 cluster x 32 + 19 x 31 + 1 x 19（被探针拖慢的 cluster (0,0)；416 + 589 + 19 = 1024，13 + 19 + 1 = 33）[M, D.3, G.1]：每个未加探针的 cluster 都收到 31 或 32 个 tile，在至多一个 tile 的差别内与静态划分的均衡相同，而拿到 32 个的那 13 个 cluster 恰好就是每个 tile 的时间最短的那 13 个 [M, D.3]。（Release 构建的分发结果没有记录；trace 构建中每个 tile 的时间为 31.9 us，对比 Release 的界 27.75 us，D.4，而它的 CLC 延迟数字就是本章通篇所引用的那些。）所以在这个均匀的问题上，CLC 换来的不是均衡，而只是健壮性：它不需要知道 N_resident，它容忍另一个 kernel 占用 SM，而且它的延迟（中位数 1.4 us）隐藏在每个 tile 28.8-37.7 us 的 throttle 等待之后 [M, D.3, trace 构建]。它的代价也几乎为零：8 个屏障、一个 50 条指令的 warp 循环 [M, G.3]、约 4% 的时间里有一个查询在途 [M, D.7]。在 tile 代价各不相同的场合（参差不齐的 K、混合的问题规模、分组 GEMM），它的均衡能力才成为重点（Colfax 的 CLC 文章）。
</details>

**G6. 一位同事提议在参数块中传递一个 `CUtensorMap const*`（指向全局内存副本的指针），而不是按值传递这些 map。设备端会有什么变化，而按值传递的选择又换来了什么？**

<details><summary>答案</summary>

TMA 接受位于 `.param`、`.const` 或 `.global` 空间中的 tensor map，所以这是可行的，但描述符将驻留在全局内存中。每条 `cp.async.bulk.tensor` 都将接受一个全局地址，而 `prefetch.tensormap` 将不再适用（PTX ISA 只为 `.const` 和 `.param` 中的 map 定义了它，9.7.10.16），如果该 map 曾被修改，kernel 将需要通用 proxy/tensormap fence 的规程，而首次访问将是一次全局内存读取而不是常量 bank 读取。以 `const __grid_constant__` 参数按值传递时，四个 map 在启动时被复制进常量 bank 0，`&params.tma_a` 是一个 `cvta.param` 地址且没有本地副本，`prefetch.tensormap` 为稍后的 `cp.async.bulk.tensor` 把该 map 的缓存行预先取入（取入 tensor-map 描述符缓存，而不是 L2 数据缓存，NVIDIA 内部信息；ISA 没有指名任何缓存；在 SASS 中它是 TMA 控制指令 `UTMACCTL.PF`），而这些字节恰好就是 `gemm.initialize()` 所编码的那些（explicit 路径把它们从 `gemm.params()` 中复制出来）。它还让启动保持为单次 640 字节的参数复制，无需额外分配。
</details>

---

## 14.3 术语表

每个条目：它是什么、这个 kernel 如何使用它、到哪里去看。"W" 编号是 `explicit.cu` 第 3 节的 PTX wrapper（W1 在 `explicit.cu:260` ... W37 在 `:601`）。

**2SM MMA (`cta_group::2`)** - 由一个 pair 中偶数 CTA 的一个线程发射的一条 `tcgen05.mma`，它从两个 CTA 的共享内存中相同的 CTA 相对地址读取 A 和 B 的两半，并把 256 x 128 结果中每个 CTA 的 128 行写入该 CTA 自己的 TMEM。该 kernel 每个 k-tile 发射 4 条（W29，`explicit.cu:516-529`）。

**累加器 stage (`st`)** - 512 个 TMEM 列中的一个 128 列切片：`tmem_c = T + 128 st` (`explicit.cu:919`)；4 个 stage 构成累加器流水线（full 的计数为 1，来自 commit；empty 的计数为 256，来自各 epilogue）。

**AlongN 光栅顺序** - 这样一种 tile-scheduler 光栅化：经过一次转置后，启动网格的 x 维索引 N 而 y 维索引 M，给出 `tm = 2 (y >> 1) + (x & 1)`、`tn = 2 (x >> 1) + (y & 1)`；在计数相等时由启发式规则 `tiles_n > tiles_m ? AlongM : AlongN` 选出（笔记 7.4）。于是连续的 CLC id 对 A 的一个固定的 256 行条带沿 N 行走。

**算术强度** - 跨某个给定接口每移动一个字节所对应的 FLOP 数；对这个 mainloop 而言，每 L2 字节 128 FLOP，TMA 送入共享内存的每字节 85.3 FLOP（如果把 tensor core 对这些字节的读取也算进去，则至多 42.7：它们至少被读取一次，而 `cta_group::2` 并不蕴含 B 只被读取一次，NVIDIA 内部信息），按每 cluster 每 k-tile 计（第 3.3.1 节和第 13.5 节），这是该 kernel 计算受限的原因。

**异步 proxy / 通用 proxy** - PTX 内存模型中内存的两种视图：普通的加载和存储（通用 proxy），以及各异步引擎——TMA (`cp.async.bulk.*`)、`tcgen05.*`、CLC 响应（异步 proxy）。二者之间的排序需要 `fence.proxy.async` (W15)；该 kernel 在每次 TMA 存储之前和每次 CLC 解码之后发射它（PTX ISA 9.4 "Async Proxy"；CUDA Programming Guide 4.12.2.1）。

**原子（layout 原子）** - 一个带 swizzle 的 smem layout 的最小重复 tile：A/B 用 SW128 K-major `Swizzle<3,4,3> o (8,64):(64,1)`（8 行 x 128 B = 1024 B，因此 stage 按 1024 字节对齐），D 用 `Swizzle<2,5,2> o (32,4):(1,32)`（32 个 fp32 x 4 列 = 512 B，因此按 512 字节对齐）。

**bank 冲突** - 一个 warp 的两个或更多 lane 在一条指令中访问映射到同一个共享内存 bank 的不同 32 位字，使该访问串行化；swizzle 的存在就是为了在 TMA、UMMA 和 epilogue 存储的行/列访问模式下避免这一点。

**屏障（命名）/ `bar.sync`、`bar.arrive`** - 按 id 选择并带有线程计数的硬件 CTA 屏障：id 1 x 128 每个 subtile 将 4 个 epilogue warp 同步两次；id 6 x 160 让 warp 0 (`bar.arrive`) 把 TMEM 基址交给 warp 4-7 (`bar.sync`) 而 warp 0 无需等待（W18/W19；id 来自 `cutlass/arch/barrier.h:169-178`）。

**`barrier.cluster.arrive.relaxed` / `.wait`** - cluster 范围的会合 (W16/W17)：每个 CTA 的每个线程先到达，然后等待；它使带 fence 的各 `mbarrier.init` 在任何远程屏障操作之前可见。该 kernel 在这两半之间计算初始 tile。

**Box (TMA)** - 一个 tensor map 每条指令所传输的 tile：A/B 为 (64, 64, 1) fp16 = 8192 B，D 为 (32, 16, 1) fp32 = 2048 B。SWIZZLE_128B 要求内层 box 范围至多为 128 B（CUDA Driver API `cuTensorMapEncodeTiled`；CUDA Programming Guide 4.12.2.2.5，表 25，该处是针对计算能力 9 给出的）。

**Bulk group (`cp.async.bulk.commit_group` / `wait_group.read N`)** - TMA 存储的完成机制：发射线程把多次存储归入一个组，提交该组，而 `wait_group.read N` 阻塞直到至多还有 N 个组仍在读取共享内存 (W24/W25)。只有发射线程才能等待，这就是只有 warp 4 发射存储的原因。

**缓存提示（`L2::cache_hint`，EVICT_NORMAL）** - 每次 TMA 加载上的 64 位操作数 `0x1000000000000000`（`kCacheHintEvictNormal`，`explicit.cu:188`），即 CUTLASS 的默认值 `Sm100MemDescDefault`。

**Carveout** - collective builder 在各流水线之间划分的共享内存预算：可选启用的 232448 B 减去 builder 的 192 B kernel carveout（152 B 的小屏障、32 B 的 CLC 响应槽位、8 B 的 TMEM 基址指针）再减去 33792 B 的 epilogue 存储，得到 floor(198464 / 24592) = 8 个 mainloop stage（笔记 6.1）；由此得到的 230400 B 迫使每个 SM 只能有一个 CTA。

**CLC（Cluster 启动控制）** - Blackwell 的硬件工作窃取原语。`clusterlaunchcontrol.try_cancel` 请求对同一网格中一个尚未启动的 cluster 进行原子取消（该请求可能失败），并返回一个 16 字节的响应，该响应被 multicast 到发出请求的 cluster 的每个 CTA，并以 16 个 tx 字节完成一个 mbarrier。（第 2.12 节）

**CTA tile / cluster tile / MMA tile** - CTA tile：一个 CTA 所拥有的 128 x 128 输出块（它的 TMEM 累加器和它的 D 存储）。cluster tile：2 x 2 cluster 所拥有的 256 x 256 块，即 CLC 分发出去的单位（共 1024 个）。MMA tile：一条 `tcgen05.mma.cta_group::2` 为一个 pair 计算的 256 (M) x 128 (N) x 16 (K) 块。

**描述符（smem，UMMA）** - 描述共享内存中一个 A 或 B 操作数 tile 的 64 位操作数：以 16 字节为单位的 14 位起始地址、LBO = 1 (16 B)、SBO = 1024 B、版本 1、layout 类型 SWIZZLE_128B（高字 `0x40004040`，低字 `0x10000 | ((addr >> 4) & 0x3FFF)`，例如 A 在 stage 0、k-block 0 时为 `0x4000404000010880`；R7）。该 kernel 对每个 stage 和 k-block 用一次对低字的 32 位加法形成它。

**描述符（指令，`idesc`）** - 编码一条 `tcgen05.mma` 的数据格式、主序性和 M/N 的 32 位操作数：这里为 0x10200010（F32 累加、F16 输入、K-major、N = 128、M = 256）。

**DSMEM（分布式共享内存）** - 一个 cluster 中的各 CTA 通过 `shared::cluster` 地址寻址彼此共享内存的能力；这里只用于屏障（远程 `mbarrier.arrive`、TMA/commit multicast、peer 位技巧）和 multicast TMA 目的地，从不用于通用加载/存储。

**`elect.sync`** - 返回一个在已收敛 warp 的恰好一个 lane 中为真的谓词 (W2)。对给定的 `membermask`，选举是确定性的：每次被选中的都是同一个 lane（PTX ISA 9.4；第 6 章，第 6.2.2 节和第 11 章，第 11.3.2 节引述了该规则），而 `.sync` 使每次选举成为一个 warp 收敛点。（第 2.10.3 节）

**Epilogue** - 把 TMEM 累加器变成 D 的阶段：每个 subtile 执行 `tcgen05.ld` -> FMUL/FFMA -> 带 swizzle 的 `st.shared` -> `fence.proxy.async` -> `bar.sync` -> 4 次 TMA 存储 -> `commit_group` -> `wait_group.read 1` -> `bar.sync` (`explicit.cu:1034-1071`)。

**`expect_tx` vs `arrive.expect_tx`** - `mbarrier.expect_tx` 只把字节数加到当前 phase 的 tx-count 上（不计入任何到达）；`mbarrier.arrive.expect_tx` 在同一个操作中既加上字节数又执行一次到达。（第 2.8.2 节）

**`fence.mbarrier_init`** - `fence.mbarrier_init.release.cluster`（`cutlass/arch/barrier.h:712-720`，`fence_barrier_init`）：在 `mbarrier.init` 的四个流水线组每一组之后发射的一个 release fence（单独的 `tmem_dealloc` init，explicit.cu:1165-1167，没有自己的 fence），使得已初始化的屏障字在 `barrier.cluster` 会合让任何远程 arrive、TMA `complete_tx` 或 `tcgen05.commit` 触碰它们之前就在 cluster 范围内可见。（第 6.4.1 节）

**`FusionStorage`** - epilogue fusion 回调的共享内存结构体；对 `Sm90LinearCombination` 而言，它是由各 visitor 节点的空 `SharedStorage` 类型组成的一个 `cute::tuple`，其本身是一个 `sizeof` 为 1 的空类型。（第 5.7.1 节）

**GDC / PDL（`griddepcontrol`，程序化依赖启动）** - `griddepcontrol.wait` 阻塞直到所有前置网格都已完成且已刷出 (W35)。`griddepcontrol.launch_dependents` 发出信号，表示一个依赖网格可以开始启动（warp 0 在其最后一个 tile 之后，W36）。（第 2.13 节）

**`__grid_constant__`** - 按值传递的 kernel 参数块上的限定符，它让 kernel 可以取该参数块的地址（`prefetch.tensormap` 和各 TMA 描述符操作数需要这一点），并把它保留在参数/常量空间中（从 `c[0x0][0x380]` 起 [M, G.3]）。

**HBM** - GPU 的主（高带宽）内存；本次运行的唯一 HBM 字节数为 512 MiB（A 128 MiB、B 128 MiB、D 256 MiB）；光栅所隐含的流量介于 0.54 到 4.70 GB 之间，取决于 B 是否留存在 L2 中，至多约为 HBM 带宽的三分之二，所以 HBM 不是限制因素（第 3.9.5 节，假设，未实测）。

**K-major / MN-major** - 一个 MMA 操作数在内存中哪一维是连续的：K-major 意味着连续的 k 值彼此相邻（这里 A 的行主序 M x K 和 B 的“列主序” N x K 都是 K-major）；在指令描述符中编码为 `a_major = b_major = 0`，并选择 K-major 的 SW128 原子。

**k-tile (`q`) / k-block (`kb`)** - 一个 k-tile 是一个 smem stage 所容纳的 64 宽的 K 条块（每个 tile 128 个）；一个 k-block 是一条 `tcgen05.mma` 所消费的 16 宽的 K 切片（每个 k-tile 4 个），在描述符上以 `+ 32 B`（以 16 字节为单位则为 2）寻址。

**L2 promotion (128 B)** - 一个 tensor-map 编码选项 (`CU_TENSOR_MAP_L2_PROMOTION_L2_128B`)，它把 L2 取数大小，即 L2 请求从 DRAM 得到填充时的字节粒度，设为 128 字节；在全部六个 map 上都已设置 [M, D.1 D3]。

**`__launch_bounds__(256, 1)`** - 告诉 ptxas 该 kernel 启动时每个线程块至多 256 个线程，并且每个 SM 应至少容纳 1 个驻留 CTA，从而允许每个线程最多 255 个寄存器；explicit kernel 使用 26 个 [M, G.3]。

**启动线性 cluster id** - 位于启动网格 cluster 坐标 `(CX, CY) = (blockIdx.x / 2, blockIdx.y / 2)` 处的 cluster 的 id `CY x 32 + CX`（网格 (64, 64) = 32 x 32 个 cluster），也就是对一个 CLC 响应的首 CTA id `(x0, y0)` 而言的 `(y0 / 2) x 32 + x0 / 2`。（第 7.5.4 节和第 12.6.3 节）

**`mapa`** - 把一个 `shared::cta` 地址映射到 cluster 中另一个 CTA 内对应 `shared::cluster` 地址的 PTX 指令 (W10/W13)；被 ptxas 下沉为将 rank 插入字节 3（`PRMT ..., 0x654`）[M, D.2]。

**mbarrier** - 一个 8 字节的共享内存同步对象，带有一个 phase、一个待到达计数和一个事务计数；当到达计数降到零且 tx-count 平衡时，一个 phase 完成。用到的操作：`init`、`arrive`、`arrive.expect_tx`、`try_wait.parity`（阻塞，带挂起提示）、`try_wait.parity`（单次 peek）、`test_wait.parity` (W5-W13)。

**Multicast（TMA / commit）** - 带 16 位 CTA 掩码的 `.multicast::cluster`：一次 TMA 加载把同一个 box 写到每个被掩码选中的 CTA 中相同的 CTA 相对偏移处，并向每个目的地的屏障发出信号；一条 `tcgen05.commit` 在每个被掩码选中的 CTA 中相同偏移处的屏障上到达。掩码：A 为 0x5 << x，release 为 0xF，累加器为 0x3 << 2y（第 4.4.1-4.4.3 节）。

**Occupancy / 驻留** - 有多少个 CTA（cluster）同时驻留：这里每个 SM 一个 CTA（smem 230400 B），33 个 cluster 各 4 个 = 132 个 SM [M, D.3]，而不是 148 / 4 = 37，因为 2 x 2 cluster 需要同一个 GPC 内的两个完整 TPC，而这块 B200 的 skyline（NVIDIA 内部）中每个 TPC 数为奇数的 GPC 都会闲置一个 TPC [A，第 2.1.1 节中的算术]。

**Pair leader / peer** - 一个 MMA pair {2y, 2y+1} 中的偶数 rank 发射 TMEM 分配原子操作（SASS `UTCATOMSWS`，仅限偶数 CTA 的被选中 lane，笔记 D.2；其背后的 pair 集体 PTX `tcgen05.alloc` 由两个 CTA 的 warp 0 执行）、全部 `tcgen05.mma` 和两次 commit，并拥有该 pair 所使用的 `full[]` 和 `acc_empty[]` 屏障；奇数 rank（peer，`r ^ 1`）加载它的两半、运行它的 epilogue，并在 leader 的屏障上到达。

**参数空间 / 常量 bank 0** - `__grid_constant__` kernel 参数所在之处；设备端从那里读取 `k`、`alpha`、`beta` 和四个 tensor map（`ld.param`，在 explicit kernel 中为 SASS `LDCU`，在 CUTLASS baseline 中为 `LDC`/`LDCU`）。

**Peer 位掩码 (`0xFEFFFFFF`)** - 清除一个 `shared::cta` 地址的第 24 位，把一个奇数 CTA 的屏障地址变成其 pair leader 的（第 2.9.3 节）。

**持久化 kernel** - 其各 CTA 在许多输出 tile 上循环而不是算完一个就退出的 kernel；这里每个 CTA 计算约 31-32 个 tile（1024 个 cluster tile 分摊在 33 个驻留 cluster 上；trace 运行实测每个 cluster 32 或 31 个，被探针拖慢的 cluster (0,0) 为 19 个，笔记 D.3），该循环由 CLC 响应驱动。

**Phase / parity** - 一个 mbarrier 的 phase 是完成次数的计数器；`try_wait.parity P` 在 parity 为 P 的 phase 完成后即成功（屏障的当前 phase 具有另一个 parity）。每个流水线状态（producer 或 consumer）在其 stage 索引旁跟踪一个 phase 位；每当索引回绕时该位翻转（第 4.2.3 节）。

**流水线（producer / consumer）** - 一个由 `Stages` 个缓冲区组成的环，每个 stage 各有一个 full 屏障和一个 empty 屏障；producer 等待 empty、填充、然后发出 full 信号（或者让 TMA 字节来完成它）；consumer 等待 full、读取、然后发出 empty 信号。这里有四个由 mbarrier 守护的流水线：mainloop (8)、累加器 (4)、CLC (2)、throttle (2)；epilogue 的 C/D 环 (4) 则改由 bulk group 守护（第 4.1 节）。

**`producer_tail`** - 一个 producer 在其最后一次填充之后额外执行的 `Stages` 次等待，使得 CTA 不会在 consumer 可能仍要在其 empty 屏障上到达时退出（第 4.8 节；第 14.2.2 节中的练习 D7）。

**光栅顺序** - 见 AlongN。

**`scale_c` / `ScaleOut::Zero` vs `One`** - `tcgen05.mma` 的谓词操作数，它选择 `D = A x B`（0，`ScaleOut::Zero`）还是 `D = D + A x B`（1，`ScaleOut::One`）。（第 9.5.2、9.5.3 和 12.2.1 节）

**Scoreboard** - SASS 层面的依赖跟踪（编码在指令控制字中的 per-warp scoreboard；`sm_100a` 保留着这类机制，即由 producer 设置（在带修饰的反汇编中为 `&wr=x`）而由 consumer 要求（`&req={x}`）的 token，这一点是 NVIDIA 内部信息；精确的控制码字段是实现细节，而这里的清单只打印十六进制控制字，所以该编码仍然是一个推断，第 12.1.3 节 [H]），ptxas 用它来让一个 consumer 停顿，直到某个异步 producer（`LDTM`、`STS` 读取、`UTMASTG` 读取）完成。（第 12.1.3 节和第 12.2 节）

**`shared::cta` / `shared::cluster`** - 用于共享内存的 PTX 状态空间，在执行中的 CTA 内寻址（32 位窗口地址，`__cvta_generic_to_shared`）或在 cluster 内寻址（一个 CTA 自己的 `shared::cta` 地址同时也是该 CTA 的一个有效 `shared::cluster` 地址，在这一编译器/硬件上 rank 位于第 [24,32) 位 [M，rank 0..3 位于第 [24,26) 位见 D5；字节 3 的 `PRMT` 下沉见 D.2]）。

**SM / TPC / GPC** - Streaming multiprocessor；一个 Texture Processing Cluster = 一组共享某些资源的 SM（"one or more SM"，Nsight Compute Profiling Guide）；B200 上每个 TPC 2 个，148 个 SM = 74 个 TPC（NVIDIA 内部，第 2.1.1 节；消费级 RTX Blackwell 白皮书公布了相同的数量），并且每个 MMA pair 都被放置在一对 SM (2j, 2j+1) 上 [M, D.3]（把那一对解读为一个 TPC 是笔记的推断）；一个 GPU Processing Cluster = 一组 TPC，一个线程块 cluster 必须被放置在其中；它的 skyline = 已启用 TPC 数的列表，每个 GPC 一个；在两种 B200 skyline（8 或 11 个 GPC，都是 74 个 TPC）中，这块 B200 的是 10/10/10/9/9/9/9/5/1/1/1（两者都是 NVIDIA 内部），这与实测的 132 个 SM 上 33 个 cluster 相符（算术见第 2.1.1 节）。

**Speed of light (SOL)** - 一个 kernel 在其唯一的约束资源以全速运行且零开销时将会有的运行时间；第 3 章建立该模型，G3 应用它。

**Stage (`s`)** - mainloop smem 环的一个槽位：位于 `OFF_SMEM_A + 16384 s` 的 16384 B 的 A（128 x 64 fp16）和位于 `OFF_SMEM_B + 8192 s` 的 8192 B 的 B（64 x 64 fp16），由 `full[s]` / `empty[s]` 守护。

**Subtile (`e`)** - epilogue 每次迭代所处理的 CTA tile 的一个 128 x 16 列切片：每个 warp 一条 `tcgen05.ld.32x32b.x16`、一个 8192 字节的 C/D smem stage (`e % 4`)、四个 TMA 存储 box。

**SW128 / `Swizzle<B,M,S>`** - CuTe 对字节偏移的 XOR swizzle：保留低 M 位，把位置 M 处的 B 个位与高出 S 个位置的 B 个位做 XOR。`Swizzle<3,4,3>`（第 [4,7) 位 ^= 第 [7,10) 位）是 TMA 单元在写入时、UMMA 单元在读取时对 A/B 施加的 SWIZZLE_128B 模式；`Swizzle<2,5,2>`（第 [5,7) 位 ^= 第 [7,9) 位）是 D 的 SWIZZLE_128B_ATOM_32B，epilogue 在 `st_off` 中计算它（`cute/swizzle.hpp:42-96`：文档注释从 42 起，`struct Swizzle` 在 55，`apply` 在 76，`operator()` 在 84）。

**`tcgen05`** - 第 5 代 tensor core 的 PTX 指令族：`alloc/dealloc/relinquish_alloc_permit`（TMEM 管理）、`mma`（UMMA）、`commit`（向一个 mbarrier 通告完成）、`ld`/`wait::ld`（TMEM 到寄存器）。需要诸如 `sm_100a`（本构建所用）这样的架构特定目标，或诸如 `sm_100f` 这样的族特定目标；单纯的 `sm_100` 目标不在受支持的列表中（PTX ISA 9.4，9.7.18.7.1 和 9.7.18.10.10.1 的 Target ISA Notes）。

**Tensor map (`CUtensorMap`)** - 由 `cuTensorMapEncodeTiled` 生成的 128 字节不透明描述符，它告诉 TMA 单元一个张量的基址、shape、stride、box、元素类型和 swizzle；在参数块中传入，并用 `prefetch.tensormap` 预取（W20）。

**Throttle** - 位于 rank 0 上、有 2 个 stage 的 CTA 本地流水线（计数 32/32），它让 scheduler 只有在 warp 2 已经开始加载 tile t 之后才发出查询 t，因此除了正在加载的 tile 之外，每个 cluster 最多只有一个 CLC 查询在途（笔记 7.4；[M] 从未超过一个在途）。

**TMA (Tensor Memory Accelerator)** - 在全局内存与共享内存之间搬运多维 tile 的异步拷贝引擎（`cp.async.bulk.tensor`），其加载经由 mbarrier 的 tx 字节完成，存储经由 bulk group 完成；它在硬件中施加 swizzle 和 multicast。

**TMEM (Tensor Memory)** - Blackwell 的每 SM 256 KB 片上累加器内存：128 个 lane x 512 列的 32 位单元，寻址方式为 `(lane << 16) | column`；由一个 warp 以列为单位分配（二的幂，至少 32）；一个 warp 只能读取 lane `[32 (warp % 4), +32)`（Colfax 的 TMEM 教程；笔记 B.6）。该 kernel 为每个 pair 分配全部 512 列，并持有 4 个累加器 stage。与 TMEM 分配相关联的启动/调度资源是 Virtual Resource Counter（VRC）：一个 CTA 从启动起就持有它，而 `relinquish_alloc_permit` 释放它，使另一个 CTA 得以继续（NVIDIA 内部，第 9.6.2 节）。

**Token (`WaitDone` / `WaitAgain`)** - 单次 `try_wait.parity` peek（W7）的结果，被携带到下一次迭代中，使得只有在 peek 失败时才进入阻塞等待（W6）；该 kernel 在推进之后立刻 peek 下一个 stage。

**tx-count（事务计数）** - 事务 mbarrier 的字节计数器：`arrive.expect_tx` 加上预期的字节数，每次异步拷贝的 `complete_tx` 减去它所交付的字节数；当到达与字节两者都达到平衡时，该 phase 完成（每个 mainloop stage 为 49152，每个 CLC 响应为 16）。

**UMMA** - CUTLASS 对 `tcgen05.mma` 这一族及其描述符的称呼（`cute/arch/mma_sm100_desc.hpp`、`mma_sm100_umma.hpp`）。

**Uniform 数据通路 / UR** - SM 中每个 warp 的标量执行路径，带有 uniform 寄存器 `URn` 和谓词 `UPn`；TMA、MMA、commit 和 CLC 的发射都是 uniform 指令（`UTMALDG`、`UTCHMMA`、`UTCBAR`、`UGETNEXTWORKID`），无论有多少个 lane 到达它们，每个 warp 都只执行一次 [M, D.2]。

**Warpgroup** - 四个连续的 warp（128 个线程），warp id 为 4k..4k+3，是 Hopper 的 `wgmma` 所针对的单位，也是 Blackwell 的 epilogue 为它的 128 线程命名屏障所继承的单位。这里 warp 0-3（MMA、scheduler、producer、空闲的 C 加载者）构成 warpgroup 0，warp 4-7（epilogue）构成 warpgroup 1；正是 `tcgen05.ld` 的 lane 规则 `[32 (warp % 4), +32)` 把 epilogue warp `w = warp_idx - 4` 映射到 TMEM lane `32 w`（第 3.7 节）。

**Warp 特化** - 把不同的角色分配给一个 CTA 的不同 warp（producer、scheduler、MMA、epilogue），使得每一个都运行一个简单的循环并通过屏障通信；这是 `explicit.cu` 第 6-9 节的结构。

**Wave 量化** - 当工作单元的数量不是工作者数量的倍数时的效率损失：1024 个 tile / 33 个 cluster = 31.03（损失约 3%），256 / 33 = 7.76（G3）。

**本讲义中用到的 SASS 助记符** - `UTCHMMA.2CTA`（`tcgen05.mma.cta_group::2`）、`UTCBAR.2CTA.MULTICAST`（`tcgen05.commit` multicast）、`UTCATOMSWS`（TMEM alloc/dealloc 原子操作）、`LDTM.x16`（`tcgen05.ld`）、`UTMALDG.3D[.MULTICAST].2CTA`（TMA 加载）、`UTMASTG.3D`（TMA 存储）、`UTMACMDFLUSH`（`commit_group`）、`DEPBAR.LE SB0, N`（`wait_group.read N`）、`UTMACCTL.PF`（`prefetch.tensormap`）、`SYNCS.PHASECHK.TRANS64[.TRYWAIT]`（`mbarrier.test_wait/try_wait`）、`SYNCS.ARRIVE.TRANS64[.RED]`（`mbarrier.arrive`；`.RED` 出现在远程 arrive 上，对远程目标是必需的，在本地也合法，NVIDIA 内部）、`SYNCS.EXCH.64`（`mbarrier.init`）、`UGETNEXTWORKID.BROADCAST`（`try_cancel`）、`ACQBULK`（`griddepcontrol.wait`）、`PREEXIT`（`launch_dependents`）、`UCGABAR_ARV/WAIT`（`barrier.cluster`）、`FENCE.VIEW.ASYNC.S` + `MEMBAR.ALL.CTA`（`fence.proxy.async`）[M，笔记 D.2 词典]。

---

## 14.4 数字卡片

以下所有内容都是来自 `explicit.cu`/`util.hpp` 的 [S]，除非标注了 [M]（实测，笔记的 D 部分和 G 部分）。

**字节与操作总量**

| 项目 | 值 |
|---|---|
| 每个 tile 每个 cluster | 来自 L2 的 8 MiB；每个 CTA 256 次 TMA 加载；每个 pair 512 条 MMA（128 个 k-tile x 4，由 pair leader 发射；每个 cluster 1024 条）；每个 CTA 32 条 `tcgen05.ld` + 32 次 TMA 存储 |

**实测结果 [M，笔记 G]**

| 项目 | 值 |
|---|---|
| 运行时间 | explicit 的中位数 0.889354 ms（最小 0.888947，最大 0.890307）；未改动版的中位数 0.889917 ms；比值 0.99937；`Avg runtime` 是 harness 的 `GpuTimer` 围绕 10 对 `initialize()` + `run()` 的计时区间（笔记第 12 节），因此它是 kernel 时间的上界（第 3.9.2 节） |
| GFLOPS | 中位数那次运行为 1.236e6（2 x 8192^3 / 0.889354 ms = 1.2363e6）；在五对运行上，explicit 的范围为 1.23498e6-1.23687e6，未改动版为 1.23501e6-1.23728e6 [M, G.2]；2026-09-09 baseline 的 1.23671e6-1.23802e6（D.4）是另一天的运行 |
| 机器 | B200，148 个 SM，CC 10.0，负载下 SM 1965 MHz，内存 3996 MHz；在整个 2026-09-11 时钟日志上功率为 188.6-306.1 W（中位数 299.9 W）（108 个样本，横跨交错进行的 explicit 与未改动版的运行）[M, G.2]（300-308 W 的稳态、峰值 308.42 W，是 2026-09-09 的 baseline 运行，D.4）；CUDA 13.3（nvcc 13.3.73），驱动 610.57.04 |
| 推导出的速率 [M/A] | 每个 tile 27.8 us，在 1965 MHz 下每个 k-tile 217 ns = 427 个周期（0.889354 ms / 32 个 tile / 128 个 k-tile；CUTLASS baseline 的 0.888 ms 给出 426，笔记 D.4，即第 3 章，第 3.9.2 节以及第 12.5.5 节和第 13.5 节的工作数值；均为 kernel 时间的上界，第 3.9.2 节的注意事项）；每条 256 x 128 x 16 MMA 约 107 个周期；整次运行的平均值 1.0995e12 / (0.889354 ms x 1.965 GHz x 132 个 SM) = 每个活跃 SM 每周期约 4766 FLOP（包含 wave 量化和固定开销）；由 k-tile 周期所隐含的稳态速率 2 x 128 x 128 x 64 / 426.7 个周期 = 每个 SM 每周期约 4915 FLOP |

---
