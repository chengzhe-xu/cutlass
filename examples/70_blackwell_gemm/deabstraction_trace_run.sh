#!/usr/bin/env bash
# De-abstraction trace experiments for examples/70_blackwell_gemm (Part C, C.5 of
# Semantics-preserving-de-abstraction.md).  Run from the CUTLASS repository root on the B200 machine.
#
#   ./examples/70_blackwell_gemm/deabstraction_trace_run.sh all          # C.5 steps 1-6 in order
#   ./examples/70_blackwell_gemm/deabstraction_trace_run.sh <step>       # one of: build inspect baseline trace restore check
#
# Only the fixed build command and the fixed run command of Section 0 are used.  Trace mode is the one-line
# toggle at the top of 70_blackwell_fp16_gemm.cu, flipped by this script with sed and flipped back afterwards.
# Outputs go to $TRACE_OUT (default ./trace_out).
set -euo pipefail

: "${CUDACXX:?set CUDACXX to the nvcc of CUDA 13.3}"
: "${CUDA_HOME:?set CUDA_HOME to the CUDA 13.3 toolkit root}"

ROOT=$(pwd)
OUT=${TRACE_OUT:-$ROOT/trace_out}
EX=examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu
BIN=build/examples/70_blackwell_gemm/70_blackwell_fp16_gemm
PRISTINE=$OUT/70_blackwell_fp16_gemm.pristine
mkdir -p "$OUT"

build() {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDAToolkit_ROOT="$CUDA_HOME" \
        -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_ENABLE_EXAMPLES=ON -DCUTLASS_ENABLE_TESTS=ON -DCUTLASS_ENABLE_PROFILER=ON \
    && cmake --build build --target 70_blackwell_fp16_gemm --parallel 16
}
run_fixed() { "$BIN" --m=8192 --n=8192 --k=8192; }
toggle_on()  { sed -i '1s|^// #define CUTLASS_DEABSTRACTION_TRACE 1|#define CUTLASS_DEABSTRACTION_TRACE 1|' "$EX"; }
toggle_off() { sed -i '1s|^#define CUTLASS_DEABSTRACTION_TRACE 1|// #define CUTLASS_DEABSTRACTION_TRACE 1|' "$EX"; }
toggle_state() { head -1 "$EX"; }

SMI_PID=""
on_exit() {
  rc=$?
  if [ -n "$SMI_PID" ]; then kill "$SMI_PID" 2>/dev/null || true; fi
  if head -1 "$EX" | grep -q '^#define CUTLASS_DEABSTRACTION_TRACE 1'; then
    echo "NOTE: the trace toggle is still ON in $EX (a step failed or 'trace' was run alone); run '$0 restore' before taking any performance number" >&2
  fi
  exit $rc
}
trap on_exit EXIT

step_build() {      # C.5 step 1: pristine (toggle off) build, keep a copy of the binary
  toggle_off
  echo "toggle: $(toggle_state)"
  build
  cp "$BIN" "$PRISTINE"
  echo "pristine binary saved to $PRISTINE"
}

step_inspect() {    # C.5 step 2: build configuration (B3), resources and SASS (B1, B2)
  {
    grep -E "CUTLASS_NVCC_ARCHS|CMAKE_BUILD_TYPE|CUTLASS_ENABLE_GDC_FOR_SM100|CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL|CUTLASS_ENABLE_CUDA_HOST_ADAPTER|CMAKE_CUDA_FLAGS|CMAKE_CUDA_COMPILER" build/CMakeCache.txt || true
    echo "--- flags.make ---"
    cat build/examples/70_blackwell_gemm/CMakeFiles/70_blackwell_fp16_gemm.dir/flags.make || true
    echo "--- toolchain ---"
    "$CUDACXX" --version
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
  } > "$OUT/config.txt" 2>&1
  cuobjdump --dump-resource-usage "$PRISTINE" > "$OUT/resources.txt" 2>&1 || true
  cuobjdump --list-elf "$PRISTINE" > "$OUT/elf.txt" 2>&1 || true
  cuobjdump --dump-sass "$PRISTINE" > "$OUT/sass.txt" 2>&1 || true
  cuobjdump --dump-ptx "$PRISTINE" > "$OUT/ptx.txt" 2>&1 || echo "(no embedded PTX)" > "$OUT/ptx.txt"
  {
    echo "--- kernel function(s) ---"
    grep -n "Function :" "$OUT/sass.txt" | head -20
    echo "--- mnemonic families (GemmUniversal kernel only; predicated lines included) ---"
    awk '/Function :/{p=/GemmUniversal/} p' "$OUT/sass.txt" > "$OUT/sass_gemm.txt"
    [ -s "$OUT/sass_gemm.txt" ] || echo "WARNING: no 'Function :' line containing GemmUniversal in sass.txt"
    for pat in UTC UTMA SYNCS CLC ACQBULK PREEXIT FFMA FMUL FADD 'STS' 'LDS' 'BAR' 'ELECT'; do
      printf "%-10s %s\n" "$pat" "$(grep -c -E "^[[:space:]]*/\*[0-9a-f]+\*/[[:space:]]+(@!?U?PT?[0-9]*[[:space:]]+)?${pat}" "$OUT/sass_gemm.txt" || true)"
    done
    echo "--- griddepcontrol in PTX ---"
    grep -c griddepcontrol "$OUT/ptx.txt" || true
  } > "$OUT/sass_summary.txt" 2>&1
  echo "inspection written to $OUT/{config,resources,elf,sass,ptx,sass_summary}.txt"
}

step_baseline() {   # C.5 step 3: R1, the fixed run command five times, clocks sampled alongside
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -lms 100 > "$OUT/clocks.csv" 2>&1 &
    SMI_PID=$!
  fi
  : > "$OUT/baseline.txt"
  for i in 1 2 3 4 5; do
    echo "=== run $i ===" >> "$OUT/baseline.txt"
    run_fixed | tee -a "$OUT/baseline.txt"
  done
  if [ -n "$SMI_PID" ]; then kill "$SMI_PID" 2>/dev/null || true; SMI_PID=""; fi
  nvidia-smi -q -d CLOCK > "$OUT/clocks_query.txt" 2>&1 || true
  echo "baseline written to $OUT/baseline.txt"
}

step_trace() {      # C.5 step 4: toggle on, same build command, fixed run command once
  toggle_on
  echo "toggle: $(toggle_state)"
  build
  rm -f launch0.csv
  run_fixed 2>&1 | tee "$OUT/host.txt"
  mv launch0.csv "$OUT/launch0.csv"
  echo "trace outputs: $OUT/host.txt $OUT/launch0.csv"
}

step_restore() {    # C.5 step 5: toggle off, rebuild, prove the performance binary is unchanged
  toggle_off
  echo "toggle: $(toggle_state)"
  build
  if cmp "$BIN" "$PRISTINE"; then
    echo "OK: toggle-off binary is byte-identical to the pristine binary"
  else
    echo "binaries differ; comparing SASS instead"
    cuobjdump --dump-sass "$BIN" > "$OUT/sass_after.txt" 2>&1 || true
    if diff -q "$OUT/sass.txt" "$OUT/sass_after.txt"; then
      echo "OK: SASS identical"
    else
      echo "WARNING: SASS differs between pristine and toggle-off builds (see $OUT/sass_after.txt)"
    fi
  fi
  git diff --stat -- include/ examples/70_blackwell_gemm/ > "$OUT/git_diff_stat.txt" || true
  git diff -- include/ > "$OUT/git_diff_include.patch" || true
  echo "source diff summary in $OUT/git_diff_stat.txt (only guarded blocks are expected under include/)"
}

step_check() {      # C.5 step 6
  python3 examples/70_blackwell_gemm/check_trace.py "$OUT/launch0.csv" "$OUT/host.txt" | tee "$OUT/checklist.txt"
}

case "${1:-all}" in
  build)    step_build ;;
  inspect)  step_inspect ;;
  baseline) step_baseline ;;
  trace)    step_trace ;;
  restore)  step_restore ;;
  check)    step_check ;;
  all)      step_build; step_inspect; step_baseline; step_trace; step_restore; step_check ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
