#!/usr/bin/env bash
# De-abstraction trace experiments for examples/70_blackwell_gemm (Part C, C.5 and Part E, E.7 of
# Semantics-preserving-de-abstraction.md).  Run from the CUTLASS repository root on the B200 machine.
#
#   ./examples/70_blackwell_gemm/deabstraction_trace_run.sh all          # F.5 / E.7.2 B5 order: trace check build inspect baseline restore (stops after check unless the trace run Passed with 0 FAIL)
#   ./examples/70_blackwell_gemm/deabstraction_trace_run.sh <step>       # one of: build inspect baseline trace restore check
#
# Only the fixed build command and the fixed run command of Section 0 are used.  Trace mode is the one-line
# toggle on line 1 of 70_blackwell_fp16_gemm_explicit_util.hpp (Part E; both .cu files include it first), flipped by
# this script with sed and flipped back afterwards.
#
# Part E (explicit kernel) conventions:
#   - outputs go to $TRACE_OUT, default ./trace_out_v2 (the committed ./trace_out holds the 2026-09-09 baseline run and
#     the pristine baseline binary trace_out/70_blackwell_fp16_gemm.pristine, which must not be overwritten: E.9 item 17);
#   - `build` saves the toggle-off binary of the new source as $TRACE_OUT/70_blackwell_fp16_gemm.explicit (not "pristine");
#   - `inspect` cuts the explicit kernel out of the SASS/PTX by the name substring $KNAME (default
#     explicit_blackwell_fp16_gemm_kernel) and also cuts the CUTLASS kernel (GemmUniversal) that stays in the binary;
#   - `baseline` interleaves five runs of the new binary with five runs of the pristine baseline binary (E.8 C2-C3);
#   - `restore` rebuilds toggle-off and proves that the explicit kernel's SASS is identical to the `build` step's (E.8 D6).
set -euo pipefail

: "${CUDACXX:?set CUDACXX to the nvcc of CUDA 13.3}"
: "${CUDA_HOME:?set CUDA_HOME to the CUDA 13.3 toolkit root}"

ROOT=$(pwd)
OUT=${TRACE_OUT:-$ROOT/trace_out_v2}
EX=examples/70_blackwell_gemm/70_blackwell_fp16_gemm_explicit_util.hpp   # line 1 = the toggle (E.2.3)
BIN=build/examples/70_blackwell_gemm/70_blackwell_fp16_gemm
EXPLICIT_BIN=$OUT/70_blackwell_fp16_gemm.explicit                        # toggle-off build of the new source (saved by `build`)
PRISTINE=${PRISTINE_BIN:-$ROOT/trace_out/70_blackwell_fp16_gemm.pristine}   # the 2026-09-09 baseline binary (read-only here)
KNAME=${KNAME:-explicit_blackwell_fp16_gemm_kernel}
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

step_build() {      # F.5 step 3: toggle-off build of the new source, keep a copy of the binary (never overwrites the baseline pristine)
  toggle_off
  echo "toggle: $(toggle_state)"
  build
  cp "$BIN" "$EXPLICIT_BIN"
  echo "toggle-off explicit binary saved to $EXPLICIT_BIN"
}

sass_of_kernel() {  # $1 = full SASS dump, $2 = function-name substring, $3 = output: the SASS of that kernel only
  awk -v pat="$2" '/^Fatbin /{p=0} /Function :/{p=index($0, pat) > 0} p' "$1" > "$3"
  [ -s "$3" ] || echo "WARNING: no 'Function :' line containing $2 in $1"
}
ptx_of_kernel() {   # $1 = full PTX dump, $2 = .entry name substring, $3 = output: the .entry of that kernel only
  awk -v pat="$2" '/^Fatbin /{p=0} /^\.visible \.entry|^\.entry/{p=index($0, pat) > 0} p' "$1" > "$3"
  [ -s "$3" ] || echo "WARNING: no .entry containing $2 in $1"
}

step_inspect() {    # F.5 step 4 (C.5 step 2): build configuration (B3), resources and SASS (B1, B2)
  {
    grep -E "CUTLASS_NVCC_ARCHS|CMAKE_BUILD_TYPE|CUTLASS_ENABLE_GDC_FOR_SM100|CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL|CUTLASS_ENABLE_CUDA_HOST_ADAPTER|CMAKE_CUDA_FLAGS|CMAKE_CUDA_COMPILER" build/CMakeCache.txt || true
    echo "--- flags.make ---"
    cat build/examples/70_blackwell_gemm/CMakeFiles/70_blackwell_fp16_gemm.dir/flags.make || true
    echo "--- toolchain ---"
    "$CUDACXX" --version
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
  } > "$OUT/config.txt" 2>&1
  cuobjdump --dump-resource-usage "$EXPLICIT_BIN" > "$OUT/resources.txt" 2>&1 || true
  cuobjdump --list-elf "$EXPLICIT_BIN" > "$OUT/elf.txt" 2>&1 || true          # two sm_100a cubins expected (one per translation unit, E.3.2)
  cuobjdump --dump-sass "$EXPLICIT_BIN" > "$OUT/sass.txt" 2>&1 || true
  cuobjdump --dump-ptx "$EXPLICIT_BIN" > "$OUT/ptx.txt" 2>&1 || echo "(no embedded PTX)" > "$OUT/ptx.txt"
  sass_of_kernel "$OUT/sass.txt" "$KNAME" "$OUT/sass_explicit.txt"
  sass_of_kernel "$OUT/sass.txt" "GemmUniversal" "$OUT/sass_gemm.txt"        # the never-launched CUTLASS kernel that stays in the binary (E.3.3)
  ptx_of_kernel "$OUT/ptx.txt" "$KNAME" "$OUT/ptx_explicit.txt"
  {
    echo "--- kernel function(s) ---"
    grep -n "Function :" "$OUT/sass.txt" | head -20
    echo "--- resource usage of the explicit kernel (E.8 D1/D2: LOCAL 0, STACK 0, SHARED 1024; REG <= 68 soft) ---"
    grep -A1 "$KNAME" "$OUT/resources.txt" | grep -E "REG|Function" || true   # the REG/STACK/SHARED/LOCAL line follows the Function line
    echo "--- mnemonic families (explicit kernel only; predicated lines included) ---"
    for pat in UTC UTMA SYNCS CLC ACQBULK PREEXIT FFMA FMUL FADD 'STS' 'LDS' 'BAR' 'ELECT' 'LDL' 'STL'; do
      printf "%-10s %s\n" "$pat" "$(grep -c -E "^[[:space:]]*/\*[0-9a-f]+\*/[[:space:]]+(@!?U?PT?[0-9]*[[:space:]]+)?${pat}" "$OUT/sass_explicit.txt" || true)"
    done
    echo "--- griddepcontrol in the explicit kernel's PTX ---"
    grep -c griddepcontrol "$OUT/ptx_explicit.txt" || true
    echo "--- CUTLASS kernel SASS unchanged from the baseline? (E.3.3) ---"
    if [ -f "$ROOT/trace_out/sass_gemm.txt" ]; then
      if diff -q -B "$ROOT/trace_out/sass_gemm.txt" "$OUT/sass_gemm.txt" >/dev/null; then echo "identical"; else echo "DIFFERS (see diff -B trace_out/sass_gemm.txt $OUT/sass_gemm.txt)"; fi
    fi
  } > "$OUT/sass_summary.txt" 2>&1
  python3 examples/70_blackwell_gemm/check_sass.py "$OUT/ptx_explicit.txt" "$OUT/sass_explicit.txt" "$OUT/resources.txt" "$KNAME" | tee "$OUT/sass_checklist.txt" || true
  echo "inspection written to $OUT/{config,resources,elf,sass,ptx,sass_explicit,ptx_explicit,sass_gemm,sass_summary,sass_checklist}.txt"
}

step_baseline() {   # F.5 step 5 (E.8 C1-C3): five runs of the new binary interleaved with five runs of the baseline binary, clocks sampled alongside
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -lms 100 > "$OUT/clocks.csv" 2>&1 &
    SMI_PID=$!
  fi
  : > "$OUT/baseline.txt"
  : > "$OUT/timing_explicit.txt"
  : > "$OUT/timing_pristine.txt"
  if [ -x "$PRISTINE" ]; then
    { echo "--- ldd of the baseline binary ---"; ldd "$PRISTINE"; echo "--- ldd of the new binary ---"; ldd "$BIN"; } > "$OUT/ldd.txt" 2>&1 || true
  else
    echo "NOTE: baseline binary $PRISTINE not found or not executable; only the new binary is timed (set PRISTINE_BIN=...)" | tee -a "$OUT/baseline.txt"
  fi
  for i in 1 2 3 4 5; do
    if [ -x "$PRISTINE" ]; then
      echo "=== baseline (pristine) run $i ===" | tee -a "$OUT/baseline.txt"
      "$PRISTINE" --m=8192 --n=8192 --k=8192 | tee -a "$OUT/baseline.txt" | tee -a "$OUT/timing_pristine.txt"
    fi
    echo "=== explicit run $i ===" | tee -a "$OUT/baseline.txt"
    run_fixed | tee -a "$OUT/baseline.txt" | tee -a "$OUT/timing_explicit.txt"
  done
  if [ -n "$SMI_PID" ]; then kill "$SMI_PID" 2>/dev/null || true; SMI_PID=""; fi
  nvidia-smi -q -d CLOCK > "$OUT/clocks_query.txt" 2>&1 || true
  python3 - "$OUT/timing_explicit.txt" "$OUT/timing_pristine.txt" <<'PY' | tee "$OUT/timing_summary.txt"
import re, statistics, sys
def read(p):
    try:
        vals = [float(m.group(1)) for m in re.finditer(r"Avg runtime: ([0-9.eE+-]+) ms", open(p).read())]
        disp = re.findall(r"Disposition: (\w+)", open(p).read())
        return vals, disp
    except FileNotFoundError:
        return [], []
e, ed = read(sys.argv[1]); b, bd = read(sys.argv[2])
print(f"explicit  : n={len(e)} dispositions={ed} min={min(e) if e else None} median={statistics.median(e) if e else None} max={max(e) if e else None}")
print(f"pristine  : n={len(b)} dispositions={bd} min={min(b) if b else None} median={statistics.median(b) if b else None} max={max(b) if b else None}")
if e and b:
    r = statistics.median(e) / statistics.median(b)
    tier = "ACCEPT (<= +0.2%)" if r <= 1.002 else ("EXPLAIN (+0.2% .. +0.5%)" if r <= 1.005 else "FAIL (> +0.5%)")
    print(f"ratio explicit/pristine medians = {r:.5f} -> {tier} (E.8 C2)")
    print(f"same-day pristine median vs 2026-09-09 median 0.888659: {statistics.median(b) / 0.888659:.5f} (gate +-0.3%, E.8 C3)")
PY
  echo "timings written to $OUT/baseline.txt, $OUT/timing_summary.txt"
}

step_trace() {      # F.5 step 1 (C.5 step 4, E.7.2 B5): toggle on, same build command, fixed run command once (hang bound 20 s + hang records)
  toggle_on
  echo "toggle: $(toggle_state)"
  build
  rm -f launch0.csv
  set +e
  run_fixed 2>&1 | tee "$OUT/host.txt"
  rc=${PIPESTATUS[0]}
  set -e
  [ -f launch0.csv ] && mv launch0.csv "$OUT/launch0.csv"
  echo "trace run exit code $rc; outputs: $OUT/host.txt $OUT/launch0.csv"
  grep -E "^TRACE_HANG|^TRACE_B4|Disposition|Got CUTLASS error|Got CUDA error" "$OUT/host.txt" | head -80 || true
}

step_restore() {    # F.5 step 6 (E.8 D6): toggle off, rebuild, prove that the explicit kernel's SASS equals the `build` step's (binary cmp is not reproducible, D.0)
  toggle_off
  echo "toggle: $(toggle_state)"
  build
  cuobjdump --dump-sass "$BIN" > "$OUT/sass_after.txt" 2>&1 || true
  sass_of_kernel "$OUT/sass_after.txt" "$KNAME" "$OUT/sass_explicit_after.txt"
  if [ -f "$OUT/sass_explicit.txt" ] && diff -q "$OUT/sass_explicit.txt" "$OUT/sass_explicit_after.txt"; then
    echo "OK: the explicit kernel's SASS is identical in the two toggle-off builds (E.8 D6)"
  else
    echo "WARNING: the explicit kernel's SASS differs between the two toggle-off builds, or 'inspect' was not run (see $OUT/sass_explicit_after.txt)"
  fi
  git diff --stat -- include/ examples/70_blackwell_gemm/ > "$OUT/git_diff_stat.txt" || true
  git diff -- include/ > "$OUT/git_diff_include.patch" || true
  echo "source diff summary in $OUT/git_diff_stat.txt (only guarded blocks are expected under include/)"
}

step_check() {      # F.5 step 2 (C.5 step 6, E.8 B2): the record checker, now including the Part E section 7 checks
  python3 examples/70_blackwell_gemm/check_trace.py "$OUT/launch0.csv" "$OUT/host.txt" | tee "$OUT/checklist.txt"
}

case "${1:-all}" in
  build)    step_build ;;
  inspect)  step_inspect ;;
  baseline) step_baseline ;;
  trace)    step_trace ;;
  restore)  step_restore ;;
  check)    step_check ;;
  all)      step_trace; step_check
            grep -q 'Disposition: Passed' "$OUT/host.txt" || { echo "trace run did not print 'Disposition: Passed'; stopping before the toggle-off runs (E.7.2 B5)" >&2; exit 1; }
            grep -q 'SUMMARY: .* PASS, 0 FAIL' "$OUT/checklist.txt" || { echo "check_trace.py reported FAIL lines; stopping before the toggle-off runs (E.7.2 B5)" >&2; exit 1; }
            step_build; step_inspect; step_baseline; step_restore ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
