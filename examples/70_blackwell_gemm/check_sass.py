#!/usr/bin/env python3
"""Structural-equivalence checks of the explicit kernel (Part E, E.7.4 / E.8 D1-D4 of Semantics-preserving-de-abstraction.md).

Usage:
    python3 check_sass.py ptx_explicit.txt sass_explicit.txt resources.txt [kernel-name-substring] [baseline_ptx.txt baseline_sass_gemm.txt]

    ptx_explicit.txt   the .entry of the explicit kernel cut out of `cuobjdump --dump-ptx` (deabstraction_trace_run.sh inspect)
    sass_explicit.txt  the explicit kernel's function cut out of `cuobjdump --dump-sass`
    resources.txt      `cuobjdump --dump-resource-usage` of the whole binary
    kernel-name        substring of the mangled kernel name (default explicit_blackwell_fp16_gemm_kernel)
    baseline files     optional: trace_out/ptx.txt (the GEMM .entry is cut out automatically) and trace_out/sass_gemm.txt, whose
                       counts are printed next to the explicit kernel's as a side-by-side reference (the baseline includes the
                       beta != 0 path that the explicit kernel omits, so those reference counts are informational)

Prints one PASS/FAIL/INFO line per row of E.7.4 (hard rows -> FAIL, soft rows -> WARN, informational -> INFO) and exits
non-zero if any hard row failed.  Values that the PTX carries in mov.b32-materialised registers (transaction bytes,
barrier ids/thread counts) are checked through the feeding `mov.b32` / SASS immediates, never by a literal match on
the instruction line (E.7.4).
"""
import re
import sys

results = []


def report(kind, name, ok, detail, where):
    tag = {"hard": "PASS" if ok else "FAIL", "soft": "PASS" if ok else "WARN", "info": "INFO"}[kind]
    if kind != "info":
        results.append((name, bool(ok), kind))
    print(f"{tag} {name}: {detail}" + ("" if ok or kind == "info" else f"   [target: {where}]"))


# ---------------------------------------------------------------- PTX
def ptx_instructions(text):
    """Instruction lines of a PTX body: strip comments, labels, directives, braces."""
    out = []
    for line in text.splitlines():
        s = line.split("//")[0].strip()
        if not s or s.startswith(".") or s in ("{", "}") or s.endswith(":"):
            continue
        out.append(s)
    return out


def ptx_count(instrs, pattern):
    rx = re.compile(pattern)
    return sum(1 for s in instrs if rx.search(s))


def cut_baseline_entry(text):
    """The GEMM .entry of trace_out/ptx.txt (the file also holds four reference kernels)."""
    parts = re.split(r"(?m)^(?=\.visible \.entry|\.entry)", text)
    for p in parts:
        if "GemmUniversal" in p.split("\n", 1)[0]:
            return p
    return text


PTX_ROWS = [
    # (name, regex, expected, kind, target note)
    ("tcgen05.mma", r"^tcgen05\.mma\.cta_group::2\.kind::f16\b", 4, "hard", "E.7.4 / D.7: 4 per k-tile"),
    ("tcgen05.commit", r"^tcgen05\.commit\.cta_group::2\.mbarrier::arrive::one\.shared::cluster\.multicast::cluster\.b64\b", 2, "hard", "mainloop release + accumulator commit"),
    ("tcgen05.ld.32x32b.x16", r"^tcgen05\.ld\.sync\.aligned\.32x32b\.x16\.b32\b", 8, "hard", "8 subtiles"),
    ("tcgen05.wait::ld", r"^tcgen05\.wait::ld\.sync\.aligned\b", 1, "hard", "one per tile before the release"),
    ("tcgen05.alloc", r"^tcgen05\.alloc\.cta_group::2\b", 1, "hard", "7.6"),
    ("tcgen05.dealloc", r"^tcgen05\.dealloc\.cta_group::2\b", 1, "hard", "7.6"),
    ("tcgen05.relinquish", r"^tcgen05\.relinquish_alloc_permit\.cta_group::2\b", 1, "hard", "7.6"),
    ("tma_load_A_multicast", r"^cp\.async\.bulk\.tensor\.3d\.cta_group::2\.shared::cluster\.global\.mbarrier::complete_tx::bytes\.multicast::cluster\.L2::cache_hint\b", 2, "hard", "one per producer loop copy (8 + 120)"),
    ("tma_load_B", r"^cp\.async\.bulk\.tensor\.3d\.cta_group::2\.shared::cluster\.global\.mbarrier::complete_tx::bytes\.L2::cache_hint\b", 2, "hard", "one per producer loop copy"),
    ("tma_load_C (beta != 0, must be absent)", r"^cp\.async\.bulk\.tensor\.3d\.shared::cluster\.global\.mbarrier::complete_tx::bytes\.L2::cache_hint\b", 0, "hard", "C path omitted"),
    ("tma_store_D", r"^cp\.async\.bulk\.tensor\.3d\.global\.shared::cta\.bulk_group\b", 32, "hard", "4 boxes x 8 subtiles"),
    ("commit_group", r"^cp\.async\.bulk\.commit_group\b", 8, "hard", "one per subtile"),
    ("wait_group.read 1", r"^cp\.async\.bulk\.wait_group\.read 1\b", 8, "hard", "one per subtile"),
    ("wait_group.read 0 (must be absent)", r"^cp\.async\.bulk\.wait_group\.read 0\b", 0, "hard", "E.6.7: no store_tail"),
    ("clc.try_cancel", r"^clusterlaunchcontrol\.try_cancel\b", 1, "hard", "7.4"),
    ("griddepcontrol.wait", r"^griddepcontrol\.wait\b", 2, "hard", "warp 2 and rank-0 warp 1"),
    ("griddepcontrol.launch_dependents", r"^griddepcontrol\.launch_dependents\b", 1, "hard", "warp 0"),
    ("prefetch.tensormap", r"^prefetch\.tensormap\b", 4, "hard", "A, B, C, D (E.10 item 7)"),
    ("mbarrier.init", r"^mbarrier\.init\.shared::cta\.b64\b", 33, "hard", "33-init variant (E.10 item 6)"),
    ("arrive.expect_tx local", r"^mbarrier\.arrive\.expect_tx\.shared::cta\.b64\b", 2, "hard", "one per producer loop copy (49152 in a register)"),
    ("arrive.expect_tx remote", r"^@p mbarrier\.arrive\.expect_tx\.shared::cluster\.b64\b|^mbarrier\.arrive\.expect_tx\.shared::cluster\.b64\b", 1, "hard", "CLC arming (16 in a register)"),
    ("fence.mbarrier_init", r"^fence\.mbarrier_init\.release\.cluster\b", 4, "hard", "4 init groups"),
    ("barrier.cluster.arrive.relaxed", r"^barrier\.cluster\.arrive\.relaxed\.aligned\b", 1, "hard", "7.3"),
    ("barrier.cluster.wait", r"^barrier\.cluster\.wait\.aligned\b", 1, "hard", "7.3"),
    ("bar.sync", r"^bar\.sync\b", 17, "hard", "16 x (1,128) + 1 x (6,160) (ids/counts in registers)"),
    ("bar.arrive", r"^bar\.arrive\b", 1, "hard", "(6,160)"),
    ("mul.f32", r"^mul\.f32\b", 128, "hard", "Section 8: alpha * acc"),
    ("fma.rn.f32", r"^fma\.rn\.f32\b", 128, "hard", "Section 8: fmaf(beta, 0, t)"),
    ("add.f32 (must be absent)", r"^add\.f32\b", 0, "hard", "Section 8: no separate add"),
    ("st.shared.b32", r"^st\.shared\.b32\b", 128, "hard", "16 per subtile"),
    ("generic st.b32/u32/f32 (must be absent)", r"^st\.(b32|u32|f32)\b", 0, "hard", "E.6.1: shared-window stores"),
    ("mbarrier.test_wait", r"^@P2 mbarrier\.test_wait\.parity\b|^mbarrier\.test_wait\.parity\b", 2, "soft", "CLC producer_tail"),
    ("elect.sync (>= 9)", r"^elect\.sync\b", None, "soft", "one per TMA pair, MMA, commit, query + prologue"),
]
PTX_ABSENT = [
    ("cp.async.ca/cg", r"^cp\.async\.c[ag]\b"), ("wgmma", r"^wgmma"), ("st.global", r"^st\.global\b"), ("atom", r"^atom\."),
    ("trap", r"^trap\b"), ("ld.global", r"^ld\.global\b"), ("ld.local/st.local", r"^(ld|st)\.local\b"), ("stmatrix/ldmatrix", r"^(st|ld)matrix"),
    ("tcgen05.fence", r"^tcgen05\.fence"),
]


def check_ptx(path, base_path):
    print("== PTX inventory (E.7.4; exact contract unless marked soft) ==")
    text = open(path, errors="replace").read()
    instrs = ptx_instructions(text)
    base = None
    if base_path:
        base = ptx_instructions(cut_baseline_entry(open(base_path, errors="replace").read()))
    if not instrs:
        report("hard", "ptx_entry_present", False, f"no instructions in {path}", "deabstraction_trace_run.sh inspect (ptx_of_kernel)")
        return
    report("info", "ptx_instructions", True, f"{len(instrs)} instruction lines" + (f" (baseline GEMM .entry: {len(base)})" if base else ""), "")
    for name, rx, exp, kind, where in PTX_ROWS:
        got = ptx_count(instrs, rx)
        ref = f" [baseline {ptx_count(base, rx)}]" if base else ""
        if exp is None:
            report(kind, name, got >= 9, f"{got}{ref}", where)
        else:
            report(kind, name, got == exp, f"{got} expected {exp}{ref}", where)
    n_dec = ptx_count(instrs, r"^clusterlaunchcontrol\.query_cancel\.is_canceled\b")
    n_first = ptx_count(instrs, r"^@p1 clusterlaunchcontrol\.query_cancel\.get_first_ctaid\b|^clusterlaunchcontrol\.query_cancel\.get_first_ctaid\b")
    report("hard", "clc.decode_sites", n_dec >= 4 and n_first == n_dec, f"is_canceled {n_dec}, get_first_ctaid {n_first} (one pair per inlined clc_consume)", "7.4")
    n_ld128 = ptx_count(instrs, r"^ld\.shared\.b128\b")
    report("hard", "ld.shared.b128 = decode sites", n_ld128 == n_dec, f"{n_ld128} vs {n_dec}", "7.4")
    n_fence = ptx_count(instrs, r"^fence\.proxy\.async\.shared::cta\b")
    report("hard", "fence.proxy.async = 8 + decode sites", n_fence == 8 + n_dec, f"{n_fence} expected {8 + n_dec}", "7.7 + 7.4")
    n_try = ptx_count(instrs, r"^mbarrier\.try_wait\.parity\.shared::cta\.b64\b")
    report("soft", "mbarrier.try_wait sites", n_try >= 20, f"{n_try} (peeks + blocking-wait sites; baseline 64 incl. the C path)", "7.2 / B.5")
    n_mapa = ptx_count(instrs, r"^@p mapa\.shared::cluster\.u32\b|^mapa\.shared::cluster\.u32\b")
    report("soft", "mapa.shared::cluster", n_mapa >= 5, f"{n_mapa} (consume sites + throttle release + 2 handshake arrivals + CLC arming; baseline 26 incl. the C path)", "E.7.4")
    peer = ptx_count(instrs, r"and\.b32 .*, -16777224\b|and\.b32 .*, -16777217\b")
    report("hard", "peer-bit mask and.b32 -16777224", peer >= 1, f"{peer} sites (0xFEFFFFF8 folded, or 0xFEFFFFFF)", "D.6 item 2")
    n_fold = ptx_count(instrs, r"^and\.b32 %r\d+, %r\d+, -16777232;")
    report("info", "peer-bit mask folded on the hoisted producer base (and.b32 ..., -16777232 = 0xFEFFFFF0)", True, f"{n_fold} sites (G.3: equivalent for the 1024-byte-aligned base and s <= 7; the literal -16777224 survives at the epilogue release)", "")
    h16 = ptx_count(instrs, r"^tcgen05\.commit.*, %rs\d+;|^cp\.async\.bulk\.tensor.*, %rs\d+, %rd\d+;")
    report("soft", "16-bit mask registers (.b16 %rs)", h16 >= 3, f"{h16} instructions carry a %rs mask operand", "7.2")
    for name, rx in PTX_ABSENT:
        got = ptx_count(instrs, rx)
        report("hard", f"absent: {name}", got == 0, f"{got}", "E.7.4 absence list")
    n_param = ptx_count(instrs, r"^ld\.param\b")
    report("info", "ld.param", True, f"{n_param} (k, alpha, beta; the map addresses are param-space constants)", "")
    # values carried in registers: the feeding mov.b32 of 49152, 16, 128/1 and 160/6 must exist
    for val, name in ((49152, "expect_tx 49152"), (16, "expect_tx 16"), (128, "bar.sync thread count 128"), (160, "bar count 160"), (10000000, "suspend hint 0x989680"), (270532624, "idesc 0x10200010"), (512, "tmem columns 512"), (800, "CLC empty count 800"), (256, "acc empty count 256")):
        got = ptx_count(instrs, rf"^mov\.b32 %r\d+, {val};")
        report("soft", f"mov.b32 {name}", got >= 1, f"{got} materialisations", "7.2 operand form")


# ---------------------------------------------------------------- SASS
SASS_LINE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(?:@!?U?P[T0-9]*\s+)?(.*?)\s*;?\s*(?:/\*.*)?$")


def sass_instructions(text):
    out = []
    for line in text.splitlines():
        m = SASS_LINE.match(line)
        if m and m.group(2):
            out.append(m.group(2).strip())
    return out


def sass_count(instrs, pattern):
    rx = re.compile(pattern)
    return sum(1 for s in instrs if rx.search(s))


SASS_ROWS = [
    ("UTCHMMA.2CTA", r"^UTCHMMA\.2CTA\b", 4, "hard"),
    ("UTCBAR.2CTA.MULTICAST", r"^UTCBAR\.2CTA\.MULTICAST\b", 2, "hard"),
    ("UTMASTG.3D", r"^UTMASTG\.3D\b", 32, "hard"),
    ("UTMACMDFLUSH", r"^UTMACMDFLUSH\b", 8, "hard"),
    ("DEPBAR.LE SB0, 0x1", r"^DEPBAR\.LE SB0, 0x1\b", 8, "hard"),
    ("DEPBAR.LE SB0, 0x0 (must be absent)", r"^DEPBAR\.LE SB0, 0x0\b", 0, "hard"),
    ("LDTM.x16", r"^LDTM(\.\w+)*\.x16\b", 8, "hard"),
    ("UGETNEXTWORKID.BROADCAST", r"^UGETNEXTWORKID\.BROADCAST\b", 1, "hard"),
    ("PREEXIT", r"^PREEXIT\b", 1, "hard"),
    ("FFMA", r"^FFMA\b", 128, "hard"),
    ("FADD (must be absent)", r"^FADD\b", 0, "hard"),
    ("LDL (must be absent)", r"^LDL\b", 0, "hard"),
    ("STL (must be absent)", r"^STL\b", 0, "hard"),
    ("UTMALDG.3D.MULTICAST.2CTA", r"^UTMALDG\.3D\.MULTICAST\.2CTA\b", 2, "soft"),
    ("UTMALDG.3D.2CTA", r"^UTMALDG\.3D\.2CTA\b", 2, "soft"),
    ("plain UTMALDG.3D (C load, must be absent)", r"^UTMALDG\.3D\b(?!\.)", 0, "soft"),
    ("UTMACCTL.PF", r"^UTMACCTL\.PF\b", 4, "soft"),
    ("ACQBULK", r"^ACQBULK\b", 2, "soft"),
    ("UTCATOMSWS.AND (dealloc)", r"^UTCATOMSWS\.AND\b", 1, "soft"),
    ("SYNCS.EXCH.64 (mbarrier.init)", r"^SYNCS\.EXCH\.64\b", 33, "soft"),
    ("BAR.SYNC.DEFER_BLOCKING 0x1, 0x80", r"^BAR\.SYNC\.DEFER_BLOCKING 0x1, 0x80\b", 16, "soft"),
    ("BAR.SYNC.DEFER_BLOCKING 0x6, 0xa0", r"^BAR\.SYNC\.DEFER_BLOCKING 0x6, 0xa0\b", 1, "soft"),
    ("BAR.ARV 0x6, 0xa0", r"^BAR\.ARV 0x6, 0xa0\b", 1, "soft"),
    ("BPT.TRAP (ptxas dealloc checks)", r"^BPT\.TRAP\b", 2, "soft"),
]


def check_sass(path, base_path):
    print("== SASS executed-path inventory (E.7.4 / D.7) ==")
    text = open(path, errors="replace").read()
    instrs = sass_instructions(text)
    base = sass_instructions(open(base_path, errors="replace").read()) if base_path else None
    if not instrs:
        report("hard", "sass_function_present", False, f"no instructions in {path}", "deabstraction_trace_run.sh inspect (sass_of_kernel)")
        return
    report("info", "sass_instructions", True, f"{len(instrs)} (baseline 3224 incl. the C path)" + (f" [baseline file {len(base)}]" if base else ""), "")
    for name, rx, exp, kind in SASS_ROWS:
        got = sass_count(instrs, rx)
        ref = f" [baseline {sass_count(base, rx)}]" if base else ""
        report(kind, name, got == exp, f"{got} expected {exp}{ref}", "E.7.4 / D.7")
    fmul_rx, sts_rx = r"^FMUL\b", r"^STS\b"
    fmul = sass_count(instrs, fmul_rx)
    fmul_ref = f" [baseline {sass_count(base, fmul_rx)}]" if base else ""
    report("soft", "FMUL 128 (+ up to 2 from the %cluster_ctaid lowering)", 128 <= fmul <= 130, f"{fmul}{fmul_ref}", "D.2")
    sts = sass_count(instrs, sts_rx)
    sts_ref = f" [baseline {sass_count(base, sts_rx)}]" if base else ""
    report("soft", "STS 128 (+ up to 3 bookkeeping)", 128 <= sts <= 131, f"{sts}{sts_ref}", "D.2")
    utcatom = sass_count(instrs, r"^UTCATOMSWS\.2CTA\.FIND_AND_SET\b")
    report("soft", "UTCATOMSWS.2CTA.FIND_AND_SET (alloc, >= 1)", utcatom >= 1, f"{utcatom} (baseline 2: ptxas retry copy)", "D.2")
    peer = sass_count(instrs, r"0xfefffff8")
    report("soft", "0xfefffff8 peer mask present", peer >= 1, f"{peer}", "D.6 item 2")
    uprmt = sass_count(instrs, r"^UPRMT\b.*0x5410")
    report("soft", "UPRMT ... 0x5410 (16-bit mask insert)", uprmt >= 1, f"{uprmt}", "D.2")
    lds128 = sass_count(instrs, r"^LDS\.128\b")
    report("info", "LDS.128 (CLC decodes)", True, f"{lds128}", "")
    spills = sass_count(instrs, r"^MOV\.SPILL\b") + sass_count(instrs, r"^R2UR\.FILL\b")
    report("info", "MOV.SPILL + R2UR.FILL (uniform-to-vector moves; baseline 44 + 44)", True, f"{spills}", "")
    exits = sass_count(instrs, r"^EXIT\b")
    report("info", "EXIT (excluding PREEXIT; baseline 8)", True, f"{exits}", "")
    nano = sass_count(instrs, r"^NANOSLEEP\b")
    report("info", "NANOSLEEP (wait retry loops + alloc retry; baseline 53)", True, f"{nano}", "")
    elect = sass_count(instrs, r"^ELECT\b")
    report("info", "ELECT (baseline 5; none expected inside the hot loops)", True, f"{elect}", "")
    # hot-loop shape (soft, D.2): the MMA k-tile loop should hold 4 UTCHMMA + 1 UTCBAR and no ELECT
    idx = [i for i, s in enumerate(instrs) if s.startswith("UTCHMMA.2CTA")]
    if len(idx) == 4:
        window = instrs[idx[0] - 30 if idx[0] >= 30 else 0: idx[3] + 6]
        report("soft", "no ELECT around the four UTCHMMA", not any(s.startswith("ELECT") for s in window), f"{sum(1 for s in window if s.startswith('ELECT'))} ELECT in the MMA window", "D.6 item 6")
        report("soft", "UTCBAR follows the fourth UTCHMMA", any(s.startswith("UTCBAR.2CTA.MULTICAST") for s in instrs[idx[3]:idx[3] + 4]), "within 3 instructions", "7.6")


# ---------------------------------------------------------------- resources
def check_resources(path, kname):
    print("== resource usage (E.8 D1/D2) ==")
    text = open(path, errors="replace").read()
    blocks = re.split(r"(?m)^\s*Function ", text)
    found = None
    for b in blocks:
        if kname in b.split("\n", 1)[0]:
            found = b
    if found is None:
        report("hard", "resources_function_found", False, f"no Function containing {kname} in {path}", "E.7.4")
        return
    # key:value pairs in any order (cuobjdump inserts e.g. CONSTANT[2]:12 between LOCAL and CONSTANT[0] when ptxas
    # emits a jump table; the first build of 2026-09-11 did, G.3)
    kv = dict((k, int(v)) for k, v in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", found))
    if not all(k in kv for k in ("REG", "STACK", "SHARED", "LOCAL", "CONSTANT[0]")):
        report("hard", "resources_line_parsed", False, found.strip().split("\n")[0][:120], "E.7.4")
        return
    reg, stack, shared, local, const0 = kv["REG"], kv["STACK"], kv["SHARED"], kv["LOCAL"], kv["CONSTANT[0]"]
    extra = {k: v for k, v in kv.items() if k not in ("REG", "STACK", "SHARED", "LOCAL", "CONSTANT[0]", "TEXTURE", "SURFACE", "SAMPLER")}
    if extra:
        report("info", "other resource banks", True, " ".join(f"{k}:{v}" for k, v in extra.items()) + " (CONSTANT[2] = ptxas jump table, G.3)", "")
    report("hard", "LOCAL 0 (no spills)", local == 0, str(local), "E.8 D1")
    report("hard", "STACK 0", stack == 0, str(stack), "E.8 D1")
    report("hard", "SHARED 1024 (system-reserved only; no static __shared__)", shared == 1024, str(shared), "E.8 D1 / D.6 item 4")
    report("soft", "REG <= 68", reg <= 68, f"{reg} (baseline 68; a higher count is a disclosed deviation, E.8 D2)", "D.7")
    report("info", "CONSTANT[0] = 0x380 + sizeof(ExplicitGemmParams)", True, f"{const0} (896 + 640 = 1536 for the 640-byte block of CUDA 13.3, alignof(CUtensorMap) 128; 1472 for 576; baseline 2944)", "")


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    ptx, sass, res = argv[1], argv[2], argv[3]
    kname = argv[4] if len(argv) > 4 else "explicit_blackwell_fp16_gemm_kernel"
    base_ptx = argv[5] if len(argv) > 5 else None
    base_sass = argv[6] if len(argv) > 6 else None
    check_ptx(ptx, base_ptx)
    check_sass(sass, base_sass)
    check_resources(res, kname)
    n_fail = sum(1 for _, ok, kind in results if not ok and kind == "hard")
    n_warn = sum(1 for _, ok, kind in results if not ok and kind == "soft")
    print(f"== SUMMARY: {sum(1 for _, ok, _k in results if ok)} PASS, {n_fail} FAIL (hard), {n_warn} WARN (soft) ==")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
