#!/usr/bin/env python3
"""Post-processing of the de-abstraction trace run (Part C, C.6 of Semantics-preserving-de-abstraction.md).

Usage:
    python3 check_trace.py launch0.csv host.txt

Reads the record CSV written by trace_dump() and the captured stdout of the trace binary, evaluates every
prediction of Parts A/B for the fixed run (--m=8192 --n=8192 --k=8192, beta = 0), prints one PASS/FAIL line
per check, and exits non-zero if any check failed.  Each FAIL names the document section that made the
prediction.
"""
import csv
import statistics
import sys
from collections import defaultdict

# ---- record kinds (deabstraction_trace.hpp) ----
K_SMEM, K_TMEM, K_MMA, K_TMA_LOAD, K_TMA_STORE, K_TMA_STORE_LANES = 1, 2, 3, 4, 5, 6
K_CLC_ISSUE, K_CLC_SCHED, K_CLC_MMA, K_PROBE0, K_SMEM2 = 7, 8, 9, 10, 11

# ---- expected constants (Sections 6.2, 7.6, B.4) ----
SMEM_TOTAL = 230400
OFF = {
    "mainloop_full0": 0, "mainloop_empty0": 64, "epi_load_full0": 128, "epi_load_empty0": 160,
    "load_order_b00": 192, "clc_full0": 208, "clc_empty0": 224, "accumulator_full0": 240,
    "accumulator_empty0": 272, "clc_throttle_full0": 304, "clc_throttle_empty0": 320, "tmem_dealloc": 336,
    "clc_response0": 352, "tmem_base_ptr": 384, "tensors": 512, "smem_C": 512, "smem_D": 512,
    "tensors_mainloop": 33792, "smem_A": 33792, "smem_B": 164864,
    "pipelines": 0, "mainloop": 0, "epi_load": 128, "load_order": 192, "clc": 208, "accumulator": 240,
    "clc_throttle": 304, "clc_response1": 368, "tensors_epilogue": 512, "load_order_b01": 200,
}
DESC_HI = 0x40004040
DESC_CONST_LO = 0x00010000
IDESC = 0x10200010
SMEM_A_OFF, SMEM_B_OFF, SMEM_D_OFF = 33792, 164864, 512
PEER_MASK = 0xFEFFFFFF

results = []


def check(name, ok, detail, where):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}" + ("" if ok else f"   [prediction: {where}]"))


def hx(v):
    return f"0x{int(v):08x}"


# ---------------------------------------------------------------- parsing
def load_records(path):
    recs = []
    with open(path) as f:
        for row in csv.reader(l for l in f if not l.startswith("#")):
            if not row or row[0] == "kind":
                continue
            r = {
                "kind": int(row[0]), "bx": int(row[1]), "by": int(row[2]), "rank": int(row[3]),
                "warp": int(row[4]), "lane": int(row[5]), "smid": int(row[6]), "seq": int(row[7]),
                "t": int(row[8]), "v": [int(x) for x in row[9:19]],
            }
            recs.append(r)
    return recs


def load_host(path):
    host, encodes, tmaps, k0, dev = {}, [], {}, [], {}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("TRACE_HOST "):
                parts = line.split()
                host[parts[1]] = parts[2:]
            elif line.startswith("TRACE_ENCODE "):
                d = {}
                for tok in line.split()[2:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        d[k] = v
                encodes.append(d)
            elif line.startswith("TRACE_TMAP "):
                parts = line.split()
                tmaps[parts[1]] = parts[2:]
            elif line.startswith("TRACE_K0 "):
                k0.append(line.split())
            elif line.startswith("TRACE_DEV "):
                parts = line.split()
                dev = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    return host, encodes, tmaps, k0, dev


def hval(host, key, idx=0, default=None):
    v = host.get(key)
    if v is None or len(v) <= idx:
        return default
    try:
        return int(v[idx])
    except ValueError:
        return v[idx]


# ---------------------------------------------------------------- checks
def main(csv_path, host_path):
    recs = load_records(csv_path)
    host, encodes, tmaps, k0_lines, dev = load_host(host_path)
    by_kind = defaultdict(list)
    for r in recs:
        by_kind[r["kind"]].append(r)
    for k in by_kind:
        by_kind[k].sort(key=lambda r: r["seq"])

    # ---- bases per rank (K0 and K1) ----
    base_k0 = {}   # (bx,by) -> record
    for r in by_kind[K_PROBE0]:
        if r["v"][9] == 0:
            base_k0[(r["bx"], r["by"])] = r
    base_k1 = {}   # rank -> base (first cluster)
    k1_by_rank, k1b_by_rank = {}, {}
    for r in by_kind[K_SMEM]:
        k1_by_rank[r["rank"]] = r
        base_k1[r["rank"]] = r["v"][0]
    for r in by_kind[K_SMEM2]:
        k1b_by_rank[r["rank"]] = r

    # 1. addresses (D5)
    print("== 1. shared-memory addresses (D5, Sections 6.2, B.6) ==")
    if base_k1:
        be = base_k1.get(0)
        bo = base_k1.get(1)
        check("smem_base_rank0_value", be in (0x0, 0x400), f"rank0 base {hx(be) if be is not None else None}", "B.6 / D5")
        for rank in (1, 2, 3):
            b = base_k1.get(rank)
            if b is not None and be is not None:
                print(f"INFO rank{rank} base {hx(b)} = rank0 base ^ {hx(b ^ be)} (rank field expected in bits [24,28): {hx(rank << 24)})")
                check(f"smem_base_rank{rank}_low_bits", (b & 0x00FFFFFF) == (be & 0x00FFFFFF), f"{hx(b)} vs {hx(be)}", "B.6 (same CTA-local offset in every rank)")
        for rank in (1, 3):
            b, bpair = base_k1.get(rank), base_k1.get(rank - 1)
            if b is not None and bpair is not None:
                check(f"smem_base_rank{rank}_peer_bit", (b & PEER_MASK) == bpair and (b & (1 << 24)) != 0,
                      f"{hx(b)} & 0xFEFFFFFF = {hx(b & PEER_MASK)} vs even partner {hx(bpair)}", "Section 7.2 peer-bit trick / B.6")
        if be is not None:
            check("smem_base_descriptor_bound", (be & 0x00FFFFFF) + SMEM_TOTAL < 262144, f"(base & 0xFFFFFF) + 230400 = {(be & 0x00FFFFFF) + SMEM_TOTAL}", "B.3.2 step 4 (14-bit start address)")
        names = ["mainloop_full0", "mainloop_empty0", "clc_full0", "clc_empty0", "accumulator_empty0", "tmem_dealloc",
                 "clc_response0", "smem_A", "smem_B"]
        for rank, r in sorted(k1_by_rank.items()):
            b = r["v"][0]
            for i, n in enumerate(names):
                check(f"rank{rank}_off_{n}", r["v"][i + 1] - b == OFF[n], f"{r['v'][i + 1] - b} expected {OFF[n]}", "Section 6.2")
            check(f"rank{rank}_smem_A_align1024", (r["v"][8] % 1024) == 0, hx(r["v"][8]), "Section 6.2")
        names2 = ["smem_D", "tmem_base_ptr", None, None, None, None, "accumulator_full0", "clc_throttle_full0", "load_order_b00", "epi_load_full0"]
        for rank, r in sorted(k1b_by_rank.items()):
            b = base_k1.get(rank)
            if b is None:
                continue
            for i, n in enumerate(names2):
                if n is None:
                    continue
                check(f"rank{rank}_off_{n}", r["v"][i] - b == OFF[n], f"{r['v'][i] - b} expected {OFF[n]}", "Section 6.2")
            check(f"rank{rank}_smem_D_align512", (r["v"][0] % 512) == 0, hx(r["v"][0]), "Section 6.2")
            check(f"rank{rank}_is_epi_load_needed", r["v"][2] == 0, str(r["v"][2]), "Section 8 (beta = 0)")
            check(f"rank{rank}_sched_participant", r["v"][3] == (1 if rank == 0 else 0), str(r["v"][3]), "Section 7.1")
            check(f"rank{rank}_rank_field", r["v"][4] == rank, str(r["v"][4]), "Section 7.1")
            check(f"rank{rank}_peer_rank", r["v"][5] == (rank ^ 1), str(r["v"][5]), "Section 7.1")
        if 0 in k1_by_rank and 1 in k1_by_rank:
            acc_e = k1_by_rank[0]["v"][5]
            acc_o = k1_by_rank[1]["v"][5]
            check("peer_mask_maps_odd_to_even", (acc_o & PEER_MASK) == acc_e, f"{hx(acc_o)} & 0xFEFFFFFF = {hx(acc_o & PEER_MASK)} vs {hx(acc_e)}", "Section 7.2 peer-bit trick")
    else:
        check("k1_records_present", False, "no K_SMEM records", "C.4.3 K1")
    if base_k0:
        cl = defaultdict(dict)
        for (bx, by), r in base_k0.items():
            cl[(bx // 2, by // 2)][(bx % 2) + 2 * (by % 2)] = r
        for cid, ranks in sorted(cl.items()):
            if len(ranks) != 4:
                continue
            for rk, r in ranks.items():
                base, mapa_peer, mapa_0, masked = r["v"][0], r["v"][1], r["v"][2], r["v"][3]
                check(f"k0_cluster{cid}_rank{rk}_ctarank", r["rank"] == rk, f"rank field {r['rank']} vs (x+2y) {rk}", "Section 7.1")
                check(f"k0_cluster{cid}_rank{rk}_mapa_peer", mapa_peer == ranks[rk ^ 1]["v"][0], f"mapa(base,{rk ^ 1})={hx(mapa_peer)} peer base {hx(ranks[rk ^ 1]['v'][0])}", "B.6")
                check(f"k0_cluster{cid}_rank{rk}_mapa_rank0", mapa_0 == ranks[0]["v"][0], f"mapa(base,0)={hx(mapa_0)} rank0 base {hx(ranks[0]['v'][0])}", "B.6")
                check(f"k0_cluster{cid}_rank{rk}_masked", masked == ranks[rk & ~1]["v"][0], f"base&mask={hx(masked)} even-partner base {hx(ranks[rk & ~1]['v'][0])}", "Section 7.2 peer-bit trick")
                check(f"k0_cluster{cid}_rank{rk}_lowbits", (base & 0x00FFFFFF) == (ranks[0]["v"][0] & 0x00FFFFFF), f"{hx(base)} vs rank0 {hx(ranks[0]['v'][0])}", "B.6 (same CTA-local offset in every rank)")
        if base_k1:
            r00 = base_k0.get((0, 0))
            if r00:
                check("k0_k1_base_agree", r00["v"][0] == base_k1.get(0), f"K0 {hx(r00['v'][0])} K1 {hx(base_k1.get(0, -1))}", "C.4.2")
    else:
        check("k0_records_present", False, "no K_PROBE0 records", "C.4.2")

    # 2. TMEM (D6)
    print("== 2. TMEM base (D6) ==")
    tmem_T = None
    pairs = defaultdict(set)
    for r in by_kind[K_TMEM]:
        pairs[(r["bx"] // 2, r["by"] // 2, r["rank"] // 2)].add(r["v"][0])
    for r in by_kind[K_PROBE0]:
        if r["v"][9] == 1:
            pairs[("k0", r["bx"] // 2, r["by"] // 2, r["rank"] // 2)].add(r["v"][0])
    if pairs:
        allv = set()
        for key, vals in sorted(pairs.items(), key=str):
            allv |= vals
            check(f"tmem_base_pair_{key}", len(vals) == 1, f"values {[hx(v) for v in vals]}", "Section 7.6 (same base in both CTAs of a pair)")
        check("tmem_base_value", allv == {0}, f"observed {[hx(v) for v in sorted(allv)]} (expected 0x0)", "Section 6.3 / D6 (expectation, not a source fact)")
        r0 = [r for r in by_kind[K_TMEM] if r["bx"] < 2 and r["by"] < 2]
        check("tmem_T_derivable", bool(r0), f"{len(r0)} K_TMEM records from cluster (0,0)", "C.4.3 K2 (cap must cover the first cluster)")
        if r0:
            tmem_T = r0[0]["v"][0]
    else:
        check("tmem_records_present", False, "no K_TMEM records", "C.4.3 K2")

    # 3. MMA operands (D7)
    print("== 3. tcgen05.mma operands (D7, Section 7.6, B.4) ==")
    mma_by_cta = defaultdict(list)
    for r in by_kind[K_MMA]:
        mma_by_cta[(r["bx"], r["by"])].append(r)
    if mma_by_cta:
        for cta, rs in sorted(mma_by_cta.items()):
            base = base_k1.get(rs[0]["rank"])
            check(f"mma_cta{cta}_leader", rs[0]["rank"] % 2 == 0, f"issuing rank {rs[0]['rank']}", "Section 7.6 (leader issues)")
            for i, r in enumerate(rs):
                s, kb = i // 4, i % 4
                da, db, tc, idesc, sc = r["v"][:5]
                check(f"mma_cta{cta}_{i}_desc_hi", (da >> 32) == DESC_HI and (db >> 32) == DESC_HI, f"a {da >> 32:#010x} b {db >> 32:#010x}", "Section 7.6 (0x4000404000010000)")
                if base is not None:
                    ea = DESC_CONST_LO | ((((base + SMEM_A_OFF) >> 4) & 0x3FFF) + 2 * kb + 1024 * s)
                    eb = DESC_CONST_LO | ((((base + SMEM_B_OFF) >> 4) & 0x3FFF) + 2 * kb + 512 * s)
                    check(f"mma_cta{cta}_{i}_desc_lo", (da & 0xFFFFFFFF) == ea and (db & 0xFFFFFFFF) == eb,
                          f"a {da & 0xFFFFFFFF:#010x} exp {ea:#010x}; b {db & 0xFFFFFFFF:#010x} exp {eb:#010x} (s={s}, kb={kb})", "B.4 / B.3.2 step 8")
                check(f"mma_cta{cta}_{i}_idesc", idesc == IDESC, f"{idesc:#010x}", "Section 7.6 (0x10200010)")
                check(f"mma_cta{cta}_{i}_scale_c", sc == (0 if i == 0 else 1), f"{sc}", "Section 7.6 (clear only on the first MMA of a tile)")
                if tmem_T is not None:
                    check(f"mma_cta{cta}_{i}_tmem_c", tc == tmem_T, f"{hx(tc)} expected {hx(tmem_T)} (first tile, stage 0)", "B.4")
    else:
        check("mma_records_present", False, "no K_MMA records", "C.4.3 K3")

    # tile sequences per CTA from K5c (for the TMA checks)
    tiles = {}
    clc_mma_by_cta = defaultdict(list)
    for r in by_kind[K_CLC_MMA]:
        clc_mma_by_cta[(r["bx"], r["by"])].append(r)
    for cta, rs in clc_mma_by_cta.items():
        bx, by = cta
        seq = [(2 * (by // 2) + bx % 2, 2 * (bx // 2) + by % 2)]
        for r in sorted(rs, key=lambda r: r["v"][6]):
            if r["v"][3] == 1:
                seq.append((r["v"][7], r["v"][8]))
        tiles[cta] = seq

    def tile_of(cta, idx):
        bx, by = cta
        seq = tiles.get(cta, [(2 * (by // 2) + bx % 2, 2 * (bx // 2) + by % 2)])
        return seq[idx] if idx < len(seq) else None

    # 4. TMA operands (D8)
    print("== 4. TMA load/store operands (D8, B.4) ==")
    ld_by_cta = defaultdict(list)
    for r in by_kind[K_TMA_LOAD]:
        ld_by_cta[(r["bx"], r["by"])].append(r)
    if ld_by_cta:
        for cta, rs in sorted(ld_by_cta.items()):
            bx, by = cta
            x, y = bx % 2, by % 2
            base = base_k1.get(rs[0]["rank"])
            qa = qb = 0
            for r in rs:
                c0, c1, c2, dst, nelem, mbar = r["v"][:6]
                if nelem == 8192:
                    q, qa = qa, qa + 1
                    which, dst_exp, c1_exp = "A", (base + SMEM_A_OFF + 16384 * (q % 8) + 8192 * y) if base is not None else None, None
                elif nelem == 4096:
                    q, qb = qb, qb + 1
                    which, dst_exp, c1_exp = "B", (base + SMEM_B_OFF + 8192 * (q % 8)) if base is not None else None, None
                else:
                    check(f"tma_load_cta{cta}_unexpected_size", False, f"size(src) = {nelem} (C load? beta must be 0)", "Section 7.5")
                    continue
                tl = tile_of(cta, q // 128)
                if tl is None:
                    continue
                tm, tn = tl
                c1_exp = 128 * tm + 64 * y if which == "A" else 128 * tn + 64 * x
                ok = c0 == 64 * (q % 128) and c1 == c1_exp and c2 == 0
                check(f"tma_load_cta{cta}_{which}{q}_coord", ok, f"({c0},{c1},{c2}) expected ({64 * (q % 128)},{c1_exp},0) tile {tl}", "B.4 (A/B TMA coordinates)")
                if dst_exp is not None:
                    check(f"tma_load_cta{cta}_{which}{q}_dst", dst == dst_exp, f"{hx(dst)} expected {hx(dst_exp)}", "B.4 (smem destinations)")
                    check(f"tma_load_cta{cta}_{which}{q}_mbar", mbar == base + 8 * (q % 8), f"{hx(mbar)} expected {hx(base + 8 * (q % 8))} (unmasked)", "B.4 (mainloop barrier operands)")
    else:
        check("tma_load_records_present", False, "no K_TMA_LOAD records", "C.4.3 K4a")
    st_by_cta = defaultdict(list)
    for r in by_kind[K_TMA_STORE]:
        st_by_cta[(r["bx"], r["by"])].append(r)
    if st_by_cta:
        for cta, rs in sorted(st_by_cta.items()):
            base = base_k1.get(rs[0]["rank"])
            for i, r in enumerate(rs):
                e, k = (i % 32) // 4, i % 4
                tl = tile_of(cta, i // 32)
                if tl is None:
                    continue
                tm, tn = tl
                c0, c1, c2, src, lanes = r["v"][:5]
                check(f"tma_store_cta{cta}_{i}_all_lanes", lanes == 32, f"popc(activemask) at issue = {lanes}", "Section 7.7 step 6 (all 32 lanes of warp 4 issue)")
                check(f"tma_store_cta{cta}_{i}_coord", c0 == 128 * tm + 32 * k and c1 == 128 * tn + 16 * e and c2 == 0,
                      f"({c0},{c1},{c2}) expected ({128 * tm + 32 * k},{128 * tn + 16 * e},0) e={e} k={k} tile {tl}", "B.4 (D TMA store coordinates)")
                if base is not None:
                    exp = base + SMEM_D_OFF + 8192 * (e % 4) + 2048 * k
                    check(f"tma_store_cta{cta}_{i}_src", src == exp, f"{hx(src)} expected {hx(exp)}", "B.4 (D smem source)")
    else:
        check("tma_store_records_present", False, "no K_TMA_STORE records", "C.4.3 K4b")

    # 5. CLC (D9)
    print("== 5. cluster launch control (D9, Section 7.4, B.3.4) ==")
    issue_by_cl = defaultdict(list)
    for r in by_kind[K_CLC_ISSUE]:
        issue_by_cl[(r["bx"] // 2, r["by"] // 2)].append(r)
    sched_by_cl = defaultdict(list)
    for r in by_kind[K_CLC_SCHED]:
        sched_by_cl[(r["bx"] // 2, r["by"] // 2)].append(r)
    be = base_k1.get(0)
    if issue_by_cl:
        total_issue = 0
        for cl, rs in sorted(issue_by_cl.items()):
            rs.sort(key=lambda r: r["v"][2])
            counts = [r["v"][2] for r in rs]
            total_issue += len(rs)
            check(f"clc_issue_{cl}_count_sequence", counts == list(range(len(rs))), f"counts {counts[:6]}... n={len(rs)}", "B.3.4 step 5a")
            check(f"clc_issue_{cl}_slot_phase", all(r["v"][0] == r["v"][2] % 2 and r["v"][1] == (1 ^ ((r["v"][2] >> 1) & 1)) for r in rs),
                  "slot = n % 2, producer parity = 1 ^ ((n >> 1) & 1)", "Section 7.2 / B.3.4 step 5a")
            check(f"clc_issue_{cl}_rank0", all(r["rank"] == 0 and r["warp"] == 1 for r in rs), "issued by rank 0 warp 1", "Section 7.4")
            if be is not None:
                check(f"clc_issue_{cl}_mbarrier", all(r["v"][3] == be + 208 + 8 * (r["v"][2] % 2) for r in rs), "full[slot] = base + 208 + 8*slot", "B.4")
        check("clc_issue_total", total_issue == 1024, f"{total_issue} queries (1024 cluster tiles)", "Section 7.4 (T queries per cluster, 1024 in total)")
    else:
        check("clc_issue_records_present", False, "no K_CLC_ISSUE records", "C.4.3 K5a")
    if sched_by_cl:
        covered = defaultdict(int)
        n_native = len(sched_by_cl)
        total_T = 0
        for cl, rs in sorted(sched_by_cl.items()):
            rs.sort(key=lambda r: r["v"][6])
            T = len(rs)
            total_T += T
            valids = [r["v"][3] for r in rs]
            check(f"clc_sched_{cl}_last_invalid", valids[-1] == 0 and all(v == 1 for v in valids[:-1]), f"valid flags {valids}", "Section 7.4 (T-1 cancelled, last not cancelled)")
            check(f"clc_sched_{cl}_slot_phase", all(r["v"][4] == r["v"][6] % 2 and r["v"][5] == ((r["v"][6] >> 1) & 1) for r in rs),
                  "slot = n % 2, consumer parity = (n >> 1) & 1", "Section 7.2 / B.3.4 step 5b")
            cx, cy = cl
            covered[(cy, cx)] += 1   # the natively launched cluster's own tile
            for r in rs:
                if r["v"][3] != 1:
                    continue
                x0, y0 = r["v"][0], r["v"][1]
                check(f"clc_sched_{cl}_n{r['v'][6]}_even_first_cta", x0 % 2 == 0 and y0 % 2 == 0, f"first CTA id ({x0},{y0})", "Section 7.4 (cluster-aligned first CTA)")
                check(f"clc_sched_{cl}_n{r['v'][6]}_tile", (r["v"][7], r["v"][8]) == (y0, x0), f"swizzled ({r['v'][7]},{r['v'][8]}) expected ({y0},{x0}) for rank 0", "B.4 (CTA tile from CLC first-CTA id)")
                covered[(y0 // 2, x0 // 2)] += 1
            if cl in issue_by_cl:
                check(f"clc_{cl}_issue_consume_count", len(issue_by_cl[cl]) == T, f"issued {len(issue_by_cl[cl])} consumed {T}", "Section 7.4")
        expected = {(m, n) for m in range(32) for n in range(32)}
        missing = expected - set(covered)
        dup = {k: v for k, v in covered.items() if v > 1}
        check("clc_coverage_exactly_once", not missing and not dup and set(covered) <= expected, f"missing {len(missing)} duplicated {len(dup)} extra {len(set(covered) - expected)}", "Section 7.4 (exactly-once coverage)")
        check("clc_sum_T", total_T == 1024, f"sum of T_c = {total_T}", "Section 7.4")
        mac = hval(host, "max_active_clusters")
        check("clc_native_clusters", mac is None or n_native <= mac, f"N_native = {n_native}, cudaOccupancyMaxActiveClusters = {mac}", "Section 7.4 (N_native <= co-resident clusters)")
        print(f"INFO tiles per native cluster: min {min(len(v) for v in sched_by_cl.values())} max {max(len(v) for v in sched_by_cl.values())} "
              f"mean {statistics.mean(len(v) for v in sched_by_cl.values()):.2f}")
        # latency issue -> consume on rank 0
        lat = []
        for cl, rs in sched_by_cl.items():
            iss = {r["v"][2]: r["t"] for r in issue_by_cl.get(cl, [])}
            for r in rs:
                if r["v"][6] in iss:
                    lat.append(r["t"] - iss[r["v"][6]])
        if lat:
            lat.sort()
            print(f"INFO CLC query->consume latency ns: min {lat[0]} median {lat[len(lat) // 2]} p90 {lat[int(len(lat) * 0.9)]} max {lat[-1]}")
        # MMA-warp copies agree with rank 0
        for cta, rs in sorted(clc_mma_by_cta.items()):
            bx, by = cta
            cl = (bx // 2, by // 2)
            ref = sched_by_cl.get(cl)
            if ref is None:
                continue
            rs.sort(key=lambda r: r["v"][6])
            def resp_key(r):   # raw M/N/L are undefined for a not-cancelled response: compare them only when valid
                return tuple(r["v"][:3]) + (1,) if r["v"][3] == 1 else (None, None, None, 0)
            same = [resp_key(r) for r in rs] == [resp_key(r) for r in ref]
            check(f"clc_mma_cta{cta}_same_responses", same, f"{len(rs)} responses vs {len(ref)} on rank 0", "Section 7.4 (response multicast to all CTAs)")
            ok = all((r["v"][7], r["v"][8]) == (r["v"][1] + bx % 2, r["v"][0] + by % 2) for r in rs if r["v"][3] == 1)
            check(f"clc_mma_cta{cta}_tile_offsets", ok, "(tm, tn) = (y0 + ctaid.x, x0 + ctaid.y)", "B.4")
    else:
        check("clc_sched_records_present", False, "no K_CLC_SCHED records", "C.4.3 K5b")

    # 6. host facts (D1-D4, D13, D14)
    print("== 6. host-side facts (D1-D4, D13, D14) ==")
    check("sizeof_SharedStorage", hval(host, "sizeof_SharedStorage") == SMEM_TOTAL, str(hval(host, "sizeof_SharedStorage")), "Section 6.2")
    check("sizeof_EpilogueSharedStorage", hval(host, "sizeof_EpilogueSharedStorage") == 33792, str(hval(host, "sizeof_EpilogueSharedStorage")), "Section 6.1")
    for n, exp in OFF.items():
        got = hval(host, f"off_{n}")
        if got is not None:
            check(f"host_off_{n}", got == exp, f"{got} expected {exp}", "Section 6.2")
    check("host_grid", host.get("grid") == ["64", "64", "1"], str(host.get("grid")), "Section 7.4")
    check("host_block", host.get("block") == ["256", "1", "1"], str(host.get("block")), "Section 5.4")
    check("host_raster_order_AlongN", (host.get("scheduler_raster_order") or [""])[0] == "AlongN", str(host.get("scheduler_raster_order")), "Section 7.4")
    check("host_swizzle_disabled", hval(host, "scheduler_swizzle_divisor") == 0, str(hval(host, "scheduler_swizzle_divisor")), "Section 7.4")
    check("host_problem_tiles", host.get("scheduler_problem_tiles") == ["32", "32", "1"], str(host.get("scheduler_problem_tiles")), "B.1.1 step 11")
    drv = hval(host, "driver_version")
    check("host_driver_version", drv is not None and drv > 13010, f"{drv}", "Section 0 / B.1.1 step 5 (fixup branch not taken)")
    check("host_encodes_six", len(encodes) == 6, f"{len(encodes)} TRACE_ENCODE lines", "Section 5.2")
    # values are printed as NAME(number); compare the names produced by the C++ side against the CUDA enumerators
    def name_of(v):
        return v.split("(")[0] if v is not None else None
    exp_ab = dict(format="FLOAT16", dim="3", shape="8192,8192,1", stride_bytes="16384,0", box="64,64,1", elem_stride="1,1,1",
                  interleave="INTERLEAVE_NONE", swizzle="SWIZZLE_128B", l2promo="L2_128B", oobfill="OOB_FILL_NONE", result="0")
    exp_cd = dict(exp_ab, format="FLOAT32", stride_bytes="32768,0", box="32,16,1", swizzle="SWIZZLE_128B_ATOM_32B")
    for i, e in enumerate(encodes[:6]):
        exp = exp_ab if i < 4 else exp_cd
        bad = {k: (e.get(k), v) for k, v in exp.items() if name_of(e.get(k)) != v}
        check(f"encode_{i}_tuple", not bad, "ok" if not bad else str(bad), "Section 5.3 / B.1.1 steps 5-9")
    for a, b in (("A", "A_fallback"), ("B", "B_fallback")):
        if a in tmaps and b in tmaps:
            check(f"tmap_{b}_identical", tmaps[a] == tmaps[b], "16 words compared", "Section 5.2")
    check("kernel_maxDynamicSharedSizeBytes", hval(host, "kernel_maxDynamicSharedSizeBytes") == SMEM_TOTAL, str(hval(host, "kernel_maxDynamicSharedSizeBytes")), "Section 5.2")
    check("kernel_static_smem_zero", hval(host, "kernel_sharedSizeBytes") == 0, str(hval(host, "kernel_sharedSizeBytes")), "Section 6.2")
    print(f"INFO kernel_localSizeBytes (trace build, informational; D11 uses the Release binary): {hval(host, 'kernel_localSizeBytes')}")
    check("kernel_nonportable_cluster_allowed", hval(host, "kernel_nonPortableClusterSizeAllowed") == 1, str(hval(host, "kernel_nonPortableClusterSizeAllowed")), "Section 5.4")
    check("cudaGetLastError_success", hval(host, "cudaGetLastError") == 0, str(host.get("cudaGetLastError")), "B.1.2 step 7")
    check("device_sm_count", hval(host, "device_sm_count") == 148, str(hval(host, "device_sm_count")), "Section 0 (B200, informational)")
    check("device_reserved_smem", hval(host, "device_reserved_smem_per_block") == 1024, str(hval(host, "device_reserved_smem_per_block")), "B.6 (informational)")
    macros = {"macro_NDEBUG": 1, "macro_CUDA_API_PER_THREAD_DEFAULT_STREAM": 0, "macro_CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL": 0,
              "macro_CUTLASS_ENABLE_GDC_FOR_SM100": 1, "macro_CUTLASS_ENABLE_SYNCLOG": 0, "macro_CUTLASS_ENABLE_CUDA_HOST_ADAPTER": 0}
    for k, v in macros.items():
        check(k, hval(host, k) == v, str(hval(host, k)), "Section 0 / D14")
    check("device_IsGdcGloballyEnabled", dev.get("IsGdcGloballyEnabled") == "1", str(dev), "Section 11 (GDC on)")
    check("device_FEAT_SM100_ALL", dev.get("FEAT_SM100_ALL") == "1" and dev.get("CUDA_ARCH") == "1000", str(dev), "Section 0 (sm_100a)")
    nv = host.get("nvcc_version")
    check("nvcc_13_3", nv is not None and nv[:2] == ["13", "3"], str(nv), "Section 0 (CUDA 13.3)")

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"== SUMMARY: {len(results) - n_fail} PASS, {n_fail} FAIL ==")
    return 1 if n_fail else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
