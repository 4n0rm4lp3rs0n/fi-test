"""Pre-registered decision rule for variants of D1 (see PREREG_neighbor_variants_vi.md). Observed endpoints only.

    python -m addons.v3_decide_variants --out results/dev_nb --d1 "v3_dedup:k=1,m=6" \
        --variants "v3_nb:k=1,m=6,mode=n1" "v3_nb:k=1,m=6,mode=n2" [--stage dev|primary|confirm]

Claim A (same budget): variant - D1 on final best-so-far val: mean >= +0.001 AND 95% bootstrap CI lower bound > 0 AND Holm p < 0.05
                       (Holm over the variants).
Claim B (fewer evaluations): variant best-so-far at 60 branch evaluations (generation 16) - D1 best-so-far at 76 (generation 20):
                       95% two-sided CI lower bound > -0.0005 (one-sided 97.5% => Bonferroni for 2 variants).
Other budgets (36, 44, 52, 68) are descriptive only. Integrity checks as in v3_decide. Python >= 3.7."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from addons import exp_stats as S
from addons import v3_decide as V

GEN_OF_BUDGET = {36: 10, 44: 12, 52: 14, 60: 16, 68: 18, 76: 20}      # branch evaluations = 4 * (generation - 1)


def curves(out):
    rows = []
    for fn in os.listdir(os.path.join(out, "cells")):
        if not fn.endswith(".json") or fn.endswith(".error.json"):
            continue
        c = json.load(open(os.path.join(out, "cells", fn)))
        h = {x["generation"]: x["best_so_far"] for x in c["history"]}
        rows.append(dict([("arm", c["arm"]), ("seed", c["seed"])] + [("g%d" % g, h[g]) for g in range(1, 21)]))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--d1", default="v3_dedup:k=1,m=6")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--stage", choices=["dev", "primary", "confirm"], default="dev")
    ap.add_argument("--min-effect", type=float, default=0.001); ap.add_argument("--margin", type=float, default=0.0005)
    ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--budget", type=int, default=60)
    a = ap.parse_args(argv)
    if any(f.endswith(".error.json") for f in os.listdir(os.path.join(a.out, "cells"))):
        print("ABORT: error cells present"); return 2
    df = S.load_cells(a.out)
    probs = V.integrity(df, [a.d1] + a.variants)
    if probs:
        print("ABORT - integrity problems:"); [print("  -", p) for p in probs]; return 2
    cv = curves(a.out)
    n = df[df.arm == a.d1]["seed"].nunique()
    print("integrity OK | seeds per arm: %d | stage: %s\n" % (n, a.stage))
    d1 = cv[cv.arm == a.d1].set_index("seed")
    pd.set_option("display.width", 220)
    # ---- Claim A
    rows = []
    for v in a.variants:
        x = df[df.arm == v].set_index("seed")["final_best_so_far_val"]; y = df[df.arm == a.d1].set_index("seed")["final_best_so_far_val"]
        b = pd.concat([x, y], axis=1, keys=["v", "d"]).dropna()
        st = S.paired_stats(b.v.values, b.d.values); st["variant"] = v; rows.append(st)
    A = pd.DataFrame(rows); A["p_holm"] = S.holm(A["wilcoxon_p"].to_numpy())
    A["pass_A"] = (A.mean_diff >= a.min_effect) & (A.ci_lo > 0) & (A.p_holm < a.alpha)
    print("### CLAIM A - variant - D1, final best-so-far val (positive = variant better)")
    print(A[["variant", "n", "mean_diff", "median_diff", "ci_lo", "ci_hi", "wins", "ties", "losses", "wilcoxon_p", "p_holm", "d_z", "pass_A"]].round(5).to_string(index=False))
    # ---- Claim B and descriptive budgets
    gB = GEN_OF_BUDGET[a.budget]; rowsB = []
    for v in a.variants:
        vv = cv[cv.arm == v].set_index("seed")
        for bud, g in sorted(GEN_OF_BUDGET.items()):
            if bud == 76:
                continue
            s_ = vv.index.intersection(d1.index)
            st = S.paired_stats(vv.loc[s_, "g%d" % g].values, d1.loc[s_, "g20"].values)
            rowsB.append(dict(variant=v, budget=bud, mean_diff=st["mean_diff"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"], wins=st["wins"], losses=st["losses"]))
    B = pd.DataFrame(rowsB)
    B["pass_B"] = (B.budget == a.budget) & (B.ci_lo > -a.margin)
    print("\n### CLAIM B - variant best-so-far at B evaluations - D1 best-so-far at 76 (pre-registered B=%d, non-inferior if CI lower > -%.4f)" % (a.budget, a.margin))
    print(B.round(5).to_string(index=False))
    # ---- secondary (report only)
    print("\n### secondary (report only): variant - D1")
    for m in ["auc_best_so_far_val", "selected_test"]:
        for v in a.variants:
            x = df[df.arm == v].set_index("seed")[m]; y = df[df.arm == a.d1].set_index("seed")[m]
            b = pd.concat([x, y], axis=1, keys=["v", "d"]).dropna(); st = S.paired_stats(b.v.values, b.d.values)
            print("  %-26s %-22s %+.5f  CI [%+.5f, %+.5f]  W/T/L %d/%d/%d" % (v, m, st["mean_diff"], st["ci_lo"], st["ci_hi"], st["wins"], st["ties"], st["losses"]))
    # ---- decision
    print("\n### DECISION")
    verdict = {}
    for v in a.variants:
        pa = bool(A[A.variant == v]["pass_A"].iloc[0]); pb = bool(B[(B.variant == v) & (B.budget == a.budget)]["pass_B"].iloc[0])
        verdict[v] = {"A": pa, "B": pb}
        print("  %-28s claim A (same budget): %s | claim B (%d evals non-inferior to D1@76): %s" % (v, "PASS" if pa else "FAIL", a.budget, "PASS" if pb else "FAIL"))
    any_pass = any(x["A"] or x["B"] for x in verdict.values())
    nxt = {"dev": "go to primary (fresh seeds, user runs)", "primary": "go to confirm (once)", "confirm": "CONFIRMED"}[a.stage]
    print("\nRESULT: %s" % (("at least one variant passed -> " + nxt + ". Report which claim passed.") if any_pass else
          "NO variant passed A or B. Per the pre-registration: STOP; V3 = D1. (Dev has low power: not evidence that the variants are useless.)"))
    json.dump({"stage": a.stage, "n_seeds": int(n), "verdict": verdict, "any_pass": any_pass}, open(os.path.join(a.out, "decision_variants.json"), "w"), indent=1)
    A.to_csv(os.path.join(a.out, "variants_claimA.csv"), index=False); B.to_csv(os.path.join(a.out, "variants_claimB.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
