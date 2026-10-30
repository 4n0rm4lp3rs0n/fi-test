"""Applies the PRE-REGISTERED decision rule to a head-to-head run. Reads observed endpoints only (no benchmark ground truth,
no oracle). Write the rule down BEFORE looking at the data; this script just applies it.

    python -m addons.v3_decide --out results/primary_v3 --d1 "v3_dedup:k=1,m=6" \
        --rivals vanilla existing_single@gini v2_imm_mut@gini v2_screen_adapt@gini [--min-effect 0.001] [--alpha 0.05]

For every rival: paired difference (D1 - rival) per seed on the PRIMARY endpoint final_best_so_far_val.
  PASS vs rival  <=>  mean diff >= min_effect  AND  95% bootstrap CI lower bound > 0  AND  Holm-adjusted Wilcoxon p < alpha
                       (Holm over all rivals).
  ADVANCE        <=>  PASS against EVERY rival.   Otherwise: report exactly which rivals failed; do not re-tune on this seed set.
Secondary (report only, no verdict): auc_best_so_far_val, selected_test.
Integrity checks (abort if violated): equal evaluation budget in every arm, one shared common population per seed, all
seeds present in all arms, no error cells. Python >= 3.7.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from addons import exp_stats as S


def integrity(df, arms):
    problems = []
    d = df[df.arm.isin(arms)]
    for a in arms:
        if a not in set(d.arm):
            problems.append("arm missing: " + a)
    if problems:
        return problems
    if d.groupby("arm")["n_evals"].nunique().max() != 1 or d["n_evals"].nunique() != 1:
        problems.append("evaluation budget differs between cells: %s" % d.groupby("arm")["n_evals"].agg(["min", "max"]).to_dict("index"))
    bad = d.groupby("seed")["common_hash"].nunique()
    if (bad > 1).any():
        problems.append("seeds with inconsistent common population: %s" % bad[bad > 1].index.tolist()[:5])
    counts = d.groupby("arm")["seed"].nunique()
    if counts.nunique() != 1:
        problems.append("different number of seeds per arm: %s" % counts.to_dict())
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--d1", default="v3_dedup:k=1,m=6")
    ap.add_argument("--rivals", nargs="+", required=True)
    ap.add_argument("--min-effect", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)

    errs = [f for f in os.listdir(os.path.join(a.out, "cells")) if f.endswith(".error.json")]
    if errs:
        print("ABORT: error cells present:", errs[:5]); return 2
    df = S.load_cells(a.out)
    probs = integrity(df, [a.d1] + a.rivals)
    if probs:
        print("ABORT - integrity problems:"); [print("  -", p) for p in probs]; return 2
    n_seeds = df[df.arm == a.d1]["seed"].nunique()
    print("integrity OK | seeds per arm: %d | budget: %d evaluations in every cell | one common population per seed\n"
          % (n_seeds, int(df[df.arm == a.d1]["n_evals"].iloc[0])))

    wide = {m: df.pivot(index="seed", columns="arm", values=m) for m in S.METRICS}
    rows = []
    for m in S.METRICS:
        for r in a.rivals:
            both = wide[m][[a.d1, r]].dropna()
            st = S.paired_stats(both[a.d1].to_numpy(), both[r].to_numpy())
            st.update(metric=m, rival=r)
            rows.append(st)
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for m in S.METRICS:
        idx = res.index[res.metric == m]
        res.loc[idx, "p_holm"] = S.holm(res.loc[idx, "wilcoxon_p"].to_numpy())

    pd.set_option("display.width", 240)
    cols = ["rival", "n", "mean_diff", "median_diff", "ci_lo", "ci_hi", "wins", "ties", "losses", "wilcoxon_p", "p_holm", "d_z"]
    labels = {"final_best_so_far_val": "PRIMARY endpoint: final best-so-far val",
              "auc_best_so_far_val": "secondary (report only): AUC of best-so-far val",
              "selected_test": "secondary (report only): observed test of the selected architecture"}
    for m in S.METRICS:
        print("### D1 - rival | %s | positive = D1 better" % labels[m])
        print(res[res.metric == m][cols].round(5).to_string(index=False)); print()

    prim = res[res.metric == "final_best_so_far_val"].copy()
    prim["passes"] = (prim.mean_diff >= a.min_effect) & (prim.ci_lo > 0) & (prim.p_holm < a.alpha)
    print("### DECISION (rule: mean diff >= %.4f AND CI lower bound > 0 AND Holm p < %.2f, against EVERY rival)" % (a.min_effect, a.alpha))
    for r in prim.itertuples():
        print("  %-28s mean %+.5f  CI [%+.5f, %+.5f]  Holm p %.4f  -> %s" % (r.rival, r.mean_diff, r.ci_lo, r.ci_hi, r.p_holm, "PASS" if r.passes else "FAIL"))
    ok = bool(prim["passes"].all())
    print("\nRESULT: %s" % ("ADVANCE to the confirmation run (all rivals passed)." if ok else
          "DO NOT ADVANCE. Failed rivals: %s. Report as is; no re-tuning on this seed set." % ", ".join(prim[~prim["passes"]].rival)))
    res.to_csv(os.path.join(a.out, "decision_stats.csv"), index=False)
    json.dump({"d1": a.d1, "rivals": a.rivals, "min_effect": a.min_effect, "alpha": a.alpha, "advance": ok,
               "n_seeds": int(n_seeds), "pass": {r.rival: bool(r.passes) for r in prim.itertuples()}},
              open(os.path.join(a.out, "decision.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
