"""OFFLINE reporting of V3 runs. THE ONLY module allowed to read benchmark ground truth (per-architecture means over the
3 repeats). It reads the logs written by addons.v3_runner / addons.diag_runner AFTER a run; it is never imported by any
code that runs the GA, builds guidance, trains a surrogate or selects candidates (checked by tests/test_v3.py).

    python -m addons.v3_report --out results/kstudy --records ./datas/nasbench_only108.tfrecord --ref vanilla \
        [--family "v3_dedup:k=1,m=6" ...] [--k-table]

Per cell (arm, seed), merging the shared common file and the branch file:
  observed_*  : what the GA could see (single noisy val/test draws)            -> same endpoints as exp_stats
  oracle_selected_val_mean / _test_mean : true 3-repeat mean of the architecture that the GA SELECTED (argmax observed val)
  oracle_discovery_val_mean / _test_mean: best true mean val among ALL architectures visited (search quality without selection noise)
  oracle_discovery_test_of_best_val_mean : TRUE mean test of that best-true-val architecture. NOTE high val does not imply high test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from nasbench import api
from selfmade import nb101
from addons import exp_stats as S


def _oracle(nb, space, code, ops, cache):
    key = (code, tuple(ops))
    if key not in cache:
        d = space.decode(nb101.GenomeNB101(code, list(ops)))
        spec = api.ModelSpec(matrix=np.array(d["matrix"]), ops=list(d["operations"]))
        _, comp = nb.get_metrics_from_spec(spec)
        runs = comp[108]
        cache[key] = (float(np.mean([r["final_validation_accuracy"] for r in runs])),
                      float(np.mean([r["final_test_accuracy"] for r in runs])))
    return cache[key]


def cell_table(out, records):
    nb = api.NASBench(records)
    space = nb101.NASBench101Space()
    cache, rows = {}, []
    cdir = os.path.join(out, "cands")
    for fn in sorted(os.listdir(os.path.join(out, "cells"))):
        if not fn.endswith(".json") or fn.endswith(".error.json"):
            continue
        c = json.load(open(os.path.join(out, "cells", fn)))
        stem = fn[:-5]
        com = [json.loads(l) for l in open(os.path.join(cdir, "common__%d.jsonl" % c["seed"]))]
        br = [json.loads(l) for l in open(os.path.join(cdir, stem + ".jsonl"))]
        allr = com + br
        vals = np.array([r["val"] for r in allr]); sel = int(np.argmax(vals))
        sv, st = _oracle(nb, space, allr[sel]["code"], allr[sel]["ops"], cache)
        orc = {}
        for r in allr:
            if r["hash"] not in orc:
                orc[r["hash"]] = _oracle(nb, space, r["code"], r["ops"], cache)
        best_h = max(orc, key=lambda h: orc[h][0])
        cnt = {}
        for r in br:
            cnt[r["hash"]] = cnt.get(r["hash"], 0) + 1
        seen = {r["hash"] for r in com}; dup = 0
        for r in br:
            dup += r["hash"] in seen; seen.add(r["hash"])
        rows.append({"arm": c["arm"], "seed": c["seed"],
                     "final_best_so_far_val": c["final_best_so_far_val"], "auc_best_so_far_val": c["auc_best_so_far_val"],
                     "selected_test": c["selected_test"],
                     "oracle_selected_val_mean": sv, "oracle_selected_test_mean": st,
                     "oracle_discovery_val_mean": orc[best_h][0], "oracle_discovery_test_of_best_val_mean": orc[best_h][1],
                     "distinct_archs": len(orc), "distinct_archs_branch": len({r["hash"] for r in br}),
                     "dup_rate": dup / max(1, len(br)), "max_evals_one_arch": max(cnt.values()) if cnt else 0,
                     "n_evals": c["n_evals"], "wall_time_s": c["wall_time_s"],
                     "fallback_total": sum(h.get("fallback", 0) for h in c["history"]),
                     "slots_total": sum(h.get("slots", 0) for h in c["history"])})
    return pd.DataFrame(rows)


METRICS = ["final_best_so_far_val", "auc_best_so_far_val", "selected_test", "oracle_selected_val_mean",
           "oracle_selected_test_mean", "oracle_discovery_val_mean", "oracle_discovery_test_of_best_val_mean",
           "distinct_archs", "dup_rate"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--records", required=True)
    ap.add_argument("--ref", default="vanilla"); ap.add_argument("--family", nargs="*")
    a = ap.parse_args(argv)
    df = cell_table(a.out, a.records)
    df.to_csv(os.path.join(a.out, "oracle_cells.csv"), index=False)
    fam = a.family or sorted(set(df["arm"]) - {a.ref})
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("per-arm means (n seeds = %d)" % df["seed"].nunique())
    print(df.groupby("arm")[METRICS + ["max_evals_one_arch", "wall_time_s"]].mean().round(5).to_string())
    res = []
    for m in METRICS:
        r = S.compare(df, a.ref, fam, m)
        if len(r):
            res.append(r)
    res = pd.concat(res, ignore_index=True)
    cols = ["metric", "arm", "n", "mean_diff", "median_diff", "ci_lo", "ci_hi", "wins", "ties", "losses", "wilcoxon_p", "wilcoxon_p_holm", "d_z"]
    print("\npaired differences vs", a.ref, "(Holm within each metric)")
    print(res[cols].round(5).to_string(index=False))
    res.to_csv(os.path.join(a.out, "oracle_paired_stats.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
