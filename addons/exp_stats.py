"""
Paired per-seed analysis of checkpointed results.

    python -m selfmade.exp_stats --out results/primary --ref vanilla \
        --family existing_nb@gini existing_nb@perm v2_imm_mut@gini v2_imm_mut@perm --metric final_best_so_far_val

Per comparison (arm - ref, paired by seed): mean & median difference, win/tie/loss, sd, 95% bootstrap CI of the
mean difference, exact-ish Wilcoxon signed-rank p (zeros dropped), two-sided sign-test p, standardized effect
d_z, the minimum detectable effect at 80% power (2.8*sd/sqrt(n)), and Holm-adjusted p within the family.
A non-significant result is NOT evidence of no effect: read the CI and the MDE.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["final_best_so_far_val", "auc_best_so_far_val", "selected_test"]


def _gm(rows, key):
    v = [x[key] for x in rows if key in x]
    return float(np.mean(v)) if v else np.nan


def load_cells(out) -> pd.DataFrame:
    rows = []
    d = os.path.join(out, "cells")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn.endswith(".error.json"):
            continue
        c = json.load(open(os.path.join(d, fn)))
        h = c["history"]
        gen_rows = [x for x in h if "diversity" in x]
        rows.append({
            "arm": c["arm"], "seed": c["seed"], "source": c["importance_source"], "common_hash": c["common_hash"],
            "final_best_so_far_val": c["final_best_so_far_val"], "auc_best_so_far_val": c["auc_best_so_far_val"],
            "selected_test": c["selected_test"], "n_evals": c["n_evals"], "wall_time_s": c["wall_time_s"],
            "guidance_time_s": c["guidance_time_s"], "valid_rate": c["valid_rate"],
            "div_final": gen_rows[-1]["diversity"] if gen_rows else np.nan,
            "div_mean": float(np.mean([x["diversity"] for x in gen_rows])) if gen_rows else np.nan,
            "unique_final": gen_rows[-1]["n_unique"] if gen_rows else np.nan,
            "mean_c": _gm(gen_rows, "c"),
            "mean_abs_ell_bits": _gm(gen_rows, "mean_abs_ell_bits"), "mean_abs_ell_ops": _gm(gen_rows, "mean_abs_ell_ops"),
            "max_abs_ell": _gm(gen_rows, "max_abs_ell"), "tau2_bits": _gm(gen_rows, "tau2_bits"), "tau2_ops": _gm(gen_rows, "tau2_ops"),
            "cv_rho": _gm(gen_rows, "cv_rho"), "frac_gens_tau0": (float(np.mean([x.get("tau2_bits", 1) == 0 and x.get("tau2_ops", 1) == 0 for x in gen_rows if "tau2_bits" in x])) if any("tau2_bits" in x for x in gen_rows) else np.nan),
            "hit_plateau": float(c["final_best_so_far_val"] >= 0.9470),
            "frac_gens_guided": float(np.mean([x.get("max_abs_ell", 0) > 0 for x in gen_rows])) if gen_rows else np.nan,
        })
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired_stats(a, b, n_boot=20000, seed=0):
    """a, b: aligned arrays (arm, ref). Returns dict for diff = a - b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    nz = d[np.abs(d) > 1e-12]
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    w = stats.wilcoxon(nz).pvalue if len(nz) > 0 else 1.0
    wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses > 0 else 1.0
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    return {"n": n, "mean_diff": float(d.mean()), "median_diff": float(np.median(d)), "sd_diff": sd,
            "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "wins": wins, "ties": int(n - wins - losses), "losses": losses, "wilcoxon_p": float(w),
            "sign_p": float(sign_p), "d_z": float(d.mean() / sd) if sd and sd > 0 else 0.0,
            "mde80": float(2.8 * sd / np.sqrt(n)) if n > 1 else np.nan}


def compare(df, ref, family, metric):
    wide = df.pivot(index="seed", columns="arm", values=metric)
    rows = []
    for arm in family:
        if arm not in wide.columns:
            continue
        both = wide[[arm, ref]].dropna()
        if len(both) < 3:
            continue
        r = paired_stats(both[arm].to_numpy(), both[ref].to_numpy())
        r["arm"] = arm; r["metric"] = metric
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].to_numpy())
        out["sign_p_holm"] = holm(out["sign_p"].to_numpy())
    return out


def check_pairing(df):
    """Every seed must share one common population across arms, and every arm must use the same eval budget."""
    bad_hash = df.groupby("seed")["common_hash"].nunique()
    bad_eval = df.groupby("arm")["n_evals"].nunique()
    return {"seeds_with_inconsistent_common_pop": bad_hash[bad_hash > 1].index.tolist(),
            "evals_per_arm": df.groupby("arm")["n_evals"].agg(["min", "max"]).to_dict("index"),
            "arms_with_varying_evals": bad_eval[bad_eval > 1].index.tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--ref", default="vanilla")
    ap.add_argument("--family", nargs="*"); ap.add_argument("--metric", nargs="*", default=METRICS)
    a = ap.parse_args(argv)
    df = load_cells(a.out)
    fam = a.family or sorted(set(df["arm"]) - {a.ref})
    print("pairing/budget check:", json.dumps(check_pairing(df), indent=1))
    res = pd.concat([compare(df, a.ref, fam, m) for m in a.metric], ignore_index=True)
    cols = ["metric", "arm", "n", "mean_diff", "median_diff", "ci_lo", "ci_hi", "wins", "ties", "losses",
            "wilcoxon_p", "wilcoxon_p_holm", "d_z", "mde80"]
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(res[cols].round(5).to_string(index=False))
    print("\ncost / behaviour per arm (means over seeds)")
    print(df.groupby("arm")[["wall_time_s", "guidance_time_s", "valid_rate", "div_mean", "div_final", "unique_final", "mean_c", "cv_rho", "mean_abs_ell_bits", "mean_abs_ell_ops", "max_abs_ell", "tau2_bits", "tau2_ops", "frac_gens_tau0", "hit_plateau", "frac_gens_guided"]]
          .mean().round(4).to_string())
    res.to_csv(os.path.join(a.out, "paired_stats.csv"), index=False)
    df.to_csv(os.path.join(a.out, "cell_summary.csv"), index=False)


if __name__ == "__main__":
    main()