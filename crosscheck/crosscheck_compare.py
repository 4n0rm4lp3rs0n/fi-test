"""Compare YOUR results/<dir>/cells/*.json with the sandbox's expected values (kstudy_expected_cells.csv).
    python crosscheck_compare.py results/kstudy kstudy_expected_cells.csv
Standalone (numpy/pandas only). Exact equality is expected: the sandbox reproduced the user's own diag1 vanilla and
v2_imm_mut cells bit-for-bit (Python 3.7.12, numpy 1.21.6, pandas 1.3.5, sklearn 1.0.2, scipy 1.7.3)."""
import csv, json, os, sys
res_dir, exp_csv = sys.argv[1], sys.argv[2]
mine = {}
for fn in os.listdir(os.path.join(res_dir, "cells")):
    if fn.endswith(".json") and not fn.endswith(".error.json"):
        c = json.load(open(os.path.join(res_dir, "cells", fn)))
        mine[(c["arm"], str(c["seed"]))] = c
exp = list(csv.DictReader(open(exp_csv)))
keys = ["final_best_so_far_val", "auc_best_so_far_val", "selected_val", "selected_test"]
missing = exact = 0; maxd = {k: 0.0 for k in keys}; bad = {}
for r in exp:
    c = mine.get((r["arm"], r["seed"]))
    if c is None:
        missing += 1; continue
    d = {k: abs(float(r[k]) - float(c[k])) for k in keys}
    for k in keys: maxd[k] = max(maxd[k], d[k])
    if all(v == 0 for v in d.values()) and int(r["n_evals"]) == c["n_evals"]:
        exact += 1
    else:
        bad.setdefault(r["arm"], []).append(r["seed"])
print("expected cells: %d | found: %d | missing: %d | bit-identical: %d" % (len(exp), len(exp) - missing, missing, exact))
print("max abs difference per metric:", {k: "%.3g" % v for k, v in maxd.items()})
for a, s in bad.items(): print("DIFFERENT in arm %s: %d cells, e.g. seeds %s" % (a, len(s), s[:3]))
print("RESULT:", "IDENTICAL" if exact == len(exp) else "NOT identical - send me this output + env_report.txt")
