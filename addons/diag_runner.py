"""
Diagnostic wrapper around addons.exp_runner: records EVERY evaluated candidate (and every surrogate-screening pool)
without modifying nb101.py / nbcore.py / guided_v2.py / exp_runner.py and WITHOUT touching any RNG stream.

    python -m addons.diag_runner run --out results/diag1 --evaluator nb101 --records ./datas/nasbench_full.tfrecord \
        --seeds diag --arms vanilla v2_screen@gini v2_screen_adapt@gini v2_imm_mut@gini

Accepts exactly the same arguments as `python -m addons.exp_runner` plus the extra seed-set name `diag`
(30 seeds, disjoint from pilot / dev / dev_b / primary / confirm). Everything exp_runner writes is written as before
(cells/, common/, manifest.json, progress.jsonl -> exp_stats works unchanged); additionally:

  <out>/cands/<cell>.jsonl     one JSON line per evaluated candidate of that cell (branch generations only)
  <out>/cands/common__<seed>.jsonl   the shared init + warm-up generation (evaluated once per seed)
  <out>/pools/<cell>.jsonl     one line per surrogate-screening call (pool, predictions, chosen) - screen arms only

Per evaluated candidate: eval index within the cell, generation, code, operations, validation / test / train accuracy,
isomorphism-invariant pruned-architecture hash (nasbench's own hash_spec if importable), edge count of the pruned
architecture, `seen_before` (same hash already evaluated earlier IN THIS FILE; the analysis merges the common__<seed>
file with the cell file to get the full history).
test_accuracy is stored for REPORTING ONLY and must never be used for fitting/selection.

Reproducibility guarantee: logging reads state only; tests/test_diag_runner.py checks that the result of every cell
is bit-identical with and without this wrapper.
Python >= 3.7 compatible.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys

import numpy as np

from addons import exp_runner as R
from addons import guided_v2 as gv
from selfmade import nbcore

DIAG_SEED_SPEC = ("diag", 90210, 30)
CANON_OPS = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]


# ---------------------------------------------------------------- seeds
def diag_seeds():
    from numpy.random import SeedSequence
    name, entropy, n = DIAG_SEED_SPEC
    return [int(x) for x in SeedSequence(entropy).generate_state(n)]


_orig_seed_list = R.seed_list


def seed_list_with_diag(name):
    if name == "diag":
        return diag_seeds()
    return _orig_seed_list(name)


def _assert_disjoint():
    d = set(diag_seeds())
    for other in ("primary", "pilot", "dev", "dev_b", "confirm"):
        assert not d & set(_orig_seed_list(other)), "diag seeds overlap with " + other


# ---------------------------------------------------------------- canonical architecture hash
_MODEL_SPEC = None


def _load_model_spec():
    """nasbench's own ModelSpec (prunes + hash_spec) if importable; otherwise the NASBENCH_SRC folder; else None."""
    global _MODEL_SPEC
    if _MODEL_SPEC is not None:
        return _MODEL_SPEC or None
    for attempt in range(2):
        try:
            if attempt == 1:
                src = os.environ.get("NASBENCH_SRC")
                if not src:
                    break
                sys.path.insert(0, src)
                # a stub 'nasbench' package (tests) must not shadow the real lib
                for m in [k for k in sys.modules if k == "nasbench" or k.startswith("nasbench.")]:
                    if getattr(sys.modules[m], "__file__", None) is None:
                        del sys.modules[m]
            ms = importlib.import_module("nasbench.lib.model_spec")
            if hasattr(ms, "ModelSpec"):
                _MODEL_SPEC = ms
                return ms
        except Exception:
            continue
    _MODEL_SPEC = False
    return None


def arch_hash(decoded):
    """Isomorphism-invariant hash of the PRUNED architecture; also returns the pruned edge count."""
    ms = _load_model_spec()
    matrix, ops = decoded["matrix"], decoded["operations"]
    if ms is not None:
        spec = ms.ModelSpec(matrix=np.array(matrix), ops=list(ops))
        if getattr(spec, "valid_spec", True) and spec.matrix is not None:
            return spec.hash_spec(CANON_OPS), int(np.sum(spec.matrix)), "nasbench"
        return "invalid", -1, "nasbench"
    # fallback (sandbox only): own pruning + sorted tuple signature; NOT isomorphism invariant -> flagged by source
    from addons.synthetic_eval import prune
    m, o = prune(matrix, ops)
    return hashlib.md5((m.tobytes().hex() + "|" + "|".join(o)).encode()).hexdigest(), int(m.sum()), "fallback"


# ---------------------------------------------------------------- the recorder
class Recorder:
    def __init__(self, out, space):
        self.out, self.space = out, space
        self.bucket = None          # file stem currently being written
        self.gen = 0
        self.i = 0
        self.seen = set()
        self.rows = []
        self.pool_rows = []

    def start(self, bucket, gen):
        self.bucket, self.gen, self.i, self.seen, self.rows, self.pool_rows = bucket, gen, 0, set(), [], []

    def log_eval(self, genome, res):
        d = self.space.decode(genome)
        h, ne, src = arch_hash(d)
        self.rows.append({"i": self.i, "gen": self.gen, "code": genome.code, "ops": list(genome.operations),
                          "val": float(res["validation_accuracy"]), "test": _f(res.get("test_accuracy")),
                          "train": _f(res.get("train_accuracy")), "hash": h, "hash_src": src, "n_edges_pruned": ne,
                          "seen_before": h in self.seen})
        self.seen.add(h)
        self.i += 1

    def flush(self, sub):
        if self.bucket is None:
            return
        for name, rows in ((sub, self.rows), ("pools", self.pool_rows)):
            if not rows:
                continue
            d = os.path.join(self.out, name)
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, self.bucket + ".jsonl.tmp")
            with open(tmp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, os.path.join(d, self.bucket + ".jsonl"))
        self.bucket = None


def _f(x):
    return None if x is None else float(x)


class LoggingEvaluator:
    """Same interface as the wrapped evaluator; records, never alters, the result."""

    def __init__(self, inner, rec):
        self.inner, self.rec = inner, rec

    def evaluate(self, genome):
        res = self.inner.evaluate(genome)
        if self.rec.bucket is not None:
            self.rec.log_eval(genome, res)
        return res

    def __getattr__(self, k):                      # pass-through (n_queries etc.)
        return getattr(self.inner, k)


# ---------------------------------------------------------------- installation (monkeypatch, reversible)
def install(out, space_holder):
    rec = Recorder(out, None)
    orig = {"make_evaluator": R.make_evaluator, "build_common": R.build_common, "run_cell": R.run_cell,
            "record": nbcore.Population.record, "operators_for": gv.operators_for, "seed_list": R.seed_list}

    def make_evaluator(kind, records, space):
        rec.space = space
        return LoggingEvaluator(orig["make_evaluator"](kind, records, space), rec)

    def build_common(seed, space, evaluator):
        rec.start("common__%d" % seed, 0)
        try:
            return orig["build_common"](seed, space, evaluator)
        finally:
            rec.flush("cands")

    def run_cell(arm, seed, common, space, cev):
        rec.start(R.cell_name(arm.spec, seed), common.current_generation + 1)
        res = orig["run_cell"](arm, seed, common, space, cev)
        rec.flush("cands")
        return res

    def record(self, generation):                  # evaluations of generation g happen BEFORE record(g)
        out_ = orig["record"](self, generation)
        rec.gen = generation + 1
        return out_

    def operators_for(pop, cfg, sig):
        ops = orig["operators_for"](pop, cfg, sig)
        if ops.screen is not None:
            inner = ops.screen

            def screen(pool):
                chosen = inner(pool)
                try:
                    pred = sig.surrogate.predict(np.array([gv.featurize(pop.space, g) for g in pool]))
                    idx = [next(k for k, g in enumerate(pool) if g is c) for c in chosen]
                    rec.pool_rows.append({"gen": rec.gen, "c": float(sig.c), "a": float(sig.a),
                                          "pool": [{"code": g.code, "ops": list(g.operations),
                                                    "hash": arch_hash(pop.space.decode(g))[0], "pred": float(p)}
                                                   for g, p in zip(pool, pred)], "chosen": idx})
                except Exception as e:             # logging must never break a run
                    rec.pool_rows.append({"gen": rec.gen, "log_error": repr(e)})
                return chosen
            ops.screen = screen
        return ops

    R.make_evaluator, R.build_common, R.run_cell = make_evaluator, build_common, run_cell
    nbcore.Population.record = record
    gv.operators_for = operators_for
    R.seed_list = seed_list_with_diag

    def uninstall():
        R.make_evaluator, R.build_common, R.run_cell = orig["make_evaluator"], orig["build_common"], orig["run_cell"]
        nbcore.Population.record = orig["record"]
        gv.operators_for = orig["operators_for"]
        R.seed_list = orig["seed_list"]
    return rec, uninstall


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_disjoint()
    out = None
    if argv and argv[0] == "run" and "--out" in argv:
        out = argv[argv.index("--out") + 1]
    if out is None:                                # status / arms: nothing to record
        return R.main(argv)
    rec, uninstall = install(out, None)
    try:
        return R.main(argv)
    finally:
        uninstall()


if __name__ == "__main__":
    sys.exit(main())