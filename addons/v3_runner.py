"""V3 experiment runner = addons.diag_runner (per-candidate logging) + V3 arms. No existing file is modified.

    python -m addons.v3_runner run --out results/kstudy --evaluator nb101 --records ./datas/nasbench_only108.tfrecord \
        --seeds kstudy --arms vanilla v3_dedup:k=inf,m=6 v3_dedup:k=1,m=6 v3_dedup:k=2,m=6 v3_dedup:k=3,m=6

All other arguments / outputs are those of addons.exp_runner (checkpointed per cell; re-run to resume).
Arms (kind "v3"):  v3_dedup:k=<int|inf>,m=<pairs per slot>
Extra seed sets:   diag (30), kstudy (100), kstudy_b (100), and parametric  --seeds ss:<entropy>:<n>  (SeedSequence(entropy),
                   n seeds; refuses any overlap with every registered set incl. confirm).
Python >= 3.7 compatible.
"""
from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np
from numpy.random import SeedSequence

from addons import exp_runner as R
from addons import guided_v2 as gv
from addons import diag_runner as D
from addons import v3_ops
from selfmade import nbcore, nb101

EXTRA_SEEDS = {"kstudy": (2026102801, 100), "kstudy_b": (2026102802, 100)}
SEED_SETS_ALL = ("primary", "pilot", "dev", "dev_b", "confirm")


def extra_seeds(name):
    entropy, n = EXTRA_SEEDS[name]
    return [int(x) for x in SeedSequence(entropy).generate_state(n)]


def assert_disjoint():
    D._assert_disjoint()
    ref = set()
    for s in SEED_SETS_ALL:
        ref |= set(D._orig_seed_list(s))
    ref |= set(D.diag_seeds())
    for name in EXTRA_SEEDS:
        assert not set(extra_seeds(name)) & ref, "seed set %s overlaps an existing set" % name


@dataclasses.dataclass(frozen=True)
class V3Arm(R.Arm):
    params: tuple = ()          # sorted (key, value) pairs

    def p(self, key, default=None):
        return dict(self.params).get(key, default)


_orig_parse_arm = R.parse_arm


def _val(v):
    if v in ("inf", "none"):
        return None
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def parse_arm(spec):
    head, _, over = spec.partition(":")
    if not head.startswith("v3_"):
        return _orig_parse_arm(spec)
    params = {}
    for kv in filter(None, over.split(",")):
        k, v = kv.split("=")
        params[k] = _val(v)
    if head in ("v3_dedup", "v3_rimm"):
        params.setdefault("k", None)
        if head == "v3_dedup" and "m" not in params:
            raise ValueError(spec + ": v3_dedup needs m=<pairs>")
        if head == "v3_rimm":                       # random immigrant (vanilla sampler), optional dedup screen when m given
            params.setdefault("n_imm", 1); params.setdefault("m", 1)
    elif head == "v3_gd":                           # V2 guided operators + dedup (D2 on top of D1)
        params.setdefault("k", 1); params.setdefault("m", 6); params.setdefault("src", "gini")
        for kk, dv in (("imm", 1), ("mut", 1), ("ops_only", 0), ("vfrac", 0.0), ("alpha", 8.0), ("beta", 8.0)):
            params.setdefault(kk, dv)
    elif head == "v3_nov":                          # dedup + surrogate veto + novelty slot (D3 on top of D1)
        params.setdefault("k", 1); params.setdefault("m", 6); params.setdefault("src", "gini")
        params.setdefault("veto", 0.3); params.setdefault("cmin", 0.05)
    else:
        raise ValueError("unknown v3 arm " + spec)
    return V3Arm(spec, "v3", "-", None, 8.0, 8.0, tuple(sorted(params.items(), key=lambda kv: kv[0])))


def run_cell_v3(arm, seed, common, space, cev):
    pop = common.copy()
    n_branch = R.TOTAL_GENS - R.WARMUP
    d_ref = gv.pop_diversity(pop.members)
    counter = v3_ops.ArchCounter(pop.evaluator, space, v3_ops.common_counts(common, space), v3_ops.common_descs(common, space))
    guide_time = 0.0
    pop.evaluator = counter
    nb101.set_seed(R.branch_seed(seed))
    evals0, t0 = cev.n, time.perf_counter()
    gen_extra = []
    with gv.count_validity() as vc, R.quiet():
        for g in range(n_branch):
            tot0, bad0 = vc.snapshot()
            extra = {"diversity_before": gv.pop_diversity(pop.members)}
            st = {}
            if arm.spec.startswith("v3_dedup"):
                ops = v3_ops.dedup_operators(pop, space, counter, arm.p("k"), arm.p("m"), st)
                gv.evolve_with(pop, ops, st)
            elif arm.spec.startswith("v3_rimm"):
                ops = gv.vanilla_operators(pop)
                if int(arm.p("m")) > 1:
                    ops.screen = v3_ops.make_dedup_screen(space, counter, arm.p("k"), st); ops.screen_pairs = int(arm.p("m"))
                ops.immigrant, ops.n_imm = (lambda: (space.random_genome(), False)), int(arm.p("n_imm"))
                gv.evolve_with(pop, ops, st)
            elif arm.spec.startswith("v3_gd"):
                a_, b_ = float(arm.p("alpha")), float(arm.p("beta"))
                cfg = gv.GuideConfig(name="v3_gd", source=arm.p("src"), alpha=a_, beta=b_, use_imm=bool(arm.p("imm")),
                                     use_mut=bool(arm.p("mut")), n_imm=1)
                if cfg.enabled():
                    sig = gv.fit_signals(gv.build_archive(pop.data, space), space, cfg, 1.0)
                    if arm.p("ops_only"):
                        sig.ell_bits[:] = 0.0                    # operation-level prior only; bits stay neutral
                    guide_time += sig.fit_time_s; extra.update(sig.diag())
                    ops = v3_ops.guided_dedup_operators(pop, cfg, sig, space, counter, arm.p("k"), arm.p("m"), st, float(arm.p("vfrac")))
                else:
                    ops = v3_ops.dedup_operators(pop, space, counter, arm.p("k"), arm.p("m"), st)
                gv.evolve_with(pop, ops, st)
            elif arm.spec.startswith("v3_nov"):
                cfg = gv.GuideConfig(name="v3_nov", source=arm.p("src"), use_screen=True)
                sig = gv.fit_signals(gv.build_archive(pop.data, space), space, cfg, 1.0)
                guide_time += sig.fit_time_s; extra.update(sig.diag())
                ops = gv.vanilla_operators(pop)
                ops.screen = v3_ops.make_nov_screen(space, counter, arm.p("k"), st, sig, float(arm.p("veto")), float(arm.p("cmin")))
                ops.screen_pairs = int(arm.p("m"))
                gv.evolve_with(pop, ops, st)
            extra.update(st)
            tot1, bad1 = vc.snapshot()
            extra.update(generation=pop.current_generation, proposals=tot1 - tot0, invalid=bad1 - bad0,
                         diversity=gv.pop_diversity(pop.members), n_unique=gv.n_unique(pop.members),
                         n_arch_evaluated=len(counter.counts), max_arch_evals=max(counter.counts.values()))
            gen_extra.append(extra)
        valid_total, valid_bad = vc.snapshot()
    hist = {h["generation"]: h for h in pop.best_history}
    for e in gen_extra:
        hist[e["generation"]].update(e)
    vals = np.array(pop.data["validation_accuracy"]); tests = np.array(pop.data["test_accuracy"], dtype=float)
    sel = int(np.argmax(vals))
    bsf = [hist[g]["best_so_far"] for g in sorted(hist)]
    return {
        "arm": arm.spec, "kind": arm.kind, "importance_source": arm.source, "seed": seed, "status": "ok",
        "cfg": {"alpha": arm.alpha, "beta": arm.beta, "v3": dict(arm.params)},
        "common_hash": R.common_hash(common), "d_ref": d_ref,
        "history": [hist[g] for g in sorted(hist)],
        "final_best_so_far_val": float(bsf[-1]), "auc_best_so_far_val": float(np.mean(bsf[R.WARMUP + 1:])),
        "selected_test": float(tests[sel]), "selected_val": float(vals[sel]),
        "n_evals": int(cev.n - evals0), "wall_time_s": time.perf_counter() - t0, "guidance_time_s": guide_time,
        "valid_rate": float(1 - valid_bad / max(1, valid_total)), "validity_checks": int(valid_total),
        "env": R.env_versions(),
    }


_orig_run_cell = R.run_cell


def run_cell(arm, seed, common, space, cev):
    if arm.kind == "v3":
        return run_cell_v3(arm, seed, common, space, cev)
    return _orig_run_cell(arm, seed, common, space, cev)


def known_seed_sets():
    """Every seed that any registered set uses (pilot/dev/dev_b/primary/confirm, diag, kstudy, kstudy_b)."""
    ref = set(D.diag_seeds())
    for s in SEED_SETS_ALL:
        ref |= set(D._orig_seed_list(s))
    for name in EXTRA_SEEDS:
        ref |= set(extra_seeds(name))
    return ref


def parametric_seeds(spec):
    """'ss:<entropy>:<n>' -> n seeds from SeedSequence(entropy); refuses any overlap with a registered set, so a fresh
    dev / primary set needs no code change (just a new entropy). The entropy is recorded in the manifest seed_set name."""
    _, ent, n = spec.split(":")
    seeds = [int(x) for x in SeedSequence(int(ent)).generate_state(int(n))]
    clash = set(seeds) & known_seed_sets()
    assert not clash, "seed set %s overlaps registered sets: %s" % (spec, sorted(clash)[:3])
    return seeds


def seed_list(name):
    if name.startswith("ss:"):
        return parametric_seeds(name)
    if name in EXTRA_SEEDS:
        return extra_seeds(name)
    return D.seed_list_with_diag(name)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    assert_disjoint()
    R.parse_arm, R.run_cell = parse_arm, run_cell               # dispatcher first, so diag_runner wraps it
    saved_seed = R.seed_list
    try:
        out = argv[argv.index("--out") + 1] if (argv and argv[0] == "run" and "--out" in argv) else None
        if out is None:
            R.seed_list = seed_list
            return R.main(argv)
        rec, uninstall = D.install(out, None)
        R.seed_list = seed_list                                  # after install(): extends diag's seed list
        try:
            return R.main(argv)
        finally:
            uninstall()
    finally:
        R.parse_arm, R.run_cell, R.seed_list = _orig_parse_arm, _orig_run_cell, saved_seed


if __name__ == "__main__":
    sys.exit(main())
