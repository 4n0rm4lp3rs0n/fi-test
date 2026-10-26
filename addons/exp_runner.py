"""
Checkpointed paired experiment runner.

    python -m selfmade.exp_runner run --out results/primary --evaluator nb101 \
        --records ./datas/nasbench_full.tfrecord --seeds primary \
        --arms vanilla existing_nb@gini existing_nb@perm v2_imm_mut@gini:alpha=2,beta=2
    python -m selfmade.exp_runner status --out results/primary
    python -m selfmade.exp_runner arms

Checkpointing (nothing is ever lost beyond the single (arm, seed) cell that was running):
  <out>/manifest.json            run settings + seed list + environment versions (resume refuses to mix settings)
  <out>/cells/<arm>__<seed>.json one atomic file per finished (arm, seed): full per-generation trace
  <out>/progress.jsonl           append-only event log (start / done / error with wall-clock + eval counts)
  <out>/common/<seed>.json       the common initial population (hash recorded in every cell; verified on resume)
Re-running the same command resumes: finished cells are skipped, error cells are retried.

Pairing guarantees
  * every arm of a seed starts from the SAME common population (state after init + 1 common generation);
  * before every arm the global RNGs are reset to the same branch seed (common random numbers), so a cell's
    result does not depend on which other arms ran or in which order;
  * identical budget: population 5, 20 generations (1 common + 19 branched), evaluations are counted and stored.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

from selfmade import nb101, nbcore
from addons.null_logger import NullLogger
from addons import guided_v2 as gv

POP_SIZE, TOTAL_GENS, WARMUP = 5, 20, 1
POP_KW = dict(candidates_per_round=3, elite_size=1, survivors=1)     # identical to the notebooks


# ------------------------------------------------------------------------------ seeds
def seed_list(name):
    from numpy.random import SeedSequence
    if name == "primary":
        return [int(x) for x in SeedSequence(20260101).generate_state(30)]
    if name == "pilot":                                               # disjoint from primary, used for tuning only
        return [int(x) for x in SeedSequence(777).generate_state(10)]
    if name == "dev":                                                 # 30 extra seeds for development/ablation BEFORE the primary run
        return [int(x) for x in SeedSequence(4242).generate_state(30)]
    if name == "dev_b":                                               # FRESH development seeds (use only if a redesign is attempted after dev)
        return [int(x) for x in SeedSequence(5151).generate_state(30)]
    if name == "confirm":                                             # fresh 75 seeds, only for a pre-registered confirmation run
        return [int(x) for x in SeedSequence(31337).generate_state(75)]
    if name.endswith(".json"):
        return [int(x) for x in json.load(open(name))]
    raise ValueError(name)


assert not set(seed_list("primary")) & set(seed_list("pilot"))
assert not set(seed_list("confirm")) & (set(seed_list("primary")) | set(seed_list("pilot")))
assert not set(seed_list("dev")) & (set(seed_list("primary")) | set(seed_list("pilot")) | set(seed_list("confirm")))
assert not set(seed_list("dev_b")) & (set(seed_list("primary")) | set(seed_list("pilot")) | set(seed_list("confirm")) | set(seed_list("dev")))


# ------------------------------------------------------------------------------ arms
@dataclasses.dataclass(frozen=True)
class Arm:
    spec: str
    kind: str                       # vanilla | existing_nb | existing_imm | v2
    source: str                     # gini | perm | -
    cfg: gv.GuideConfig = None
    alpha: float = 8.0
    beta: float = 8.0


def parse_arm(spec: str) -> Arm:
    head, _, over = spec.partition(":")
    name, _, source = head.partition("@")
    overrides = {}
    for kv in filter(None, over.split(",")):
        k, v = kv.split("=")
        overrides[k] = float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v
    if name == "vanilla":
        return Arm(spec, "vanilla", "-")
    if source not in ("gini", "perm"):
        raise ValueError(f"{spec}: arm needs an explicit importance source, e.g. {name}@gini")
    a, b = float(overrides.pop("alpha", 8.0)), float(overrides.pop("beta", 8.0))
    if name in ("existing_nb", "existing_refit", "existing_single", "existing_ctrl", "existing_imm"):
        kind = "existing_refit" if name == "existing_nb" else name      # existing_nb kept as alias of existing_refit
        return Arm(spec, kind, source, alpha=a, beta=b)
    if name.startswith("v2_"):
        toks = set(name[3:].split("_"))
        bad = toks - {"imm", "mut", "cut", "screen", "adapt"}
        if bad:
            raise ValueError(f"{spec}: unknown tokens {bad}")
        cfg = gv.GuideConfig(name=name, source=source, alpha=a, beta=b, use_imm="imm" in toks, use_mut="mut" in toks,
                             use_cut="cut" in toks, use_screen="screen" in toks, adaptive="adapt" in toks)
        if overrides:
            ints = {"n_imm", "screen_m", "min_n", "cv_folds", "rf_trees", "perm_repeats"}
            cfg = dataclasses.replace(cfg, **{k: (int(v) if k in ints else v) for k, v in overrides.items()})
        return Arm(spec, "v2", source, cfg, a, b)
    raise ValueError(f"unknown arm {spec}")


def cell_name(spec, seed):
    safe = spec.replace("@", "-").replace(":", "_").replace(",", "_").replace("=", "")
    return f"{safe}__{seed}"


# ------------------------------------------------------------------------------ plumbing
class CountingEvaluator:
    def __init__(self, inner):
        self.inner, self.n = inner, 0

    def evaluate(self, genome):
        self.n += 1
        return self.inner.evaluate(genome)


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def log_event(out, **kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(out, "progress.jsonl"), "a") as f:
        f.write(json.dumps(kw, default=_json_default) + "\n")


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def env_versions():
    import pandas, sklearn, scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__}


def branch_seed(seed):
    from numpy.random import SeedSequence
    return int(SeedSequence([seed, 1]).generate_state(1)[0])


def build_common(seed, space, evaluator):
    nb101.set_seed(seed)
    pop = nbcore.Population(POP_SIZE, space, evaluator, logger=NullLogger(), **POP_KW)
    with quiet():
        pop.initialize()
        pop.evolve_loop(total_gen=WARMUP)
    return pop


def common_hash(pop):
    h = hashlib.sha256()
    for m in pop.members:
        h.update(repr((m.code, tuple(m.operations), round(float(m.fitness), 12))).encode())
    h.update(str(len(pop.data["fitness"])).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------ one (arm, seed)
def run_cell(arm: Arm, seed: int, common, space, cev: CountingEvaluator):
    pop = common.copy()
    n_branch = TOTAL_GENS - WARMUP
    d_ref = gv.pop_diversity(pop.members)
    nb101.set_seed(branch_seed(seed))
    evals0, t0 = cev.n, time.perf_counter()
    gen_extra, guide_time = [], 0.0
    rf = fi = None
    if arm.kind in ("existing_refit", "existing_single", "existing_ctrl", "existing_imm"):
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
        fi = nbcore.FeatureImportance(space)

    with gv.count_validity() as vc, quiet():
        for g in range(n_branch):
            tot0, bad0 = vc.snapshot()
            extra = {"diversity_before": gv.pop_diversity(pop.members)}
            t1 = time.perf_counter()
            if arm.kind == "vanilla":
                pop.evolve()
            elif arm.kind == "existing_refit":                  # notebook cell 8: guidance re-fitted every generation
                gui = _existing_guidance(fi, rf, pop, space, arm)
                guide_time += time.perf_counter() - t1
                pop.guidance = gui
                pop.evolve()
            elif arm.kind in ("existing_single", "existing_ctrl"):   # cell 44: guidance fitted ONCE at the branch point
                if g == 0:                                      # existing_ctrl: same code path with kappa forced to 0
                    pop.guidance = _existing_guidance(fi, rf, pop, space, arm, kappa0=(arm.kind == "existing_ctrl"))
                    guide_time += time.perf_counter() - t1
                pop.evolve()
            elif arm.kind == "existing_imm":
                gui = _existing_guidance(fi, rf, pop, space, arm)
                guide_time += time.perf_counter() - t1
                pop.guidance = None
                ops = gv.vanilla_operators(pop)
                ops.immigrant, ops.n_imm = (lambda gui=gui: (space.guided_genome(gui), False)), 1
                gv.evolve_with(pop, ops)
            else:
                cfg = arm.cfg
                if cfg.enabled():
                    a_t = gv.adapt_factor(extra["diversity_before"], d_ref, cfg.rho) if cfg.adaptive else 1.0
                    sig = gv.fit_signals(gv.build_archive(pop.data, space), space, cfg, a_t)
                    guide_time += sig.fit_time_s
                    extra.update(sig.diag())
                    if sig.active or (cfg.use_screen and sig.surrogate is not None and sig.c > 0):
                        st = {}
                        gv.evolve_with(pop, gv.operators_for(pop, cfg, sig), st)
                        extra.update(st)
                    else:
                        pop.evolve()                              # zero signal -> baseline code path (== vanilla)
                else:
                    pop.evolve()
            tot1, bad1 = vc.snapshot()
            extra.update(generation=pop.current_generation, proposals=tot1 - tot0, invalid=bad1 - bad0,
                         diversity=gv.pop_diversity(pop.members), n_unique=gv.n_unique(pop.members))
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
        "cfg": arm.cfg.to_dict() if arm.cfg else {"alpha": arm.alpha, "beta": arm.beta},
        "common_hash": common_hash(common), "d_ref": d_ref,
        "history": [hist[g] for g in sorted(hist)],
        "final_best_so_far_val": float(bsf[-1]),
        "auc_best_so_far_val": float(np.mean(bsf[WARMUP + 1:])),
        "selected_test": float(tests[sel]), "selected_val": float(vals[sel]),
        "n_evals": int(cev.n - evals0), "wall_time_s": time.perf_counter() - t0, "guidance_time_s": guide_time,
        "valid_rate": float(1 - valid_bad / max(1, valid_total)), "validity_checks": int(valid_total),
        "env": env_versions(),
    }


def _existing_guidance(fi, rf, pop, space, arm, kappa0=False):
    imp = fi.pipeline(pop.data, rf)
    gui = gv.SourceGuidance(space, imp, pop.config, alpha=arm.alpha, beta=arm.beta, source_key=gv.SOURCE_KEY[arm.source])
    if kappa0:
        gui.kappa = 0.0          # CONTROL: all guided weights/mutation reductions = 0; only the operator type differs from vanilla
    gui.sortNsplit()
    gui.calculate_guidance()
    return gui


# ------------------------------------------------------------------------------ driver
def make_evaluator(kind, records, space):
    if kind == "synthetic":
        from addons.synthetic_eval import SyntheticNB101Evaluator
        return SyntheticNB101Evaluator(space)
    return nb101.NASBench101Evaluator(records, space)


def cmd_run(a):
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "common"), exist_ok=True)
    seeds = seed_list(a.seeds) if a.seeds_list is None else [int(s) for s in a.seeds_list]
    arms = [parse_arm(s) for s in a.arms]
    manifest = {"evaluator": a.evaluator, "records": a.records, "pop_size": POP_SIZE, "total_gens": TOTAL_GENS,
                "warmup": WARMUP, "pop_kw": POP_KW, "seeds": seeds, "seed_set": a.seeds, "env": env_versions()}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        keys = ("evaluator", "pop_size", "total_gens", "warmup", "pop_kw", "seeds")
        if any(old[k] != manifest[k] for k in keys) and not a.force:
            sys.exit("manifest mismatch with existing results; use a fresh --out (or --force)")
        manifest["arms_seen"] = sorted(set(old.get("arms_seen", [])) | {x.spec for x in arms})
    else:
        manifest["arms_seen"] = sorted(x.spec for x in arms)
    atomic_write_json(mpath, manifest)

    space = nb101.NASBench101Space()
    cev = CountingEvaluator(make_evaluator(a.evaluator, a.records, space))
    todo = [(s, x) for s in seeds for x in arms
            if not os.path.exists(os.path.join(a.out, "cells", cell_name(x.spec, s) + ".json"))]
    print(f"{len(todo)} cells to run ({len(seeds) * len(arms) - len(todo)} already done)")
    log_event(a.out, event="start", todo=len(todo))
    cache_seed, common = None, None
    for n, (seed, arm) in enumerate(todo, 1):
        try:
            if cache_seed != seed:
                common, cache_seed = build_common(seed, space, cev), seed
                h = common_hash(common)
                cpath = os.path.join(a.out, "common", f"{seed}.json")
                if os.path.exists(cpath):
                    if json.load(open(cpath))["hash"] != h:
                        sys.exit(f"common population for seed {seed} differs from checkpoint (evaluator/env changed?)")
                else:
                    atomic_write_json(cpath, {"seed": seed, "hash": h,
                                              "members": [{"code": m.code, "ops": m.operations, "val": m.fitness} for m in common.members]})
            res = run_cell(arm, seed, common, space, cev)
            atomic_write_json(os.path.join(a.out, "cells", cell_name(arm.spec, seed) + ".json"), res)
            log_event(a.out, event="done", arm=arm.spec, seed=seed, final=res["final_best_so_far_val"],
                      evals=res["n_evals"], wall=round(res["wall_time_s"], 2))
            print(f"[{n}/{len(todo)}] {arm.spec:40s} seed={seed} best_val={res['final_best_so_far_val']:.5f} "
                  f"evals={res['n_evals']} {res['wall_time_s']:.1f}s", flush=True)
        except KeyboardInterrupt:
            log_event(a.out, event="interrupted"); print("interrupted: finished cells are saved; rerun to resume"); return 130
        except Exception as e:                                         # keep going, retry on next resume
            tb = traceback.format_exc()
            atomic_write_json(os.path.join(a.out, "cells", cell_name(arm.spec, seed) + ".error.json"),
                              {"arm": arm.spec, "seed": seed, "status": "error", "error": repr(e), "traceback": tb})
            log_event(a.out, event="error", arm=arm.spec, seed=seed, error=repr(e))
            print(f"[{n}/{len(todo)}] {arm.spec} seed={seed} ERROR {e!r}", flush=True)
    return 0


def cmd_status(a):
    mp = json.load(open(os.path.join(a.out, "manifest.json")))
    cells = os.listdir(os.path.join(a.out, "cells"))
    done = [c for c in cells if c.endswith(".json") and not c.endswith(".error.json")]
    errs = [c for c in cells if c.endswith(".error.json")]
    want = len(mp["seeds"]) * len(mp["arms_seen"])
    print(f"done {len(done)}/{want}  errors {len(errs)}  evaluator={mp['evaluator']}  seed_set={mp['seed_set']}")
    for arm in mp["arms_seen"]:
        k = sum(1 for s in mp["seeds"] if cell_name(arm, s) + ".json" in done)
        print(f"  {arm:45s} {k}/{len(mp['seeds'])}")


def cmd_arms(_):
    print(__doc__)
    print("arm grammar: name[@gini|perm][:key=val,...]\n  vanilla\n  existing_nb@S      notebook protocol (guided crossover+mutation of nbcore.Guidance, refit each generation)\n"
          "  existing_ctrl@S    CONTROL for existing_single: identical but kappa=0 (operator swap only: uniform crossover with f1/(f1+f2) bias, flat mutation)\n"
          "  existing_imm@S     existing Formula 1/2 init probabilities applied through 1 immigrant per generation\n"
          "  v2_<tokens>@S      tokens from {imm,mut,cut,screen,adapt}, e.g. v2_imm_mut_adapt@perm:alpha=4,beta=4\n"
          "  overrides: alpha beta rho n_imm n0 ell_max p_floor screen_m min_n ...")


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True); r.add_argument("--evaluator", choices=["synthetic", "nb101"], required=True)
    r.add_argument("--records"); r.add_argument("--seeds", default="primary"); r.add_argument("--seeds-list", nargs="*")
    r.add_argument("--arms", nargs="+", required=True); r.add_argument("--force", action="store_true")
    s = sub.add_parser("status"); s.add_argument("--out", required=True)
    sub.add_parser("arms")
    a = p.parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "arms": cmd_arms}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())