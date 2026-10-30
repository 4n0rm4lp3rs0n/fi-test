"""V3 building blocks (new file; nb101.py / nbcore.py / guided_v2.py / exp_runner.py are NOT modified).

ArchKey            isomorphism-invariant hash of the PRUNED architecture (nasbench ModelSpec.hash_spec) - uses structure only.
ArchCounter        evaluator wrapper that counts evaluations per architecture (reads nothing but the genome).
dedup_operators    vanilla operators + a pool screen that enforces a per-architecture re-evaluation cap k (k=None -> no cap).

NOTHING here may read fitness/val/test of other architectures or any benchmark ground truth; see tests/test_v3.py.
Python >= 3.7 compatible.
"""
from __future__ import annotations

import numpy as np

from nasbench import api
from addons import guided_v2 as gv
from selfmade import nb101

AVAILABLE_OPS = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]    # nasbench config order (hash input)


def arch_key(space, genome):
    d = space.decode(genome)
    spec = api.ModelSpec(matrix=np.array(d["matrix"]), ops=list(d["operations"]))
    if not getattr(spec, "valid_spec", True):
        return "invalid:%s|%s" % (genome.code, ",".join(genome.operations))
    return spec.hash_spec(AVAILABLE_OPS)


def descriptor(space, genome):
    """Isomorphism-invariant structural descriptor of the PRUNED architecture, scaled to ~[0,1]:
    [#nodes/7, #edges/9, longest input->output path/6, #conv3x3/5, #conv1x1/5, #maxpool/5, input->output skip]."""
    d = space.decode(genome)
    spec = api.ModelSpec(matrix=np.array(d["matrix"]), ops=list(d["operations"]))
    if not getattr(spec, "valid_spec", True):
        return np.zeros(7)
    m, ops = np.array(spec.matrix), list(spec.ops)
    n = len(ops)
    depth = [0] * n
    for j in range(1, n):
        depth[j] = max([depth[i] + 1 for i in range(j) if m[i][j]] or [0])
    inner = ops[1:-1]
    return np.array([n / 7.0, float(m.sum()) / 9.0, depth[-1] / 6.0, inner.count("conv3x3-bn-relu") / 5.0,
                     inner.count("conv1x1-bn-relu") / 5.0, inner.count("maxpool3x3") / 5.0, float(m[0][-1])])


def desc_dist(a, b):
    return float(np.mean(np.abs(a - b)))


class ArchCounter(object):
    """Wraps an evaluator; `counts[key]` = number of evaluations of that architecture so far; `desc[key]` = descriptor."""

    def __init__(self, inner, space, counts=None, descs=None):
        self.inner, self.space = inner, space
        self.counts = dict(counts or {})
        self.desc = dict(descs or {})

    def evaluate(self, genome):
        res = self.inner.evaluate(genome)
        k = arch_key(self.space, genome)
        self.counts[k] = self.counts.get(k, 0) + 1
        if k not in self.desc:
            self.desc[k] = descriptor(self.space, genome)
        return res

    def __getattr__(self, name):
        return getattr(self.inner, name)


def common_counts(common_pop, space):
    """Evaluation counts of the shared init + warm-up generation, derived from the pop's public data rows.
    Rows: generation 0 = all members evaluated; generation 1 = elite(s) first (NOT re-evaluated) then children."""
    reps, gens = common_pop.data["representation"], common_pop.data["generation"]
    elite = common_pop.elite_size
    counts, seen_gen = {}, {}
    for r, g in zip(reps, gens):
        j = seen_gen.get(g, 0); seen_gen[g] = j + 1
        if g > 0 and j < elite:
            continue
        gen = nb101.GenomeNB101(r["code"], r["operations"])
        k = arch_key(space, gen)
        counts[k] = counts.get(k, 0) + 1
    return counts


def common_descs(common_pop, space):
    out = {}
    for r in common_pop.data["representation"]:
        g = nb101.GenomeNB101(r["code"], r["operations"])
        k = arch_key(space, g)
        if k not in out:
            out[k] = descriptor(space, g)
    return out


def make_dedup_screen(space, counter, k, stats):
    """Pool screen. Walk the pool in generation order and take the first two children whose architecture still has
    evaluation budget (count + already-chosen-in-this-pair < k). If fewer than two qualify, fill with the least-evaluated
    children (ties: pool order) and count a fallback. k=None disables the cap (first two pool children)."""

    def screen(pool):
        keys = [arch_key(space, g) for g in pool]
        stats["pool_total"] = stats.get("pool_total", 0) + len(pool)
        stats["pool_unique"] = stats.get("pool_unique", 0) + len(set(keys))
        stats["slots"] = stats.get("slots", 0) + 1
        if k is None:
            return pool[0], pool[1]
        virt, chosen = {}, []
        for i, key in enumerate(keys):
            if counter.counts.get(key, 0) + virt.get(key, 0) < k:
                chosen.append(i); virt[key] = virt.get(key, 0) + 1
                if len(chosen) == 2:
                    break
        if len(chosen) < 2:
            stats["fallback"] = stats.get("fallback", 0) + 1
            rest = sorted((i for i in range(len(pool)) if i not in chosen),
                          key=lambda i: (counter.counts.get(keys[i], 0) + virt.get(keys[i], 0), i))
            for i in rest:
                chosen.append(i); virt[keys[i]] = virt.get(keys[i], 0) + 1
                if len(chosen) == 2:
                    break
        return pool[chosen[0]], pool[chosen[1]]
    return screen


def dedup_operators(pop, space, counter, k, m, stats):
    ops = gv.vanilla_operators(pop)
    ops.screen, ops.screen_pairs = make_dedup_screen(space, counter, k, stats), int(m)
    return ops


def make_nov_screen(space, counter, k, stats, sig, veto, cmin):
    """D3 (dev-stage hypothesis). Pool -> eligible (arch count + virtual < k, one child per architecture) ->
    slot 1 = first eligible in pool order (PROTECTED: ignores surrogate and novelty) ->
    slot 2 = among survivors of a surrogate VETO (drop the lowest `veto` fraction of predicted values; only when the
    surrogate exists, sig.c >= cmin and >= 3 eligible children) the child with the largest minimum structural distance to
    (all evaluated architectures + slot 1). Falls back to the D1 rule when nothing qualifies."""
    base = make_dedup_screen(space, counter, k, stats)

    def screen(pool):
        keys = [arch_key(space, g) for g in pool]
        seen, elig = set(), []
        for i, key in enumerate(keys):
            if key not in seen and (k is None or counter.counts.get(key, 0) < k):
                seen.add(key); elig.append(i)
        stats["pool_total"] = stats.get("pool_total", 0) + len(pool)
        stats["pool_unique"] = stats.get("pool_unique", 0) + len(set(keys))
        stats["slots"] = stats.get("slots", 0) + 1
        if len(elig) < 2:
            stats["slots"] -= 1                                  # base() counts it
            stats["pool_total"] -= len(pool); stats["pool_unique"] -= len(set(keys))
            return base(pool)
        first = elig[0]
        rest = [i for i in elig[1:]]
        used_veto = False
        if getattr(sig, "surrogate", None) is not None and sig.c >= cmin and len(elig) >= 3 and veto > 0:
            pred = sig.surrogate.predict(np.array([gv.featurize(space, pool[i]) for i in elig]))
            cut = np.quantile(pred, veto)
            keep = [i for i, p in zip(elig, pred) if p > cut and i != first]
            if keep:
                rest, used_veto = keep, True
        stats["veto_used"] = stats.get("veto_used", 0) + int(used_veto)
        ref = list(counter.desc.values()) + [descriptor(space, pool[first])]
        best, best_nov = None, -1.0
        for i in rest:
            dsc = descriptor(space, pool[i])
            nov = min(desc_dist(dsc, r) for r in ref)
            if nov > best_nov + 1e-12:
                best, best_nov = i, nov
        stats["nov_sum"] = stats.get("nov_sum", 0.0) + best_nov
        return pool[first], pool[best]
    return screen


def guided_dedup_operators(pop, cfg, sig, space, counter, k, m, stats, vfrac=0.0):
    """D2 on top of D1: V2 guided mutation / guided immigrants (from `sig`), a protected vanilla fraction `vfrac` of the
    mutation and immigrant calls, and the D1 dedup pool screen. alpha = beta = 0 (or sig inactive) == D1 exactly."""
    import random
    vops = gv.vanilla_operators(pop)
    ops = gv.operators_for(pop, cfg, sig)
    if vfrac > 0:
        g_mut = ops.mutate
        if g_mut is not vops.mutate and g_mut != vops.mutate:
            ops.mutate = lambda g, gm=g_mut, vm=vops.mutate, vf=vfrac: vm(g) if random.random() < vf else gm(g)
        if ops.n_imm > 0:
            g_imm = ops.immigrant
            ops.immigrant = lambda gi=g_imm, vf=vfrac: ((space.random_genome(), False) if random.random() < vf else gi())
    ops.screen, ops.screen_pairs = make_dedup_screen(space, counter, k, stats), int(m)
    return ops
