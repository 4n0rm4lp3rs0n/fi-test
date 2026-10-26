"""
Confidence-aware, diversity-preserving feature-importance guidance (v2).

nb101.py / nbcore.py are NOT modified. Everything here is opt-in:
  * GuideConfig(alpha=0 or no operator enabled)  ->  baseline Population.evolve() is called,
    i.e. bit-identical to vanilla (see tests/test_guided_v2.py::test_exact_vanilla_fallback).
  * Guidance never sees test accuracy: the archive holds only `fitness` (= validation accuracy).

Design in one paragraph (derivations: docs/DIAGNOSIS_AND_PLAN.md, section 2)
  Every guided operator is a *tilt of the vanilla distribution* by a log-odds vector `ell`
  (bits: ell_i toward bit=1; operations: ell_{j,o}).  ell = 0 reproduces vanilla.
      ell = clip( gain * a_t * c_t * m * theta , -ell_max, +ell_max )
  theta : empirical-Bayes-shrunk standardized marginal effect (unobserved / tiny-n cells -> 0)
  m     : importance weight in [0,1] from the ACTIVE importance source ("gini" | "perm")
  c_t   : global confidence = positive grouped-CV rank correlation of a surrogate * n/(n+n0)
  a_t   : diversity factor in [0,1] (1 unless adaptive=True and population diversity collapsed)
  gain  : alpha (edges) / beta (operations)
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from selfmade import nb101, nbcore

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GuideConfig:
    name: str = "vanilla"
    source: str = "gini"          # ACTIVE importance source: "gini" | "perm"  (recorded in every output)
    alpha: float = 8.0            # gain for edge (bit) tilts
    beta: float = 8.0             # gain for operation tilts
    use_imm: bool = False         # guided "immigrants" (init-style sampling) replace n_imm offspring / generation
    n_imm: int = 1
    use_mut: bool = False         # guided mutation
    use_cut: bool = False         # guided crossover cut point (ablation only)
    use_screen: bool = False      # surrogate pre-selection of offspring (interaction-aware, exploratory)
    screen_m: int = 6             # max candidate pairs per slot pair when screening
    adaptive: bool = False        # fade guidance when diversity falls below rho * D_ref
    rho: float = 0.5
    ell_max: float = math.log(3)  # no edge/op ever gets odds more extreme than 3:1 against vanilla
    p_floor: float = 0.05         # minimum probability of any operation
    mut_fmin: float = 0.5         # mutation-rate multiplier bounds (exploration stays possible)
    mut_fmax: float = 2.0
    n0: float = 20.0              # sample-size shrinkage of the confidence: n / (n + n0)
    min_n: int = 10               # fewer unique evaluated rows -> no guidance
    cv_folds: int = 5
    rf_trees: int = 100
    perm_repeats: int = 3

    def operators_on(self) -> bool:
        return self.use_imm or self.use_mut or self.use_cut or self.use_screen

    def enabled(self) -> bool:
        return self.operators_on() and (self.alpha > 0 or self.beta > 0)

    def to_dict(self):
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Small numerical helpers (all outputs finite; probabilities sum to 1)
# --------------------------------------------------------------------------------------


def sanitize(x, fill=0.0):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, fill)


def sigmoid(x):
    x = np.clip(sanitize(x), -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def op_probs(ell_row, p_floor):
    """softmax(ell) mixed with uniform so every operation has probability >= p_floor."""
    ell = sanitize(ell_row)
    k = len(ell)
    z = np.exp(ell - ell.max())
    p = z / z.sum()
    eps = min(1.0, k * float(p_floor))
    p = (1 - eps) * p + eps / k
    p = np.clip(p, 0.0, 1.0)
    return p / p.sum()


def pop_diversity(members) -> float:
    """Mean pairwise normalized Hamming distance over (21 bits + 5 operations)."""
    vecs = []
    for g in members:
        vecs.append([c for c in g.code if c in "01"] + list(g.operations[1:-1]))
    n = len(vecs)
    if n < 2:
        return 0.0
    L = len(vecs[0])
    tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += sum(a != b for a, b in zip(vecs[i], vecs[j]))
    return tot / (L * n * (n - 1) / 2)


def n_unique(members) -> int:
    return len({(g.code, tuple(g.operations)) for g in members})


def adapt_factor(d_now, d_ref, rho) -> float:
    if d_ref <= 0:
        return 1.0
    return float(np.clip(d_now / (rho * d_ref), 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Validity accounting (valid-architecture rate) without touching nb101.py on disk
# --------------------------------------------------------------------------------------


class ValidityCounter:
    def __init__(self):
        self.total = 0
        self.invalid = 0

    def snapshot(self):
        return self.total, self.invalid


@contextlib.contextmanager
def count_validity():
    """Temporarily wraps nb101.valid_architecture to count every validity check."""
    counter = ValidityCounter()
    orig = nb101.valid_architecture

    def wrapped(mat, edge_limit=-1):
        ok, why = orig(mat, edge_limit)
        counter.total += 1
        counter.invalid += (not ok)
        return ok, why

    nb101.valid_architecture = wrapped
    try:
        yield counter
    finally:
        nb101.valid_architecture = orig


# --------------------------------------------------------------------------------------
# Feature <-> genome indexing (single source of truth; unit-tested)
# --------------------------------------------------------------------------------------


def n_bits(space):
    """Segments of length 1..layer_limit-1 -> 21 bits for 7 nodes."""
    return sum(range(1, space.layer_limit))


def n_op_positions(space):
    return space.layer_limit - 2                              # internal nodes (=5)


def bit_to_edge(i, space):
    """0-based flat bit index -> (src, dst) node pair, matching nb101.string_to_matrix."""
    seg = 0
    while i >= seg + 1:
        i -= seg + 1
        seg += 1
    return i, seg + 1                                         # bit c of segment s is edge c -> s+1


def bit_feature_name(i):          # 0-based -> name used by FeatureImportance/Guidance
    return f"code_bit{i + 1}"


def op_feature_name(j):           # 0-based internal position -> name used by Guidance ("operations2".."operations6")
    return f"operations{j + 2}"


# --------------------------------------------------------------------------------------
# Archive (deduplicated evaluated rows) and signal estimation
# --------------------------------------------------------------------------------------


@dataclass
class Archive:
    bits: np.ndarray      # (n, NB) int
    ops: np.ndarray       # (n, NP) int index into space.operations
    y: np.ndarray         # (n,) fitness (validation accuracy)
    groups: np.ndarray    # (n,) architecture id (for grouped CV: no architecture straddles train/test)


def build_archive(data, space) -> Archive:
    """data = Population.data. Drops rows repeated only because elites are re-recorded every generation."""
    op_index = {o: i for i, o in enumerate(space.operations)}
    seen, bits, ops, y, gid, gmap = set(), [], [], [], [], {}
    for rep, fit in zip(data["representation"], data["fitness"]):
        key = (rep["code"], tuple(rep["operations"]), float(fit))
        if key in seen:
            continue
        seen.add(key)
        bits.append([int(ch) for ch in rep["code"] if ch in "01"])
        ops.append([op_index[o] for o in rep["operations"][1:-1]])
        y.append(float(fit))
        gid.append(gmap.setdefault((rep["code"], tuple(rep["operations"])), len(gmap)))
    return Archive(np.array(bits, dtype=int).reshape(len(y), -1), np.array(ops, dtype=int).reshape(len(y), -1),
                   np.array(y, dtype=float), np.array(gid, dtype=int))


def _design(arch: Archive, K, NB, NP):
    n = len(arch.y)
    X = np.zeros((n, NB + NP * K), dtype=float)
    X[:, :NB] = arch.bits
    for j in range(NP):
        X[np.arange(n), NB + j * K + arch.ops[:, j]] = 1.0
    blocks = [[i] for i in range(NB)] + [list(range(NB + j * K, NB + (j + 1) * K)) for j in range(NP)]
    return X, blocks


def eb_shrink(d, se):
    """Empirical-Bayes (method-of-moments, positive-part) normal shrinkage of z-like effects.
    tau2 = max(0, mean(d^2 - se^2)) capped at 1; theta = d * tau2/(tau2+se^2).
    If the effects are no larger than their noise (tau2=0) every theta is exactly 0 (= vanilla)."""
    d = np.asarray(d, float)
    se = np.asarray(se, float)
    ok = np.isfinite(d) & np.isfinite(se) & (se > 0)
    theta = np.zeros_like(d)
    if ok.sum() < 3:
        return theta, 0.0
    tau2 = float(np.clip(np.mean(d[ok] ** 2 - se[ok] ** 2), 0.0, 1.0))
    if tau2 > 0:
        theta[ok] = d[ok] * tau2 / (tau2 + se[ok] ** 2)
    return theta, tau2


def importance_weights(imp):
    """m = clip(I / mean(I), 0, 1) within a group. Features at/above average importance get full weight,
    so one dominant feature cannot suppress the rest (max-normalisation would)."""
    imp = np.clip(sanitize(imp), 0, None)
    mu = imp.mean() if len(imp) else 0.0
    return np.zeros_like(imp) if mu <= 0 else np.clip(imp / mu, 0.0, 1.0)


@dataclass
class Signals:
    ell_bits: np.ndarray
    ell_ops: np.ndarray
    c: float = 0.0
    a: float = 1.0
    n: int = 0
    tau2_bits: float = 0.0
    tau2_ops: float = 0.0
    source: str = "gini"
    cv_rho: float = 0.0
    fit_time_s: float = 0.0
    surrogate: object = None   # only filled when cfg.use_screen

    @property
    def active(self) -> bool:
        return bool(np.any(self.ell_bits != 0) or np.any(self.ell_ops != 0))

    def diag(self):
        return {"c": self.c, "a": self.a, "n_archive": self.n, "tau2_bits": self.tau2_bits, "tau2_ops": self.tau2_ops,
                "cv_rho": self.cv_rho, "mean_abs_ell_bits": float(np.mean(np.abs(self.ell_bits))),
                "mean_abs_ell_ops": float(np.mean(np.abs(self.ell_ops))), "max_abs_ell": float(max(np.abs(self.ell_bits).max(initial=0), np.abs(self.ell_ops).max(initial=0))),
                "fit_time_s": self.fit_time_s}


def zero_signals(space, source="gini", n=0, a=1.0):
    return Signals(np.zeros(n_bits(space)), np.zeros((n_op_positions(space), len(space.operations))), n=n, a=a, source=source)


def fit_signals(arch: Archive, space, cfg: GuideConfig, a_t: float = 1.0) -> Signals:
    t0 = time.perf_counter()
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    NB, NP, K = n_bits(space), n_op_positions(space), len(space.operations)
    n = len(arch.y)
    sig = zero_signals(space, cfg.source, n=n, a=a_t)
    s = float(np.std(arch.y, ddof=1)) if n > 1 else 0.0
    n_groups = len(set(arch.groups.tolist()))
    k = min(cfg.cv_folds, n_groups)
    if n < cfg.min_n or s < 1e-12 or k < 3:
        sig.fit_time_s = time.perf_counter() - t0
        return sig                                            # not enough evidence -> vanilla behaviour

    X, blocks = _design(arch, K, NB, NP)
    y = arch.y
    oof = np.full(n, np.nan)
    perm_acc = np.zeros(len(blocks))
    rng = np.random.default_rng(0)
    for tr, te in GroupKFold(n_splits=k).split(X, y, arch.groups):
        m = RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=0, n_jobs=1).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        oof[te] = pred
        if cfg.source == "perm":
            base = np.mean((pred - y[te]) ** 2)
            stack = []
            for cols in blocks:
                for _ in range(cfg.perm_repeats):
                    Xp = X[te].copy()
                    Xp[:, cols] = X[te][rng.permutation(len(te))][:, cols]
                    stack.append(Xp)
            pp = m.predict(np.vstack(stack)).reshape(len(blocks), cfg.perm_repeats, len(te))
            perm_acc += (((pp - y[te][None, None, :]) ** 2).mean(axis=2).mean(axis=1) - base) / k

    rho = 0.0
    if np.nanstd(oof) > 1e-12:
        r = spearmanr(y, oof)[0]
        rho = float(r) if np.isfinite(r) else 0.0
    c = max(0.0, rho) * n / (n + cfg.n0)
    sig.cv_rho, sig.c = rho, c

    if cfg.source == "gini":
        full = RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=0, n_jobs=1).fit(X, y)
        fi = full.feature_importances_
        block_imp = np.array([fi[cols].sum() for cols in blocks])
    else:
        block_imp = perm_acc
    m_bits, m_ops = importance_weights(block_imp[:NB]), importance_weights(block_imp[NB:])

    # --- marginal standardized effects, shrunk -------------------------------------------------
    yc = y - y.mean()
    d_b, se_b = np.full(NB, np.nan), np.full(NB, np.nan)
    for i in range(NB):
        n1 = int(arch.bits[:, i].sum()); n0 = n - n1
        if n1 >= 2 and n0 >= 2:
            d_b[i] = (y[arch.bits[:, i] == 1].mean() - y[arch.bits[:, i] == 0].mean()) / s
            se_b[i] = math.sqrt(1.0 / n1 + 1.0 / n0)
    th_b, sig.tau2_bits = eb_shrink(d_b, se_b)

    d_o, se_o = np.full((NP, K), np.nan), np.full((NP, K), np.nan)
    for j in range(NP):
        for o in range(K):
            mask = arch.ops[:, j] == o
            no = int(mask.sum())
            if 2 <= no < n:
                d_o[j, o] = (y[mask].mean() - y.mean()) / s
                se_o[j, o] = math.sqrt(1.0 / no - 1.0 / n)
    th_o, sig.tau2_ops = eb_shrink(d_o.ravel(), se_o.ravel())
    th_o = th_o.reshape(NP, K)

    amp_b = cfg.alpha * a_t * c
    amp_o = cfg.beta * a_t * c
    sig.ell_bits = np.clip(sanitize(amp_b * m_bits * th_b), -cfg.ell_max, cfg.ell_max)
    sig.ell_ops = np.clip(sanitize(amp_o * m_ops[:, None] * th_o), -cfg.ell_max, cfg.ell_max)

    if cfg.use_screen:
        sig.surrogate = RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=0, n_jobs=1).fit(X, y)
    sig.fit_time_s = time.perf_counter() - t0
    return sig


# --------------------------------------------------------------------------------------
# Constrained edge sampling: exact conditional law of independent Bernoulli(p_i) given sum <= K
# --------------------------------------------------------------------------------------


def sample_bits_constrained(p, K):
    """Sequential exact sampling from  prod_i Bern(p_i)  conditioned on  sum_i x_i <= K.
    DP: W[i][k] = P(sum_{t>=i} x_t <= k).  Controls the *actual* edge count (not just its mean)
    while preserving the relative odds of the tilted probabilities."""
    p = np.clip(sanitize(p, 0.5), 1e-9, 1 - 1e-9)
    N = len(p)
    K = int(min(K, N))
    W = np.zeros((N + 1, K + 1))
    W[N, :] = 1.0
    for i in range(N - 1, -1, -1):
        for k in range(K + 1):
            W[i, k] = (1 - p[i]) * W[i + 1, k] + (p[i] * W[i + 1, k - 1] if k >= 1 else 0.0)
    x, k = [], K
    for i in range(N):
        num = p[i] * (W[i + 1, k - 1] if k >= 1 else 0.0)
        p1 = num / W[i, k]
        bit = random.random() < p1
        x.append(int(bit))
        k -= int(bit)
    return x


def sample_guided_genome(space, sig: Signals, cfg: GuideConfig, max_tries=200):
    """Guided 'immigrant': tilted Bernoulli edges (edge limit enforced exactly), tilted operations.
    Remaining validity rules (reachability) by rejection; falls back to vanilla random_genome()."""
    p_bits = sigmoid(sig.ell_bits)                            # vanilla prior is p = 0.5 (ell = 0)
    NP = n_op_positions(space)
    for _ in range(max_tries):
        bits = sample_bits_constrained(p_bits, space.edge_limit)
        code = nb101.rebuild_code([str(b) for b in bits], space.actual_layers + 1)
        ok, _ = nb101.valid_architecture(nb101.string_to_matrix(code), space.edge_limit)
        if ok:
            break
    else:
        return space.random_genome(), True
    ops = ["input"]
    for j in range(NP):
        pr = op_probs(sig.ell_ops[j], cfg.p_floor)
        ops.append(random.choices(space.operations, weights=pr)[0])
    ops.append("output")
    return nb101.GenomeNB101(code, ops), False


# --------------------------------------------------------------------------------------
# Guided operators (each is vanilla when ell = 0)
# --------------------------------------------------------------------------------------


def guided_mutation_v2(space, genome, sig: Signals, cfg: GuideConfig, mu):
    """Asymmetric flip rates  q(b->1-b) = clip(mu * exp(-0.5*ell_i*(2b-1)), [fmin, fmax]*mu).
    The two-state chain's stationary odds are exp(ell_i) = the same tilt used for immigrants, but the rates stay
    within [mu*fmin, mu*fmax] so every bit can still move.  Operations: mutate-away rate scaled by how much the
    current op is above the softmax-mean tilt; replacement drawn from softmax(ell) over the *other* ops."""
    NB = n_bits(space)
    original = nb101.flatten_code(genome.code)
    code = None
    for _ in range(nb101.MAX_PAIR_RETRIES):
        pre = original.copy()
        for i in range(NB):
            b = int(pre[i])
            f = float(np.clip(math.exp(-0.5 * sig.ell_bits[i] * (2 * b - 1)), cfg.mut_fmin, cfg.mut_fmax))
            if random.random() < min(0.5, mu * f):
                pre[i] = "0" if b == 1 else "1"
        cand = nb101.rebuild_code(pre, space.actual_layers + 1)
        if space.validate(cand)[0]:
            code = cand
            break
    if code is None:
        code = genome.code                                    # (baseline vanilla_mutation returns a *list* here: bug)
    ops = list(genome.operations)
    index = {o: i for i, o in enumerate(space.operations)}
    for j in range(n_op_positions(space)):
        cur = index[ops[j + 1]]
        e = sig.ell_ops[j]
        pr = np.exp(e - e.max()); pr /= pr.sum()
        f = float(np.clip(math.exp(-0.5 * (e[cur] - float(pr @ e))), cfg.mut_fmin, cfg.mut_fmax))
        if random.random() < min(0.5, mu * f):
            others = [o for o in range(len(space.operations)) if o != cur]
            w = np.exp(e[others] - e[others].max())
            ops[j + 1] = space.operations[random.choices(others, weights=w)[0]]
    return nb101.GenomeNB101(code, ops)


def _cut_weights(ell_unsigned):
    """P(cut after position c) ∝ exp(-(|ell_c| + |ell_{c+1}|)): avoid cutting between two informative genes."""
    r = ell_unsigned[:-1] + ell_unsigned[1:]
    w = np.exp(-r)
    return (w / w.sum()).tolist()


def guided_cut_crossover(space, parent1, parent2, sig: Signals):
    """Same retry/validity logic as NASBench101Space.vanilla_crossover; only the cut-point law differs."""
    g1, g2 = nb101.flatten_code(parent1.code), nb101.flatten_code(parent2.code)
    wb = _cut_weights(np.abs(sig.ell_bits))
    cuts = list(range(1, len(g1)))
    ok = False
    for _ in range(nb101.MAX_PAIR_RETRIES):
        cut = random.choices(cuts, weights=wb)[0]
        c1 = nb101.rebuild_code(g1[:cut] + g2[cut:], space.actual_layers + 1)
        c2 = nb101.rebuild_code(g2[:cut] + g1[cut:], space.actual_layers + 1)
        if space.validate(c1)[0] and space.validate(c2)[0]:
            ok = True
            break
    if not ok:
        return None, None
    o1, o2 = parent1.operations[1:-1], parent2.operations[1:-1]
    wo = _cut_weights(np.abs(sig.ell_ops).max(axis=1))
    oc = random.choices(list(range(1, len(o1))), weights=wo)[0]
    return (nb101.GenomeNB101(c1, ["input"] + o1[:oc] + o2[oc:] + ["output"]),
            nb101.GenomeNB101(c2, ["input"] + o2[:oc] + o1[oc:] + ["output"]))


def featurize(space, genome):
    NB, NP, K = n_bits(space), n_op_positions(space), len(space.operations)
    x = np.zeros(NB + NP * K)
    x[:NB] = [int(ch) for ch in genome.code if ch in "01"]
    for j in range(NP):
        x[NB + j * K + space.operations.index(genome.operations[j + 1])] = 1.0
    return x


# --------------------------------------------------------------------------------------
# Generic evolve step. With vanilla callables it consumes the RNG in exactly the order of
# nbcore.Population.evolve(), so it is a faithful drop-in (tested).
# --------------------------------------------------------------------------------------


@dataclass
class Operators:
    cross: callable            # (p1, p2) -> (c1, c2) | (None, None)
    mutate: callable           # genome -> genome
    immigrant: callable = None  # () -> genome ; None => no immigrants
    n_imm: int = 0
    screen: callable = None    # (children) -> children, see evolve_with
    screen_pairs: int = 1


def vanilla_operators(pop) -> Operators:
    return Operators(cross=lambda a, b: pop.space.vanilla_crossover(a, b),
                     mutate=lambda g: pop.space.vanilla_mutation(g, pop.space.operations, pop.mutation_rate))


def evolve_with(pop, ops: Operators, stats: dict = None):
    pop.current_generation += 1
    gen = pop.current_generation
    pop.config["generations"] = gen
    nxt = pop.elitism(pop.elite_size)
    imm_left = ops.n_imm if ops.immigrant is not None else 0
    while len(nxt) < pop.population_size:
        while True:
            p1 = pop.selection(pop.selector, pop.survivors, pop.candidates_per_round)[0]
            p2 = pop.selection(pop.selector, pop.survivors, pop.candidates_per_round)[0]
            if p1 is not p2:
                break
        if ops.screen is not None and ops.screen_pairs > 1:
            pool = []
            for _ in range(ops.screen_pairs):
                a, b = ops.cross(p1, p2)
                if a is None and b is None:
                    continue
                pool += [ops.mutate(a), ops.mutate(b)]
            if len(pool) < 2:
                continue
            m1, m2 = ops.screen(pool)
        else:
            c1, c2 = ops.cross(p1, p2)
            if c1 is None and c2 is None:
                continue
            m1, m2 = ops.mutate(c1), ops.mutate(c2)
        if imm_left > 0:
            g, fell_back = ops.immigrant()
            if stats is not None:
                stats["imm"] = stats.get("imm", 0) + 1
                stats["imm_fallback"] = stats.get("imm_fallback", 0) + int(fell_back)
            m2, imm_left = g, imm_left - 1
        for child in (m1, m2):
            pop.single_evaluation(child)
            if len(nxt) < pop.population_size:
                nxt.append(child)
    pop.members = nxt
    for g in pop.members:
        if g.fitness is None:
            pop.single_evaluation(g)
    pop.record(gen)


def operators_for(pop, cfg: GuideConfig, sig: Signals) -> Operators:
    """Build the operator bundle for one generation from the current signals."""
    space = pop.space
    ops = vanilla_operators(pop)
    if cfg.use_screen and sig.surrogate is not None:          # whole-architecture surrogate: independent of marginal tilts
        m_eff = 1 + int(round((cfg.screen_m - 1) * min(1.0, sig.c) * sig.a))
        if m_eff > 1:
            def screen(pool):
                pred = sig.surrogate.predict(np.array([featurize(space, g) for g in pool]))
                top = np.argsort(-pred)[:2]
                return pool[top[0]], pool[top[1]]
            ops.screen, ops.screen_pairs = screen, m_eff
    if not sig.active:
        return ops
    if cfg.use_cut:
        ops.cross = lambda a, b: guided_cut_crossover(space, a, b, sig)
    if cfg.use_mut:
        ops.mutate = lambda g: guided_mutation_v2(space, g, sig, cfg, pop.mutation_rate)
    if cfg.use_imm:
        ops.immigrant = lambda: sample_guided_genome(space, sig, cfg)
        ops.n_imm = cfg.n_imm
    return ops


# --------------------------------------------------------------------------------------
# Baseline (existing) guidance with an EXPLICIT importance source. nbcore.py is untouched.
# --------------------------------------------------------------------------------------


class SourceGuidance(nbcore.Guidance):
    """Identical to nbcore.Guidance except sortNsplit reads the importance table named by `source_key`
    ("df_imp" = Gini, "perm_imp" = permutation) instead of the hand-edited line in nbcore.py."""

    def __init__(self, *args, source_key="df_imp", **kwargs):
        super().__init__(*args, **kwargs)
        if source_key not in ("df_imp", "perm_imp"):
            raise ValueError(source_key)
        self.source_key = source_key

    def sortNsplit(self):
        df = self.fi[self.source_key].copy()
        extracted = df["feature"].str.extract(r"^([A-Za-z_]+)\s*(\d+)$")
        df["family"] = extracted[0]
        df["position"] = nbcore.pd.to_numeric(extracted[1], errors="coerce")
        df["type"] = df["feature"].map(self.feature_types)
        df = df[df["type"] != "constant"].copy()
        if df["type"].isna().any():
            raise ValueError(f"No feature type found for: {df.loc[df['type'].isna(), 'feature'].tolist()}")
        groups = {}
        for (family, feature_type), group in df.groupby(["family", "type"], sort=True):
            groups[(family, feature_type)] = group.sort_values("position", ascending=True).reset_index(drop=True)
        return groups


SOURCE_KEY = {"gini": "df_imp", "perm": "perm_imp"}