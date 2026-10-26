"""
Run:  cd selfmade_v2 && python -m unittest discover -s tests -v
(uses the synthetic evaluator; these are CODE-correctness tests, not performance evidence)
"""
import hashlib
import itertools
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.dirname(__file__))
import _env  # noqa: F401  (installs stubs only if the real selfmade.abstract / nasbench are absent)

import numpy as np

from selfmade import nb101, nbcore
from addons.synthetic_eval import SyntheticNB101Evaluator
from addons import guided_v2 as gv, exp_runner as R, exp_stats as S

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(__file__))
BASELINE_SHA = {
    "nb101.py": "3006851d5f122845c663ad56676ee01e4e2586af043ee1c113d008be28c9da71",
    "nbcore.py": "206f2b54ff2cc4a82a8b58cc49453335a3dbb86310f8221661edaf7c72fa7bbf",
}


def fresh():
    space = nb101.NASBench101Space()
    return space, R.CountingEvaluator(SyntheticNB101Evaluator(space))


def random_archive(space, n, seed=0):
    rng = random.Random(seed)
    random.seed(seed)
    ev = SyntheticNB101Evaluator(space)
    bits, ops, y = [], [], []
    for _ in range(n):
        g = space.random_genome()
        bits.append([int(c) for c in g.code if c in "01"])
        ops.append([space.operations.index(o) for o in g.operations[1:-1]])
        y.append(ev.evaluate(g)["fitness"])
    return gv.Archive(np.array(bits), np.array(ops), np.array(y), np.arange(n))


class TestBaselineUntouched(unittest.TestCase):
    def test_hashes(self):
        for fn, sha in BASELINE_SHA.items():
            with open(os.path.join(ROOT, "selfmade", fn), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), sha, f"{fn} was modified")


class TestIndexing(unittest.TestCase):
    def test_bit_to_edge_matches_string_to_matrix(self):
        space = nb101.NASBench101Space()
        nbit = gv.n_bits(space)
        self.assertEqual(nbit, 21)
        for i in range(nbit):
            flat = ["0"] * nbit; flat[i] = "1"
            mat = nb101.string_to_matrix(nb101.rebuild_code(flat, space.actual_layers + 1))
            src, dst = gv.bit_to_edge(i, space)
            self.assertEqual(int(mat.sum()), 1)
            self.assertEqual(int(mat[src][dst]), 1, f"bit {i} -> edge {(src, dst)}")
            self.assertLess(src, dst)

    def test_feature_names_match_guidance_naming(self):
        self.assertEqual(gv.bit_feature_name(0), "code_bit1")
        self.assertEqual(gv.bit_feature_name(20), "code_bit21")
        self.assertEqual(gv.op_feature_name(0), "operations2")      # operations1 / operations7 are input / output
        self.assertEqual(gv.op_feature_name(4), "operations6")

    def test_archive_encoding_and_dedup(self):
        space, cev = fresh()
        random.seed(1)
        pop = nbcore.Population(5, space, cev, logger=__import__("selfmade.null_logger", fromlist=["x"]).NullLogger(), elite_size=1, candidates_per_round=3)
        pop.initialize(); pop.evolve_loop(total_gen=2)
        arch = gv.build_archive(pop.data, space)
        self.assertLess(len(arch.y), len(pop.data["fitness"]))        # elites re-recorded each generation are dropped
        rep = pop.data["representation"][0]
        self.assertEqual(arch.bits[0].tolist(), [int(c) for c in rep["code"] if c in "01"])
        self.assertEqual([space.operations[k] for k in arch.ops[0]], rep["operations"][1:-1])


class TestProbabilities(unittest.TestCase):
    def test_op_probs_finite_nonneg_sum1_floor(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            ell = rng.normal(0, 50, size=3)
            if rng.random() < .2: ell[rng.integers(3)] = np.nan
            if rng.random() < .1: ell[rng.integers(3)] = np.inf
            p = gv.op_probs(ell, 0.05)
            self.assertTrue(np.all(np.isfinite(p)) and np.all(p >= 0.05 - 1e-12))
            self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        np.testing.assert_allclose(gv.op_probs(np.zeros(3), 0.05), np.ones(3) / 3)   # neutral = uniform = vanilla

    def test_sigmoid_zero_is_half(self):
        self.assertEqual(float(gv.sigmoid(0.0)), 0.5)
        self.assertTrue(np.all(np.isfinite(gv.sigmoid(np.array([np.nan, np.inf, -np.inf, 1e9])))))

    def test_eb_shrink(self):
        th, tau2 = gv.eb_shrink([0.1, -0.1, 0.1, -0.1], [1, 1, 1, 1])        # effects below noise
        self.assertEqual(tau2, 0.0); self.assertTrue(np.all(th == 0))
        th, tau2 = gv.eb_shrink([2, -2, 2, 2], [0.1, 0.1, 0.1, 0.1])         # strong, precise effects
        self.assertGreater(tau2, 0.5); np.testing.assert_allclose(th, [2, -2, 2, 2], rtol=0.05)
        th, _ = gv.eb_shrink([np.nan, 1, 1, 1, 1], [np.nan, 1, 1, 1, 1])      # missing -> exactly 0
        self.assertEqual(th[0], 0.0)

    def test_importance_weights(self):
        m = gv.importance_weights([100.0, 1.0, 1.0, 1.0])                    # one dominant feature
        self.assertTrue(np.all((m >= 0) & (m <= 1)))
        self.assertGreater(m[1], 0.0)                                         # does not suppress the others to ~0
        self.assertTrue(np.all(gv.importance_weights([0, 0, 0]) == 0))
        self.assertTrue(np.all(gv.importance_weights([-1, np.nan, 0]) == 0))


class TestConstrainedSampling(unittest.TestCase):
    def test_dp_is_exact_conditional_law(self):
        N, K = 8, 3
        p = np.array([.9, .2, .6, .5, .7, .1, .8, .4])
        subsets = [x for x in itertools.product([0, 1], repeat=N) if sum(x) <= K]
        w = np.array([np.prod([p[i] if x[i] else 1 - p[i] for i in range(N)]) for x in subsets]); w /= w.sum()
        exact_marg = (np.array(subsets) * w[:, None]).sum(axis=0)
        random.seed(0)
        S_ = np.array([gv.sample_bits_constrained(p, K) for _ in range(40000)])
        self.assertTrue((S_.sum(axis=1) <= K).all())
        np.testing.assert_allclose(S_.mean(axis=0), exact_marg, atol=0.012)

    def test_edge_limit_enforced_not_just_in_expectation(self):
        space = nb101.NASBench101Space()
        random.seed(0)
        for _ in range(300):                                                  # p = .99 => mean 20.8 edges unconstrained
            x = gv.sample_bits_constrained(np.full(21, .99), space.edge_limit)
            self.assertLessEqual(sum(x), space.edge_limit)
        cfg = gv.GuideConfig(use_imm=True)
        sig = gv.zero_signals(space); sig.ell_bits[:] = cfg.ell_max; sig.ell_ops[:, 0] = cfg.ell_max
        for _ in range(300):
            g, _fb = gv.sample_guided_genome(space, sig, cfg)
            ok, why = space.validate(g.code)
            self.assertTrue(ok, why)
            self.assertLessEqual(int(nb101.string_to_matrix(g.code).sum()), space.edge_limit)

    def test_neutral_tilt_matches_vanilla_init_distribution(self):
        space = nb101.NASBench101Space(); cfg = gv.GuideConfig(use_imm=True)
        sig = gv.zero_signals(space)
        random.seed(3); np.random.seed(3)
        v = [space.random_genome() for _ in range(4000)]
        random.seed(4)
        g = [gv.sample_guided_genome(space, sig, cfg)[0] for _ in range(4000)]
        e = lambda L: np.mean([nb101.string_to_matrix(x.code).sum() for x in L])
        self.assertAlmostEqual(e(v), e(g), delta=0.15)
        mb = lambda L: np.mean([[int(c) for c in x.code if c in "01"] for x in L], axis=0)
        np.testing.assert_allclose(mb(v), mb(g), atol=0.04)
        fo = lambda L: np.mean([[o == "conv3x3-bn-relu" for o in x.operations[1:-1]] for x in L], axis=0)
        np.testing.assert_allclose(fo(v), fo(g), atol=0.04)


class TestSignals(unittest.TestCase):
    def test_not_enough_evidence_is_vanilla(self):
        space = nb101.NASBench101Space(); cfg = gv.GuideConfig(use_imm=True)
        sig = gv.fit_signals(random_archive(space, 6), space, cfg)            # n < min_n
        self.assertFalse(sig.active); self.assertEqual(sig.c, 0.0)
        a = random_archive(space, 40); a.y[:] = 0.9                           # constant fitness
        self.assertFalse(gv.fit_signals(a, space, cfg).active)

    def test_missing_and_constant_features_are_zero_and_finite(self):
        space = nb101.NASBench101Space()
        for source in ("gini", "perm"):
            cfg = gv.GuideConfig(use_imm=True, source=source)
            a = random_archive(space, 60, seed=2)
            a.bits[:, 0] = 1                                                  # constant edge
            a.ops[:, 2] = np.where(a.ops[:, 2] == 2, 0, a.ops[:, 2])          # operation 2 never observed at position 2
            a.y = a.y + 0.01 * a.bits[:, 5] + 0.01 * (a.ops[:, 0] == 1)       # plant a real signal
            sig = gv.fit_signals(a, space, cfg)
            self.assertTrue(np.all(np.isfinite(sig.ell_bits)) and np.all(np.isfinite(sig.ell_ops)))
            self.assertEqual(sig.ell_bits[0], 0.0); self.assertEqual(sig.ell_ops[2, 2], 0.0)
            self.assertLessEqual(np.abs(sig.ell_bits).max(), cfg.ell_max + 1e-12)
            self.assertLessEqual(np.abs(sig.ell_ops).max(), cfg.ell_max + 1e-12)

    def test_planted_signal_gets_correct_sign(self):
        space = nb101.NASBench101Space(); cfg = gv.GuideConfig(use_imm=True, alpha=8, beta=8)
        a = random_archive(space, 150, seed=5)
        a.y = 0.9 + 0.03 * a.bits[:, 4] - 0.03 * (a.ops[:, 1] == 2) + np.random.default_rng(0).normal(0, .003, 150)
        sig = gv.fit_signals(a, space, cfg)
        self.assertGreater(sig.ell_bits[4], 0)
        self.assertLess(sig.ell_ops[1, 2], 0)

    def test_no_signal_gives_near_zero_tilt(self):
        space = nb101.NASBench101Space(); cfg = gv.GuideConfig(use_imm=True)
        rng = np.random.default_rng(1); mags = []
        for s in range(5):
            a = random_archive(space, 40, seed=10 + s); a.y = rng.normal(0.9, .01, 40)   # pure noise
            sg = gv.fit_signals(a, space, cfg); mags.append(max(np.abs(sg.ell_bits).max(), np.abs(sg.ell_ops).max()))
        self.assertLess(np.median(mags), 0.35)                                # far below ell_max = 1.10

    def test_adapt_factor(self):
        self.assertEqual(gv.adapt_factor(0.3, 0.3, 0.5), 1.0)
        self.assertEqual(gv.adapt_factor(0.0, 0.3, 0.5), 0.0)
        self.assertAlmostEqual(gv.adapt_factor(0.075, 0.3, 0.5), 0.5)
        self.assertEqual(gv.adapt_factor(0.1, 0.0, 0.5), 1.0)


class TestOperators(unittest.TestCase):
    def _random_parents(self, space, n=2):
        return [space.random_genome() for _ in range(n)]

    def test_guided_children_always_valid(self):
        space = nb101.NASBench101Space(); cfg = gv.GuideConfig(use_mut=True, use_cut=True)
        sig = gv.zero_signals(space)
        rng = np.random.default_rng(0)
        sig.ell_bits = rng.uniform(-cfg.ell_max, cfg.ell_max, 21); sig.ell_ops = rng.uniform(-cfg.ell_max, cfg.ell_max, (5, 3))
        random.seed(0); n_valid = 0
        for _ in range(250):
            p1, p2 = self._random_parents(space)
            c1, c2 = gv.guided_cut_crossover(space, p1, p2, sig)
            if c1 is None: continue
            for c in (c1, c2, gv.guided_mutation_v2(space, c1, sig, cfg, 0.05)):
                ok, why = space.validate(c.code); self.assertTrue(ok, why)
                self.assertEqual(len(c.operations), 7); self.assertEqual(c.operations[0], "input")
                self.assertTrue(all(o in space.operations for o in c.operations[1:-1]))
                n_valid += 1
        self.assertGreater(n_valid, 600)

    def test_mutation_rate_stays_in_bounds_and_neutral_equals_vanilla(self):
        """Neutral tilt must reproduce vanilla_mutation's REALIZED flip rate (both retry until valid, so the
        realized rate is below the nominal mu: ~0.03 vs 0.05 -- a property of the baseline, not of v2)."""
        space = nb101.NASBench101Space(); cfg = gv.GuideConfig(use_mut=True); sig = gv.zero_signals(space)
        random.seed(0); g = space.random_genome(); trials = 5000
        flips = lambda c: sum(a != b for a, b in zip(nb101.flatten_code(g.code), nb101.flatten_code(c.code)))
        random.seed(10); van = np.mean([flips(space.vanilla_mutation(g, space.operations, 0.05)) for _ in range(trials)]) / 21
        random.seed(11); neu = np.mean([flips(gv.guided_mutation_v2(space, g, sig, cfg, 0.05)) for _ in range(trials)]) / 21
        self.assertAlmostEqual(neu, van, delta=0.004)
        f = [float(np.clip(np.exp(-0.5 * cfg.ell_max * s), cfg.mut_fmin, cfg.mut_fmax)) for s in (+1, -1)]
        self.assertTrue(all(cfg.mut_fmin <= x <= cfg.mut_fmax for x in f))     # exploration floor/ceiling hold
        self.assertGreater(min(f) * 0.05, 0.0)                                 # preferred bits still mutate

    def test_cut_weights_uniform_when_neutral(self):
        w = gv._cut_weights(np.zeros(21)); self.assertAlmostEqual(sum(w), 1.0); np.testing.assert_allclose(w, 1 / 20)
        w = gv._cut_weights(np.array([0, 0, 1.0, 1.0, 0, 0])); self.assertLess(w[2], w[0])   # avoid cutting between informative genes


class TestExactFallbackAndReproducibility(unittest.TestCase):
    def _pair(self, spec_a, spec_b, seed=5):
        space, cev = fresh()
        common = R.build_common(seed, space, cev)
        ra = R.run_cell(R.parse_arm(spec_a), seed, common, space, cev)
        rb = R.run_cell(R.parse_arm(spec_b), seed, common, space, cev)
        return ra, rb

    def test_exact_vanilla_fallback_when_guidance_disabled(self):
        ra, rb = self._pair("vanilla", "v2_imm_mut_cut@gini:alpha=0,beta=0")
        for x, y in zip(ra["history"], rb["history"]):
            self.assertEqual(x["best_val"], y["best_val"]); self.assertEqual(x["best_test"], y["best_test"])
        self.assertEqual(ra["final_best_so_far_val"], rb["final_best_so_far_val"])
        self.assertEqual(ra["common_hash"], rb["common_hash"]); self.assertEqual(ra["n_evals"], rb["n_evals"])

    def test_zero_signal_generation_is_baseline_path(self):
        ra, rb = self._pair("vanilla", "v2_imm_mut@perm")                       # first guided gen has n < min_n unique rows
        g2 = lambda r: [h for h in r["history"] if h["generation"] == 2][0]
        self.assertEqual(g2(ra)["best_val"], g2(rb)["best_val"])
        self.assertEqual(g2(ra)["diversity"], g2(rb)["diversity"])

    def test_evolve_with_vanilla_ops_equals_baseline_evolve(self):
        space1, cev1 = fresh(); space2, cev2 = fresh()
        p1 = R.build_common(9, space1, cev1); p2 = R.build_common(9, space2, cev2)
        self.assertEqual(R.common_hash(p1), R.common_hash(p2))
        a, b = p1.copy(), p2.copy()
        nb101.set_seed(123)
        for _ in range(4): a.evolve()
        nb101.set_seed(123)
        for _ in range(4): gv.evolve_with(b, gv.vanilla_operators(b))
        self.assertEqual(a.data["fitness"], b.data["fitness"])
        self.assertEqual(a.data["representation"], b.data["representation"])

    def test_same_seed_same_result_and_budget(self):
        space, cev = fresh()
        c1 = R.build_common(7, space, cev); c2 = R.build_common(7, space, cev)
        self.assertEqual(R.common_hash(c1), R.common_hash(c2))
        arm = R.parse_arm("v2_imm_mut@gini")
        r1 = R.run_cell(arm, 7, c1, space, cev); r2 = R.run_cell(arm, 7, c2, space, cev)
        self.assertEqual([h["best_val"] for h in r1["history"]], [h["best_val"] for h in r2["history"]])
        self.assertEqual(r1["n_evals"], 4 * (R.TOTAL_GENS - R.WARMUP))         # pop 5, elite 1 -> 4 evaluations / generation
        v = R.run_cell(R.parse_arm("vanilla"), 7, c1, space, cev)
        self.assertEqual(v["n_evals"], r1["n_evals"])                          # identical budget across arms
        self.assertEqual(v["common_hash"], r1["common_hash"])


class TestExistingGuidanceSource(unittest.TestCase):
    def test_source_guidance_equals_baseline_for_gini_and_runs_for_perm(self):
        from sklearn.ensemble import RandomForestRegressor
        space, cev = fresh()
        pop = R.build_common(3, space, cev)
        imp = nbcore.FeatureImportance(space).pipeline(pop.data, RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1))
        base = nbcore.Guidance(space, imp, pop.config); gs = base.sortNsplit()
        sg = gv.SourceGuidance(space, imp, pop.config, source_key="df_imp"); ss = sg.sortNsplit()
        self.assertEqual(set(gs), set(ss))
        for k in gs:
            self.assertTrue(gs[k].equals(ss[k]))
        gp = gv.SourceGuidance(space, imp, pop.config, source_key="perm_imp"); gp.sortNsplit(); gp.calculate_guidance()
        self.assertIn(("code_bit", "binary"), gp.guides)
        with self.assertRaises(ValueError):
            gv.SourceGuidance(space, imp, pop.config, source_key="bogus")


class TestControlArm(unittest.TestCase):
    def test_kappa0_control_has_zero_guidance_and_runs(self):
        from sklearn.ensemble import RandomForestRegressor
        space, cev = fresh(); pop = R.build_common(3, space, cev)
        fi = nbcore.FeatureImportance(space); rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        g = R._existing_guidance(fi, rf, pop, space, R.parse_arm("existing_ctrl@gini"), kappa0=True)
        self.assertEqual(g.kappa, 0.0)
        bk, lk = space.key_data["bit_key"], space.key_data["layer_key"]
        self.assertTrue(all(g.get_weight(bk, f"code_bit{i}") == 0 for i in range(1, 22)))
        self.assertTrue(np.allclose(g.get_cell(bk, "mutation_prob", "all"), g.mrate))
        r = R.run_cell(R.parse_arm("existing_ctrl@gini"), 3, pop, space, cev)
        self.assertEqual(r["n_evals"], 76)
        self.assertEqual(R.parse_arm("existing_ctrl@perm").kind, "existing_ctrl")

    def test_seed_sets_disjoint(self):
        sets = [set(R.seed_list(k)) for k in ("pilot", "primary", "dev", "confirm")]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(sets[i] & sets[j])


class TestCheckpointResume(unittest.TestCase):
    def test_resume_skips_done_and_recomputes_missing(self):
        out = tempfile.mkdtemp()
        try:
            args = ["run", "--out", out, "--evaluator", "synthetic", "--seeds-list", "1", "2", "3", "--arms", "vanilla"]
            self.assertEqual(R.main(args), 0)
            cells = os.path.join(out, "cells")
            names = sorted(os.listdir(cells)); self.assertEqual(len(names), 3)
            keep = {n: os.path.getmtime(os.path.join(cells, n)) for n in names}
            os.remove(os.path.join(cells, names[1]))                              # simulate an interruption
            self.assertEqual(R.main(args), 0)
            self.assertEqual(len(os.listdir(cells)), 3)
            self.assertEqual(os.path.getmtime(os.path.join(cells, names[0])), keep[names[0]])    # untouched
            self.assertEqual(os.path.getmtime(os.path.join(cells, names[2])), keep[names[2]])
            self.assertFalse([f for f in os.listdir(cells) if ".tmp." in f])      # atomic writes leave no partial files
            with self.assertRaises(SystemExit):                                   # different seeds into same dir is refused
                R.main(["run", "--out", out, "--evaluator", "synthetic", "--seeds-list", "9", "--arms", "vanilla"])
            lines = [json.loads(l) for l in open(os.path.join(out, "progress.jsonl"))]
            self.assertTrue(any(l["event"] == "done" for l in lines))
        finally:
            shutil.rmtree(out)

    def test_arm_needs_explicit_importance_source(self):
        with self.assertRaises(ValueError):
            R.parse_arm("v2_imm_mut")
        self.assertEqual(R.parse_arm("v2_imm_mut_adapt@perm:alpha=2,beta=2").cfg.source, "perm")


class TestStats(unittest.TestCase):
    def test_holm_and_paired(self):
        np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        r = S.paired_stats([1, 2, 3, 4, 5, 6.0], [0, 0, 0, 0, 0, 0.0])
        self.assertEqual((r["wins"], r["ties"], r["losses"]), (6, 0, 0)); self.assertLess(r["wilcoxon_p"], 0.05)
        r = S.paired_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]); self.assertEqual(r["wilcoxon_p"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestScreenArm(unittest.TestCase):
    def test_screen_runs_with_budget_and_ignores_marginal_activity(self):
        space, cev = fresh(); common = R.build_common(4, space, cev)
        for spec in ("v2_screen@gini", "v2_imm_screen@gini"):
            r = R.run_cell(R.parse_arm(spec), 4, common, space, cev)
            self.assertEqual(r["n_evals"], 76)
            self.assertTrue(any(h.get("c", 0) > 0 for h in r["history"] if "c" in h), spec)