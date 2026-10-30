"""Tests for the V3 building blocks (dedup arms, runner, offline-oracle isolation). Uses the SYNTHETIC evaluator, so it
needs no dataset, but it needs the real `nasbench` package (ModelSpec.hash_spec). Python >= 3.7."""
import os, sys, json, shutil, tempfile, unittest, warnings, ast, re
sys.path.insert(0, os.path.dirname(__file__))
import _env  # noqa
warnings.simplefilter("ignore")
from nasbench import api
if not hasattr(api.ModelSpec(matrix=__import__("numpy").array([[0, 1], [0, 0]]), ops=["input", "output"]), "hash_spec"):
    raise unittest.SkipTest("real nasbench ModelSpec not available")
from addons import exp_runner as R, v3_runner as V, v3_ops, diag_runner as D
from selfmade import nb101

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["vanilla", "v3_dedup:k=inf,m=6", "v3_dedup:k=1,m=6", "v3_dedup:k=2,m=6"]
SEEDS = [11, 22]


def run(out, arms=ARMS, seeds=SEEDS):
    argv = ["run", "--out", out, "--evaluator", "synthetic", "--arms"] + arms + ["--seeds-list"] + [str(s) for s in seeds]
    assert V.main(argv) == 0


def cells(out):
    r = {}
    for fn in sorted(os.listdir(os.path.join(out, "cells"))):
        c = json.load(open(os.path.join(out, "cells", fn))); r[fn] = c
    return r


class TestV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = tempfile.mkdtemp(), tempfile.mkdtemp()
        run(cls.a); run(cls.b)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.a, True); shutil.rmtree(cls.b, True)

    def test_runner_restores_exp_runner(self):
        self.assertIs(R.parse_arm, V._orig_parse_arm)
        self.assertIs(R.run_cell, V._orig_run_cell)

    def test_reproducible(self):
        ca, cb = cells(self.a), cells(self.b)
        for fn in ca:
            if fn.startswith("v3_dedup"):
                self.assertEqual(ca[fn]["final_best_so_far_val"], cb[fn]["final_best_so_far_val"])
                self.assertEqual([h["best_so_far"] for h in ca[fn]["history"]], [h["best_so_far"] for h in cb[fn]["history"]])

    def test_equal_budget_and_schema(self):
        ca = cells(self.a)
        van = next(c for fn, c in ca.items() if fn.startswith("vanilla"))
        for fn, c in ca.items():
            self.assertEqual(c["n_evals"], 76, fn)
            self.assertEqual(set(c.keys()), set(van.keys()), fn)
            self.assertEqual(len(c["history"]), len(van["history"]), fn)
            self.assertEqual(len(c["history"]), 21, fn)       # generations 0..20

    def test_common_population_shared(self):
        by_seed = {}
        for c in cells(self.a).values():
            by_seed.setdefault(c["seed"], set()).add(c["common_hash"])
        self.assertTrue(all(len(v) == 1 for v in by_seed.values()))

    def test_every_evaluation_logged(self):
        for fn in os.listdir(os.path.join(self.a, "cands")):
            if not fn.startswith("common__"):
                self.assertEqual(sum(1 for _ in open(os.path.join(self.a, "cands", fn))), 76, fn)

    def test_common_counts_match_logged_hashes(self):
        for s in SEEDS:
            rows = [json.loads(l) for l in open(os.path.join(self.a, "cands", "common__%d.jsonl" % s))]
            space = nb101.NASBench101Space()
            pop = R.build_common(s, space, R.CountingEvaluator(R.make_evaluator("synthetic", None, space)))
            counts = v3_ops.common_counts(pop, space)
            self.assertEqual(sum(counts.values()), 9)
            logged = {}
            for r in rows:
                key = v3_ops.arch_key(space, nb101.GenomeNB101(r["code"], r["ops"]))
                logged[key] = logged.get(key, 0) + 1
            self.assertEqual(counts, logged)

    def test_cap_respected_except_logged_fallbacks(self):
        for fn, c in cells(self.a).items():
            if not fn.startswith("v3_dedup") or c["cfg"]["v3"].get("k") is None:
                continue
            k = c["cfg"]["v3"]["k"]
            rows = [json.loads(l) for l in open(os.path.join(self.a, "cands", fn[:-5] + ".jsonl"))]
            com = [json.loads(l) for l in open(os.path.join(self.a, "cands", "common__%d.jsonl" % c["seed"]))]
            cnt, over = {}, 0
            for r in com:
                cnt[r["hash"]] = cnt.get(r["hash"], 0) + 1
            for r in rows:
                if cnt.get(r["hash"], 0) >= k:
                    over += 1
                cnt[r["hash"]] = cnt.get(r["hash"], 0) + 1
            fallbacks = sum(h.get("fallback", 0) for h in c["history"])
            self.assertLessEqual(over, 2 * fallbacks, fn)          # a fallback slot can add at most 2 over-cap evaluations

    def test_k1_means_fewer_duplicates_than_vanilla(self):
        agg = {}
        for fn, c in cells(self.a).items():
            rows = [json.loads(l) for l in open(os.path.join(self.a, "cands", fn[:-5] + ".jsonl"))]
            agg.setdefault(c["arm"], []).append(len({r["hash"] for r in rows}))
        self.assertGreater(sum(agg["v3_dedup:k=1,m=6"]), sum(agg["vanilla"]))

    def test_guided_dedup_neutral_equals_D1(self):
        """v3_gd with alpha=beta=0 must reproduce D1 (v3_dedup:k=1,m=6) bit for bit (exact fallback)."""
        out = tempfile.mkdtemp()
        try:
            run(out, ["v3_dedup:k=1,m=6", "v3_gd:alpha=0,beta=0,k=1,m=6"], [11])
            c = {v["arm"]: v for v in cells(out).values()}
            a, b = c["v3_dedup:k=1,m=6"], c["v3_gd:alpha=0,beta=0,k=1,m=6"]
            self.assertEqual([h["best_so_far"] for h in a["history"]], [h["best_so_far"] for h in b["history"]])
            self.assertEqual(a["final_best_so_far_val"], b["final_best_so_far_val"])
        finally:
            shutil.rmtree(out, True)

    def test_new_arms_budget_and_logging(self):
        out = tempfile.mkdtemp()
        try:
            run(out, ["v3_gd:k=1,m=6,ops_only=1,vfrac=0.5", "v3_nov:k=1,m=6,veto=0.3"], [11])
            for fn, c in cells(out).items():
                self.assertEqual(c["n_evals"], 76, fn)
                self.assertEqual(sum(1 for _ in open(os.path.join(out, "cands", fn[:-5] + ".jsonl"))), 76, fn)
        finally:
            shutil.rmtree(out, True)

    def test_descriptor_is_isomorphism_invariant(self):
        space = nb101.NASBench101Space()
        import random as rnd
        rnd.seed(3); by = {}
        for _ in range(400):
            g = space.random_genome(); by.setdefault(v3_ops.arch_key(space, g), []).append(v3_ops.descriptor(space, g))
        multi = [v for v in by.values() if len(v) > 1]
        self.assertTrue(multi)
        for v in multi:
            for d in v[1:]:
                self.assertTrue((d == v[0]).all())

    def test_arm_parsing(self):
        a = V.parse_arm("v3_dedup:k=inf,m=6"); self.assertIsNone(a.p("k")); self.assertEqual(a.p("m"), 6)
        self.assertEqual(V.parse_arm("v3_dedup:k=2,m=3").p("k"), 2)
        r = V.parse_arm("v3_rimm:k=1,m=6"); self.assertEqual((r.p("k"), r.p("m"), r.p("n_imm")), (1, 6, 1))
        with self.assertRaises(ValueError): V.parse_arm("v3_dedup:k=2")
        with self.assertRaises(ValueError): V.parse_arm("v3_bogus:k=2,m=3")
        self.assertEqual(V.parse_arm("vanilla").kind, "vanilla")

    def test_seed_sets(self):
        V.assert_disjoint()
        self.assertEqual(len(V.extra_seeds("kstudy")), 100)
        self.assertFalse(set(V.extra_seeds("kstudy")) & set(V.extra_seeds("kstudy_b")))
        self.assertEqual(V.extra_seeds("kstudy"), V.extra_seeds("kstudy"))

    def test_decide_script(self):
        from addons import v3_decide
        two = tempfile.mkdtemp()
        try:
            seeds = [11, 22, 33, 44]
            run(two, ["vanilla", "v3_dedup:k=1,m=6"], seeds)
            self.assertEqual(v3_decide.main(["--out", two, "--d1", "v3_dedup:k=1,m=6", "--rivals", "vanilla"]), 0)
            d = json.load(open(os.path.join(two, "decision.json")))
            self.assertEqual(d["n_seeds"], 4); self.assertIn("vanilla", d["pass"])
            self.assertEqual(v3_decide.main(["--out", two, "--rivals", "no_such_arm"]), 2)      # integrity abort
        finally:
            shutil.rmtree(two, True)

    def test_parametric_seed_sets(self):
        a = V.seed_list("ss:20261101:30"); b = V.seed_list("ss:20261101:30"); c = V.seed_list("ss:20261102:75")
        self.assertEqual(a, b); self.assertEqual((len(a), len(c)), (30, 75)); self.assertFalse(set(a) & set(c))
        self.assertFalse(set(a + c) & V.known_seed_sets())
        # a 'new' set that collides with a registered set must be refused
        with self.assertRaises(AssertionError): V.parametric_seeds("ss:%d:%d" % (V.EXTRA_SEEDS["kstudy"][0], 5))

    def test_ground_truth_never_reaches_the_search(self):
        """Only addons/v3_report.py may touch benchmark ground truth (per-architecture means / get_metrics_from_spec)."""
        banned = re.compile(r"get_metrics_from_spec|v3_report|oracle", re.I)
        for rel in ["addons/v3_ops.py", "addons/v3_runner.py", "addons/guided_v2.py", "addons/exp_runner.py",
                    "addons/diag_runner.py", "selfmade/nbcore.py", "selfmade/nb101.py"]:
            p = os.path.join(ROOT, rel)
            if os.path.exists(p):
                self.assertIsNone(banned.search(open(p, encoding="utf-8", errors="ignore").read()), rel)
        for c in cells(self.a).values():
            self.assertFalse(any("mean" in k for k in c.keys()))

    def test_evaluator_result_has_no_oracle_fields(self):
        space = nb101.NASBench101Space()
        ev = R.make_evaluator("synthetic", None, space)
        res = ev.evaluate(space.random_genome())
        self.assertFalse(any("mean" in k for k in res))


if __name__ == "__main__":
    unittest.main()
