"""Test/sandbox bootstrap. Makes `selfmade.nb101` importable when the private
`selfmade.abstract` and the `nasbench` package are NOT installed.
In your real environment neither stub is used (real modules win)."""
import sys, types, importlib, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def install_stubs():
    try:
        importlib.import_module("selfmade.abstract")
    except Exception:
        m = types.ModuleType("selfmade.abstract")
        class Genome:
            def __init__(self, representation):
                self.representation = representation
                self.fitness = None
                self.metrics = None
            def __repr__(self):
                return f"Genome({self.representation}, fitness={self.fitness})"
        m.Genome = Genome
        m.Evaluator = type("Evaluator", (), {})
        m.SearchSpace = type("SearchSpace", (), {})
        m.FeatureImportance = type("FeatureImportance", (), {})
        sys.modules["selfmade.abstract"] = m
        import selfmade
        selfmade.abstract = m
    try:
        importlib.import_module("nasbench.api")
    except Exception:
        pkg = types.ModuleType("nasbench"); api = types.ModuleType("nasbench.api")
        api.ModelSpec = type("ModelSpec", (), {"__init__": lambda s, matrix, ops: None})
        api.NASBench = type("NASBench", (), {"__init__": lambda s, *a, **k: (_ for _ in ()).throw(
            RuntimeError("real NASBench dataset not available in sandbox"))})
        pkg.api = api
        sys.modules["nasbench"] = pkg; sys.modules["nasbench.api"] = api
install_stubs()
