"""
SYNTHETIC NASBench-101-like evaluator — for SMOKE TESTS ONLY.

It is NOT NASBench-101 and results obtained with it say NOTHING about real
NASBench-101 performance. It exists so the harness, checkpointing, resume,
unit tests and statistics code can be exercised without the 1 GB dataset.

Mimicked properties of the real API (nasbench.query):
  * the spec is pruned (nodes not on an input->output path are removed);
  * identical pruned specs give identical expected accuracy;
  * each query returns ONE of 3 repeated trainings chosen with Python's global
    `random` module (so evaluation consumes the global RNG stream, as the real
    evaluator does);
  * `evaluate()` returns the same keys as NASBench101Evaluator.
Ground truth includes marginal op/edge effects AND pairwise interactions.
"""
import hashlib
import random
import numpy as np
from selfmade import nb101


def prune(matrix, ops):
    n = len(matrix)
    fwd = [False] * n; fwd[0] = True
    for i in range(n):
        if fwd[i]:
            for j in range(n):
                if matrix[i][j]: fwd[j] = True
    bwd = [False] * n; bwd[-1] = True
    for i in range(n - 1, -1, -1):
        for j in range(n):
            if matrix[i][j] and bwd[j]: bwd[i] = True
    keep = [i for i in range(n) if fwd[i] and bwd[i]]
    m = np.array(matrix)[np.ix_(keep, keep)]
    return m, [ops[i] for i in keep]


def _h(*parts):
    d = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64          # uniform [0,1)


class SyntheticNB101Evaluator:
    OP_EFF = {"conv3x3-bn-relu": 0.012, "conv1x1-bn-relu": 0.004,
              "maxpool3x3": -0.010, "input": 0.0, "output": 0.0}

    def __init__(self, space, world_seed=0):
        self.space = space
        self.world_seed = world_seed
        self.n_queries = 0

    def expected_val(self, genome):
        d = self.space.decode(genome)
        mat, ops = prune(d["matrix"], d["operations"])
        n = len(ops)
        if n < 3:
            return 0.80
        s = 0.905 + 0.004 * (n - 2)                         # deeper helps a bit
        s += sum(self.OP_EFF[o] for o in ops[1:-1])
        s -= 0.0015 * max(0, int(mat.sum()) - 7)            # too dense hurts
        # pairwise interaction: adjacent conv3x3 chain helps, maxpool->maxpool hurts
        for i in range(1, n - 2):
            if mat[i][i + 1]:
                a, b = ops[i], ops[i + 1]
                if a == b == "conv3x3-bn-relu": s += 0.006
                if a == b == "maxpool3x3": s -= 0.008
                if a == "maxpool3x3" and b == "conv3x3-bn-relu": s += 0.004
        if mat[0][-1]: s += 0.002                            # skip input->output
        s += 0.02 * (_h(self.world_seed, "arch", mat.tobytes(), tuple(ops)) - 0.5) * 0.35
        return float(min(s, 0.955))

    def evaluate(self, genome):
        self.n_queries += 1
        mu = self.expected_val(genome)
        d = self.space.decode(genome)
        mat, ops = prune(d["matrix"], d["operations"])
        rep = random.randrange(3)                            # global RNG, like the real API
        key = (self.world_seed, mat.tobytes(), tuple(ops), rep)
        val = mu + (_h("val", *key) - 0.5) * 0.006
        test = mu - 0.002 + (_h("test", *key) - 0.5) * 0.007
        return {"fitness": val, "validation_accuracy": val, "test_accuracy": test,
                "training_time": 1000.0, "train_accuracy": 1.0, "parameters": 1_000_000}
