import json
import numpy as np
import pandas as pd
from pathlib import Path

pth = Path('./records/vanilla')

cfg      = json.load(open(pth / "config.json"))
bitgui   = pd.read_csv(pth / "bitgui.csv")
layergui = pd.read_csv(pth /"layergui.csv")
fi       = pd.read_csv(pth / "fi.csv")
pop      = pd.read_csv(pth / "population.csv")   # needed for the layer baseline (§2.1)

# ---- global confidence -----------------------------------------------------
G = np.clip(cfg["r2"] * abs(cfg["corr"]) * (1 - cfg["p_value"]) / (1 + cfg["mae"]), 0, 1)

# print(layergui.dtypes)
# print(bitgui.dtypes)

# bad_mask = pd.to_numeric(layergui["layer 1"], errors="coerce").isna()
# print(layergui.loc[bad_mask, ["operation", "layer 1"]])

# fi_layer = fi[fi["feature"].str.startswith("layer")].copy()
# fi_layer["pos"] = fi_layer["feature"].str.replace("layer ", "").astype(int)

# print(fi_layer)

layergui = layergui.set_index("operation")
layergui.columns = [int(c.replace("layer ", "")) for c in layergui.columns]
print(layergui)
spread = layergui.max() - layergui.min()
G_hat = (spread / spread.max()).reindex(range(1, 5 + 1))
print(G_hat)

exit()
# ---- structural constants from config --------------------------------------
N_bits        = cfg["layer_limit"] * (cfg["layer_limit"] - 1) // 2
num_positions = cfg["layer_limit"] - 2
mutation_rate = cfg["mutation_rate"]

# ============================== FORMULA 1: bits =============================
def compute_p0(edge_limit, N_bits, num_eps=1e-6):
    """edge_limit is a MAX-edges cap. It only binds when 0 <= edge_limit < N_bits.
    Outside that range (sentinel -1, or a cap >= every possible edge) there is
    no structural sparsity information, so fall back to the uninformative 0.5."""
    if edge_limit is None or edge_limit < 0 or edge_limit >= N_bits:
        p0 = 0.5
    else:
        p0 = edge_limit / N_bits
    return np.clip(p0, num_eps, 1 - num_eps)   # numerical safety only


def bit_probabilities(alpha=4.0):
    fi_bits = fi[fi["feature"].str.startswith("bit")].copy()
    fi_bits["I_norm"] = fi_bits["importance"] / fi_bits["importance"].max()

    df = bitgui.merge(fi_bits[["feature", "I_norm"]], on="feature")

    p0 = compute_p0(cfg["edge_limit"], N_bits)
    logit0 = np.log(p0 / (1 - p0))

    # NaN guard: direction is undefined if a bit never varies (mean_1 or mean_0
    # empty in this population) -> zero evidence, not zero-poisoning. See Appendix C.
    direction = df["direction"].fillna(0.0)
    strength  = df["direction_strength"].fillna(0.0)

    g = np.sign(direction) * np.tanh(strength) * df["I_norm"]
    p_guided = 1 / (1 + np.exp(-(logit0 + alpha * G * g)))
    p_final = (1 - mutation_rate) * p_guided + mutation_rate * 0.5

    return dict(zip(df["feature"], p_final))


# ============================== FORMULA 2: layers ===========================
def layer_probabilities(beta=4.0, eps=1e-3):
    fi_layers = fi[fi["feature"].str.startswith("layer")].copy()
    fi_layers["I_norm"] = fi_layers["importance"] / fi_layers["importance"].max()

    result = {}
    for j in range(1, num_positions + 1):
        col = f"layer {j}"
        all_dist = pop[col].value_counts(normalize=True)
        tendency = layergui.set_index("operation")[col]
        I_j = fi_layers.loc[fi_layers["feature"] == col, "I_norm"].values[0]
        ops = np.array(all_dist.index)

        logits = np.array([
            np.log(all_dist[o] + eps) + beta * G * I_j * tendency.get(o, 0.0)
            for o in ops
        ])
        p_guided = np.exp(logits - logits.max())
        p_guided /= p_guided.sum()

        K = len(ops)
        p_final = (1 - mutation_rate) * p_guided + mutation_rate * (1 / K)

        # sort descending by probability, and renormalize so the sum is
        # exactly 1.0 (guards against float drift that np.random.choice
        # is strict about) -> ready to unzip straight into np.random.choice
        order = np.argsort(-p_final)
        ops_sorted   = ops[order]
        probs_sorted = p_final[order]
        probs_sorted = probs_sorted / probs_sorted.sum()

        result[col] = list(zip(ops_sorted, probs_sorted))

    return result


def sample_layer(p_layer, col):
    """Draw one operation for position `col` (e.g. "layer 1") from its guided distribution."""
    operations, probs = zip(*p_layer[col])
    return np.random.choice(operations, p=probs)


p_bit   = bit_probabilities()
p_layer = layer_probabilities()

# usage:
# p_layer["layer 1"] -> [('conv3x3-bn-relu', 0.631), ('maxpool3x3', 0.284), ('conv1x1-bn-relu', 0.086)]
# sample_layer(p_layer, "layer 1")  ->  a single sampled operation string