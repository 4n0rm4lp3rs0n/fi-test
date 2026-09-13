'''
Genome = Matrix/String + Operations
'''
import numpy as np
import random
from nasbench import api
from nb201.nas_201_api import NASBench201API as API
from pathlib import Path
import pandas as pd
from collections import Counter
import json
import selfmade.abstract

from time import time

# GLOBAL PARAMETERS
MAX_PAIR_RETRIES = 1000

class GenomeNB101(selfmade.abstract.Genome):
    def __init__(self, code, operations):
        super().__init__(code, operations)

    def random_genome(self, operations, layer_limit, edge_limit):
        ops = self.random_operation(layer_limit, operations)
        # dimension decrease to match nasbench's evaluation requirement
        dims = len(ops) - 1
        while True:
            code = rebuild_code(self.random_gen(round((dims*(dims+1)) / 2)), dims)
            valid, _ = valid_architecture(string_to_matrix(code), edge_limit)
            if valid:
                break
        return GenomeNB101(code, ops)

    def guided_genome(self, layer_limit, edge_limit, guidance : classmethod):
        ops = guided_layers(layer_limit, guidance.prob_layers)
        dims = len(ops) - 1
        # print("generating guided bits...")
        while True:
            code = guided_bits(dims, guidance.p_final_bits)
            valid, _ = valid_architecture(string_to_matrix(code), edge_limit)
            if valid:
                break
        return GenomeNB101(code, ops)

    def random_string(num_nodes):
        genome = []
        for i in range(1, num_nodes + 1):
            gene = ''.join(
                random.choice(['0', '1']) 
                for _ in range(i)
            )
            genome.append(gene)
        return '-'.join(genome)

    def random_operation(limit, operations):
        '''
        Genrating operations randomly
        
        Needs limit and operation list (list of strings)
        '''
        ops = ['input']
        operation_nums = limit - 2 # one for input and one for output
        while operation_nums:
            ops.append(random.choice(operations))
            operation_nums -= 1
        ops.append('output')
        return ops

def flatten_code(code):
    return list(code.replace("-", ""))
 
def rebuild_code(flat_bits, layers):
    segs, pos = [], 0
    for L in range(1, layers + 1):
        segs.append("".join(flat_bits[pos:pos + L]))
        pos += L
    return "-".join(segs)

def string_to_matrix(string):
    cons = string.split('-')
    dims = len(cons)
    mat = np.zeros((dims+1, dims+1), dtype=int)
    for i in range(dims):
        for c in range(len(cons[i])):
            if int(cons[i][c]) == 1:
                mat[i+1][c] = 1
    mat = mat.T
    return mat

def valid_architecture(mat, edge_limit = -1):
    '''A function used to check if a genome works
    
    Conditions:
    - Input has outgoing nodes
    - Output has incoming nodes
    - Input must reach output (Direct/Indirect)
    - Edges and vertices limit (optional)
    '''
    
    # output must have incoming edge
    if np.sum(mat[:, -1]) == 0:
        return False, "output must have incoming edge"
    
    # input must have outgoing edge
    if np.sum(mat[0, :]) == 0:
        return False, "input must have outgoing edge"
    
    # edge limit (-1 if no limit)
    if edge_limit != -1:
        if np.sum(mat) > edge_limit:
            return False, "too many edges"
    
    # input must reach output
    reachable = input_reachable(mat)
    if not reachable[-1]:
        return False, "input must reach output"

    # allow for bit1 to be 0 or 1, so beginning to check reachable at 2
    # for i in range(1, len(mat) - 1):
    for i in range(2, len(mat) - 1):
        if not reachable[i]:
            return False, f"unreachable node detected at layer {i}"
    
    # # no isolated nodes
    # for i in range(1, len(mat)-1):
    #     if np.sum(mat[i,:]) == 0 and np.sum(mat[:,i]) == 0:
    #         return False, "no isolated nodes"
    return True, None

def input_reachable(mat):
    '''
    Check if every intermediate nodes is reachable from input, and at least one path reaches the output
    '''
    n = len(mat)
    reachable = [False] * n
    reachable[0] = True      # input is reachable

    for node in range(n):
        if reachable[node]:
            for nxt in range(n):
                if mat[node, nxt] == 1:
                    reachable[nxt] = True

    return reachable

def guided_bits(num_nodes, probs : list):
    """created bits should be 1 level lower than expected"""
    genome = []

    for i in range(len(probs)):
        p_one = probs[i]
        p_zero = 1 - p_one
        bit = np.random.choice(["0", "1"], p = [p_zero, p_one])
        genome.append(bit)

    genome = rebuild_code(genome, num_nodes)

    return genome
    # return '-'.join(genome)

def guided_layers(limit, guidance : dict):
    ops = ['input']
    operation_nums = limit - 2 # one for input and one for output
    for o in range(1, operation_nums + 1):
        operations, probs = zip(*guidance[f"layer {o}"])
        ops.append(np.random.choice(operations, p = probs))
    ops.append('output')
    return ops

# Output results
def parse_out(path_dir : Path, run_res : dict, eval_res : dict):
    run_checks = {"population_data", "configs"}
    eval_checks = {"df_importance", "bit_directions", "layer_report", "r2", "mae", "corr", "p_value"}
    
    if not path_dir.exists() or not any(path_dir.iterdir()):
        path_dir.mkdir(parents=True)

    if run_res.keys() < run_checks:
        raise KeyError(f"run_res is missing values: {run_res.keys()}")
    if eval_res.keys() < eval_checks:
        raise KeyError(f"eval_res is missing values: {eval_res.keys()}")

    df = run_res["population_data"]
    cfg = run_res["configs"]

    cfg["r2"] = eval_res["r2"]
    cfg["mae"] = eval_res["mae"]
    cfg["corr"] = eval_res["corr"]
    cfg["p_value"] = eval_res["p_value"]
    
    df.to_csv(path_dir / "population.csv")
    eval_res["df_importance"].to_csv(path_dir / "fi.csv")
    eval_res["bit_directions"].to_csv(path_dir / "bitgui.csv")
    eval_res["layer_report"].to_csv(path_dir / "layergui.csv")

    with open(path_dir / "config.json", "w") as c:
        json.dump(cfg, c, indent=4)