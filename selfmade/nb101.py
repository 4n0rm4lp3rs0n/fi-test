'''
Genome = Matrix/String + Operations
'''
import numpy as np
import random
from nasbench import api
from pathlib import Path
import pandas as pd
from collections import Counter
import json
import selfmade.abstract as abstract

# GLOBAL PARAMETERS
MAX_PAIR_RETRIES = 10000

class GenomeNB101(abstract.Genome):
    def __init__(self, code, operations):
        representation = {"code" : code,
                          "operations" : operations,
                          }
        super().__init__(representation)

    @property
    def code(self):
        return self.representation["code"]

    @property
    def operations(self):
        return self.representation["operations"]

class NASBench101Evaluator(abstract.Evaluator):

    def __init__(self, data_path, search_space):
        self.nasbench = api.NASBench(data_path)
        self.space = search_space

    def evaluate(self, genome):

        decoded = self.space.decode(genome)
        spec = api.ModelSpec(
            matrix=decoded["matrix"],
            ops=decoded["operations"]
        )

        result = self.nasbench.query(spec)

        return {
                "fitness": result["validation_accuracy"],
                "validation_accuracy": result["validation_accuracy"],
                "test_accuracy": result.get("test_accuracy"),
                "training_time": result.get("training_time"),
                "train_accuracy": result.get("train_accuracy"),
                "parameters": result.get("trainable_parameters"),
            }

class NASBench101Space(abstract.SearchSpace):
    def __init__(self, layer_limit= 7, edge_limit= 9, operations=None,
                 alpha = 4, beta = 4):

        self.layer_limit = layer_limit
        self.edge_limit = edge_limit

        if operations is None:
            operations = [
                "conv1x1-bn-relu",
                "conv3x3-bn-relu",
                "maxpool3x3"
            ]

        self.operations = operations
        self.actual_layers = layer_limit - 2

        self.key_data = {
            "bit_label": "code_bit",
            "bit_type": "binary",
            "layer_label": "operations",
            "layer_type": "categorical",
        }

        self.key_data["bit_key"] = (self.key_data["bit_label"], self.key_data["bit_type"])
        self.key_data["layer_key"] = (self.key_data["layer_label"], self.key_data["layer_type"])
        
    def random_genome(self):
        ops = random_operation(self.layer_limit, self.operations)
        dims = len(ops) - 1

        while True:
            code = random_string(dims)
            valid, _ = valid_architecture(string_to_matrix(code), self.edge_limit)

            if valid:
                return GenomeNB101(code, ops)

    def guided_genome(self, guidance):
        layer_prob_init = guidance.get_cell(self.key_data["layer_key"], "init_prob", "all")
        bit_prob_init = guidance.get_cell(self.key_data["bit_key"], "init_prob", "all")
        
        ops = guided_layers(self.layer_limit, layer_prob_init, self.key_data["layer_label"])
        dims = len(ops) - 1

        while True:
            code = guided_bits(dims, bit_prob_init)
            valid, _ = valid_architecture(string_to_matrix(code), self.edge_limit)

            if valid:
                return GenomeNB101(code, ops)

    def validate(self, code):
        matrix = string_to_matrix(code)
        return valid_architecture(matrix, self.edge_limit)

    def decode(self, genome):
        matrix = string_to_matrix(genome.representation["code"])

        return {
            "matrix" : matrix,
            "operations" : genome.representation["operations"]
        }

    def feature_data(self, genome):

        bits = [int(x) for x in genome.code.replace("-", "")]
        data = {}

        for i, bit in enumerate(bits, start=1):
            data[f"bit{i}"] = bit

        for i, operation in enumerate(genome.operations[1:-1],start=1):
            data[f"layer {i}"] = operation

        return data

    def feature_schema(self):

        schema = []
        num_bits = (self.layer_limit - 2) * (self.layer_limit - 1) // 2

        for i in range(1, num_bits + 1):
            schema.append({
                "name": f"bit{i}",
                "type": "binary",
                "states": [0, 1]
            })

        for i in range(1, self.layer_limit - 1):
            schema.append({
                "name": f"layer {i}",
                "type": "categorical",
                "states": self.operations
            })

        return schema

    def to_genes(self, genome):
        return flatten_code(genome.code)

    def vanilla_crossover(self, parent1, parent2):
        # while True:
        genome1 = self.to_genes(parent1)
        genome2 = self.to_genes(parent2)

        invalids = Counter()
        success = False
        fails = []
        
        for attempt in range(MAX_PAIR_RETRIES):
        
            gen_cut = random.randint(1, len(genome1) - 1)
        
            # child1_code = '-'.join(genome1[:gen_cut] + genome2[gen_cut:])
            # child2_code = '-'.join(genome2[:gen_cut] + genome1[gen_cut:])
            
            child1_code = rebuild_code(genome1[:gen_cut] + genome2[gen_cut:], self.actual_layers + 1)
            child2_code = rebuild_code(genome2[:gen_cut] + genome1[gen_cut:], self.actual_layers + 1)
            
            # print(f"child 1: {child1_code}")
            # print(f"child 2: {child2_code}")
        
            valid1, reason1 = self.validate(child1_code)
            valid2, reason2 = self.validate(child2_code)
        
            if not valid1:
                invalids[reason1] += 1
                fails.append(["child 1", child1_code, reason1])
            if not valid2:
                invalids[reason2] += 1
                fails.append(["child 2", child2_code, reason2])
        
            # if attempt % 1000 == 0:
            #     print(f"attempt no. {attempt}")
        
            if valid1 and valid2:
                success = True
                break

        # print("total attempts: ", attempt)

        # print(invalids)
        # input()

        # Children are still invalid after too many crossovers
        if not success:
            return None, None

        ops1 = parent1.operations[1:-1]
        ops2 = parent2.operations[1:-1]

        ops_cut = random.randint(1, len(ops1) - 1)
        
        child1_ops = ops1[:ops_cut] + ops2[ops_cut:]       
        child2_ops = ops2[:ops_cut] + ops1[ops_cut:]

        child1 = GenomeNB101(child1_code, ["input"] + child1_ops + ["output"])
        child2 = GenomeNB101(child2_code, ["input"] + child2_ops + ["output"])
        
        return child1, child2

    def guided_crossover(self, parent1, parent2, guidance):

        invalids = Counter()
        success = False
        # fails = []
        genome1 = flatten_code(parent1.code)
        genome2 = flatten_code(parent2.code)
        
        f1, f2 = parent1.fitness, parent2.fitness
        base_parents = f1 / (f1 + f2)
        
        for attempt in range(MAX_PAIR_RETRIES):
            c1, c2 = [], []
            for i in range(1, len(genome1) + 1):
                label = self.key_data["bit_label"]
                w = guidance.get_weight(self.key_data["bit_key"], f"{label}{i}")
                b = guidance.get_b_star(self.key_data["bit_key"], f"{label}{i}")
                bias = w * ((genome1[i-1] == b) - (genome2[i-1] == b))
                probs = min(1.0, max(0.0, base_parents + bias))
        
                if random.random() < probs:
                    c1.append(genome1[i-1])
                    c2.append(genome2[i-1])
                else:
                    c1.append(genome2[i-1])
                    c2.append(genome1[i-1])
        
            child1_code = rebuild_code(c1, self.actual_layers + 1)
            child2_code = rebuild_code(c2, self.actual_layers + 1)
        
            valid1, reason1 = self.validate(child1_code)
            valid2, reason2 = self.validate(child2_code)
        
            if not valid1:
                invalids[reason1] += 1
                # fails.append(["child 1", child1_code, reason1])
            if not valid2:
                invalids[reason2] += 1
                # fails.append(["child 2", child2_code, reason2])
        
            # if attempt % 1000 == 0:
            #     print(f"attempt no. {attempt}")
        
            if valid1 and valid2:
                success = True
                break
        
        # Children are still invalid after too many crossovers
        if not success:
            return None, None

        # Operation crossover
        # Exclude IO
        ops1 = parent1.operations[1:-1]
        ops2 = parent2.operations[1:-1]

        child1_ops, child2_ops = [], []
        base_parents = f1 / (f1 + f2)
        
        layer_key = self.key_data["layer_key"]
        layer_label = self.key_data["layer_label"]
    
        for j in range(1, len(ops1) + 1):
            cur_feature = f"{layer_label}{j+1}"
            op1, op2 = ops1[j-1], ops2[j-1]
            g1, g2 = guidance.get_tendency(cur_feature, op1), guidance.get_tendency(cur_feature, op2)
            g_min, g_max = guidance.get_tendency_range(cur_feature)
            span = g_max - g_min
            A = (g1 - g2) / span if span > 0 else 0.0
            w = guidance.get_weight(layer_key, cur_feature)
            p = min(1.0, max(0.0, base_parents + w * A))
            if random.random() < p:
                child1_ops.append(op1)
                child2_ops.append(op2)
            else:
                child1_ops.append(op2)
                child2_ops.append(op1)
        
        child1 = GenomeNB101(child1_code, ["input"] + child1_ops + ["output"])
        child2 = GenomeNB101(child2_code, ["input"] + child2_ops + ["output"])
        
        return child1, child2

    def vanilla_mutation(self, genome, avail_ops, chance=0.05):
        # Genome mutation
        original = list(genome.code)
        # while True:
        invalids = Counter()
        success = False
        for attempt in range(MAX_PAIR_RETRIES):
            pre_code = original.copy()
            for p in range(len(pre_code)):
                if pre_code[p] in ('0', '1'):
                    if random.random() < chance:
                        pre_code[p] = '0' if pre_code[p] == '1' else '1'
            new_code = ''.join(pre_code)
            valid, reason = self.validate(new_code)
        
            if valid:
                success = True
                break
            else:
                invalids[reason] += 1
        
            # if attempt % 1000 == 0:
            #     print(f"attempt no. {attempt}")
        
        # print(invalids)
        # print("total attempts: ", attempt)

        if not success:
            new_code = pre_code

        # Operation mutation  
        # copy() is used to make a new variable 
        # (equal sign means connection to an existed)
        pre_ops = genome.operations.copy()

        # Exclude IO
        for o in range(1, len(pre_ops) - 1):
            if random.random() < chance:
                choices = [op for op in avail_ops if op != pre_ops[o]]
                pre_ops[o] = random.choice(choices)

        return GenomeNB101(new_code, pre_ops)

    def guided_mutation(self, genome, avail_ops, guidance):
        # Guided mutation
        original = flatten_code(genome.code)
        new_code = None
        invalids = Counter()
        success = False
        bit_key = self.key_data["bit_key"]
        for attempt in range(MAX_PAIR_RETRIES):
        # while True:
            pre_code = original.copy()
            for p in range(len(pre_code)):
                cur_bit = f"code_bit{p+1}"
                if (pre_code[p] in ('0', '1') and
                    random.random() < guidance.get_feature(bit_key, "mutation_prob", cur_bit)):
                    pre_code[p] = '0' if pre_code[p] == '1' else '1'
            new_code = rebuild_code(pre_code, self.actual_layers + 1)
            valid, reason = self.validate(new_code)
            if valid:
                success = True
                break
            else:
                invalids[reason] += 1
        
            # if attempt % 100 == 0:
            #     print(f"attempt no. {attempt}")
        
        # print(invalids)
        
        # print("total attempts: ", attempt)
        
        if not success:
            new_code = genome.code

        # Operation mutation  
        # copy() is used to make a new variable 
        # (equal sign means connection to an existed)
        pre_ops = genome.operations.copy()
        # print(pre_ops)
        # Exclude IO
        layer_key = self.key_data["layer_key"]
        layer_label = self.key_data["layer_label"]
        for o in range(1, len(pre_ops) - 1):
            # print(f"current op no.{o}: {pre_ops[o]}")
            cur_feature = f"{layer_label}{o+1}"
            if random.random() < guidance.get_feature(layer_key, "mutation_prob", cur_feature):
                choices = [op for op in avail_ops if op != pre_ops[o]]
                pre_ops[o] = random.choice(choices)

        return GenomeNB101(new_code, pre_ops)

# Helper Functions

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

def random_string(num_nodes):
    genome = []

    for i in range(1, num_nodes + 1):
        gene = ''.join(
            random.choice(['0', '1']) 
            for _ in range(i)
        )
        genome.append(gene)

    return '-'.join(genome)

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

def guided_layers(limit, guidance : dict, label):
    ops = ['input']
    operation_nums = limit - 2 # one for input and one for output
    for o in range(1, operation_nums + 1):
        cur_feature = f"{label}{o+1}"
        operations, probs = zip(*guidance[cur_feature])
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# testing ground
    
if __name__ == "__main__":

    # records = Path('./datas/nasbench_full.tfrecord')
    records = './datas/nasbench_full.tfrecord'
    
    from selfmade.nbcore import Population
    
    space = NASBench101Space()
    evaluator = NASBench101Evaluator(records, space)
    pop = Population(100, space, evaluator)
    print("initializing...")
    pop.initialize()
    pop.evolve(generation=2)

    data = pd.DataFrame(pop.data)
    # cfg = pd.DataFrame(pop.config)

    data.to_csv("data.csv")
    # cfg.to_csv("cfg.csv")
    print(pop.config)
