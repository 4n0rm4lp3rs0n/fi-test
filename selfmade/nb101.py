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
import selfmade.abstract as abstract

from time import time

# GLOBAL PARAMETERS
MAX_PAIR_RETRIES = 1000

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
    def code(self):
        return self.representation["operations"]

class NASBench101Evaluator(abstract.Evaluator):

    def __init__(self, data_path, search_space):
        from nasbench import api

        self.api = api.NASBench(data_path)
        self.space = search_space

    def evaluate(self, genome):

        decoded = self.space.decode(genome)
        spec = self.api.ModelSpec(
            matrix=decoded["matrix"],
            ops=decoded["operations"]
        )

        result = self.api.query(spec)

        return {
                "fitness": result["validation_accuracy"],
                "validation_accuracy": result["validation_accuracy"],
                "test_accuracy": result.get("test_accuracy"),
                "training_time": result.get("training_time"),
                "train_accuracy": result.get("train_accuracy"),
                "parameters": result.get("trainable_parameters"),
            }

class NASBench101Space(abstract.SearchSpace):
    def __init__(self, layer_limit=7, edge_limit=7, operations=None,
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
        
        for i in range(self.layer_limit - 2):
            self.data[f"layer {i+1}"] = []

    def random_genome(self):
        ops = random_operation(self.layer_limit, self.operations)
        dims = len(ops) - 1

        while True:
            code = random_string(dims)
            valid, _ = valid_architecture(string_to_matrix(code), self.edge_limit)

            if valid:
                return GenomeNB101(code, ops)

    def guided_genome(self, guidance : classmethod):
        ops = guided_layers(self.layer_limit, guidance.prob_layers)
        dims = len(ops) - 1

        while True:
            code = guided_bits(dims, guidance.p_final_bits)
            valid, _ = valid_architecture(string_to_matrix(code), self.edge_limit)

            if valid:
                return GenomeNB101(code, ops)

    def validate(self, code):
        matrix = string_to_matrix(code)
        return valid_architecture(matrix, self.edge_limit)

    def decode(self, genome):
        matrix = string_to_matrix(genome.code)

        return {
            "matrix" : matrix,
            "operations" : genome.operations
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
        genome1 = self.space.to_genes(parent1.code)
        genome2 = self.space.to_genes(parent2.code)

        invalids = Counter()
        # fails = []
        
        for attempt in range(MAX_PAIR_RETRIES):
        
            gen_cut = random.randint(1, len(genome1) - 1)
        
            child1_code = '-'.join(genome1[:gen_cut] + genome2[gen_cut:])
            child2_code = '-'.join(genome2[:gen_cut] + genome1[gen_cut:])
        
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
                break

        # print("total attempts: ", attempt)

        # Children are still invalid after too many crossovers
        if attempt == MAX_PAIR_RETRIES - 1 and not (child1_code and child2_code):
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
        # fails = []
        genome1 = flatten_code(parent1.code)
        genome2 = flatten_code(parent2.code)
        
        f1, f2 = parent1.fitness, parent2.fitness
        base_parents = f1 / (f1 + f2)
        
        for attempt in range(MAX_PAIR_RETRIES):
            c1, c2 = [], []
            for i in range(1, len(genome1) + 1):
                w = guidance.bit_weight(i)
                b = guidance.get_b_star(i)
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
                break
        
        # Children are still invalid after too many crossovers
        if attempt == MAX_PAIR_RETRIES - 1 and not (child1_code and child2_code):
            return None, None

        # Operation crossover
        # Exclude IO
        ops1 = parent1.operations[1:-1]
        ops2 = parent2.operations[1:-1]

        child1_ops, child2_ops = [], []
        base_parents = f1 / (f1 + f2)
    
        for j in range(1, len(ops1) + 1):
            op1, op2 = ops1[j-1], ops2[j-1]
            g1, g2 = self.guidance.layer_eff(op1, j), self.guidance.layer_eff(op2, j)
            g_min, g_max = self.guidance.layer_eff_range(j)
            span = g_max - g_min
            A = (g1 - g2) / span if span > 0 else 0.0
            w = self.guidance.layer_weight(j)
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
        pre_code = list(genome.code)
        # while True:
        invalids = Counter()
        for attempt in range(MAX_PAIR_RETRIES):
            for p in range(len(pre_code)):
                if pre_code[p] in ('0', '1'):
                    if random.random() < chance:
                        pre_code[p] = '0' if pre_code[p] == '1' else '1'
            new_code = ''.join(pre_code)
            valid, reason = self.validate(new_code)
        
            if valid:
                break
            else:
                invalids[reason] += 1
        
            # if attempt % 1000 == 0:
            #     print(f"attempt no. {attempt}")
        
        # print(invalids)
        # print("total attempts: ", attempt)

        if attempt == MAX_PAIR_RETRIES - 1 and new_code is None:
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
        pre_code = flatten_code(genome.code)
        new_code = None
        invalids = Counter()
        for attempt in range(MAX_PAIR_RETRIES):
        # while True:
            for p in range(len(pre_code)):
                if pre_code[p] in ('0', '1') and random.random() < guidance.get_prob_bit(p + 1):
                    pre_code[p] = '0' if pre_code[p] == '1' else '1'
            new_code = rebuild_code(pre_code, self.actual_layers + 1)
            valid, reason = self.validate(new_code)
            if valid:
                break
            else:
                invalids[reason] += 1
        
            # if attempt % 100 == 0:
            #     print(f"attempt no. {attempt}")
        
        # print(invalids)
        
        # print("total attempts: ", attempt)
        
        if attempt == MAX_PAIR_RETRIES - 1 and new_code is None:
            new_code = pre_code

        # Operation mutation  
        # copy() is used to make a new variable 
        # (equal sign means connection to an existed)
        pre_ops = genome.operations.copy()
        # print(pre_ops)
        # Exclude IO
        for o in range(1, len(pre_ops) - 1):
            # print(f"current op no.{o}: {pre_ops[o]}")
            if random.random() < guidance.get_prob_layer(o):
                choices = [op for op in avail_ops if op != pre_ops[o]]
                pre_ops[o] = random.choice(choices)

        return GenomeNB101(new_code, pre_ops)

class Population:
    """General population for all NASes (maybe)"""

    def __init__(self, population_size, search_space, evaluator,
                 guidance=None, mutation_rate=0.05):
        self.population_size = population_size
        self.space = search_space
        self.evaluator = evaluator
        self.guidance = guidance
        self.mutation_rate = mutation_rate
        self.members = []
        
        self.data = {
                "generation" : [],
                "code" : [],
                "validation_accuracy" : [],
                "test_accuracy" : [],
                "training_time" : [],
                "train_accuracy" : [],
                "parameters" : [],
            }
            
        self.config = {
            "population_size" : None,
            "generations" : None,
            "mutation_rate" : None,
            "elite_size" : None,
            "edge_limit" : self.edge_limit,
            "layer_limit" : self.layer_limit,
            "selection" : None,
            "best_val_acc" : 0,
            "best_genome" : None,
            "best_operation" : None,
        }

    def initialize(self):
        if self.guidance is None:
            self.members = [self.space.random_genome() for _ in range(self.population_size)]
        else:
            self.members = [self.space.guided_genome(self.guidance) for _ in range(self.population_size)]

        self.evaluation()
        self.record(0)

    # __repr__ return string, so if the list is done, it will print None, causing error
    def __repr__(self):
        return '\n'.join(str(member) for member in self.members)

    def evaluation(self):
        for genome in self.members:
            res = self.evaluator.evaluate(genome)

            genome.metrics = res
            genome.fitness = res["fitness"]
            
    def single_evaluation(self, child):
        res = self.evaluator.evaluate(child)
        child.metrics = res
        child.fitness = res["fitness"]

    def selection(self, method = "tournament", survivors = 1, tournament_size = 4):
        '''
        Find the best genomes using these methods:
        - Tournament
        - Roulette
        - Rank selection
        - Truncation
        - Random
        - Stochastic Universal Sampling
        - Boltzmann Selection
        '''

        if tournament_size > len(self.members):
            raise ValueError("Tournament size exceeds population")

        # Tournament
        if method == "tournament":
            winners = []
            for _ in range(survivors):
                contestants = random.sample(self.members, tournament_size)
                winner = max(contestants, key = lambda g: g.fitness)
                contestants.remove(winner)
                winners.append(winner)
                # self.members += contestants
            return winners

    def elitism(self, elites = 2):
        return sorted(self.members, key=lambda g: g.fitness, reverse = True)[:elites]

    def crossover(self, parent1, parent2):
        # Genome crossover
        if self.guidance is None:
            c1, c2 = self.space.vanilla_crossover(parent1, parent2)
        else:
            c1, c2 = self.space.guided_crossover(parent1, parent2, self.guidance)
        return c1, c2

    def mutation(self, genome, avail_ops, chance=0.05):
        # Vanilla mutation
        if self.bit_guidance is None:
            return self.space.vanilla_mutation(genome, avail_ops, chance)
        else:
            return self.space.guided_mutation(genome, avail_ops, self.guidance)

    def evolve(self, operations, generation,
                elite_size = 2, selector = "tournament", survivors = 1,
                candidates_per_round = 4):

        # Config should be added once to save a bit of time
        self.config["generations"] = generation
        # if generation == 1:
        #     self.config["mutation_rate"] = self.mutation_rate
        #     self.config["elite_size"] = elite_size
        #     self.config["selection"] = selector

        next_generation = self.elitism(elite_size)
        # print("varying genes...")
        while len(next_generation) < self.population_size:
            while True:
                parent1 = self.selection(selector, survivors, candidates_per_round)[0]
                parent2 = self.selection(selector, survivors, candidates_per_round)[0]

                if parent1 is not parent2:
                    break

            child1, child2 = self.crossover(parent1, parent2)

            if child1 is None and child2 is None:
                continue

            child1 = self.mutation(child1, self.space.operations, self.mutation_rate)
            child2 = self.mutation(child2, self.space.operations, self.mutation_rate)

            # print(child1)
            # print(child2)

            for child in (child1, child2):
                self.single_evaluation(child)

                if len(next_generation) < self.population_size:
                    next_generation.append(child)

        self.members = next_generation

        for genome in self.members:
            if genome.fitness is None:
                self.single_evaluation(genome)

        self.record(generation)

    def record(self, generation):
        print(f"recording generation {generation}")
        best = max(self.members, key = lambda g: g.fitness)

        self.config["best_val_acc"] = best.fitness
        self.config["best_genome"] = best.code
        self.config["best_operation"] = best.operations

        for m in self.members:

            self.data["generation"].append(generation)
            features = self.space.feature_data(m)
            
            for name, value in features.items():
                if name not in self.data:
                    self.data[name] = []
                self.data.setdefault(name, []).append(value)
                
            for name, value in m.metrics.items():
                if name == "fitness":
                    continue
                self.data.setdefault(name, []).append(value)

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
    
# testing ground
    
if __name__ == "__main__":
    
    space = NASBench101Space()
    evaluator = NASBench101Evaluator()
    pop = Population(100, space, evaluator)
    pop.initialize()
    pop.evolve()