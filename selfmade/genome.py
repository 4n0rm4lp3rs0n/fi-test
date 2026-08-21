'''
Genome = Matrix/String + Operations
'''
import numpy as np
import random
from nasbench import api
from pathlib import Path
import pandas as pd
import json
from collections import Counter

class Genome:
    def __init__(self, code, operations):
        self.code = code
        self.matrix = string_to_matrix(code)
        self.operations = operations
        self.fitness = None
        self.metrics = None
    # classmethod decorator used to imitate self.Genome
    @classmethod
    def random(cls, operations, layer_limit, edge_limit):
        ops = random_operation(layer_limit, operations)
        dims = len(ops)
        while True:
            code = random_string(dims - 1)
            valid, _ = valid_architecture(string_to_matrix(code), edge_limit)
            if valid:
                break
        return cls(code, ops)

    def __repr__(self):
        return (
            f"Genome(code='{self.code}', "
            # f"Matrix={self.matrix}, "
            f"fitness={self.fitness})"
        )
        
class Population:
    def __init__(self, population_size, layer_nums, edge_nums, evaluator="nasbench"):
        self.population_size = population_size
        self.members = []
        self.edge_limit = edge_nums
        self.layer_nums = layer_nums
        self.evaluator = evaluator
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
            "population_size" : population_size,
            "generations" : None,
            "mutation_rate" : None,
            "elite_size" : None,
            "edge_limit" : edge_nums,
            "layer_limit" : layer_nums,
            "selection" : None,
        }
        
        for i in range(layer_nums - 2):
            self.data[f"layer {i+1}"] = []
        
    def initialize(self, operations):
        '''
        Initiate parents and evaluate immediately
        '''
        self.members = [Genome.random(operations, self.layer_nums, self.edge_limit) 
                        for _ in range(self.population_size)]

        for genome in self.members:
            res = self.evaluator.evaluate(genome)
            genome.metrics = res
            genome.fitness = res["validation_accuracy"]

        self.record(0)
    
    # __repr__ return string, so if the list is done, it will print None, cause error
    def __repr__(self):
        return '\n'.join(str(member) for member in self.members)
            
    def evaluation(self, evaluator):
        for genome in self.members:
            # print(f"Evaluating genome with code {genome.code}")
            res = evaluator.evaluate(genome)
            genome.fitness = res["validation_accuracy"]
            genome.metrics = res
            
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
        genome1 = parent1.code.split(sep="-")
        genome2 = parent2.code.split(sep="-")
        
        while True:
            gen_cut = random.randint(1, len(genome1) - 1)
            
            child1_code = '-'.join(genome1[:gen_cut] + genome2[gen_cut:])
            child2_code = '-'.join(genome2[:gen_cut] + genome1[gen_cut:])
            
            valid1, _ = valid_architecture(string_to_matrix(child1_code), self.edge_limit)
            valid2, _ = valid_architecture(string_to_matrix(child2_code), self.edge_limit)
            
            if valid1 and valid2:
                break
        
        # Operation crossover
        ops1 = parent1.operations[1:-1]
        ops2 = parent2.operations[1:-1]
        
        ops_cut = random.randint(1, len(ops1) - 1)

        inp = ["input"]
        out = ["output"]

        child1_ops = inp + ops1[:ops_cut] + ops2[ops_cut:] + out        
        child2_ops = inp + ops2[:ops_cut] + ops1[ops_cut:] + out
        
        child1 = Genome(child1_code, child1_ops)
        child2 = Genome(child2_code, child2_ops)
        
        return child1, child2
                
    def mutation(self, genome, avail_ops, chance=0.05):
        # Genome mutation
        while True:
            pre_code = list(genome.code)
            for p in range(len(pre_code)):
                if pre_code[p] in ('0', '1'):
                    if random.random() < chance:
                        pre_code[p] = '0' if pre_code[p] == '1' else '1'
            new_code = ''.join(pre_code)
            valid, _ = valid_architecture(string_to_matrix(new_code), self.edge_limit)
            if valid:
                break
        # Operation mutation  
        # copy() is used to make a new variable 
        # (equal sign means connection to an existed)
        pre_ops = genome.operations.copy()
        
        # Exclude IO
        for o in range(1, len(pre_ops) - 1):
            if random.random() < chance:
                choices = [op for op in avail_ops if op != pre_ops[o]]
                pre_ops[o] = random.choice(choices)
                
        # print(f"before mutation: {genome.operations}")
        # print(f"after mutation: {pre_ops}")
        
        return Genome(new_code, pre_ops)
    
    def evolve(self, operations, generation,
               elite_size = 2, selector = "tournament", survivors = 1,
               candidates_per_round = 4, mutation_rate = 0.05):

        # Config should be added once to save a bit of time
        if generation == 1:
            self.config["generations"] = generation
            self.config["mutation_rate"] = mutation_rate
            self.config["elite_size"] = elite_size
            self.config["selection"] = selector

        next_generation = self.elitism(elite_size)
        
        while len(next_generation) < self.population_size:
            while True:
                parent1 = self.selection(selector, survivors, candidates_per_round)[0]
                parent2 = self.selection(selector, survivors, candidates_per_round)[0]
            
                if parent1 is not parent2:
                    break
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutation(child1, operations, mutation_rate)
            child2 = self.mutation(child2, operations, mutation_rate)
            
            for child in (child1, child2):
                # print(child)
                res = self.evaluator.evaluate(child)
                child.metrics = res
                child.fitness = res["validation_accuracy"]
                
                if len(next_generation) < self.population_size:
                    next_generation.append(child)
        
        self.members = next_generation

        
        for genome in self.members:
            if genome.fitness is None:
                res = self.evaluator.evaluate(genome)
                genome.metrics = res
                genome.fitness = res["validation_accuracy"]

        self.record(generation)
        
    def record(self, generation):
        print(f"recording generation {generation}")
        layer_patterns = Counter()
        for m in self.members:
            
            layers = m.operations[1:-1]
            metrics = m.metrics
            
            layer_patterns[tuple(layers)] += 1
            
            self.data["generation"].append(generation)
            self.data["code"].append(m.code)
            self.data["validation_accuracy"].append(metrics["validation_accuracy"])
            self.data["test_accuracy"].append(metrics["test_accuracy"])
            self.data["training_time"].append(metrics["training_time"])
            self.data["train_accuracy"].append(metrics["train_accuracy"])
            self.data["parameters"].append(metrics["trainable_parameters"])

            for i in range(len(layers)):
                self.data[f"layer {i+1}"].append(layers[i])
                
        # for pattern, count in layer_patterns.items():
        #     print(count, pattern)

class Evaluator:
    
    def __init__(self, method):
        self.method = method
        
        if method == "nasbench":
            self.nasbench = api.NASBench('./nasbench_full.tfrecord')
    
    def evaluate(self, genome):
        
        if self.method == "nasbench":
            
            model_spec = api.ModelSpec(matrix=genome.matrix, ops = genome.operations)
            data = self.nasbench.query(model_spec)
            # print(f"Data: {data}")
            
            # fixed_metrics, computed_metrics = self.nasbench.get_metrics_from_spec(model_spec)
            # for epochs in self.nasbench.valid_epochs:
            #     for repeat_index in range(len(computed_metrics[epochs])):
            #         data_point = computed_metrics[epochs][repeat_index]
            #         print('Epochs trained %d, repeat number: %d' % (epochs, repeat_index + 1))
            #         print(data_point)
            # return data["validation_accuracy"]
            return data
        
# Helper functions

def string_to_matrix(string):
    cons = string.split('-')
    dims = len(cons)
    mat = np.zeros((dims + 1, dims + 1), dtype=int)
    for i in range(dims):
        for c in range(len(cons[i])):
            if int(cons[i][c]) == 1:
                mat[i+1][c] = 1
    mat = mat.T
    return mat

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
    
    for i in range(1, len(mat) - 1):
        if not reachable[i]:
            return False, "unreachable node detected"
    
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

# Main pipeline

def run(experiments, operations, population_size, 
        edge_limit = 9, layer_limit = 7, generations = 100, eval_method = "nasbench",
        selector = "tournament", elite_size = 2, survivors = 1,
        candidates_per_round = 4, mutation_rate = 0.05):
    
    # Conditions check:
    if not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("Population must be integer and larger than 0") 
    
    if not isinstance(experiments, int) or experiments <= 0:
        raise ValueError("Experiments must be an integer larger than 0") 
    
    if not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("Population size must be an integer larger than 0") 
    
    if not isinstance(edge_limit, int) or edge_limit <= 0:
        raise ValueError("Edge limit must be an integer larger than 0") 
    
    if not isinstance(layer_limit, int) or edge_limit <= 0:
        raise ValueError("Layer limit must be an integer larger than 0") 
    
    if not isinstance(generations, int) or generations <= 0:
        raise ValueError("Generation must be an integer larger than 0") 
    
    valid_evals = {"nasbench"}
    if eval_method not in valid_evals:
        raise ValueError(f"Unknown evaluation method: {eval_method}")
     
    valid_selectors = {"tournament"}
    if selector not in valid_selectors:
        raise ValueError(f"Unknown selector: {selector}") 

    if not isinstance(elite_size, int) or elite_size < 0:
        raise ValueError("Elite size must be an integer >= 0") 

    if elite_size >= population_size:
        raise ValueError("Elite size must be smaller than population size") 
    
    if not isinstance(survivors, int) or survivors <= 0:
        raise ValueError("Survivors must be an integer larger than 0") 

    if not isinstance(candidates_per_round, int) or candidates_per_round <= 0:
        raise ValueError("Tournament size must be an integer larger than 0") 

    if candidates_per_round > population_size:
        raise ValueError("Tournament size cannot exceed population size") 

    if not isinstance(mutation_rate, (int, float)):
        raise ValueError("Mutation rate must be a number") 

    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1") 

    if not isinstance(operations, (list, tuple)):
        raise ValueError("Operations must be a list or tuple") 

    if len(operations) == 0:
        raise ValueError("Operations cannot be empty") 

    if len(operations) != len(set(operations)):
        raise ValueError("Operations contain duplicates") 

    # Main pipeline

    for i in range(experiments):
        print(f"experiment no.{i + 1}")

        evaluator = Evaluator(eval_method)
        pop = Population(population_size, layer_limit, edge_limit, evaluator)
        pop.initialize(operations)
        print("Population created, evaluating...")
        
        for generation in range(1, generations + 1):
            pop.evolve(operations, generation, elite_size, selector, 
                       survivors, candidates_per_round, mutation_rate)
            
            # for col in pop.data:
            #     if col.startswith("layer"):
            #         print(col)
            #         print(pop.data[f"{col}"])
            
            best = max(pop.members, key = lambda g: g.fitness)
            
            print(f"Generation {generation} - Validation accuracy: {best.fitness}")
            
        df = pd.DataFrame(pop.data)
        root = Path("./records")
        
        i = 1
        while (root / f"exp{i:03d}").exists():
            i += 1
        
        exp_dir = root / f"exp{i:03d}"
        exp_dir.mkdir(parents=True)
        
        df.to_csv(exp_dir / "population.csv", index = False)
        with open(exp_dir / "config.json", "w") as file:
            json.dump(pop.config, file, indent=4)
            
        print(f"log created at {exp_dir}")
    
    
# Testing ground
if __name__ == "__main__":
    # conv1x1 = 'conv1x1-bn-relu'
    # conv3x3 = 'conv3x3-bn-relu'
    # maxpool3x3 = 'maxpool3x3'

    # operations = [conv1x1, conv3x3, maxpool3x3]
    # try:
    #     run(1, operations, population_size=100, generations=100)
    # except ValueError as e:
    #     print(e)

    # print(random_string(7))
    test_list = [random_string(7) for _ in range(100)]
    for t in test_list:
        print(t, valid_architecture(string_to_matrix(t)))