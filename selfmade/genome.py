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

from time import time

# GLOBAL PARAMETERS
MAX_PAIR_RETRIES = 1000

class Guidance:
    def __init__(self, feature_importance: pd.DataFrame = None, bit_guidance: pd.DataFrame = None, 
                 layer_guidance : pd.DataFrame = None, config : dict = None, population : pd.DataFrame = None,
                 alpha = 4, beta = 4, eps = 1e-3):
        self.fi = feature_importance
        self.bitgui = bit_guidance
        self.layergui = layer_guidance.set_index("operation")
        self.config = config
        self.population = population
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        if config is not None:
            self.r2 = config["r2"]
            self.mae = config["mae"]
            self.corr = config["corr"]
            self.p_val = config["p_value"]
            self.mrate = config["mutation_rate"]
            self.layer_limit = config["layer_limit"]
            self.edge_limit = config["edge_limit"]

            sbit = self.layer_limit * (self.layer_limit - 1) // 2
            if self.edge_limit is not None or self.edge_limit < 0 or self.edge_limit <= sbit:
                self.sparsity_rate =  np.clip(self.edge_limit / sbit, 1e-6, 1 - 1e-6)
            else:
                self.sparsity_rate = 0.5
        else:
            self.r2 = 0
            self.mae = 0
            self.corr = 0
            self.p_val = 0
            self.mrate = 0
            self.layer_limit = 0
            self.sparsity_rate = 0.5

        self.G = np.clip(self.r2 * abs(self.corr) * (1 - self.p_val) / (1 + self.mae), 0, 1)
        self.kappa = self.r2

        self.calc_bit_guidance()
        self.calc_layer_guidance()

    def calc_bit_guidance(self):
        # Cross usage
        imp_bit = self.fi[self.fi["feature"].str.startswith("bit")].copy()
        # init phase
        i_norm = imp_bit["importance"] / imp_bit["importance"].max()
        direction = self.bitgui["direction"].fillna(0.0)
        strength = self.bitgui["direction_strength"].fillna(0.0)
        logit0 = np.log(self.sparsity_rate / (1 - self.sparsity_rate))

        g = np.sign(direction) * np.tanh(strength) * i_norm.to_numpy()
        p_gui = 1 / (1 + np.exp(-(logit0 + self.alpha * self.G * g)))
        self.p_final_bits = list((1 - self.mrate) * p_gui + self.mrate * 0.5)

        # variance phase
        i_share = imp_bit["importance"] / imp_bit["importance"].sum()
        i_rel = i_share / i_share.max()

        self.b_star = (self.bitgui["direction"] > 0).astype(int).to_numpy()

        d_hat = self.bitgui["direction_strength"] / self.bitgui["direction_strength"].max()

        self.i_share_bit = i_share.to_numpy()

        s_bit = (i_rel.to_numpy() + d_hat.to_numpy()) / 2
        self.bit_mutation_prob = self.mrate * ((1 - self.kappa) + self.kappa * (1-s_bit))
    
    def calc_layer_guidance(self):
        # cross usage
        imp_layer = self.fi[self.fi["feature"].str.startswith("layer")].copy()
        imp_layer["pos"] = imp_layer["feature"].str.replace("layer ", "").astype(int)
        imp_layer = imp_layer.set_index("pos").sort_index()

        self.layergui.columns = [int(c.replace("layer ", "")) for c in self.layergui.columns]

        # init phase
        # i_norm = imp_layer["importance"] / imp_layer["importance"].max()
        self.prob_layers = {}
        imp_layer["I_norm"] = imp_layer["importance"] / imp_layer["importance"].max()
        actual_ops = self.layer_limit - 2
        # print(self.layergui)
        for j in range(1, actual_ops + 1):
            col = f"layer {j}"
            all_dist = self.population[col].value_counts(normalize=True)
            tendency = self.layergui[j]
            I_j = imp_layer.loc[imp_layer["feature"] == col, "I_norm"].values[0]
            ops = all_dist.index

            logits = np.array([np.log(all_dist[o] + self.eps) + self.beta * self.G * I_j * tendency.get(o, 0.0) for o in ops])
            p_gui = np.exp(logits - logits.max())
            p_gui /= p_gui.sum()

            K = len(ops)
            p_final_layers = (1 - self.mrate) * p_gui + self.mrate * (1/K)
            self.prob_layers[col] = list(zip(ops, p_final_layers))

        # variance phase
        i_share = imp_layer["importance"] / imp_layer["importance"].sum()
        i_rel = i_share / i_share.max()

        self.i_share_layer = i_share.to_numpy()

        spread = self.layergui.max() - self.layergui.min()
        G_hat = (spread / spread.max()).reindex(range(1, self.layer_limit - 2 + 1))

        s_layer = (i_rel.to_numpy() + G_hat.to_numpy()) / 2
        self.layer_mutation_prob = self.mrate * ((1- self.kappa) + self.kappa * (1 - s_layer))

    def get_bit_cell(self, content = None, pos = 0):
        if content in self.imp_bit.columns:
            return self.imp_bit.iloc[content, pos]
        elif content in self.bitgui.columns:
            return self.bitgui.iloc[content, pos]
        else:
            raise ValueError(f"{content} not recognized at position {pos}")
        
    def get_layer_cell(self, content = None, pos = 0):
        if content in self.imp_layer.columns:
            return self.imp_layer.iloc[content, pos]
        elif content in self.layergui.columns:
            return self.layergui.iloc[content, pos]
        else:
            raise ValueError(f"{content} not recognized at position {pos}")

    def get_b_star(self, i = 0): return self.b_star[i-1]
    def get_prob_bit(self, i): return self.bit_mutation_prob[i-1]
    def get_prob_layer(self, j): return self.layer_mutation_prob[j-1]
    def bit_weight(self, i): return self.kappa * self.i_share_bit[i-1]
    def layer_weight(self, j): return self.kappa * self.i_share_layer[j-1]
    def layer_eff(self, op, j): return self.layergui.loc[op, j]
    def layer_eff_range(self, j):
        col = self.layergui[j]
        return col.min(), col.max()

class Genome:
    def __init__(self, code, operations):
        self.code = code
        self.operations = operations
        self.fitness = None
        self.metrics = None
    # classmethod decorator used to imitate self.Genome
    @classmethod
    def random(cls, operations, layer_limit, edge_limit):
        ops = random_operation(layer_limit, operations)
        # dimension decrease to match nasbench's evaluation requirement
        dims = len(ops) - 1
        while True:
            code = random_string(dims)
            valid, _ = valid_architecture(string_to_matrix(code), edge_limit)
            if valid:
                break
        return cls(code, ops)

    @classmethod
    # def guided_genome(cls, layer_limit, edge_limit, bit_guidance, layer_guidance, 
    #                   feature_importance, global_var, pop, p0 = 0, mutation_rate = 0,
    #                   alpha = 4, beta = 4, eps = 1e-3):
    def guided_genome(cls, layer_limit, edge_limit, guidance : Guidance):
        # print("generating guided ops...")

        # # Calculating bit probabilities (P(bit = 1))
        # bit_imp = feature_importance[feature_importance["feature"].str.startswith("bit")].copy()
        # bit_imp["I_norm"] = bit_imp["importance"] / bit_imp["importance"].max()

        # bgui = bit_guidance.merge(bit_imp[["feature", "I_norm"]], on = "feature")
        # logit0 = np.log(p0 / (1 - p0))

        # direction = bgui["direction"].fillna(0.0)
        # strength = bgui["direction_strength"].fillna(0.0)

        # g = np.sign(direction) * np.tanh(strength) * bgui["I_norm"]
        # p_gui = 1 / (1 + np.exp(-(logit0 + alpha * global_var * g )))
        # p_final_bits = list((1 - mutation_rate) * p_gui + mutation_rate * 0.5)

        # # Calculating layer probabilities (P(layer(i,o)))
        # layer_imp = feature_importance[feature_importance["feature"].str.startswith("layer")].copy()
        # layer_imp["I_norm"] = layer_imp["importance"] / layer_imp["importance"].max()

        # prob_layers = {}
        # lcount = layer_limit - 2
        # for j in range(1, lcount + 1):
        #     col = f"layer {j}"
        #     all_dist = pop[col].value_counts(normalize = True)
        #     tendency = layer_guidance.set_index("operation")[col]
        #     I_j = layer_imp.loc[layer_imp["feature"] == col, "I_norm"].values[0]
        #     ops = all_dist.index

        #     logits = np.array([np.log(all_dist[o] + eps) + beta * global_var * I_j * tendency.get(o, 0.0) for o in ops])
        #     p_gui = np.exp(logits - logits.max())
        #     p_gui /= p_gui.sum()

        #     K = len(ops)
        #     p_final_layers = (1 - mutation_rate) * p_gui + mutation_rate * (1/K)
        #     prob_layers[col] = list(zip(ops, p_final_layers))

        ops = guided_layers(layer_limit, guidance.prob_layers)
        dims = len(ops) - 1
        # print("generating guided bits...")
        while True:
            code = guided_bits(dims, guidance.p_final_bits)
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
    def __init__(self, population_size, layer_nums, edge_nums, evaluator="nasbench", 
                 bit_guidance = None, layer_guidance = None, feature_importance = None, population = None,
                 preconfig = None, mutation_rate = 0, alpha = 4, beta = 4, eps = 1e-3):
        self.population_size = population_size
        self.members = []
        self.edge_limit = edge_nums
        self.layer_nums = layer_nums
        self.actual_layers = layer_nums - 2
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
            "best_val_acc" : 0,
            "best_genome" : None,
            "best_operation" : None,
        }

        self.preconfig = preconfig
        self.bit_guidance = bit_guidance
        self.layer_guidance = layer_guidance
        self.feature_importance = feature_importance
        self.population = population
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta = beta

        if preconfig is not None:
            self.guidance = Guidance(feature_importance, bit_guidance, layer_guidance,
                                 preconfig, population, alpha, beta, eps)
        else:
            self.guidance = None
        
        for i in range(layer_nums - 2):
            self.data[f"layer {i+1}"] = []
        
    def initialize(self, operations):
        '''
        Initiate parents and evaluate immediately
        '''
        if self.bit_guidance is None and self.layer_guidance is None:
            self.members = [Genome.random(operations, self.layer_nums, self.edge_limit) 
                            for _ in range(self.population_size)]

        else:
            if self.bit_guidance is None or not isinstance(self.bit_guidance, pd.DataFrame):
                raise ValueError("bit_guidance is missing or not the correct type")
            if self.layer_guidance is None or not isinstance(self.layer_guidance, pd.DataFrame):
                raise ValueError("layer_guidance is missing or is not the correct type")
            if self.feature_importance is None or not isinstance(self.feature_importance, pd.DataFrame):
                raise ValueError("feature_importance is missing or is not the correct type")

            # self.members = [Genome.guided_genome(self.layer_nums, self.edge_limit, self.bit_guidance,
            #                                     self.layer_guidance, self.feature_importance, pop = self.population, mutation_rate = self.mutation_rate,
            #                                     global_var = self.global_var, p0 = self.sparsity_rate, alpha = self.alpha, beta = self.beta) 
            #                 for _ in range(self.population_size)]
            self.members = [Genome.guided_genome(self.layer_nums, self.edge_limit, self.guidance) 
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
        
    def crossover(self, parent1 : Genome, parent2 : Genome):
        # Genome crossover

        # print("parent1: ", parent1)
        # print("parent2: ", parent2)

        invalids = Counter()
        # fails = []
        
        # Vanilla crossover
        if self.bit_guidance is None and self.layer_guidance is None:
            # while True:
            genome1 = parent1.code.split(sep="-")
            genome2 = parent2.code.split(sep="-")

            for attempt in range(MAX_PAIR_RETRIES):

                gen_cut = random.randint(1, len(genome1) - 1)
                
                child1_code = '-'.join(genome1[:gen_cut] + genome2[gen_cut:])
                child2_code = '-'.join(genome2[:gen_cut] + genome1[gen_cut:])
                
                valid1, reason1 = valid_architecture(string_to_matrix(child1_code), self.edge_limit)
                valid2, reason2 = valid_architecture(string_to_matrix(child2_code), self.edge_limit)

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
                # attempts += 1
            
            # print("total attempts: ", attempt)
        else:
            genome1 = flatten_code(parent1.code)
            genome2 = flatten_code(parent2.code)

            f1, f2 = parent1.fitness, parent2.fitness
            base_parents = f1 / (f1 + f2)
            # Exclude IO

            for attempt in range(MAX_PAIR_RETRIES):
                c1, c2 = [], []
                for i in range(1, len(genome1) + 1):
                    w = self.guidance.bit_weight(i)
                    b = self.guidance.get_b_star(i)
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

                valid1, reason1 = valid_architecture(string_to_matrix(child1_code), self.edge_limit)
                valid2, reason2 = valid_architecture(string_to_matrix(child2_code), self.edge_limit)
        
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
                # attempts += 1

        # Children are still invalid after too many crossovers
        if attempt == MAX_PAIR_RETRIES - 1 and not (child1_code and child2_code):
            return None, None

        # Operation crossover
        ops1 = parent1.operations[1:-1]
        ops2 = parent2.operations[1:-1]
        
        if self.bit_guidance is None and self.layer_guidance is None:
            ops_cut = random.randint(1, len(ops1) - 1)

            child1_ops = ops1[:ops_cut] + ops2[ops_cut:]       
            child2_ops = ops2[:ops_cut] + ops1[ops_cut:]
        else:
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
        
        child1 = Genome(child1_code, ["input"] + child1_ops + ["output"])
        child2 = Genome(child2_code, ["input"] + child2_ops + ["output"])
        
        return child1, child2
                
    def mutation(self, genome, avail_ops, chance=0.05):
        # Vanilla mutation
        if self.bit_guidance is None and self.layer_guidance is None:
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
                valid, reason = valid_architecture(string_to_matrix(new_code), self.edge_limit)

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
                    
            # print(f"before mutation: {genome.operations}")
            # print(f"after mutation: {pre_ops}")
            
            return Genome(new_code, pre_ops)
        else:
            # Guided mutation
            pre_code = flatten_code(genome.code)
            new_code = None
            invalids = Counter()
            for attempt in range(MAX_PAIR_RETRIES):
            # while True:
                for p in range(len(pre_code)):
                    if pre_code[p] in ('0', '1') and random.random() < self.guidance.get_prob_bit(p + 1):
                        pre_code[p] = '0' if pre_code[p] == '1' else '1'
                new_code = rebuild_code(pre_code, self.actual_layers + 1)
                valid, reason = valid_architecture(string_to_matrix(new_code), self.edge_limit)
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
                if random.random() < self.guidance.get_prob_layer(o):
                    choices = [op for op in avail_ops if op != pre_ops[o]]
                    pre_ops[o] = random.choice(choices)
                
            # print(f"before mutation: {genome.operations}")
            # print(f"after mutation: {pre_ops}")
            return Genome(new_code, pre_ops)
    
    def evolve(self, operations, generation,
               elite_size = 2, selector = "tournament", survivors = 1,
               candidates_per_round = 4):

        # Config should be added once to save a bit of time
        self.config["generations"] = generation
        if generation == 1:
            self.config["mutation_rate"] = self.mutation_rate
            self.config["elite_size"] = elite_size
            self.config["selection"] = selector

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

            child1 = self.mutation(child1, operations, self.mutation_rate)
            child2 = self.mutation(child2, operations, self.mutation_rate)

            # print(child1)
            # print(child2)
            
            for child in (child1, child2):
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
        best = max(self.members, key = lambda g: g.fitness)

        self.config["best_val_acc"] = best.fitness
        self.config["best_genome"] = best.code
        self.config["best_operation"] = best.operations

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
        allowed_methods = ["nasbench101", "nasbench201"]
        if method not in allowed_methods:
            raise ValueError(f"{method} not allowed")
        
        self.method = method
        
        if method == "nasbench101":
            self.nasbench = api.NASBench('./datas/nasbench_full.tfrecord')
        elif method == "nasbench201":
            self.nasbench = API("./datas/NAS-Bench-201-v1_1-096897.pth")
    
    def evaluate(self, genome : Genome):
        if self.method == "nasbench101":
            # print("evaluating genome with code ", genome.code)
            model_spec = api.ModelSpec(matrix=string_to_matrix(genome.code), ops = genome.operations)
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
        if self.method == "nasbench201":
            return "breh"
        
    def nb201parser(inp_layers : list):
        # exp list : [1,3,2,3,1]
        op_dict = {0 : "none", 1 : "skip_connect", 2 : "nor_conv_1x1", 3 : "nor_conv_3x3", 4 : "avg_pool_3x3"}
        idx = 0
        query = []
        for layer in range(3):
            qp = []
            for i in range(layer):
                op = op_dict[inp_layers[idx]]
                qp.append(f"{op}~{i}")
                idx += 1
            query.append(qp)
                
        return query

# HELPER FUNCTIONS

def flatten_code(code):
    return list(code.replace("-", ""))
 
def rebuild_code(flat_bits, layers):
    segs, pos = [], 0
    # SEG_LENS = []
    # for l in range(1, layers + 1):
    #     SEG_LENS.append(l)
    # for L in SEG_LENS:
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

def guided_bits(num_nodes, probs : list):
    """created bits should be 1 level lower than expected"""
    genome = []
    # bit_idx = 0

    # for i in range(1, num_nodes + 1):
    #     gene = []
    #     for _ in range(i):
    #         p_one = probs[bit_idx]
    #         p_zero = 1 - p_one
    #         bit = np.random.choice(["0", "1"], p = [p_zero, p_one])
    #         gene.append(bit)
    #         bit_idx += 1
    #     genome.append(''.join(gene))

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

# Operation methods (Vanilla & Guided)

def vanilla_run(operations, population_size, 
        edge_limit = 9, layer_limit = 7, generations = 100, evaluator : Evaluator = None,
        selector = "tournament", elite_size = 2, survivors = 1,
        candidates_per_round = 4, mutation_rate = 0.05):
    
    # Conditions check:
    if not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("Population must be integer and larger than 0") 
    
    if not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("Population size must be an integer larger than 0") 
    
    if not isinstance(edge_limit, int) or edge_limit <= 0:
        raise ValueError("Edge limit must be an integer larger than 0") 
    
    if not isinstance(layer_limit, int) or edge_limit <= 0:
        raise ValueError("Layer limit must be an integer larger than 0") 
    
    if not isinstance(generations, int) or generations <= 0:
        raise ValueError("Generation must be an integer larger than 0") 
     
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
    # evaluator = Evaluator(eval_method)

    pop = Population(population_size, layer_limit, edge_limit, evaluator, mutation_rate=mutation_rate)
    pop.initialize(operations)
    print("Population created, evaluating...")
    
    for generation in range(1, generations + 1):
        pop.evolve(operations, generation, elite_size, selector, 
                    survivors, candidates_per_round)
        
        best = max(pop.members, key = lambda g: g.fitness)
        
        print(f"Generation {generation} - Validation accuracy: {best.fitness}")
        
    # df = pd.DataFrame(pop.data)
    # root = Path("./records")
    
    # exp_dir = root / f"va_records"
    # exp_dir.mkdir(parents=True)
    
    # df.to_csv(exp_dir / "population.csv", index = False)
    # with open(exp_dir / "config.json", "w") as file:
    #     json.dump(pop.config, file, indent=4)
        
    # print(f"log created at {exp_dir}")

    return {"population_data" : pd.DataFrame(pop.data), "configs" : pop.config}

def guided_run(operations, population_size, 
        edge_limit = 9, layer_limit = 7, generations = 100, evaluator : Evaluator = None,
        selector = "tournament", elite_size = 2, survivors = 1,
        candidates_per_round = 4, mutation_rate = 0.05, bit_guidance = None, 
        layer_guidance = None, feature_importance = None, config = None, pre_pop = None, alpha = 4, beta = 4) -> pd.DataFrame:
    
    # Conditions check:
    if not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("Population must be integer and larger than 0") 
    
    if not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("Population size must be an integer larger than 0") 
    
    if not isinstance(edge_limit, int) or edge_limit <= 0:
        raise ValueError("Edge limit must be an integer larger than 0") 
    
    if not isinstance(layer_limit, int) or edge_limit <= 0:
        raise ValueError("Layer limit must be an integer larger than 0") 
    
    if not isinstance(generations, int) or generations <= 0:
        raise ValueError("Generation must be an integer larger than 0") 
     
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
    # evaluator = Evaluator(eval_method)
    # print("initiating population...")
    pop = Population(population_size, layer_limit, edge_limit, evaluator, 
                     bit_guidance, layer_guidance, feature_importance,
                    pre_pop, config, mutation_rate, alpha, beta)
    pop.initialize(operations)
    # print("Population created, evaluating...")
    
    for generation in range(1, generations + 1):
        pop.evolve(operations, generation, elite_size, selector, 
                    survivors, candidates_per_round)
        
        # for col in pop.data:
        #     if col.startswith("layer"):
        #         print(col)
        #         print(pop.data[f"{col}"])
        
        best = max(pop.members, key = lambda g: g.fitness)
        
        print(f"Generation {generation} - Validation accuracy: {best.fitness}")
        
    # df = pd.DataFrame(pop.data)
    # root = Path("./guided_records")
    
    # i = 1
    # while (root / f"exp{i:03d}").exists():
    #     i += 1
    
    # exp_dir = root / f"exp{i:03d}"
    # exp_dir.mkdir(parents=True)
    
    # df.to_csv(exp_dir / "population.csv", index = False)
    # with open(exp_dir / "config.json", "w") as file:
    #     json.dump(pop.config, file, indent=4)
        
    # print(f"log created at {exp_dir}")

    return {"population_data" : pd.DataFrame(pop.data), "configs" : pop.config}


def get_stats(mode):
    allowed_modes = ["nasbench101", "nasbench201"]
    if mode in allowed_modes:
        if mode == "nasbench101":
            edge_limit = 9
            layer_limit = 7

            conv1x1 = 'conv1x1-bn-relu'
            conv3x3 = 'conv3x3-bn-relu'
            maxpool3x3 = 'maxpool3x3'

            operations = [conv1x1, conv3x3, maxpool3x3]

            return {'edge_limit': edge_limit,
                    'layer_limit': layer_limit,
                    'operations': operations,
                    'evaluator' : Evaluator("nasbench101")
                    }
        elif mode == "nasbench201":
            edge_limit = 6
            layer_limit = 6
            operations = ["zeroize", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    else:
        raise ValueError("mode not existed")

# Testing ground
if __name__ == "__main__":
    start = time()
    conv1x1 = 'conv1x1-bn-relu'
    conv3x3 = 'conv3x3-bn-relu'
    maxpool3x3 = 'maxpool3x3'

    operations = [conv1x1, conv3x3, maxpool3x3]

    # fi_report_path = Path("./fi_reports/fi1")
    fi_report_path = Path("./fi_reports/fi2")

    fi = pd.read_csv(fi_report_path / "fi.csv")
    bit_gui = pd.read_csv(fi_report_path / "bit_guidance.csv")
    layer_gui = pd.read_csv(fi_report_path / "layer_guidance.csv")
    try:
        # run(10, operations, population_size=10000, generations=100)
        guided_run(operations, population_size=100000, generations=100, bit_guidance=bit_gui, layer_guidance=layer_gui)
    except ValueError as e:
        print(e)
    print("Operation time: ", time() - start)

    # print(random_string(7))

    # test_list = [random_string(7) for _ in range(100)]
    # for t in test_list:
    #     print(t, valid_architecture(string_to_matrix(t)))