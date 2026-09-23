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

from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from time import time

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

    def random_genome(self):
        ops = random_operation(self.layer_limit, self.operations)
        dims = len(ops) - 1

        while True:
            code = random_string(dims)
            valid, _ = valid_architecture(string_to_matrix(code), self.edge_limit)

            if valid:
                return GenomeNB101(code, ops)

    def guided_genome(self, guidance):
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
        for attempt in range(MAX_PAIR_RETRIES):
        # while True:
            pre_code = original.copy()
            for p in range(len(pre_code)):
                if pre_code[p] in ('0', '1') and random.random() < guidance.get_prob_bit(p + 1):
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
                 guidance=None, mutation_rate=0.05, elite_size = 2, 
                 selector = "tournament", survivors = 1, candidates_per_round = 4):
        self.population_size = population_size
        self.space = search_space
        self.evaluator = evaluator
        self.guidance = guidance
        self.mutation_rate = mutation_rate
        self.members = []
        self.elite_size = elite_size
        self.selector = selector
        self.survivors = survivors
        self.candidates_per_round = candidates_per_round
        
        self.data = {
                "generation" : [],
                "representation" : [],
                "fitness" : [],
            }
            
        self.config = {
            "population_size" : self.population_size,
            "generations" : None,
            "mutation_rate" : self.mutation_rate,
            "elite_size" : self.elite_size,
            "edge_limit" : self.space.edge_limit,
            "layer_limit" : self.space.layer_limit,
            "selection" : self.selector,
            "best_val_acc" : 0,
            "best_candidate" : None,
        }

    def initialize(self):
        if self.guidance is None:
            self.members = [self.space.random_genome() for _ in range(self.population_size)]
        else:
            self.members = [self.space.guided_genome(self.guidance) for _ in range(self.population_size)]

        self.evaluation()
        best = max(self.members, key=lambda g: g.fitness)
        
        print(f"Best performance: {best.fitness}")
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

    def mutation(self, genome):
        # Vanilla mutation
        if self.guidance is None:
            return self.space.vanilla_mutation(genome, self.space.operations, self.mutation_rate)
        else:
            return self.space.guided_mutation(genome, self.space.operations, self.guidance)

    def evolve(self, generation = 1):
        """
        Evolution loop, including selection, crossover and mutation
        """
        self.config["generations"] = generation

        for gen in range(1, generation + 1):
            next_generation = self.elitism(self.elite_size)
            print(f"Generation {gen}")

            while len(next_generation) < self.population_size:
                while True:
                    parent1 = self.selection(self.selector, self.survivors, self.candidates_per_round)[0]
                    parent2 = self.selection(self.selector, self.survivors, self.candidates_per_round)[0]

                    if parent1 is not parent2:
                        break

                # print("crossover attempt")
                child1, child2 = self.crossover(parent1, parent2)

                if child1 is None and child2 is None:
                    continue

                # print("mutation attempt")
                child1_mut = self.mutation(child1)
                child2_mut = self.mutation(child2)

                # print(child1)
                # print(child2)

                for child in (child1_mut, child2_mut):
                    self.single_evaluation(child)

                    if len(next_generation) < self.population_size:
                        next_generation.append(child)

            self.members = next_generation

            for genome in self.members:
                if genome.fitness is None:
                    self.single_evaluation(genome)

            best = max(self.members, key=lambda g: g.fitness)

            print(f"Best performance: {best.fitness}")

            self.record(gen)

    def record(self, generation):
        """Record all candidates into pop.data"""
        print(f"recording generation {generation}")
        best = max(self.members, key = lambda g: g.fitness)

        self.config["best_val_acc"] = best.fitness
        self.config["best_candidate"] = best.representation

        for m in self.members:

            self.data["generation"].append(generation)
            self.data["representation"].append(m.representation)
            self.data["fitness"].append(m.fitness)

        print("finished recording")

class FeatureImportance(abstract.FeatureImportance):
    def __init__(self, search_space):
        self.space = search_space
        self.data = None

    def extract_data(self, population_data):
        """
        population_data must contain genome representation and fitness
        """
        rows = []
        for rep, fitness in zip(population_data["representation"], population_data["fitness"]):
            row = {"fitness" : fitness}
            self._extract_rep(rep, row)
            rows.append(row)

        self.data = pd.DataFrame(rows)
        return self.data

    def _extract_rep(self, value, row, prefix = ""):
        # dict
        if isinstance(value, dict):
            for name, part in value.items():
                new_pref = (f"{prefix}_{name}" if prefix else name)
                self._extract_rep(part, row, new_pref)
        # list / tuple
        elif isinstance(value, (list, tuple)):
            # empty
            if not value:
                return
            # numeric
            if all(self._is_num(x) for x in value):
                for i, x in enumerate(value, start = 1):
                    row[f"{prefix}{i}"] = x
            
            # string / cat
            else:
                for i, x in enumerate(value, start = 1):
                    row[f"{prefix}{i}"] = x
        # string
        elif isinstance(value, str):
            # binary rep
            if self._is_bin_str(value):
                bits = [char for char in value if char in ("0", "1")]
                for i, bit in enumerate(bits, start = 1):
                    row[f"{prefix}_bit{i}"] = int(bit)
            # cat scalar
            else:
                row[prefix] = value
        
        # num scalar
        elif self._is_num(value):
            row[prefix] = value
        # others
        else:
            row[prefix] = value
        
    @staticmethod
    def _is_num(value):
        return isinstance(value, (int, float, np.integer, np.floating))
    
    @staticmethod
    def _is_bin_str(value):
        chars = [char for char in value if not char.isspace()]
        return (len(chars) > 0 and all(char in "01-" for char in chars) and any(char in "01" for char in chars))

    def encode(self):
        encoders = {}
        df = self.data.copy()
        y = self.data["fitness"].copy()
        X = df.drop(columns=["fitness"]).copy()

        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                continue
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            encoders[c] = le
        self.data = pd.concat([X,y], axis = 1)
        if encoders:
            self.encoders = encoders
        return X, y, self.encoders

    def get_importance(self, model):
        # Encode
        _, _, _ = self.encode()

        df = self.data.copy()

        # Remove constants
        constant_cols = [
            c for c in df.columns
            if c != "fitness" and df[c].nunique(dropna=True) <= 1
        ]

        df = df.drop(columns=constant_cols)

        y = df["fitness"]
        X = df.drop(columns = ["fitness"]).copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = None
        
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        }).sort_values("importance", ascending=False).reset_index(drop = True)
        
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10,
            random_state=42, n_jobs=1)
        
        corr, p_val = spearmanr(importance, perm.importances_mean)

        perm_df = pd.DataFrame({
                "feature": X.columns,
                "importance": perm.importances_mean,
                "std": perm.importances_std
                }).sort_values("importance", ascending=False).reset_index(drop=True)
                    
        self.r2 = r2
        self.mae = mae
        self.imp_df = importance_df
        self.perm = perm_df

        return {"df_imp" : importance_df, "perm_imp" : perm_df, "r2": r2, "mae" : mae, "corr": corr, "p_value": p_val}
        
    def calculate_effect(self, feature, feature_type):
        x = self.data[feature]
        y = self.data["fitness"]
        if feature_type == "binary":
            return self._bin_eff(x, y)
        elif feature_type == "categorical":
            return self._cat_eff(x, y)
        else:
            raise ValueError(f"{feature_type} not existed in {feature}")
        
    def get_feature_types(self):
        feature_types = {}
        for feature in self.data.columns:
            if feature == "fitness":
                continue
            feature_types[feature] = self.label_check(feature)
        return feature_types
        
    def _bin_eff(self, x, y):
        states = sorted(x.dropna().unique())

        if states != [0,1]:
            raise ValueError(f"Binary feature must contain 0 or 1, got {states}")

        y0 = y[x == 0]
        y1 = y[x == 1]

        mean_0 = y0.mean()
        mean_1 = y1.mean()

        delta = mean_1 - mean_0
        
        std_1 = y1.std()
        std_0 = y0.std()
        
        pooled_std = np.sqrt((std_1**2 + std_0**2) / 2)

        if pooled_std == 0 or np.isnan(pooled_std):
            strength = 0
        else:
            strength = abs(delta) / pooled_std
            
        return {"type" : "binary",
                "states" : states,
                "n_states" : len(states),
                "means" : {0 : mean_0, 1 : mean_1},
                "effect" : delta,
                "preferred" : 1 if delta > 0 else 0,
                "std_1": std_1,
                "std_0": std_0,
                "strength": strength
                }

    def _cat_eff(self, x, y):
        states = list(x.dropna().unique())
        overall_mean = y.mean()

        means = {}

        for state in states:
            means[state] = y[x == state].mean()

        effects = {
            state: mean - overall_mean
            for state, mean in means.items()
        }

        preferred = max(effects, key=effects.get)
        spread = max(effects.values()) - min(effects.values())

        return {
            "type": "categorical",
            "states": states,
            "n_states": len(states),
            "means": means,
            "effects": effects,
            "preferred": preferred,
            "spread": spread
        }

    def label_check(self, feature):
        state = self.data[feature].dropna().unique()
        if set(state).issubset({0, 1}):
            return "binary"
        if len(state) <= 1:
            return "constant"
        return "categorical"

    def calculate_effects(self):
        effects = {}
        for feature in self.data.columns:
            if feature == "fitness":
                continue
            feature_type = self.label_check(feature)
            if feature_type == "constant":
                continue
            effects[feature] = self.calculate_effect(feature, feature_type)
        return effects

    def make_effect_reports(self):

        effects = self.calculate_effects()

        directions = {}
        tendencies = {}

        for feature, result in effects.items():

            if result["type"] == "binary":
                directions[feature] = result

            elif result["type"] == "categorical":
                tendencies[feature] = result

        return directions, tendencies

    def pipeline(self, pop_data, model):

        # 1. Extract
        self.extract_data(pop_data)

        # 2. Get feature types and calculate effects
        feature_types = self.get_feature_types()
        directions, tendencies = self.make_effect_reports()

        # 3. Model-based importance
        self.encode()
        importance = self.get_importance(model)

        # 4. Convert reports to DataFrames
        direction_rows = []
        for feature, result in directions.items():
            direction_rows.append({
                "feature": feature,
                "mean_0": result["means"][0],
                "mean_1": result["means"][1],
                "direction": result["effect"],
                "preferred": result["preferred"]
            })

        direction_df = pd.DataFrame(direction_rows)

        tendency_rows = []
        for feature, result in tendencies.items():
            for state, effect in result["effects"].items():
                tendency_rows.append({
                    "feature": feature,
                    "state": state,
                    "mean_fitness": result["means"][state],
                    "effect": effect,
                    "preferred": state == result["preferred"],
                    "spread": result["spread"]
                })

        tendency_df = pd.DataFrame(tendency_rows)

        return {
            "importance": importance,
            "direction": direction_df,
            "tendency": tendency_df,
            "feature_types": feature_types,
        }

class Guidance:
    
    def __init__(self, search_space, importance_data, config, alpha = 4, beta = 4, eps = 1e-3):
        self.space = search_space
        self.fi = importance_data["importance"]
        self.direction = importance_data["direction"]
        self.tendency = importance_data["tendency"]
        self.feature_types = importance_data["feature_types"]
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.r2 = self.fi["r2"]
        self.mae = self.fi["mae"]
        self.corr = self.fi["corr"]
        self.p_val = self.fi["p_value"]
        self.layer_limit = config["layer_limit"]
        self.edge_limit = config["edge_limit"]
        self.mrate = config["mutation_rate"]
        
        sbit = self.layer_limit * (self.layer_limit - 1) // 2
        if self.edge_limit is not None and 0 <= self.edge_limit <= sbit:
            self.sparsity_rate = np.clip(self.edge_limit / sbit, 1e-6, 1 - 1e-6)
        else:
            self.sparsity_rate = 0.5

        self.G = np.clip(self.r2 * abs(self.corr) * (1 - self.p_val) / (1 + self.mae), 0, 1)
        self.kappa = np.clip(self.r2, 0.0, 1.0)
        
    def sortNsplit(self):
        df = self.fi["df_imp"].copy()

        # Extract family and numerical position
        extracted = df["feature"].str.extract(r"^([A-Za-z_]+)\s*(\d+)$")

        df["family"] = extracted[0]
        df["position"] = pd.to_numeric(extracted[1], errors="coerce")

        # Get type from FeatureImportance
        df["type"] = df["feature"].map(self.feature_types)

        # Check that every feature got a type
        if df["type"].isna().any():
            unknown = df.loc[df["type"].isna(), "feature"].tolist()
            raise ValueError(f"No feature type found for: {unknown}")

        # Group by family + type
        groups = {}

        for (family, feature_type), group in df.groupby(["family", "type"], sort=True):
            group = (group.sort_values("position",ascending=True).reset_index(drop=True))
            groups[(family, feature_type)] = group
            
        self.groups = groups

        return groups
        
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

    # records = Path('./datas/nasbench_full.tfrecord')
    records = './datas/nasbench_full.tfrecord'
    
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
