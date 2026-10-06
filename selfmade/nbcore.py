import numpy as np
import random
import pandas as pd
import selfmade.abstract as abstract

from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

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
                "validation_accuracy": [],
                "test_accuracy": [], # test accuracy is for record. not for GA
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

        self.best_so_far = 0.0
        self.best_history = []

        self.history = {
            "generation": [],
            "best_val": [],
            "test_of_best_val": [],
        }
        self.best_so_far_history = []

        self.current_generation = 0

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

    def evolve(self):
        """
        Evolution ONCE, including selection, crossover and mutation
        """
        self.current_generation += 1
        gen = self.current_generation
        self.config["generations"] = gen
        next_generation = self.elitism(self.elite_size)

        while len(next_generation) < self.population_size:
            while True:
                parent1 = self.selection(self.selector, self.survivors, self.candidates_per_round)[0]
                parent2 = self.selection(self.selector, self.survivors, self.candidates_per_round)[0]

                if parent1 is not parent2:
                    break

            child1, child2 = self.crossover(parent1, parent2)

            if child1 is None and child2 is None:
                continue

            child1_mut = self.mutation(child1)
            child2_mut = self.mutation(child2)

            for child in (child1_mut, child2_mut):
                self.single_evaluation(child)

                if len(next_generation) < self.population_size:
                    next_generation.append(child)

        self.members = next_generation

        for genome in self.members:
            if genome.fitness is None:
                self.single_evaluation(genome)

        best = max(self.members, key=lambda g: g.fitness)
        print(f"Generation {gen} - Best performance: {best.fitness}")

        self.record(gen)

    def guidance_switch(self, switch, gui = None):
        if switch == False:
            self.guidance = None
        else:
            if self.guidance is None:
                self.guidance = gui

    def evolve_cfg(self, guidance, gui_cross = True, gui_mut = True):
        """
        Config for ablation studies
        """
        self.current_generation += 1
        gen = self.current_generation
        self.config["generations"] = gen
        next_generation = self.elitism(self.elite_size)

        while len(next_generation) < self.population_size:
            while True:
                parent1 = self.selection(self.selector, self.survivors, self.candidates_per_round)[0]
                parent2 = self.selection(self.selector, self.survivors, self.candidates_per_round)[0]

                if parent1 is not parent2:
                    break

            self.guidance_switch(gui_cross, guidance.copy())
            child1, child2 = self.crossover(parent1, parent2)

            if child1 is None and child2 is None:
                continue

            self.guidance_switch(gui_mut, guidance.copy())
            child1_mut = self.mutation(child1)
            child2_mut = self.mutation(child2)

            for child in (child1_mut, child2_mut):
                self.single_evaluation(child)

                if len(next_generation) < self.population_size:
                    next_generation.append(child)

        self.members = next_generation

        for genome in self.members:
            if genome.fitness is None:
                self.single_evaluation(genome)

        best = max(self.members, key=lambda g: g.fitness)
        print(f"Generation {gen} - Best performance: {best.fitness}")

        self.record(gen)

    def record(self, generation):
        print(f"Recording generation {generation}")

        best_val = max(g.fitness for g in self.members)

        best_candidates = [
            g for g in self.members
            if abs(g.fitness - best_val) <= 1e-12
        ]

        best = best_candidates[0]

        self.config["best_val_acc"] = best_val
        self.config["best_test_acc"] = best.metrics["test_accuracy"]
        self.config["best_candidate"] = best.representation

        for m in self.members:
            self.data["generation"].append(generation)
            self.data["representation"].append(m.representation)
            self.data["fitness"].append(m.fitness)
            self.data["validation_accuracy"].append(
                m.metrics["validation_accuracy"]
            )
            self.data["test_accuracy"].append(
                m.metrics["test_accuracy"]
            )

        self.best_so_far = max(self.best_so_far, best_val)

        tied_tests = [g.metrics["test_accuracy"] for g in best_candidates]

        self.best_history.append({
            "generation": generation,
            "best_val": best_val,
            "best_so_far": self.best_so_far,
            "best_test": best.metrics["test_accuracy"],
            "num_tied": len(best_candidates),
            "tied_test_min": min(tied_tests),
            "tied_test_max": max(tied_tests),
            "tied_test_mean": np.mean(tied_tests),
        })

        print("Finished recording")

    def evolve_loop(self, total_gen = 1, guidance = None):
        if guidance is not None:
            self.guidance = guidance
        for _ in range(1, total_gen + 1):
            self.evolve()

    def true_test(self, genome):
        test_results = self.evaluator.evaluate(genome)
        return test_results

class FeatureImportance(abstract.FeatureImportance):
    """Extract values from previous runs"""
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

        self.raw_data = pd.DataFrame(rows)
        self.data = self.raw_data.copy()
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
                "std": {0: std_0, 1: std_1},
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
        eff_spread = max(effects.values()) - min(effects.values())

        threshold = y.quantile(0.8)
        good_mask = y >= threshold

        all_dist = x.value_counts(normalize=True)

        good_dist = x[good_mask].value_counts(normalize=True)

        tendency = good_dist.subtract(all_dist, fill_value=0.0)

        tendency = {
            state: tendency.get(state, 0.0)
            for state in states
        }

        tendency_spread = (max(tendency.values()) - min(tendency.values()))

        return {
            "type": "categorical",
            "states": states,
            "n_states": len(states),
            "means": means,
            "effects": effects,
            "preferred": preferred,
            "effect_spread": eff_spread,
            "tendency": {state: tendency.get(state, 0.0) for state in states},
            "tendency_spread": tendency_spread,
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
                "preferred": result["preferred"],
                "std_0": result["std"][0],
                "std_1": result["std"][1],
                "strength": result["strength"]
            })

        direction_df = pd.DataFrame(direction_rows)

        tendency_rows = []
        for feature, result in tendencies.items():
            for state in result["states"]:
                tendency_rows.append({
                    "feature": feature,
                    "state": state,
                    "mean_fitness": result["means"][state],
                    "effect": result["effects"][state],
                    "effect_spread": result["effect_spread"],
                    "preferred": state == result["preferred"],
                    "tendency": result["tendency"][state],
                    "tendency_spread": result["tendency_spread"]
                })

        tendency_df = pd.DataFrame(tendency_rows)

        return {
            "importance": importance,
            "direction": direction_df,
            "tendency": tendency_df,
            "feature_types": feature_types,
            "population": self.raw_data.copy()
        }

class Guidance:
    """Shape the population to get better results"""
    
    def __init__(self, search_space, importance_data, config, alpha = 4, beta = 4, eps = 1e-3):
        self.space = search_space
        self.fi = importance_data["importance"]
        self.direction = importance_data["direction"]
        self.tendency = importance_data["tendency"]
        self.feature_types = importance_data["feature_types"]
        self.population = importance_data["population"]
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

        return groups

    def calculate_guidance(self):
        self.groups = self.sortNsplit()
        self.guides = {}

        for (family, feature_type) in self.groups:
            if feature_type == "binary":
                res = self.calc_bit_guidance(self.groups[(family, feature_type)], family)
            elif feature_type == "categorical":
                res = self.calc_cat_guidance(self.groups[(family, feature_type)], family)
            else:
                raise ValueError(f"{feature_type} not existed for guidance calculation")
            self.guides[(family, feature_type)] = res

        return self.guides
            
    def calc_bit_guidance(self, data, family):
        # Cross usage
        # imp_bit = self.fi[self.fi["feature"].str.startswith("bit")].copy()
        imp_bit = data.copy()
        # init phase
        i_norm = imp_bit["importance"] / imp_bit["importance"].max()
        # direction = self.bitgui["direction"].fillna(0.0)
        # strength = self.bitgui["direction_strength"].fillna(0.0)
        direction_df = (
            self.direction
            .set_index("feature")
            .reindex(imp_bit["feature"])
        )

        direction = direction_df["direction"].fillna(0.0)
        strength = direction_df["strength"].fillna(0.0)
        logit0 = np.log(self.sparsity_rate / (1 - self.sparsity_rate))

        g = np.sign(direction) * np.tanh(strength) * i_norm.to_numpy()
        p_gui = 1 / (1 + np.exp(-(logit0 + self.alpha * self.G * g)))
        
        p_final = list((1 - self.mrate) * p_gui + self.mrate * 0.5)

        # variance phase
        i_share = imp_bit["importance"] / imp_bit["importance"].sum()
        i_rel = i_share / i_share.max()

        # self.b_star = (self.bitgui["direction"] > 0).astype(int).to_numpy()
        b_star = (direction > 0).astype(int).to_numpy()

        # d_hat = self.bitgui["direction_strength"] / self.bitgui["direction_strength"].max()
        d_hat = strength / strength.max()

        # i_share_bit = i_share.to_numpy()

        s_bit = (i_rel.to_numpy() + d_hat.to_numpy()) / 2
        mutation_prob = self.mrate * ((1 - self.kappa) + self.kappa * (1-s_bit))

        return {
            "family": family,
            "type": "binary",
            "features": imp_bit["feature"].tolist(),
            "init_prob": p_final,
            "preferred": b_star,
            "mutation_prob": mutation_prob,
            "importance_share": i_share.to_numpy()
        }
        
    def calc_cat_guidance(self, data, family):
        # cross usage
        data = data.copy()
        features = data["feature"].to_list()

        # init phase
        self.prob_layers = {}
        max_imp = data["importance"].max()
        if max_imp > 0:
            data["I_norm"] = data["importance"] / max_imp
        else:
            data["I_norm"] = 0.0

        tendency_df = self.tendency[self.tendency["feature"].isin(features)].copy()
        prob_init = {}
        spreads = {}

        
        for _, row in data.iterrows():

            feature = row["feature"]
            I_j = row["I_norm"]

            all_dist = (self.population[feature].value_counts(normalize=True))
            tdf = tendency_df[tendency_df["feature"] == feature]
            tendency_map = (tdf.set_index("state")["tendency"].to_dict())

            states = list(all_dist.index)

            logits = np.array([np.log(all_dist[s] + self.eps) + self.beta * self.G * I_j * tendency_map.get(s, 0.0) for s in states])
            p_gui = np.exp(logits - logits.max())
            p_gui /= p_gui.sum()

            K = len(states)
            p_final = (1 - self.mrate) * p_gui + self.mrate * (1/K)
            prob_init[feature] = list(zip(states, p_final))

            if len(tdf) > 0:
                spreads[feature] = (tdf["tendency"].max() - tdf["tendency"].min())
            else:
                spreads[feature] = 0.0

        # variance phase
        i_sum = data["importance"].sum()
        if i_sum > 0:
            i_share = data["importance"] / i_sum
        else:
            i_share = pd.Series(0.0, index = data.index)
        max_share = i_share.max()
        if max_share > 0:
            i_rel = i_share / i_share.max()
        else:
            i_rel = pd.Series(0.0, index = data.index)

        spread = pd.Series(spreads)
        max_spread = spread.max()
        if max_spread > 0:
            G_hat = (spread / max_spread)
        else:
            G_hat = pd.Series(0.0, index = spread.index)

        G_hat = G_hat.reindex(data["feature"]).fillna(0.0)
        i_rel = pd.Series(i_rel.to_numpy(), index = data["feature"])

        s_cat = (i_rel + G_hat) / 2
        mutation_prob = self.mrate * ((1- self.kappa) + self.kappa * (1 - s_cat))

        return {
            "family": family,
            "type": "categorical",
            "features": features,
            "init_prob": prob_init,
            "importance_share": i_share.to_numpy(),
            "importance_relative": i_rel.to_numpy(),
            "mutation_prob": mutation_prob.to_numpy(),
            "tendency_spread": G_hat.to_numpy()
        }
        
    def get_cell(self, key, field, pos="all"):
        group = self.guides[key]

        if field not in group:
            raise KeyError(
                f"{field} not found in {key}"
            )

        value = group[field]

        if pos == "all":
            return value

        return value[pos]


    def get_feature(self, key, field, feature):
        group = self.guides[key]
        value = group[field]

        if isinstance(value, dict):
            return value[feature]

        idx = group["features"].index(feature)
        return value[idx]


    def get_weight(self, key, feature):
        share = self.get_feature(
            key,
            "importance_share",
            feature
        )

        return self.kappa * share

    def get_b_star(self, key, feature):
        return self.get_feature(key, "preferred", feature)

    def get_tendency(self, feature, state):
        rows = self.tendency[
            (self.tendency["feature"] == feature) &
            (self.tendency["state"] == state)
        ]

        if rows.empty:
            return 0.0

        return rows.iloc[0]["tendency"]


    def get_tendency_range(self, feature):
        rows = self.tendency[
            self.tendency["feature"] == feature
        ]

        if rows.empty:
            return 0.0, 0.0

        return (
            rows["tendency"].min(),
            rows["tendency"].max()
        )

def full_vanilla(pop_size, evo_loop, space, evaluator) -> Population:
    """Full Vanilla, no Guidance"""
    fv_pop = Population(pop_size, space, evaluator)
    fv_pop.initialize()
    fv_pop.evolve_loop(total_gen=evo_loop)
    return fv_pop

def full_guided(pop_size: int, evo_loop: int, space, evaluator, p_split = 0.5,
                 gui_init = True, gui_cross = True, gui_mut = True) -> Population:
    """Vanilla at beginning, then guided at init, crossover and mutation"""
    if p_split <= 0 or p_split >= 1:
        raise ValueError(f"p_split must be in range (0,1), got {p_split}")
    bp = int(evo_loop * p_split)
    if bp < 1:
        bp = 1
    elif (evo_loop - bp) == 0:
        bp = evo_loop - 1
    van_res = full_vanilla(pop_size, bp, space, evaluator)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
    fi = FeatureImportance(space)
    imp = fi.pipeline(van_res.data, rf)

    gui = Guidance(space, imp, van_res.config)
    gui.sortNsplit()
    gui.calculate_guidance()
