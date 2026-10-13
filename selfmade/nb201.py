from nb201.nas_201_api import NASBench201API as API
from pathlib import Path
import numpy as np
import random
import pandas as pd
from collections import Counter
import json
import selfmade.abstract as abstract
from selfmade.nbcore import Population, Guidance, FeatureImportance


class GenomeNB201(abstract.Genome):
    def __init__(self, code, operations):
        representation = {"code": code,
                          "operations" : operations}
        super().__init__(representation)
    
    @property
    def code(self):
        return self.representation["code"]
    
    @property
    def operations(self):
        return self.representation["operations"]
    
class NASBench201Evaluator(abstract.Evaluator):
    def __init__(self, data_path, search_space):
            self.api = API(Path(data_path))
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