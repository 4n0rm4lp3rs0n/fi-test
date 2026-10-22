
# selfmade/evolution_logger.py

import copy
import json
from pathlib import Path


class EvolutionLogger:
    def __init__(self):
        self.events = []
        self._next_genome_id = 0
        self._genome_ids = {}

    def genome_id(self, genome):
        # Track object identity during a run.
        key = id(genome)
        if key not in self._genome_ids:
            self._genome_ids[key] = f"G{self._next_genome_id:04d}"
            self._next_genome_id += 1
        return self._genome_ids[key]

    @staticmethod
    def snapshot(genome):
        if genome is None:
            return None

        return {
            "code": str(genome.code),
            "operations": list(genome.operations),
            "fitness": getattr(genome, "fitness", None),
            "metrics": copy.deepcopy(
                getattr(genome, "metrics", None)
            ),
        }

    def log(self, generation, phase, **details):
        self.events.append({
            "step": len(self.events),
            "generation": generation,
            "phase": phase,
            **copy.deepcopy(details),
        })

    def save(self, path="evolution_trace.json"):
        Path(path).write_text(
            json.dumps(self.events, indent=2, default=str),
            encoding="utf-8",
        )
