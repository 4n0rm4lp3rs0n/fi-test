"""No-op logger so the UNMODIFIED nbcore.Population.evolve() can run.

Why this exists (confirmed defect in uploaded nbcore.py): evolve() builds
`self.logger.genome_id(...)` / `self.logger.snapshot(...)` arguments eagerly for
the 'selection', 'crossover', 'mutation' and 'evaluation' traces without an
`if self.logger` guard, so Population(..., logger=None) raises AttributeError.
Passing this object keeps nbcore.py byte-identical. It touches no RNG.
"""
class NullLogger:
    def genome_id(self, genome):
        return 0
    def snapshot(self, genome):
        return {}
    def log(self, *args, **kwargs):
        return None
