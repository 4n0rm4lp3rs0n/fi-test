from abc import ABC, abstractmethod


# ============================================================
# Genome
# ============================================================

class Genome(ABC):
    """
    Generic representation of one NAS candidate.

    The representation is intentionally opaque to the generic GA.
    Each SearchSpace decides what the representation contains.
    """

    def __init__(self, representation):
        self.representation = representation
        self.fitness = None
        self.metrics = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"representation={self.representation!r}, "
            f"fitness={self.fitness})"
        )


# ============================================================
# Search Space
# ============================================================

class SearchSpace(ABC):
    """
    Defines everything that is specific to a particular NAS
    search space.

    The generic GA should not need to know whether this is
    NAS-Bench-101, NAS-Bench-201, NATS-Bench, etc.
    """

    @abstractmethod
    def random_genome(self):
        """
        Generate one valid random genome.
        """
        pass

    @abstractmethod
    def guided_genome(self, guidance):
        """
        Generate one valid genome using feature guidance.
        """
        pass

    @abstractmethod
    def validate(self, genome):
        """
        Check whether a genome represents a valid architecture.

        Returns:
            (bool, reason)
        """
        pass

    @abstractmethod
    def decode(self, genome):
        """
        Convert the genome into the benchmark-native
        architecture representation.
        """
        pass

    @abstractmethod
    def feature_data(self, genome):
        """
        Convert a genome into its architectural decision features.

        This is what the feature-importance module sees.

        Example:
            NAS101 -> [bit1, ..., bit21, layer1, ..., layer5]
            NAS201 -> [edge1, ..., edge6]
        """
        pass

    @abstractmethod
    def feature_schema(self):
        """
        Describe the features and their types/domains.
        """
        pass


# ============================================================
# Evaluator
# ============================================================

class Evaluator(ABC):
    """
    Converts a valid genome into benchmark performance.
    """

    @abstractmethod
    def evaluate(self, genome):
        """
        Return a normalized result dictionary.

        The generic GA should be able to rely on at least:
            result["fitness"]
        """
        pass


# ============================================================
# Feature Importance
# ============================================================

class FeatureImportance(ABC):
    """
    Generic interface for learning feature importance from
    evolutionary population data.
    """

    @abstractmethod
    def extract_data(self, population_data, search_space):
        """
        Convert population history into an FI-ready dataset.
        """
        pass

    # @abstractmethod
    # def calculate(self, data):
    #     """
    #     Calculate importance/effect statistics.
    #     """
    #     pass