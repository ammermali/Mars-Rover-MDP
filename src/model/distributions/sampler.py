from typing import Mapping, TypeVar
import random

EPSILON = 0.000001

T = TypeVar('T')
def sample_distribution(distribution: Mapping[T, float], rng: random.Random) -> T:
    if not distribution:
        raise ValueError("Distribution necessary.")
    outcomes = list(distribution)
    probabilities = list(distribution.values())
    if any(probability < 0 for probability in probabilities):
        raise ValueError("Probabilities must be non-negative.")
    total = sum(probabilities)
    if not (1 - EPSILON) <= total <= (1 + EPSILON):
        raise ValueError("Probabilities must sum to 1.")
    return rng.choices(outcomes, weights=probabilities, k=1)[0]