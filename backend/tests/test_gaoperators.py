"""Tests for genetic operators (mutationSA, (μ+λ) λ offspring count)."""
import random

import pytest

from optimization_framework.operators import gaoperators
from optimization_framework.problems.onemax import fitnessOnemax


class TestMutationSA:
    def test_flips_exactly_one_bit(self):
        bit = "0" * 32
        for _ in range(200):
            mutant = gaoperators.mutationSA(bit)
            assert len(mutant) == len(bit)
            assert sum(a != b for a, b in zip(bit, mutant)) == 1

    def test_preserves_length(self):
        bit = "1010101010"
        assert len(gaoperators.mutationSA(bit)) == 10


class TestMuPlusLambdaOffspringCount:
    def test_lambda_offspring_count(self):
        random.seed(0)
        population = {
            f"Bitstring{i}": {"bit": format(i, "010b"), "fitness": fitnessOnemax(format(i, "010b"))}
            for i in range(4)
        }
        mu, lam = 2, 5
        next_gen = gaoperators.createNextGenerationMuPlusLambda(
            population,
            fitnessOnemax,
            mu_size=mu,
            lambda_size=lam,
            tournament_k=2,
            mutation_prob=1 / 10,
        )
        assert len(next_gen) == mu
        # Offspring dict is internal; count via repeated calls to offsprings helper
        random.seed(1)
        offsprings = gaoperators.createNextGenerationOffsprings(
            population,
            fitnessOnemax,
            tournament_k=2,
            mutation_prob=1 / 10,
            lambda_size=lam,
        )
        assert len(offsprings) == lam
