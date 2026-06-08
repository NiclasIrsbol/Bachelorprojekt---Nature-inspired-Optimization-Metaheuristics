"""Tests for the bitstring crossover operators used by the GA-style (μ+λ) EA."""
import random

import pytest

from optimization_framework.operators import gaoperators

VALID_BITS = set("01")


def _is_valid_bitstring(s, n):
    return len(s) == n and set(s) <= VALID_BITS


class TestSinglePointCrossover:
    def test_length_and_bits(self):
        random.seed(0)
        p1, p2 = "0" * 16, "1" * 16
        for _ in range(50):
            o1, o2 = gaoperators.crossover(p1, p2)
            assert _is_valid_bitstring(o1, 16)
            assert _is_valid_bitstring(o2, 16)

    def test_offspring_are_recombinations(self):
        random.seed(1)
        p1, p2 = "0" * 10, "1" * 10
        o1, o2 = gaoperators.crossover(p1, p2)
        # Single-point: o1 = prefix(p1)+suffix(p2) -> some 0s then some 1s.
        assert "0" in o1 and "1" in o1
        # Bit conservation: at each position offspring come from one of the parents.
        for a, b, x, y in zip(p1, p2, o1, o2):
            assert {x, y} == {a, b} or {x, y} == {a} == {b}


class TestTwoPointCrossover:
    def test_length_and_bits(self):
        random.seed(2)
        p1, p2 = "0" * 20, "1" * 20
        for _ in range(50):
            o1, o2 = gaoperators.two_point_crossover(p1, p2)
            assert _is_valid_bitstring(o1, 20)
            assert _is_valid_bitstring(o2, 20)

    def test_position_wise_from_some_parent(self):
        random.seed(3)
        p1 = "0101010101"
        p2 = "1010101010"
        o1, o2 = gaoperators.two_point_crossover(p1, p2)
        for a, b, x in zip(p1, p2, o1):
            assert x in (a, b)
        # complementary: o1[i] and o2[i] together cover both parent bits
        for a, b, x, y in zip(p1, p2, o1, o2):
            assert {x, y} == {a, b}

    def test_identical_parents_return_parent(self):
        random.seed(4)
        p = "1100110011"
        o1, o2 = gaoperators.two_point_crossover(p, p)
        assert o1 == p and o2 == p


class TestUniformCrossover:
    def test_length_and_bits(self):
        random.seed(5)
        p1, p2 = "0" * 32, "1" * 32
        for _ in range(50):
            o1, o2 = gaoperators.uniform_crossover(p1, p2)
            assert _is_valid_bitstring(o1, 32)
            assert _is_valid_bitstring(o2, 32)

    def test_all_same_parents_returns_that_parent(self):
        random.seed(6)
        p = "1" * 24
        for _ in range(20):
            o1, o2 = gaoperators.uniform_crossover(p, p)
            assert o1 == p and o2 == p

    def test_complementary_offspring(self):
        random.seed(7)
        p1, p2 = "0" * 40, "1" * 40
        o1, o2 = gaoperators.uniform_crossover(p1, p2)
        # With complementary parents, the two offspring are bitwise complements.
        for x, y in zip(o1, o2):
            assert x != y

    def test_mixes_both_parents(self):
        random.seed(8)
        p1, p2 = "0" * 200, "1" * 200
        o1, _ = gaoperators.uniform_crossover(p1, p2)
        # Over 200 independent coin flips, expect both symbols present.
        assert "0" in o1 and "1" in o1


def _origin_switches(offspring, parent1, parent2):
    """Count positions where the offspring's source parent changes.

    With complementary parents (all-0 vs all-1) each origin switch flips the
    symbol, so this equals the number of 0<->1 transitions in the offspring.
    """
    switches = 0
    prev_src = None
    for o, a, b in zip(offspring, parent1, parent2):
        if o == a and o != b:
            src = 1
        elif o == b and o != a:
            src = 2
        else:
            src = prev_src  # ambiguous (a == b); keep previous source
        if prev_src is not None and src != prev_src:
            switches += 1
        prev_src = src
    return switches


class TestKPointCrossover:
    def test_length_and_bits_various_k(self):
        for k in (1, 2, 3, 5):
            random.seed(100 + k)
            p1, p2 = "0" * 24, "1" * 24
            for _ in range(30):
                o1, o2 = gaoperators.k_point_crossover(p1, p2, k)
                assert _is_valid_bitstring(o1, 24)
                assert _is_valid_bitstring(o2, 24)

    def test_identical_parents_return_parent(self):
        for k in (1, 2, 3, 7):
            random.seed(200 + k)
            p = "1100110011"
            o1, o2 = gaoperators.k_point_crossover(p, p, k)
            assert o1 == p and o2 == p

    def test_complementary_offspring(self):
        random.seed(300)
        p1, p2 = "0" * 30, "1" * 30
        o1, o2 = gaoperators.k_point_crossover(p1, p2, 3)
        for x, y in zip(o1, o2):
            assert x != y

    def test_k1_single_switch(self):
        random.seed(301)
        p1, p2 = "0" * 20, "1" * 20
        o1, _ = gaoperators.k_point_crossover(p1, p2, 1)
        assert _origin_switches(o1, p1, p2) == 1

    def test_switches_at_most_k(self):
        for k in (1, 2, 3, 5, 8):
            random.seed(400 + k)
            p1, p2 = "0" * 40, "1" * 40
            for _ in range(20):
                o1, _ = gaoperators.k_point_crossover(p1, p2, k)
                assert _origin_switches(o1, p1, p2) <= k

    def test_k_clamped_when_too_large(self):
        random.seed(500)
        p1, p2 = "01", "10"
        # k far larger than n-1 must not raise and must stay valid.
        o1, o2 = gaoperators.k_point_crossover(p1, p2, 99)
        assert _is_valid_bitstring(o1, 2)
        assert _is_valid_bitstring(o2, 2)


class TestCrossoverRegistry:
    def test_get_crossover_returns_callables(self):
        for name in ("single_point", "two_point", "three_point", "four_point", "uniform"):
            fn = gaoperators.get_crossover(name)
            assert callable(fn)

    def test_k_point_variants_produce_valid_offspring(self):
        random.seed(600)
        p1, p2 = "0" * 16, "1" * 16
        for name in ("three_point", "four_point"):
            o1, o2 = gaoperators.get_crossover(name)(p1, p2)
            assert _is_valid_bitstring(o1, 16)
            assert _is_valid_bitstring(o2, 16)

    def test_unknown_crossover_raises(self):
        with pytest.raises(ValueError):
            gaoperators.get_crossover("does_not_exist")

