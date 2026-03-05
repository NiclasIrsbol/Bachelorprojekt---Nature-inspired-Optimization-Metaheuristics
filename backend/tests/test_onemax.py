from optimization_framework.problems.onemax import fitnessOnemax
def test_all_ones():
    assert fitnessOnemax("11111") == 5
def test_all_zeros():
    assert fitnessOnemax("00000") == 0