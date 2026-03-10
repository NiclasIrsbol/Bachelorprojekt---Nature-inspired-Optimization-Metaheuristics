import random
from optimization_framework.operators.gaoperators import map_bitstring


def ant_colony_optimization(fitness_fn, bit_length=100, rho=0.1, max_iterations=10000):
    """MMAS (Max-Min Ant System) for pseudo-Boolean optimisation.

    1-to-1 implementation of the MMAS pseudocode (Algorithm 4):
      1: τ_j = 1/2 for all j
      2: x* = CONSTRUCT(C, τ)
      3: update pheromone using x*
      6: repeat
      7:   y = CONSTRUCT(C, τ)
      8:   if f(y) >= f(x*) then x* = y
      9:   update pheromone using x*
     10: until stop
    """
    tau_min = 1 / bit_length
    tau_max = 1 - 1 / bit_length

    # Line 1: Set τ = 1/2 for all bits
    pheromone = [0.5] * bit_length

    iterations = 0
    fitness_evaluations = 0
    coords = []

    def construct():
        bits = []
        for j in range(bit_length):
            bits.append("1" if random.random() < pheromone[j] else "0")
        return "".join(bits)

    def update_pheromone(x_star):
        for j in range(bit_length):
            if x_star[j] == "1":
                pheromone[j] = min((1 - rho) * pheromone[j] + rho, tau_max)
            else:
                pheromone[j] = max((1 - rho) * pheromone[j], tau_min)

    # Line 2: x* = CONSTRUCT(C, τ)
    best = construct()
    best_fit = fitness_fn(best)
    fitness_evaluations += 1
    fitness_over_time = [best_fit]
    coords.append(map_bitstring(best))

    # Lines 3-5: initial pheromone update using x*
    update_pheromone(best)

    # Line 6: repeat
    while best_fit != bit_length and iterations < max_iterations:
        iterations += 1

        # Line 7: y = CONSTRUCT(C, τ)
        y = construct()
        y_fit = fitness_fn(y)
        fitness_evaluations += 1

        # Line 8: if f(y) >= f(x*) then x* = y
        if y_fit >= best_fit:
            best = y
            best_fit = y_fit

        # Lines 9-10: update pheromone using x*
        update_pheromone(best)
        fitness_over_time.append(best_fit)
        coords.append(map_bitstring(best))

    population = {"Solution": {"bit": best, "fitness": best_fit}}
    return best, iterations, 0.0, population, fitness_evaluations, coords, fitness_over_time
