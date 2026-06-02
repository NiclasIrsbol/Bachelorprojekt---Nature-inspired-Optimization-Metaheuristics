from optimization_framework.operators import gaoperators
import math
import random
from optimization_framework.problems.tsp import tour_cost


# Bitstrings
def simulated_annealing(fitness_fn, bit_length=20, cooling=0.99, T0=100.0, prob=None, max_iterations=None):
    if prob is None:
        prob = 1 / bit_length
    iterations = 0

    current = gaoperators.generateSingleBitstring(bit_length)
    current_fit = fitness_fn(current)
    fitness_evaluations = 1
    best = current
    best_fit = current_fit
    T = float(T0)

    coords = [gaoperators.map_bitstring(best)]
    fitness_over_time = [best_fit]

    while best_fit != bit_length:
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1
        neighbor = gaoperators.mutation(current, prob)
        neighbor_fit = fitness_fn(neighbor)
        fitness_evaluations += 1
        delta = neighbor_fit - current_fit
        if delta >= 0:
            accept = True
        else:
            if T <= 0.0:
                accept = False
            else:
                accept = (random.random() < math.exp(delta / T))
        if accept:
            current = neighbor 
            current_fit = neighbor_fit
            if current_fit > best_fit:
                best = current 
                best_fit = current_fit
                coords.append(gaoperators.map_bitstring(best))
        fitness_over_time.append(best_fit)
        T *= cooling
    return best, iterations, T, {}, fitness_evaluations, coords, fitness_over_time

# TSP
def simulated_annealingTSP(distance_matrix, city_coords, cooling=0.9995, T0=1000.0, max_iterations=100000):

    current = gaoperators.generate_random_ham_cycle(distance_matrix)
    current_cost = tour_cost(current, distance_matrix)
    best = current[:]
    best_cost = current_cost
    fitness_evaluations = 1
    iterations = 0
    T = float(T0)

    tour_coords = _tour_to_coords(best, city_coords)
    cost_over_time = [best_cost]

    for _ in range(max_iterations):
        iterations += 1
        neighbor = gaoperators.two_opt_mutation(current)
        neighbor_cost = tour_cost(neighbor, distance_matrix)
        fitness_evaluations += 1
        delta = neighbor_cost - current_cost

        if delta <= 0:
            accept = True
        elif T > 0:
            accept = (random.random() < math.exp(-delta / T))
        else:
            accept = False

        if accept:
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
                tour_coords = _tour_to_coords(best, city_coords)

        cost_over_time.append(best_cost)
        T *= cooling

    return best, iterations, T, {}, fitness_evaluations, tour_coords, cost_over_time

def _tour_to_coords(tour, city_coords):
    """Convert a tour (list of 0-based indices) to (x, y) in tour order."""
    if not city_coords:
        return []
    node_ids = sorted(city_coords.keys())
    if max(tour) >= len(node_ids):
        return []
    return [(city_coords[node_ids[i]][0], city_coords[node_ids[i]][1]) for i in tour]