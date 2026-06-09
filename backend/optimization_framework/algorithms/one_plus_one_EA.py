from optimization_framework.operators import gaoperators
from optimization_framework.problems.tsp import tour_cost, tour_to_coords

# Bitstrings
def OnePlusOneEA(fitness_fn, bit_length=20, prob=None, max_iterations=None):
    if prob is None:
        prob = 1 / bit_length
    iterations = 0
    parent = gaoperators.generateSingleBitstring(bit_length)
    fitness_parent = fitness_fn(parent)
    fitness_evaluations = 1

    coords = [gaoperators.map_bitstring(parent)]
    fitness_over_time = [fitness_parent]

    while fitness_parent != bit_length:
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1
        offspring = gaoperators.mutation(parent, prob)
        fitness_offspring = fitness_fn(offspring)
        fitness_evaluations += 1

        if fitness_offspring >= fitness_parent:
            parent = offspring
            fitness_parent = fitness_offspring
            coords.append(gaoperators.map_bitstring(parent))

        fitness_over_time.append(fitness_parent)

    return parent, iterations, 0.0, {}, fitness_evaluations, coords, fitness_over_time

# TSP
def OnePlusOneEATSP(distance_matrix, city_coords, max_iterations=10000, mutation="2opt"):
    current = gaoperators.generate_random_ham_cycle(distance_matrix)
    current_cost = tour_cost(current, distance_matrix)
    best = current[:]
    best_cost = current_cost
    fitness_evaluations = 1
    iterations = 0

    tour_coords = tour_to_coords(best, city_coords)
    cost_over_time = [best_cost]

    for _ in range(max_iterations):
        iterations += 1
        neighbor = gaoperators.tsp_mutation(current, distance_matrix, mutation)
        neighbor_cost = tour_cost(neighbor, distance_matrix)
        fitness_evaluations += 1

        if neighbor_cost <= current_cost:
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
                tour_coords = tour_to_coords(best, city_coords)

        cost_over_time.append(best_cost)

    return best, iterations, 0.0, {}, fitness_evaluations, tour_coords, cost_over_time
