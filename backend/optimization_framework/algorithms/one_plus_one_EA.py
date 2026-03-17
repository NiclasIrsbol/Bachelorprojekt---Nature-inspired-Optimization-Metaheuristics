from optimization_framework.operators import gaoperators

# Bitstrings
def OnePlusOneEA(fitness_fn, bit_length=20, prob=None):
    if prob is None:
        prob = 1 / bit_length
    iterations = 0
    parent = gaoperators.generateSingleBitstring(bit_length)
    fitness_parent = fitness_fn(parent)
    fitness_evaluations = 1

    coords = [gaoperators.map_bitstring(parent)]
    fitness_over_time = [fitness_parent]

    while fitness_parent != bit_length:
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
def OnePlusOneEATSP(distance_matrix, city_coords, max_iterations=10000):
    from optimization_framework.problems.tsp import tour_cost

    n = len(distance_matrix)
    current = gaoperators.generate_random_ham_cycle(distance_matrix)
    current_cost = tour_cost(current, distance_matrix)
    best = current[:]
    best_cost = current_cost
    fitness_evaluations = 1
    iterations = 0

    tour_coords = _tour_to_coords(best, city_coords)
    cost_over_time = [best_cost]

    for iteration in range(max_iterations):
        iterations += 1
        neighbor = gaoperators.two_opt_mutation(current)
        neighbor_cost = tour_cost(neighbor, distance_matrix)
        fitness_evaluations += 1

        if neighbor_cost <= current_cost:
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
                tour_coords = _tour_to_coords(best, city_coords)

        cost_over_time.append(best_cost)

    return best, iterations, 0.0, {}, fitness_evaluations, tour_coords, cost_over_time


def _tour_to_coords(tour, city_coords):
    """Convert a tour (list of 0-based indices) to (x, y) in tour order."""
    if not city_coords:
        return []
    node_ids = sorted(city_coords.keys())
    if max(tour) >= len(node_ids):
        return []
    return [(city_coords[node_ids[i]][0], city_coords[node_ids[i]][1]) for i in tour]