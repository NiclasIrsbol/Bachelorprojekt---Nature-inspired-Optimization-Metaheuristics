from optimization_framework.operators import gaoperators
import random
from optimization_framework.problems.tsp import tour_cost


# Bitstrings
def MuPlusLambdaEA(
    fitness_fn,
    bit_length=20,
    mu_size=20,
    lambda_size=40,
    tournament_k=3,
    mutation_prob=None,
    max_iterations=None,
    crossover_type="uniform",
):
    """GA-style (mu+lambda) EA for bitstring optimisation.

    This is the framework variant used in the application and report
    experiments: tournament parent selection, configurable crossover,
    standard bit mutation, and elitist plus-selection.
    """
    if mutation_prob is None:
        mutation_prob = 1 / bit_length
    iterations = 0
    population = gaoperators.generatePopulation(bit_length, fitness_fn, size=mu_size)
    best = max(population.values(), key=lambda ind: ind["fitness"])
    fitness_evaluations = len(population)

    coords = [gaoperators.map_bitstring(best["bit"])]
    fitness_over_time = [best["fitness"]]


    while best["fitness"] != bit_length:
        if max_iterations is not None and iterations >= max_iterations:
            break
        population, offspring = gaoperators.createNextGenerationMuPlusLambda(
            population,
            fitness_fn,
            mu_size,
            lambda_size,
            tournament_k,
            mutation_prob,
            crossover_type=crossover_type,
        )
        best = max(population.values(), key=lambda ind: ind["fitness"])
        fitness_evaluations += len(offspring)
        iterations += 1
        coords.append(gaoperators.map_bitstring(best["bit"]))
        fitness_over_time.append(best["fitness"])

    return best, iterations, 0.0, population, fitness_evaluations, coords, fitness_over_time


def MuPlusLambdaGA(*args, **kwargs):
    """Backward-compatible name for the GA-style (mu+lambda) EA.

    Older experiment scripts use this name when emphasizing that the framework
    implementation includes crossover. It delegates to ``MuPlusLambdaEA``.
    """
    return MuPlusLambdaEA(*args, **kwargs)

# TSP
def MuPlusLambdaEATSP(distance_matrix, city_coords, mu_size=20, lambda_size=40, tournament_k=3, max_iterations=5000):

    population = []
    for _ in range(mu_size):
        tour = gaoperators.generate_random_ham_cycle(distance_matrix)
        cost = tour_cost(tour, distance_matrix)
        population.append({"tour": tour, "cost": cost})
    fitness_evaluations = mu_size

    best = min(population, key=lambda ind: ind["cost"])
    tour_coords = _tour_to_coords(best["tour"], city_coords)
    cost_over_time = [best["cost"]]
    iterations = 0

    for _ in range(max_iterations):
        iterations += 1
        offspring = []
        for _ in range(lambda_size):
            competitors = random.sample(population, min(tournament_k, len(population)))
            p1 = min(competitors, key=lambda ind: ind["cost"])
            competitors = random.sample(population, min(tournament_k, len(population)))
            p2 = min(competitors, key=lambda ind: ind["cost"])

            child_tour = gaoperators.order_crossover(p1["tour"], p2["tour"])
            child_tour = gaoperators.two_opt_mutation(child_tour)
            child_cost = tour_cost(child_tour, distance_matrix)
            fitness_evaluations += 1
            offspring.append({"tour": child_tour, "cost": child_cost})

        combined = population + offspring
        combined.sort(key=lambda ind: ind["cost"])
        population = combined[:mu_size]
        best = population[0]
        cost_over_time.append(best["cost"])
        tour_coords = _tour_to_coords(best["tour"], city_coords)

    pop_dict = {
        f"Tour{i}": {"tour": ind["tour"], "cost": ind["cost"]}
        for i, ind in enumerate(population)
    }
    return best["tour"], iterations, 0.0, pop_dict, fitness_evaluations, tour_coords, cost_over_time

def _tour_to_coords(tour, city_coords):
    """Convert a tour (list of 0-based indices) to (x, y) in tour order."""
    if not city_coords:
        return []
    node_ids = sorted(city_coords.keys())
    if max(tour) >= len(node_ids):
        return []
    return [(city_coords[node_ids[i]][0], city_coords[node_ids[i]][1]) for i in tour]
