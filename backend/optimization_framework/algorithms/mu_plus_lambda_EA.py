from optimization_framework.operators import gaoperators

def MuPlusLambdaEA(fitness_fn, bit_length=20):
    bit_length = 20
    iterations = 0
    tournament_k = 3
    mutation_prob = 1/bit_length
    mu_size = 20
    lambda_size = 40
    population = gaoperators.generatePopulation(bit_length, fitness_fn, size=mu_size)
    best = max(population.values(), key=lambda ind: ind["fitness"])
    fitness_evaluations = len(population)

    coords = [gaoperators.map_bitstring(best["bit"])]
    fitness_over_time = [best["fitness"]]


    while best["fitness"] != bit_length:
        population, offspring = gaoperators.createNextGenerationMuPlusLambda(population, fitness_fn, mu_size, lambda_size, tournament_k, mutation_prob,)
        best = max(population.values(), key=lambda ind: ind["fitness"])
        fitness_evaluations += len(offspring)
        iterations += 1
        coords.append(gaoperators.map_bitstring(best["bit"]))
        fitness_over_time.append(best["fitness"])

    return best, iterations, 0.0, population, fitness_evaluations, coords, fitness_over_time
