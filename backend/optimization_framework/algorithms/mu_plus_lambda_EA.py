from optimization_framework.operators import gaoperators

def MuPlusLambdaEA(fitness_fn):
    bit_length = 20
    iterations = 0
    tournament_k = 3
    mutation_prob = 1/bit_length
    mu_size = 20
    lambda_size = 40
    population = gaoperators.generatePopulation(bit_length, fitness_fn, size=mu_size)
    best = max(population.values(), key=lambda ind: ind["fitness"])
    fitness_evaluations = len(population)

    while best["fitness"] != bit_length:
        population, offspring = gaoperators.createNextGenerationMuPlusLambda(population, fitness_fn, mu_size, lambda_size, tournament_k, mutation_prob,)
        best = max(population.values(), key=lambda ind: ind["fitness"])
        fitness_evaluations += len(offspring)
        iterations += 1

    return best, iterations, population, fitness_evaluations
