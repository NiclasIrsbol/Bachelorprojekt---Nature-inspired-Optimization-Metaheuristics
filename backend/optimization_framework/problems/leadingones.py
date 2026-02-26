from optimization_framework.operators import gaoperators

def fitnessLeadingOnes(bitstring):
    fitness = 0
    for i in range(len(bitstring)):
        if (bitstring[i] == "1"):
            fitness +=1
        else:
            break
    return fitness

def leadingonesMuPlusLambdaEA():
    bit_length = 20
    iterations = 0
    tournament_k = 3
    mutation_prob = 1/bit_length
    mu_size = 20
    lambda_size = 40
    population = gaoperators.generatePopulation(bit_length, fitnessLeadingOnes, size=mu_size)
    best = max(population.values(), key=lambda ind: ind["fitness"])
    fitness_evaluations = len(population)

    while best["fitness"] != bit_length:
        population, offspring = gaoperators.createNextGenerationMuPlusLambda(population, fitnessLeadingOnes, mu_size, lambda_size, tournament_k, mutation_prob,)
        best = max(population.values(), key=lambda ind: ind["fitness"])
        fitness_evaluations += len(offspring)
        iterations += 1

    return best, iterations, population, fitness_evaluations

def leadingonesOnePlusOneEA():
    bit_length = 20
    iterations = 0
    prob = 1/bit_length
    parent = gaoperators.generateSingleBitstring(bit_length)
    fitnessparent = fitnessLeadingOnes(parent)
    fitness_evaluations = 1

    while fitnessparent != bit_length:
        iterations+=1
        offspring = gaoperators.mutation(parent, prob)
        fitnessoffspring = fitnessLeadingOnes(offspring)
        fitness_evaluations += 1
        if fitnessoffspring>=fitnessparent:
            parent = offspring
            fitnessparent = fitnessoffspring

    return parent, iterations, {}, fitness_evaluations

if __name__ == "__main__":
    print(leadingonesMuPlusLambdaEA())
