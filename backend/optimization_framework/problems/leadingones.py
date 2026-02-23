from optimization_framework.operators import gaoperators

def fitnessLeadingOnes(bitstring):
    fitness = 0
    for i in range(len(bitstring)):
        if (bitstring[i] == "1"):
            fitness +=1
        else:
            break
    return fitness

def leadingonesEA():
    bit_length = 20
    iterations = 0
    tournament_k = 3
    mutation_prob = 1/bit_length
    pop_size = 20
    population = gaoperators.generatePopulation(bit_length, fitnessLeadingOnes)
    best = max(population.values(), key=lambda ind: ind["fitness"])
    fitness_evaluations = pop_size

    while best["fitness"] != bit_length:
        population = gaoperators.createNextGeneration(population, fitnessLeadingOnes, tournament_k=tournament_k, mutation_prob=mutation_prob)
        best = max(population.values(), key=lambda ind: ind["fitness"])
        fitness_evaluations += len(population)
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
    print(leadingonesOnePlusOneEA())
