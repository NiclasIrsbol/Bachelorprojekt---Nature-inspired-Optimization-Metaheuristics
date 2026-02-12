import random
from optimization_framework.operators import gaoperators

def fitnessLeadingOnes(bitstring):
    fitness = 0
    for i in range(len(bitstring)):
        if (bitstring[i] == "1"):
            fitness +=1
        else:
            fitness = 0
    return fitness

def leadingones():
    bit_length = 20
    iterations = 0
    tournament_k = 3
    mutation_prob = 1/bit_length
    population = gaoperators.generatePopulation(bit_length, fitnessLeadingOnes)
    best = max(population.values(), key=lambda ind: ind["fitness"])

    while best["fitness"] != bit_length:
        population = gaoperators.createNextGeneration(population, fitnessLeadingOnes, tournament_k=tournament_k, mutation_prob=mutation_prob)
        best = max(population.values(), key=lambda ind: ind["fitness"])
        iterations += 1
    return best, iterations, population

if __name__ == "__main__":
    print(leadingones())