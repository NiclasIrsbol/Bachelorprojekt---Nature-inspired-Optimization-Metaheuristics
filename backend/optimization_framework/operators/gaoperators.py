import random
from optimization_framework.problems import onemax

def generatePopulation(length, fitness_fn):
    bitstrings = {}
    size = 20
    for i in range(size):
        bit = "".join(random.choice("01") for _ in range(length))
        fitness = fitness_fn(bit)
        bitstrings[f"Bitstring{i}"] = {"bit": bit, "fitness": fitness}
    return bitstrings


def selectparents(population: dict, tournament_k: int):
    individuals = list(population.values())
    k = min(max(1, tournament_k), len(individuals))
    competitors = random.sample(individuals, k)
    return max(competitors, key=lambda ind: ind["fitness"])

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2

def mutation(bit, prob):
    if random.random() < prob:
        index = random.randint(0, len(bit) - 1)
        new_char = "0" if bit[index] == "1" else "1"
        bit = bit[:index] + new_char + bit[index+1:]
    return bit

def createNextGeneration(population, fitness_fn ,tournament_k, mutation_prob):
    generation_size = len(population)
    next_generation = {}

    pair_count = (generation_size + 1) // 2 # number of parents pairs needed to create the next generation
    for pair_index in range(pair_count):
        parent1 = selectparents(population, tournament_k)
        parent2 = selectparents(population, tournament_k)

        offspring1, offspring2 = crossover(parent1["bit"], parent2["bit"])
        offspring1 = mutation(offspring1, mutation_prob)
        offspring2 = mutation(offspring2, mutation_prob)

        i1 = 2 * pair_index # index/key for the 1st child from this parent pair (0,2,4,...)
        i2 = i1 + 1 # index/key for the 2nd child from this parent pair (1,3,5,...)
        next_generation[f"Bitstring{i1}"] = {"bit": offspring1, "fitness": fitness_fn(offspring1)}
        if i2 < generation_size:
            next_generation[f"Bitstring{i2}"] = {"bit": offspring2, "fitness": fitness_fn(offspring2)}

    return next_generation