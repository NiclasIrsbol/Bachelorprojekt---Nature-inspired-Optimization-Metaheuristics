import random
from optimization_framework.problems import onemax

def generateSingleBitstring(length):
    bit = "".join(random.choice("01") for _ in range(length))
    return bit

def generatePopulation(length, fitness_fn, size: int = 20):
    """Create an initial population of `size` individuals."""
    bitstrings = {}
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

def createNextGenerationOffsprings(population, fitness_fn ,tournament_k, mutation_prob, lambda_size=None):
    lambda_size = len(population)
    offsprings = {}
    created = 0

    pair_count = (lambda_size + 1) // 2 # number of parents pairs needed to create the next generation
    for _ in range(pair_count):
        parent1 = selectparents(population, tournament_k)
        parent2 = selectparents(population, tournament_k)
        offspring1, offspring2 = crossover(parent1["bit"], parent2["bit"])
        offspring1 = mutation(offspring1, mutation_prob)
        offspring2 = mutation(offspring2, mutation_prob)

        offsprings[f"Offspring{created}"] = {"bit": offspring1, "fitness": fitness_fn(offspring1)}
        created += 1
        if created < lambda_size:
            offsprings[f"Offspring{created}"] = {"bit": offspring2, "fitness": fitness_fn(offspring2)}
            created += 1
    return offsprings

def selectMuBest(parents, offsprings, mu):
    combined = list(parents.values()) + list(offsprings.values())
    combined.sort(key=lambda ind: ind["fitness"], reverse=True)
    survivors = combined[:mu]
    return {f"Bitstring{i}": ind for i, ind in enumerate(survivors)}

def createNextGenerationMuPlusLambda(population, fitness_fn, mu_size, lambda_size, tournament_k, mutation_prob):
    offspring = createNextGenerationOffsprings(population, fitness_fn, tournament_k, mutation_prob, lambda_size=lambda_size,)
    next_population = selectMuBest(population, offspring, mu=mu_size)
    return next_population, offspring
