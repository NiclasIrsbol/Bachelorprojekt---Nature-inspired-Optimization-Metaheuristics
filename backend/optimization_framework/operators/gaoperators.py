import random
import math

def map_bitstring(x):
    if isinstance(x, str):
        bits = [1 if c == "1" else 0 for c in x]
    else:

        bits = []
        for v in x:
            if isinstance(v, str):
                if v not in ("0", "1"):
                    raise ValueError(f"Expected bits '0'/'1', got {v!r}")
                bits.append(1 if v == "1" else 0)
            else:
                bits.append(1 if int(v) == 1 else 0)

    n = len(bits)
    o = sum(bits)
    Y = 2 * (o / n) - 1
    score = sum(i * bits[i] for i in range(n))
    if o == 0:
        lowestScore = 0
        highestScore = 0
    else:
        lowestScore = sum(range(o))
        highestScore = sum(range(n-o, n))
    b = math.sqrt(1 - Y**2)
    a = -b
    if lowestScore == highestScore:
        X = 0
    else:
        X = (b - a) * (score - lowestScore) / (highestScore - lowestScore) + a
    return (X, Y)

def generateSingleBitstring(length):
    """Create a single bitstring of size length."""
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
    """Select k parents with best fitness and add to dict."""
    individuals = list(population.values())
    k = min(max(1, tournament_k), len(individuals))
    competitors = random.sample(individuals, k)
    return max(competitors, key=lambda ind: ind["fitness"])

def crossover(parent1, parent2):
    """Perform single point crossover to generate offsprings."""
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2

def mutation(bit, prob):
    """Mutate individual by flipping random bit"""
    if random.random() < prob:
        index = random.randint(0, len(bit) - 1)
        new_char = "0" if bit[index] == "1" else "1"
        bit = bit[:index] + new_char + bit[index+1:]
    return bit

def createNextGenerationOffsprings(population, fitness_fn ,tournament_k, mutation_prob, lambda_size=None):
    """Creates the next generation of offsprings using crossover and mutation"""
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
    """Select mu best parents based on fitness"""
    combined = list(parents.values()) + list(offsprings.values())
    combined.sort(key=lambda ind: ind["fitness"], reverse=True)
    survivors = combined[:mu]
    return {f"Bitstring{i}": ind for i, ind in enumerate(survivors)}

def createNextGenerationMuPlusLambda(population, fitness_fn, mu_size, lambda_size, tournament_k, mutation_prob):
    """Create the next generation"""
    offspring = createNextGenerationOffsprings(population, fitness_fn, tournament_k, mutation_prob, lambda_size=lambda_size,)
    next_population = selectMuBest(population, offspring, mu=mu_size)
    return next_population, offspring
