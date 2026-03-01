import random


def ant_colony_optimization(fitness_fn):
    bit_length = 20
    num_ants = 20
    evaporation_rate = 0.5
    initial_pheromone = 1.0

    pheromone = [initial_pheromone] * bit_length

    best = None
    best_fit = -1
    iterations = 0
    fitness_evaluations = 0
    population = {}

    max_iterations = 10_000

    while best_fit != bit_length and iterations < max_iterations:
        iterations += 1
        ants = []

        for a in range(num_ants):
            bits = []
            for j in range(bit_length):
                prob_one = pheromone[j] / (pheromone[j] + 1.0)
                bits.append("1" if random.random() < prob_one else "0")
            bitstring = "".join(bits)
            fitness = fitness_fn(bitstring)
            fitness_evaluations += 1
            ants.append({"bit": bitstring, "fitness": fitness})

            if fitness > best_fit:
                best = bitstring
                best_fit = fitness

        # Evaporate
        for j in range(bit_length):
            pheromone[j] *= (1 - evaporation_rate)

        # Deposit pheromone from the global best
        deposit = best_fit / bit_length
        for j in range(bit_length):
            if best[j] == "1":
                pheromone[j] += deposit

        population = {f"Ant{i}": ant for i, ant in enumerate(ants)}

    return best, iterations, population, fitness_evaluations
