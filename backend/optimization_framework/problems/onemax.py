import random

def generateBitstrings(length):
    bitstrings = {}
    size = 20
    for i in range(size):
        bit = "".join(random.choice("01") for _ in range(5))
        fitness = fitnessOnemax(bit)
        bitstrings[f"Bitstring{i}"] = {"bit": bit, "fitness": fitness}
    return bitstrings

def fitnessOnemax(bitstring):
    fitness = 0
    for i in range((len(bitstring))):
        if bitstring[i] == "1":
            fitness += 1
    return fitness

intial_population = generateBitstrings(5)
bitstring, fitness = random.choice(list(intial_population.items()))