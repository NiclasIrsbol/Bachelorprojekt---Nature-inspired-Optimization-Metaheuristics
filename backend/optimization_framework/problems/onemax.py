from optimization_framework.operators import gaoperators

def fitnessOnemax(bitstring):
    fitness = 0
    for i in range((len(bitstring))):
        if bitstring[i] == "1":
            fitness += 1
    return fitness