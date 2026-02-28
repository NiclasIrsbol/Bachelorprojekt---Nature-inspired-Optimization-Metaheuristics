from optimization_framework.operators import gaoperators
from optimization_framework.problems import onemax
from optimization_framework.problems import leadingones
import math

def simulated_annealing(fitness_fn):
    iterations = 0
    bit_length = 20
    prob = 1/bit_length
    bit = gaoperators.generateSingleBitstring(bit_length)
    temp = 100
    fitnessparent = fitness_fn(bit)

    while (fitnessparent != bit_length and temp>0):
        iterations += 1
        neighbor = gaoperators.mutation(bit, prob)
        fitness_neighbor = fitness_fn(neighbor)
        delta = fitness_neighbor-fitnessparent

        if delta <= 0:
            bit = neighbor
            fitnessparent = fitness_neighbor
        else:
            if (temp == 0):
                p = 0
            else:
                p = math.exp(-delta/temp)
                r = gaoperators.generateRandomInt(0,1)
            if (r<=p):
                bit = neighbor
                fitnessparent = fitness_neighbor
        temp -= 1
    return bit, iterations, temp

print(simulated_annealing(onemax.fitnessOnemax))