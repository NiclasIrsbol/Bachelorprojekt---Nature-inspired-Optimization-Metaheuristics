from optimization_framework.operators import gaoperators
import math
import random

def simulated_annealing(fitness_fn, bit_length=20):
    cooling = 0.99
    iterations = 0
    T0 = 100.0
    bit_length = 20
    prob = 1 / bit_length

    current = gaoperators.generateSingleBitstring(bit_length)
    current_fit = fitness_fn(current)
    fitness_evaluations = 1
    best = current
    best_fit = current_fit
    T = float(T0)

    coords = [gaoperators.map_bitstring(best)]
    while best_fit != bit_length:
        iterations += 1
        neighbor = gaoperators.mutation(current, prob)
        neighbor_fit = fitness_fn(neighbor)
        fitness_evaluations += 1
        delta = neighbor_fit - current_fit
        if delta >= 0:
            accept = True
        else:
            if T <= 0.0:
                accept = False
            else:
                accept = (random.random() < math.exp(delta / T))
        if accept:
            current = neighbor 
            current_fit = neighbor_fit
            if current_fit > best_fit:
                best = current 
                best_fit = current_fit
                coords.append(gaoperators.map_bitstring(best))
        T *= cooling
    return best, iterations, T, {}, fitness_evaluations, coords
