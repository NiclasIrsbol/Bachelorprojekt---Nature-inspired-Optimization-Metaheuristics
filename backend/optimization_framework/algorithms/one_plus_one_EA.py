from optimization_framework.operators import gaoperators


def OnePlusOneEA(fitness_fn, bit_length=20):
    iterations = 0
    prob = 1 / bit_length
    parent = gaoperators.generateSingleBitstring(bit_length)
    fitness_parent = fitness_fn(parent)
    fitness_evaluations = 1

    coords = [gaoperators.map_bitstring(parent)]

    while fitness_parent != bit_length:
        iterations += 1
        offspring = gaoperators.mutation(parent, prob)
        fitness_offspring = fitness_fn(offspring)
        fitness_evaluations += 1

        if fitness_offspring >= fitness_parent:
            parent = offspring
            fitness_parent = fitness_offspring
            coords.append(gaoperators.map_bitstring(parent))

    return parent, iterations, 0.0, {}, fitness_evaluations, coords
