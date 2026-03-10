from optimization_framework.operators import gaoperators


def OnePlusOneEA(fitness_fn, bit_length=20, prob=None):
    if prob is None:
        prob = 1 / bit_length
    iterations = 0
    parent = gaoperators.generateSingleBitstring(bit_length)
    fitness_parent = fitness_fn(parent)
    fitness_evaluations = 1

    coords = [gaoperators.map_bitstring(parent)]
    fitness_over_time = [fitness_parent]

    while fitness_parent != bit_length:
        iterations += 1
        offspring = gaoperators.mutation(parent, prob)
        fitness_offspring = fitness_fn(offspring)
        fitness_evaluations += 1

        if fitness_offspring >= fitness_parent:
            parent = offspring
            fitness_parent = fitness_offspring
            coords.append(gaoperators.map_bitstring(parent))

        fitness_over_time.append(fitness_parent)

    return parent, iterations, 0.0, {}, fitness_evaluations, coords, fitness_over_time