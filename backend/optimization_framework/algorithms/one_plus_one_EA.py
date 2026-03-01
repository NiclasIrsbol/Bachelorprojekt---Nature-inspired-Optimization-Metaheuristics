from optimization_framework.operators import gaoperators

def OnePlusOneEA(fitness_fn):
    bit_length = 20
    iterations = 0
    prob = 1/bit_length
    parent = gaoperators.generateSingleBitstring(bit_length)
    fitnessparent = fitness_fn(parent)
    fitness_evaluations = 1

    while fitnessparent != bit_length:
        iterations+=1
        offspring = gaoperators.mutation(parent, prob)
        fitnessoffspring = fitness_fn(offspring)
        fitness_evaluations += 1
        if fitnessoffspring>=fitnessparent:
            parent = offspring
            fitnessparent = fitnessoffspring

    return parent, iterations, {}, fitness_evaluations
