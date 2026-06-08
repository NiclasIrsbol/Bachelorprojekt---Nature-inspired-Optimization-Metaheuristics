import random
import math
from functools import partial

# Bitstrings
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
    """Perform single point crossover to generate offsprings.

    A single cut point is chosen uniformly in ``[1, n-1]`` and the tails are
    swapped. This is the classic single-point bitstring crossover.
    """
    if len(parent1) < 2:
        return parent1, parent2
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2


def two_point_crossover(parent1, parent2):
    """Two-point crossover: swap the middle segment between two cut points.

    Two cut points ``i <= j`` are chosen uniformly in ``[1, n-1]``; the segment
    ``[i, j)`` is exchanged between the parents. Same signature/return shape as
    :func:`crossover` (returns two offspring strings).
    """
    n = len(parent1)
    if n < 2:
        return parent1, parent2
    i = random.randint(1, n - 1)
    j = random.randint(1, n - 1)
    if i > j:
        i, j = j, i
    offspring1 = parent1[:i] + parent2[i:j] + parent1[j:]
    offspring2 = parent2[:i] + parent1[i:j] + parent2[j:]
    return offspring1, offspring2


def uniform_crossover(parent1, parent2):
    """Uniform crossover: each position is independently inherited from either
    parent with probability 1/2.

    If both parents are identical, both offspring equal that parent. Same
    signature/return shape as :func:`crossover` (returns two offspring strings).
    """
    offspring1 = []
    offspring2 = []
    for a, b in zip(parent1, parent2):
        if random.random() < 0.5:
            offspring1.append(a)
            offspring2.append(b)
        else:
            offspring1.append(b)
            offspring2.append(a)
    return "".join(offspring1), "".join(offspring2)


def k_point_crossover(parent1, parent2, k):
    """k-point crossover: k distinct cut points, alternating segments.

    Chooses ``k`` distinct cut points uniformly in ``[1, n-1]``, splitting the
    strings into ``k+1`` segments, and alternates which parent each segment comes
    from. Generalizes single-point (k=1) and two-point (k=2). Same signature/
    return shape as :func:`crossover` (returns two offspring strings).
    """
    n = len(parent1)
    if n < 2 or k < 1:
        return parent1, parent2
    k = min(k, n - 1)
    points = sorted(random.sample(range(1, n), k))
    o1, o2, swap, prev = [], [], False, 0
    for p in points + [n]:
        if swap:
            o1.append(parent2[prev:p]); o2.append(parent1[prev:p])
        else:
            o1.append(parent1[prev:p]); o2.append(parent2[prev:p])
        swap = not swap
        prev = p
    return "".join(o1), "".join(o2)


# Registry of bitstring crossover operators used by the GA-style (μ+λ) variant
# and the crossover-comparison experiment.
CROSSOVER_OPERATORS = {
    "single_point": crossover,
    "two_point": two_point_crossover,
    "three_point": partial(k_point_crossover, k=3),
    "four_point": partial(k_point_crossover, k=4),
    "uniform": uniform_crossover,
}


def get_crossover(crossover_type):
    """Return the crossover function for ``crossover_type``.

    Valid keys are the keys in ``CROSSOVER_OPERATORS``.
    """
    try:
        return CROSSOVER_OPERATORS[crossover_type]
    except KeyError:
        raise ValueError(
            f"Unknown crossover_type {crossover_type!r}; "
            f"expected one of {sorted(CROSSOVER_OPERATORS)}"
        )

def mutation(bit, prob):
    """Standard bit-flip mutation: flip each bit independently with probability prob.

    With prob = 1/n this flips one bit in expectation, unlike flipping a single
    bit only with probability prob.
    """
    return "".join(
        ("0" if c == "1" else "1") if random.random() < prob else c
        for c in bit
    )

def mutationSA(bit):
    """Single-bit-flip neighborhood for Simulated Annealing.

    Flips exactly one uniformly random bit.
    """
    index = random.randint(0, len(bit) - 1)
    flipped = "0" if bit[index] == "1" else "1"
    return bit[:index] + flipped + bit[index + 1:]

def createNextGenerationOffsprings(population, fitness_fn, tournament_k, mutation_prob, lambda_size=None, crossover_type="single_point"):
    """Create the next generation of offsprings using crossover and mutation
    (GA-style path).

    Each offspring pair is produced by tournament-selecting two parents, applying
    the chosen crossover, then bit-flip mutation. This is used by the GA-style
    (μ+λ) variant and the crossover-comparison experiment.
    """
    if lambda_size is None:
        lambda_size = len(population)
    cross = get_crossover(crossover_type)
    offsprings = {}
    created = 0

    pair_count = (lambda_size + 1) // 2 # number of parents pairs needed to create the next generation
    for _ in range(pair_count):
        parent1 = selectparents(population, tournament_k)
        parent2 = selectparents(population, tournament_k)
        offspring1, offspring2 = cross(parent1["bit"], parent2["bit"])
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

# TSP
def order_crossover(parent1, parent2):
    """Order Crossover (OX) for permutation-based representations."""
    n = len(parent1)
    i, j = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[i:j+1] = parent1[i:j+1]
    fill = [x for x in parent2 if x not in child[i:j+1]]
    pos = 0
    for k in range(n):
        if child[k] is None:
            child[k] = fill[pos]
            pos += 1
    return child

def generate_random_ham_cycle(distance_matrix): 
    nodes = list(range(len(distance_matrix)))
    tour = nodes[:]
    random.shuffle(tour)
    return tour

def two_opt_mutation(ham_cycle):
    n = len(ham_cycle)
    i = random.randint(0, n - 1)
    k = random.randint(0, n - 1)
    while k == i:
        k = random.randint(0, n - 1)
    if i > k:
        i, k = k, i
    new_tour = ham_cycle[:i+1] + ham_cycle[i+1:k+1][::-1] + ham_cycle[k+1:]
    return new_tour

def three_opt_mutation(ham_cycle, distance_matrix):
    n = len(ham_cycle)
    positions = sorted(random.sample(range(n), 3))
    i, j, k = positions
    A = ham_cycle[i+1:j+1]
    B = ham_cycle[j+1:k+1]
    C = ham_cycle[k+1:] + ham_cycle[:i+1]

    candidates = [
        C + A + B,               # original
        C + A + B[::-1],         # reverse B
        C + A[::-1] + B,         # reverse A
        C + A[::-1] + B[::-1],   # reverse both
        C + B + A,               # swap A and B
        C + B + A[::-1],         # swap, reverse A
        C + B[::-1] + A,         # swap, reverse B
        C + B[::-1] + A[::-1],   # swap, reverse both
    ]

    def tour_cost(tour):
        return sum(
            distance_matrix[tour[idx]][tour[(idx + 1) % n]]
            for idx in range(n)
        )

    best_tour = min(candidates, key=tour_cost)
    return best_tour
