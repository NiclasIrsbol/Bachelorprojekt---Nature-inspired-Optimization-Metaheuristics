import random
from optimization_framework.operators.gaoperators import map_bitstring

# Bitstrings
def ant_colony_optimization(fitness_fn, bit_length=100, rho=0.1, max_iterations=10000):
    """MMAS (Max-Min Ant System) for pseudo-Boolean optimisation.

    1-to-1 implementation of the MMAS pseudocode (Algorithm 4):
      1: τ_j = 1/2 for all j
      2: x* = CONSTRUCT(C, τ)
      3: update pheromone using x*
      6: repeat
      7:   y = CONSTRUCT(C, τ)
      8:   if f(y) >= f(x*) then x* = y
      9:   update pheromone using x*
     10: until stop
    """
    tau_min = 1 / bit_length
    tau_max = 1 - 1 / bit_length

    # Line 1: Set τ = 1/2 for all bits
    pheromone = [0.5] * bit_length

    iterations = 0
    fitness_evaluations = 0
    coords = []

    def construct():
        bits = []
        for j in range(bit_length):
            bits.append("1" if random.random() < pheromone[j] else "0")
        return "".join(bits)

    def update_pheromone(x_star):
        for j in range(bit_length):
            if x_star[j] == "1":
                pheromone[j] = min((1 - rho) * pheromone[j] + rho, tau_max)
            else:
                pheromone[j] = max((1 - rho) * pheromone[j], tau_min)

    # Line 2: x* = CONSTRUCT(C, τ)
    best = construct()
    best_fit = fitness_fn(best)
    fitness_evaluations += 1
    fitness_over_time = [best_fit]
    coords.append(map_bitstring(best))

    # Lines 3-5: initial pheromone update using x*
    update_pheromone(best)

    # Line 6: repeat
    while best_fit != bit_length and iterations < max_iterations:
        iterations += 1

        # Line 7: y = CONSTRUCT(C, τ)
        y = construct()
        y_fit = fitness_fn(y)
        fitness_evaluations += 1

        # Line 8: if f(y) >= f(x*) then x* = y
        if y_fit >= best_fit:
            best = y
            best_fit = y_fit

        # Lines 9-10: update pheromone using x*
        update_pheromone(best)
        fitness_over_time.append(best_fit)
        coords.append(map_bitstring(best))

    population = {"Solution": {"bit": best, "fitness": best_fit}}
    return best, iterations, 0.0, population, fitness_evaluations, coords, fitness_over_time

# TSP
def ant_colony_optimizationTSP(distance_matrix, city_coords,
                                rho=0.1, max_iterations=1000,
                                alpha=1, beta=2):
    """MMAS* for TSP (Kötzing, Neumann, Röglin, Witt).

    Algorithm 1 — MMAS* on G = (V, E):
      1: τ(e) ← 1/|V| for all e ∈ E
      2: x* ← construct(τ)
      3: update(τ, x*)
      4: while true do
      5:   x ← construct(τ)
      6:   if f(x) < f(x*) then x* ← x
      7:   τ ← update(τ, x*)

    Algorithm 2 — construct: choose edges with prob ∝ τ^α · η^β
    Update: τ'(e) = min{(1-ρ)·τ(e)+ρ, τ_max} if e ∈ E(x*), else max{(1-ρ)·τ(e), τ_min}
    """
    from optimization_framework.problems.tsp import tour_cost

    n = len(distance_matrix)
    tau_min = 1 / n
    tau_max = 1 - 1 / n

    pheromone = [[1 / n] * n for _ in range(n)]

    heuristic = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] > 0:
                heuristic[i][j] = 1.0 / distance_matrix[i][j]

    def construct():
        visited = set()
        start = random.randint(0, n - 1)
        tour = [start]
        visited.add(start)
        for _ in range(n - 1):
            current = tour[-1]
            probs = []
            for j in range(n):
                if j not in visited:
                    p = (pheromone[current][j] ** alpha) * (heuristic[current][j] ** beta)
                    probs.append((j, p))
            total = sum(p for _, p in probs)
            if total == 0:
                city = random.choice([j for j, _ in probs])
            else:
                r = random.random() * total
                cumulative = 0.0
                city = probs[-1][0]
                for j, p in probs:
                    cumulative += p
                    if cumulative >= r:
                        city = j
                        break
            tour.append(city)
            visited.add(city)
        return tour

    def update_pheromone(best_tour):
        best_edges = set()
        for k in range(len(best_tour)):
            i = best_tour[k]
            j = best_tour[(k + 1) % n]
            best_edges.add((i, j))
            best_edges.add((j, i))

        for i in range(n):
            for j in range(n):
                if (i, j) in best_edges:
                    pheromone[i][j] = min((1 - rho) * pheromone[i][j] + rho, tau_max)
                else:
                    pheromone[i][j] = max((1 - rho) * pheromone[i][j], tau_min)

    best_tour = construct()
    best_cost = tour_cost(best_tour, distance_matrix)
    fitness_evaluations = 1
    update_pheromone(best_tour)

    cost_over_time = [best_cost]
    iterations = 0

    for iteration in range(max_iterations):
        iterations += 1
        x = construct()
        x_cost = tour_cost(x, distance_matrix)
        fitness_evaluations += 1

        if x_cost < best_cost:
            best_tour = x
            best_cost = x_cost

        update_pheromone(best_tour)
        cost_over_time.append(best_cost)

    tour_coords = _tour_to_coords(best_tour, city_coords)
    population = {"Solution": {"tour": best_tour, "cost": best_cost}}
    return best_tour, iterations, 0.0, population, fitness_evaluations, tour_coords, cost_over_time


def _tour_to_coords(tour, city_coords):
    """Convert a tour (list of 0-based indices) to (x, y) in tour order."""
    if not city_coords:
        return []
    node_ids = sorted(city_coords.keys())
    if max(tour) >= len(node_ids):
        return []
    return [(city_coords[node_ids[i]][0], city_coords[node_ids[i]][1]) for i in tour]