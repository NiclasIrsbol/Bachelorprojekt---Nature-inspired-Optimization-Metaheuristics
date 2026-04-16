import random
from optimization_framework.operators.gaoperators import map_bitstring
from optimization_framework.problems.tsp import tour_cost

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
def ant_colony_optimizationTSP(distance_matrix, city_coords, rho=0.1, max_iterations=1000, alpha=1, beta=2):
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

    for _ in range(max_iterations):
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


# ============================================================================
# Population-Based ACO (PACO)
# ============================================================================

# Bitstrings
def population_based_aco(fitness_fn, bit_length=100, archive_size=10, max_iterations=10000, num_ants=30, q0=0.9, beta=2):
    """Population-Based ACO for pseudo-Boolean optimisation.
    
    Uses a fixed-size archive of best solutions instead of a pheromone matrix.
    Pheromone is derived from how many archive solutions use each bit position.
    
    Args:
        fitness_fn: Function to evaluate solutions
        bit_length: Length of bitstring
        archive_size: Max size of solution archive (K)
        max_iterations: Max iterations
        num_ants: Number of ants per iteration
        q0: Probability of greedy selection (0.9 = 90% greedy)
        beta: Heuristic exponent (not used for bitstrings, kept for API compatibility)
    """
    tau0 = 0.5
    delta = 0.1
    
    archive = []
    count = [0] * bit_length
    iterations = 0
    fitness_evaluations = 0
    best_fit = -1
    best_solution = None
    
    fitness_over_time = []
    coords = []
    
    def get_pheromone(j):
        """Calculate pheromone for bit j"""
        return tau0 + delta * count[j]
    
    def construct_solution():
        """Construct a solution using pheromone and randomness"""
        bits = []
        for j in range(bit_length):
            tau = get_pheromone(j)
            if random.random() < q0:
                # Greedy: choose 1 if pheromone > 0.5, else choose 0
                bits.append("1" if tau > 0.5 else "0")
            else:
                # Probabilistic: choose 1 with probability tau
                bits.append("1" if random.random() < tau else "0")
        return "".join(bits)
    
    # Initial solutions
    for _ in range(num_ants):
        sol = construct_solution()
        fit = fitness_fn(sol)
        fitness_evaluations += 1
        
        if fit > best_fit:
            best_fit = fit
            best_solution = sol
        
        archive.append((sol, fit))
    
    # Sort archive by fitness and keep top archive_size
    archive.sort(key=lambda x: x[1], reverse=True)
    archive = archive[:archive_size]
    
    # Initialize count from archive
    count = [0] * bit_length
    for sol, _ in archive:
        for j in range(bit_length):
            if sol[j] == "1":
                count[j] += 1
    
    fitness_over_time.append(best_fit)
    coords.append(map_bitstring(best_solution))
    
    # Main loop
    while best_fit != bit_length and iterations < max_iterations:
        iterations += 1
        
        # Construct solutions
        new_solutions = []
        for _ in range(num_ants):
            sol = construct_solution()
            fit = fitness_fn(sol)
            fitness_evaluations += 1
            new_solutions.append((sol, fit))
            
            if fit > best_fit:
                best_fit = fit
                best_solution = sol
        
        # Select iteration-best
        iter_best = max(new_solutions, key=lambda x: x[1])
        
        # Update archive: add iteration-best
        archive.append(iter_best)
        archive.sort(key=lambda x: x[1], reverse=True)
        
        # Remove oldest if archive exceeds size
        removed = None
        if len(archive) > archive_size:
            removed = archive.pop()
        
        # Update count
        for j in range(bit_length):
            if iter_best[0][j] == "1":
                count[j] += 1
        
        if removed:
            for j in range(bit_length):
                if removed[0][j] == "1":
                    count[j] -= 1
        
        fitness_over_time.append(best_fit)
        coords.append(map_bitstring(best_solution))
    
    population = {"Solution": {"bit": best_solution, "fitness": best_fit}}
    return best_solution, iterations, 0.0, population, fitness_evaluations, coords, fitness_over_time


# TSP
def population_based_acoTSP(distance_matrix, city_coords, archive_size=10, max_iterations=1000, num_ants=30, alpha=1, beta=2, q0=0.9):
    """Population-Based ACO for TSP.
    
    Uses a fixed-size archive of best tours instead of a pheromone matrix.
    Pheromone is derived from how many archive tours use each edge.
    
    Args:
        distance_matrix: Distance matrix
        city_coords: City coordinates for visualization
        archive_size: Max size of tour archive (K)
        max_iterations: Max iterations
        num_ants: Number of ants per iteration
        alpha: Pheromone exponent
        beta: Heuristic exponent
        q0: Probability of greedy selection
    """
    n = len(distance_matrix)
    tau0 = 1.0 / n
    delta = 0.1
    
    # Precompute heuristic
    heuristic = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] > 0:
                heuristic[i][j] = 1.0 / distance_matrix[i][j]
    
    # Archive: list of (tour, cost)
    archive = []
    
    # Count edges: count[i][j] = how many archive tours use edge i->j
    count = [[0] * n for _ in range(n)]
    
    iterations = 0
    fitness_evaluations = 1
    best_cost = float('inf')
    best_tour = None
    
    cost_over_time = []
    
    def get_pheromone(i, j):
        """Calculate pheromone for edge (i, j)"""
        return tau0 + delta * count[i][j]
    
    def construct_tour():
        """Construct a tour using pheromone and heuristic"""
        visited = set()
        start = random.randint(0, n - 1)
        tour = [start]
        visited.add(start)
        
        for _ in range(n - 1):
            current = tour[-1]
            
            if random.random() < q0:
                # Greedy: choose edge with highest tau * eta
                best_j = -1
                best_val = -1.0
                for j in range(n):
                    if j not in visited:
                        val = (get_pheromone(current, j) ** alpha) * (heuristic[current][j] ** beta)
                        if val > best_val:
                            best_val = val
                            best_j = j
                next_city = best_j
            else:
                # Probabilistic: select based on tau * eta
                probs = []
                for j in range(n):
                    if j not in visited:
                        p = (get_pheromone(current, j) ** alpha) * (heuristic[current][j] ** beta)
                        probs.append((j, p))
                
                total = sum(p for _, p in probs)
                if total == 0:
                    next_city = random.choice([j for j, _ in probs])
                else:
                    r = random.random() * total
                    cumulative = 0.0
                    next_city = probs[-1][0]
                    for j, p in probs:
                        cumulative += p
                        if cumulative >= r:
                            next_city = j
                            break
            
            tour.append(next_city)
            visited.add(next_city)
        
        return tour
    
    # Initial population
    for _ in range(num_ants):
        tour = construct_tour()
        cost = tour_cost(tour, distance_matrix)
        fitness_evaluations += 1
        
        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]
        
        archive.append((tour, cost))
    
    # Sort and trim archive
    archive.sort(key=lambda x: x[1])
    archive = archive[:archive_size]
    
    # Initialize edge counts
    count = [[0] * n for _ in range(n)]
    for tour, _ in archive:
        for k in range(len(tour)):
            i = tour[k]
            j = tour[(k + 1) % n]
            count[i][j] += 1
    
    cost_over_time.append(best_cost)
    
    # Main loop
    while iterations < max_iterations:
        iterations += 1
        
        # Construct new solutions
        new_tours = []
        for _ in range(num_ants):
            tour = construct_tour()
            cost = tour_cost(tour, distance_matrix)
            fitness_evaluations += 1
            new_tours.append((tour, cost))
            
            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]
        
        # Select iteration-best
        iter_best = min(new_tours, key=lambda x: x[1])
        
        # Update archive
        archive.append(iter_best)
        archive.sort(key=lambda x: x[1])
        
        # Remove oldest if exceeds size
        removed = None
        if len(archive) > archive_size:
            removed = archive.pop()
        
        # Update edge counts
        for k in range(len(iter_best[0])):
            i = iter_best[0][k]
            j = iter_best[0][(k + 1) % n]
            count[i][j] += 1
        
        if removed:
            for k in range(len(removed[0])):
                i = removed[0][k]
                j = removed[0][(k + 1) % n]
                count[i][j] -= 1
        
        cost_over_time.append(best_cost)
    
    tour_coords = _tour_to_coords(best_tour, city_coords)
    population = {"Solution": {"tour": best_tour, "cost": best_cost}}
    return best_tour, iterations, 0.0, population, fitness_evaluations, tour_coords, cost_over_time