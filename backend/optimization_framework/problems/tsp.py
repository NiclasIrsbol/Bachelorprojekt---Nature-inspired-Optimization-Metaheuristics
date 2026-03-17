import random
import requests
import tsplib95

# Small TSPLIB instances with EUC_2D coordinates (fast to solve, have city coords)
TSPLIB_INSTANCES = [
    "att48", "berlin52", "burma14", "ch130", "ch150",
    "eil51", "eil76", "eil101", "fri26", "gr17", "gr21",
    "gr24", "gr48", "gr96", "gr120", "kroA100", "kroB100",
    "kroC100", "kroD100", "kroE100", "lin105", "pr76",
    "pr107", "pr124", "pr136", "pr144", "pr152",
    "rat99", "rd100", "st70", "ulysses16", "ulysses22",
]

BASE_URL = "https://raw.githubusercontent.com/mastqe/tsplib/master"

def fetch_random_tsp_instance():
    name = random.choice(TSPLIB_INSTANCES)
    url = f"{BASE_URL}/{name}.tsp"
    response = requests.get(url, timeout=10)
    problem = tsplib95.parse(response.text)
    coords = problem.node_coords
    nodes = list(problem.get_nodes())
    distance_matrix = [
    [problem.get_weight(i, j) for j in nodes] for i in nodes]
    return name, problem, coords, nodes, distance_matrix


def tour_cost(tour, distance_matrix):
    n = len(tour)
    return sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
