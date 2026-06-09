import os
import random
import tsplib95

TSPLIB_DIR = os.path.join(os.path.dirname(__file__), "tsplib")

# TSPLIB instances with NODE_COORD_SECTION and <= 300 cities
TSPLIB_INSTANCES = [
    "a280", "att48", "berlin52", "bier127", "burma14",
    "ch130", "ch150", "d198", "eil101", "eil51", "eil76",
    "gil262", "gr137", "gr202", "gr229", "gr96",
    "kroA100", "kroA150", "kroA200", "kroB100", "kroB150",
    "kroB200", "kroC100", "kroD100", "kroE100", "lin105",
    "pr107", "pr124", "pr136", "pr144", "pr152", "pr226",
    "pr264", "pr299", "pr76", "rat195", "rat99", "rd100",
    "st70", "ts225", "tsp225", "u159", "ulysses16", "ulysses22",
]

# --- Commented out for now: large instances (> 300 cities) for benchmarking later ---
# "ali535", "att532", "d1291", "d1655", "d2103", "d493", "d657",
# "dsj1000", "fl1400", "fl1577", "fl3795", "fl417", "fnl4461",
# "gr431", "gr666", "lin318", "linhp318", "nrw1379", "p654",
# "pcb1173", "pcb3038", "pcb442", "pr1002", "pr2392", "pr439",
# "rat575", "rat783", "rd400", "rl1304", "rl1323", "rl1889",
# "rl5915", "rl5934", "rl11849", "u574", "u724", "u1060",
# "u1432", "u1817", "u2152", "u2319", "vm1084", "vm1748",
# "brd14051", "d15112", "d18512", "pla7397", "pla33810",
# "pla85900", "usa13509"
#
# --- Commented out: no NODE_COORD_SECTION (can't visualize) ---
# "bayg29", "bays29", "brazil58", "brg180", "dantzig42",
# "fri26", "gr17", "gr21", "gr24", "gr48", "gr120",
# "hk48", "swiss42", "pa561", "si175", "si535", "si1032"

def fetch_tsp_instance(instance_name: str = None):
    """
    Fetch a specific TSP instance or a random one if instance_name is None.
    
    Args:
        instance_name: Name of the TSP instance (without .tsp extension). 
                      If None, a random instance is chosen.
    
    Returns:
        Tuple of (name, problem, coords, nodes, distance_matrix)
    """
    if instance_name is None:
        name = random.choice(TSPLIB_INSTANCES)
    else:
        if instance_name not in TSPLIB_INSTANCES:
            raise ValueError(f"Unknown TSP instance: {instance_name}")
        name = instance_name
    
    filepath = os.path.join(TSPLIB_DIR, f"{name}.tsp")

    with open(filepath) as f:
        text = f.read()

    problem = tsplib95.parse(text)
    coords = problem.node_coords
    nodes = list(problem.get_nodes())
    distance_matrix = [
        [problem.get_weight(i, j) for j in nodes] for i in nodes
    ]
    return name, problem, coords, nodes, distance_matrix


def fetch_random_tsp_instance():
    """Fetch a random TSP instance (deprecated - use fetch_tsp_instance instead)"""
    return fetch_tsp_instance(None)


def get_tsp_instances_metadata():
    """
    Get metadata for all available TSP instances.
    
    Returns:
        List of dicts with 'name' and 'num_cities' for each instance.
    """
    metadata = []
    for instance_name in TSPLIB_INSTANCES:
        try:
            filepath = os.path.join(TSPLIB_DIR, f"{instance_name}.tsp")
            with open(filepath) as f:
                text = f.read()
            problem = tsplib95.parse(text)
            num_cities = len(list(problem.get_nodes()))
            metadata.append({
                "name": instance_name,
                "num_cities": num_cities
            })
        except Exception as e:
            # Skip instances that can't be parsed
            print(f"Warning: Could not parse {instance_name}: {e}")
            continue
    
    # Sort by number of cities, then by name
    metadata.sort(key=lambda x: (x["num_cities"], x["name"]))
    return metadata


def tour_cost(tour, distance_matrix):
    n = len(tour)
    return sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


def tour_to_coords(tour, city_coords):
    """Convert a tour (list of 0-based indices) to (x, y) points in tour order.

    ``city_coords`` maps TSPLIB node ids to (x, y); nodes are taken in sorted-id
    order, so index ``i`` in the tour maps to the i-th city. Returns ``[]`` if the
    coordinates are missing or a tour index is out of range. Shared by all TSP
    solvers for building the tour-map visualization.
    """
    if not city_coords:
        return []
    node_ids = sorted(city_coords.keys())
    if max(tour) >= len(node_ids):
        return []
    return [(city_coords[node_ids[i]][0], city_coords[node_ids[i]][1]) for i in tour]


# Published optimal tour lengths for TSPLIB instances (integer-rounded weights,
# as used here). Source: TSPLIB optimal-solutions tables. Instances not listed
# return None from get_optimum/gap_percent.
TSPLIB_OPTIMA = {
    "burma14": 3323, "ulysses16": 6859, "ulysses22": 7013, "att48": 10628,
    "eil51": 426, "berlin52": 7542, "st70": 675, "eil76": 538, "pr76": 108159,
    "rat99": 1211, "kroA100": 21282, "kroB100": 22141, "kroC100": 20749,
    "kroD100": 21294, "kroE100": 22068, "rd100": 7910, "eil101": 629,
    "lin105": 14379, "pr107": 44303, "pr124": 59030, "bier127": 118282,
    "ch130": 6110, "pr136": 96772, "pr144": 58537, "ch150": 6528,
    "kroA150": 26524, "kroB150": 26130, "pr152": 73682, "u159": 42080,
    "rat195": 2323, "d198": 15780, "kroA200": 29368, "kroB200": 29437,
    "ts225": 126643, "tsp225": 3916, "pr226": 80369, "gil262": 2378,
    "pr264": 49135, "a280": 2579, "pr299": 48191,
}


def get_optimum(instance_name):
    """Return the known optimal tour length for an instance, or None if unknown."""
    return TSPLIB_OPTIMA.get(instance_name)


def gap_percent(cost, instance_name):
    """Percentage above the known optimum: (cost - opt) / opt * 100, or None."""
    opt = TSPLIB_OPTIMA.get(instance_name)
    if opt is None or opt <= 0:
        return None
    return (cost - opt) / opt * 100.0
