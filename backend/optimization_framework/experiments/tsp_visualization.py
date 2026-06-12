# @author: Niclas Søe Irsbøl
import matplotlib
matplotlib.use("Agg")

import base64
import io

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx


def generate_tour_map(city_coords, best_tour, distance_matrix, instance_name=""):
    """Return a base64-encoded PNG of the TSP tour visualised with networkx.

    Parameters
    ----------
    city_coords : dict[str, list[float]]
        Mapping of ``"0"``, ``"1"``, … to ``[x, y]`` coordinates.
    best_tour : list[int]
        Ordered list of city indices forming the tour.
    distance_matrix : list[list[float]]
        Full pairwise distance matrix.
    instance_name : str, optional
        Name shown as the plot title.

    Returns
    -------
    str | None
        Base64-encoded PNG string, or *None* for very large instances.
    """
    n = len(best_tour)
    if n > 2000:
        return None

    # -- build graph --------------------------------------------------------
    G = nx.Graph()
    pos = {}
    for idx in range(n):
        key = str(best_tour[idx])
        coord = city_coords.get(key, city_coords.get(best_tour[idx]))
        pos[best_tour[idx]] = (coord[0], coord[1])
        G.add_node(best_tour[idx])

    edges = []
    distances = []
    for i in range(n):
        u, v = best_tour[i], best_tour[(i + 1) % n]
        G.add_edge(u, v)
        edges.append((u, v))
        distances.append(distance_matrix[u][v])

    # -- edge colours & widths ----------------------------------------------
    min_d = min(distances)
    max_d = max(distances)
    rng = max_d - min_d if max_d != min_d else 1.0

    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=min_d, vmax=max_d)
    edge_colors = [cmap(norm(d)) for d in distances]
    edge_widths = [0.8 + 1.7 * (1 - (d - min_d) / rng) for d in distances]

    # -- figure setup (dark theme) ------------------------------------------
    bg = "#1a1a2e"
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    # -- draw edges ---------------------------------------------------------
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color=edge_colors,
        width=edge_widths, alpha=0.85, ax=ax,
    )

    # -- draw regular nodes -------------------------------------------------
    regular_nodes = [node for node in best_tour[1:-1]]
    nx.draw_networkx_nodes(
        G, pos, nodelist=regular_nodes,
        node_size=20, node_color="white", alpha=0.55, ax=ax,
    )

    # -- draw start node ----------------------------------------------------
    nx.draw_networkx_nodes(
        G, pos, nodelist=[best_tour[0]],
        node_size=100, node_color="#646cff", edgecolors="white",
        linewidths=1.2, ax=ax,
    )
    start_x, start_y = pos[best_tour[0]]
    ax.annotate(
        "start", (start_x, start_y),
        textcoords="offset points", xytext=(0, 10),
        ha="center", fontsize=9, color="#646cff", fontweight="bold",
    )

    # -- draw end node -------------------------------------------------------
    nx.draw_networkx_nodes(
        G, pos, nodelist=[best_tour[-1]],
        node_size=100, node_color="#34d399", edgecolors="white",
        linewidths=1.2, ax=ax,
    )
    end_x, end_y = pos[best_tour[-1]]
    ax.annotate(
        "end", (end_x, end_y),
        textcoords="offset points", xytext=(0, -15),
        ha="center", fontsize=9, color="#34d399", fontweight="bold",
    )

    # -- city labels (small instances only) ---------------------------------
    if n <= 50:
        labels = {node: str(node) for node in best_tour}
        nx.draw_networkx_labels(
            G, pos, labels, font_size=7, font_color="white", ax=ax,
        )

    # -- colour-bar legend --------------------------------------------------
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.04, pad=0.02, aspect=40)
    cbar.set_label("Edge distance (short → long)", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white", labelcolor="white")

    # -- title --------------------------------------------------------------
    if instance_name:
        ax.set_title(instance_name, color="white", fontsize=14, pad=12)

    # -- export -------------------------------------------------------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
