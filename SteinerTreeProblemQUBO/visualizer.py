import networkx as nx
import matplotlib.pyplot as plt
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.random_problem_generator import (
    generate_geometric_steiner_tree,
    generate_erdos_renyi_steiner_tree,
)


def draw_steiner_tree(problem: SteinerTree) -> None:
    G = nx.Graph()

    for node in problem.nodes:
        G.add_node(node)

    for u, v, w in problem.edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    terminal_set = set(problem.terminals)
    terminal_nodes = [node for node in problem.nodes if node in terminal_set]
    normal_nodes = [node for node in problem.nodes if node not in terminal_set]

    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, node_size=700, node_color="red")

    nx.draw_networkx_edges(G, pos, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10)

    edge_labels = {(u, v): w for u, v, w in problem.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title("Steiner Tree Instance")
    plt.axis("off")
    plt.show()

problem = generate_geometric_steiner_tree(
                        node_count=8,
                        terminal_count=3,
                        max_weight=100,
                        connectivity="knn",
                        k=8,
                        seed=1,
                    )
"""problem = generate_erdos_renyi_steiner_tree(
                        node_count=8,
                        terminal_count=4,
                        edge_probability=0.3,
                        weight_range=(1, 100),
                        seed=1,
                    )"""
draw_steiner_tree(problem)