import csv
import random

def generate_adjacency_list_csv(
    filename: str,
    num_nodes: int,
    num_edges: int,
    min_weight: int = 1,
    max_weight: int = 10,
    directed: bool = True
):
    """
    Generates a random graph and saves it as a CSV file representing an adjacency list.
    
    :param filename: Output CSV filename
    :param num_nodes: Number of nodes in the graph
    :param num_edges: Number of edges to generate
    :param min_weight: Minimum edge weight
    :param max_weight: Maximum edge weight
    :param directed: If False, adds edges in both directions
    """
    assert num_edges <= num_nodes * (num_nodes - 1) if directed else num_nodes * (num_nodes - 1) // 2, \
        "Too many edges for the number of nodes"

    edges = set()

    while len(edges) < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(max(u-100,0), min(u+100,num_nodes-1))
        if u == v:
            continue  # Skip self-loops
        if directed:
            edge = (u, v)
        else:
            edge = tuple(sorted((u, v)))
        if edge not in edges:
            edges.add(edge)

    sorted_edges = sorted(edges, key=lambda edge: edge[0])

    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for u, v in sorted_edges:
            weight = random.randint(min_weight, max_weight)
            writer.writerow([u, v, weight])
            if not directed:
                writer.writerow([v, u, weight])  # Add reverse edge for undirected

    print(f"Graph with {num_nodes} nodes and {len(edges)} edges written to '{filename}'.")

generate_adjacency_list_csv("adjacency_list_100000.csv", num_nodes=100000, num_edges=3000000, min_weight=-50, max_weight=50, directed=True)
generate_adjacency_list_csv("adjacency_list_250000.csv", num_nodes=250000, num_edges=750000, min_weight=-50, max_weight=50, directed=True)
generate_adjacency_list_csv("adjacency_list_500000.csv", num_nodes=500000, num_edges=1500000, min_weight=-50, max_weight=50, directed=True)
generate_adjacency_list_csv("adjacency_list_750000.csv", num_nodes=750000, num_edges=2250000, min_weight=-50, max_weight=50, directed=True)