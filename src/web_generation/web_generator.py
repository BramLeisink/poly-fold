import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import numpy as np
from stl import mesh as stl_mesh
import json
from src.mesh_import import mesh_importer
from src import utils


def kruskals_max_weight(D):
    # Create a list of all edges sorted in descending order of weight
    sorted_edges = sorted(
        D.edges(data=True), key=lambda x: x[2]["weight"], reverse=True
    )

    # Create a disjoint-set data structure to keep track of connected components
    parent = {node: node for node in D.nodes()}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            parent[root1] = root2

    # Initialize an empty graph to store the maximum weight spanning tree
    T = nx.Graph()

    for edge in sorted_edges:
        u, v, data = edge
        if find(u) != find(v):
            union(u, v)
            T.add_edge(u, v, weight=data["weight"])

    return T


def graph_of_a_mesh(mesh):
    """
    This function takes a mesh and returns a graph, where:
        V: [vertices of the mesh]
        E: [edges of the mesh]
    """
    edges = mesh.edges
    G = nx.Graph()
    for edge in edges:
        edge = list(map(int, edge))
        weight = round(
            calculate_edge_weight(
                mesh, mesh.vertices[int(edge[0])], mesh.vertices[int(edge[1])], 0, 1
            ),
            6,
        )
        G.add_edge(edge[0], edge[1], weight=weight)

    return G


def dual_graph_of_a_mesh(mesh):
    """
    This function takes a mesh and returns a dual graph, where:
        V: [faces of the mesh]
        E: [(f, g) where f and g are faces with a common edge in the mesh]
    """
    D = nx.Graph()

    for edge in mesh.edges:
        edge_faces = []
        for face_index in range(len(mesh.faces)):
            if edge[0] in mesh.faces[face_index] and edge[1] in mesh.faces[face_index]:
                edge_faces.append(face_index)
        weight = round(
            calculate_edge_weight(
                mesh, mesh.vertices[int(edge[0])], mesh.vertices[int(edge[1])], 0, 1
            ),
            6,
        )
        D.add_edge(edge_faces[0], edge_faces[1], weight=weight)

    return D


def min_spanning_tree(edges, vertices):
    result = 2 * math.ceil(0.5 * (-1 + math.sqrt(1 + 8 * (edges - vertices + 1))))
    return result


# I really hope this works, otherwise the whole algorithm is fugged.
def calculate_edge_weight(mesh, endpoint1, endpoint2, var1, var2):
    m = calculate_minimum_perimeter_weight(
        endpoint1, endpoint2, mesh.shortest_edge_length, mesh.longest_edge_length
    )
    f = calculate_flat_spanning_tree_weight(endpoint1, endpoint2)

    h = (1 - var1) * m + var2 * f
    return h


def calculate_minimum_perimeter_weight(
    endpoint1, endpoint2, shortest_edge_length, longest_edge_length
):
    edge_length = utils.calculate_distance(endpoint1, endpoint2)

    if longest_edge_length == shortest_edge_length:
        return 1
    else:
        # Calculate the weight using the formula
        weight = 1 - (edge_length - shortest_edge_length) / (
            longest_edge_length - shortest_edge_length
        )

    # Ensure the weight is within [0, 1] bounds
    weight = max(0, min(1, weight))

    return weight


def calculate_flat_spanning_tree_weight(
    endpoint1, endpoint2, reference_direction=np.array([1, 0, 0])
):
    endpoint_a, endpoint_b = endpoint1, endpoint2
    constant_direction = reference_direction
    edge_vector = np.array(endpoint_b) - np.array(endpoint_a)
    dot_product = np.dot(constant_direction, edge_vector)
    edge_length = np.linalg.norm(edge_vector)
    flat_weight = abs(dot_product) / edge_length

    return flat_weight


if __name__ == "__main__":
    # Load the angel mesh. Trimesh directly detects that the mesh is textured and contains a material
    mesh_path = "models/json/icosahedron.json"
    vertices, faces = mesh_importer.from_file(mesh_path)
    mesh = mesh_importer.Mesh(vertices, faces)

    G = graph_of_a_mesh(mesh)
    D = dual_graph_of_a_mesh(mesh)
    T = kruskals_max_weight(D)

    # Draw the graph
    # Create a figure with two subplots
    plt.figure(figsize=(10, 5))

    # Plot the first graph in the first subplot
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(
        G, k=0.2
    )  # You can adjust the 'k' parameter to control edge length
    nx.draw(G, pos, with_labels=True)
    plt.title("Graph G")

    # Plot the second graph in the second subplot
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(
        D, k=0.2
    )  # You can adjust the 'k' parameter to control edge length
    nx.draw(D, pos, with_labels=True)
    plt.title("Graph D")

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    # Show the combined plot
    plt.show()

    nx.draw(T, with_labels=True)

    plt.show()
