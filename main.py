import numpy as np
import poly
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load the angel mesh. Trimesh directly detects that the mesh is textured and contains a material
    mesh_path = (
        "models/json/cube.json"
    )
    vertices, faces = poly.from_file(mesh_path)
    mesh = poly.Mesh(vertices, faces)

    G = poly.graph_of_a_mesh(mesh)
    D = poly.dual_graph_of_a_mesh(mesh)

    # Draw the graph
    # Create a figure with two subplots
    plt.figure(figsize=(10, 5))

    # Plot the first graph in the first subplot
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    edge_labels = {(u, v): str(d["weight"]) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Graph G")

    # Plot the second graph in the second subplot
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(D)
    nx.draw(D, pos, with_labels=True, node_size=500, font_size=8)
    edge_labels = {(u, v): str(d["weight"]) for u, v, d in D.edges(data=True)}
    nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels)
    plt.title("Graph D")

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()

    # Show the combined plot
    plt.show()
