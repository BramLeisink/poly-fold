import numpy as np
from src.web_generation import web_generator
from src.mesh_import import mesh_importer
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load the angel mesh. Trimesh directly detects that the mesh is textured and contains a material
    mesh_path = "data/json/truncated-cube.json"
    vertices, faces = mesh_importer.from_file(mesh_path)
    mesh = mesh_importer.Mesh(vertices, faces)

    print(vertices)

    G = web_generator.graph_of_a_mesh(mesh)
    D = web_generator.dual_graph_of_a_mesh(mesh)
    T = web_generator.kruskals_max_weight(D)

    # Draw the graph
    # Create a figure with two subplots
    plt.figure(figsize=(10, 5))

    # Plot the first graph in the first subplot
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, k=0.2)  # You can adjust the 'k' parameter to control edge length
    nx.draw(G, pos, with_labels=True)
    plt.title("Graph G")

    # Plot the second graph in the second subplot
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(D, k=0.2)  # You can adjust the 'k' parameter to control edge length
    nx.draw(D, pos, with_labels=True)
    plt.title("Graph D")

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    # Show the combined plot
    plt.show()

    nx.draw(T, with_labels=True)

    plt.show()
