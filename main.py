import numpy as np
import poly
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load the angel mesh. Trimesh directly detects that the mesh is textured and contains a material
    mesh_path = "models/cube.obj"
    vertices = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    ]
    faces = [
        (1, 5, 7, 3),
        (1, 3, 4, 2),
        (1, 2, 6, 5),
        (2, 4, 8, 6),
        (3, 7, 8, 4),
        (5, 6, 8, 7),
    ]
    mesh = poly.Mesh(vertices, faces, poly.Mesh.calculate_edges(faces))

    G = poly.graph_of_a_mesh(mesh)
    D = poly.dual_graph_of_a_mesh(mesh)

    # Draw the graph
    nx.draw_spring(D, with_labels=True)
    plt.show()
