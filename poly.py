import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import numpy as np
from stl import mesh as stl_mesh
import json


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices  # list of the position for each vertex
        print(f"vertices: {self.vertices}")
        self.faces = faces  # list of the indices for each face

        self.edges = []  # list of the indices for each edge
        for face in self.faces:
            for vertex_index in range(len(face)):
                if self.edges.__contains__(
                    tuple(
                        sorted(
                            [
                                face[vertex_index],
                                face[vertex_index + 1]
                                if vertex_index + 1 < len(face)
                                else face[0],
                            ]
                        )
                    )
                ):
                    continue
                else:
                    self.edges.append(
                        tuple(
                            sorted(
                                [
                                    face[vertex_index],
                                    face[vertex_index + 1]
                                    if vertex_index + 1 < len(face)
                                    else face[0],
                                ]
                            )
                        )
                    )
        self.edges = sorted(self.edges)


def calculate_distance(point1, point2, round_to=6):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    distance = round(distance, round_to)
    return distance


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
        distance = calculate_distance(mesh.vertices[edge[0]], mesh.vertices[edge[1]])
        edge.append(distance)
        G.add_edge(edge[0], edge[1], weight=distance)

    return G


def dual_graph_of_a_mesh(mesh):
    """
    This function takes a mesh and returns a dual graph, where:
        V: [faces of the mesh]
        E: [(f, g) where f and g are faces with a common edge in the mesh]
    """

    face_connections = []
    for edge in mesh.edges:
        connected_faces = []
        for face_index in range(len(mesh.faces)):
            if edge[0] in mesh.faces[face_index] and edge[1] in mesh.faces[face_index]:
                connected_faces.append(face_index)
        if len(connected_faces) > 1:
            face_connections.append(connected_faces)
        else:
            print(
                "You somehow got a mesh, that contains an edge with just one plane attached."
            )
            quit()
    D = nx.Graph()
    for edge in face_connections:
        D.add_edge(edge[0], edge[1], weight=calculate_edge_weight(mesh, edge))

    return D

# Function not working correctly.
def calculate_edge_weight(mesh, edge):
    """
    This function takes a mesh and an edge and returns the weight of the edge.
    The weight is based on the sum of the lengths of the two edges in the corresponding faces.
    """
    face1 = mesh.faces[edge[0]]
    face2 = mesh.faces[edge[1]]

    edge_indices = [(face1[i], face1[(i + 1) % len(face1)]) for i in range(len(face1))]
    edge_indices += [(face2[i], face2[(i + 1) % len(face2)]) for i in range(len(face2))]

    edge_count = {}
    for edge in edge_indices:
        # Sort the edge to ensure order doesn't affect the comparison
        sorted_edge = tuple(sorted(edge))

        if sorted_edge in edge_count:
            duplicate_edge = sorted_edge
        else:
            edge_count[sorted_edge] = 1
    vertex1 = mesh.vertices[int(duplicate_edge[0]) - 1]
    vertex2 = mesh.vertices[int(duplicate_edge[1]) - 1]
    edge_length = calculate_distance(vertex1, vertex2)
    return round(edge_length, 6)


def min_spanning_tree(edges, vertices):
    result = 2 * math.ceil(0.5 * (-1 + math.sqrt(1 + 8 * (edges - vertices + 1))))
    return result


# This function needs to be fixed. It terrible. (little less terrible now.)
def from_file(path):
    file_extension = os.path.splitext(path)[1]
    vertices = []
    faces = []

    if file_extension == ".obj":
        print(path)
        with open(path) as file:
            for line in file:
                if line.startswith("v "):
                    vertices.append([float(coord) for coord in line.split()[1:]])
                elif line.startswith("f "):
                    face = []
                    for connection in [coord for coord in line.split()[1:]]:
                        face.append(int(connection.split("/")[0]))
                    faces.append(face)
        return vertices, faces
    elif file_extension == ".json":
        with open(path) as file:
            data = json.load(file)
        constants = data["constants"]
        vertices_const = data["vertices"]
        faces = data["faces"]
        vertices = []

        for vertex_constant in vertices_const:
            vertex = []
            for coordinate in vertex_constant:
                if str(coordinate).__contains__("C"):
                    if coordinate.startswith("-"):
                        negative = True
                    constant = round(
                        constants[int(coordinate.strip("-").strip("C"))][0], 6
                    )
                    if coordinate[0] == "-":
                        coordinate = constant * -1
                    else:
                        coordinate = constant
                else:
                    coordinate = float(coordinate)
                vertex.append(coordinate)
            vertices.append(vertex)
        return vertices, faces
    else:
        print(f"Error: Unsupported file extension {file_extension}")
        return


if __name__ == "__main__":
    # Load the angel mesh. Trimesh directly detects that the mesh is textured and contains a material
    mesh_path = "models/json/icosahedron.json"
    vertices, faces = from_file(mesh_path)
    mesh = Mesh(vertices, faces)

    G = graph_of_a_mesh(mesh)
    D = dual_graph_of_a_mesh(mesh)

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
    nx.draw(D, pos, with_labels=True)
    edge_labels = {(u, v): str(d["weight"]) for u, v, d in D.edges(data=True)}
    nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels)
    plt.title("Graph D")

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()

    # Show the combined plot
    plt.show()
