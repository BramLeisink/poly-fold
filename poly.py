import networkx as nx
import matplotlib.pyplot as plt
import math
import trimesh

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices  # list of the position for each vertex
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

def graph_of_a_mesh(mesh):
    """
    This function takes a mesh and returns a graph, where:
        V: [vertices of the mesh]
        E: [edges of the mesh]
    """

    G = nx.from_edgelist(mesh.edges)

    return G


# NOT WORKING!
def dual_graph_of_a_mesh(mesh):
    """
    This function takes a mesh and returns a dual graph, where:
        V: [faces of the mesh]
        E: [(f, g) where f and g are faces with a common edge in the mesh]
    """

    face_connections = []
    for edge in mesh.edges:
        polygons = [[int(idx) for idx in poly] for poly in mesh.faces]
        connected_faces = []
        for face_index in range(len(polygons)):
            face_score = 0
            for vertex in mesh.faces[face_index]:
                if edge[0] == vertex or edge[1] == vertex:
                    face_score += 1
            if face_score == 2:
                connected_faces.append(face_index)
        face_connections.append(connected_faces)

    D = nx.from_edgelist(face_connections)

    return D


def min_spanning_tree(edges, vertices):
    result = 2 * math.ceil(0.5 * (-1 + math.sqrt(1 + 8 * (edges - vertices + 1))))
    return result




if __name__ == "__main__":
    vertices = []
    faces = [
        (1, 5, 7, 3),
        (1, 3, 4, 2),
        (1, 2, 6, 5),
        (2, 4, 8, 6),
        (3, 7, 8, 4),
        (5, 6, 8, 7),
    ]
    mesh = Mesh(vertices, faces)

    print(mesh.edges)
    print(mesh.faces)
