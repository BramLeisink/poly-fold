from src import utils
import json
import os

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

        self.shortest_edge_length, self.longest_edge_length = None, None

        utils.edge_length_xtremes(self)


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
