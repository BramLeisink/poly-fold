import math


def graph_of_a_mesh(mesh):
    """
    This function takes a mesh and returns a graph, where:
        V: [vertices of the mesh]
        E: [edges of the mesh]
    """

    return


def dual_graph_of_a_mesh(mesh):
    """
    This function takes a mesh and returns a dual graph, where:
        V: [faces of the mesh]
        E: [(f, g) where f and g are faces with a common edge in the mesh]
    """

    return


def min_spanning_tree(edges, vertices):
    result = 2 * math.ceil(0.5 * (-1 + math.sqrt(1 + 8 * (edges - vertices + 1))))
    return result
