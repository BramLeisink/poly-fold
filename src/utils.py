import numpy as np
import math


def edge_length_xtremes(mesh):
    for edge in mesh.edges:
        length = calculate_distance(
            mesh.vertices[int(edge[0])], mesh.vertices[int(edge[1])]
        )
        if mesh.shortest_edge_length is None or length < mesh.shortest_edge_length:
            mesh.shortest_edge_length = length
        if mesh.longest_edge_length is None or length > mesh.longest_edge_length:
            mesh.longest_edge_length = length


def calculate_distance(endpoint1, endpoint2, round_to=6):
    x1, y1, z1 = endpoint1
    x2, y2, z2 = endpoint2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    distance = round(distance, round_to)
    return distance


def calculate_angle(a, b, c, round_to=6):
    # Define your three points as numpy arrays
    point1 = np.array(a)
    point2 = np.array(b)
    point3 = np.array(c)

    # Calculate vectors A and B
    vector_A = point2 - point1
    vector_B = point3 - point2

    # Calculate the dot product of A and B
    dot_product = np.dot(vector_A, vector_B)

    # Calculate the magnitudes of A and B
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)

    # Calculate the angle in radians
    cos_theta = dot_product / (magnitude_A * magnitude_B)
    angle_rad = np.arccos(cos_theta)

    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    return round(angle_deg, round_to)
