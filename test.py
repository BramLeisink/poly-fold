import json


def generate_edges():
    num_faces = int(input("Enter the number of faces: "))
    faces = {}
    edges = []

    for face in range(1, num_faces + 1):
        faces[face] = []

    for face in range(1, num_faces + 1):
        print(f"---FACE {face}---")
        for connection in faces[face]:
            print(f"Already connected to: {connection}")
        while True:
            inpt = input("Enter a new connection (or next for next face): ")

            try:
                faces[face].append(int(inpt))
                faces[int(inpt)].append(face)
                edges.append([face, int(inpt)])
            except ValueError:
                break
    return faces, edges


if __name__ == "__main__":
    faces, edges = generate_edges()
