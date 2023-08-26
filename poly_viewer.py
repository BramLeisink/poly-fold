import trimesh
import numpy as np


if __name__ == "__main__":
    # Load the angel mesh. Trimesh directly detects that the mesh is textured and contains a material
    mesh_path = "/home/bramleisink/Documents/poly/models/icosahedron.obj"
    mesh = trimesh.load(mesh_path)

    # Show the scene and set a callback function, which will be used to rotate the objects
    mesh.show()
