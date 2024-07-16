import numpy as np
import pymesh


vertices = np.array([[0.0, 0.0],
[1.0, 0.0],
[1.0, 1.0],
[0.0, 1.0],
])
tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.05
tri.split_boundary = False
tri.verbosity = 0
tri.run() # Execute triangle.
mesh = tri.mesh # output triangulation.

