"""
Voor nu lijkt het er op dat CGAL / Delaunau_triangulation_3 een mooie storage/data structure zijn
voor het opslaan van een getrianguleerde mesh.

Dus je kan er makkelijk punten in kwijt, net zoals edges e.d.
Maar ik zie nog geen methods om van een bepaalde set aan punten een triangulatie te maken..

"""
import sys
# This is where the installation is put....
# I could fix this..
# But this also works
sys.path.append('/home/bugger/PycharmProjects/cgal-swig-bindings/examples/python')
import CGAL

import scipy.io
import helper.plot_class as hplotc
import h5py
ddata = '/home/bugger/Documents/data/vessel_meshing/ModelAU.mat'
import numpy as np
with h5py.File(ddata, 'r') as f:
    A = np.array(f['Imagen_binary'][450:500, 250:500, 200:500])

hplotc.SlidingPlot(A)

from CGAL.CGAL_Kernel import Point_3, Point_2
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Cell_handle
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Vertex_handle
from CGAL.CGAL_Triangulation_3 import Ref_Locate_type_3
from CGAL.CGAL_Triangulation_3 import VERTEX
from CGAL.CGAL_Kernel import Ref_int
#
# # Following example simple_triangulation_3.py
# # under /home/bugger/PycharmProjects/cgal-swig-bindings/examples/python
# list_of_points = []
# for i_point in np.argwhere(A):
#     res = list(i_point.astype(int))
#     point_3d = Point_3(int(res[0]), int(res[1]), int(res[2]))
#     list_of_points.append(point_3d)
#
# # Geen idee wat dit moet voorstellen eigenlijk..
# T = Delaunay_triangulation_3(list_of_points)


"""
Run the full example simple_triangulation_3.py
"""

L = []
L.append(Point_3(0, 0, 0))
L.append(Point_3(1, 0, 0))
L.append(Point_3(0, 1, 0))

T = Delaunay_triangulation_3(L)

n = T.number_of_vertices()

V = []
V.append(Point_3(0, 0, 1))
V.append(Point_3(1, 1, 1))
V.append(Point_3(2, 2, 2))

n = n + T.insert(V)

assert n == 6
assert T.is_valid()

lt = Ref_Locate_type_3()
li = Ref_int()
lj = Ref_int()
p = Point_3(0, 0, 0)

c = T.locate(p, lt, li, lj)
assert lt.object() == VERTEX
assert c.vertex(li.object()).point() == p

v = c.vertex((li.object() + 1) & 3)
nc = c.neighbor(li.object())

nli = Ref_int()
assert nc.has_vertex(v, nli)

T.write_to_file("output", 14)

T1 = Delaunay_triangulation_3()
T1.read_from_file("output")

assert T1.is_valid()
assert T1.number_of_vertices() == T.number_of_vertices()
assert T1.number_of_cells() == T.number_of_cells()

# Trying my own 2D case
from CGAL.CGAL_Triangulation_2 import Triangulation_2, Delaunay_triangulation_2
V = []
V.append(Point_2(0, 0))
V.append(Point_2(0, 1))
V.append(Point_2(1, 0))

tri_obj = Delaunay_triangulation_2()
tri_obj.insert(V[0])
tri_obj.insert(V[1])
tri_obj.insert(V[2])

dir(tri_obj)
tri_obj.number_of_vertices()