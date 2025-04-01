# Scaled variable
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl
import numpy as np
import meshio

L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

# BC dirichlet
def clamped_boundary(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# BC neumann
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
ds = ufl.Measure("ds", domain=domain)  # integral over this is integral over boundary

# Variational formulation
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

# Defining and solving the problem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

points = domain.geometry.x

# Get the connectivity of 3D cells (hexahedra) and reshape so each cell has 8 nodes.
cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape((-1, 8))

# Reshape your solution vector to have one row per node and three components (for 3D displacement).
u_array = uh.x.array.reshape((-1, 3))

# Create a meshio mesh. Here the cell type is "hexahedron" (which PyVista understands).
meshio_mesh = meshio.Mesh(points, [("hexahedron", cells)], point_data={"u": u_array})

# Write out a VTU file that PyVista can read.
meshio.write("clamped_beam.vtu", meshio_mesh)

# # Visualizing the solution
# pyvista.start_xvfb()

# # Create plotter and pyvista grid
# p = pyvista.Plotter()
# topology, cell_types, geometry = plot.vtk_mesh(V)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# # Attach vector values to grid and warp grid by vector
# grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
# actor_0 = p.add_mesh(grid, style="wireframe", color="k")
# warped = grid.warp_by_vector("u", factor=1.5)
# actor_1 = p.add_mesh(warped, show_edges=True)
# p.show_axes()
# if not pyvista.OFF_SCREEN:
#     p.show()
# else:
#     figure_as_array = p.screenshot("deflection.png")