from dolfinx import fem, la, mesh, plot
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

# Create mesh
length, height = 1, 1
Nx, Ny = 100, 100
extent = [[0.0, 0.0], [length, height]]
domain = mesh.create_rectangle(MPI.COMM_SELF, extent, [Nx, Ny], mesh.CellType.quadrilateral)

# # Get geometry and connectivity
# points = domain.geometry.x
# cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape((-1, 4))

# # Plot with matplotlib
# fig, ax = plt.subplots(figsize=(10, 4))
# for cell in cells:
#     quad = points[cell]
#     quad = np.vstack((quad, quad[0]))  # close the loop
#     ax.plot(quad[:, 0], quad[:, 1], color="black", linewidth=0.5)

# ax.set_aspect('equal')
# ax.set_title("Structured quadrilateral mesh")
# plt.axis("off")
# plt.savefig("mesh.png", dpi=300, bbox_inches="tight")


from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    dx,
    grad,
    inner,
    system,
)

V = fem.functionspace(domain, ("Lagrange", 1))
u = TrialFunction(V)
v = TestFunction(V)
un = fem.Function(V)
f = fem.Constant(domain, 0.0)
mu = fem.Constant(domain, 2.3)
dt = fem.Constant(domain, 0.05)

F = inner(u - un, v) * dx + dt * mu * inner(grad(u), grad(v)) * dx
F -= dt * inner(f, v) * dx
(a, L) = system(F)

import numpy as np


def uD_function(t):
    return lambda x: x[1] * np.cos(0.25 * t)


uD = fem.Function(V)
t = 0
uD.interpolate(uD_function(t))

def dirichlet_facets(x):
    return np.isclose(x[0], length)


tdim = domain.topology.dim
bc_facets = mesh.locate_entities_boundary(domain, tdim - 1, dirichlet_facets)

bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, bc_facets)

bcs = [fem.dirichletbc(uD, bndry_dofs)]

import dolfinx.fem.petsc as petsc

compiled_a = fem.form(a)
A = petsc.assemble_matrix(compiled_a, bcs=bcs)
A.assemble()

compiled_L = fem.form(L)
b = fem.Function(V)

from petsc4py import PETSc

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")


topology, cells, geometry = plot.vtk_mesh(V)
uh = fem.Function(V)

T = np.pi
i = 0
while t < T:
    # Update boundary condition
    t += dt.value
    uD.interpolate(uD_function(t))

    # Assemble RHS
    b.x.array[:] = 0
    petsc.assemble_vector(b.x.petsc_vec, compiled_L)

    # Apply boundary condition
    petsc.apply_lifting(b.x.petsc_vec, [compiled_a], [bcs])
    b.x.scatter_reverse(la.InsertMode.add)
    fem.petsc.set_bc(b.x.petsc_vec, bcs)

    # Solve linear problem
    solver.solve(b.x.petsc_vec, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update un
    un.x.array[:] = uh.x.array
    # print( uh.x.array)

    # Save a snapshot every 10 iterations
    if i % 10 == 0:
        # Get nodal coordinates and cell connectivity from the mesh
        points = domain.geometry.x
        cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape((-1, 4))

        # Convert each quadrilateral cell into two triangles
        triangles = []
        for quad in cells:
            triangles.append([quad[0], quad[1], quad[2]])
            triangles.append([quad[0], quad[2], quad[3]])
        triangles = np.array(triangles)

        # Create a triangulation object
        triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)

        # Plot the solution field (uh.x.array) using tricontourf
        plt.figure(figsize=(10, 4))
        contour = plt.tricontourf(triang, uh.x.array, cmap="viridis")
        plt.colorbar(contour)
        plt.title(f"Solution at time t = {t:.2f}")
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(f"solution_{t}.png", dpi=300, bbox_inches="tight")
        plt.close()
    i += 1
    # Update plotter
    # plotter.update_scalars(uh.x.array, render=False)
    # plotter.write_frame()