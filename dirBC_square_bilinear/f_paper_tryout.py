import petsc4py
petsc4py.init()
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from basix.ufl import element
from dolfinx.cpp.mesh import CellType
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, nabla_grad, sym)
from dolfinx.fem import (Constant, Function, functionspace, dirichletbc, form,
                          locate_dofs_topological)
import dolfinx.fem as fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time

# -------------------------------
# Parameters and time settings
# -------------------------------
t = 0.0
T = 1.0
L_val = 1.0
grid_size = 200
num_steps = int(5 * grid_size)
dt_led = T / num_steps
dx_led = L_val / grid_size

c_k_led = 1.4
c_mu_led = 0.1

# -------------------------------
# Manufactured Dirichlet BC and body load functions
# -------------------------------
def u_dirBC(x, t):
    # Manufactured Dirichlet boundary condition.
    values = np.zeros((2, x.shape[1]))
    values[0] = np.sin(4.*np.pi*x[0]) * np.sin(2.*np.pi*x[1]) * np.sin(4.*np.pi*(t-0.1))
    values[1] = np.sin(4.*np.pi*x[0]) * np.sin(2.*np.pi*x[1]) * np.sin(4.*np.pi*(t+0.3))
    return values

def velocity_init(X, t):
    values = np.zeros((2, X.shape[1]))
    x = X[0]
    y = X[1]
    values[0] = 4.*np.pi*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.cos(4.*np.pi*(t - 1./10.))
    values[1] = 4.*np.pi*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.cos(4.*np.pi*(t + 3./10.))
    return values

def u_dir_BC_time(t):
    return lambda x: u_dirBC(x, t)

def body_load(X, t):
    # Manufactured body load (your expression)
    values = np.zeros((2, X.shape[1]))
    # x = X[0]
    # y = X[1]
    # values[0] = (c_mu_led*(8.*np.pi**2.*np.cos(4.*np.pi*x)*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.)) + 16.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.))) - c_mu_led*(8.*np.pi**2.*np.cos(4.*np.pi*x)*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.)) - 4.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.))) - c_k_led*(8.*np.pi**2.*np.cos(4.*np.pi*x)*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.)) - 16.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.))) - 16.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)))
    # values[1] = (c_mu_led*(8.*np.pi**2.*np.cos(4.*np.pi*x)*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)) + 4.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.))) - c_k_led*(8.*np.pi**2.*np.cos(4.*np.pi*x)*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)) - 4.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.))) - c_mu_led*(8.*np.pi**2.*np.cos(4.*np.pi*x)*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)) - 16.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.))) - 16.*np.pi**2.*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.)))
    return values

def u_body_load_time(t):
    return lambda x: body_load(x, t)

# -------------------------------
# Create mesh and function space
# -------------------------------
domain = create_unit_square(comm=MPI.COMM_WORLD, 
                            nx=grid_size,
                            ny=grid_size,
                            cell_type=CellType.triangle)
assert L_val == 1  # unit square
coords = domain.geometry.x
xmin = np.min(coords[:, 0]); xmax = np.max(coords[:, 0])
ymin = np.min(coords[:, 1]); ymax = np.max(coords[:, 1])
print(f"x goes from {xmin} to {xmax}, y goes from {ymin} to {ymax}")

v_bl = element(
    family="Lagrange",
    cell=domain.topology.cell_name(),
    degree=1,
    shape=(domain.geometry.dim,)
)
V = functionspace(domain, v_bl)

# -------------------------------
# Define trial/test functions and BC functions
# -------------------------------
u = TrialFunction(V)
v = TestFunction(V)
u_D = Function(V)  # Dirichlet BC
b = Function(V)    # Body load

def walls(x):
    return (
        np.isclose(x[0], 0) | np.isclose(x[0], 1)
        | np.isclose(x[1], 0) | np.isclose(x[1], 1)
    )

fdim = domain.topology.dim - 1  # In 2D, fdim = 1
boundary_facets = locate_entities_boundary(domain, fdim, walls)
dofs = locate_dofs_topological(V, fdim, boundary_facets)
bc = dirichletbc(u_D, dofs)

# -------------------------------
# Define strain and stress
# -------------------------------
def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u):
    lambda_ = c_k_led - 2*c_mu_led
    mu_     = c_mu_led
    print(mu_)
    print(lambda_)

    return (lambda_*div(u)*Identity(2) + 2*mu_*epsilon(u))

# -------------------------------
# Create a Constant to wrap the mass form
# -------------------------------
one = Constant(domain, PETSc.ScalarType(1.))

# -------------------------------
# Define bilinear forms for mass and stiffness matrices
# -------------------------------
a_mass = one * inner(u, v) * dx
a_stiff = inner(sigma(u), epsilon(v)) * dx

# -------------------------------
# Assemble mass and stiffness matrices
# -------------------------------
M = assemble_matrix(form(a_mass))
M.assemble()
K = assemble_matrix(form(a_stiff))
K.assemble()

# Lumped mass: get inverse diagonal of M
M_diag = M.getDiagonal().copy()
M_diag.reciprocal()

# -------------------------------
# Create solution vectors for time stepping
# -------------------------------
u_nm1 = Function(V)  # u^{n-1}
u_n   = Function(V)  # u^n
u_np1 = Function(V)  # u^{n+1}

# Set initial conditions using u_dirBC at t=0:
u_n.interpolate(lambda x: u_dirBC(x, 0.0))
# Assuming zero initial velocity, set u_nm1 equal to u_n:
u_nm1.x.array[:] = u_n.x.array

# Set initial Dirichlet BC on the boundary (using the manufactured function)
u_D.interpolate(u_dir_BC_time(t))
bc = dirichletbc(u_D, dofs)

v_func = Function(V)
v_func.interpolate(lambda x: velocity_init(x, 0.0))
temp = u_n.x.array - dt_led * v_func.x.array
u_nm1.x.array[:] = temp

# -------------------------------
# Time-Stepping Loop (Explicit Central Difference)
# -------------------------------
T_final = 1
dt_val = dt_led  # already defined above
t_ = t
snapshot = 0
snapshot_interval = 10  # adjust as needed

iteration = 0
while t_ < T_final:
    t_ += dt_val

    # Update Dirichlet BC and body force for current time
    u_D.interpolate(u_dir_BC_time(t_))
    b.interpolate(u_body_load_time(t_))

    # Assemble force vector: L_form = dot(b, v) * dx
    L_form = dot(b, v) * dx
    f_vec = assemble_vector(form(L_form))
    apply_lifting(f_vec, [form(a_stiff)], [[bc]])
    f_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(f_vec, [bc])

    # Compute stiffness term: K * u_n
    Ku_n = u_n.x.petsc_vec.duplicate()
    K.mult(u_n.x.petsc_vec, Ku_n)

    # Compute right-hand side: f_vec - K*u_n, then explicit update
    rhs_vec = f_vec.copy()
    rhs_vec.axpy(-1.0, Ku_n)   # rhs_vec = f_vec - K*u_n
    rhs_vec.scale(dt_val**2)
    rhs_vec.axpy(2.0, u_n.x.petsc_vec)
    rhs_vec.axpy(-1.0, u_nm1.x.petsc_vec)

    # Multiply componentwise by the inverse lumped mass diagonal
    u_np1.x.array[:] = M_diag.array[:] * rhs_vec.array[:]

    # Enforce Dirichlet BC on u_np1 by overwriting DOF values on the boundary
    u_np1.x.array[dofs] = u_D.x.array[dofs]
    u_np1.x.scatter_forward()

    # Rotate solution vectors for next time step: u_nm1 <- u_n, u_n <- u_np1
    u_nm1.x.array[:] = u_n.x.array
    u_n.x.array[:] = u_np1.x.array

    # Optional: snapshot/plot
    if iteration % 1 == 0 and domain.comm.rank == 0:
        print(f"[{iteration}/{num_steps}] Time: {t_:.4f}")

        # Get nodal coordinates
        points = domain.geometry.x

        # Get cell connectivity
        cell_conn = domain.topology.connectivity(domain.topology.dim, 0).array

        # Check if the domain is made of triangles or quads
        # (CellType.triangle or CellType.quadrilateral, etc.)
        if domain.topology.cell_type.name.lower().startswith("quadrilateral"):
            # Reshape the connectivity to (num_cells, 4)
            cells = cell_conn.reshape((-1, 4))
            # Subdivide each quad into two triangles
            triangles = []
            for quad in cells:
                triangles.append([quad[0], quad[1], quad[2]])
                triangles.append([quad[0], quad[2], quad[3]])
            triangles = np.array(triangles, dtype=np.int32)
        else:
            # Otherwise assume triangular cells already
            cells = cell_conn.reshape((-1, 3))
            triangles = cells

        # Compute displacement magnitude
        u_flat = u_n.x.array.reshape((-1, domain.geometry.dim))
        mag = np.linalg.norm(u_flat, axis=1)

        # Create a Triangulation object
        triangulation = tri.Triangulation(points[:, 0], points[:, 1], triangles)

        # Plot the magnitude
        plt.figure(figsize=(10, 4))
        contour = plt.tricontourf(triangulation, mag, cmap="viridis")
        plt.colorbar(contour)
        plt.title(f"Displacement magnitude at time t = {t_:.2f}")
        plt.axis("equal")
        plt.axis("off")

        plt.savefig(
            f"displacement_magnitude_{t_:.2f}_{int(time.time())}.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    iteration += 1

if domain.comm.rank == 0:
    print("Time integration complete.")
