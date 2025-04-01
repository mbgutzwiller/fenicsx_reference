import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import HTML, clear_output

from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType
import time

L = 8.0
H = 0.2
B = 0.1
# T_total = 1e-3  # delete this afterwards
# grid_size = 100  # delete this afeterwards
domain = create_box(
    MPI.COMM_WORLD,
    [[0.0, -B / 2, -H / 2], [L, B / 2, H / 2]],
    [8, 2, 2],
    CellType.hexahedron,
)

dim = domain.topology.dim
dx = ufl.Measure("dx", domain=domain)

degree = 2
shape = (dim,)
V = fem.functionspace(domain, ("Q", degree, shape))

u = fem.Function(V, name="Displacement")

def left(x):
    return np.isclose(x[0], 0.0)


def point(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)


# Locate dofs for clamped (left) boundary
clamped_dofs = fem.locate_dofs_geometrical(V, left)

# Each rank finds its local dofs for the point condition
local_point_dofs = fem.locate_dofs_geometrical(V, point)

# Gather the local dof arrays from all processes
all_point_dofs = MPI.COMM_WORLD.allgather(local_point_dofs)

# Find the global point dof (choose the first non-empty result)
global_point_dof = None
for pd in all_point_dofs:
    if len(pd) > 0:
        global_point_dof = pd[0]
        break

if global_point_dof is None:
    raise RuntimeError("No point dof found!")

# Compute the indices for the vector components at that dof
point_dofs = np.arange(global_point_dof * dim, (global_point_dof + 1) * dim)


bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]

E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)
rho = fem.Constant(domain, 7.8e-3)
f = fem.Constant(domain, (0.0,) * dim)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)

u_old = fem.Function(V)
v_old = fem.Function(V)
a_old = fem.Function(V)
a_new = fem.Function(V)
v_new = fem.Function(V)

beta_ = 0.25
beta = fem.Constant(domain, beta_)
gamma_ = 0.5
gamma = fem.Constant(domain, gamma_)
dt = fem.Constant(domain, 0.0)

a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
a_expr = fem.Expression(a, V.element.interpolation_points())

v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
v_expr = fem.Expression(v, V.element.interpolation_points())

# beta_ = 0.25
# # beta_ = 0.  # explicit scheme
# beta = fem.Constant(domain, beta_)
# gamma_ = 0.5
# gamma = fem.Constant(domain, gamma_)
# dt = fem.Constant(domain, 0.0)
# # dt = fem.Constant(domain, T_total / (2.5 * grid_size))

# a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
# # a = (u - u_old) / (dt ** 2) + a_old  # Acceleration term
# a_expr = fem.Expression(a, V.element.interpolation_points())

# # v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
# v = v_old + dt * a  # Update velocity explicitly
# v_expr = fem.Expression(v, V.element.interpolation_points())

u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)



eta_M = fem.Constant(domain, 1e-4)
eta_K = fem.Constant(domain, 1e-4)


def mass(u, u_):
    return rho * ufl.dot(u, u_) * ufl.dx


def stiffness(u, u_):
    return ufl.inner(sigma(u), epsilon(u_)) * ufl.dx


def damping(u, u_):
    return eta_M * mass(u, u_) + eta_K * stiffness(u, u_)


Residual = mass(a, u_) + damping(v, u_) + stiffness(u, u_) - ufl.dot(f, u_) * ufl.dx

Residual_du = ufl.replace(Residual, {u: du})
a_form = ufl.lhs(Residual_du)
L_form = ufl.rhs(Residual_du)

problem = fem.petsc.LinearProblem(
    a_form, L_form, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "jacobi"}
)

# problem = fem.petsc.LinearProblem(
#     a_form, L_form, u=u, bcs=bcs,
#     petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-6}
# )

E_kin = fem.form(0.5 * mass(v_old, v_old))
E_el = fem.form(0.5 * stiffness(u_old, u_old))
P_damp = fem.form(damping(v_old, v_old))

vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")

t = 0.0

Nsteps = 400
Nsave = 100
times = np.linspace(0, 2, Nsteps + 1)
save_freq = Nsteps // Nsave
energies = np.zeros((Nsteps + 1, 3))
tip_displacement = np.zeros((Nsteps + 1, 3))
stime = time.time()
for i, dti in enumerate(np.diff(times)):
    if MPI.COMM_WORLD.rank == 0:
        print(i/Nsteps, flush=True)
    if i % save_freq == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti

    if t <= 0.2:
        f.value = np.array([0.0, 1.0, 1.5]) * t / 0.2
    else:
        f.value *= 0.0

    problem.solve()

    u.x.scatter_forward()  # updates ghost values for parallel computations

    # compute new velocity v_n+1
    v_new.interpolate(v_expr)

    # compute new acceleration a_n+1
    a_new.interpolate(a_expr)

    # update u_n with u_n+1
    # u.copy(u_old.x.array)

    # update v_n with v_n+1
    # v_new.copy(v_old.x.array)

    # update a_n with a_n+1
    # a_new.copy(a_old.x.array)
    u.x.petsc_vec.copy(u_old.x.petsc_vec)
    v_new.x.petsc_vec.copy(v_old.x.petsc_vec)
    a_new.x.petsc_vec.copy(a_old.x.petsc_vec)


    energies[i + 1, 0] = fem.assemble_scalar(E_el)
    energies[i + 1, 1] = fem.assemble_scalar(E_kin)
    energies[i + 1, 2] = energies[i, 2] + dti * fem.assemble_scalar(P_damp)

    tip_displacement[i + 1, :] = u.x.array[point_dofs]

    # clear_output(wait=True)
    print(f"Time increment {i+1}/{Nsteps}", flush=True)
print(f"simulation took {int(time.time()-stime)}s seconds to run")
vtk.close()