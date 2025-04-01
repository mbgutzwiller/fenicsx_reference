import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType
import time
from petsc4py import PETSc

# Domain setup
L, H, B = 8.0, 0.2, 0.1
domain = create_box(MPI.COMM_WORLD, [[0.0, -B/2, -H/2], [L, B/2, H/2]], [8, 2, 2], CellType.hexahedron)
dim = domain.topology.dim
dx = ufl.Measure("dx", domain=domain)

# Function space (vector-valued)
degree = 2
shape = (dim,)
V = fem.functionspace(domain, ("Q", degree, shape))

# Material properties
E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)
rho = fem.Constant(domain, 7.8e-3)
# f = fem.Constant(domain, (0.0,) * dim)
f = fem.Function(V)



mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Kinematics
def epsilon(u): return ufl.sym(ufl.grad(u))
def sigma(u): return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim) + 2 * mu * epsilon(u)

# Boundary conditions
def left(x): return np.isclose(x[0], 0.0)
def point(x): return np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)

clamped_dofs = fem.locate_dofs_geometrical(V, left)
bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]

# Observation point

local_point_dofs = fem.locate_dofs_geometrical(V, point)
all_point_dofs = MPI.COMM_WORLD.allgather(local_point_dofs)

point_dofs = next((pd for pd in all_point_dofs if len(pd) > 0), None)

if point_dofs is None:
    raise RuntimeError("No point DOFs found!")

# Functions
u, u_old, u_old2 = fem.Function(V), fem.Function(V), fem.Function(V)
v, a = fem.Function(V), fem.Function(V)
u_new = fem.Function(V)

# Forms for mass, stiffness, damping
u_, du = ufl.TestFunction(V), ufl.TrialFunction(V)
eta_M, eta_K = fem.Constant(domain, 0.), fem.Constant(domain, 0.)
# eta_M, eta_K = fem.Constant(domain, 1e-4), fem.Constant(domain, 1e-4)

# def mass(u, v): return rho * ufl.dot(u, u_) * dx
# def stiffness(u, v): return ufl.inner(sigma(u), epsilon(v)) * dx
# def damping(u, v): return eta_M * mass(u, v) + eta_K * stiffness(u, v)

def mass(u, v): return rho * ufl.dot(u, v) * dx
def stiffness(u, v): return ufl.inner(sigma(u), epsilon(v)) * dx
def damping(u, v): return eta_M * mass(u, v) + eta_K * stiffness(u, v)


mass_form = fem.form(mass(du, u_))
M = fem.petsc.assemble_matrix(mass_form, bcs=[])
M.assemble()
M_diag = M.getDiagonal().copy()
M_diag.assemble()
M_lumped = M_diag.getArray()

# Time stepping setup
c = float(np.sqrt(E.value / rho.value))
L_min = min(np.linalg.norm(np.ptp(domain.geometry.x[domain.topology.connectivity(dim, 0).links(cell)], axis=0)) for cell in range(domain.topology.index_map(dim).size_local))
dt_value = 0.5 * L_min / c
dt = fem.Constant(domain, dt_value)

Nsteps, Nsave = 1000, 100
times = np.linspace(0, 2, Nsteps + 1)
save_freq = Nsteps // Nsave

vtk = io.VTKFile(domain.comm, "results/explicit_beam.pvd", "w")
energies = np.zeros((Nsteps + 1, 3))
tip_displacement = np.zeros((Nsteps + 1, 3))

# Predefine force vector
rhs_vec = fem.petsc.create_vector(fem.form(ufl.dot(f, u_)*dx))
a_vec = fem.Function(V)

# Initial conditions
# vx = fem.Function(fem.functionspace(domain, ("Q", 1)))
# vy = fem.Function(fem.functionspace(domain, ("Q", 1)))
# vx.interpolate(lambda x: np.full(x.shape[1], 1.0))
# vy.interpolate(lambda x: np.zeros(x.shape[1]))
# v.x.array[0::2] = vx.x.array
# v.x.array[1::2] = vy.x.array
def initial_velocity(x):
    values = np.zeros((dim, x.shape[1]))
    values[0, :] = 1.0  # vx = 1.0, vy = vz = 0.0
    return values

v.interpolate(initial_velocity)
v.x.scatter_forward()

# Main time loop
t = 0.0
stime = time.time()
for i, dti in enumerate(np.diff(times)):
    if i == 0:
        u_old2.x.array[:] = u_old.x.array

    if MPI.COMM_WORLD.rank == 0:
        print(f"{i/Nsteps:.2f}", flush=True)
    if i % save_freq == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti
    if t <= 0.2:
        f.interpolate(lambda x: np.array([
            np.zeros_like(x[0]),
            np.full_like(x[0], 100 * t / 0.2),
            np.full_like(x[0], 150 * t / 0.2)
        ]))
    else:
        f.interpolate(lambda x: np.zeros((dim, x.shape[1])))

    f_form = fem.form(ufl.dot(f, u_) * dx)
    ff = fem.assemble_vector(f_form)
    Kuu = fem.assemble_vector(fem.form(stiffness(u, u_)))
    Cvv = fem.assemble_vector(fem.form(damping(v, u_)))

    ff.petsc_vec.axpy(-1.0, Kuu.petsc_vec)
    ff.petsc_vec.axpy(-1.0, Cvv.petsc_vec)

    ff.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)

    ff.petsc_vec.pointwiseDivide(ff.petsc_vec, M_diag)
    a_vec.x.petsc_vec.copy(ff.petsc_vec)
    u_new.x.petsc_vec.copy(u.x.petsc_vec)
    u_new.x.petsc_vec.scale(2.0)
    u_new.x.petsc_vec.axpy(-1.0, u_old2.x.petsc_vec)
    u_new.x.petsc_vec.axpy(dt.value**2, a_vec.x.petsc_vec)


    v.x.petsc_vec.copy(u_new.x.petsc_vec)
    v.x.petsc_vec.axpy(-1.0, u_old2.x.petsc_vec)
    v.x.petsc_vec.scale(1.0 / (2 * dt.value))
    
    u_old2.x.petsc_vec.copy(u.x.petsc_vec)
    u.x.petsc_vec.copy(u_new.x.petsc_vec)

    tip_displacement[i + 1, :] = u.x.array[point_dofs]
    if MPI.COMM_WORLD.rank == 0:
        print(f"Tip disp: {tip_displacement[i+1, :]}")

print(f"Simulation took {int(time.time()-stime)}s")
vtk.close()
