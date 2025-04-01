import dolfinx.fem.petsc
import dolfinx, ufl
import numpy as np
import sympy as sp
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
import sys, time

comm = MPI.COMM_WORLD

# =============================================================================
# Problem parameters
# =============================================================================
geometry_parameters = {'Lx': 1, 'Ly': 1}
_c_k = 1.4 ** 0.5
_c_mu = 0.1 ** 0.5
_rho = 1
_mu = _c_mu**2 * _rho

_K = 1 / 3 * (3 * _rho * _c_k**2 - _mu)
_G = _mu
_E = (9 * _K * _G) / (3 * _K + _G)
_nu = (3 * _K - 2 * _G) / (2 * (3 * _K + _G))

material_properties = {
    'E':   _E,
    'nu':  _nu,
    'rho': _rho,
    'c1':  0,
    'c2':  0,
}

mesh_parameters = {'nx': 40, 'ny': 40}
timestepping_parameters = {'initial_time': 0.,
                           'total_time': 1,
                           'explicit_safety_factor': 3}
OTP_settings = {'xdmf_filename': "explicit_elastodynamics.xdmf"}

# For possibly time dependent BC.
t_sp = sp.Symbol('t', real=True)
U_imp = sp.Piecewise((0, True))
V_imp = sp.diff(U_imp, t_sp)
A_imp = sp.diff(V_imp, t_sp)

# =============================================================================
# Mesh and geometry
# =============================================================================
mesh = dolfinx.mesh.create_rectangle(comm, 
           [np.array([0., 0.]), np.array([geometry_parameters['Lx'], geometry_parameters['Ly']])],
           [mesh_parameters['nx'], mesh_parameters['ny']], 
           dolfinx.mesh.CellType.triangle)

# =============================================================================
# Define function spaces and functions
# =============================================================================
V_t = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))  # Galerkin, bilinear.
u = dolfinx.fem.Function(V_t, name="Displacement")
u_new = dolfinx.fem.Function(V_t)
v = dolfinx.fem.Function(V_t, name="Velocity")
v_new = dolfinx.fem.Function(V_t)
a = dolfinx.fem.Function(V_t, name="Acceleration")
a_new = dolfinx.fem.Function(V_t)
ones_a = dolfinx.fem.Function(V_t)  # used for lumped mass assembly

# =============================================================================
# Boundary conditions and body force
# =============================================================================
gdim = mesh.topology.dim
fdim = gdim - 1

# Apply zero Dirichlet on all boundaries
all_boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
all_boundary_dofs = dolfinx.fem.locate_dofs_topological(V_t, fdim, all_boundary_facets)
zero_disp = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
bc_u = dolfinx.fem.dirichletbc(zero_disp, all_boundary_dofs, V_t)
bcs_u = [bc_u]
bcs_v = [bc_u]
bcs_a = [bc_u]

# Body force
V_vec = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
b = dolfinx.fem.Function(V_vec, name="Body force")

# =============================================================================
# Define weak form
# =============================================================================
dx = ufl.Measure("dx", domain=mesh)
n = ufl.FacetNormal(mesh)

# Kinetic energy density
kinetic_energy_density = 0.5 * ufl.inner(v, v) * _rho

# Strain and stress
lmbda = material_properties["E"] * material_properties["nu"] / ((1.0 + material_properties["nu"]) * (1.0 - 2.0 * material_properties["nu"]))
# eps = ufl.sym(ufl.grad(u))
eps = ufl.variable(ufl.sym(ufl.grad(u)))
elastic_energy_density = lmbda/2 * ufl.tr(eps)**2 + _mu * ufl.inner(eps, eps)

# Define energy forms
external_work = ufl.dot(b, u) * dx
elastic_energy = elastic_energy_density * dx
potential_energy = elastic_energy - external_work
kinetic_energy = kinetic_energy_density * dx

# Energy derivatives (variations)
u_test = ufl.TestFunction(V_t)
P_du = ufl.derivative(potential_energy, u, u_test)  # internal force vector
K_dv = ufl.derivative(kinetic_energy, v, u_test)  # inertial forces
Q_dv = 0  # damping

Res = ufl.replace(K_dv, {v: a}) + Q_dv + P_du

# =============================================================================
# Assemble Lumped Mass Vector
# =============================================================================
M = ufl.lhs(ufl.replace(K_dv, {v: ufl.TrialFunction(V_t)}))
M_lumped_form = dolfinx.fem.form(ufl.action(M, ones_a))  # row sum trick
M_lumped = dolfinx.fem.petsc.assemble_vector(M_lumped_form)
M_lumped.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
M_lumped.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
# =============================================================================
# Define Force form (residual)
# =============================================================================
F_form = dolfinx.fem.form(ufl.replace(- (P_du + Q_dv), {u: u_new}))

# =============================================================================
# Time stepping parameters
# =============================================================================
t0 = timestepping_parameters['initial_time']
total_time = timestepping_parameters["total_time"]
explicit_safety_factor = timestepping_parameters["explicit_safety_factor"]
t       = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
delta_t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))

t.value = t0

# Calculate h_min (minimal internodal distance)
tdim = mesh.topology.dim
h_min_local = min(mesh.h(tdim, np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)))
h_min = np.asarray(comm.gather(h_min_local, root=0))
if comm.rank == 0:
    h_min = min(h_min)
h_min = MPI.COMM_WORLD.bcast(h_min, root=0)

# Approximate CFL condition
c_val = np.sqrt((material_properties["E"]/material_properties["rho"]) * 
                ((1-material_properties["nu"])/((1+material_properties["nu"])*(1-2*material_properties["nu"]))))
delta_t_critic = h_min / c_val
delta_t0 = delta_t_critic / explicit_safety_factor
delta_t.value = delta_t0

# =============================================================================
# Output mesh and initial condition
# =============================================================================
with dolfinx.io.XDMFFile(comm, OTP_settings['xdmf_filename'], "w") as xdmf_file:
    xdmf_file.write_mesh(mesh)
    xdmf_file.write_function(u, t.value)

if comm.rank == 0:
    print('RESOLUTION STATUS')
    sys.stdout.flush()

# =============================================================================
# INITIALIZE: Set initial displacement and velocity fields
# =============================================================================
x = mesh.geometry.x
u_values = u.x.array
v_values = v.x.array
for i, xi in enumerate(x):
    u_values[2*i] = np.sin(4*np.pi*xi[0]) * np.sin(2*np.pi*xi[1]) * np.sin(4.*np.pi*(-0.1))
    u_values[2*i+1] = np.sin(4*np.pi*xi[0]) * np.sin(2*np.pi*xi[1]) * np.sin(4.*np.pi*(0.3))
    v_values[2*i] = 4. * np.pi * np.sin(4. * np.pi * xi[0]) * np.sin(2. * np.pi * xi[1]) * np.cos(4. * np.pi * (0. - 0.1))
    v_values[2*i+1] = 4. * np.pi * np.sin(4. * np.pi * xi[0]) * np.sin(2. * np.pi * xi[1]) * np.cos(4. * np.pi * (0. + 0.3))
u.x.scatter_forward()
v.x.scatter_forward()

b_values = b.x.array
for i, xi in enumerate(x):
    b_values[2*i] = _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.)) + 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))) - _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.)) - 4.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))) - _c_k ** 2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.)) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))
    # b_values[2*i]     = np.sin(4.*np.pi*xi[0]i[0]) * np.sin(2.*np.pi*xi[1]) * np.sin(4.*np.pi*(t.value0 - 0.1))
    b_values[2*i + 1] = _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.)) + 4.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))) - _c_k ** 2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.)) - 4.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))) - _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.)) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))
    # b_values[2*i + 1] = np.sin(4.*np.pi*xi[0]) * np.sin(2.*np.pi*xi[1]) * np.sin(4.*np.pi*(t0 + 0.3))
b.x.scatter_forward()

# =============================================================================
# INITIAL ACCELERATION: a = M⁻¹ F
# =============================================================================
F_init = dolfinx.fem.petsc.assemble_vector(F_form)
F_init.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
F_init.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

a_petsc = a.x.petsc_vec
F_petsc = F_init
# Get local forms
with F_petsc.localForm() as F_local, M_lumped.localForm() as M_local, a.x.petsc_vec.localForm() as a_local:
    a_local.pointwiseDivide(F_local, M_local)
a.x.scatter_forward()
dolfinx.fem.set_bc(a.x.array, bcs_a)

F_init.destroy()

# =============================================================================
# Time stepping loop
# =============================================================================
ts = []
c_led = (1 / mesh_parameters['nx']) / delta_t.value

stime = time.time()
while float(t.value) < total_time:

    # Update displacement field: u_new = u + delta_t*v + 0.5*delta_t^2*a
    with u.x.petsc_vec.localForm() as u_local, \
         v.x.petsc_vec.localForm() as v_local, \
         a.x.petsc_vec.localForm() as a_local, \
         u_new.x.petsc_vec.localForm() as u_new_local:
        # Perform vectorized update using the NumPy arrays from localForm:
        u_new_local.array[:] = u_local.array + float(delta_t) * v_local.array + 0.5 * float(delta_t)**2 * a_local.array
    u_new.x.scatter_forward()

    dolfinx.fem.set_bc(u_new.x.array, bcs_u)

    # Update body force b (if time-dependent)
    b_values = b.x.array
    for i, xi in enumerate(mesh.geometry.x):
        b_values[2*i] = _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.)) + 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))) - _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.)) - 4.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))) - _c_k ** 2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.)) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.))
        b_values[2*i+1] = _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.)) + 4.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))) - _c_k ** 2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.)) - 4.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))) - _c_mu**2*(8.*np.pi**2.*np.cos(4.*np.pi*xi[0])*np.cos(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value - 1./10.)) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))) - 16.*np.pi**2.*np.sin(4.*np.pi*xi[0])*np.sin(2.*np.pi*xi[1])*np.sin(4.*np.pi*(t.value + 3./10.))
    b.x.scatter_forward()

    # Update acceleration: F = assemble_vector(F_form), then a_new = F / M
    F = dolfinx.fem.petsc.assemble_vector(F_form)
    F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    with F.localForm() as F_local, M_lumped.localForm() as M_local, a_new.x.petsc_vec.localForm() as a_new_local:
        a_new_local.pointwiseDivide(F_local, M_local)
    a_new.x.scatter_forward()
    F.destroy()

    dolfinx.fem.set_bc(a_new.x.array, bcs_a)

    # Update velocity: v_new = v + 0.5*delta_t*(a_new + a)
    with v.x.petsc_vec.localForm() as v_local, \
         a_new.x.petsc_vec.localForm() as a_new_local, \
         a.x.petsc_vec.localForm() as a_local, \
         v_new.x.petsc_vec.localForm() as v_new_local:
        v_new_local.array[:] = v_local.array + 0.5 * float(delta_t) * (a_new_local.array + a_local.array)
    v_new.x.scatter_forward()
    
    dolfinx.fem.set_bc(v_new.x.array, bcs_v)

    # Copy new state into current state
    u.x.array[:] = u_new.x.array
    v.x.array[:] = v_new.x.array
    a.x.array[:] = a_new.x.array
    u.x.scatter_forward()
    v.x.scatter_forward()
    a.x.scatter_forward()

    t.value += float(delta_t)
    delta_t.value = float(ufl.conditional(ufl.lt(delta_t0, total_time-t), delta_t0, total_time-t))
    
    ts = np.concatenate((ts, [t.value]))
    
    with dolfinx.io.XDMFFile(comm, OTP_settings['xdmf_filename'], 'a') as xdmf_file:
       xdmf_file.write_function(u, t.value)
    
    if comm.rank == 0:
        print(f"Progress: {100*t.value/total_time:.0f}% ", end="\r")
        sys.stdout.flush()

print(f"Simulation took {int(time.time() - stime)} s to run.")
