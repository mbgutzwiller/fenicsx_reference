import ffcx.compiler
import dolfinx, ufl
import numpy as np
import sympy as sp
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
import sys
#import mesh_plotter
import dolfinx.fem.petsc
import time
import pandas as pd
from tqdm import tqdm
import os
import json
# force local cache (for HPCs)
import ffcx

try:
    os.mkdir("cache")
except FileExistsError:
    pass

jit_options = {
    "cache_dir": "./cache/",
    "cffi_extra_compile_args": ["-Ofast", "-march=native"],
    # "quadrature_degree": 1
}

with open("dolfinx_jit_options.json", "w") as jit_options_file:
    json.dump(jit_options, jit_options_file)

dolfinx.log.set_output_file("output_file_dolfinx")
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

comm = MPI.COMM_WORLD

# Characteristic dimensions of the domain
geometry_parameters = {'Lx': 1, 
                       'Ly': 1}

_c_k = 0.8 ** 0.5
_c_mu = 0.7 ** 0.5
_rho = 1
_mu = _c_mu**2 * _rho

_K = 1 / 3 * (3 * _rho * _c_k**2 - _mu)
_G = _mu
_E = (9 * _K * _G) / (3 * _K + _G)
_nu = (3 * _K - 2 * _G) / (2 * (3 * _K + _G))

material_properties = {
    'E': _E,
    'nu': _nu,
    'rho': _rho,
    'c1': 0,
    'c2': 0,
}

S_pulse = 5e-3  # sharpness of initial pulse
snapshot_interval = 1

l2_err_plot = []
grid_sizes_run = []
runtimes_no_comp = []
runtimes_comp = []

# grid_sizes = np.array([int(gs*1**(1/3)) for gs in [320]])
# grid_sizes = [50, 100, 200, 400, 800, 1600, 3200, 4000]
grid_sizes = [50, 100, 200, 400, 800, 1600, 3200, 4000]
# grid_sizes = [50, 100, 200, 400, 800, 1200, 1600]#, 2400, 3200, 4000]
# grid_sizes = [2400]

for i, grid_size in zip(range(len(grid_sizes)), grid_sizes):
    grid_sizes_run.append(grid_size)
    stime_comp = time.time()
    # Mesh control
    mesh_parameters = {'nx': grid_size,
                       'ny': grid_size}

    # Time stepping control
    timestepping_parameters = {'initial_time':0., 
                            'total_time':1,
                            'total_steps':1000,
                            'explicit_safety_factor': 2.886751359}

    # Output parameters
    OTP_settings = {'xdmf_filename': f"explicit_elastodynamics_pulse.xdmf"}

    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, 
                                        [np.array([0., 0.]), 
                                        np.array([geometry_parameters['Lx'], geometry_parameters['Ly']])], 
                                        [mesh_parameters['nx'], mesh_parameters['ny']], 
                                        dolfinx.mesh.CellType.triangle)

    gdim = mesh.topology.dim
    fdim = gdim - 1 

    tdim = mesh.topology.dim
    h_min_local= min(mesh.h(tdim, np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32))) #Minimum internodal distance h of every processor submesh
    h_min = np.asarray(comm.gather(h_min_local, root=0)) # Gather the minimum internodal distance of every processor submesh in an array at processor 0
    if comm.rank == 0:  #Calculate the global minimum of h at processor 0
        h_min = min(h_min)  
    h_min = MPI.COMM_WORLD.bcast(h_min, root=0) #Broadcast to every processor the global minimum of h.

    # Geometrical regions  
    def top(x):
        return np.isclose(x[1], geometry_parameters["Ly"])
    def bottom(x):
        return np.isclose(x[1], 0.)
    def right (x):
        return np.isclose(x[0], geometry_parameters["Lx"])
    def left (x):
        return np.isclose(x[0], 0.)

    # Geometrical sets
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right)
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)

    tagged_facets = np.hstack([top_facets, 
                            bottom_facets, 
                            right_facets, 
                            left_facets])

    tag_values = np.hstack([np.full_like(top_facets,    1), 
                            np.full_like(bottom_facets, 2),
                            np.full_like(right_facets,  3),
                            np.full_like(left_facets,   4)])

    tagged_facets_sorted = np.argsort(tagged_facets)

    mt = dolfinx.mesh.meshtags(mesh, fdim, 
                            tagged_facets[tagged_facets_sorted], 
                            tag_values[tagged_facets_sorted])

    # Domain and subdomain measures
    # dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 1})
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)      # External Boundary measure
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=mt)      # External/Internal measure
    n = ufl.FacetNormal(mesh)                                  # External normal to the boundary

    # --------- Main functions and function spaces
    V_t = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    u = dolfinx.fem.Function(V_t, name="Displacement")
    u_new = dolfinx.fem.Function(V_t)
    v = dolfinx.fem.Function(V_t, name="Velocity")
    v_new = dolfinx.fem.Function(V_t)
    a = dolfinx.fem.Function(V_t, name="Acceleration")
    a_new = dolfinx.fem.Function(V_t)

    # --------- Unit nodal acceleration
    ones_a = dolfinx.fem.Function(V_t)

    all_boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    all_boundary_dofs = dolfinx.fem.locate_dofs_topological(V_t, fdim, all_boundary_facets)
    zero_disp = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
    bc_u = dolfinx.fem.dirichletbc(zero_disp, all_boundary_dofs, V_t)

    # Set bcs for u, v and a
    bcs_u = [bc_u]
    bcs_v = [bc_u]
    bcs_a = [bc_u]

    t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
    delta_t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))

    # Material properties
    E = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_E))
    nu = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_nu))
    rho = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_rho))
    c1 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))
    c2 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))

    # Kinetic energy density
    kinetic_energy_density = 0.5 * rho * ufl.inner(v,v)

    # Dissipated energy density
    eps_v = ufl.variable(ufl.sym(ufl.grad(v)))
    # dissipated_power_density = 0.5 * (c1 * ufl.inner(v,v) + c2 * ufl.inner(eps_v,eps_v))

    # Lame constants (Plane strain)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Infinitesimal strain tensor
    eps = ufl.variable(ufl.sym(ufl.grad(u)))

    # Strain energy density (Linear elastic)
    elastic_energy_density = lmbda / 2 * ufl.tr(eps) ** 2 + mu * ufl.inner(eps,eps)

    # Stress tensor
    sigma = ufl.diff(elastic_energy_density, eps) + c2*eps_v

    # External workd density
    # external_work_density = ufl.dot(b,u)

    # System's energy components
    kinetic_energy = kinetic_energy_density * dx
    # dissipated_power = dissipated_power_density * dx
    elastic_energy = elastic_energy_density * dx 
    # external_work = external_work_density * dx 

    potential_energy = elastic_energy #- external_work
    total_energy = kinetic_energy + potential_energy

    # Energy derivatives
    u_test = ufl.TestFunction(V_t)
    P_du = ufl.derivative(potential_energy, u, u_test)  # internal forces from potential energy
    K_dv = ufl.derivative(kinetic_energy,   v, u_test)  # inertial term from kinetic energy
    # Q_dv = ufl.derivative(dissipated_power, v, u_test)  # damping forces (irrelevant here)

    # Residual 
    # Res = ufl.replace(K_dv, {v: a}) + Q_dv + P_du
    Res = ufl.replace(K_dv, {v: a}) + P_du  # no dmaping

    M = ufl.lhs(ufl.replace(K_dv,{v: ufl.TrialFunction(V_t)}))
    M_lumped_form = dolfinx.fem.form(ufl.action(M, ones_a))
    # F_form = dolfinx.fem.form(ufl.replace(-(P_du+Q_dv),{u:u_new}))
    F_form = dolfinx.fem.form(ufl.replace(-P_du,{u:u_new}))

    # Initialization
    t0 = timestepping_parameters['initial_time']
    total_time = timestepping_parameters["total_time"]
    explicit_safety_factor = timestepping_parameters["explicit_safety_factor"]

    E.value = material_properties["E"]
    nu.value = material_properties["nu"]
    rho.value = material_properties["rho"]
    c1.value = material_properties["c1"]
    c2.value = material_properties["c2"]

    t.value = t0

    for func in [u,v,a] :
        func.x.array[:] = 0.
        
    ones_a.x.array[:] = 1.

    stp_cont = 0    
    ts = []

    M_lumped = dolfinx.fem.petsc.assemble_vector(M_lumped_form)
    M_lumped.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    M_lumped.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
    # Critical time increment (approximated CFL condition)
    c = np.sqrt((E.value/rho.value)*((1-nu.value)/((1+nu.value)*(1-2*nu.value)))) 
    delta_t_critic = h_min/c
    delta_t0 = delta_t_critic/explicit_safety_factor
    delta_t.value = delta_t0
    # print(1/delta_t0)

    # Create the file once, in write mode, collectively on *all* ranks
    # with dolfinx.io.XDMFFile(comm, OTP_settings['xdmf_filename'], "w") as xdmf:
    #     xdmf.write_mesh(mesh)
    #     xdmf.write_function(u, t.value)

    if comm.rank == 0:
        print ('RESOLUTION STATUS')
        sys.stdout.flush()

    current_time = float(t.value)

    x = mesh.geometry.x
    u_values = u.x.array
    for i, xi in enumerate(x):
        x_ = xi[0]
        y_ = xi[1]
        u_values[2*i]     = np.exp(-((x_-1/2)**2 + (y_-1/2)**2)/S_pulse) #np.sin(4.*np.pi*xi[0]) * np.sin(2.*np.pi*xi[1]) * np.sin(4.*np.pi*(t.value-0.1))
        u_values[2*i + 1] = np.exp(-((x_-1/2)**2 + (y_-1/2)**2)/S_pulse) #np.sin(4.*np.pi*xi[0]) * np.sin(2.*np.pi*xi[1]) * np.sin(4.*np.pi*(t.value+0.3))
    u.x.scatter_forward()

    v_values = v.x.array
    for i, xi in enumerate(x):
        # vx = 0. #4. * np.pi * np.sin(4. * np.pi * xi[0]) * np.sin(2. * np.pi * xi[1]) * np.cos(4. * np.pi * (current_time - 0.1))
        # vy = 0. #4. * np.pi * np.sin(4. * np.pi * xi[0]) * np.sin(2. * np.pi * xi[1]) * np.cos(4. * np.pi * (current_time + 0.3))
        v_values[2*i] = 0.
        v_values[2*i + 1] = 0.
    v.x.scatter_forward()

    a_values = a.x.array
    for i, xi in enumerate(x):
        # ax = 0. #-16. * np.pi**2 * np.sin(4. * np.pi * xi[0]) * np.sin(2. * np.pi * xi[1]) * np.sin(4. * np.pi * (current_time - 0.1))
        # ay = 0. #-16. * np.pi**2 * np.sin(4. * np.pi * xi[0]) * np.sin(2. * np.pi * xi[1]) * np.sin(4. * np.pi * (current_time + 0.3))
        a_values[2*i] = 0.
        a_values[2*i + 1] = 0.
    a.x.scatter_forward()

    stime_no_comp = time.time()
    step = 0

    #----------
    # Main loop
    #----------

    dof_coords = V_t.tabulate_dof_coordinates()[:, :2]
    # Only drop z-dimension
    node_coords = dof_coords[:, :2]  # shape: (1681, 2)

    if comm.rank == 0:
        pbar = tqdm(total=total_time, desc="Integrating")

    stime_no_comp = time.time()
    F = dolfinx.fem.petsc.create_vector(F_form)

    while float(t.value) < total_time:
        # Update u
        with u.x.petsc_vec.localForm() as u_local, \
            v.x.petsc_vec.localForm() as v_local, \
            a.x.petsc_vec.localForm() as a_local, \
            u_new.x.petsc_vec.localForm() as u_new_local:
            u_new_local.array[:] = u_local.array + float(delta_t) * v_local.array + 0.5 * float(delta_t)**2 * a_local.array
        dolfinx.fem.set_bc(u_new.x.array, bcs_u)
        u_new.x.scatter_forward()
        
        # Update acceleration
        F.zeroEntries()
        dolfinx.fem.petsc.assemble_vector(F, F_form)
        # F = dolfinx.fem.petsc.assemble_vector(F_form)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as F_local, M_lumped.localForm() as M_local, a_new.x.petsc_vec.localForm() as a_new_local:
            a_new_local.pointwiseDivide(F_local, M_local)
        dolfinx.fem.set_bc(a_new.x.array, bcs_a)
        a_new.x.scatter_forward()
        # F.destroy()

        with v.x.petsc_vec.localForm() as v_local, \
            a_new.x.petsc_vec.localForm() as a_new_local, \
            a.x.petsc_vec.localForm() as a_local, \
            v_new.x.petsc_vec.localForm() as v_new_local:
            v_new_local.array[:] = v_local.array + 0.5 * float(delta_t) * (a_new_local.array + a_local.array)    
        dolfinx.fem.set_bc(v_new.x.array, bcs_v)
        v_new.x.scatter_forward()

        # Copy i+1 into i
        u.x.array[:] = u_new.x.array
        v.x.array[:] = v_new.x.array
        a.x.array[:] = a_new.x.array

        # u.x.petsc_vec.copy(u_new.x.petsc_vec)
        # v.x.petsc_vec.copy(v_new.x.petsc_vec)
        # a.x.petsc_vec.copy(a_new.x.petsc_vec)

        
        # Is this needed?
        u.x.scatter_forward()
        v.x.scatter_forward()
        a.x.scatter_forward()

        t.value += float(delta_t)
        delta_t.value = float(ufl.conditional(ufl.lt(delta_t0,total_time-t), 
                                                    delta_t0,
                                                    total_time - t))
        
        ts = np.concatenate((ts,[t.value]))
        
        # if step % snapshot_interval == 0:
        #     with dolfinx.io.XDMFFile(comm, OTP_settings['xdmf_filename'], 'a') as xdmf_file:
        #         xdmf_file.write_function(u, t.value)
        
        if comm.rank == 0:
            pbar.update(float(delta_t))
        
        step += 1
    ftime = time.time()
    runtimes_no_comp.append(ftime-stime_no_comp)
    runtimes_comp.append(ftime-stime_comp)

    comm = MPI.COMM_WORLD

    # Only rank 0 will get the full vector
    if comm.rank == 0:
        print(f"runtimes no comp: {runtimes_no_comp}")
        print(f"loop runtimes: {runtimes_comp}")
        pbar.close()

if comm.rank == 0:
    print(f"runtimes no comp: {runtimes_no_comp}")
    print(f"loop runtimes: {runtimes_comp}")
    # delta_xs = 1 / np.array(grid_sizes_run)
    # plt.figure(figsize=(10, 5))
    # x1 = delta_xs[0]
    # y1 = l2_err_plot[0]
    # x2 = delta_xs[0]
    # y2 = l2_err_plot[0]
    # ref_line = y1 * (delta_xs / x1)  # Slope 1
    # ref_line2 = y2 * (delta_xs / x2)**2  # Slope 2
    # plt.loglog(delta_xs, l2_err_plot, marker='o', label="L2 Error", color="black")
    # plt.loglog(delta_xs, ref_line, "-.", label="Order 1")
    # plt.loglog(delta_xs, ref_line2, "--", label="Order 2")
    # plt.xlabel("Mesh size (h)")
    # plt.ylabel("L2 error")
    # plt.legend()
    # plt.grid(True, which="both")
    # plt.savefig("convergence_fencisx_loglog", dpi=300)
    # plt.show()
