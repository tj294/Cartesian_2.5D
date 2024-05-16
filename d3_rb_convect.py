"""
A Dedalus v3 script for running 2.5D numerical simulations of in a Cartesian box. This script
allows for an internal heating function, such as in Currie et al. 2020.
To Do:

Usage:
    d3_rb_convect.py [options]
    d3_rb_convect.py [--currie | --kazemi] [options]

Options:
    --Ra=<Ra>                       # Rayleigh number
    --Pr=<Pr>                       # Prandtl number [default: 1]
    --Ta=<Ta>                       # Taylor number [default: 1e-1]
    --theta=<theta>                 # co-latitude of box to rotation vector [default: 5]
    --Ly=<Ly>                       # Aspect Ratio of the box [default: 4]
    --Lz=<Lz>                       # Depth of the box [default: 1]
    --Ny=<Ny>                       # Horizontal resolution [default: 128]
    --Nz=<Nz>                       # Vertical resolution [default: 256]
    --tau=<tau>                     # timescale [default: viscous]
    --maxdt=<maxdt>                 # Maximum timestep [default: 1e-5]
    --stop=<stop>                   # Simulation stop time [default: 1.0]
    --currie                        # Run with Currie 2020 heating function
    --kazemi                        # Run with Kazemi 2022 heating function
    --Hwidth=<Hwidth>               # Width of heating zone [default: 0.2]
    --ff                            # Use fixed-flux boundary conditions
    --slip=SLIP                     # Boundary conditions No/Free [default: free]
    --top=TOP                       # Top boundary condition [default: insulating]
    --bottom=BOTTOM                 # Bottom boundary condition [default: insulating]
    --snaps=<snaps>                 # Snapshot interval [default: 500]
    --horiz=<horiz>                 # Horizontal analysis interval [default: 100]
    --scalar=<scalar>               # Scalar analysis interval [default: 1]
    -o OUT_PATH, --output OUT_PATH  # output file [default= ../DATA/output/]
    -i IN_PATH, --input IN_PATH     # path to read in initial conditions from
    -m=<mesh>, --mesh=<mesh>        # Processor Mesh
    -t --test                       # Do not save any output
    -k, --kill                      # Kills the program after building the solver.
    -f, --function                  # Plots the heating function
"""

import numpy as np
import dedalus.public as d3
import logging
import os
import sys
import pathlib
from glob import glob
from docopt import docopt
import json
from mpi4py import MPI

ncpu = MPI.COMM_WORLD.size

# import rb_params as rp

logger = logging.getLogger(__name__)


class NaNFlowError(Exception):
    exit_code = -50
    pass


def argcheck(argument, params, type=float):
    if argument:
        return type(argument)
    else:
        return params


exit_code = 0
args = docopt(__doc__, version="2.0")

mesh = args["--mesh"]
if mesh is not None:
    mesh = mesh.split(",")
    mesh = [int(mesh[0]), int(mesh[1])]
logger.info("ncpu = {}".format(ncpu))
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2 ** np.ceil(log2 / 2)), int(2 ** np.floor(log2 / 2))]
logger.info("running on processor mesh={}".format(mesh))

if not (args["--test"]):
    outpath = os.path.normpath(args["--output"]) + "/"
    os.makedirs(outpath, exist_ok=True)
    logger.info("Writing to {}".format(outpath))

if args["--input"]:
    restart_path = os.path.normpath(args["--input"]) + "/"
    logger.info("Reading from {}".format(restart_path))

Ly = float(args["--Ly"])
Lz = float(args["--Lz"])
if args["--Nz"]:
    Nz = int(args["--Nz"])
    if args["--Ny"]:
        Ny = int(args["--Ny"])
    else:
        Ny = 2 * Nz
else:
    # ? Might never run since resolution arguments have default arguments?
    if args["--input"]:
        with open(restart_path + "run_params/runparams.json", "r") as f:
            inparams = json.load(f)
        Ny = inparams["Ny"]
        Nz = inparams["Nz"]
    else:
        Ny = int(args["--Ny"])
        Nz = int(args["--Nz"])

try:
    Ra = float(args["--Ra"])
except ValueError:
    print("Must provide a valid Ra value")
Pr = float(args["--Pr"])
Ta = float(args["--Ta"])

logger.info(f"Ro_c = {np.sqrt(Ra / (Pr * Ta)):1.2e}")

snapshot_iter = int(args["--snaps"])
horiz_iter = int(args["--horiz"])
scalar_iter = int(args["--scalar"])

if args["--kazemi"]:
    heat_type = "Kazemi"
elif args["--currie"]:
    heat_type = "Currie"
else:
    heat_type = None
if args["--slip"] == "no":
    slip_type = "No Slip"
else:
    slip_type = "Free Slip"

logger.info(
    f"Ra={Ra:1.1e}, Pr={Pr:1.1e}, Ta={Ta:1.1e}\nLy={Ly}, Lz={Lz}, Ny={Ny}, Nz={Nz}, Heated={heat_type}, {slip_type}"
)

# parallel = "gather"
parallel = None

# ====================
# SET UP PROBLEM
# ====================
dealias = 3 / 2
dtype = np.float64
timestepper = d3.RK443

stop_sim_time = float(args["--stop"])
stop_wall_time = np.inf
stop_iteration = np.inf

max_timestep = float(args["--maxdt"])
logger.info(f"max_timestep = {max_timestep}")

# ===Initialise basis===
coords = d3.CartesianCoordinates("x", "y", "z")
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords["x"], size=2, bounds=(0, Ly), dealias=dealias)
ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords["z"], size=Nz, bounds=(0, Lz), dealias=dealias)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
all_bases = (xbasis, ybasis, zbasis)
hor_bases = (xbasis, ybasis)

# Add fields (e.g. variables of the equations)
# Velocity
u = dist.VectorField(coords, name="u", bases=all_bases)
# Pressure
p = dist.Field(name="p", bases=all_bases)
# Temperature
Temp = dist.Field(name="Temp", bases=all_bases)

# Add Tau Terms
# Velocity tau terms, tau_u1 = (tau_1, tau_2)
tau_u1 = dist.VectorField(coords, name="tau_u1", bases=hor_bases)
tau_u2 = dist.VectorField(coords, name="tau_u2", bases=hor_bases)
# Temperature Tau Terms
tau_T3 = dist.Field(name="tau_T3", bases=hor_bases)
tau_T4 = dist.Field(name="tau_T4", bases=hor_bases)
# Scalar tau term for pressure gauge fixing
tau_p = dist.Field(name="tau_p")

# Substitutions
x_hat, y_hat, z_hat = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)  # Chebyshev U Basis
lift = lambda A: d3.Lift(A, lift_basis, -1)  # Shortcut for multiplying by U_{N-1}(y)
uz = d3.Differentiate(u, coords["z"])
Tz = d3.Differentiate(Temp, coords["z"])

u_x = u @ x_hat
u_y = u @ y_hat
u_z = u @ z_hat
dzu_y = d3.Differentiate(u_y, coords["z"])
dzu_x = d3.Differentiate(u_x, coords["z"])


f_cond = -d3.Differentiate(Temp, coords["z"])
f_conv = u_z * Temp
g_operator = d3.grad(u) - z_hat * lift(tau_u1)
h_operator = d3.grad(Temp) - z_hat * lift(tau_T3)
F = 1

# Add coriolis term
Tah = np.sqrt(Ta)
theta_deg = float(args["--theta"])
theta = theta_deg * np.pi / 180
# rotation vector
omega = dist.VectorField(coords, name="omega", bases=all_bases)
omega["g"][0] = 0
omega["g"][1] = np.sin(theta)
omega["g"][2] = np.cos(theta)

# #? =================
# #! HEATING FUNCTION
# #? =================
# Following Currie et al. 2020 Set-up B
# Width of middle 'convection zone' with no heating/cooling
heating_width = float(args["--Hwidth"])

H = Lz / (1 + 2 * heating_width)
# Width of heating and cooling layers
Delta = heating_width * H

heat = dist.Field(bases=zbasis)
if args["--currie"]:
    heat_func = lambda z: (F / Delta) * (
        1 + np.cos((2 * np.pi * (z - (Delta / 2))) / Delta)
    )
    cool_func = lambda z: (F / Delta) * (
        -1 - np.cos((2 * np.pi * (z - Lz + (Delta / 2))) / Delta)
    )

    heat["g"] = np.piecewise(
        z, [z <= Delta, z >= Lz - Delta], [heat_func, cool_func, 0]
    )
elif args["--kazemi"]:
    l = 0.1
    beta = 1
    a = 1 / (0.1 * (1 - np.exp(-1 / l)))
    heat_func = lambda z: a * np.exp(-z / l) - beta
    heat["g"] = heat_func(z)
else:
    #! === No Heating ===
    heat["g"] = np.zeros(heat["g"].shape)

# === Initialise Problem ===
problem = d3.IVP(
    [u, p, Temp, tau_u1, tau_u2, tau_T3, tau_T4, tau_p], time="t", namespace=locals()
)
# Thermal diffusion time
if args["--tau"] == "thermal":
    # Mass Conservation
    problem.add_equation("trace(g_operator) + tau_p= 0")  # needs a gauge fixing term
    # Momentum Equation
    problem.add_equation(
        "dt(u) - Pr * (div(g_operator)) + grad(p) - (Ra * Pr)*Temp*z_hat + lift(tau_u2) = - u@g_operator - (Tah / Pr) * cross(omega, u)"
    )
    # Temp Evolution
    problem.add_equation(
        "dt(Temp) + lift(tau_T4) -  (div(h_operator)) = -(u@h_operator) + heat"
    )
# Viscous diffusion time
elif args["--tau"] == "viscous":
    # Mass Conservation
    problem.add_equation("trace(g_operator) + tau_p= 0")  # needs a gauge fixing term
    # Momentum Equation
    problem.add_equation(
        "dt(u) - (div(g_operator)) + grad(p) - (Ra / Pr)*Temp*z_hat + lift(tau_u2) = - u@g_operator - Tah*cross(omega, u)"
    )
    # Temp Evolution
    problem.add_equation(
        "dt(Temp) + lift(tau_T4) - (1 / Pr) * (div(h_operator)) = -(u@h_operator) + (1 / Pr) * heat"
    )
else:
    raise ValueError(
        f'Invalid tau value {args["--tau"]}. Must be "viscous" or "thermal".'
    )

# ? === Driving Boundary Conditions ===
if args["--top"] == "insulating":
    problem.add_equation("Tz(z=Lz) = 0")
    boundary_conditions = "Insulating top"
elif args["--top"] == "vanishing":
    problem.add_equation("Temp(z=Lz) = 0")
    boundary_conditions = "Vanishing top"
elif args["--top"] == "fixed_flux":
    problem.add_equation("Tz(z=Lz) = -F")
    boundary_conditions = "Fixed Flux top"
else:
    raise ValueError(
        f"Invalid top boundary condition {args['--top']}, must be 'insulating', 'vanishing' or 'fixed_flux'"
    )

if args["--bottom"] == "insulating":
    problem.add_equation("Tz(z=0) = 0")
    boundary_conditions += "; Insulating bottom"
elif args["--bottom"] == "vanishing":
    problem.add_equation("Temp(z=0) = 0")
    boundary_conditions += "; Vanishing bottom"
elif args["--bottom"] == "fixed_flux":
    problem.add_equation("Tz(z=0) = -F")
    boundary_conditions += "; Fixed flux bottom"
else:
    raise ValueError(
        f'Invalid bottom boundary condition {args["--bottom"]}, must be "insulating", "vanishing" or "fixed_flux"'
    )


#! === Other ===
# * === Currie et al. 2020 ===
# # Fixed temp bottom, insulating top:
# problem.add_equation("Temp(z=0) = 0")
# problem.add_equation("Tz(z=Lz) = 0")

# ? === Velocity Boundary Conditions ===
# * === Stress-Free ===
# d(ux)/dz|(z=0, D) = 0
if args["--slip"] == "no":
    # * === No-Slip  ===
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("u(z=Lz) = 0")
    boundary_conditions += "; No-slip"
elif args["--slip"] == "free":
    # * === Free-Slip ===
    problem.add_equation("dzu_y(z=0) = 0")
    problem.add_equation("dzu_y(z=Lz) = 0")
    problem.add_equation("dzu_x(z=0) = 0")
    problem.add_equation("dzu_x(z=Lz) = 0")
    problem.add_equation("u_z(z=0) = 0")
    problem.add_equation("u_z(z=Lz) = 0")
    boundary_conditions += "; Free-Slip"
else:
    raise ValueError(
        f"invalid slip condition {args['--slip']}, must be 'no' or 'free'."
    )

# Pressure gauge fixing
problem.add_equation("integ(p) = 0")

logger.info("Boundary conditions: {}".format(boundary_conditions))

solver = problem.build_solver(timestepper)
logger.info("Solver built")

# ====================
# INITIAL CONDITIONS
# ====================
if args["--input"]:
    if pathlib.Path(restart_path + "snapshots/").exists():
        restart_file = sorted(glob(restart_path + "snapshots/*.h5"))[-1]
        write, last_dt = solver.load_state(restart_file, -1)
        dt = last_dt
        first_iter = solver.iteration
        fh_mode = "append"
    else:
        logger.error(
            "Problem reading file.\n{} does not exist.".format(
                restart_path + "snapshots_s1.h5"
            )
        )
        exit(-10)
else:
    # ? Need to change this to a better initial condition
    Temp.fill_random("g", seed=42, distribution="normal", scale=1e-5)
    # Temp.low_pass_filter(scales=0.25)
    # Temp.high_pass_filter(scales=0.125)
    Temp["g"] *= z * (Lz - z)
    # Temp["g"] += Lz - z

    first_iter = 0
    dt = max_timestep
    fh_mode = "overwrite"

if not args["--test"]:
    os.makedirs(outpath + "run_params/", exist_ok=True)
    run_params = {
        "Ly": Ly,
        "Lz": Lz,
        "Ny": Ny,
        "Nz": Nz,
        "Ra": Ra,
        "Pr": Pr,
        "Ta": Ta,
        "theta": theta_deg,
        "F": F,
        "max_timestep": max_timestep,
        "snapshot_iter": snapshot_iter,
        "horiz_iter": horiz_iter,
        "scalar_iter": scalar_iter,
    }
    run_params = json.dumps(run_params, indent=4)

    with open(outpath + "run_params/runparams.json", "w") as run_file:
        run_file.write(run_params)

    with open(outpath + "run_params/args.txt", "w") as file:
        file.write(" ".join(sys.argv))

    # ====================
    #   2.5D DATA FIELD
    # ====================
    snapshots = solver.evaluator.add_file_handler(
        outpath + "snapshots",
        iter=snapshot_iter,
        max_writes=5000,
        mode=fh_mode,
        parallel=parallel,
    )
    snapshots.add_tasks(solver.state, layout="g")
    snapshots.add_task(heat, name="heat", layout="g")
    # ==================
    #   HORIZONTAL AVE
    # ==================
    horiz_aves = solver.evaluator.add_file_handler(
        outpath + "horiz_aves",
        iter=horiz_iter,
        max_writes=2500,
        mode=fh_mode,
        parallel=parallel,
    )
    horiz_aves.add_task(
        d3.Integrate(d3.Integrate(Temp, "x"), "y") / (Ly * Ly), name="<T>", layout="g"
    )
    horiz_aves.add_task(
        d3.Integrate(d3.Integrate(f_cond, "x"), "y") / Ly, name="<F_cond>", layout="g"
    )
    horiz_aves.add_task(
        d3.Integrate(d3.Integrate(f_conv, "x"), "y") / Ly, name="<F_conv>", layout="g"
    )

    # ==================
    #      SCALARS
    # ==================
    scalars = solver.evaluator.add_file_handler(
        outpath + "scalars",
        iter=scalar_iter,
        max_writes=2500,
        mode=fh_mode,
        parallel=parallel,
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(0.5 * u @ u, "y"), "z"), "x")
        / (Lz * Ly),
        name="KE",
        layout="g",
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(np.sqrt(u @ u), "x"), "y"), "z")
        / (Lz * Ly),
        name="Re",
        layout="g",
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(Temp(z=0), "y"), "x") / Ly, name="<T(0)>", layout="g"
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(Temp, "x"), "y"), "z") / (Ly * Lz),
        name="<<T>>",
        layout="g",
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(f_cond + f_conv, "x"), "y"), "z")
        / (Ly * Lz),
        name="F_tot",
        layout="g",
    )

    # analysis = solver.evaluator.add_file_handler(
    #     outpath + "analysis",
    #     iter=analysis_iter,
    #     max_writes=5000,
    #     mode=fh_mode,
    #     parallel=parallel,
    # )
    # analysis.add_task(f_cond, name='F_cond', layout='g') #? F_cond
    # analysis.add_task(f_conv, name='F_conv', layout='g') #? F_conv
    # analysis.add_task(0.5*u@u, name='KE', layout='g') #? KE
    # analysis.add_task(d3.Integrate(Temp, 'y') / Ly, name='<T>y', layout='g') #? <T>y
    # analysis.add_task(d3.Integrate(d3.Integrate(Temp, 'y'), 'z') / (Lz*Ly), name='<T>', layout='g') #? <T>
    # analysis.add_task((d3.Integrate(f_cond, coords['z']) / Lz) / (d3.Integrate(f_conv, coords['z']) / Lz),
    #                   name='Nu_inst', layout='g') #? Nu_inst

solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = first_iter + stop_iteration + 1
solver.warmup_iterations = solver.iteration + 2000

CFL = d3.CFL(
    solver,
    initial_dt=dt,
    cadence=10,
    safety=0.5,
    threshold=0.1,
    max_change=1.5,
    min_change=0.5,
    max_dt=max_timestep,
)
CFL.add_velocity(u)
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u @ u), name="Re")
if args["--kill"]:
    exit(-99)
try:
    logger.info("Starting main loop")
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            max_Re = flow.max("Re")
            logger.info(
                "Iteration=%i,\n\tTime=%e, dt=%e, max(Re)=%f"
                % (solver.iteration - first_iter, solver.sim_time, timestep, max_Re)
            )
        if np.isnan(max_Re) or np.isinf(max_Re):
            raise NaNFlowError
except KeyboardInterrupt:
    logger.error("User quit loop. Triggering end of main loop")
    exit_code = -1
except NaNFlowError:
    logger.error("Max Re is NaN or inf. Triggering end of loop")
    exit_code = -50
except:
    logger.error("Unknown error raised. Triggering end of loop")
    exit_code = -10
finally:
    # if not args.test:
    #     # logger.info("Merging outputs...")
    #     # combine_outputs.merge_files(outpath)
    solver.evaluate_handlers_now(timestep)
    solver.log_stats()
    total_iterations = solver.iteration - first_iter
    snap_writes = (total_iterations) // snapshot_iter
    horiz_writes = (total_iterations) // horiz_iter
    scalar_writes = (total_iterations) // scalar_iter
    logger.info(
        "Snaps = {}, Horiz = {}, Scalars = {}".format(
            snap_writes, horiz_writes, scalar_writes
        )
    )
    logger.info("Written to {}".format(outpath))
    exit(exit_code)
