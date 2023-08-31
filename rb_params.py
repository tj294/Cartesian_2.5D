import numpy as np
from dedalus.public import RK443

Ly = 2
Lz = 1

heating_width = 0.2
convection_height = Lz / (1 + 2 * heating_width)

Ny, Nz = 256, 128
dealias = 3 / 2

F = 1
Ra = 1e6
Pr = 1
Ta = 1e3
theta = np.pi / 4

timestepper = RK443

dt = 5e-5
max_timestep = 5e-5

stop_sim_time = 1
stop_wall_time = np.inf
end_iteration = np.inf

snapshot_iter = 500
horiz_iter = 100
scalar_iter = 1
