from Nozzle_1D import Nozzle
import matplotlib.pyplot as plt
import numpy as np
import pickle


tolerance = 1e-12

# Initialize MEDIUM ISENTROPIC CASE
MEDIUM_ISENTROPIC = Nozzle("inputs/Medium_inputs.nml")
MEDIUM_ISENTROPIC.CFL = .05
MEDIUM_ISENTROPIC.p_back = 120
# RUN MEDIUM SIMULATION
p_compute_Medium,u_compute_Medium,rho_compute_Medium,p_exact_Medium,\
    u_exact_Medium,rho_exact_Medium,convergence_history_Medium = MEDIUM_ISENTROPIC.RUN_SIMULATION(verbose = True,convergence_criteria=tolerance,jameson_damping = False,iter_max=1,first_order=True,local_timestep=True)