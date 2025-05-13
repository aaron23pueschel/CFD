from EulerSolver import EulerSolver
from DataStructures import DataStructures
import matplotlib.pyplot as plt
import numpy as np


Test = EulerSolver("AOE_6145/CFD/CFD_FINAL/inputs/Medium_inputs.nml")
Test.set_initial_conditions()
Test.Data.plot_primitive("pressure")
Test.set_inflow_bcs()
Test.set_pressure_bc()

s = Test.iteration_step(delta_t = .1)