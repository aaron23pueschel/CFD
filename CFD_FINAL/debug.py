from EulerSolver import EulerSolver
from DataStructures import DataStructures
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.getcwd())

Test = EulerSolver("CFD_FINAL/inputs/Medium_inputs.nml")
#Test = EulerSolver("inputs/Medium_inputs.nml")
Test.set_initial_conditions()
Test.set_inflow_bcs()
Test.set_pressure_bc()
Test.set_RK4_vals()


for i in range(1):
    Test.set_pressure_bc()
    Test.set_inflow_bcs()
    s = Test.RK_iteration(type_ = "Euler")
    Test.set_pressure_bc()



    
    print(np.linalg.norm(s[3,:,:]))
