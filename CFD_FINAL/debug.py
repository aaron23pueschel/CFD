from Nozzle_1D import Nozzle
import matplotlib.pyplot as plt
import numpy as np
import pickle


TestNozzle = Nozzle("inputs/Medium_inputs.nml")
TestNozzle.set_arrays()
TestNozzle.set_curved_geometry()
TestNozzle.plot_Geometry()
TestNozzle.set_initial_conditions()
TestNozzle.set_normals()
TestNozzle.compute_all_areas()
TestNozzle.set_boundary_conditions()
#TestNozzle.nx_ny_right[0].shape
#plt.scatter(TestNozzle.V[:,:,3])
TestNozzle.plot_primitive(type_="velocity")
#TestNozzle.set_boundary_conditions()


s = TestNozzle.iteration_step(return_error=True)
