

from Nozzle_2D_checkpoint4 import Nozzle
import matplotlib.pyplot as plt
import numpy as np
import pickle

SUPERCOARSE = Nozzle("CFD/CFD_FINAL2/inputs/Medium_inputs.nml")
SUPERCOARSE.CFL = .1


self = SUPERCOARSE 
self.CFL = .1
self.iter_max = 78000

# RUN MEDIUM SIMULATION
self.set_arrays()
#self.set_geometry()
self.set_curved_geometry()
#self.set_nozzle_BC()
#self.set_ramp_BC()
#self.load_grid("Meshes/Inlet.53x17.grd")
#self.load_grid("Meshes/Inlet.417x129.grd") # boundary conditions increments of 20,40,80,160
self.compute_all_areas()
self.set_initial_conditions()
self.set_boundary_conditions()
self.set_normals()
#R1 = self.iteration_step()
#self.set_boundary_conditions()


self.set_boundary_conditions()
self.set_normal_bcs()
R1 = self.iteration_step()
plt.imshow(self.V[2,::-1])
plt.colorbar()