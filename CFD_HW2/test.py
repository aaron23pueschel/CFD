from Nozzle_1D import Nozzle
temp = Nozzle("inputs.nml")
import matplotlib.pyplot as plt

temp.set_geometry()
temp.set_initial_conditions()
temp.set_boundary_conditions()
temp.set_conserved_variables()
temp.set_primitive_variables()
temp.set_source_term()

temp.iteration_step()
temp.update_all()