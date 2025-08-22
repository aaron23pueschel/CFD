#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <utility> 
#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include "Solver.h"
#include "ramp_inputs.h"
#include <map>
#include <array>

class Simulation{
public:

    Inputs in;
    Mesh simulation_mesh;
    Flux simulation_flux;
    Solver simulation_solver;

    Simulation(const Inputs& input_config)
        : in(input_config),
          simulation_mesh(input_config.MeshName,
                          input_config.xx_filename, input_config.yy_filename),
          simulation_flux(simulation_mesh, input_config.upwind_order,
                          input_config.kappa, input_config.epsilon),
          simulation_solver(simulation_mesh, simulation_flux,
                            {
                                {"p0", input_config.p0},
                                {"t0", input_config.t0},
                                {"ru", input_config.ru},
                                {"mach", input_config.mach}
                            },
                            input_config.cfl, input_config.local_timestep)
    {}






};



