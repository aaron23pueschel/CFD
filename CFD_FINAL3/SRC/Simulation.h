#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <utility> 
#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include "Solver.h"
#include "inputs.h"
#include <map>
#include <array>

class Simulation{
public:

    Inputs in;
    Mesh simulation_mesh;
    Flux simulation_flux;
    Solver simulation_solver;


    Simulation(const string& inputs_str)
        : in(load_inputs(inputs_str)),
          simulation_mesh(in.MeshName,
                          in.xx_filename, in.yy_filename),
          simulation_flux(simulation_mesh, in.upwind_order,
                          in.kappa, in.epsilon,in.damping_scheme),
          simulation_solver(simulation_mesh, simulation_flux,
                            {
                                {"p0", in.p0},
                                {"t0", in.t0},
                                {"ru", in.ru},
                                {"mach", in.mach},
                                {"is_mms",in.is_mms}
                            },
                            in.cfl, in.local_timestep)
    {}





    Simulation(const Inputs& input_config)
        : in(input_config),
          simulation_mesh(input_config.MeshName,
                          input_config.xx_filename, input_config.yy_filename),
          simulation_flux(simulation_mesh, input_config.upwind_order,
                          input_config.kappa, input_config.epsilon,input_config.damping_scheme),
          simulation_solver(simulation_mesh, simulation_flux,
                            {
                                {"p0", input_config.p0},
                                {"t0", input_config.t0},
                                {"ru", input_config.ru},
                                {"mach", input_config.mach}
                            },
                            input_config.cfl, input_config.local_timestep)
    {}

    void write_primitives_csv(const string& filename);
    Inputs load_inputs(const string& filename);
    void write_residuals_csv(const string& filename);
    void write_Cd(const string& filename);
    void write_sources_csv(const string& filename);



};



