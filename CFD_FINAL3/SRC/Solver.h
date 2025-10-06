#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <utility> 
#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include <array>
#include <map>




using namespace std;
class Solver{
public:


    const double gamma = 1.4;
    int upwind_order = 1;
    Mesh mesh;
    Flux flux;
    map<string,double> inputs;
    double CFL;
    double kappa;
    double epsilon;










    
    bool local_timestep;
    

    
    


    Solver(Mesh mesh_, Flux flux_, map<string,double> inputs_,double cfl_,double local_timestep_) : mesh(mesh_),flux(flux_),inputs(inputs_), CFL(cfl_),local_timestep(local_timestep_){}




    void iteration_step(int i);


    void update_delta_t();          
    double delta_t(Cell* cell);
    void step();
    void set_flow_initial_conditions();
    void set_boundary_conditions();
    void set_normal_bcs(Cell* cell);
    void set_outflow_bcs(Cell* cell);
    void set_inflow_bcs(Cell* cell,double nx,double ny);
    void set_ambient_conditions();
    void set_mms_bcs(Cell* outflow_cell);
    void RK4_step();
    void set_MMS_initial_conditions();
    void RK2_step();











};
