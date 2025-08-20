
#include <iostream>
#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include "Solver.h"
#include "Simulation.h"
#include "ramp_inputs.h"

using namespace std;
int testmesh();

int main(){
    
    Inputs inputs;
    Simulation test(inputs);
    test.simulation_solver.set_ambient_conditions();
    test.simulation_solver.iteration_step();
    test.simulation_mesh.print_conserved();
    //Mesh mesh("Coarse Ramp",17,53,"Ramp_Coarse_xx.bin","Ramp_Coarse_yy.bin");
    //Mesh mesh("test","Ramp_Coarse_xx.csv","Ramp_Coarse_yy.csv");
    //mesh.print_conserved();
    //Mesh::set_mesh();
    //auto temp = Mesh::load_csv_file("Ramp_Coarse_xx.bin");
    //mesh.write_vector_to_binary(temp,"test_file.bin");
    
    //testmesh();
    //Mesh mesh("Coarse Ramp",17,53,"Ramp_Coarse_xx.bin","Ramp_Coarse_yy.bin");
    
    //Flux flux(mesh);

    //flux.compute_fluxes();
    //flux.compute_residual();
    return 0;
}


int testmesh() {
    
    //Mesh mesh("test",10,10);
    //mesh.set_uniform_points();
    Mesh mesh("Coarse Ramp",17,53,"Ramp_Coarse_xx.bin","Ramp_Coarse_yy.bin");
    mesh.set_mesh();
    mesh.check_cell_points();
    mesh.test_boundary_normals();
    mesh.check_mesh_cellwise();
    mesh.check_mesh();
    mesh.check_interior_cells();
    mesh.set_ramp_boundary_types();
    mesh.set_square_boundary_types();
    return 0;
}


