
#include <iostream>
#include "SRC/Cell.h"
#include "SRC/Mesh.h"
#include "SRC/Flux.h"
#include "SRC/Solver.h"
#include "SRC/Simulation.h"
#include "SRC/inputs.h"

using namespace std;
int testmesh();

int main(){
    
    Simulation test("inputs/curve_inputs.json");
    test.simulation_solver.set_ambient_conditions();
    test.simulation_solver.set_boundary_conditions();
    test.simulation_solver.flux.set_MMS_source(true);
     test.simulation_flux.compute_residual();
    
    // //test.simulation_solver.set_flow_initial_conditions();
    const string primvar = "primitives.csv";
    //test.write_primitives_csv(primvar);
    // // test.simulation_mesh.test_boundary_normals();
     for(int i=0;i<4000;i++){
    //     cout << "Iteration: " << i<<endl; 
    //      test.simulation_solver.iteration_step();
    //  }
    test.simulation_solver.iteration_step(1);
      auto norms = test.simulation_flux.compute_norm();

     cout<< "  Density   : "   << norms[0] 
     << "  U-velocity: "   << norms[1] 
     << "  V-velocity: "   << norms[2]  
     << "  Pressure  : "   << norms[3] << "\n";
    
     }

    test.write_primitives_csv(primvar);

    return 0;
}


int testmesh() {
    
    //Mesh mesh("test",10,10);
    //mesh.set_uniform_points();
    Mesh mesh("Curve","Meshes/Curvilinear_xx.csv","Meshes/Curvilinear_yy.csv");
    
    //mesh.set_mesh();
    //mesh.check_cell_points();
    //mesh.print_mesh();
    mesh.check_divergence();
    mesh.test_boundary_normals();
    mesh.check_mesh_cellwise();
    mesh.check_mesh();
    mesh.check_interior_cells();
    //mesh.set_ramp_boundary_types();
    //mesh.set_square_boundary_types();
    return 0;
}


