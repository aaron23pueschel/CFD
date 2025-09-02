
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

    
    Simulation test("airfoil_inputs.json");
    test.simulation_solver.set_ambient_conditions();
    test.simulation_solver.set_flow_initial_conditions();
    const string primvar = "primitives.csv";

    for(int i=0;i<5550;i++){
       cout << "Iteration: " << i<<endl; 
       test.simulation_solver.iteration_step();
    }
     //test.simulation_solver.iteration_step();
     

    //test.write_primitives_csv(primvar);
    test.write_primitives_csv(primvar);
    //test.write_residuals_csv(primvar);
    auto norms = test.simulation_flux.compute_norm();
    cout << "Norms:\n"
     << "  Density   : "   << norms[0] << "\n"
     << "  U-velocity: "   << norms[1] << "\n"
     << "  V-velocity: "   << norms[2] << "\n"
     << "  Pressure  : "   << norms[3] << "\n";
    
   
    //     cout<<"Iteration "<<i<<endl;
    // }
    // //}
    // for(Cell* c : test.simulation_mesh.interior_cells){
    //     cout<<c->Residual[1]<<"\n";
    // }

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
    Mesh mesh("Ramp","Ramp_Coarse_cells_xx.csv","Ramp_Coarse_cells_yy.csv");
    
    //mesh.set_mesh();
    //mesh.check_cell_points();
    //mesh.print_mesh();
    mesh.check_divergence();
    //mesh.test_boundary_normals();
    mesh.check_mesh_cellwise();
    //mesh.check_mesh();
    //mesh.check_interior_cells();
    //mesh.set_ramp_boundary_types();
    //mesh.set_square_boundary_types();
    return 0;
}


