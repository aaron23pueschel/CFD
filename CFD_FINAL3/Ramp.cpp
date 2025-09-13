
#include <iostream>
#include "SRC/Cell.h"
#include "SRC/Mesh.h"
#include "SRC/Flux.h"
#include "SRC/Solver.h"
#include "SRC/Simulation.h"
#include "SRC/inputs.h"

using namespace std;
int testmesh();

extern "C" {

// rho(x,y; L) → rho (out)
void rmassconv(double* length, double* x, double* y, double* rho);


}
int main(){

    Simulation test("inputs/ramp_inputs.json");
    test.simulation_solver.set_ambient_conditions();
    test.simulation_solver.set_flow_initial_conditions();
    test.simulation_flux.set_MMS_source(test.in.is_mms);
    const string primvar = "primitives.csv";
    for(int i=0;i< 7006;i++){
       cout << "Iteration: " << i<<endl; 

       test.simulation_solver.iteration_step(i);
       
        if(i<1000)
            test.simulation_solver.flux.upwind_order = 0;
        else
            test.simulation_solver.flux.upwind_order = 1;



       auto norms = test.simulation_flux.compute_norm();
    cout <<"Upwind Order: "<<test.simulation_flux.upwind_order<< ";  Norms:\n"
     << "  Density   : "   << norms[0] 
     << "  U-velocity: "   << norms[1] 
     << "  V-velocity: "   << norms[2]  
     << "  Pressure  : "   << norms[3] << "\n";
    
    }
     //test.simulation_solver.iteration_step();
     
    //test.write_residuals_csv(primvar);
    test.write_residuals_csv(primvar);
    // test.write_primitives_csv(primvar);
    
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


