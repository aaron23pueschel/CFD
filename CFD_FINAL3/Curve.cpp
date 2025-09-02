
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
    
    Simulation test("curve_inputs.json");
    //test.simulation_mesh.set_ramp_boundary_types();
    // test.simulation_mesh.set_square_boundary_types();
    // for (Cell* c : test.simulation_mesh.ghost_cells) {
    //     if (c->cell_L && c->cell_L->cell_R)
    //         std::cout << "Left->Right: " << c->cell_L->cell_R->type << std::endl;

    //     if (c->cell_R && c->cell_R->cell_L)
    //         std::cout << "Right->Left: " << c->cell_R->cell_L->type << std::endl;

    //     if (c->cell_U && c->cell_U->cell_D)
    //         std::cout << "Up->Down: " << c->cell_U->cell_D->type << std::endl;

    //     if (c->cell_D && c->cell_D->cell_U)
    //         std::cout << "Down->Up: " << c->cell_D->cell_U->type << std::endl;

    // }

    //testmesh();
    // 
    test.simulation_solver.set_ambient_conditions();
    test.simulation_solver.set_boundary_conditions();
     test.simulation_flux.compute_residual();
    // //test.simulation_solver.set_flow_initial_conditions();
    const string primvar = "primitives.csv";
    // // test.simulation_mesh.test_boundary_normals();
     for(int i=0;i<40;i++)
    //     cout << "Iteration: " << i<<endl; 
    //      test.simulation_solver.iteration_step();
    //  }
    test.simulation_solver.iteration_step();
    
    Cell* iterator = test.simulation_mesh.interior_cells[0];
   
    // //test.write_primitives_csv(primvar);
    test.write_primitives_csv(primvar);
    // test.write_residuals_csv(primvar);
    // auto norms = test.simulation_flux.compute_norm();
    // cout << "Norms:\n"
    //  << "  Density   : "   << norms[0] << "\n"
    //  << "  U-velocity: "   << norms[1] << "\n"
    //  << "  V-velocity: "   << norms[2] << "\n"
    //  << "  Pressure  : "   << norms[3] << "\n";
    
   
    //     cout<<"Iteration "<<i<<endl;
    // }
    // //}
     for(Cell* c : test.simulation_mesh.interior_cells){
         cout<<"NX_L: "<<c->nx_L<<",  "<<c->ny_L<<"   ";
         auto cL = (c->cell_L->U);
         auto cR = (c->U);
         auto temp= test.simulation_flux.vanleer_flux({cL[0],cL[1],cL[2],cL[3]},{cR[0],cR[1],cR[2],cR[3]},c->nx_L,c->ny_L);
         cout << temp[0]<<"  "<<temp[1]<< "  "<<temp[2]<<"   "<<temp[3]<<endl;
     }

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


