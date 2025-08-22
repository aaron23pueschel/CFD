
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

    Mesh mesh("Ramp","Ramp_Coarse_cells_xx.csv","Ramp_Coarse_cells_yy.csv");
    mesh.set_ramp_boundary_types();
    // for(Cell* c: mesh.ghost_cells){

    //     cout<< c->type<<endl;
    // }
    //testmesh();
    // for(int i=0;i<mesh.NI;i++){
    //     std::cout << "Cell " << i << ":\n";
    //     std::cout << "  Left:  " 
    //             << mesh.interior_cells[i]->nx_L << ", " 
    //             << mesh.interior_cells[i]->cell_L->nx_R << "\n";
    //     std::cout << "  Right: " 
    //             << mesh.interior_cells[i]->nx_R << ", " 
    //             << mesh.interior_cells[i]->cell_R->nx_L << "\n";
    //     std::cout << "  Up:    " 
    //             << mesh.interior_cells[i]->nx_U << ", " 
    //             << mesh.interior_cells[i]->cell_U->nx_D << "\n";
    //     std::cout << "  Down:  " 
    //             << mesh.interior_cells[i]->nx_D << ", " 
    //             << mesh.interior_cells[i]->cell_D->nx_U << "\n";
    // }
    Inputs inputs;
    Simulation test(inputs);
    test.simulation_solver.set_ambient_conditions();

    for(int i=0;i<100;i++){
        test.simulation_solver.iteration_step();
        cout<<"Iteration "<<i<<endl;
    }
    //}
    for(Cell* c : test.simulation_mesh.interior_cells){
        cout<<c->Residual[1]<<"\n";
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


