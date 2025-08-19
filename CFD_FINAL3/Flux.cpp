#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include <utility> 
#include <cmath>
#include <tuple>
#include <array>
using namespace std;




array<double, 4> Flux::vanleer_flux(double* UL, double* UR, double nx_L, double nx_R) {
    array<double, 4> flux = {1.0, 2.0, 3.0, 4.0};

    return flux;

}




void Flux::compute_residual(){

    std::array<double, 4> FL = {0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> FR = {0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> FD = {0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> FU = {0.0, 0.0, 0.0, 0.0};
  
    for (Cell* cell : mesh.interior_cells) {

        // Upwind here
        if (cell->cell_L) 
            FL = vanleer_flux(cell->cell_L->U, cell->U, cell->nx_L, cell->ny_L);
        if (cell->cell_R) 
            FR = vanleer_flux(cell->U, cell->cell_R->U, cell->nx_R, cell->ny_R);
        if (cell->cell_D) 
            FD = vanleer_flux(cell->cell_D->U, cell->U, cell->nx_D, cell->ny_D);
        if (cell->cell_U) 
            FU = vanleer_flux(cell->U, cell->cell_U->U, cell->nx_U, cell->ny_U);
        
        // Compute residual
        for(int i=0;i<4;i++)
            cell->Residual[i] = (FL[i]*cell->A_L + FR[i]*cell->A_R + FU[i]*cell->A_U + FD[i]*cell->A_D);
            
        cout << cell->Residual[1] <<"\n";
    }

    

    



}