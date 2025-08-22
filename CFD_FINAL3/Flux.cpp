#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include <utility> 
#include <cmath>
#include <tuple>
#include <array>
using namespace std;




array<double, 4> Flux::vanleer_flux(double* U_L, double* U_R, double nx, double ny) {
    
    double rho_L = U_L[0];
    double u_L   = U_L[1];
    double v_L   = U_L[2];
    double p_L   = U_L[3];

    double rho_R = U_R[0];
    double u_R   = U_R[1];
    double v_R   = U_R[2];
    double p_R   = U_R[3];



    double Uhat_L = u_L * nx + v_L * ny;
    double aL = sqrt(gamma * p_L/ rho_L);
    double ML = Uhat_L / aL;
    double htL = (gamma / (gamma - 1)) * (p_L / rho_L) + .5 * (u_L * u_L + v_L * v_L);


    double Uhat_R = u_R * nx + v_R * ny;
    double aR = sqrt(gamma * p_R/ rho_R);
    double MR = Uhat_R / aR;
    double htR = (gamma / (gamma - 1)) * (p_R / rho_R) + .5 * (u_R * u_R + v_R * v_R);




    // Flux sp_Litting
    double alpha_p = .5 * (1.0 + copysign(1.0, ML));
    double alpha_m = .5 * (1.0 - copysign(1.0, MR));

    double betaL = -std::max(0.0, static_cast<double>(1 - static_cast<int>(std::abs(ML))));
    double betaR = -std::max(0.0, static_cast<double>(1 - static_cast<int>(std::abs(MR))));


    double Mp = .25 * pow((ML + 1.0), 2);
    double Mm = -.25 * pow((MR - 1.0), 2);

    double Cp = alpha_p * (1.0 + betaL) * ML - betaL * Mp;
    double Cm = alpha_m * (1.0 + betaR) * MR - betaR * Mm;

    array<double, 4> F_convective;
    F_convective[0] = rho_L * aL * Cp * 1.0 + rho_R * aR * Cm * 1.0;
    F_convective[1] = rho_L * aL * Cp * u_L + rho_R * aR * Cm * u_R;
    F_convective[2] = rho_L * aL * Cp * v_L + rho_R * aR * Cm * v_R;
    F_convective[3] = rho_L * aL * Cp * htL + rho_R * aR * Cm * htR;

    double Pp = Mp * (-ML + 2.0);
    double Pm = Mm * (-MR - 2.0);

    double Dp = (alpha_p * (1.0 + betaL)) - (betaL * Pp);
    double Dm = (alpha_m * (1.0 + betaR)) - (betaR * Pm);

    array<double, 4> F_pressure;
    F_pressure[0] = 0.0;
    F_pressure[1] = Dp * (nx * p_L) + Dm * (nx * p_R);
    F_pressure[2] = Dp * (ny * p_L) + Dm * (ny * p_R);
    F_pressure[3] = 0.0;

    array<double, 4> F;
    for (int i = 0; i < 4; ++i) 
        F[i] = F_convective[i]+ F_pressure[i];
    
    return F;

}

























void Flux::compute_residual(){

    array<double, 4> FL = {0.0, 0.0, 0.0, 0.0};
    array<double, 4> FR = {0.0, 0.0, 0.0, 0.0};
    array<double, 4> FD = {0.0, 0.0, 0.0, 0.0};
    array<double, 4> FU = {0.0, 0.0, 0.0, 0.0};
  
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

        //cout << cell->Residual[1]<< ", " << cell->Residual[2];
    }

    

    



}



array<double, 4> Flux::get_primvars(double* U){

    double rho = max(U[0],epsilon);
    double u = U[1] / rho;
    double v = U[2]/rho;
    double p = max(0.0, (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v)));


    return {rho,u,v,p};
}










array<double, 4> Flux::get_primvars(Cell* cell){

    double rho = max(cell->U[0],epsilon);
    double u = cell->U[1] / rho;
    double v = cell->U[2]/rho;
    double p = max(0.0, (gamma - 1.0) * (cell->U[3] - 0.5 * rho * (u * u + v * v)));


    return {rho,u,v,p};
}

void Flux::set_conserved(Cell* cell, array<double, 4> primitive){

        double rho = primitive[0];
        double u = primitive[1];  
        double v = primitive[2];
        double p = primitive[3];  
        
        
        double et = p / ((gamma - 1) * rho) + 0.5 * (u*u+v*v);

        cell->U[0] = rho;
        cell->U[1] = rho*u;
        cell->U[2] = rho*v;
        cell->U[3] = rho*et;
         


}
