#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include "Solver.h"
#include <utility> 
#include <cmath>
#include <tuple>
#include <array>
using namespace std;


void Solver::iteration_step(){

    flux.compute_residual();
    update_delta_t();
    step();








}

void Solver::step(){
    
    for (Cell* cell : mesh.interior_cells){
        for(int i=0;i<4;i++)
            cell->U[i] = cell->U[i] - (cell->Residual[i]*cell->delta_t)/cell->Volume;
    }



}
void Solver::update_delta_t(){
    for (Cell* cell : mesh.interior_cells) 
        cell->delta_t = delta_t(cell);
}

static double psi(double gamma, double M) {
    return 1.0 + 0.5 * (gamma - 1.0) * M * M;
}

static double total_T(double gamma, double M, double T0) {
    // T = T0 / psi(gamma, M)
    return T0 / psi(gamma, M);
}

static double total_p(double gamma, double M, double P0) {
    // p = P0 / psi(gamma, M)^(gamma/(gamma-1))
    const double e = gamma / (gamma - 1.0);
    return P0 / std::pow(psi(gamma, M), e);
}

static double total_density(double P, double R, double T, double epsilon) {
    // rho = P / max(R*T, epsilon)
    return P / std::max(R * T, epsilon);
}

static double total_velocity(double gamma, double M, double R, double T) {
    // |u| = M * a = M * sqrt(gamma * R * T)
    return M * std::sqrt(gamma * R * T);
}








void Solver::set_boundary_conditions(){

    return;


}
void Solver::set_ambient_conditions(){
    cout << "Setting ambient Conditions...";
    for (Cell* cell : mesh.interior_cells){

        double p = total_p(gamma,inputs["mach"],inputs["p0"]);
        double rho = total_density(inputs["p0"],inputs["ru"],inputs["t0"],epsilon);
        flux.set_conserved(cell,{rho,0,0,p});
    }

    for (Cell* cell : mesh.ghost_cells){

        double p = total_p(gamma,inputs["mach"],inputs["p0"]);
        double rho = total_density(inputs["p0"],inputs["ru"],inputs["t0"],epsilon);
        flux.set_conserved(cell,{rho,0,0,p});
    }


}
void Solver::set_inflow_bcs(Cell* inflow_cell,double nx, double ny){

    //double nx = 1.0;
    //double ny = 0.0;
    double p = total_p(gamma,inputs["mach"],inputs["p0"]);
    double T = total_T(gamma,inputs["mach"],inputs["t0"]);
    double rho = total_density(inputs["p0"],inputs["ru"],T,epsilon);
    double u = nx*total_velocity(gamma,inputs["mach"],inputs["r0"],T);
    double v = ny*total_velocity(gamma,inputs["mach"],inputs["r0"],T);

    flux.set_conserved(inflow_cell,{rho,u,v,p});

}



void Solver::set_outflow_bcs(Cell* outflow_cell){
    Cell* neighbor = nullptr;
    if (outflow_cell->cell_L) 
        neighbor = outflow_cell->cell_L;
     else if (outflow_cell->cell_R) 
        neighbor = outflow_cell->cell_R;
     else if (outflow_cell->cell_U) 
        neighbor = outflow_cell->cell_U;
     else if (outflow_cell->cell_D) 
        neighbor = outflow_cell->cell_D;
    else
        throw std::invalid_argument("Cannot set outflow");
    for(int i=0; i<4;i++)
        outflow_cell->U[i] = neighbor->U[i];




}
void Solver::set_normal_bcs(Cell* normal_bc_cell){

    if(normal_bc_cell->type != 1){
        throw std::invalid_argument("Cannot set normal");
        return;
    }

    Cell* neighbor = nullptr;
    double nx;
    double ny;


    if (normal_bc_cell->cell_L) {
        neighbor = normal_bc_cell->cell_L;
        nx = neighbor->nx_R;
        ny = neighbor->ny_R;    
    } else if (normal_bc_cell->cell_R) {
        neighbor = normal_bc_cell->cell_R;
        nx = neighbor->nx_L;
        ny = neighbor->ny_L;    
    } else if (normal_bc_cell->cell_U) {
        neighbor = normal_bc_cell->cell_U;
        nx = neighbor->nx_D;
        ny = neighbor->ny_D;    
    } else if (normal_bc_cell->cell_D) {
        neighbor = normal_bc_cell->cell_D;
        nx = neighbor->nx_U;
        ny = neighbor->ny_U;    
    }
    if(neighbor==nullptr)
        throw std::invalid_argument("Cannot set normal");

    array<double,4> primvars = flux.get_primvars(neighbor);

    double u = primvars[1];
    double v = primvars[2];
    double dot = u * nx + v * ny;
    double u_out = u - 2.0 * dot * nx;
    double v_out = v - 2.0 * dot * ny;

    double p = primvars[3]; // TODO revisit this for second order
    double rho = primvars[0];

    flux.set_conserved(normal_bc_cell,{rho,u_out,v_out,p});
    



}





double Solver::delta_t(Cell* cell){


        array<double,4> V = flux.get_primvars(cell);
        double rho = V[0];
        double u = V[1];
        double v = V[2];
        double p = V[3];
    
        double a = sqrt(max(0.0,(gamma*(p/rho))));

        double nx_psi = (cell->nx_L + cell->nx_R) / 2.0;
        double ny_psi = (cell->ny_L + cell->ny_R) / 2.0;

        double nx_eta = (cell->nx_U + cell->nx_D) / 2.0;
        double ny_eta = (cell->ny_U + cell->ny_D) / 2.0;

        double Area_LR = 0.5 * (cell->A_L + cell->A_R);
        double Area_UD = 0.5 * (cell->A_U + cell->A_D);

        double lambda_LR = std::abs(u * nx_psi + v * ny_psi) + a;
        double lambda_UD = std::abs(u * nx_eta + v * ny_eta) + a;

        double delta_t_ = cell->Volume / std::max(epsilon, lambda_LR * Area_LR + lambda_UD * Area_UD);
        return CFL*delta_t_;
        // self.delta_t =self.CFL*delta_t




}


