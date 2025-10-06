#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include "Solver.h"
#include <utility> 
#include <cmath>
#include <tuple>
#include <array>
#include <cmath>       // std::isnan, std::isinf
#include <stdexcept>   // std::runtime_error
#include <string>      // std::to_string
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;








void print_norm(const array<double,4>& norm) {


cout << 
    "Norms:  Density   : " << norm[0] << 
    " U-velocity: " << norm[1] << 
    "  V-velocity: " << norm[2] << 
    "  Pressure  : " << norm[3] << "\n";

}







void Solver::iteration_step(int i){

    set_boundary_conditions();
    flux.compute_residual();
    update_delta_t();
    step();
    //RK4_step();
    //RK2_step();


}
void Solver::RK2_step() {
    const bool is_mms = inputs["is_mms"];

    // 1) Save U^n
    for (Cell* c : mesh.interior_cells)
        for (int i = 0; i < 4; ++i)
            c->temp_U[i] = c->U[i];

    // ---------- Stage 1: K1 = L(U^n) ----------
    set_boundary_conditions();
    flux.compute_residual();                 // fills c->Residual for current c->U

    for (Cell* c : mesh.interior_cells) {
        const double dt = c->delta_t;        // keep same dt across both stages per cell
        for (int i = 0; i < 4; ++i) {
            // L(U) = -(Residual/Vol - Source)   (your sign convention)
            c->K1[i] = - (c->Residual[i] / c->Volume - c->Source[i]);
            c->U[i]  =  c->temp_U[i] + dt * c->K1[i];   // U^(1)
        }
    }

    // ---------- Stage 2: K2 = L(U^(1)) ----------
    set_boundary_conditions();               // BCs must be consistent with stage state
    flux.compute_residual();

    for (Cell* c : mesh.interior_cells) {
        const double dt = c->delta_t;
        for (int i = 0; i < 4; ++i) {
            c->K2[i] = - (c->Residual[i] / c->Volume - c->Source[i]);

            // U^{n+1} = 0.5*U^n + 0.5*( U^(1) + dt*K2 )
            const double U1_i   = c->U[i];             // currently holds U^(1)
            const double update = U1_i + dt * c->K2[i];
            c->U[i] = 0.5 * c->temp_U[i] + 0.5 * update;

            // optional diagnostics
            c->total_residual[i] = c->K1[i] + c->K2[i];
        }
    }
}



void Solver::RK4_step() {

    for (Cell* c : mesh.interior_cells)
        for (int i = 0; i < 4; ++i)
            c->temp_U[i] = c->U[i];

    auto compute_stage = [&](int stage){
        set_boundary_conditions();
        flux.compute_residual();      
        

        for (Cell* c : mesh.interior_cells) {
            for (int i = 0; i < 4; ++i) {
                const double f = - (c->Residual[i] / c->Volume - c->Source[i]); // slope f(U)
                if      (stage == 1) c->K1[i] = f;
                else if (stage == 2) c->K2[i] = f;
                else if (stage == 3) c->K3[i] = f;
                else                  c->K4[i] = f;
            }
        }
    };

    compute_stage(1);


    for (Cell* c : mesh.interior_cells) {
        const double dt = c->delta_t;   
        for (int i = 0; i < 4; ++i)
            c->U[i] = c->temp_U[i] + 0.5 * dt * c->K1[i];
    }
    compute_stage(2);


    for (Cell* c : mesh.interior_cells) {
        const double dt = c->delta_t;
        for (int i = 0; i < 4; ++i)
            c->U[i] = c->temp_U[i] + 0.5 * dt * c->K2[i];
    }
    compute_stage(3);


    for (Cell* c : mesh.interior_cells) {
        const double dt = c->delta_t;
        for (int i = 0; i < 4; ++i)
            c->U[i] = c->temp_U[i] + dt * c->K3[i];
    }
    compute_stage(4);


    for (Cell* c : mesh.interior_cells) {
        const double dt = c->delta_t;
        for (int i = 0; i < 4; ++i) {
            c->total_residual[i] = (c->K1[i] + 2.0*c->K2[i] + 2.0*c->K3[i] + c->K4[i]);
            c->U[i] = c->temp_U[i]
                    + (dt/6.0) * (c->K1[i] + 2.0*c->K2[i] + 2.0*c->K3[i] + c->K4[i]);
        }
    }
}



void Solver::step(){
    bool is_mms = inputs["is_mms"];
    for (Cell* cell : mesh.interior_cells){
        if (is_mms &&( (cell->cell_L && cell->cell_L->name == "Ghost") ||
     (cell->cell_R && cell->cell_R->name == "Ghost") ||
     (cell->cell_U && cell->cell_U->name == "Ghost") ||
     (cell->cell_D && cell->cell_D->name == "Ghost") ))
     continue;
        for(int i=0;i<4;i++){
            cell->total_residual[i] = (((cell->Residual[i]))/cell->Volume-cell->Source[i]);
            cell->U[i] = cell->U[i] - (((cell->Residual[i]))/cell->Volume-cell->Source[i])*cell->delta_t;
        }

        
    }



}
void Solver::update_delta_t(){
    double min = 10000.0;


    if(local_timestep)
        for (Cell* cell : mesh.interior_cells)
            cell->delta_t = delta_t(cell);
    else{
        for (Cell* cell : mesh.interior_cells){
            double temp_min = delta_t(cell);
                if(temp_min<min)
                    min = temp_min;
        }

        for (Cell* cell : mesh.interior_cells){
            cell->delta_t = min;
        }

    }
    
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


void Solver::set_mms_bcs(Cell* outflow_cell){

    Cell* neighbor = nullptr;
    double midpointx;
    double midpointy;
    if (outflow_cell->cell_L){
        neighbor = outflow_cell->cell_L;
        midpointx = (neighbor->x12 +neighbor->x22)/2;
        midpointy = (neighbor->y12 +neighbor->y22)/2;
    }
     else if (outflow_cell->cell_R){
        neighbor = outflow_cell->cell_R;
        midpointx = (neighbor->x11 +neighbor->x21)/2;
        midpointy = (neighbor->y11 +neighbor->y21)/2;
     }
     else if (outflow_cell->cell_D){
        neighbor = outflow_cell->cell_D;
        midpointx = (neighbor->x21 +neighbor->x22)/2;
        midpointy = (neighbor->y21 +neighbor->y22)/2;
     }
     else if (outflow_cell->cell_U) {
        neighbor = outflow_cell->cell_U;
        midpointx = (neighbor->x11 +neighbor->x12)/2;
        midpointy = (neighbor->y11 +neighbor->y12)/2;

     }

    
    else
        throw std::invalid_argument("Cannot set outflow");

    midpointx = neighbor->midpoint_x;
    midpointy = neighbor->midpoint_y;
     
    double length = 1.0;
    double gamma = 1.4;
    
    array<double,4> temp;

    temp[0] = flux.rho_mms(length,midpointx,midpointy);
    temp[1] = flux.uvel_mms(length,midpointx,midpointy);
    temp[2] = flux.vvel_mms(length,midpointx,midpointy);
    temp[3] = flux.press_mms(length,midpointx,midpointy);

    flux.set_conserved(outflow_cell,temp);
    flux.set_conserved(neighbor,temp);
    
    // for(int i=0;i<4;i++)
    //     cout << outflow_cell->U[i] <<",  "<<endl;



}




void Solver::set_boundary_conditions(){
    bool use_normal_bcs = true;
    for (Cell* c : mesh.ghost_cells) {

        
        if(c->type == 2){
            
            
            set_inflow_bcs(c,1,0);
        }

        if(c->type == 0)
            set_outflow_bcs(c);

        if(c->type==1 && use_normal_bcs)
            set_normal_bcs(c);

        if(c->type==3)
            set_mms_bcs(c);

        

    }


}



void Solver::set_MMS_initial_conditions(){
    cout << "Setting inflow Conditions...";

    
    

    for (Cell* cell : mesh.interior_cells){
        double rho_ = flux.rho_mms(1.0,cell->midpoint_x,cell->midpoint_y);
        double u_ = flux.uvel_mms(1.0,cell->midpoint_x,cell->midpoint_y);
        double v_ = flux.vvel_mms(1.0,cell->midpoint_x,cell->midpoint_y);
        double p_ = flux.press_mms(1.0,cell->midpoint_x,cell->midpoint_y);
        flux.set_conserved(cell,{rho_,u_,v_,p_});
    }



    

}














void Solver::set_flow_initial_conditions(){
    cout << "Setting inflow Conditions...";

    // double p = total_p(gamma,inputs["mach"],inputs["p0"]);
    // double T = total_T(gamma,inputs["mach"],inputs["t0"]);
    // double rho = total_density(p,inputs["ru"],T,epsilon);
    // double pi = 3.1415926;
    // double u = total_velocity(gamma,inputs["mach"],inputs["ru"],T)*cos(0*(pi/180.0));
    // double v = total_velocity(gamma,inputs["mach"],inputs["ru"],T)*sin(0*(pi/180.0));



    const double gamma = this->gamma;
    const double M     = inputs["mach"];     // freestream Mach (static)
    const double R     = inputs["ru"];       // specific gas constant
    const double p     = inputs["p0"];       // static pressure (Pa)
    const double T     = inputs["t0"];       // static temperature (K)
    const double ang   = 8.0;
    const double ang_r = ang * M_PI / 180.0;

    const double rho = p / (R * T);
    const double a   = std::sqrt(gamma * R * T);
    const double U   = M * a;           // speed magnitude
    const double u   = U * std::cos(ang_r);
    const double v   = U * std::sin(ang_r);
    
    cout << "Freestream Pressure: "<<p<<"   Freestream Density: "<<rho<<" Freestream vel"<<u;

    for (Cell* cell : mesh.interior_cells){
        flux.set_conserved(cell,{rho,u,v,p});
    }

    for (Cell* cell : mesh.ghost_cells){
        flux.set_conserved(cell,{rho,u,v,p});
    }

    

}
















void Solver::set_ambient_conditions(){
    cout << "Setting ambient Conditions...";
    for (Cell* cell : mesh.interior_cells){

        double p = inputs["p0"];
        double rho = total_density(p,inputs["ru"],inputs["t0"],epsilon);
        flux.set_conserved(cell,{rho,0,0,p});
        
    }

    for (Cell* cell : mesh.ghost_cells){

        double p = inputs["p0"];
        double rho = total_density(p,inputs["ru"],inputs["t0"],epsilon);
        flux.set_conserved(cell,{rho,0,0,p});
    }


}
// void Solver::set_inflow_bcs(Cell* inflow_cell,double nx, double ny){

//     //double nx = 1.0;
//     //double ny = 0.0;
//     double p = total_p(gamma,inputs["mach"],inputs["p0"]);
//     double T = total_T(gamma,inputs["mach"],inputs["t0"]);
//     double rho = total_density(p,inputs["ru"],T,epsilon);
//     double pi = 3.1415926;
//     double u = total_velocity(gamma,inputs["mach"],inputs["ru"],T)*cos(0*(pi/180.0));
//     double v = total_velocity(gamma,inputs["mach"],inputs["ru"],T)*sin(0*(pi/180.0));

//     // cout << "SETTING INFLOW";
//     // std::cout << "rho = " << rho << "\n";
//     // std::cout << "u   = " << u   << "\n";
//     // std::cout << "v   = " << v   << "\n";
//     // std::cout << "p   = " << p   << "\n";
    
//     flux.set_conserved(inflow_cell,{rho,u,v,p});

// }



void Solver::set_inflow_bcs(Cell* inflow_cell, double nx, double ny)
{
    // Inputs: static freestream
    
    const double gamma = this->gamma;
    const double M     = inputs["mach"];   // freestream Mach (static)
    const double R     = inputs["ru"];     // gas constant
    const double p     = inputs["p0"];     // static pressure (Pa)
    const double T     = inputs["t0"];     // static temperature (K)

    // Derived freestream
    const double rho = p / (R * T);
    const double a   = std::sqrt(gamma * R * T);
    const double U   = M * a;              // speed magnitude

    const double ang   = 8.0;
    const double ang_r = ang * M_PI / 180.0;

    
    const double u   = U * std::cos(ang_r);
    const double v   = U * std::sin(ang_r);
    
    

    // Set full state (Dirichlet) for supersonic inflow
    flux.set_conserved(inflow_cell, {rho, u, v, p});
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
    //cout << "Setting normal bcs: "<<" p: "<<p <<" rho: "<<rho<<"  u: "<< u <<"  v: "<<v<<"  u_out: "<<u_out<<"  v_out: "<< v_out<<" nx: "<< nx<< " ny: "<<ny<<endl;
    

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


