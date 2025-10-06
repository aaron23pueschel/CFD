
#include <vector>
#include <iostream>
#include <cmath>
#include <utility> 
#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include "Solver.h"
#include "Simulation.h"
#include "inputs.h"
#include <map>
#include <array>

#include <iomanip>  // Required for std::setprecision and std::scientific
using namespace std;


void Simulation::write_primitives_csv(const string& filename)
{
    ofstream file(filename);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }

    const auto& cells = simulation_mesh.interior_cells;
    const size_t N = cells.size();

    // Collect each primitive across all cells
    vector<double> rho; rho.reserve(N);
    vector<double> u;   u.reserve(N);
    vector<double> v;   v.reserve(N);
    vector<double> p;   p.reserve(N);
    vector<double> p1;   p1.reserve(N);

    for (const auto* elem : cells) {
        array<double,4> V = simulation_flux.get_primvars(elem->U); // {rho,u,v,p}
        rho.push_back(V[0]);
        u.push_back(V[1]);
        v.push_back(V[2]);
        p.push_back(V[3]);
        p1.push_back(elem->p1[0]);
    }

    auto write_line = [&](const std::vector<double>& vec) {
        for (size_t i = 0; i < vec.size(); ++i) {
            file << std::scientific << std::setprecision(17) << vec[i];
            if (i + 1 < vec.size()) file << ",";
        }
        file << "\n";
    };

    // One variable per line (newline separates variables)
    write_line(rho);
    write_line(u);
    write_line(v);
    write_line(p);
    write_line(p1);
}



void Simulation::write_Cd(const string& filename){
    ofstream file(filename);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    double Cd = 0.0;
    for(Cell* cell:simulation_mesh.ghost_cells){

        if(cell->type==1 && cell->cell_D==nullptr){

        
            auto V0 = simulation_flux.get_primvars(cell->cell_U);
            auto V1 = simulation_flux.get_primvars(cell->cell_U->cell_U);
            double p0 = V0[3]; double p1 = V1[3];


            double pressure_at_face = p0 + .5*(p1-p0); 


            double nx = cell->cell_U->nx_D;

            double Area = sqrt(pow(cell->cell_U->x22-cell->cell_U->x21,2) + pow(cell->cell_U->y22-cell->cell_U->y21,2));

            file << std::fixed << std::setprecision(17) << pressure_at_face * nx * Area;



            


        

        }



    }


    //file << Cd;





}








void Simulation::write_sources_csv(const string& filename)
{
    ofstream file(filename);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }

    const auto& cells = simulation_mesh.interior_cells;
    const size_t N = cells.size();

    // Collect each primitive across all cells
    vector<double> rho; rho.reserve(N);
    vector<double> u;   u.reserve(N);
    vector<double> v;   v.reserve(N);
    vector<double> p;   p.reserve(N);

    for (const auto* elem : cells) {
        //cout << simulation_flux.vvel_mms(1,elem->midpoint_x,elem->midpoint_y)<<","<<elem->midpoint_x<<","<<elem->midpoint_y<<endl;
        rho.push_back(simulation_flux.rho_mms(1,elem->midpoint_x,elem->midpoint_y));
        u.push_back(simulation_flux.uvel_mms(1,elem->midpoint_x,elem->midpoint_y));
        v.push_back(simulation_flux.vvel_mms(1,elem->midpoint_x,elem->midpoint_y));
        p.push_back(simulation_flux.press_mms(1,elem->midpoint_x,elem->midpoint_y));

        // rho.push_back(elem->Source[0]);
        // u.push_back(elem->Source[1]);
        // v.push_back(elem->Source[2]);
        // p.push_back(elem->Source[3]);
    }

    auto write_line = [&](const std::vector<double>& vec) {
        for (size_t i = 0; i < vec.size(); ++i) {
            file << std::setprecision(17) << std::scientific << vec[i];
            if (i + 1 < vec.size()) file << ",";
        }
        file << "\n";
    };

    // One variable per line (newline separates variables)
    write_line(rho);
    write_line(u);
    write_line(v);
    write_line(p);
}












void Simulation::write_residuals_csv(const string& filename)
{
    ofstream file(filename);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }

    const auto& cells = simulation_mesh.interior_cells;
    const size_t N = cells.size();

    // Collect each primitive across all cells
    vector<double> rho; rho.reserve(N);
    vector<double> u;   u.reserve(N);
    vector<double> v;   v.reserve(N);
    vector<double> p;   p.reserve(N);

    for (const auto* elem : cells) {
        auto V = elem->total_residual;
        rho.push_back(V[0]);
        u.push_back(V[1]);
        v.push_back(V[2]);
        p.push_back(V[3]);
    }


auto write_line = [&](const std::vector<double>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        file << std::scientific << std::setprecision(17) << vec[i];
        if (i + 1 < vec.size()) file << ",";
    }
    file << "\n";
};


    // One variable per line (newline separates variables)
    write_line(rho);
    write_line(u);
    write_line(v);
    write_line(p);
}


Inputs Simulation::load_inputs(const string& filename) {
    Inputs inputs;
    ifstream f(filename);
    nlohmann::json j;
    f >> j;

    j.at("NI").get_to(inputs.NI);
    j.at("NJ").get_to(inputs.NJ);
    j.at("MeshName").get_to(inputs.MeshName);
    j.at("xx_filename").get_to(inputs.xx_filename);
    j.at("yy_filename").get_to(inputs.yy_filename);
    j.at("mach").get_to(inputs.mach);
    j.at("p0").get_to(inputs.p0);
    j.at("t0").get_to(inputs.t0);
    j.at("ru").get_to(inputs.ru);
    j.at("cfl").get_to(inputs.cfl);
    j.at("gamma").get_to(inputs.gamma);
    j.at("extrapolation_order").get_to(inputs.extrapolation_order);
    j.at("epsilon").get_to(inputs.epsilon);
    j.at("kappa").get_to(inputs.kappa);
    j.at("upwind_order").get_to(inputs.upwind_order);
    j.at("convergence_criteria").get_to(inputs.convergence_criteria);
    j.at("damping_scheme").get_to(inputs.damping_scheme);
    j.at("iter_max").get_to(inputs.iter_max);
    j.at("local_timestep").get_to(inputs.local_timestep);
    j.at("flux_limiter_scheme").get_to(inputs.flux_limiter_scheme);
    j.at("is_mms").get_to(inputs.is_mms);

    return inputs;
}







