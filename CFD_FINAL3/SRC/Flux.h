#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <utility> 
#include "Cell.h"
#include "Mesh.h"
#include <array>


using namespace std;
class Flux{
public:


    const double gamma = 1.4;


    Mesh mesh;
    int upwind_order;
    double kappa;
    double epsilon;
    double damping_scheme;

    

    Flux(Mesh mesh_, int upwind_order_, double kappa_, double epsilon_,double damping_scheme_): mesh(mesh_), upwind_order(upwind_order_), kappa(kappa_), epsilon(epsilon_),damping_scheme(damping_scheme_) {}




    array<double,4> roe_flux(array<double,4>U_L,array<double,4> U_R,double nx, double ny);
    pair<array<double,4>, array<double,4>>  MusclExtrapolation(Cell* cell,char direction);
    array<double, 4> vanleer_flux(array<double, 4> UL, array<double, 4> UR, double nx_L, double nx_R);

    void compute_residual();
    array<double, 4> get_primvars(Cell* cell);
    array<double, 4> get_primvars(array<double,4>);
    array<double,4> compute_norm();
    array<double,4> get_primvars(double* cell);
    void set_conserved(Cell*,array<double, 4>);



    































};