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

    

    Flux(Mesh mesh_, int upwind_order_, double kappa_, double epsilon_): mesh(mesh_), upwind_order(upwind_order_), kappa(kappa_), epsilon(epsilon_) {}




    void roe_flux();
    array<double, 4> vanleer_flux(double* UL, double* UR, double nx_L, double nx_R);

    void compute_residual();
    array<double, 4> get_primvars(Cell* cell);
    array<double,4> get_primvars(double* cell);
    void set_conserved(Cell*,array<double, 4>);



    































};