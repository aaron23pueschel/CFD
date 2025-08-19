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
    const double pi = 3.141526;
    int upwind_order = 1;

    Mesh mesh;

    Flux(Mesh mesh_) : mesh(mesh_){}




    void roe_flux();
    array<double, 4> vanleer_flux(double* UL, double* UR, double nx_L, double nx_R);

    void compute_residual();




    































};