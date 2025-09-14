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
    double upwind_order;
    double kappa;
    double epsilon;
    double damping_scheme;
















    const double pi     = 3.14159265358979323846;

    const double rho0   = 1.0;
    const double rhox   = 0.15;
    const double rhoy   = -0.1;

    const double uvel0  = 800.0;
    const double uvelx  = 50.0;
    const double uvely  = -30.0;

    const double vvel0  = 800.0;
    const double vvelx  = -75.0;
    const double vvely  = 40.0;

    const double press0 = 100000.0;
    const double pressx = 20000.0;
    const double pressy = 50000.0;










    

    Flux(Mesh mesh_, double upwind_order_, double kappa_, double epsilon_,double damping_scheme_): mesh(mesh_), upwind_order(upwind_order_), kappa(kappa_), epsilon(epsilon_),damping_scheme(damping_scheme_) {}




    array<double,4> roe_flux(array<double,4>U_L,array<double,4> U_R,double nx, double ny);
    pair<array<double,4>, array<double,4>>  MusclExtrapolation(Cell* cell,char direction);
    array<double, 4> vanleer_flux(array<double, 4> UL, array<double, 4> UR, double nx_L, double nx_R);

    void compute_residual();
    array<double, 4> get_primvars(Cell* cell);
    array<double, 4> get_primvars(array<double,4>);
    array<double,4> compute_norm();
    array<double,4> get_primvars(double* cell);
    void set_conserved(Cell*,array<double, 4>);
    void set_source(Cell*,array<double, 4>);
    void set_MMS_source(bool is_mms);
    double rho_mms   (double length, double x, double y);
    double uvel_mms  (double length, double x, double y);
    double vvel_mms  (double length, double x, double y);
    double press_mms (double length, double x, double y);

    double mass_mms(double length, double x, double y);
    double xmtm_mms(double length, double x, double y);
    double ymtm_mms(double length, double x, double y);
    double energy_mms(double length, double x, double y);




    































};