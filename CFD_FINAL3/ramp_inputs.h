#pragma once
#include <string>
using namespace std;
struct Inputs {

int NI = 17;
int NJ = 53;

string MeshName = "Ramp";
string xx_filename = "Ramp_Coarse_cells_xx.csv";
string yy_filename = "Ramp_Coarse_cells_yy.csv";
double mach = 2.1;
double p0 = 65.8558;
double t0 = 300.0;
double ru = 8314.0;
double cfl = 0.0001;
double gamma = 1.4;

int extrapolation_order = 0;
double epsilon = 1e-10;
double kappa = -1;
int upwind_order = 0;
double convergence_criteria = 1e-12;

int damping_scheme = 1;
int iter_max = 1000000;

bool local_timestep = true;
int flux_limiter_scheme = 1;




};