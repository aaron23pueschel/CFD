#pragma once
#include <string>
using namespace std;
struct Inputs {

int NI = 17;
int NJ = 53;

string MeshName = "Coarse";
string xx_filename = "Ramp_Fine_xx.csv";
string yy_filename = "Ramp_Fine_yy.csv";
double mach = 2.0;
double p0 = 65855.8;
double t0 = 300.0;
double ru = 8314.0;
double m  = 28.97;
double cfl = 0.1;
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