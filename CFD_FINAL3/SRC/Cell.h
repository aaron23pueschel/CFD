#pragma once
#include <vector>
#include <iostream>
#include <cmath>
 #include <limits>
using namespace std;
class Cell{
public:
//Class variables
    string name;

    int type = -1; // 0 for outflow, 1 for no slip, 2 for inflow, 3 for periodic
    
    // Euler variables
    double* U = new double[4]();  // Initializes all to 0
    double* temp_U = new double[4]();
    double* Residual = new double[4]();
    double* Source = new double[4]();
    double* K1 = new double[4]();
    double* K2 = new double[4]();
    double* K3 = new double[4]();
    double* K4 = new double[4]();
    double* total_residual = new double[4]();
    double* p1 = new double[1]();

    // Solver variables
    double delta_t;

    int i_idx;
    int j_idx;

    // Cell pointers
    Cell* cell_L = nullptr;
    Cell* cell_R = nullptr;
    Cell* cell_U = nullptr;
    Cell* cell_D = nullptr;

    //Outward pointing fluxes
// Face normals



    double nx_L = numeric_limits<double>::quiet_NaN();
    double nx_R = numeric_limits<double>::quiet_NaN();
    double nx_U = numeric_limits<double>::quiet_NaN();
    double nx_D = numeric_limits<double>::quiet_NaN();

    double ny_L = numeric_limits<double>::quiet_NaN();
    double ny_R = numeric_limits<double>::quiet_NaN();
    double ny_U = numeric_limits<double>::quiet_NaN();
    double ny_D = numeric_limits<double>::quiet_NaN();

    double A_L = numeric_limits<double>::quiet_NaN();
    double A_R = numeric_limits<double>::quiet_NaN();
    double A_U = numeric_limits<double>::quiet_NaN();
    double A_D = numeric_limits<double>::quiet_NaN();
    double Volume = numeric_limits<double>::quiet_NaN();

    double x11 = numeric_limits<double>::quiet_NaN();
    double y11 = numeric_limits<double>::quiet_NaN();
    double x12 = numeric_limits<double>::quiet_NaN();
    double y12 = numeric_limits<double>::quiet_NaN();
    double x21 = numeric_limits<double>::quiet_NaN();
    double y21 = numeric_limits<double>::quiet_NaN();
    double x22 = numeric_limits<double>::quiet_NaN();
    double y22 = numeric_limits<double>::quiet_NaN();

    double midpoint_x = numeric_limits<double>::quiet_NaN();
    double midpoint_y = numeric_limits<double>::quiet_NaN();





// Class functions
    void initialize_areas();
    void initialize_normals();
    void set_points(double X11, double Y11,
                    double X12, double Y12,
                    double X21, double Y21,
                    double X22, double Y22);

// Constructor
    Cell(string name_): name(name_){}





























};
