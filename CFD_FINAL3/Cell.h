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
    double* V = new double[4]();  // Same
    double* Residual = new double[4]();
    // Solver variables
    double delta_t;

    // Cell pointers
    Cell* cell_L = nullptr;
    Cell* cell_R = nullptr;
    Cell* cell_U = nullptr;
    Cell* cell_D = nullptr;

    //Outward pointing fluxes
// Face normals



    double nx_L = std::numeric_limits<double>::quiet_NaN();
    double nx_R = std::numeric_limits<double>::quiet_NaN();
    double nx_U = std::numeric_limits<double>::quiet_NaN();
    double nx_D = std::numeric_limits<double>::quiet_NaN();

    double ny_L = std::numeric_limits<double>::quiet_NaN();
    double ny_R = std::numeric_limits<double>::quiet_NaN();
    double ny_U = std::numeric_limits<double>::quiet_NaN();
    double ny_D = std::numeric_limits<double>::quiet_NaN();

    double A_L = std::numeric_limits<double>::quiet_NaN();
    double A_R = std::numeric_limits<double>::quiet_NaN();
    double A_U = std::numeric_limits<double>::quiet_NaN();
    double A_D = std::numeric_limits<double>::quiet_NaN();
    double Volume = std::numeric_limits<double>::quiet_NaN();

    double x11 = std::numeric_limits<double>::quiet_NaN();
    double y11 = std::numeric_limits<double>::quiet_NaN();
    double x12 = std::numeric_limits<double>::quiet_NaN();
    double y12 = std::numeric_limits<double>::quiet_NaN();
    double x21 = std::numeric_limits<double>::quiet_NaN();
    double y21 = std::numeric_limits<double>::quiet_NaN();
    double x22 = std::numeric_limits<double>::quiet_NaN();
    double y22 = std::numeric_limits<double>::quiet_NaN();

    double midpoint_x = std::numeric_limits<double>::quiet_NaN();
    double midpoint_y = std::numeric_limits<double>::quiet_NaN();





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
