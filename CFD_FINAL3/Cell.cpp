#include "Cell.h"
#include <cmath>
using namespace std;
void Cell::set_points(double X11, double Y11,
                double X12, double Y12,
                double X21, double Y21,
                double X22, double Y22){

        x11 = X11; y11 = Y11;
        x12 = X12; y12 = Y12;
        x21 = X21; y21 = Y21;
        x22 = X22; y22 = Y22;

        // compute midpoint automatically
        midpoint_x = 0.25 * (x11 + x12 + x21 + x22);
        midpoint_y = 0.25 * (y11 + y12 + y21 + y22);
}


void Cell::initialize_areas(){


    A_L = sqrt(pow(x11 - x21, 2) + pow(y11 - y21, 2));
    A_R = sqrt(pow(x12 - x22, 2) + pow(y12 - y22, 2));
    A_D = sqrt(pow(x11 - x12, 2) + pow(y11 - y12, 2));
    A_U = sqrt(pow(x21 - x22, 2) + pow(y21 - y22, 2));

    double tri1 = 0.5 * fabs(x11*(y12 - y21) + x12*(y21 - y11) + x21*(y11 - y12));
    double tri2 = 0.5 * fabs(x21*(y12 - y22) + x12*(y22 - y21) + x22*(y21 - y12));
    Volume = tri1 + tri2;


}




void Cell::initialize_normals(){

    // Left face: (x11,y11) -> (x21,y21)
    {
        double ex = x21 - x11;
        double ey = y21 - y11;
        nx_L =  -ey / A_L;
        ny_L = ex / A_L;

       
    }

    // Right face: (x12,y12) -> (x22,y22)
    {
        double ex = x22 - x12;
        double ey = y22 - y12;
        nx_R =  ey / A_R;
        ny_R = -ex / A_R;

    }

    // Bottom face: (x11,y11) -> (x12,y12)
    {
        double ex = x12 - x11;
        double ey = y12 - y11;
        nx_D =  -ey / A_D;
        ny_D = ex / A_D;

       
    }

    // Top face: (x21,y21) -> (x22,y22)
    {
        double ex = x22 - x21;
        double ey = y22 - y21;
        nx_U =  ey / A_U;
        ny_U = -ex / A_U;

       
    }




}