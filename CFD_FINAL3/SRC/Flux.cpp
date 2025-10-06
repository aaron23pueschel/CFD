#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include <utility> 
#include <cmath>
#include <tuple>
#include <array>
#include <functional>
#include <stdexcept>
using namespace std;








double Flux::rho_mms(double length, double x, double y) {
    return rho0
         + rhoy * cos((pi * y) / (2.0 * length))
         + rhox * sin((pi * x) / length);
}

double Flux::uvel_mms(double length, double x, double y) {
    return uvel0
         + uvely * cos((3.0 * pi * y) / (5.0 * length))
         + uvelx * sin((3.0 * pi * x) / (2.0 * length));
}

double Flux::vvel_mms(double length, double x, double y) {
    return vvel0
         + vvelx * cos((pi * x) / (2.0 * length))
         + vvely * sin((2.0 * pi * y) / (3.0 * length));
}

double Flux::press_mms(double length, double x, double y) {
    return press0
         + pressx * cos((2.0 * pi * x) / length)
         + pressy * sin((pi * y) / length);
}

double Flux::mass_mms(double length, double x, double y){

   

   double mass = (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                 
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length))) /        
    (two*length) + (two*pi*vvely*cos((two*pi*y)/(three*length)) *              
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length))) /        
    (three*length) + (pi*rhox*cos((pi*x)/length) *                             
    (uvel0 + uvely*cos((three*pi*y)/(five*length)) + uvelx*sin((three*pi*x)/   
    (two*length))))/length - (pi*rhoy*sin((pi*y)/(two*length)) *               
    (vvel0 + vvelx*cos((pi*x)/(two*length)) + vvely*sin((two*pi*y) /           
    (three*length))))/(two*length);


    return mass;

}



double Flux::xmtm_mms(double length,double x,double y){




 double xmtmconv = (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                  
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length)) *         
    (uvel0 + uvely*cos((three*pi*y)/(five*length)) +                           
    uvelx*sin((three*pi*x)/(two*length))))/length +                            
    (two*pi*vvely*cos((two*pi*y) /                                             
    (three*length))*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    
    rhox*sin((pi*x)/length))*(uvel0 + uvely*cos((three*pi*y) /                 
    (five*length)) + uvelx*sin((three*pi*x)/(two*length))))/(three*length) +   
    (pi*rhox*cos((pi*x)/length)*pow(uvel0 + uvely*cos((three*pi*y) /              
    (five*length)) + uvelx*sin((three*pi*x)/(two*length)),2))/length -        
    (two*pi*pressx*sin((two*pi*x)/length))/length -                            
    (pi*rhoy*(uvel0 + uvely*cos((three*pi*y)/(five*length)) +                  
    uvelx*sin((three*pi*x)/(two*length)))*sin((pi*y)/(two*length))*            
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  
    vvely*sin((two*pi*y)/(three*length))))/(two*length) -                      
    (three*pi*uvely*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    
    rhox*sin((pi*x)/length))*sin((three*pi*y)/(five*length))*(vvel0 + vvelx *  
    cos((pi*x)/(two*length)) + vvely*sin((two*pi*y)/(three*length)))) /        
    (five*length);

    return xmtmconv;



}



double Flux::ymtm_mms(double length,double x,double y){


double ymtmconv = (pi*pressy*cos((pi*y)/length))/length -                           
    (pi*vvelx*sin((pi*x)/(two*length))*(rho0 + rhoy*cos((pi*y)/(two*length)) + 
    rhox*sin((pi*x)/length))*(uvel0 + uvely*cos((three*pi*y)/(five*length)) +  
    uvelx*sin((three*pi*x)/(two*length))))/(two*length) +                      
    (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                           
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length)) *         
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  
    vvely*sin((two*pi*y)/(three*length))))/(two*length) +                      
    (four*pi*vvely*cos((two*pi*y) /                                            
    (three*length))*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    
    rhox*sin((pi*x)/length))*(vvel0 + vvelx*cos((pi*x)/(two*length)) +         
    vvely*sin((two*pi*y)/(three*length))))/(three*length) +                    
    (pi*rhox*cos((pi*x)/length) *                                              
    (uvel0 + uvely*cos((three*pi*y)/(five*length)) +                           
    uvelx*sin((three*pi*x)/(two*length))) *                                    
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  
    vvely*sin((two*pi*y)/(three*length))))/length -                            
    (pi*rhoy*sin((pi*y)/(two*length)) *                                        
    pow(vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  
    vvely*sin((two*pi*y)/(three*length)),2))/(two*length);

    return ymtmconv;


}





double Flux::energy_mms(double length,double x,double y){


   double energyconv = (uvel0 + uvely*cos((three*pi*y)/(five*length)) +                
    uvelx*sin((three*pi*x)/(two*length)))*((-two*pi*pressx*sin((two*pi*x) /    
    length))/length + (rho0 + rhoy*cos((pi*y)/(two*length)) +                  
    rhox*sin((pi*x)/length))*((-two*pi*pressx*sin((two*pi*x)/length))/         
    ((-one + gamma)*length*(rho0 + rhoy*cos((pi*y)/(two*length)) +             
    rhox*sin((pi*x)/length))) + ((three*pi*uvelx*cos((three*pi*x) /            
    (two*length))*(uvel0 + uvely*cos((three*pi*y)/(five*length)) +             
    uvelx*sin((three*pi*x)/(two*length))))/length - (pi*vvelx*sin((pi*x) /     
    (two*length))*(vvel0 + vvelx*cos((pi*x)/(two*length)) +                    
    vvely*sin((two*pi*y)/(three*length))))/length)/two - (pi*rhox*cos((pi*x) / 
    length)*(press0 + pressx*cos((two*pi*x)/length) +                           
    pressy*sin((pi*y)/length)))/((-one + gamma)*length*pow(rho0 + rhoy*cos((pi*y)/
    (two*length)) + rhox*sin((pi*x)/length),2))) +                            
    (pi*rhox*cos((pi*x)/length)*((0.0 + pow(uvel0 + uvely*cos((three*pi*y) / 
    (five*length)) + uvelx*sin((three*pi*x)/(two*length)),2) +                
    pow(vvel0 + vvelx*cos((pi*x)/(two*length)) + vvely*sin((two*pi*y) /           
    (three*length)),2))/two + (press0 + pressx*cos((two*pi*x)/length) +       
    pressy*sin((pi*y)/length))/((-one + gamma) *                               
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    
    rhox*sin((pi*x)/length)))))/length) +                                      
    (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                           
    (press0 + pressx*cos((two*pi*x)/length) + pressy*sin((pi*y)/length) +      
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length))*          
    ((0.0 + pow(uvel0 + uvely*cos((three*pi*y)/(five*length)) +              
    uvelx*sin((three*pi*x)/(two*length)),2) +                                 
    pow(vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  
    vvely*sin((two*pi*y)/(three*length)),2))/two +                            
    (press0 + pressx*cos((two*pi*x)/length) +                                  
    pressy*sin((pi*y)/length))/((-one + gamma) *                               
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    
    rhox*sin((pi*x)/length))))))/(two*length) +                                
    (two*pi*vvely*cos((two*pi*y)/(three*length)) *                             
    (press0 + pressx*cos((two*pi*x)/length) +                                  
    pressy*sin((pi*y)/length) + (rho0 + rhoy*cos((pi*y)/(two*length)) +        
    rhox*sin((pi*x)/length))*((0.0 +                                      
    pow(uvel0 + uvely*cos((three*pi*y)/(five*length)) +                           
    uvelx*sin((three*pi*x)/(two*length)),2) +                                 
    pow(vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  
    vvely*sin((two*pi*y)/(three*length)),2))/two +                            
    (press0 + pressx*cos((two*pi*x)/length) + pressy*sin((pi*y)/length)) /     
	((-one + gamma)*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    
    rhox*sin((pi*x)/length))))))/(three*length) + (vvel0 + vvelx*cos((pi*x) /  
    (two*length)) + vvely*sin((two*pi*y)/(three*length))) *                    
    ((pi*pressy*cos((pi*y)/length))/length - (pi*rhoy*sin((pi*y)/(two*length))*
    ((0.0 + pow(uvel0 + uvely*cos((three*pi*y)/(five*length)) +              
    uvelx*sin((three*pi*x)/(two*length)),2) + pow(vvel0 + vvelx *                
    cos((pi*x)/(two*length)) + vvely*sin((two*pi*y)/(three*length)),2))/two + 
    (press0 + pressx*cos((two*pi*x)/length) +                                  
    pressy*sin((pi*y)/length))/((-one + gamma) *                               
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    
    rhox*sin((pi*x)/length)))))/(two*length) +                                 
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    
    rhox*sin((pi*x)/length))*((pi*pressy*cos((pi*y)/length)) /                 
    ((-one + gamma)*length*(rho0 + rhoy*cos((pi*y)/(two*length)) +             
    rhox*sin((pi*x)/length))) +                                                
    ((-six*pi*uvely*(uvel0 + uvely*cos((three*pi*y) /                          
    (five*length)) + uvelx*sin((three*pi*x)/(two*length))) *                   
    sin((three*pi*y)/(five*length)))/(five*length) +                           
    (four*pi*vvely*cos((two*pi*y) /                                            
    (three*length))*(vvel0 + vvelx*cos((pi*x)/(two*length)) +                  
    vvely*sin((two*pi*y)/(three*length))))/(three*length))/two +               
    (pi*rhoy*sin((pi*y)/(two*length))*(press0 + pressx*cos((two*pi*x)/length) +
    pressy*sin((pi*y)/length)))/(two*(-one + gamma)*length*                    
    pow(rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length),2))));




    return energyconv;













}












void Flux::set_MMS_source(bool is_mms){

    for (Cell* cell : mesh.interior_cells) {
        double length = 1.0;
        // double rho = rho_mms(length, cell->midpoint_x, cell->midpoint_y);
        // double u   = uvel_mms(length, cell->midpoint_x, cell->midpoint_y);
        // double v   = vvel_mms(length, cell->midpoint_x, cell->midpoint_y);
        // double p   = press_mms(length, cell->midpoint_x, cell->midpoint_y);


        double mass = mass_mms(length, cell->midpoint_x, cell->midpoint_y);
        double xmtm   = xmtm_mms(length, cell->midpoint_x, cell->midpoint_y);
        double ymtm   = ymtm_mms(length, cell->midpoint_x, cell->midpoint_y);
        double E   = energy_mms(length, cell->midpoint_x, cell->midpoint_y);



        if(is_mms){
            set_source(cell,{mass,xmtm,ymtm,E});
        }
        else{
            for(int i=0;i<4;i++)
            cell->Source[i] = 0.0; 
        }


    }




}



array<double,4> Flux::roe_flux(array<double,4>U_L,array<double,4> U_R,double nx, double ny)
{

    const double half = 0.5;
    const double two  = 2.0;
    const double four = 4.0;

    const double eps = 0.1;      // entropy fix parameter (Harten)
    const double tol = 1.0e-6;   // clipping threshold

    array<double,4> F{};

    auto VL = get_primvars(U_L);
    auto VR = get_primvars(U_R);
    double rho_L = VL[0];
    double u_L   = VL[1];
    double v_L   = VL[2];
    double p_L   = VL[3];

    double rho_R = VR[0];
    double u_R   = VR[1];
    double v_R   = VR[2];
    double p_R   = VR[3];

    double keL  = half*(u_L*u_L + v_L*v_L);
    double htL  = (gamma/(gamma-1.0))*(p_L/rho_L) + keL;
    double UhatL = u_L*nx + v_L*ny;



    double keR  = half*(u_R*u_R + v_R*v_R);
    double htR  = (gamma/(gamma-1.0))*(p_R/rho_R) + keR;
    double UhatR = u_R*nx + v_R*ny;

    // ---- Roe averages ----
    double Ri     = std::sqrt(rho_R / rho_L);
    double rhobar = Ri * rho_L;
    double ubar   = (Ri*u_R + u_L) / (Ri + 1.0);
    double vbar   = (Ri*v_R + v_L) / (Ri + 1.0);
    double htbar  = (Ri*htR + htL) / (Ri + 1.0);
    double keBar  = half*(ubar*ubar + vbar*vbar);
    double abar   = std::sqrt( std::max( (gamma-1.0)*(htbar - keBar), tol ) );
    double Uhat   = ubar*nx + vbar*ny;

    // ---- Eigenvalues (projected) ----
    double lam[4];
    lam[0] = Uhat;
    lam[1] = Uhat;
    lam[2] = Uhat + abar;
    lam[3] = Uhat - abar;

    // ---- Entropy fix (Harten) ----
    double lam_mod[4];
    for (int i = 0; i < 4; ++i) {
        if (std::abs(lam[i]) >= two*eps*abar) {
            lam_mod[i] = std::abs(lam[i]);
        } else {
            lam_mod[i] = (lam[i]*lam[i])/(four*eps*abar) + eps*abar;
        }
    }

    // ---- Right eigenvectors (r1..r4) ----
    // r1: convective
    double r1[4];
    r1[0] = 1.0;
    r1[1] = ubar;
    r1[2] = vbar;
    r1[3] = keBar;

    // r2: shear (tangential)
    double r2[4];
    r2[0] = 0.0;
    r2[1] =  ny*rhobar;
    r2[2] = -nx*rhobar;
    r2[3] =  rhobar*(ny*ubar - nx*vbar);

    // r3: acoustic (+a)
    double fac = half*(rhobar/abar);
    double r3[4];
    r3[0] =  fac;
    r3[1] =  fac*(ubar + nx*abar);
    r3[2] =  fac*(vbar + ny*abar);
    r3[3] =  fac*(htbar + Uhat*abar);

    // r4: acoustic (-a)
    double r4[4];
    r4[0] = -fac;
    r4[1] = -fac*(ubar - nx*abar);
    r4[2] = -fac*(vbar - ny*abar);
    r4[3] = -fac*(htbar - Uhat*abar);

    // ---- Jumps and wave strengths (dw) ----
    double drho = rho_R - rho_L;
    double dP   = p_R   - p_L;
    double du   = u_R   - u_L;
    double dv   = v_R   - v_L;

    double dw[4];
    dw[0] = drho + dP/(abar*abar);
    dw[1] = ny*du - nx*dv;
    dw[2] = (nx*du + ny*dv) + dP/(rhobar*abar);
    dw[3] = (nx*du + ny*dv) - dP/(rhobar*abar);

    // ---- Physical fluxes on each side ----
    array<double,4> fL{
        rho_L*UhatL,
        rho_L*u_L*UhatL + nx*p_L,
        rho_L*v_L*UhatL + ny*p_L,
        rho_L*htL*UhatL
    };

    array<double,4> fR{
        rho_R*UhatR,
        rho_R*u_R*UhatR + nx*p_R,
        rho_R*v_R*UhatR + ny*p_R,
        rho_R*htR*UhatR
    };

    // ---- Roe flux: average minus dissipation ----
    for (int i = 0; i < 4; ++i) {
        double diss =
            lam_mod[0]*dw[0]*r1[i] +
            lam_mod[1]*dw[1]*r2[i] +
            lam_mod[2]*dw[2]*r3[i] +
            lam_mod[3]*dw[3]*r4[i];

        F[i] = 0.5*(fL[i] + fR[i]) - 0.5*diss;
    }

    return F;
}

array<double, 4> Flux::vanleer_flux(array<double, 4> U_L, array<double, 4> U_R, double nx, double ny) {
    auto VL = get_primvars(U_L);
    auto VR = get_primvars(U_R);
    double rho_L = VL[0];
    double u_L   = VL[1];
    double v_L   = VL[2];
    double p_L   = VL[3];

    double rho_R = VR[0];
    double u_R   = VR[1];
    double v_R   = VR[2];
    double p_R   = VR[3];



    double Uhat_L = u_L * nx + v_L * ny;
    double aL = sqrt(gamma * p_L/ rho_L);
    double ML = Uhat_L / aL;
    double htL = (gamma / (gamma - 1)) * (p_L / rho_L) + .5 * (u_L * u_L + v_L * v_L);


    double Uhat_R = u_R * nx + v_R * ny;
    double aR = sqrt(gamma * p_R/ rho_R);
    double MR = Uhat_R / aR;
    double htR = (gamma / (gamma - 1)) * (p_R / rho_R) + .5 * (u_R * u_R + v_R * v_R);




    // Flux sp_Litting
    double alpha_p = .5 * (1.0 + copysign(1.0, ML));
    double alpha_m = .5 * (1.0 - copysign(1.0, MR));

    double betaL = -std::max(0.0, static_cast<double>(1 - static_cast<int>(std::abs(ML))));
    double betaR = -std::max(0.0, static_cast<double>(1 - static_cast<int>(std::abs(MR))));


    double Mp = .25 * pow((ML + 1.0), 2);
    double Mm = -.25 * pow((MR - 1.0), 2);

    double Cp = alpha_p * (1.0 + betaL) * ML - betaL * Mp;
    double Cm = alpha_m * (1.0 + betaR) * MR - betaR * Mm;

    array<double, 4> F_convective;
    F_convective[0] = rho_L * aL * Cp * 1.0 + rho_R * aR * Cm * 1.0;
    F_convective[1] = rho_L * aL * Cp * u_L + rho_R * aR * Cm * u_R;
    F_convective[2] = rho_L * aL * Cp * v_L + rho_R * aR * Cm * v_R;
    F_convective[3] = rho_L * aL * Cp * htL + rho_R * aR * Cm * htR;

    double Pp = Mp * (-ML + 2.0);
    double Pm = Mm * (-MR - 2.0);

    double Dp = (alpha_p * (1.0 + betaL)) - (betaL * Pp);
    double Dm = (alpha_m * (1.0 + betaR)) - (betaR * Pm);

    array<double, 4> F_pressure;
    F_pressure[0] = 0.0;
    F_pressure[1] = Dp * (nx * p_L) + Dm * (nx * p_R);
    F_pressure[2] = Dp * (ny * p_L) + Dm * (ny * p_R);
    F_pressure[3] = 0.0;

    array<double, 4> F;
    for (int i = 0; i < 4; ++i) 
        F[i] = F_convective[i]+ F_pressure[i];
    
    return F;

}



double p_func(double i4,double i3,double i2,double i1){
    return 1.0;
    if(isnan(i4)|| isnan(i3)||isnan(i2)||isnan(i1))
         throw invalid_argument("NaN encountered in Flux::p_func");
    double NUM = i4-i3;
    double DEN = i2-i1;


    if (abs(DEN) < 1e-6)
        return 0.0;

    double r = NUM / DEN;
    //return std::max(0.0, std::min(1.0, r));
    //return (r + std::abs(r)) / (1.0 + std::abs(r));
    //return std::max(0.0, std::max(std::min(2*r,1.0), std::min(r,2.0)));
    return (r + std::abs(r)) / (1.0 + std::abs(r));
    //return (r*r+r)/max(.00000001,1+r*r);
    //return 1.0;
        

}


inline double van_albada(double a, double b) {
    // Symmetric van Albada (smooth, TVD). eps avoids 0/0 near extrema.
    const double eps = 1e-16;
    if (a * b <= 0.0) return 0.0;
    return ((a * b + eps) / (a * a + b * b + eps)) * (a + b);
}

inline double van_leer(double a, double b) {
    // Equivalent to φ(r) = (r + |r|)/(1 + |r|) with r = a/b.
    if (a * b <= 0.0) return 0.0;
    return (2.0 * a * b) / (a + b);  // harmonic mean
}
inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}
inline double mc_limiter(double a, double b) {
    // MC: minmod( (a+b)/2, 2a, 2b )
    //return minmod(0.5*(a + b), minmod(2.0*a, 2.0*b));
    //return van_leer(a,b);
    return van_albada(a,b);
}

inline double limiter_frozen(double a, double b) {
    // a = (U_i - U_{i-1}), b = (U_{i+1} - U_i) on a uniform grid
    return 0.5 * (a + b);  // central slope
}



pair<array<double,4>, array<double,4>> Flux::MusclExtrapolation(Cell* cell,char direction){
        array<double,4> FL;
        array<double,4> FR;
        int i=0;
        double p1; double p3;
        
        double upwind_order_ = upwind_order;
        
        for(;i<4;i++){

            // Put these helpers somewhere accessible (e.g., top of file or a utils header)

// ... inside your loop over components i and given `cell` & `direction`:

// ---------- RIGHT face (cell | CR) ----------
if (direction == 'R') {
    auto CL  = cell->cell_L;
    auto CR  = cell->cell_R;
    auto CRR = (CR && CR->cell_R) ? CR->cell_R : CR;

    double Ui   = cell->U[i];
    double UCL  = CL->U[i];
    double UCR  = CR->U[i];
    double UCRR = CRR->U[i];

    // slopes toward the RIGHT face
    double slopeC  = (upwind_order_ == 1) ? 0.0 : mc_limiter(Ui  - UCL,  UCR - Ui );
    double slopeCR = (upwind_order_ == 1) ? 0.0 : mc_limiter(UCR - Ui,   UCRR - UCR);

    // states at i+1/2
    FL[i] = Ui  + 0.5 * slopeC;     // from cell (left of face)
    FR[i] = UCR - 0.5 * slopeCR;    // from CR   (right of face)
}

// ---------- LEFT face (CL | cell) ----------
if (direction == 'L') {
    auto CL  = cell->cell_L;
    auto CLL = (CL && CL->cell_L) ? CL->cell_L : CL;
    auto CR  = cell->cell_R;

    double Ui   = cell->U[i];
    double UCL  = CL->U[i];
    double UCLL = CLL->U[i];
    double UCR  = CR->U[i];

    // slopes toward the LEFT face
    double slopeCL = (upwind_order_ == 1) ? 0.0 : mc_limiter(UCL - UCLL, Ui  - UCL);
    double slopeC  = (upwind_order_ == 1) ? 0.0 : mc_limiter(Ui  - UCL,  UCR - Ui );

    // states at i-1/2
    FL[i] = UCL + 0.5 * slopeCL;    // from CL   (left of face)
    FR[i] = Ui  - 0.5 * slopeC;     // from cell (right of face)
}

// ---------- UP face (cell | CU) ----------
if (direction == 'U') {
    auto CD  = cell->cell_D;
    auto CU  = cell->cell_U;
    auto CUU = (CU && CU->cell_U) ? CU->cell_U : CU;

    double Ui   = cell->U[i];
    double UCD  = CD->U[i];
    double UCU  = CU->U[i];
    double UCUU = CUU->U[i];

    // slopes toward the UP face
    double slopeC  = (upwind_order_ == 1) ? 0.0 : mc_limiter(Ui  - UCD,  UCU - Ui );
    double slopeCU = (upwind_order_ == 1) ? 0.0 : mc_limiter(UCU - Ui,   UCUU - UCU);

    // states at j+1/2
    FL[i] = Ui  + 0.5 * slopeC;     // from cell (below face)
    FR[i] = UCU - 0.5 * slopeCU;    // from CU   (above face)
}

// ---------- DOWN face (CD | cell) ----------
if (direction == 'D') {
    auto CD  = cell->cell_D;
    auto CDD = (CD && CD->cell_D) ? CD->cell_D : CD;
    auto CU  = cell->cell_U;

    double Ui   = cell->U[i];
    double UCD  = CD->U[i];
    double UCDD = CDD->U[i];
    double UCU  = CU->U[i];

    // slopes toward the DOWN face
    double slopeCD = (upwind_order_ == 1) ? 0.0 : mc_limiter(UCD - UCDD, Ui  - UCD);
    double slopeC  = (upwind_order_ == 1) ? 0.0 : mc_limiter(Ui  - UCD,  UCU - Ui );

    // states at j-1/2
    FL[i] = UCD + 0.5 * slopeCD;    // from CD   (below face)
    FR[i] = Ui  - 0.5 * slopeC;     // from cell (above face)
}

        }

    return make_pair(FL,FR);





        

}






array<double,4> Flux::compute_norm(){
    double sumP = 0.0; double sumU = 0.0;double sumV=0.0;double sumRho = 0.0;
    int N = mesh.interior_cells.size();
    for(Cell* cell: mesh.interior_cells){
        auto V = cell->total_residual;
        sumRho += V[0]*V[0];
        sumU += V[1]*V[1];
        sumV += V[2]*V[2];
        sumP += V[3]*V[3];


    }

    return{sqrt(sumRho)/N,sqrt(sumU)/N,sqrt(sumV)/N,sqrt(sumP)/N};





}





void Flux::compute_residual(){
    bool use_flux_bcs = true;
    array<double, 4> FL = {0.0, 0.0, 0.0, 0.0};
    array<double, 4> FR = {0.0, 0.0, 0.0, 0.0};
    array<double, 4> FD = {0.0, 0.0, 0.0, 0.0};
    array<double, 4> FU = {0.0, 0.0, 0.0, 0.0};
    array<double,4> UL, UR;
    for (Cell* cell : mesh.interior_cells) {

        
        std::function<array<double,4>(array<double,4>,array<double,4>,double, double)> flux_function;

           if (damping_scheme == 1) {
            flux_function = [this](array<double,4> UL, array<double,4> UR, double nx, double ny) {
                return this->vanleer_flux(UL, UR, nx, ny);
            };
        } else {
            //cout<<"Using Roes method";
            flux_function = [this](array<double,4> UL, array<double,4> UR, double nx, double ny) {
                return this->roe_flux(UL, UR, nx, ny);
            };
        }

        if (cell->cell_L) {
            auto ULR = MusclExtrapolation(cell, 'L');  // {UL, UR}
            UL = ULR.first;
            UR = ULR.second;
            FL = flux_function(UR, UL, cell->nx_L, cell->ny_L);
            if(cell->cell_L->name=="Ghost" && cell->cell_L->type == 1 && use_flux_bcs){
                auto V = get_primvars(cell);
                FL = {0,cell->nx_L*V[3],cell->ny_L*V[3],0};
            }   

        }

        if (cell->cell_R) {
            auto ULR = MusclExtrapolation(cell, 'R');
            UL = ULR.first;
            UR = ULR.second;
            FR = flux_function(UL, UR, cell->nx_R, cell->ny_R);
            if(cell->cell_R->name=="Ghost" && cell->cell_R->type == 1 && use_flux_bcs){
                auto V = get_primvars(cell);
                FR = {0,cell->nx_R*V[3],cell->ny_R*V[3],0};
            }   

        }

        if (cell->cell_D) {
            auto ULR = MusclExtrapolation(cell, 'D');
            UL = ULR.first;
            UR = ULR.second;
            FD = flux_function(UR, UL, cell->nx_D, cell->ny_D);
            if(cell->cell_D->name=="Ghost" && cell->cell_D->type == 1 && use_flux_bcs){
                auto V = get_primvars(cell);
                FD = {0,cell->nx_D*V[3],cell->ny_D*V[3],0};
                
            }   
            // else if(cell->cell_D->type == 4)
            //     throw invalid_argument("Encountered type 4");


        }

        if (cell->cell_U) {
            auto ULR = MusclExtrapolation(cell, 'U');
            UL = ULR.first;
            UR = ULR.second;
            FU = flux_function(UL, UR, cell->nx_U, cell->ny_U);
            if(cell->cell_U->name=="Ghost" && cell->cell_U->type == 1 && use_flux_bcs){
                auto V = get_primvars(cell);
                FU = {0,cell->nx_U*V[3],cell->ny_U*V[3],0};
            }   
            //  else if(cell->cell_U->type == 4)
            //     throw invalid_argument("Encountered type 4");

        }
        for(int i=0;i<4;i++)
            if(isnan(FL[i])||isnan(FR[i])||isnan(FU[i])||isnan(FD[i])){
                cout<<"Nan encountered at index: "<<i<<"  ";
                if(i==0)
                    cout<<"Density";
                if(i==3)
                    cout<<"pressure";
                throw invalid_argument("NaN encountered");
            }
        double temp_num = 1.0;
        // if(flag)
        //     temp_num = 0.0;
        for(int i=0;i<4;i++)
            cell->Residual[i] = temp_num*(FL[i]*cell->A_L + FR[i]*cell->A_R + FU[i]*cell->A_U + FD[i]*cell->A_D);

    }

    

    




}

array<double, 4> Flux::get_primvars(array<double,4> U){

    double rho = max(U[0],epsilon);
    double u = U[1] / rho;
    double v = U[2]/rho;
    double p = max(epsilon, (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v)));


    return {rho,u,v,p};
}


array<double, 4> Flux::get_primvars(double* U){

    double rho = max(U[0],epsilon);
    double u = U[1] / rho;
    double v = U[2]/rho;
    double p = max(epsilon, (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v)));


    return {rho,u,v,p};
}










array<double, 4> Flux::get_primvars(Cell* cell){

    double rho = max(cell->U[0],epsilon);
    double u = cell->U[1] / rho;
    double v = cell->U[2]/rho;
    double p = max(epsilon, (gamma - 1.0) * (cell->U[3] - 0.5 * rho * (u * u + v * v)));


    return {rho,u,v,p};
}

void Flux::set_conserved(Cell* cell, array<double, 4> primitive){

        double rho = primitive[0];
        double u = primitive[1];  
        double v = primitive[2];
        double p = primitive[3];  
        
        
        double et = p / ((gamma - 1) * rho) + 0.5 * (u*u+v*v);

        cell->U[0] = rho;
        cell->U[1] = rho*u;
        cell->U[2] = rho*v;
        cell->U[3] = rho*et;
         


}



void Flux::set_source(Cell* cell, array<double, 4> primitive){

        

        cell->Source[0] = primitive[0]; // This is actually a conserved source term. 
        cell->Source[1] =  primitive[1];
        cell->Source[2] =  primitive[2];
        cell->Source[3] =  primitive[3];
         


}

