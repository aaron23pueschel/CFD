#include "Cell.h"
#include "Mesh.h"
#include "Flux.h"
#include <utility> 
#include <cmath>
#include <tuple>
#include <array>
#include <functional>
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

double Flux::mass_mms(double L, double x, double y){

    double mass = (3*pi*uvelx*cos((3*pi*x)/(2.*L))*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L)))/(2.*L) + 
   (2*pi*vvely*cos((2*pi*y)/(3.*L))*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L)))/(3.*L) + 
   (pi*rhox*cos((pi*x)/L)*(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L))))/L - 
   (pi*rhoy*sin((pi*y)/(2.*L))*(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L))))/(2.*L);


    return mass;

}



double Flux::xmtm_mms(double L,double x,double y){


    double xmtm = (2*pi*vvely*cos((2*pi*y)/(3.*L))*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))*
      (uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L))))/(3.*L) - 
   (pi*rhoy*(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L)))*sin((pi*y)/(2.*L))*
      (vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L))))/(2.*L) - 
   (3*pi*uvely*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))*sin((3*pi*y)/(5.*L))*
      (vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L))))/(5.*L);

    return xmtm;


}



double Flux::ymtm_mms(double L,double x,double y){

    double ymtm = (pi*pressy*cos((pi*y)/L))/L + (4*pi*vvely*cos((2*pi*y)/(3.*L))*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))*
      (vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L))))/(3.*L) - 
   (pi*rhoy*sin((pi*y)/(2.*L))*pow(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L)),2))/(2.*L);


    return ymtm;


}





double Flux::energy_mms(double L,double x,double y){


    double energy = (uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L)))*
    ((-2*pi*pressx*sin((2*pi*x)/L))/L + (rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))*
       ((-2*pi*pressx*sin((2*pi*x)/L))/((-1 + gamma)*L*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))) + 
         ((3*pi*uvelx*cos((3*pi*x)/(2.*L))*(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L))))/L - 
            (pi*vvelx*sin((pi*x)/(2.*L))*(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L))))/L)/2. - 
         (pi*rhox*cos((pi*x)/L)*(press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L)))/
          ((-1 + gamma)*L*pow(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L),2))) + 
      (pi*rhox*cos((pi*x)/L)*((pow(0.0,2) + pow(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L)),2) + 
              pow(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L)),2))/2. + 
           (press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L))/
            ((-1 + gamma)*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L)))))/L) + 
   (3*pi*uvelx*cos((3*pi*x)/(2.*L))*(press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L) + 
        (rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))*
         ((pow(0.0,2) + pow(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L)),2) + 
              pow(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L)),2))/2. + 
           (press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L))/
            ((-1 + gamma)*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))))))/(2.*L) + 
   (2*pi*vvely*cos((2*pi*y)/(3.*L))*(press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L) + 
        (rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))*
         ((pow(0.0,2) + pow(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L)),2) + 
              pow(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L)),2))/2. + 
           (press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L))/
            ((-1 + gamma)*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))))))/(3.*L) + 
   (vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L)))*
    ((pi*pressy*cos((pi*y)/L))/L - (pi*rhoy*sin((pi*y)/(2.*L))*
         ((pow(0.0,2) + pow(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L)),2) + 
              pow(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L)),2))/2. + 
           (press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L))/
            ((-1 + gamma)*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L)))))/(2.*L) + 
      (rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))*
       ((pi*pressy*cos((pi*y)/L))/((-1 + gamma)*L*(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L))) + 
         ((-6*pi*uvely*(uvel0 + uvely*cos((3*pi*y)/(5.*L)) + uvelx*sin((3*pi*x)/(2.*L)))*sin((3*pi*y)/(5.*L)))/(5.*L) + 
            (4*pi*vvely*cos((2*pi*y)/(3.*L))*(vvel0 + vvelx*cos((pi*x)/(2.*L)) + vvely*sin((2*pi*y)/(3.*L))))/(3.*L))/2. + 
         (pi*rhoy*sin((pi*y)/(2.*L))*(press0 + pressx*cos((2*pi*x)/L) + pressy*sin((pi*y)/L)))/
          (2.*(-1 + gamma)*L*pow(rho0 + rhoy*cos((pi*y)/(2.*L)) + rhox*sin((pi*x)/L),2))));




    return energy;













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












pair<array<double,4>, array<double,4>> Flux::MusclExtrapolation(Cell* cell,char direction){
        array<double,4> FL;
        array<double,4> FR;
        int i=0;
        

        
        for(;i<4;i++){

            const double tol = 1e-14;

            // ---------- RIGHT face ----------
            if (direction=='R') {
                auto CRR = cell->cell_R->cell_R ? cell->cell_R->cell_R : cell->cell_R;
                auto CL  = cell->cell_L;
                auto CR  = cell->cell_R;

                // p1 at center (cell): r = (U - UL)/(UR - U)
                double dL  =  cell->U[i] - CL->U[i];
                double dR  =  CR->U[i]   - cell->U[i];
                double r   = (std::abs(dR) > tol) ? dL/dR : (dL>0 ? 1e9 : (dL<0 ? -1e9 : 0.0));
                double p1  = (r + std::abs(r)) / (1.0 + std::abs(r)); // van Leer

                // p3 at right cell (CR): rinv = 1/((UR - U)/(URR - UR))
                double dLr =  CR->U[i]   - cell->U[i];
                double dRr =  CRR->U[i]  - CR->U[i];
                double rr  = (std::abs(dRr) > tol) ? dLr/dRr : (dLr>0 ? 1e9 : (dLr<0 ? -1e9 : 0.0));
                double rinv= (std::abs(rr)  > tol) ? 1.0/rr : (rr>0 ? 1e9 : (rr<0 ? -1e9 : 0.0));
                double p3  = (rinv + std::abs(rinv)) / (1.0 + std::abs(rinv)); // van Leer

                FL[i] = cell->U[i]+ 0.5*upwind_order * (p1 * (cell->U[i] - CL->U[i]));
                FR[i] = CR->U[i]- 0.5*upwind_order * (p3 * (CRR->U[i]  - CR->U[i]));
            }

            // ---------- LEFT face ----------
            if (direction=='L') {
                auto CLL = cell->cell_L->cell_L ? cell->cell_L->cell_L : cell->cell_L;
                auto CL  = cell->cell_L;
                auto CR  = cell->cell_R;

                // p1 at left cell (CL): r = (U_CL - U_CLL)/(U_cell - U_CL)
                double dL  =  CL->U[i]   - CLL->U[i];
                double dR  =  cell->U[i] - CL->U[i];
                double r   = (std::abs(dR) > tol) ? dL/dR : (dL>0 ? 1e9 : (dL<0 ? -1e9 : 0.0));
                double p1  = (r + std::abs(r)) / (1.0 + std::abs(r));

                // p3 at center (cell): rinv = 1/((U_cell - U_CL)/(U_CR - U_cell))
                double dLc =  cell->U[i] - CL->U[i];
                double dRc =  CR->U[i]   - cell->U[i];
                double rc  = (std::abs(dRc) > tol) ? dLc/dRc : (dLc>0 ? 1e9 : (dLc<0 ? -1e9 : 0.0));
                double rinv= (std::abs(rc)  > tol) ? 1.0/rc : (rc>0 ? 1e9 : (rc<0 ? -1e9 : 0.0));
                double p3  = (rinv + std::abs(rinv)) / (1.0 + std::abs(rinv));

                FL[i] = CL->U[i]+ 0.5*upwind_order * (p1 * (CL->U[i]   - CLL->U[i]));
                FR[i] = cell->U[i]- 0.5*upwind_order * (p3 * (CR->U[i]   - cell->U[i]));
            }

            // ---------- UP face ----------
            if (direction=='U') {
                auto CUU = cell->cell_U->cell_U ? cell->cell_U->cell_U : cell->cell_U;
                auto CD  = cell->cell_D;
                auto CU  = cell->cell_U;

                // p1 at center (cell): r = (U - UD)/(UU - U)
                double dL  =  cell->U[i] - CD->U[i];
                double dR  =  CU->U[i]   - cell->U[i];
                double r   = (std::abs(dR) > tol) ? dL/dR : (dL>0 ? 1e9 : (dL<0 ? -1e9 : 0.0));
                double p1  = (r + std::abs(r)) / (1.0 + std::abs(r));

                // p3 at up cell (CU): rinv = 1/((U_U - U)/(U_UU - U_U))
                double dLu =  CU->U[i]   - cell->U[i];
                double dRu =  CUU->U[i]  - CU->U[i];
                double ru  = (std::abs(dRu) > tol) ? dLu/dRu : (dLu>0 ? 1e9 : (dLu<0 ? -1e9 : 0.0));
                double rinv= (std::abs(ru)  > tol) ? 1.0/ru : (ru>0 ? 1e9 : (ru<0 ? -1e9 : 0.0));
                double p3  = (rinv + std::abs(rinv)) / (1.0 + std::abs(rinv));

                FL[i] = cell->U[i]+ 0.5*upwind_order * (p1 * (cell->U[i] - CD->U[i]));
                FR[i] = CU->U[i]- 0.5*upwind_order * (p3 * (CUU->U[i]  - CU->U[i]));
            }

            // ---------- DOWN face ----------
            if (direction=='D') {
                auto CDD = cell->cell_D->cell_D ? cell->cell_D->cell_D : cell->cell_D;
                auto CD  = cell->cell_D;
                auto CU  = cell->cell_U;

                // p1 at down cell (CD): r = (U_CD - U_CDD)/(U_cell - U_CD)
                double dL  =  CD->U[i]   - CDD->U[i];
                double dR  =  cell->U[i] - CD->U[i];
                double r   = (std::abs(dR) > tol) ? dL/dR : (dL>0 ? 1e9 : (dL<0 ? -1e9 : 0.0));
                double p1  = (r + std::abs(r)) / (1.0 + std::abs(r));

                // p3 at center (cell): rinv = 1/((U_cell - U_CD)/(U_CU - U_cell))
                double dLc =  cell->U[i] - CD->U[i];
                double dRc =  CU->U[i]   - cell->U[i];
                double rc  = (std::abs(dRc) > tol) ? dLc/dRc : (dLc>0 ? 1e9 : (dLc<0 ? -1e9 : 0.0));
                double rinv= (std::abs(rc)  > tol) ? 1.0/rc : (rc>0 ? 1e9 : (rc<0 ? -1e9 : 0.0));
                double p3  = (rinv + std::abs(rinv)) / (1.0 + std::abs(rinv));

                FL[i] = CD->U[i] + 0.5*upwind_order * (p1 * (CD->U[i]   - CDD->U[i]));
                FR[i] = cell->U[i]- 0.5*upwind_order * (p3 * (CU->U[i]   - cell->U[i]));
            }


    
        }


    return make_pair(FL,FR);







}






array<double,4> Flux::compute_norm(){
    double sumP = 0.0; double sumU = 0.0;double sumV=0.0;double sumRho = 0.0;
    int N = mesh.interior_cells.size();
    for(Cell* cell: mesh.interior_cells){
        auto V = cell->Residual;
        sumRho += V[0]*V[0];
        sumU += V[1]*V[1];
        sumV += V[2]*V[2];
        sumP += V[3]*V[3];


    }

    return{sqrt(sumRho)/N,sqrt(sumU)/N,sqrt(sumV)/N,sqrt(sumP)/N};





}





void Flux::compute_residual(){

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

        bool ghost_flag = false;
        if (cell->cell_L) {
            auto ULR = MusclExtrapolation(cell, 'L');  // {UL, UR}
            UL = ULR.first;
            UR = ULR.second;
            FL = flux_function(UR, UL, cell->nx_L, cell->ny_L);
            if(cell->cell_L->name=="Ghost" && cell->cell_L->type == 1){
                auto V = get_primvars(cell);
                FL = {0,cell->nx_L*V[3],cell->ny_L*V[3],0};
            }   
            // if(cell->cell_L->name=="Ghost" && cell->cell_L->type == 2){
            //     auto V = get_primvars(cell->cell_L);
            //     auto rho = V[0];
            //     auto u = V[1];
            //     auto v = V[2];
            //     auto p = V[3];
            //     auto nx = cell->nx_L;
            //     auto ny = cell->ny_L;
            //     double U = u*nx + v*ny;
            //     auto E =  p/((gamma - 1.0) * rho) + 0.5*(u*u + v*v);
            //     FL = {rho*U,rho*u*U+p*nx,rho*v*U+p*ny,(rho*E+p)*U};
            // }   

        }

        if (cell->cell_R) {
            auto ULR = MusclExtrapolation(cell, 'R');
            UL = ULR.first;
            UR = ULR.second;
            FR = flux_function(UL, UR, cell->nx_R, cell->ny_R);
            if(cell->cell_R->name=="Ghost" && cell->cell_R->type == 1){
                auto V = get_primvars(cell);
                FR = {0,cell->nx_R*V[3],cell->ny_R*V[3],0};
            }   

        }

        if (cell->cell_D) {
            auto ULR = MusclExtrapolation(cell, 'D');
            UL = ULR.first;
            UR = ULR.second;
            FD = flux_function(UR, UL, cell->nx_D, cell->ny_D);
            if(cell->cell_D->name=="Ghost" && cell->cell_D->type == 1){
                auto V = get_primvars(cell);
                FD = {0,cell->nx_D*V[3],cell->ny_D*V[3],0};
            }   

        }

        if (cell->cell_U) {
            auto ULR = MusclExtrapolation(cell, 'U');
            UL = ULR.first;
            UR = ULR.second;
            FU = flux_function(UL, UR, cell->nx_U, cell->ny_U);
            if(cell->cell_U->name=="Ghost" && cell->cell_U->type == 1){
                auto V = get_primvars(cell);
                FU = {0,cell->nx_U*V[3],cell->ny_U*V[3],0};
            }   

        }
        

        for(int i=0;i<4;i++)
            cell->Residual[i] = (FL[i]*cell->A_L + FR[i]*cell->A_R + FU[i]*cell->A_U + FD[i]*cell->A_D);

        //cout << cell->Residual[1]<< ", " << cell->Residual[2];
    }

    

    




}

array<double, 4> Flux::get_primvars(array<double,4> U){

    double rho = max(U[0],epsilon);
    double u = U[1] / rho;
    double v = U[2]/rho;
    double p = max(0.0, (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v)));


    return {rho,u,v,p};
}


array<double, 4> Flux::get_primvars(double* U){

    double rho = max(U[0],epsilon);
    double u = U[1] / rho;
    double v = U[2]/rho;
    double p = max(0.0, (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v)));


    return {rho,u,v,p};
}










array<double, 4> Flux::get_primvars(Cell* cell){

    double rho = max(cell->U[0],epsilon);
    double u = cell->U[1] / rho;
    double v = cell->U[2]/rho;
    double p = max(0.0, (gamma - 1.0) * (cell->U[3] - 0.5 * rho * (u * u + v * v)));


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

