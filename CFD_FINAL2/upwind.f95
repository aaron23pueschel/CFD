module upwind_module

implicit none





contains

subroutine vanleer_flux(VLeft, VRight, nx, ny, F)



integer, parameter :: imax = 200, jmax = 150,zero=0
real(kind=8) :: F_xsi(4,1:imax,1:jmax-1), F_eta(4,1:imax-1,1:jmax)
logical :: vanleer = .true.
real(kind=8), parameter :: epsilon = 1.0d0
real(kind=8) :: conv

real(kind=8), parameter :: freeze_tol = 4.1d-4,one=1.0d0,two = 2.0d0,quarter = .25d0,half=.5d0,gamma=1.4,four=4.0d0
integer :: clip_count_rho = 0
integer :: clip_count_p = 0
integer :: total_flux_calls = 0






real(kind=8), intent(in) :: VLeft(4), VRight(4), nx, ny
real(kind=8), intent(out) :: F(4)
real(kind=8) :: rhoL, uL, vL, PL, aL, ML, htL, Uhat_L
real(kind=8) :: rhoR, uR, vR, PR, aR, MR, htR, Uhat_R
real(kind=8) :: alpha_p, alpha_m, betaL, betaR, Mp, Mm
real(kind=8) :: Cp, Cm, Dp, Dm, Pp, Pm, FC_normal(4), FP_normal(4), tolerance = 1.0e-6

!$$$$$$ total_flux_calls = total_flux_calls + 1

! Left State Variables
!$$$$$$ rhoL = VLeft(1)
!$$$$$$ rhoL = max(rhoL, tolerance)
if (VLeft(1) < tolerance) clip_count_rho = clip_count_rho + 1
rhoL = max(VLeft(1), tolerance)
uL = VLeft(2)
vL = VLeft(3)
!$$$$$$ PL   = VLeft(4)
!$$$$$$ !$$$$$$ PL   = max(PL, tolerance)
if (VLeft(4) < tolerance) clip_count_p = clip_count_p + 1
PL = max(VLeft(4), tolerance)
Uhat_L = uL*nx + vL*ny
if (rhoL <= zero .or. PL <=zero) then
  print *, "Unphysical left state: rho =", rhoL, " P =", PL
  stop
end if
aL = sqrt(gamma * PL/rhoL)
ML = Uhat_L/aL
htL = (gamma / (gamma - one)) * (PL / rhoL) + half * (uL**two + vL**two)

! Right State Variables
!$$$$$$ rhoR = VRight(1)
!$$$$$$ rhoR = max(rhoR, tolerance)
if (VRight(1) < tolerance) clip_count_rho = clip_count_rho + 1
rhoR = max(VRight(1), tolerance)
uR = VRight(2)
vR = VRight(3)
!$$$$$$ PR   = VRight(4)
!$$$$$$ PR   = max(PR, tolerance)
if (VRight(4) < tolerance) clip_count_p = clip_count_p + 1
PR = max(VRight(4), tolerance)
Uhat_R = uR*nx + vR*ny
if (PR <= zero) then
  print *, "==== CRASH TRACE ===="
  print *, "rhoR =", rhoR
  print *, "PR   =", PR
  print *, "uR   =", uR, "vR =", vR
  print *, "nx, ny =", nx, ny
  print *, "Uhat =", Uhat_R
  print *, "VRight =", VRight
  stop
end if
  
aR = sqrt(gamma * PR/rhoR)
MR = Uhat_R/aR
htR = (gamma / (gamma - one)) * (PR / rhoR) + half * (uR**two + vR**two)

! Flux Splitting
alpha_p = half * (one + sign(one, ML))
alpha_m = half * (one - sign(one, MR))

betaL = -max(zero, int(one) - int(abs(ML)))
betaR = -max(zero, int(one) - int(abs(MR)))

Mp = quarter*(ML + one)**two
Mm = -quarter*(MR - one)**two

Cp = alpha_p*(one + betaL)*ML - betaL*Mp
Cm = alpha_m*(one + betaR)*MR - betaR*Mm

FC_normal(1) = rhoL*aL*Cp*(one) + rhoR*aR*Cm*(1)
FC_normal(2) = rhoL*aL*Cp*(uL) + rhoR*aR*Cm*(uR)
FC_normal(3) = rhoL*aL*Cp*(vL) + rhoR*aR*Cm*(vR)
FC_normal(4) = rhoL*aL*Cp*(htL) + rhoR*aR*Cm*(htR)

Pp = Mp * (-ML + 2.0)
Pm = Mm * (-MR - 2.0)

Dp = (alpha_p * (one + betaL)) - (betaL * Pp)
Dm = (alpha_m * (one + betaR)) - (betaR * Pm)

FP_normal(1) = Dp*(zero) + Dm*(zero)
FP_normal(2) = Dp*(nx*PL) + Dm*(nx*PR)
FP_normal(3) = Dp*(ny*PL) + Dm*(ny*PR)
FP_normal(4) = Dp*(zero) + Dm*(zero)

F(1) = FC_normal(1) + FP_normal(1)
F(2) = FC_normal(2) + FP_normal(2)
F(3) = FC_normal(3) + FP_normal(3)
F(4) = FC_normal(4) + FP_normal(4)

end subroutine vanleer_flux




subroutine roe_flux(VLeft, VRight, nx, ny, F)
  real(kind=8), intent(in) :: VLeft(4), VRight(4), nx, ny
  real(kind=8), intent(out) :: F(4)
  real(kind=8) :: rhoL, uL, vL, PL, htL, Uhat_L, fL(4)
  real(kind=8) :: rhoR, uR, vR, PR, htR, Uhat_R, fR(4)
  real(kind=8) :: Ri, rhobar, ubar, vbar, htbar, abar, Uhatbar
  real(kind=8) :: lambda(4), lambda_mod(4)
  real(kind=8), parameter :: eps = 0.1, tolerance = 1.0e-6
  integer :: i
  real(kind=8) :: r1(4), r2(4), r3(4), r4(4)
  real(kind=8) :: dw(4), drho, dP, du, dv
  
    real(kind=8), parameter :: freeze_tol = 4.1d-4,zero = 0.0d0,one=1.0d0,two = 2.0d0,quarter = .25d0,half=.5d0,gamma=1.4,four=4.0d0
    


    integer :: clip_count_rho = 0
integer :: clip_count_p = 0
integer :: total_flux_calls = 0




  ! Left State Variables
!$$$$$$   rhoL = VLeft(1)
!$$$$$$   rhoL = max(rhoL, 1.0e-6)
  if (VLeft(1) < tolerance) clip_count_rho = clip_count_rho + 1
  rhoL = max(VLeft(1), tolerance)
  uL = VLeft(2)
  vL = VLeft(3)
!$$$$$$   PL = VLeft(4)
!$$$$$$   PL   = max(PL, 1.0e-6)
  if (VLeft(4) < tolerance) clip_count_p = clip_count_p + 1
  PL = max(VLeft(4), tolerance)
  htL = (gamma / (gamma - one)) * (PL / rhoL) + half * (uL**two + vL**two)
  Uhat_L = uL*nx + vL*ny
  
  ! Right State Variables
!$$$$$$   rhoR = VRight(1)
!$$$$$$   rhoR = max(rhoR, 1.0e-6)
if (VRight(1) < tolerance) clip_count_rho = clip_count_rho + 1
rhoR = max(VRight(1), tolerance)
  uR = VRight(2)
  vR = VRight(3)
!$$$$$$   PR   = VRight(4)
!$$$$$$   PR = max(PR, 1.0e-6)
if (VRight(4) < tolerance) clip_count_p = clip_count_p + 1
PR = max(VRight(4), tolerance)
  htR = (gamma / (gamma - one)) * (PR / rhoR) + half * (uR**two + vR**two)
  Uhat_R = uR*nx + vR*ny
  
  ! Roe Averaged Variables
  Ri = sqrt(rhoR/rhoL)
  rhobar = Ri*rhoL
  ubar = ((Ri*uR) + uL)/(Ri + one)
  vbar = ((Ri*vR) + vL)/(Ri + one)
  htbar = ((Ri*htR) + htL)/(Ri + one)
  abar = sqrt((gamma - one)*(htbar - (half*(ubar**two + vbar**two))))
  Uhatbar = ubar*nx + vbar*ny
  
  ! Eigenvalues
  lambda(1) = Uhatbar
  lambda(2) = Uhatbar
  lambda(3) = Uhatbar + abar
  lambda(4) = Uhatbar - abar
  
  do i = 1,4
    if (abs(lambda(i)) >= two*eps*abar) then
      lambda_mod(i) = abs(lambda(i))
    else
      lambda_mod(i) = (((lambda(i))**two)/(four*eps*abar)) + (eps*abar)
    end if
  end do
  
  ! Eigenvectors
  r1(1) = one
  r1(2) = ubar
  r1(3) = vbar
  r1(4) = half*(ubar**two + vbar**two)
  
  r2(1) = zero
  r2(2) = ny*rhobar
  r2(3) = -nx*rhobar
  r2(4) = rhobar*(ny*ubar - nx*vbar)
  
  r3(1) = (half*(rhobar/abar))*one
  r3(2) = (half*(rhobar/abar))*(ubar + nx*abar)
  r3(3) = (half*(rhobar/abar))*(vbar + ny*abar)
  r3(4) = (half*(rhobar/abar))*(htbar + Uhatbar*abar)
  
  r4(1) = -(half*(rhobar/abar))*one
  r4(2) = -(half*(rhobar/abar))*(ubar - nx*abar)
  r4(3) = -(half*(rhobar/abar))*(vbar - ny*abar)
  r4(4) = -(half*(rhobar/abar))*(htbar - Uhatbar*abar)
  
  ! Wave Amplitudes
  drho = rhoR - rhoL
  dP = PR - PL
  du = uR - uL
  dv = vR - vL
  
  dw(1) = drho + (dP/abar**two)
  dw(2) = ny*du - nx*dv
  dw(3) = nx*du + ny*dv + (dP/(rhobar*abar))
  dw(4) = nx*du + ny*dv - (dP/(rhobar*abar))
  
  ! Calculate Flux
  fL(1) = rhoL*Uhat_L
  fL(2) = rhoL*uL*Uhat_L + nx*PL
  fL(3) = rhoL*vL*Uhat_L + ny*PL
  fL(4) = rhoL*htL*Uhat_L
  
  fR(1) = rhoR*Uhat_R
  fR(2) = rhoR*uR*Uhat_R + nx*PR
  fR(3) = rhoR*vR*Uhat_R + ny*PR
  fR(4) = rhoR*htR*Uhat_R
  
  do i = 1,4
    F(i) = (half*(fL(i)+fR(i))) - (half*(lambda_mod(1)*dw(1)*r1(i) &
    + lambda_mod(2)*dw(2)*r2(i) &
    + lambda_mod(3)*dw(3)*r3(i) &
    + lambda_mod(4)*dw(4)*r4(i)))
  end do
  
  
  end subroutine roe_flux





subroutine compute_L_R_states_eta(V,imax,jmax, vl_eta, vr_eta) bind(C)
  use iso_c_binding
  integer(c_int), intent(in) :: imax,jmax
  integer(c_int), parameter :: one = 1
  real(kind=8), intent(in) :: V(4,imax+one,jmax+one)
  real(kind=8), intent(out) :: VL_eta(4,imax-one,jmax), VR_eta(4,imax-one,jmax)


  real(kind=8),parameter :: epsilon=0.0d0
  integer :: i,j,k

  

  do k = 1,4
    do i = 1,imax-1
      ! Internal faces: j = 2 to jmax-2 (so we can safely access j-1 and j+1)
      do j = 1,jmax-2
        VL_eta(k,i,j+1) = V(k,i+1,j+1) + 0.5d0*epsilon*( (V(k,i+1,j+1) - V(k,i+1,j+1-1)))

        VR_eta(k,i,j+1) = V(k,i+1,j+1+1) - 0.5d0*epsilon * ( (V(k,i+1,j+1+2) - V(k,i+1,j+1+1)))
      end do

      ! Left boundary face j = 1
      VL_eta(k,i,1) = V(k,i+1,1+0) + 0.5d0 * (V(k,i+1,1+1) - V(k,i+1,1+0))   ! No limiter
      VR_eta(k,i,1) = V(k,i+1,1+1) - 0.5d0 * (V(k,i+1,1+2) - V(k,i+1,1+1))   ! No limiter

      ! Right boundary face j = jmax
      VL_eta(k,i,jmax) = V(k,i+1,jmax+1-1) + 0.5d0 * (V(k,i+1,jmax+1-1) - V(k,i+1,jmax+1-2))
      VR_eta(k,i,jmax) = V(k,i+1,jmax+1)   - 0.5d0  * (V(k,i+1,jmax+1)   - V(k,i+1,jmax+1-1))


    end do
  end do

end subroutine compute_L_R_states_eta











end  module upwind_module