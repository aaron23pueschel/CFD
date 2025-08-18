module flux_module


implicit none







integer, parameter :: dp = kind(1.0d0)
integer, parameter :: imax = 200, jmax = 150,zero=0
real(dp) :: F_xsi(4,1:imax,1:jmax-1), F_eta(4,1:imax-1,1:jmax)
logical :: vanleer = .true.
real(dp), parameter :: epsilon = 1.0_dp
real(dp) :: conv


integer :: clip_count_rho = 0
integer :: clip_count_p = 0
integer :: total_flux_calls = 0


logical :: freeze_limiters = .false.
real(dp) :: psi_p_xsi_frozen(4,1:imax,1:jmax-1), psi_m_xsi_frozen(4,1:imax,1:jmax-1)
real(dp) :: psi_p_eta_frozen(4,1:imax-1,1:jmax), psi_m_eta_frozen(4,1:imax-1,1:jmax)
real(dp), parameter :: freeze_tol = 4.1e-4_dp,one=1.0_dp,two = 2.0_dp,quarter = .25_dp,half=.5_dp,gamma=1.4,four=4.0_dp


contains

subroutine compute_limiter_xsi(V, psi_p_xsi, psi_m_xsi)

real(dp), intent(in) :: V(4,0:imax,0:jmax)
real(dp), intent(out) :: psi_p_xsi(4,1:imax,1:jmax-1), psi_m_xsi(4,1:imax,1:jmax-1)
real(dp) :: rp_xsi(4,1:imax,1:jmax-1), rm_xsi(4,1:imax,1:jmax-1), den
integer :: i,j, k
real(dp), parameter :: delta = 1.0e-6_dp

if (freeze_limiters) then
  psi_p_xsi = psi_p_xsi_frozen
  psi_m_xsi = psi_m_xsi_frozen
  return
end if

! Calculate xsi direction limiters (i+1/2,j)
do k = 1,4
  do j = 1,jmax-1
    do i = 1,imax-2
      den = sign(one, V(k,i+1,j) - V(k,i,j))*max(abs(V(k,i+1,j) - V(k,i,j)), delta)
      rp_xsi(k,i+1,j) = (V(k,i+2,j) - V(k,i+1,j))/den
      rm_xsi(k,i+1,j) = (V(k,i,j) - V(k,i-1,j))/den
    end do
    
    ! Face 1
    rp_xsi(k,1,j) = (V(k,2,j) - V(k,1,j)) / &
    (sign(one, V(k,1,j) - V(k,0,j))*max(abs(V(k,1,j) - V(k,0,j)),delta))
    
    rm_xsi(k,1,j) = (V(k,1,j) - V(k,0,j)) / &
    (sign(one, V(k,2,j) - V(k,1,j))*max(abs(V(k,2,j) - V(k,1,j)),delta))

    ! Face imax
    rp_xsi(k,imax,j) = (V(k,imax,j) - V(k,imax-1,j)) / &
    (sign(1.0_dp, V(k,imax-1,j) - V(k,imax-2,j)) * max(abs(V(k,imax-1,j) - V(k,imax-2,j)), delta))
    
    rm_xsi(k,imax,j) = (V(k,imax-1,j) - V(k,imax-2,j)) / &
    (sign(1.0_dp, V(k,imax,j) - V(k,imax-1,j)) * max(abs(V(k,imax,j) - V(k,imax-1,j)), delta))
  
    do i = 1,imax
      psi_p_xsi(k,i,j) = (rp_xsi(k,i,j) + abs(rp_xsi(k,i,j)))/(one + abs(rp_xsi(k,i,j)))
      psi_m_xsi(k,i,j) = (rm_xsi(k,i,j) + abs(rm_xsi(k,i,j)))/(one + abs(rm_xsi(k,i,j)))

!$$$$$$       psi_p_xsi(k,i,j) = max(0.1, psi_p_xsi(k,i,j))
!$$$$$$       psi_m_xsi(k,i,j) = max(0.1, psi_m_xsi(k,i,j))
    end do

  end do
end do

psi_p_xsi_frozen = psi_p_xsi
psi_m_xsi_frozen = psi_m_xsi

end subroutine compute_limiter_xsi

subroutine compute_L_R_states_xsi(V, VL_xsi, VR_xsi)
real(dp), intent(in) :: V(4,0:imax,0:jmax)
real(dp), intent(out) :: VL_xsi(4,1:imax,1:jmax-1), VR_xsi(4,1:imax,1:jmax-1)
real(dp) :: psi_p_xsi(4,1:imax,1:jmax-1), psi_m_xsi(4,1:imax,1:jmax-1)
integer :: i,j,k

! Check residual norm here
if (conv < freeze_tol .and. .not. freeze_limiters) then
  freeze_limiters = .true.
end if

  call compute_limiter_xsi(V, psi_p_xsi, psi_m_xsi)

do k = 1,4
  do j = 1,jmax-1
    do i = 1,imax-2   ! <-- start at i=2 to allow V(i-1), and stop at i=imax-2 to allow V(i+2)
      VL_xsi(k,i+1,j) = V(k,i,j) + 0.5_dp*epsilon *(psi_p_xsi(k,i,j) * (V(k,i,j) - V(k,i-1,j)))
      VR_xsi(k,i+1,j) = V(k,i+1,j) - 0.5_dp*epsilon *(psi_m_xsi(k,i+2,j)   * (V(k,i+2,j) - V(k,i+1,j)))
    end do

    ! === Extrapolate left and right states at i = 1 (left boundary face)
    VL_xsi(k,1,j) = V(k,0,j) + 0.5_dp * psi_p_xsi(k,1,j) * (V(k,1,j) - V(k,0,j))  ! No limiter
    VR_xsi(k,1,j) = V(k,1,j) - 0.5_dp * psi_m_xsi(k,2,j) * (V(k,2,j) - V(k,1,j))  ! No limiter

    ! === Extrapolate at i = imax (right boundary face)
    VL_xsi(k,imax,j) = V(k,imax-1,j) + 0.5_dp * psi_p_xsi(k,imax-1,j) * (V(k,imax-1,j) - V(k,imax-2,j))
    VR_xsi(k,imax,j) = V(k,imax,j)   - 0.5_dp * psi_m_xsi(k,imax,j) * (V(k,imax,j)   - V(k,imax-1,j))

!$$$$$$ ! === Extrapolate left and right states at i = 1 (left boundary face)
!$$$$$$     VL_xsi(k,1,j) = V(k,0,j) + 0.5_dp * (V(k,1,j) - V(k,0,j))  ! No limiter
!$$$$$$     VR_xsi(k,1,j) = V(k,1,j) - 0.5_dp * (V(k,2,j) - V(k,1,j))  ! No limiter
!$$$$$$ 
!$$$$$$     ! === Extrapolate at i = imax (right boundary face)
!$$$$$$     VL_xsi(k,imax,j) = V(k,imax-1,j) + 0.5_dp * (V(k,imax-1,j) - V(k,imax-2,j))
!$$$$$$     VR_xsi(k,imax,j) = V(k,imax,j)   - 0.5_dp * (V(k,imax,j)   - V(k,imax-1,j))
   
  end do
end do

end subroutine compute_L_R_states_xsi

subroutine compute_limiter_eta(V, psi_p_eta, psi_m_eta)
real(dp), intent(in) :: V(4,0:imax,0:jmax)
real(dp), intent(out) :: psi_p_eta(4,1:imax-1,1:jmax), psi_m_eta(4,1:imax-1,1:jmax)
real(dp) :: rp_eta(4,1:imax-1,1:jmax), rm_eta(4,1:imax-1,1:jmax), den
integer :: i,j, k
real(dp), parameter :: delta = 1.0e-6_dp

if (freeze_limiters) then
  psi_p_eta = psi_p_eta_frozen
  psi_m_eta = psi_m_eta_frozen
  return
end if


! Calculate eta direction limiters (i,j+1/2)
do k = 1,4
  do i = 1,imax-1
    do j = 1,jmax-2
      den = sign(one,V(k,i,j+1) - V(k,i,j))*max(abs(V(k,i,j+1) - V(k,i,j)), delta)
      rp_eta(k,i,j+1) = (V(k,i,j+2) - V(k,i,j+1))/den
      rm_eta(k,i,j+1) = (V(k,i,j) - V(k,i,j-1))/den
    end do

    ! Face 1
    rp_eta(k,i,1) = (V(k,i,2) - V(k,i,1)) / &
    (sign(one, V(k,i,1) - V(k,i,0))*max(abs(V(k,i,1) - V(k,i,0)),delta))
    
    rm_eta(k,i,1) = (V(k,i,1) - V(k,i,0)) / &
    (sign(one, V(k,i,2) - V(k,i,1))*max(abs(V(k,i,2) - V(k,i,1)),delta))

    ! Face jmax
    rp_eta(k,i,jmax) = (V(k,i,jmax) - V(k,i,jmax-1)) / &
    (sign(one, V(k,i,jmax-1) - V(k,i,jmax-2)) * max(abs(V(k,i,jmax-1) - V(k,i,jmax-2)), delta))
    
    rm_eta(k,i,jmax) = (V(k,i,jmax-1) - V(k,i,jmax-2)) / &
    (sign(one, V(k,i,jmax) - V(k,i,jmax-1)) * max(abs(V(k,i,jmax) - V(k,i,jmax-1)), delta))

    do j = 1,jmax
      psi_p_eta(k,i,j) = (rp_eta(k,i,j) + abs(rp_eta(k,i,j)))/(one + abs(rp_eta(k,i,j)))
      psi_m_eta(k,i,j) = (rm_eta(k,i,j) + abs(rm_eta(k,i,j)))/(one + abs(rm_eta(k,i,j)))

!$$$$$$       psi_p_eta(k,i,j) = max(0.1, psi_p_eta(k,i,j))
!$$$$$$       psi_m_eta(k,i,j) = max(0.1, psi_m_eta(k,i,j))
    end do
  end do
end do

psi_p_eta_frozen = psi_p_eta
psi_m_eta_frozen = psi_m_eta

end subroutine compute_limiter_eta

      
subroutine compute_L_R_states_eta(V, VL_eta, VR_eta)

  implicit none
  real(dp), intent(in) :: V(4,0:imax,0:jmax)
  real(dp), intent(out) :: VL_eta(4,1:imax-1,1:jmax), VR_eta(4,1:imax-1,1:jmax)
  real(dp) :: psi_p_eta(4,1:imax-1,1:jmax), psi_m_eta(4,1:imax-1,1:jmax)
  integer :: i,j,k

  call compute_limiter_eta(V, psi_p_eta, psi_m_eta)

  do k = 1,4
    do i = 1,imax-1
      ! Internal faces: j = 2 to jmax-2 (so we can safely access j-1 and j+1)
      do j = 1,jmax-2
        VL_eta(k,i,j+1) = V(k,i,j) + 0.5_dp*epsilon*(psi_p_eta(k,i,j) * (V(k,i,j) - V(k,i,j-1)))

        VR_eta(k,i,j+1) = V(k,i,j+1) - 0.5_dp*epsilon * (psi_m_eta(k,i,j+2)   * (V(k,i,j+2) - V(k,i,j+1)))
      end do

      ! Left boundary face j = 1
      VL_eta(k,i,1) = V(k,i,0) + 0.5_dp * psi_p_eta(k,i,1) * (V(k,i,1) - V(k,i,0))   ! No limiter
      VR_eta(k,i,1) = V(k,i,1) - 0.5_dp * psi_m_eta(k,i,2) * (V(k,i,2) - V(k,i,1))   ! No limiter

      ! Right boundary face j = jmax
      VL_eta(k,i,jmax) = V(k,i,jmax-1) + 0.5_dp * psi_p_eta(k,i,jmax-1) * (V(k,i,jmax-1) - V(k,i,jmax-2))
      VR_eta(k,i,jmax) = V(k,i,jmax)   - 0.5_dp * psi_m_eta(k,i,jmax) * (V(k,i,jmax)   - V(k,i,jmax-1))

!$$$$$$ ! Left boundary face j = 1
!$$$$$$       VL_eta(k,i,1) = V(k,i,0) + 0.5_dp * (V(k,i,1) - V(k,i,0))   ! No limiter
!$$$$$$       VR_eta(k,i,1) = V(k,i,1) - 0.5_dp * (V(k,i,2) - V(k,i,1))   ! No limiter
!$$$$$$ 
!$$$$$$       ! Right boundary face j = jmax
!$$$$$$       VL_eta(k,i,jmax) = V(k,i,jmax-1) + 0.5_dp * (V(k,i,jmax-1) - V(k,i,jmax-2))
!$$$$$$       VR_eta(k,i,jmax) = V(k,i,jmax)   - 0.5_dp * (V(k,i,jmax)   - V(k,i,jmax-1))
    end do
  end do

end subroutine compute_L_R_states_eta


subroutine vanleer_flux(VLeft, VRight, nx, ny, F)
real(dp), intent(in) :: VLeft(4), VRight(4), nx, ny
real(dp), intent(out) :: F(4)
real(dp) :: rhoL, uL, vL, PL, aL, ML, htL, Uhat_L
real(dp) :: rhoR, uR, vR, PR, aR, MR, htR, Uhat_R
real(dp) :: alpha_p, alpha_m, betaL, betaR, Mp, Mm
real(dp) :: Cp, Cm, Dp, Dm, Pp, Pm, FC_normal(4), FP_normal(4), tolerance = 1.0e-6

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
  real(dp), intent(in) :: VLeft(4), VRight(4), nx, ny
  real(dp), intent(out) :: F(4)
  real(dp) :: rhoL, uL, vL, PL, htL, Uhat_L, fL(4)
  real(dp) :: rhoR, uR, vR, PR, htR, Uhat_R, fR(4)
  real(dp) :: Ri, rhobar, ubar, vbar, htbar, abar, Uhatbar
  real(dp) :: lambda(4), lambda_mod(4)
  real(dp), parameter :: eps = 0.1, tolerance = 1.0e-6
  integer :: i
  real(dp) :: r1(4), r2(4), r3(4), r4(4)
  real(dp) :: dw(4), drho, dP, du, dv
  

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



subroutine compute_flux_xsi(vanleer, V, nhat_xsi, F_xsi)
logical, intent(in) :: vanleer
real(dp), intent(in) :: V(4,0:imax,0:jmax), nhat_xsi(2,1:imax, 1:jmax-1)
real(dp), intent(out) :: F_xsi(4,1:imax,1:jmax-1)
integer i, j
real(dp) :: VL_xsi(4,1:imax,1:jmax-1), VR_xsi(4,1:imax,1:jmax-1), F(4)

! Calculate Left and Right States
call compute_L_R_states_xsi(V, VL_xsi, VR_xsi)

if (vanleer) then
  do j = 1,jmax-1
    do i = 1,imax
      call vanleer_flux(VL_xsi(:,i,j), VR_xsi(:,i,j), nhat_xsi(1,i,j), nhat_xsi(2,i,j), F)
      F_xsi(:,i,j) = F
    end do
  end do
else
  do j = 1,jmax-1
    do i = 1,imax
      call roe_flux(VL_xsi(:,i,j), VR_xsi(:,i,j), nhat_xsi(1,i,j), nhat_xsi(2,i,j), F)
      F_xsi(:,i,j) = F
    end do
  end do
end if

end subroutine compute_flux_xsi

subroutine compute_flux_eta(vanleer, V, nhat_eta, F_eta)
logical, intent(in) :: vanleer
real(dp), intent(in) :: V(4,0:imax,0:jmax), nhat_eta(2,1:imax-1, 1:jmax)
real(dp), intent(out) :: F_eta(4,1:imax-1,1:jmax)
integer i, j
real(dp) :: VL_eta(4,1:imax-1,1:jmax), VR_eta(4,1:imax-1,1:jmax), F(4)

! Calculate Left and Right States
call compute_L_R_states_eta(V, VL_eta, VR_eta)

if (vanleer) then
  do i = 1,imax-1
    do j = 1,jmax
      call vanleer_flux(VL_eta(:,i,j), VR_eta(:,i,j), nhat_eta(1,i,j), nhat_eta(2,i,j), F)
      F_eta(:,i,j) = F
    end do
  end do
else
  do i = 1,imax-1
    do j = 1,jmax
      call roe_flux(VL_eta(:,i,j), VR_eta(:,i,j), nhat_eta(1,i,j), nhat_eta(2,i,j), F)
      F_eta(:,i,j) = F
    end do
  end do
end if
end subroutine compute_flux_eta

end module flux_module