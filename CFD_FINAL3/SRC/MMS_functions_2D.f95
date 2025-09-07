!=============================================================================80
module set_precision
  use, intrinsic :: iso_c_binding
  implicit none


  integer, parameter :: dp = c_double

end module
!=============================================================================80
module constants
  use, intrinsic :: iso_c_binding
  use set_precision, only : dp

  implicit none


  real(dp), parameter :: one   = 1.0_dp
  real(dp), parameter :: two   = 2.0_dp
  real(dp), parameter :: three = 3.0_dp
  real(dp), parameter :: four  = 4.0_dp
  real(dp), parameter :: five  = 5.0_dp
  real(dp), parameter :: six   = 6.0_dp

end module

!=============================================================================80
module set_inputs

  use set_precision, only : dp

  private


  public :: initialize_constants
  

  contains
  
  subroutine initialize_constants

    use constants,  only : one

    implicit none


  
  end subroutine initialize_constants 

end module
!=============================================================================80
module mms_constants

  use set_precision, only : dp

  implicit none

 ! NOTE: These are currently set up to run the supersonic manufactured solution
  real(dp), parameter :: rho0   = 1.0_dp
  real(dp), parameter :: rhox   = 0.15_dp
  real(dp), parameter :: rhoy   = -0.1_dp
  real(dp), parameter :: uvel0  = 800.0_dp
  real(dp), parameter :: uvelx  = 50.0_dp
  real(dp), parameter :: uvely  = -30.0_dp
  real(dp), parameter :: vvel0  = 800.0_dp
  real(dp), parameter :: vvelx  = -75.0_dp
  real(dp), parameter :: vvely  = 40.0_dp
  real(dp), parameter :: wvel0  = 0.0_dp
  real(dp), parameter :: wvelx  = 0.0_dp
  real(dp), parameter :: wvely  = 0.0_dp
  real(dp), parameter :: press0 = 100000.0_dp
  real(dp), parameter :: pressx = 20000.0_dp
  real(dp), parameter :: pressy = 50000.0_dp
  
end module mms_constants



module mms_interface
  use, intrinsic :: iso_c_binding
  use set_precision, only : dp
  use constants,      only : one, two, three, four, five, six
  use mms_constants,  only : &
       rho0, rhox, rhoy, &
       uvel0, uvelx, uvely, &
       vvel0, vvelx, vvely, &
       wvel0, wvelx, wvely, &
       press0, pressx, pressy
  implicit none

contains
!=============================================================================80
!!!!MMS functions
!=============================================================================80
subroutine rmassconv(length,x,y,rmassconv_) bind(C, name="rmassconv")



  implicit none

  real(dp), intent(in) :: x
  real(dp), intent(in) :: y
  real(dp), intent(in) :: length
  real(dp),intent(out) :: rmassconv_
  real(dp), parameter :: pi = 3.1415926_dp




  rmassconv_ = (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                 &
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length))) /        &
    (two*length) + (two*pi*vvely*cos((two*pi*y)/(three*length)) *              &
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length))) /        &
    (three*length) + (pi*rhox*cos((pi*x)/length) *                             &
    (uvel0 + uvely*cos((three*pi*y)/(five*length)) + uvelx*sin((three*pi*x)/   &
    (two*length))))/length - (pi*rhoy*sin((pi*y)/(two*length)) *               &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) + vvely*sin((two*pi*y) /           &
    (three*length))))/(two*length)

end subroutine rmassconv
!=============================================================================80
subroutine xmtmconv(length,x,y,xmtmconv_) bind(C, name="xmtmconv")


  implicit none

  real(dp), intent(in) :: x
  real(dp), intent(in) :: y
  real(dp), intent(in) :: length
  real(dp),intent(out) :: xmtmconv_
  real(dp), parameter :: pi = 3.1415926_dp

  xmtmconv_ = (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                  &
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length)) *         &
    (uvel0 + uvely*cos((three*pi*y)/(five*length)) +                           &
    uvelx*sin((three*pi*x)/(two*length))))/length +                            &
    (two*pi*vvely*cos((two*pi*y) /                                             &
    (three*length))*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    &
    rhox*sin((pi*x)/length))*(uvel0 + uvely*cos((three*pi*y) /                 &
    (five*length)) + uvelx*sin((three*pi*x)/(two*length))))/(three*length) +   &
    (pi*rhox*cos((pi*x)/length)*(uvel0 + uvely*cos((three*pi*y) /              &
    (five*length)) + uvelx*sin((three*pi*x)/(two*length)))**2)/length -        &
    (two*pi*pressx*sin((two*pi*x)/length))/length -                            &
    (pi*rhoy*(uvel0 + uvely*cos((three*pi*y)/(five*length)) +                  &
    uvelx*sin((three*pi*x)/(two*length)))*sin((pi*y)/(two*length))*            &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  &
    vvely*sin((two*pi*y)/(three*length))))/(two*length) -                      &
    (three*pi*uvely*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    &
    rhox*sin((pi*x)/length))*sin((three*pi*y)/(five*length))*(vvel0 + vvelx *  &
    cos((pi*x)/(two*length)) + vvely*sin((two*pi*y)/(three*length)))) /        &
    (five*length)

end subroutine xmtmconv
!=============================================================================80
subroutine ymtmconv(length,x,y,ymtmconv_) bind(C, name="ymtmconv")


  real(dp), intent(in) :: x
  real(dp), intent(in) :: y
  real(dp), intent(in) :: length
  real(dp),intent(out) :: ymtmconv_
  real(dp), parameter :: pi = 3.1415926_dp

  ymtmconv_ = (pi*pressy*cos((pi*y)/length))/length -                           &
    (pi*vvelx*sin((pi*x)/(two*length))*(rho0 + rhoy*cos((pi*y)/(two*length)) + &
    rhox*sin((pi*x)/length))*(uvel0 + uvely*cos((three*pi*y)/(five*length)) +  &
    uvelx*sin((three*pi*x)/(two*length))))/(two*length) +                      &
    (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                           &
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length)) *         &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  &
    vvely*sin((two*pi*y)/(three*length))))/(two*length) +                      &
    (four*pi*vvely*cos((two*pi*y) /                                            &
    (three*length))*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    &
    rhox*sin((pi*x)/length))*(vvel0 + vvelx*cos((pi*x)/(two*length)) +         &
    vvely*sin((two*pi*y)/(three*length))))/(three*length) +                    &
    (pi*rhox*cos((pi*x)/length) *                                              &
    (uvel0 + uvely*cos((three*pi*y)/(five*length)) +                           &
    uvelx*sin((three*pi*x)/(two*length))) *                                    &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  &
    vvely*sin((two*pi*y)/(three*length))))/length -                            &
    (pi*rhoy*sin((pi*y)/(two*length)) *                                        &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  &
    vvely*sin((two*pi*y)/(three*length)))**2)/(two*length)

end subroutine ymtmconv
!=============================================================================80
subroutine energyconv(gamma, length,x,y,energyconv_) bind(C, name="energyconv")



  implicit none

  real(dp), intent(in) :: gamma
  real(dp), intent(in) :: length
  real(dp), intent(in) :: x
  real(dp), intent(in) :: y
  real(dp),intent(out) :: energyconv_
  real(dp), parameter :: pi = 3.1415926_dp

  energyconv_ = (uvel0 + uvely*cos((three*pi*y)/(five*length)) +                &
    uvelx*sin((three*pi*x)/(two*length)))*((-two*pi*pressx*sin((two*pi*x) /    &
    length))/length + (rho0 + rhoy*cos((pi*y)/(two*length)) +                  &
    rhox*sin((pi*x)/length))*((-two*pi*pressx*sin((two*pi*x)/length))/         &
    ((-one + gamma)*length*(rho0 + rhoy*cos((pi*y)/(two*length)) +             &
    rhox*sin((pi*x)/length))) + ((three*pi*uvelx*cos((three*pi*x) /            &
    (two*length))*(uvel0 + uvely*cos((three*pi*y)/(five*length)) +             &
    uvelx*sin((three*pi*x)/(two*length))))/length - (pi*vvelx*sin((pi*x) /     &
    (two*length))*(vvel0 + vvelx*cos((pi*x)/(two*length)) +                    &
    vvely*sin((two*pi*y)/(three*length))))/length)/two - (pi*rhox*cos((pi*x) / &
    length)*(press0 + pressx*cos((two*pi*x)/length) +                          & 
    pressy*sin((pi*y)/length)))/((-one + gamma)*length*(rho0 + rhoy*cos((pi*y)/&
    (two*length)) + rhox*sin((pi*x)/length))**2)) +                            &
    (pi*rhox*cos((pi*x)/length)*((wvel0**2 + (uvel0 + uvely*cos((three*pi*y) / &
    (five*length)) + uvelx*sin((three*pi*x)/(two*length)))**2 +                &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) + vvely*sin((two*pi*y) /           &
    (three*length)))**2)/two + (press0 + pressx*cos((two*pi*x)/length) +       &
    pressy*sin((pi*y)/length))/((-one + gamma) *                               &
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    &
    rhox*sin((pi*x)/length)))))/length) +                                      &
    (three*pi*uvelx*cos((three*pi*x)/(two*length)) *                           &
    (press0 + pressx*cos((two*pi*x)/length) + pressy*sin((pi*y)/length) +      &
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length))*          &
    ((wvel0**2 + (uvel0 + uvely*cos((three*pi*y)/(five*length)) +              &
    uvelx*sin((three*pi*x)/(two*length)))**2 +                                 &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  &
    vvely*sin((two*pi*y)/(three*length)))**2)/two +                            &
    (press0 + pressx*cos((two*pi*x)/length) +                                  &
    pressy*sin((pi*y)/length))/((-one + gamma) *                               &
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    &
    rhox*sin((pi*x)/length))))))/(two*length) +                                &
    (two*pi*vvely*cos((two*pi*y)/(three*length)) *                             &
    (press0 + pressx*cos((two*pi*x)/length) +                                  &
    pressy*sin((pi*y)/length) + (rho0 + rhoy*cos((pi*y)/(two*length)) +        &
    rhox*sin((pi*x)/length))*((wvel0**2 +                                      &
    (uvel0 + uvely*cos((three*pi*y)/(five*length)) +                           &
    uvelx*sin((three*pi*x)/(two*length)))**2 +                                 &
    (vvel0 + vvelx*cos((pi*x)/(two*length)) +                                  &
    vvely*sin((two*pi*y)/(three*length)))**2)/two +                            &
    (press0 + pressx*cos((two*pi*x)/length) + pressy*sin((pi*y)/length)) /     &
	((-one + gamma)*(rho0 + rhoy*cos((pi*y)/(two*length)) +                    &
    rhox*sin((pi*x)/length))))))/(three*length) + (vvel0 + vvelx*cos((pi*x) /  &
    (two*length)) + vvely*sin((two*pi*y)/(three*length))) *                    &
    ((pi*pressy*cos((pi*y)/length))/length - (pi*rhoy*sin((pi*y)/(two*length))*&
    ((wvel0**2 + (uvel0 + uvely*cos((three*pi*y)/(five*length)) +              &
    uvelx*sin((three*pi*x)/(two*length)))**2 + (vvel0 + vvelx *                &
    cos((pi*x)/(two*length)) + vvely*sin((two*pi*y)/(three*length)))**2)/two + &
    (press0 + pressx*cos((two*pi*x)/length) +                                  &
    pressy*sin((pi*y)/length))/((-one + gamma) *                               &
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    &
    rhox*sin((pi*x)/length)))))/(two*length) +                                 &
    (rho0 + rhoy*cos((pi*y)/(two*length)) +                                    &
    rhox*sin((pi*x)/length))*((pi*pressy*cos((pi*y)/length)) /                 &
    ((-one + gamma)*length*(rho0 + rhoy*cos((pi*y)/(two*length)) +             &
    rhox*sin((pi*x)/length))) +                                                &
    ((-six*pi*uvely*(uvel0 + uvely*cos((three*pi*y) /                          &
    (five*length)) + uvelx*sin((three*pi*x)/(two*length))) *                   &
    sin((three*pi*y)/(five*length)))/(five*length) +                           &
    (four*pi*vvely*cos((two*pi*y) /                                            &
    (three*length))*(vvel0 + vvelx*cos((pi*x)/(two*length)) +                  &
    vvely*sin((two*pi*y)/(three*length))))/(three*length))/two +               &
    (pi*rhoy*sin((pi*y)/(two*length))*(press0 + pressx*cos((two*pi*x)/length) +&
    pressy*sin((pi*y)/length)))/(two*(-one + gamma)*length*                    &
    (rho0 + rhoy*cos((pi*y)/(two*length)) + rhox*sin((pi*x)/length))**2)))

end subroutine energyconv


end module mms_interface