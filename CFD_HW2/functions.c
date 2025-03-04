#include <stdio.h>
#include <math.h>
#include <stdlib.h>

//C FUNCTIONS

double phi(double gamma, double A_bar, double M){
    return ((2/(gamma+1))*((1+((gamma-1)/2)*(M*M))));
}


double FM(double gamma, double A_bar,double M){
    double temp= (pow(phi(gamma,A_bar,M),(gamma+1)/(gamma-1)))-((A_bar*A_bar)*(M*M));
    return temp;

}


double DfdM(double gamma, double A_bar, double M){
    return 2*M*((pow(phi(gamma,A_bar,M),2/(gamma-1)))-A_bar*A_bar);
}


double newton(double gamma, double A_bar, double M,double tolerance){

    double initial_error = fabs(FM(gamma,A_bar,M));
    double error = 1000000.0;
    int max_iter = 100;
    int count = 1;
    double Mk1 = 0;
    
    while(fabs(error/initial_error) > tolerance ) {
        Mk1 = M-FM(gamma,A_bar,M)/DfdM(gamma,A_bar,M);
        error = fabs(FM(gamma,A_bar,Mk1));
        count = count+1;
        if(count==max_iter)
            return 0;
        M = Mk1;  
    } 
    
    //printf("Converged with error tolerance %e",error);
    return Mk1;

}

