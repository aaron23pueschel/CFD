import f90nml
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
from DataStructures import DataStructures
from Upwind import Upwind
class EulerSolver:
    def __init__(self,input_file):
        print(os.getcwd())
        if input_file is not None:
            try:
                nml = f90nml.read(input_file)
            except FileNotFoundError:
                print("Check namelist filename")

            # Input vars
            self.p0 = nml["SimulationVariables"]["p0"]
            self.NI = nml["SimulationVariables"]["NI"]
            self.NJ = nml["SimulationVariables"]["NJ"]
            self.T0 = nml["SimulationVariables"]["T0"]
            self.Ru = nml["SimulationVariables"]["Ru"]
            self.M = nml["SimulationVariables"]["M"]
            self.R = self.Ru/self.M
            self.CFL = nml["SimulationVariables"]["CFL"]
            
            self.gamma = nml["SimulationVariables"]["gamma"]
            self.epsilon = nml["SimulationVariables"]["epsilon"]
            self.kappa = nml["UpwindVariables"]["kappa"]
            self.upwind_order = nml["UpwindVariables"]["upwind_order"] #! 0 for first order 1 for second order
            self.convergence_criteria = nml["SimulationVariables"]["convergence_criteria"]
            self.damping_scheme = nml["UpwindVariables"]["damping_scheme"]#1 ! 0 for jameson damping, 1 for van leer upwind
            self.iter_max = nml["SimulationVariables"]["iter_max"]
            self.local_timestep = nml["SimulationVariables"]["local_timestep"]
            self.flux_limiter_scheme = nml["UpwindVariables"]["flux_limiter_scheme"] #0 for freeze, 1 for van leer, 2 for van albada
            self.return_conserved = nml["SimulationVariables"]["return_conserved"]
        # Class variables

        self.p_idx = 3
        self.u_idx = 1
        self.v_idx = 2
        self.rho_idx = 0

        self.xxnormal = 0
        self.yynormal = 1
        self.unormal = 2
        self.vnormal = 3

        self.p = None
        self.u = None
        self.x = None
        self.rho = None
        
       
        self.residual = None
        self.mach = None
        self.delta_t = None
        self.AREA = None

        self.Data = DataStructures(self.NI,self.NJ)
        self.Upwind = Upwind(self.Data,self.kappa,self.upwind_order,self.damping_scheme,self.flux_limiter_scheme)



    
    def set_boundary_multiplier(self):
        bc_multiplier_top = np.ones_like(self.Data.F)
        bc_multiplier_top[:,-1,:] = 0 # Top

        bc_multiplier_bottom = np.ones_like(self.Data.F)
        bc_multiplier_bottom[:,0,:] = 0 
        #bc_multiplier[:,:,0] = 0 
        
        self.Data.bc_multiplier = [bc_multiplier_top,bc_multiplier_bottom]
        
    
        
    def set_initial_conditions(self):
        
        mach = 1.5*np.ones((self.NI-1,self.NJ-1))
        T = self.total_T(self.gamma,mach,self.T0)
        p = self.total_p(self.gamma, mach,self.p0) # Number of cells
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,mach,self.R,T)
        
        
        self.Data.V[self.rho_idx] = rho
        self.Data.V[self.u_idx] = 0*u*self.Data.upward_S[self.unormal]
        self.Data.V[self.v_idx] = 0*u*self.Data.upward_S[self.vnormal]
        self.Data.V[self.p_idx] = p

        self.Data.U,_,_ = self.primitive_to_conserved(self.Data.V)

    def set_inflow_bcs(self):

        mach = 2.1*np.ones((self.NI-1))
        T = self.total_T(self.gamma,mach,self.T0)
        p = self.total_p(self.gamma, mach,self.p0) # Number of cells
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,mach,self.R,T)
        
        
        self.Data.V[self.rho_idx,:,0] = rho
        self.Data.V[self.u_idx,:,0] = u*self.Data.upward_S[self.unormal,:,0]
        self.Data.V[self.v_idx,:,0] = u*self.Data.upward_S[self.vnormal,:,0]
        self.Data.V[self.p_idx,:,0] = p


        self.Data.U,_,_ = self.primitive_to_conserved(self.Data.V)


    
    


    def RUN_SIMULATION(self,output_quantity = 100,verbose=False ):
        self.set_arrays()
        self.set_geometry()
        rho_exact,u_exact,p_exact = ([],[],[])
        if self.compute_isentropic:
            rho_exact,u_exact,p_exact = self.exact_isentropic()
        self.set_initial_conditions()
        self.set_boundary_conditions()

        R1 = self.iteration_step()
        self.set_boundary_conditions()
        
        convergence_history = []

        for i in range(self.iter_max):
            compute_norm = False
            if i%output_quantity==0:
                compute_norm = True
            R = self.iteration_step(return_error = compute_norm)
            ##print(self.Data.V[-1,1])
            if compute_norm:
                convergence_history.append(R/R1)
                if np.max(R/R1)<self.converge_at_second_order:
                    self.upwind_order = 1
                if np.max(R/R1)<self.convergence_criteria:
                    print("Converged at Rk/R1: "+ str(np.max(R/R1))+" in "+str(i)+" iterations.") 
                    break
                if verbose:
                    print("Iteration: "+str((i+1)),np.max(R/R1))
            self.set_boundary_conditions()
            
        p_compute = self.Data.V[:,2]
        u_compute = self.Data.V[:,1]
        rho_compute = self.Data.V[:,0]
        if not self.return_conserved:
            return p_compute,u_compute,rho_compute,np.array(p_exact),np.array(u_exact),np.array(rho_exact),convergence_history
        else:
            Mass_compute = self.Data.U[1:-1,0]
            Momentum_compute = self.Data.U[1:-1,1]
            Energy_compute = self.Data.U[1:-1,2]
            U,_ = self.primitive_to_conserved(np.array([rho_exact,u_exact,p_exact]).T)
            
            Mass_exact = U[:,0]
            Momentum_exact = U[:,1]
            Energy_exact = U[:,2]

            return Mass_compute,Momentum_compute,Energy_compute,Mass_exact,Momentum_exact,Energy_exact,convergence_history




            

        
        


    def compute_timestep(self):
        delta_x = np.abs(self.x[1]-self.x[0])
        self.delta_t =self.CFL*delta_x/(np.abs(self.Data.V[:,1]) + np.sqrt(np.abs(self.gamma*self.Data.V[:,2]/self.Data.V[:,0])))

    def set_pressure_bc(self):
        self.Data.V[self.p_idx,-1,:] = self.Data.V[self.p_idx,-2,:]
        self.Data.V[self.p_idx,0,:] =self.Data.V[self.p_idx,1,:]



    def iteration_step(self,return_error=True,delta_t = .000001):
        

        
        #self.compute_timestep()
        #if not self.local_timestep:
        #    self.delta_t = np.min(self.delta_t)
        
        self.delta_t = delta_t
     
        #F_left = 
        Volume =self.Data.AREA 
        self.set_boundary_multiplier()
        F_left,F_right,F_top,F_bottom = self.Upwind.GetFluxes()
       
        
        




        
        
        
        
    
        residual = F_left*self.Data.A_left + F_right*self.Data.A_right+\
            F_top*self.Data.A_bottom+F_bottom*self.Data.A_top
        
        self.residual = residual
        #if self.local_timestep:       
        #    self.Data.U[1:-1] = self.Data.U[1:-1]-(residual*np.tile(self.delta_t[1:-1],(3,1)).T/np.tile(Volume,(3,1)).T)
        #else:
        
        self.Data.U = self.Data.U-(residual*self.delta_t/Volume)
        # Update all 

        self.Data.V = self.conserved_to_primitive(self.Data.U)
        
        return residual
        if return_error:
            return np.linalg.norm(residual,axis=0)
        
        

    def update_source(self):
        #deltax = np.abs(self.x[2]-self.x[1])
        #self.S[:,1] = self.Data.V[1:-1,2]*(self.A[1:]-self.A[0:-1])/deltax
        return



    def lambda_ibar(self,shift=1):
        rho =self.Data.V[:, 0]  # Density
        u = self.Data.V[:, 1]  # Velocity
        p = self.Data.V[:, 2]  # Pressure
        a = np.sqrt(np.maximum(0,(self.gamma*(p/rho))))
        

        lambda_bar = np.abs(u)+a
        if shift==1:
            lambda_bar = (lambda_bar[2:]+lambda_bar[1:-1])/2
        else:
            lambda_bar = (lambda_bar[1:-1]+lambda_bar[0:-2])/2

        return lambda_bar


    def d2(self,shift=1):
        temp_U = self.Data.U
        
        if shift==1:
            return np.tile(self.lambda_ibar(shift=1)*self.epsilon2(shift=1),(3,1)).T*(temp_U[2:]-temp_U[1:-1])
        else:
            return np.tile(self.lambda_ibar(shift=-1)*self.epsilon2(shift=-1),(3,1)).T*(temp_U[1:-1]-temp_U[0:-2])

    def d4(self,shift=1):
        temp_U = np.zeros((self.Data.U.shape[0]+2,3))
        temp_U[1:-1] = self.Data.U
        temp_U[0] = 2*temp_U[1]-temp_U[2]
        temp_U[-1] = 2*temp_U[-2]-temp_U[-3]
        #print(self.lambda_ibar(shift=1).shape,self.epsilon4(shift=1).shape,temp_U[4:].shape)
        if shift==1:
            return np.tile(self.lambda_ibar(shift=1)*self.epsilon4(shift=1),(3,1)).T*(temp_U[4:]-3*temp_U[3:-1]+3*temp_U[2:-2]-temp_U[1:-3])
        else:
            return np.tile(self.lambda_ibar(shift=-1)*self.epsilon4(shift=-1),(3,1)).T*(temp_U[3:-1]-3*temp_U[2:-2]+3*temp_U[1:-3]-temp_U[0:-4])




    def nu(self):
        temp_p = np.zeros((self.NI+5)) # 3 ghost cells on each side
        temp_p[2:-2] = self.Data.V[:,2]
        
        
        temp_p[1] = 2*temp_p[2]-temp_p[3]
        temp_p[0] = 2*temp_p[1]-temp_p[2]
        temp_p[-2] = 2*temp_p[-3]-temp_p[-4]
        temp_p[-1] = 2*temp_p[-2]-temp_p[-3]


        pi_plus1 = temp_p[2:]
        pi = temp_p[1:-1]
        pi_minus1 = temp_p[0:-2]

        num = pi_plus1-2*pi+pi_minus1
        den =  pi_plus1+2*pi+pi_minus1

        return np.abs(num/den)

    def epsilon2(self,shift=1):
        nu = self.nu()
        epsilon = np.zeros((self.NI-1))
        if shift==1:
            for i in range(2,self.NI+2-1):
                epsilon[i-2] = self.K2*np.max([nu[i-1],nu[i],nu[i+1],nu[i+2]])
        if shift ==-1:
            for i in range(2,self.NI-1+2):
                epsilon[i-2] = self.K2*np.max([nu[i-2],nu[i-1],nu[i],nu[i+1]])
        return epsilon

    def epsilon4(self,shift=1):
        return np.maximum(0,self.K4 - self.epsilon2(shift=shift))


    def exact_isentropic(self):
        x = (self.x[1:]+self.x[0:-1])/2
        A_x = 0.2 + 0.4*(1 + np.sin(np.pi*(x - 0.5)))
        A_star = 0.2 + 0.4*(1 + np.sin(np.pi*(0 - 0.5)))
        A_Astar = A_x/A_star
        RHO,U,P = ([],[],[])
        for i,A in enumerate(A_Astar):
            if x[i]<=0:
                rho,u,p = self.exact_solution(A,subsonic=True)
            else:
                rho,u,p = self.exact_solution(A,subsonic=False)
            RHO.append(rho)
            U.append(u)
            P.append(p)
        return RHO,U,P
    def exact_solution(self,A_Astar,subsonic=True):
       
        
        def mach_from_area_ratio(A_Astar, gamma=1.4):

            def mach_eq(M):
                return (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2))**((gamma+1)/(2*(gamma-1))) - A_Astar
            if subsonic:
                Mach_initial_guess = 0.5 
            else:
                Mach_initial_guess =  2.0  # Subsonic for A/A*<1, supersonic for A/A*>1
            return scipy.optimize.fsolve(mach_eq, Mach_initial_guess)[0]
        Mach = mach_from_area_ratio(A_Astar) 
        # Temperature relation
        T = self.T0 / (1 + (self.gamma - 1) / 2 * Mach**2)
        
        # Pressure relation
        p = self.p0 * (T / self.T0) ** (self.gamma / (self.gamma - 1))
        
        # Density relation using ideal gas law
        rho = p / (self.R * T)
        velocity = Mach * np.sqrt(self.gamma * self.R * T)
        return rho,velocity,p



    
    def total_T(self,gamma,M,T0):
        def psi(gamma,M):
            return 1+((gamma-1)/2)*M**2
        return T0/psi(gamma,M)

    def total_p(self,gamma,M,P0):
        def psi(gamma,M):
            return 1+((gamma-1)/2)*M**2
        return P0/(psi(gamma,M)**(gamma/(gamma-1)))
    def total_density(self,P,R,T):
        
        return P/(R*T)
    def total_velocity(self,gamma,M,R,T):
        return M*np.sqrt(gamma*R*T)

    #def compute_mach(self):




#### TEST FUNCTIONS

    def test_variable_changes(self):
        U = np.random.rand(*self.Data.U.shape)
        V = self.primitive_to_conserved(U)[0]
        temp_U = self.conserved_to_primitive(V)
        if np.allclose(temp_U,U):
            print("Conserved to primitive is valid")
        else:
            print("ERROR: Conserved to primitive has an error")

    def test_nu(self):
        self.Data.V = np.ones_like(self.Data.V)
        if np.allclose(self.nu(),0):
            print("Nu is valid")
        else:
            print("ERROR: Nu() has an error")

            
    def test_lambda_ibar(self):
        self.Data.V = np.ones_like(self.Data.V)
        self.Data.V[:,1] = 0*self.Data.V[:,1] # u = 0
        expected_val = np.sqrt(self.gamma)*np.ones_like(self.Data.V[:,1])[1:-1]
        test_val = self.lambda_ibar(shift = 1)
        if np.allclose(expected_val,test_val):
            print("Lambda_ibar() function is valid")
        else:
            print("ERROR: Lambda_ibar() is invalid")


    def primitive_to_conserved(self,primitive):
        rho = primitive[self.rho_idx]  # Density
        u = primitive[self.u_idx]  # Velocity
        v = primitive[self.v_idx] 
        p = primitive[self.p_idx]

        # Compute total energy per unit mass (et)
        et = p / ((self.gamma - 1) * rho) + 0.5 * ((u)**2+v**2)
        ht = (self.gamma / (self.gamma - 1)) * (p / rho) + 0.5 * ((u)**2+v**2)
        
        U = np.array([rho, rho * u,rho*v, rho * et])

        F = np.column_stack((rho * u, rho * u**2 + p,rho*u*v, rho * u * ht))
        G = np.array([rho * v,rho*u*v, rho * v**2 + p, rho * v * ht])
        
        return U, F,G  # Return both conserved variables and fluxes

    def conserved_to_primitive(self,conserved):
        rho = conserved[self.rho_idx]  # Density
       
        u = conserved[self.u_idx] / rho  # Velocity
        v = conserved[self.v_idx]/rho
        #if np.any(u<=0):
        #    u = np.sign(u)*np.maximum(np.abs(u), self.epsilon)
        p = (self.gamma - 1) * (conserved[self.p_idx] - 0.5 * rho * (u**2+v**2))  # Pressure
        V = np.array([rho,u,v,p])

        return V
    
         
        
def verify_solution(v1, v2, s, n, tol=1e-10):
    """
    Verifies that:
      - dot(v1, s) == dot(v2, s)
      - dot(v1, n) == -dot(v2, n)
    within some tolerance.
    """
    left1 = np.tensordot(v1, s)
    right1 = np.tensordot(v2, s)

    left2 = np.tensordot(v1, n)
    right2 = -np.tensordot(v2, n)

    print("Check 1: v1·s == v2·s →", np.isclose(left1, right1, atol=tol))
    print("       ", left1, "==", right1)

    print("Check 2: v1·n == -v2·n →", np.isclose(left2, right2, atol=tol))
    print("       ", left2, "==", right2)
    
    return np.isclose(left1, right1, atol=tol) and np.isclose(left2, right2, atol=tol)

"""

def get_boundary_multiplier(self,Flux_up,Flux_right):
        top_bc_multiplier = np.ones_like(self.Data.F)
        right_bc_multiplier = np.ones_like(self.Data.F)
        u0 = Flux_up[:,-1,:]
        v0 = Flux_right[:,-1,:]
        nx0 = np.einsum("mj,jk->mk",np.ones((4,1)),self.Data.upward_normal[np.newaxis,self.unormal,0,:])
        ny0 = np.einsum("mj,jk->mk",np.ones((4,1)),self.Data.upward_normal[np.newaxis,self.vnormal,0,:])

        #u1 = self.Data.V[self.u_idx,-1,:]
        #v1 = self.Data.V[self.v_idx,-1,:]
        #nx1 = self.Data.upward_normal[self.unormal,-1,:]
        #ny1 = self.Data.upward_normal[self.vnormal,-1,:]

        dot0 = u0*nx0 + v0*ny0
        #dot1 = u1*nx1 + v1*ny1
        Vnorm0 = np.linalg.norm(np.array([u0,v0]),axis=0)
        Nnorm0 = np.linalg.norm(np.array([nx0,ny0]),axis=0)

        #Vnorm1 = np.linalg.norm(np.array([u1,v1]),axis=0)
        #Nnorm1 = np.linalg.norm(np.array([nx1,ny1]),axis=0)
        indeces0 = np.where(np.any(Vnorm0<.0000001))
        #indeces1 = np.where(np.any(Vnorm1<.0000001))
        Vnorm0[indeces0] = 1
        #Vnorm1[indeces1] = 1


        top_bc_multiplier[:,-1,:] = 1-np.abs(dot0)/(Vnorm0*Nnorm0) 
        right_bc_multiplier[:,-1,:] = np.abs(dot0)/(Vnorm0*Nnorm0)
        #bc_multiplier[:,-1,:] = 1-np.abs(dot1)/(Vnorm1*Nnorm1) 
        #bc_multiplier[:,:,0] = 0 
        return top_bc_multiplier,right_bc_multiplier
        """