import f90nml
import numpy as np
import scipy
import os
import copy
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
            self.MMS_length = nml["SimulationVariables"]["MMS_len"]
            self.grd_name = nml["SimulationVariables"]["grd"]
            self.gamma = nml["SimulationVariables"]["gamma"]
            self.epsilon = nml["SimulationVariables"]["epsilon"]
            self.kappa = nml["UpwindVariables"]["kappa"]
            self.Upwind_order = nml["UpwindVariables"]["Upwind_order"] #! 0 for first order 1 for second order
            self.convergence_criteria = nml["SimulationVariables"]["convergence_criteria"]
            self.damping_scheme = nml["UpwindVariables"]["damping_scheme"]#1 ! 0 for jameson damping, 1 for van leer Upwind
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

        self.Data = DataStructures(self.NI,self.NJ,self.grd_name)
        self.Data.R = self.R
        self.Data.p0 = self.p0
        self.Upwind = Upwind(self.Data,self.kappa,self.Upwind_order,self.damping_scheme,self.flux_limiter_scheme)


    
    def set_boundary_multiplier(self):
        bc_multiplier_top = np.ones_like(self.Data.F)
        bc_multiplier_top[:,-1,:] = 0 # Top

        bc_multiplier_bottom = np.ones_like(self.Data.F)
        bc_multiplier_bottom[:,0,:] = 0 
        #bc_multiplier[:,:,0] = 0 
        
        self.Data.bc_multiplier = [bc_multiplier_top,bc_multiplier_bottom]
        
    
        
    def set_initial_conditions(self):
        
        mach = .1*np.ones((self.NI-1,self.NJ-1))
        T = self.total_T(self.gamma,mach,self.T0)
        p = self.total_p(self.gamma, mach,self.p0) # Number of cells
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,mach,self.R,T)
        
        
        self.Data.V[self.rho_idx] = rho
        self.Data.V[self.u_idx] = u*self.Data.upward_S[self.unormal]
        self.Data.V[self.v_idx] = u*self.Data.upward_S[self.vnormal]

        self.Data.V[self.p_idx] = p

        self.Data.U,_,_ = self.primitive_to_conserved(self.Data.V)

    def set_RK4_vals(self):
        self.U1 = copy.deepcopy(self)
        self.U2 = copy.deepcopy(self)
        self.U3 = copy.deepcopy(self)
        self.U4 = copy.deepcopy(self)


    

        
    def set_inflow_bcs(self):

        mach = .2*np.ones((self.NI-1))
        T = self.total_T(self.gamma,mach,self.T0)
        p = self.total_p(self.gamma, mach,self.p0) # Number of cells
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,mach,self.R,T)
        
        
        #self.Data.V[self.rho_idx,:,0] = rho
        #self.Data.V[self.u_idx,:,0] = u*self.Data.rightward_normal[self.unormal,:,0]
        #self.Data.V[self.v_idx,:,0] = u*self.Data.rightward_normal[self.vnormal,:,0]
        #self.Data.V[self.p_idx,:,0] = p


        #self.Data.U,_,_ = self.primitive_to_conserved(self.Data.V)

        self.Data.inflow = np.array([rho, u*self.Data.rightward_normal[self.unormal,:,0],\
            u*self.Data.rightward_normal[self.vnormal,:,0],p])
    
    


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
                    self.Upwind_order = 1
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
        rho =self.Data.V[self.rho_idx]  # Density
        u = self.Data.V[self.u_idx]
        v = self.Data.V[self.v_idx]  # Velocity
        p = self.Data.V[self.p_idx]  # Pressure
        a = np.sqrt(np.maximum(0,(self.gamma*(p/self.Upwind.min_func(rho)))))


        nx_psi_p1,ny_psi_p1 = self.Upwind.get_normal_directions("left")
        nx_psi_p2,ny_psi_p2 = self.Upwind.get_normal_directions("right")
        nx_psi = (nx_psi_p2+nx_psi_p1)/2
        ny_psi = (ny_psi_p2+ny_psi_p1)/2

        nx_eta_p1,ny_eta_p1 = self.Upwind.get_normal_directions("up")
        nx_eta_p2,ny_eta_p2 = self.Upwind.get_normal_directions("down")
        nx_eta = (nx_eta_p2+nx_eta_p1)/2
        ny_eta = (ny_eta_p2+ny_eta_p1)/2
        u,v = self.Data.V[self.u_idx],self.Data.V[self.v_idx]
        
        Area_LR = .5*(self.Data.A_left +self.Data.A_right)
        Area_UD = .5*(self.Data.A_top+self.Data.A_bottom)
        lambda_LR = np.abs(u*nx_psi +v*ny_psi)+a
        lambda_UD = np.abs(u*nx_eta+v*ny_eta)+a

        delta_t = self.Data.AREA/self.Upwind.min_func(lambda_LR*Area_LR+lambda_UD*Area_UD)
        self.delta_t =self.CFL*delta_t

    def set_pressure_bc(self):
        self.Data.V[self.p_idx,-1,:] = self.Data.V[self.p_idx,-2,:]
        self.Data.V[self.p_idx,0,:] =self.Data.V[self.p_idx,1,:]

   


    def iteration_step(self,alpha=1):
        

        
        self.compute_timestep()
        if not self.local_timestep:
            self.delta_t = np.min(self.delta_t)
        

        Volume =self.Data.AREA 
        self.set_boundary_multiplier()
        
        F_left,F_right,F_top,F_bottom = self.Upwind.GetFluxes("Roe")
    
    
        residual = F_left*self.Data.A_left + F_right*self.Data.A_right\
            +0*F_bottom*self.Data.A_bottom+0*F_top*self.Data.A_top -self.Data.MMS_conserved*Volume
        
        self.residual = residual

        
        self.Data.U = self.Data.U+alpha*(residual*self.delta_t/Volume)
        self.Data.V = self.conserved_to_primitive(self.Data.U)
    
        return residual
        
        
        
    def RK_iteration(self,type_ = "RK4"):
        if type_ =="Euler":
            self.residual = self.iteration_step(alpha = 1)
        elif type_=="RK4":
            self.U1.set_inflow_bcs()
            self.U2.set_inflow_bcs()
            self.U3.set_inflow_bcs()
            self.U4.set_inflow_bcs()
            R1 = self.U1.iteration_step(alpha=1/4)
            self.U2.Data.V = copy.deepcopy(self.U1.Data.V)
            R2 = self.U2.iteration_step(alpha = 1/3)
            self.U3.Data.V = copy.deepcopy(self.U2.Data.V)
            R3 = self.U3.iteration_step(alpha = .5)
            self.U4.Data.V = copy.deepcopy(self.U3.Data.V)
            R4 = self.U4.iteration_step(alpha = 1)


            self.Data = copy.deepcopy(self.U4.Data)
            self.residual = self.U4.residual
            self.U1.Data = self.U2.Data = self.U3.Data = copy.deepcopy(self.Data)
            self.U1.set_inflow_bcs()
            self.U2.set_inflow_bcs()
            self.U3.set_inflow_bcs()
            self.U4.set_inflow_bcs()
        return self.residual




    
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

    def compute_mach(self):
        pass



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
        rho = np.maximum(self.epsilon,rho)
        u = conserved[self.u_idx] / rho  # Velocity
        v = conserved[self.v_idx]/rho
        #if np.any(u<=0):
        #    u = np.sign(u)*np.maximum(np.abs(u), self.epsilon)
        p = np.maximum(0,(self.gamma - 1) * (conserved[self.p_idx] - 0.5 * rho * (u**2+v**2)))  # Pressure
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