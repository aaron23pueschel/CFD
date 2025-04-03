import f90nml
import numpy as np
import scipy
import ctypes
class Nozzle:
    def __init__(self,input_file):
        if input_file is not None:
            try:
                nml = f90nml.read(input_file)
            except FileNotFoundError:
                print("Check namelist filename")

            # Input vars
            self.p0 = nml["inputs"]["p0"]
            self.NI = nml["inputs"]["NI"]
            self.T0 = nml["inputs"]["T0"]
            self.Ru = nml["inputs"]["Ru"]
            self.M = nml["inputs"]["M"]
            self.R = self.Ru/self.M
            self.CFL = nml["inputs"]["CFL"]
            self.ghost_cells = nml["inputs"]["ghost_cells"]
            self.domain = nml["inputs"]["domain"]
            self.p_back = nml["inputs"]["p_back"]
            self.gamma = nml["inputs"]["gamma"]
            self.K4 = nml["inputs"]["K4"]
            self.K2 = nml["inputs"]["K2"]
            self.epsilon = nml["inputs"]["epsilon"]
            self.extrapolation_order = []
        # Class variables
        self.p = None
        self.u = None
        self.x = None
        self.rho = None
        self.F = None
        self.U =  None
        self.V = None
        self.A = None
        self.S = None
        self.residual = None
        self.mach = None
        self.delta_t = None

                ## C functions
        #self.FM = functions().FM()
        #self.DfdM = functions().DfdM()
        #self.newton = functions().newton()


    def set_arrays(self):
        self.U = np.zeros((self.NI+1,3)) # Number of faces
        self.F = np.zeros((self.NI+1,3))
        self.V = np.zeros((self.NI+1,3))
        self.S = np.zeros((self.NI-1,3)) # No ghost cells for source

    def primitive_to_conserved(self,primitive):
        rho = primitive[:, 0]  # Density
        u = primitive[:, 1]  # Velocity
        p = primitive[:, 2]  # Pressure

        # Compute total energy per unit mass (et)
        et = p / ((self.gamma - 1) * rho) + 0.5 * u**2
        ht = (self.gamma / (self.gamma - 1)) * (p / rho) + 0.5 * u**2
        U = np.column_stack((rho, rho * u, rho * et))
        F = np.column_stack((rho * u, rho * u**2 + p, rho * u * ht))

        return U, F  # Return both conserved variables and fluxes

    def conserved_to_primitive(self,conserved):
        rho = conserved[:,0]  # Density
       
        u = conserved[:,1] / rho  # Velocity
        #if np.any(u<=0):
        #    u = np.sign(u)*np.maximum(np.abs(u), self.epsilon)
        p = (self.gamma - 1) * (conserved[:,2] - 0.5 * rho * u**2)  # Pressure
        V = np.array([rho,u,p]).T
        return V 

    def set_geometry(self):
        self.x = np.linspace(self.domain[0],self.domain[1],self.NI)
        self.A = self.Area()
    def extrapolate1(self,a):
        temp0 = 2*a[1]-a[2]
        temp1 = 2*a[-2]-a[-3]
        return temp0,temp1
    def set_initial_conditions(self):
        tempx = (self.x[1:]+self.x[0:-1])/2
        self.mach = np.zeros((self.NI+1)) # only a left ghost node
        self.mach[1:-1] = (9/10)*tempx+1
        self.mach[0],self.mach[-1] = self.extrapolate1(self.mach)

        T = self.total_T(self.gamma,self.mach,self.T0)


        p = self.total_p(self.gamma, self.mach,self.p0) # Number of cells
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,self.mach,self.R,T)

        self.V = np.array([rho,u,p]).T
        self.U,self.F = self.primitive_to_conserved(self.V)
      
    def set_boundary_conditions(self):

        self.mach= np.abs(self.V[:,1])/np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0]))
        #self.mach[0] = np.max([epsilon,2*self.mach[1]-self.mach[2]]) # Maybe dont need this line...

        T = self.total_T(self.gamma,self.mach[0],self.T0)
        p_bndry = self.total_p(self.gamma,self.mach[0],self.p0)
        rho_bndry = self.total_density(p_bndry,self.R,T)
        u_bndry = self.total_velocity(self.gamma,self.mach[0],self.R,T)
        
        self.V[0] = np.array([rho_bndry,u_bndry,p_bndry]).T
        self.V[-1] = np.array([self.extrapolate1(self.V)[1]])
        if not self.p_back==-1:
            self.V[-1,2] = self.p_back
        self.U,self.F = self.primitive_to_conserved(self.V)


    def set_conserved_variables(self): # Set F_{i+1/2} and F_{i-1/2}
        T = self.V[:,2]/(self.V[:,0]*self.R)
        et = (self.R/(self.gamma-1))*T+.5*(self.V[:,1]**2)
        ht = (self.gamma*self.R/(self.gamma-1))*T+.5*self.V[:,1]**2

        self.U = np.zeros((self.NI+1,3)) # Number of faces
        self.F = np.zeros((self.NI+1,3))
        
       
        
        self.U = np.array([
        self.V[:, 0],  # Density
        self.V[:, 0] * self.V[:, 1],  # Momentum (rho * u)
        self.V[:, 0] * et ]).T

    # Set the fluxes (F) using primitive variables (density, velocity, and pressure)
        self.F = np.array([
        self.V[:, 0] * self.V[:, 1], 
        self.V[:, 0] * self.V[:, 1]**2 + self.V[:, 2],  
        self.V[:, 0] * self.V[:, 1] * ht ]).T
        


    def Area(self):
        return .2+.4*(1+np.sin(np.pi*(self.x-.5)))
    def Darea(self):
        x = (self.x[1:]+self.x[0:-1])/2
        return 0.4 * np.pi * np.cos(np.pi * (x - 0.5))

    def RUN_SIMULATION(self,iter_max = 500000,output_quantity = 100,convergence_criteria = 10e-12, verbose=False, 
                        compute_exact = True,local_timestep = True,jameson_damping = False,first_order = True):
        self.set_arrays()
        self.set_geometry()
        rho_exact,u_exact,p_exact = ([],[],[])
        if compute_exact:
            rho_exact,u_exact,p_exact = self.exact_isentropic()
        self.set_initial_conditions()
        self.set_boundary_conditions()

        R1 = self.iteration_step(local_timestep=local_timestep,jameson_damping = jameson_damping,first_order=first_order,return_error=True)
        self.set_boundary_conditions()
        
        convergence_history = []

        for i in range(iter_max):
            
            R = self.iteration_step(jameson_damping = jameson_damping,first_order=first_order,return_error=True)
            ##print(self.V[-1,1])
            if i%output_quantity == 0:
                convergence_history.append(R/R1)
                
                if np.max(R/R1)<convergence_criteria:
                    print("Converged at Rk/R1: "+ str(np.max(R/R1)))
                    break
                if verbose:
                    print("Iteration: "+str((i+1)),np.max(R/R1))
            self.set_boundary_conditions()
            
        p_compute = self.V[:,2]
        u_compute = self.V[:,1]
        rho_compute = self.V[:,0]

        return p_compute,u_compute,rho_compute,np.array(p_exact),np.array(u_exact),np.array(rho_exact),convergence_history



    def c_plus_minus(self,alpha_plus_minus,beta_LR,mach_LR,M_plus_minus):
        return alpha_plus_minus*(1+beta_LR)*mach_LR-beta_LR*M_plus_minus




    def f_convective(self,i_plus_half=True,first_order=False):
        shift = self.FL_FR_FUNC(i_plus_half=i_plus_half,first_order=first_order)
            

        rho_L,rho_R =shift(self.V[:,0]) # Density
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        u_L,u_R = shift(self.V[:,1])  # Velocity
        p_L,p_R = shift(self.V[:, 2])  # Pressure
        if i_plus_half and not self.p_back== -1:
            p_R[-1] = self.p_back
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        M_L,M_R = (u_L/a_L,u_R/a_R)
        
        alpha_plus = .5*(1+np.sign(M_L))
        alpha_minus = .5*(1-np.sign(M_R))
        beta_L = -np.maximum(0,1-np.floor(np.abs(M_L)))
        beta_R = -np.maximum(0,1-np.floor(np.abs(M_R)))

        M_plus = .25*(M_L+1)**2
        M_minus = -.25*(M_R-1)**2

        
        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * u_L**2
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * u_R**2

        temp_L = rho_L*a_L*self.c_plus_minus(alpha_plus,beta_L,M_L,M_plus)
        temp_R = rho_R*a_R*self.c_plus_minus(alpha_minus,beta_R,M_R,M_minus)
        FC_i_half = np.array([temp_L,u_L*temp_L,ht_L*temp_L]).T+np.array([temp_R,u_R*temp_R,ht_R*temp_R]).T

        return FC_i_half 
    def FL_FR_FUNC(self,i_plus_half = True,first_order=False,epsilon=0):
        if first_order:
            if i_plus_half:
                def shift(F,tuple_len = 1): return (F[1:-1],F[2:])
            else:
                def shift(F,tuple_len = 1): return (F[0:-2],F[1:-1])
        else:
            if i_plus_half:
                def shift(F,tuple_len = 1): 
                    F_temp = np.zeros((F.shape[0]+2))
                    F_temp[1:-1] = F
                    F_temp[0],F_temp[-1] = self.extrapolate1(F_temp)
                    
                    F = F_temp
                    
                    #return (F[2:-2],F[3:-1])
                    return F[2:-2]+.5*(F[2:-2]-F[1:-3]),F[3:-1]-.5*(F[4:]-F[3:-1])
            else:
                def shift(F,tuple_len = 1):
                    F_temp = np.zeros((F.shape[0]+2))
                    F_temp[1:-1] = F
                    F_temp[0],F_temp[-1] = self.extrapolate1(F_temp)
                    F = F_temp
                    #return (F[1:-3],F[2:-2])
                    return F[1:-3]+.5*(F[1:-3]-F[:-4]),F[2:-2]-.5*(F[3:-1]-F[2:-2])
        return shift
    def f_pressure_flux(self,i_plus_half=True,first_order=False):
        
        shift = self.FL_FR_FUNC(i_plus_half=i_plus_half,first_order=first_order)
        rho_L,rho_R =shift(self.V[:,0]) # Density

        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        p_L,p_R = shift(self.V[:, 2])  # Pressure
        if i_plus_half and not self.p_back== -1:
            p_R[-1] = self.p_back
        u_L,u_R = shift(self.V[:,1])  # Velocity

        
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        M_L,M_R = (u_L/a_L,u_R/a_R)
        
        alpha_plus = .5*(1+np.sign(M_L))
        alpha_minus = .5*(1-np.sign(M_R))
        beta_L = -np.maximum(0,1-np.floor(np.abs(M_L)))
        beta_R = -np.maximum(0,1-np.floor(np.abs(M_R)))

        M_plus = .25*(M_L+1)**2
        M_minus = -.25*(M_R-1)**2

        p_doubleBar_plus = M_plus*(-M_L+2)
        p_doubleBar_minus = M_minus*(-M_R-2)

        D_plus = alpha_plus*(1+beta_L)-beta_L*p_doubleBar_plus
        D_minus = alpha_minus*(1+beta_R)-beta_R*p_doubleBar_minus

        temp_zero = np.zeros_like(p_L)
        

        return np.array([temp_zero,D_plus*p_L,temp_zero]).T+np.array([temp_zero,D_minus*p_R,temp_zero]).T


    def compute_timestep(self):
        delta_x = np.abs(self.x[1]-self.x[0])
        self.delta_t =self.CFL*delta_x/(np.abs(self.V[:,1]) + np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0])))
    def iteration_step(self,return_error=True,local_timestep = True,jameson_damping = False,first_order = False):
        deltax = np.abs(self.x[2]-self.x[1])
        d_minus_half = 0
        d_plus_half = 0
        if jameson_damping:
            d_plus_half = -(self.d2(shift=1)-self.d4(shift=1))
            d_minus_half = -(self.d2(shift=-1)-self.d4(shift=-1))
        residual = np.zeros((self.NI-1,3))
        
        self.compute_timestep()
        if not local_timestep:
            self.delta_t = np.min(self.delta_t)
        
        Volume = (self.A[0:-1]+self.A[1:])*deltax/2
        if jameson_damping:
            F_plus_1_2 = (self.F[2:]+self.F[1:-1])/2 + d_plus_half
            F_minus_1_2 = (self.F[0:-2]+self.F[1:-1])/2 +d_minus_half
        else:
            F_plus_1_2 = self.f_convective(i_plus_half=True,first_order=first_order)+self.f_pressure_flux(i_plus_half=True,first_order=first_order)
            F_minus_1_2 = self.f_convective(i_plus_half=False,first_order=first_order)+self.f_pressure_flux(i_plus_half=False,first_order=first_order)

        A_plus_1_2 = np.tile((self.A[1:]),(3,1)).T
        A_minus_1_2 = np.tile(self.A[0:-1],(3,1)).T
        
        residual = ( F_plus_1_2*A_plus_1_2-F_minus_1_2*A_minus_1_2 -self.S*deltax)
        
        if local_timestep:       
            self.U[1:-1] = self.U[1:-1]-(residual*np.tile(self.delta_t[1:-1],(3,1)).T/np.tile(Volume,(3,1)).T)
        else:
            self.U[1:-1] = self.U[1:-1]-(residual*self.delta_t/np.tile(Volume,(3,1)).T)
        # Update all 

        
        self.U[0],self.U[-1] = self.extrapolate1(self.U)
        self.V = self.conserved_to_primitive(self.U)
        self.update_source()
        if return_error:
            return np.linalg.norm(residual,axis=0)
        
        

    def update_source(self):
        deltax = np.abs(self.x[2]-self.x[1])
        self.S[:,1] = self.V[1:-1,2]*(self.A[1:]-self.A[0:-1])/deltax



    def lambda_ibar(self,shift=1):
        rho =self.V[:, 0]  # Density
        u = self.V[:, 1]  # Velocity
        p = self.V[:, 2]  # Pressure
        a = np.sqrt(np.maximum(0,(self.gamma*(p/rho))))
        

        lambda_bar = np.abs(u)+a
        if shift==1:
            lambda_bar = (lambda_bar[2:]+lambda_bar[1:-1])/2
        else:
            lambda_bar = (lambda_bar[1:-1]+lambda_bar[0:-2])/2

        return lambda_bar


    def d2(self,shift=1):
        temp_U = self.U
        
        if shift==1:
            return np.tile(self.lambda_ibar(shift=1)*self.epsilon2(shift=1),(3,1)).T*(temp_U[2:]-temp_U[1:-1])
        else:
            return np.tile(self.lambda_ibar(shift=-1)*self.epsilon2(shift=-1),(3,1)).T*(temp_U[1:-1]-temp_U[0:-2])

    def d4(self,shift=1):
        temp_U = np.zeros((self.U.shape[0]+2,3))
        temp_U[1:-1] = self.U
        temp_U[0] = 2*temp_U[1]-temp_U[2]
        temp_U[-1] = 2*temp_U[-2]-temp_U[-3]
        #print(self.lambda_ibar(shift=1).shape,self.epsilon4(shift=1).shape,temp_U[4:].shape)
        if shift==1:
            return np.tile(self.lambda_ibar(shift=1)*self.epsilon4(shift=1),(3,1)).T*(temp_U[4:]-3*temp_U[3:-1]+3*temp_U[2:-2]-temp_U[1:-3])
        else:
            return np.tile(self.lambda_ibar(shift=-1)*self.epsilon4(shift=-1),(3,1)).T*(temp_U[3:-1]-3*temp_U[2:-2]+3*temp_U[1:-3]-temp_U[0:-4])




    def nu(self):
        temp_p = np.zeros((self.NI+5)) # 3 ghost cells on each side
        temp_p[2:-2] = self.V[:,2]
        
        
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
        self.set_arrays()
        U = np.random.rand(*self.U.shape)
        V = self.primitive_to_conserved(U)[0]
        temp_U = self.conserved_to_primitive(V)
        if np.allclose(temp_U,U):
            print("Conserved to primitive is valid")
        else:
            print("ERROR: Conserved to primitive has an error")

    def test_nu(self):
        self.V = np.ones_like(self.V)
        if np.allclose(self.nu(),0):
            print("Nu is valid")
        else:
            print("ERROR: Nu() has an error")

            
    def test_lambda_ibar(self):
        self.V = np.ones_like(self.V)
        self.V[:,1] = 0*self.V[:,1] # u = 0
        expected_val = np.sqrt(self.gamma)*np.ones_like(self.V[:,1])[1:-1]
        test_val = self.lambda_ibar(shift = 1)
        if np.allclose(expected_val,test_val):
            print("Lambda_ibar() function is valid")
        else:
            print("ERROR: Lambda_ibar() is invalid")


            
        
