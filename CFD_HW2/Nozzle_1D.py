import f90nml
import numpy as np
from cfunctions import C_FUNCTIONS as functions
import ctypes
class Nozzle:
    def __init__(self,input_file):
        try:
            nml = f90nml.read(input_file)
        except FileNotFoundError:
            print("Check namelist filename")

        # Input vars
        self.p0 = nml["inputs"]["p0"]
        self.NI = nml["inputs"]["NI"]
        self.p0 = nml["inputs"]["p0"]
        self.T0 = nml["inputs"]["T0"]
        self.Ru = nml["inputs"]["Ru"]
        self.M = nml["inputs"]["M"]
        self.R = self.Ru/self.M
        self.p_inf = nml["inputs"]["p_inf"]
        self.T_inf = nml["inputs"]["T_inf"]
        self.CFL = nml["inputs"]["CFL"]
        self.ghost_cells = nml["inputs"]["ghost_cells"]
        self.domain = nml["inputs"]["domain"]
        #self.delta_t = nml["inputs"]["delta_t"]
        self.p_back = nml["inputs"]["p_back"]
        self.gamma = nml["inputs"]["gamma"]
        self.K4 = nml["inputs"]["K4"]
        self.K2 = nml["inputs"]["K2"]
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
        self.FM = functions().FM()
        self.DfdM = functions().DfdM()
        self.newton = functions().newton()
        
    def compute_newton(self):
        P0 = self.p0
        T0 = self.T0
        gamma = ctypes.c_double(self.gamma)
        tolerance = ctypes.c_double(1e-12)

        # Compute A(x)
        if self.x is None:
            self.set_geometry()
        x = self.x
        A_x = 0.2 + 0.4*(1 + np.sin(np.pi*(x - 0.5)))
        A_star = 0.2 + 0.4*(1 + np.sin(np.pi*(0 - 0.5)))
        A_bar = A_x/A_star
        p = []
        u = []
        RHO = []
        
        for i,A_ in enumerate(A_bar):
            A_ = ctypes.c_double(A_)
            if abs(x[i])<.01:
                initial_mach = ctypes.c_double(1+x[i])
            if x[i]<0:
                initial_mach = ctypes.c_double(.3)

            elif x[i]==0:
                initial_mach = ctypes.c_double(1+10e-14)
            else:
                initial_mach = ctypes.c_double(3)
            R = self.R
            M = (self.newton(gamma,A_,initial_mach,tolerance))
            T = functions.T(gamma.value,M,T0)
            P = functions.P(gamma.value,M,P0)
            U = functions.u(gamma.value,M,R,T)
            rho = functions.rho(P,R,T)
            p.append(P)
            RHO.append(rho)
            u.append(U)
        return p,u,RHO
            

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

        # Compute total enthalpy per unit mass (ht)
        ht = (self.gamma / (self.gamma - 1)) * (p / rho) + 0.5 * u**2

        # Compute conserved variables (U)
        U = np.column_stack((rho, rho * u, rho * et))

        # Compute fluxes (F)
        F = np.column_stack((rho * u, rho * u**2 + p, rho * u * ht))

        return U, F  # Return both conserved variables and fluxes

    def conserved_to_primitive(self,conserved):
        rho = conserved[:,0]  # Density
        if np.any(rho<0):
            rho = np.maximum(rho, 0.0001)
        u = conserved[:,1] / rho  # Velocity
        if np.any(u<0):
            u = np.maximum(u, 0.0001)
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
        epsilon = .001
        self.mach= np.abs(self.V[:,1])/np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0]))
        #self.mach[0] = np.max([epsilon,2*self.mach[1]-self.mach[2]]) # Maybe dont need this line...

        T = self.total_T(self.gamma,self.mach[0],self.T0)
        p_bndry = self.total_p(self.gamma,self.mach[0],self.p0)
        rho_bndry = self.total_density(p_bndry,self.R,T)
        u_bndry = self.total_velocity(self.gamma,self.mach[0],self.R,T)
        if not self.p_back==-1:
            self.p[-1] = self.p_back
        self.V[0] = np.array([rho_bndry,u_bndry,p_bndry]).T
        self.V[-1] = np.array([self.extrapolate1(self.V)[1]])
        self.U,self.F = self.primitive_to_conserved(self.V)


    def set_conserved_variables(self): # Set F_{i+1/2} and F_{i-1/2}
        T = self.V[:,2]/(self.V[:,0]*self.R)
        et = (self.R/(self.gamma-1))*T+.5*(self.V[:,1]**2)
        ht = (self.gamma*self.R/(self.gamma-1))*T+.5*self.V[:,1]**2

        self.U = np.zeros((self.NI+1,3)) # Number of faces
        self.F = np.zeros((self.NI+1,3))
        
        #self.U[1:-1] = np.array([self.V[-1:1,0],self.V[-1:1,1]*self.V[-1:1,0],self.V[-1:1,0]*et]).T
        #self.F[1:-1] = np.array([self.V[-1:1,0]*self.V[-1:1,1],self.V[-1:1,0]*self.V[-1:1,1]**2+self.V[-1:1,3],self.V[-1:1,0]*self.V[-1:1,1]*ht]).T
        
        self.U = np.array([
        self.V[:, 0],  # Density
        self.V[:, 0] * self.V[:, 1],  # Momentum (rho * u)
        self.V[:, 0] * et ]).T

    # Set the fluxes (F) using primitive variables (density, velocity, and pressure)
        self.F = np.array([
        self.V[:, 0] * self.V[:, 1], 
        self.V[:, 0] * self.V[:, 1]**2 + self.V[:, 2],  
        self.V[:, 0] * self.V[:, 1] * ht ]).T
        

    #def update_conserved(self):
 
    #    F3 = (self.gamma/(self.gamma-1))*self.p*self.u+self.rho*(self.u**3)/2
    #    self.F = np.array([self.rho*self.u,self.rho*self.u**2+self.p,F3]).T
    #    T = self.p/(self.rho*self.R)
    #    et = (self.R/(self.gamma-1))*T+.5*(self.u**2)
    #    self.U = np.array([self.rho,self.rho*self.u,self.rho*et]).T

        

    def Area(self):
        return .2+.4*(1+np.sin(np.pi*(self.x-.5)))
    def Darea(self):
        x = (self.x[1:]+self.x[0:-1])/2
        return 0.4 * np.pi * np.cos(np.pi * (x - 0.5))
    
    def extrapolate_conserved(self):
        self.U[0] = 2*self.U[1]-self.U[2]
        self.U[-1] = 2*self.U[-2]-self.U[-3]
            

    def set_primitive_variables(self):
        self.V = np.zeros((self.NI+2*self.ghost_cells-1,3))
        self.V[0:-1] = np.array([self.rho[0:-1],self.u[0:-1],self.p[0:-1]]).T
        self.extrapolate_primitive()

    
    def compute_CFL(self):
        delta_x = np.abs(self.x[1]-self.x[0])
        self.delta_t =self.CFL*delta_x/(np.abs(self.V[:,1]) + np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0])))
    def iteration_step(self,return_error=True,local_timestep = True):
        deltax = np.abs(self.x[2]-self.x[1])
        
        d_plus_half = -(self.d2(shift=1)-self.d4(shift=1))
        d_minus_half = -(self.d2(shift=-1)-self.d4(shift=-1))
        residual = np.zeros((self.NI-1,3))
        
        self.compute_CFL()
        if not local_timestep:
            self.delta_t = np.min(self.delta_t)
        
        Volume = (self.A[0:-1]+self.A[1:])*deltax/2
        F_plus_1_2 = (self.F[2:]+self.F[1:-1])/2 + d_plus_half
        F_minus_1_2 = (self.F[0:-2]+self.F[1:-1])/2 +d_minus_half

        A_plus_1_2 = np.tile((self.A[1:]),(3,1)).T
        A_minus_1_2 = np.tile(self.A[0:-1],(3,1)).T
        
        residual = ( F_plus_1_2*A_plus_1_2-F_minus_1_2*A_minus_1_2 -self.S*deltax)
        
        
        self.U[1:-1] = self.U[1:-1]-(residual*np.tile(self.delta_t[1:-1],(3,1)).T/np.tile(Volume,(3,1)).T)
        
        # Update all 

        
        self.U[0],self.U[-1] = self.extrapolate1(self.U)
        self.V = self.conserved_to_primitive(self.U)
        self.update_source()
        if return_error:
            return np.linalg.norm(residual,axis=0)
        
    def set_source_term(self):
        deltax = np.abs(self.x[2]-self.x[1])
        self.S[:,1] = self.V[1:-1,2]*(self.A[1:]-self.A[0:-1])/deltax
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
        #temp_U[1] = 2*temp_U[2]-temp_U[3]
        temp_U[0] = 2*temp_U[1]-temp_U[2]
        #temp_U[-2] = 2*temp_U[-3]-temp_U[-4]
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
                epsilon[i-2] = np.max([nu[i-1],nu[i],nu[i+1],nu[i+2]])
        if shift ==-1:
            for i in range(2,self.NI-1+2):
                epsilon[i-2] = np.max([nu[i-2],nu[i-1],nu[i],nu[i+1]])
        return epsilon

    def epsilon4(self,shift=1):
        return np.maximum(0,self.K4 - self.epsilon2(shift=shift))




    def update_all(self):
        self.conserved_to_primitive() # Updates primitive variables
        self.update_mach()
        self.set_boundary_conditions()
        self.update_source()
        self.update_conserved()




    def display(self):
        print("p:", self.p)
        print("u:", self.u)
        print("x:", self.x)
        print("rho:", self.rho)
        print("F:", self.F)
        print("U:", self.U)
        print("V:", self.V)
        print("A:", self.A)
        print("S:", self.S)


    
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







            
        
