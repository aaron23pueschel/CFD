import f90nml
import numpy as np


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
        self.delta_t = nml["inputs"]["delta_t"]
        self.p_back = nml["inputs"]["p_back"]
        self.gamma = nml["inputs"]["gamma"]
        self.K4 = nml["inputs"]["K4"]
        self.K2 = nml["inputs"]["K2"]
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
        

    def set_geometry(self):
        self.x = np.linspace(self.domain[0],self.domain[1],self.NI)
        self.A = self.Area()
        #self.A[1:-1] = self.Area()
        #self.A[0] = 2*self.A[1]-self.A[2]
        #self.A[-1] = 2*self.A[-2]-self.A[-3]
    def set_initial_conditions(self):
        self.p = np.zeros(self.NI-1) + self.p_inf # Number of cells
        self.rho = np.zeros(self.NI-1)+self.p_inf/(self.R*self.T_inf)
        self.u = np.zeros(self.NI-1)
    def set_boundary_conditions(self):
        
        self.p[0] = self.p0
        self.rho[0] = self.p0/(self.R*self.T0)
        if not self.p_back==-1:
            self.p[-1] = self.p_back


    def set_conserved_variables(self): # Set F_{i+1/2} and F_{i-1/2}
        T = self.p/(self.rho*self.R)
        et = (self.R/(self.gamma-1))*T+.5*(self.u**2)
        ht = (self.gamma*self.R/(self.gamma-1))*T+.5*self.u**2

        self.U = np.zeros((self.NI-1,3)) # Number of faces
        self.F = np.zeros((self.NI+2*self.ghost_cells-1,3))
        
        self.U = np.array([self.rho,self.rho*self.u,self.rho*et]).T
        self.F[1:-1] = np.array([self.rho*self.u,self.rho*self.u**2+self.p,self.rho*self.u*ht]).T

        self.extrapolate_conserved()

    def update_conserved(self):
 
        F3 = (self.gamma/(self.gamma-1))*self.p*self.u+self.rho*(self.u**3)/2
        self.F[1:-1] = np.array([self.rho*self.u,self.rho*self.u**2+self.p,F3]).T
        T = self.p/(self.rho*self.R)
        et = (self.R/(self.gamma-1))*T+.5*(self.u**2)
        self.U = np.array([self.rho,self.rho*self.u,self.rho*et]).T

        self.extrapolate_conserved()

    def Area(self):
        return .2+.4*(1+np.sin(np.pi*(self.x-.5)))
    def extrapolate_conserved(self):
        self.F[0] = 2*self.F[1]-self.F[2]
        self.F[-1] = 2*self.F[-2]-self.F[-3]

        #self.U[0] = 2*self.U[1]-self.U[2]
        #self.U[-1] = 2*self.U[-2]-self.U[-3]
    def extrapolate_primitive(self):
        self.V[0] = 2*self.V[1]-self.V[2]
        self.V[-1] = 2*self.V[-2]-self.V[-3]
    def set_primitive_variables(self):
        self.V = np.zeros((self.NI+2*self.ghost_cells-1,3))
        self.V[1:-1] = np.array([self.rho,self.u,self.p]).T
        self.extrapolate_primitive()

    def iteration_step(self):
        deltax = np.abs(self.x[2]-self.x[1])
        
        d_plus_half = -(self.d2(shift=1)-self.d4(shift=1))
        d_minus_half = -(self.d2(shift=-1)-self.d4(shift=-1))
        residual = np.zeros_like(self.S)
        for i in range(self.NI-1):
            Volume = (self.A[i]+self.A[i+1])*deltax/2
            F_plus_1_2 = (self.F[i+2]+self.F[i+1])/2 #+ d_plus_half[i]
            F_minus_1_2 = (self.F[i]+self.F[i+1])/2 #+d_minus_half[i]

            A_plus_1_2 = (self.A[i+1])
            A_minus_1_2 = (self.A[i])
            residual[i] = ( F_plus_1_2*A_plus_1_2-F_minus_1_2*A_minus_1_2 -self.S[i]*deltax)
            self.U[i] = self.U[i]-(( F_plus_1_2*A_plus_1_2-F_minus_1_2*A_minus_1_2 -self.S[i]*deltax)*self.delta_t/(Volume))
        self.residual = residual
    def set_source_term(self):
        self.S = np.zeros((self.NI-1,3))
        self.S[:,1] = self.p*(self.A[1:]-self.A[0:-1])
    def update_source(self):
        self.S[:,1] = self.p*(self.A[1:]-self.A[0:-1])
    def conserved_to_primitive(self):
        self.rho = self.U[:,0]  # Density
        if np.any(self.rho<0):
            np.maximum(self.rho, 0)
        self.u = self.U[:,1] / self.rho  # Velocity
        if np.any(self.u<0):
            self.u = np.maximum(self.u, 0)
        self.p = (self.gamma - 1) * (self.U[:,2] - 0.5 * self.rho * self.u**2)  # Pressure
        if np.any(self.p > self.p0):  # Check if any element is greater than p0
            self.p = np.minimum(self.p, self.p0)



    def lambda_ibar(self,shift=1):
        u = np.zeros((self.u.shape[0]+2))
        a = np.zeros((self.u.shape[0]+2))
        u[1:-1] = np.abs(self.u)
        a[1:-1] = np.sqrt(self.gamma*(self.p/self.rho))
        
        a[0] = 2*a[1]-a[2]
        a[-1] = 2*a[-2]-a[-3]

        u[0] = 2*u[1]-u[2]
        u[-1] = 2*u[-2]-u[-3]

        lambda_bar = np.abs(u)+a
        if shift==1:
            lambda_bar = (lambda_bar[2:]+lambda_bar[1:-1])/2
        else:
            lambda_bar = (lambda_bar[1:-1]+lambda_bar[0:-2])/2

        return lambda_bar


    def d2(self,shift=1):
        temp_U = np.zeros((self.U.shape[0]+2,3))
        temp_U[0] = 2*temp_U[1]-temp_U[2]
        temp_U[-1] = 2*temp_U[-2]-temp_U[-3]
        if shift==1:
            return np.tile(self.lambda_ibar(shift=1)*self.epsilon2(shift=1),(3,1)).T*(temp_U[2:]-temp_U[1:-1])
        else:
            return np.tile(self.lambda_ibar(shift=-1)*self.epsilon2(shift=-1),(3,1)).T*(temp_U[1:-1]-temp_U[0:-2])

    def d4(self,shift=1):
        temp_U = np.zeros((self.U.shape[0]+4,3))
        temp_U[1] = 2*temp_U[2]-temp_U[3]
        temp_U[0] = 2*temp_U[1]-temp_U[2]
        temp_U[-2] = 2*temp_U[-3]-temp_U[-4]
        temp_U[-1] = 2*temp_U[-2]-temp_U[-3]
        if shift==1:
            return np.tile(self.lambda_ibar(shift=1)*self.epsilon4(shift=1),(3,1)).T*(temp_U[4:]-3*temp_U[3:-1]+3*temp_U[2:-2]-temp_U[1:-3])
        else:
            return np.tile(self.lambda_ibar(shift=-1)*self.epsilon4(shift=-1),(3,1)).T*(temp_U[3:-1]-3*temp_U[2:-2]+3*temp_U[1:-3]-temp_U[0:-4])



    def nu(self):
        temp_p = np.zeros((self.p.shape[0]+6)) # 3 ghost cells on each side
        temp_p[3:-3] = self.p
        
        temp_p[2] = 2*temp_p[3]-temp_p[4]
        temp_p[1] = 2*temp_p[2]-temp_p[3]
        temp_p[0] = 2*temp_p[1]-temp_p[2]
        temp_p[-3] = 2*temp_p[-4]-temp_p[-5]
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




            
        
