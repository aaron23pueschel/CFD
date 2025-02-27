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

    def set_geometry(self):
        self.x = np.linspace(self.domain[0],self.domain[1],self.NI)
        self.A = np.zeros(self.NI+2)
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
        T = self.p/(self.rho*self.R)
        et = (self.R/(self.gamma-1))*T+.5*(self.u**2)
        ht = (self.gamma*self.R/(self.gamma-1))*T+.5*self.u**2
        self.F[1:-1] = np.array([self.rho*self.u,self.rho*self.u**2+self.p,self.rho*self.u*ht]).T
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
        deltax = self.x[2]-self.x[1]
        for i in range(self.NI-1):
            Volume = (self.A[i]+self.A[i+1])*deltax/2
            F_plus_1_2 = (self.F[i+2]+self.F[i+1])/2
            F_minus_1_2 = (self.F[i]+self.F[i+1])/2 

            A_plus_1_2 = (self.A[i+1])
            A_minus_1_2 = (self.A[i])
            
            self.U[i] = self.U[i]-( F_plus_1_2*A_plus_1_2-F_minus_1_2*A_minus_1_2 -self.S[i]*deltax)*self.delta_t/(Volume)

    def set_source_term(self):
        self.S = np.zeros((self.NI-1,3))
        self.S[:,1] = self.p*(self.A[1:]-self.A[0:-1])
    def update_source(self):
        self.S[:,1] = self.p*(self.A[1:]-self.A[0:-1])
    def conserved_to_primitive(self):
        self.rho = self.U[:,0]  # Density
        self.u = self.U[:,1] / self.rho  # Velocity
        self.p = (self.gamma - 1) * (self.U[:,2] - 0.5 * self.rho * self.u**2)  # Pressure
    def update_all(self):
        self.conserved_to_primitive() # Updates primitive variables
        self.set_boundary_conditions()
        self.update_source()
        self.update_conserved()






            
        
