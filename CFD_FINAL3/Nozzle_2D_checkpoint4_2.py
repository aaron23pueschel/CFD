import f90nml
import numpy as np
import scipy
import ctypes
import fmodpy
#library = fmodpy.fimport("upwind.f95")
import matplotlib.pyplot as plt
class Nozzle:
    def __init__(self,input_file):
        if input_file is not None:
            try:
                nml = f90nml.read(input_file)
            except FileNotFoundError:
                print("Check namelist filename")
                import os

                print(os.getcwd())
            # Input vars
            self.p0 = nml["inputs"]["p0"]
            self.NI = nml["inputs"]["NI"]
            self.NJ = nml["inputs"]["NJ"]
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
            self.kappa = nml["inputs"]["kappa"]
            self.upwind_order = nml["inputs"]["upwind_order"] #! 0 for first order 1 for second order
            self.convergence_criteria = nml["inputs"]["convergence_criteria"]
            self.damping_scheme = nml["inputs"]["damping_scheme"]#1 ! 0 for jameson damping, 1 for van leer upwind
            self.iter_max = nml["inputs"]["iter_max"]
            self.local_timestep = nml["inputs"]["local_timestep"]
            self.flux_limiter_scheme = nml["inputs"]["flux_limiter_scheme"] #0 for freeze, 1 for van leer, 2 for van albada
            self.flat_initial_mach = nml["inputs"]["flat_initial_mach"]
            self.converge_at_second_order = nml["inputs"]["converge_at_second_order"]
            self.extrapolation_order = []
            self.compute_isentropic =nml["inputs"]["compute_isentropic"]
            self.return_conserved = nml["inputs"]["return_conserved"]
            self.temp_plot = None
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


        self.rho_idx = 0
        self.u_idx = 1
        self.v_idx = 2
        self.p_idx = 3

        self.xxnormal = 0
        self.yynormal = 1
        self.unormal = 2
        self.vnormal = 3

                ## C functions
        #self.FM = functions().FM()
        #self.DfdM = functions().DfdM()
        #self.newton = functions().newton()


    def set_arrays(self):
        self.U = np.zeros((4,self.NI+1,self.NJ+1)) # Number of faces
        self.F = np.zeros((4,self.NI+1,self.NJ+1))
        self.V = np.zeros((4,self.NI+1,self.NJ+1))
        self.S = np.zeros((4,self.NI,self.NJ)) # No ghost cells for source
        self.residual = np.zeros_like(self.U)

        self.FL = np.zeros((self.V.shape[0],self.V.shape[1]-2,self.V.shape[2]-1))
        self.FR = np.zeros((self.V.shape[0],self.V.shape[1]-2,self.V.shape[2]-1))
        self.FD = np.zeros((self.V.shape[0], self.V.shape[1]-1, self.V.shape[2]-2))
        self.FU = np.zeros((self.V.shape[0], self.V.shape[1]-1, self.V.shape[2]-2))



    def Area(self):
        #return .2+.4*(1+np.sin(np.pi*(self.x-.5)))
        return np.ones((self.x.shape[0],self.x.shape[1]))
    def set_geometry(self):
        x = np.linspace(self.domain[0],self.domain[1],self.NI)
        y = np.linspace(self.domain[0],self.domain[1],self.NJ) # Remove this +1 soon
        xdomain = np.zeros((self.NI,self.NJ))
        ydomain = np.zeros((self.NI,self.NJ))
        for i,x_ in enumerate(x):
            for j,y_ in enumerate(y):
                xdomain[i,j] = x_
                ydomain[i,j] = y_ +.1*x_

        
        self.x = xdomain
        self.y = ydomain
        
        self.A = self.Area()
    def set_curved_geometry(self):
        x = np.linspace(self.domain[0], self.domain[1], self.NI)
        y = np.linspace(self.domain[0], self.domain[1], self.NJ)
        xx_, yy_ = np.meshgrid(y, x)

        # Flat top (y = 1.0 everywhere)
        top_y = self.domain[1]

        # Sloped curved bottom: more downward at right side (x = L)
        L = self.domain[1] - self.domain[0]
        bottom_y = self.domain[0] - 0.4 * (xx_ / L)**2  # or use linear: -0.1 * (xx_ / L)

        # Interpolation factor from top to bottom (eta = 0 at top, 1 at bottom)
        eta = 1.0 - (yy_ - self.domain[0]) / (self.domain[1] - self.domain[0])

        # Interpolate vertically between top and curved bottom
        yy = top_y * (1 - eta) + bottom_y * eta

        self.x = xx_
        self.y =  yy_ #+.3*xx_ -.1*(yy_+xx_**2)
        self.A = self.Area()

    #def set_curved_geometry(self):
    #    x = np.linspace(self.domain[0],self.domain[1],self.NI)
    #    y = np.linspace(self.domain[0],self.domain[1],self.NJ)
    #    xx_,yy_ = np.meshgrid(y,x)
    #    xx = (xx_)
    #    yy = yy_ +.1*xx_ -.1*(yy_+xx**2)


    #    self.x = xx
    #    self.y = yy
    #    self.A = self.Area()
    def primitive_to_conserved(self,primitive):
        rho = primitive[0]  # Density
        u = primitive[1]  # Velocity
        v = primitive[2]
        p = primitive[3]  # Pressure
        
  
        et = p / ((self.gamma - 1) * rho) + 0.5 * ((u)**2+v**2)

        U = np.array([rho, rho * u,rho*v, rho * et])
       

        return U
    def compute_timestep(self):
        delta_x = np.abs(self.x[1:,1:]-self.x[0:-1,1:])
        

        
        self.delta_t = self.CFL*np.ones_like(np.abs(self.V[1,1:-1,1:-1]))
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

    
    def extrapolate1(self,a):
        temp0 = 2*a[1]-a[2]
        temp1 = 2*a[-2]-a[-3]
        return temp0,temp1
    def set_airfoil_bcs(self,num_vals = 16):
        #self.V[:,0,0:num_vals] = (self.V[:,1,self.NJ-num_vals+1:])[::-1]
        #self.V[1:3,0,0:num_vals] = -self.V[1:3,0,0:num_vals] 
        #self.V[:,0,self.NJ-num_vals+1:]= (self.V[:,1,0:num_vals])[::-1]
        #self.V[1:3,0,self.NJ-num_vals+1:] = - self.V[1:3,0,self.NJ-num_vals+1:]
        return
    def set_foil_inflow(self,boundary_mach = .01):
        #self.mach= np.abs(self.V[:,1])/np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0]))
        #self.mach[0] = np.max([epsilon,2*self.mach[1]-self.mach[2]]) # Maybe dont need this line...

        #index = self.NJ//5
        #self.mach[0,(self.NJ//2-index):(self.NJ//2+index)] = boundary_mach*np.ones_like(self.mach[0,(self.NJ//2-index):(self.NJ//2+index)])
        #T = self.total_T(self.gamma,self.mach[0,(self.NJ//2-index):(self.NJ//2+index)],self.T0)
        #p_bndry = self.total_p(self.gamma,self.mach[0,(self.NJ//2-index):(self.NJ//2+index)],self.p0)
        #rho_bndry = self.total_density(p_bndry,self.R,T)
        #u_bndry = 0*self.total_velocity(self.gamma,self.mach[0,(self.NJ//2-index):(self.NJ//2+index)],self.R,T)
        #v_bndry = -self.total_velocity(self.gamma,self.mach[0,(self.NJ//2-index):(self.NJ//2+index)],self.R,T)
        
        #self.V[:,-1,(self.NJ//2-index):(self.NJ//2+index)] = np.array([rho_bndry,u_bndry,v_bndry,p_bndry])
        #self.set_airfoil_bcs()
        #self.V[:,:,0] = self.V[:,:,1]
        #self.V[:,:,-1] = self.V[:,:,-2]
        # Apply inlet to entire top boundary (j-direction fully)
        mach = self.mach[0,:]
        mach = boundary_mach * np.ones_like(mach)

        T = self.total_T(self.gamma, mach, self.T0)
        p_bndry = self.total_p(self.gamma, mach, self.p0)
        rho_bndry = self.total_density(p_bndry, self.R, T)

        u_bndry =  self.total_velocity(self.gamma, mach, self.R, T)
        v_bndry = 0*self.total_velocity(self.gamma, mach, self.R, T)

        self.V[:, -1, :] = np.array([rho_bndry, u_bndry, v_bndry, p_bndry])


    def set_ramp_inflow(self,boundary_mach = .01):

        mach = self.mach[0,:]
        mach = boundary_mach * np.ones_like(mach)

        T = self.total_T(self.gamma, mach, self.T0)
        p_bndry = self.total_p(self.gamma, mach, self.p0)
        rho_bndry = self.total_density(p_bndry, self.R, T)

        u_bndry =  self.total_velocity(self.gamma, mach, self.R, T)
        v_bndry = 0*self.total_velocity(self.gamma, mach, self.R, T)

        self.V[:, -1, :] = np.array([rho_bndry, u_bndry, v_bndry, p_bndry])
        


        mach = self.mach[:-1,0]
        mach = boundary_mach * np.ones_like(mach)

        T = self.total_T(self.gamma, mach, self.T0)
        p_bndry = self.total_p(self.gamma, mach, self.p0)
        rho_bndry = self.total_density(p_bndry, self.R, T)

        u_bndry =  self.total_velocity(self.gamma, mach, self.R, T)
        v_bndry = 0*self.total_velocity(self.gamma, mach, self.R, T)

        #self.V[:, :-1, 0] = np.array([rho_bndry, u_bndry, v_bndry, p_bndry])


    def set_initial_foil_conditions(self,boundary_mach = .01):
        self.mach = boundary_mach*np.ones((self.NI+1,self.NJ+1)) # only a left ghost node
        #self.mach = #np.linspace(.3,.5,self.NI+1)[:,np.newaxis]@np.ones((1,self.NJ+1))
        T = self.total_T(self.gamma,self.mach,self.T0)


        p = self.total_p(self.gamma, self.mach,self.p0) # Number of cells
        nx,ny =self.get_normal_directions("down")
        rho = self.total_density(p,self.R,T)
        u = 0*self.total_velocity(self.gamma,self.mach,self.R,T)[1:-1,1:-1]
        v = 0*self.total_velocity(self.gamma,self.mach,self.R,T)[1:-1,1:-1]
        
        u_full = np.zeros((u.shape[0]+2, u.shape[1]+2))
        v_full = np.zeros_like(u_full)

        # Insert interior
        u_full[1:-1, 1:-1] = u
        v_full[1:-1, 1:-1] = v

        # --- Extrapolate in all 4 directions ---

        # Top & bottom rows
        u_full[0, 1:-1]   = 2*u_full[1, 1:-1]   - u_full[2, 1:-1]    # bottom
        u_full[-1, 1:-1]  = 2*u_full[-2, 1:-1]  - u_full[-3, 1:-1]   # top

        v_full[0, 1:-1]   = 2*v_full[1, 1:-1]   - v_full[2, 1:-1]
        v_full[-1, 1:-1]  = 2*v_full[-2, 1:-1]  - v_full[-3, 1:-1]

        # Left & right columns
        u_full[1:-1, 0]   = 2*u_full[1:-1, 1]   - u_full[1:-1, 2]    # left
        u_full[1:-1, -1]  = 2*u_full[1:-1, -2]  - u_full[1:-1, -3]   # right

        v_full[1:-1, 0]   = 2*v_full[1:-1, 1]   - v_full[1:-1, 2]
        v_full[1:-1, -1]  = 2*v_full[1:-1, -2]  - v_full[1:-1, -3]

        # --- Corner extrapolations (optional, if needed) ---
        u_full[0, 0]     = 2*u_full[1, 1]     - u_full[2, 2]
        u_full[0, -1]    = 2*u_full[1, -2]    - u_full[2, -3]
        u_full[-1, 0]    = 2*u_full[-2, 1]    - u_full[-3, 2]
        u_full[-1, -1]   = 2*u_full[-2, -2]   - u_full[-3, -3]

        v_full[0, 0]     = 2*v_full[1, 1]     - v_full[2, 2]
        v_full[0, -1]    = 2*v_full[1, -2]    - v_full[2, -3]
        v_full[-1, 0]    = 2*v_full[-2, 1]    - v_full[-3, 2]
        v_full[-1, -1]   = 2*v_full[-2, -2]   - v_full[-3, -3]






        v = v_full
        u = u_full
        self.V = np.array([rho,u,v,p])
        self.U = self.primitive_to_conserved(self.V)








    def set_inflow_conditions(self,direction = "down",boundary_mach = .01):
        

        if direction=="down" or direction=="up":
            mach = self.mach[0,:]
        else:
            mach = self.mach[:,0]
        mach = boundary_mach * np.ones_like(mach)

        T = self.total_T(self.gamma, mach, self.T0)
        p_bndry = self.total_p(self.gamma, mach, self.p0)
        rho_bndry = self.total_density(p_bndry, self.R, T)
        
        if direction=="up":
            u_bndry =  0*self.total_velocity(self.gamma, mach, self.R, T)
            v_bndry = self.total_velocity(self.gamma, mach, self.R, T)
            self.V[:, 0, :] = np.array([rho_bndry, u_bndry, v_bndry, p_bndry])
            self.V[:,-1,:] = self.V[:,-2,:]
        elif direction=="down":
            u_bndry =  0*self.total_velocity(self.gamma, mach, self.R, T)
            v_bndry = -self.total_velocity(self.gamma, mach, self.R, T)
            self.V[:, -1, :] = np.array([rho_bndry, u_bndry, v_bndry, p_bndry])
            self.V[:,0,:] = self.V[:,1,:]
        elif direction=="right":
            u_bndry =  self.total_velocity(self.gamma, mach, self.R, T)
            v_bndry = 0*self.total_velocity(self.gamma, mach, self.R, T)
            self.V[:, :, 0] = np.array([rho_bndry, u_bndry, v_bndry, p_bndry])
            self.V[:, :, -1] = self.V[:, :, -2]
        elif direction=="left":
            u_bndry =  -self.total_velocity(self.gamma, mach, self.R, T)
            v_bndry = 0*self.total_velocity(self.gamma, mach, self.R, T)
            self.V[:, :, -1] = np.array([rho_bndry, u_bndry, v_bndry, p_bndry])
            self.V[:, :, 0] = self.V[:, :, 1]
        
    def set_initial_conditions2(self,direction="down",boundary_mach=.01):
        self.mach = boundary_mach*np.ones((self.NI+1,self.NJ+1)) # only a left ghost node
        #self.mach = #np.linspace(.3,.5,self.NI+1)[:,np.newaxis]@np.ones((1,self.NJ+1))
        T = self.total_T(self.gamma,self.mach,self.T0)


        p = self.total_p(self.gamma, self.mach,self.p0) # Number of cells
        nx,ny =self.get_normal_directions(direction,UPWIND = False)
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,self.mach,self.R,T)[1:-1,1:-1]
        v = 0*self.total_velocity(self.gamma,self.mach,self.R,T)[1:-1,1:-1]
        
        u_full = np.zeros((u.shape[0]+2, u.shape[1]+2))
        v_full = np.zeros_like(u_full)

        # Insert interior
        u_full[1:-1, 1:-1] = u
        v_full[1:-1, 1:-1] = v

        # --- Extrapolate in all 4 directions ---

        # Top & bottom rows
        u_full[0, 1:-1]   = 2*u_full[1, 1:-1]   - u_full[2, 1:-1]    # bottom
        u_full[-1, 1:-1]  = 2*u_full[-2, 1:-1]  - u_full[-3, 1:-1]   # top

        v_full[0, 1:-1]   = 2*v_full[1, 1:-1]   - v_full[2, 1:-1]
        v_full[-1, 1:-1]  = 2*v_full[-2, 1:-1]  - v_full[-3, 1:-1]

        # Left & right columns
        u_full[1:-1, 0]   = 2*u_full[1:-1, 1]   - u_full[1:-1, 2]    # left
        u_full[1:-1, -1]  = 2*u_full[1:-1, -2]  - u_full[1:-1, -3]   # right

        v_full[1:-1, 0]   = 2*v_full[1:-1, 1]   - v_full[1:-1, 2]
        v_full[1:-1, -1]  = 2*v_full[1:-1, -2]  - v_full[1:-1, -3]

        # --- Corner extrapolations (optional, if needed) ---
        u_full[0, 0]     = 2*u_full[1, 1]     - u_full[2, 2]
        u_full[0, -1]    = 2*u_full[1, -2]    - u_full[2, -3]
        u_full[-1, 0]    = 2*u_full[-2, 1]    - u_full[-3, 2]
        u_full[-1, -1]   = 2*u_full[-2, -2]   - u_full[-3, -3]

        v_full[0, 0]     = 2*v_full[1, 1]     - v_full[2, 2]
        v_full[0, -1]    = 2*v_full[1, -2]    - v_full[2, -3]
        v_full[-1, 0]    = 2*v_full[-2, 1]    - v_full[-3, 2]
        v_full[-1, -1]   = 2*v_full[-2, -2]   - v_full[-3, -3]






        v = v_full
        u = u_full
        self.V = np.array([rho,u,v,p])
        self.U = self.primitive_to_conserved(self.V)
 

    


    

    def set_initial_conditions(self,initial_mach = .01):
        self.mach = initial_mach*np.ones((self.NI+1,self.NJ+1)) # only a left ghost node
        #self.mach = #np.linspace(.3,.5,self.NI+1)[:,np.newaxis]@np.ones((1,self.NJ+1))
        T = self.total_T(self.gamma,self.mach,self.T0)


        p = self.total_p(self.gamma, self.mach,self.p0) # Number of cells
        nx,ny = 1,0 #self.get_normal_directions("right")
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,self.mach,self.R,T)[1:-1,1:-1]
        v = 0*ny*self.total_velocity(self.gamma,self.mach,self.R,T)[1:-1,1:-1]
        
        u_full = np.zeros((u.shape[0]+2, u.shape[1]+2))
        v_full = np.zeros_like(u_full)

        # Insert interior
        u_full[1:-1, 1:-1] = u
        v_full[1:-1, 1:-1] = v

        # --- Extrapolate in all 4 directions ---

        # Top & bottom rows
        u_full[0, 1:-1]   = 2*u_full[1, 1:-1]   - u_full[2, 1:-1]    # bottom
        u_full[-1, 1:-1]  = 2*u_full[-2, 1:-1]  - u_full[-3, 1:-1]   # top

        v_full[0, 1:-1]   = 2*v_full[1, 1:-1]   - v_full[2, 1:-1]
        v_full[-1, 1:-1]  = 2*v_full[-2, 1:-1]  - v_full[-3, 1:-1]

        # Left & right columns
        u_full[1:-1, 0]   = 2*u_full[1:-1, 1]   - u_full[1:-1, 2]    # left
        u_full[1:-1, -1]  = 2*u_full[1:-1, -2]  - u_full[1:-1, -3]   # right

        v_full[1:-1, 0]   = 2*v_full[1:-1, 1]   - v_full[1:-1, 2]
        v_full[1:-1, -1]  = 2*v_full[1:-1, -2]  - v_full[1:-1, -3]

        # --- Corner extrapolations (optional, if needed) ---
        u_full[0, 0]     = 2*u_full[1, 1]     - u_full[2, 2]
        u_full[0, -1]    = 2*u_full[1, -2]    - u_full[2, -3]
        u_full[-1, 0]    = 2*u_full[-2, 1]    - u_full[-3, 2]
        u_full[-1, -1]   = 2*u_full[-2, -2]   - u_full[-3, -3]

        v_full[0, 0]     = 2*v_full[1, 1]     - v_full[2, 2]
        v_full[0, -1]    = 2*v_full[1, -2]    - v_full[2, -3]
        v_full[-1, 0]    = 2*v_full[-2, 1]    - v_full[-3, 2]
        v_full[-1, -1]   = 2*v_full[-2, -2]   - v_full[-3, -3]






        v = v_full
        u = u_full
        self.V = np.array([rho,u,v,p])
        self.U = self.primitive_to_conserved(self.V)
    def compute_doubleBar_values(self,compute_deltas = False):
        shift = self.FLUXL_FLUXR_FUNC(direction)

        rho_L,rho_R =shift(self.V[:,0]) # Density
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        u_L,u_R = shift(self.V[:,1],is_velocity = True)  # Velocity
        p_L,p_R = shift(self.V[:, 2],is_velocity = True)  # Pressure
        
        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * u_L**2
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * u_R**2

        R = np.sqrt(np.maximum(0,rho_R/self.min_func(rho_L)))
        p_double_bar = np.sqrt(np.maximum(self.epsilon,p_L*p_R))
        rho_double_bar = R*rho_L
        u_double_bar = (R*u_R+u_L)/(R+1)
        ht_double_bar = (R*ht_R+ht_L)/(R+1)
        if not compute_deltas:
            return (p_double_bar,rho_double_bar,u_double_bar,ht_double_bar)
        return (p_double_bar,rho_double_bar,u_double_bar,ht_double_bar),(p_R-p_L,rho_R-rho_L,u_R-u_L)
    
    
    def set_ramp_BC(self):
        
        def generate_ramp_mesh(x_flat=1.0, x_ramp=6.0, height=1.0, angle_deg=15, 
                        nx=self.NJ, ny=self.NI):
            angle_rad = np.radians(angle_deg)

            # x-coordinate lines (structured)
            x = np.linspace(0, x_ramp, nx)
            
            # Create y-boundaries for each x based on flat + ramp
            y_bottom = np.zeros_like(x)
            y_top = np.zeros_like(x)

            for i, xi in enumerate(x):
                if xi <= x_flat:
                    y_bottom[i] = 0.0
                else:
                    y_bottom[i] = np.tan(angle_rad) * (xi - x_flat)

                y_top[i] = y_bottom[i] + height  # domain height constant

            # Now create structured mesh in y-direction between y_bottom and y_top
            Y = np.zeros((ny, nx))
            X = np.zeros((ny, nx))

            for i in range(nx):
                yi = np.linspace(y_bottom[i], y_top[i], ny)
                Y[:, i] = yi
                X[:, i] = x[i]
            return X,Y
        X, Y = generate_ramp_mesh()
        self.x = X
        self.y = Y

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def set_nozzle_BC(self):
        # Parameters
        length = 1.0      # nozzle length in x-direction
        height_in = 1.0   # inlet height
        height_throat = 0.3  # throat height (minimum)
        height_out = 1.0  # outlet height
          # number of y points

        # x coordinates
        x = np.linspace(0, length, self.NJ)


        theta = np.linspace(0, np.pi, self.NJ)
        top_wall = height_throat + (height_out - height_throat) * (1 - np.cos(theta)) / 2

        bottom_wall = -top_wall  # symmetric nozzle

        X, Y = np.meshgrid(x, np.linspace(0, 1, self.NI))

        # Stretch Y between bottom and top walls
        for i in range(self.NJ):
            Y[:, i] = bottom_wall[i] + Y[:, i] * (top_wall[i] - bottom_wall[i])
        self.x = X
        self.y = Y
    def set_normal_bcs(self):
        G = self.V
        #T = self.V[self.p_idx,1,:]/(self.V[self.rho_idx,1,:]*self.R)
        #rho0 = self.total_density(self.p0,self.R,T)
        
        G[self.rho_idx,0,:]= G[self.rho_idx,1,:] #- G[self.rho_idx,2,:]

        #T = self.V[self.p_idx,-2,:]/(self.V[self.rho_idx,-2,:]*self.R)
        #rho0 = self.total_density(self.p0,self.R,T)
        
        G[self.rho_idx,-1,:]= G[self.rho_idx,-2,:] #- G[self.rho_idx,-3,:]
        


        G[self.p_idx,0,:]= self.V[self.p_idx,1,:]
        G[self.p_idx,-1,:]= self.V[self.p_idx,-2,:]
        G[self.rho_idx,0,:]= self.V[self.rho_idx,1,:]
        G[self.rho_idx,-1,:]= self.V[self.rho_idx,-2,:]

        


        top,bottom = self.get_boundary_conditions_NORMALS()
        

        # TOP
        # TOP
        
        G[self.u_idx,-1,1:-1] = top[0]
        G[self.v_idx,-1,1:-1] = top[1]


        G[self.u_idx,0,1:-1] = bottom[0]
        G[self.v_idx,0,1:-1] = bottom[1]



        self.V = G
    def set_normal_bcs_FOIL(self,num_vals = 16):
        G = self.V
        #T = self.V[self.p_idx,1,:]/(self.V[self.rho_idx,1,:]*self.R)
        #rho0 = self.total_density(self.p0,self.R,T)
        
       # G[self.rho_idx,0,:]= G[self.rho_idx,1,:] #- G[self.rho_idx,2,:]

        #T = self.V[self.p_idx,-2,:]/(self.V[self.rho_idx,-2,:]*self.R)
        #rho0 = self.total_density(self.p0,self.R,T)
        
        #G[self.rho_idx,-1,:]= G[self.rho_idx,-2,:] #- G[self.rho_idx,-3,:]
        


        G[self.p_idx,0,num_vals+1:self.NJ-num_vals]= self.V[self.p_idx,1,num_vals+1:self.NJ-num_vals]
        G[self.rho_idx,0,num_vals+1:self.NJ-num_vals]= self.V[self.rho_idx,1,num_vals+1:self.NJ-num_vals]
       

        


        top,bottom = self.get_boundary_conditions_NORMALS()


        G[self.u_idx,0,num_vals+1:self.NJ-num_vals] = (bottom[0][num_vals+1:self.NJ-num_vals])
        G[self.v_idx,0,num_vals+1:self.NJ-num_vals] = (bottom[1][num_vals+1:self.NJ-num_vals])


        self.V = G

    def get_boundary_conditions_ramp_NORMALS(self,num_top = 20):
        
        
        v2x = self.V[1,-2,(num_top+1):-1]
        v2y = self.V[2,-2,(num_top+1):-1]

        u_top,v_top = self.compute_slip_walls(self.upward_S[self.unormal,-1,num_top:],\
                                                    self.upward_S[self.vnormal,-1,num_top:],self.upward_normal[self.unormal,-1,num_top:],\
                                                        self.upward_normal[self.vnormal,-1,num_top:],v2x,v2y)

        top = [u_top,v_top]
        top = [u_top,v_top]

        v2x = self.V[1,1,1:-1]
        v2y = self.V[2,1,1:-1]
        u_top,v_top = self.compute_slip_walls(self.downward_S[self.unormal,0,:],\
                                            self.downward_S[self.vnormal,0,:],self.downward_normal[self.unormal,0,:],\
                                                self.downward_normal[self.vnormal,0,:],v2x,v2y)
        bottom = [u_top,v_top]


        # bottom nook (inflow)

        v2x = self.V[1,1:-1,1]
        v2y = self.V[2,1:-1,1]
        u_top,v_top = self.compute_slip_walls(self.leftward_S[self.unormal,:,0],\
                                            self.leftward_S[self.vnormal,:,0],self.leftward_normal[self.unormal,:,0],\
                                                self.leftward_normal[self.vnormal,:,0],v2x,v2y)
        tiny_nook = [u_top,v_top]
                

        return top,bottom,tiny_nook
    def set_ramp_bcs(self,num_top=20):
        T = self.V[self.p_idx,:,1]/(self.V[self.rho_idx,:,1]*self.R)
        rho0 = self.total_density(self.p0,self.R,T)
        G = self.V
        #G[self.rho_idx,:,0]= rho0

        T = self.V[self.p_idx,:,-2]/(self.V[self.rho_idx,:,-2]*self.R)
        rho0 = self.total_density(self.p0,self.R,T)
        G = self.V
        #G[self.rho_idx,:,-1]= rho0

        


        G[self.p_idx,-1,(num_top+1):-1] = G[self.p_idx,-2,(num_top+1):-1] 
        G[self.rho_idx,-1,(num_top+1):-1] = G[self.rho_idx,-2,(num_top+1):-1] 


        G[self.p_idx,0,:] = G[self.p_idx,1,:] 
        G[self.rho_idx,0,:] = G[self.rho_idx,1,:] 

        G[self.p_idx,:,0] = G[self.p_idx,:,1]
        G[self.rho_idx,:,0] = G[self.rho_idx,:,1]


        top,bottom,nook = self.get_boundary_conditions_ramp_NORMALS()
        

        # TOP
        # TOP
        
        G[1,-1,(num_top+1):-1] = top[0] 
        G[2,-1,(num_top+1):-1] = top[1] 

        G[self.u_idx,1:-1,0] = nook[0]
        G[self.v_idx,1:-1,0] = nook[1] 

        G[self.u_idx,0,1:-1] = bottom[0] 
        G[self.v_idx,0,1:-1] = bottom[1]

        G[self.u_idx,0,0] = (G[self.u_idx,0,1] + G[self.u_idx,1,0])/2
        G[self.v_idx,0,0] = (G[self.v_idx,0,1] + G[self.v_idx,1,0])/2
        



        self.V = G
    def set_normal_bcs_RAMP(self):







        G = self.V


        G[self.p_idx,0,:]= self.V[self.p_idx,1,:]
        G[self.rho_idx,0,:]= self.V[self.rho_idx,1,:]
       

        


        top,bottom = self.get_boundary_conditions_NORMALS()


        G[self.u_idx,0,:] = (bottom[0])
        G[self.v_idx,0,:] = (bottom[1])


        self.V = G
   
    def compute_all_areas(self,visualize=False):
        def triangle_area_2d(a, b, c):
            u = np.array(b) - np.array(a)
            v = np.array(c) - np.array(a)
            return 0.5 * abs(u[0]*v[1] - u[1]*v[0])
        xx =self.x
        yy = self.y
        p1 = [xx[0:-1,0:-1],yy[0:-1,0:-1]]
        p2 = [xx[1:,0:-1],yy[1:,0:-1]] 
        p3 = [xx[0:-1,1:],yy[0:-1,1:]] 
        p4 = [xx[1:,1:],yy[1:,1:]]
        areas = triangle_area_2d(p1,p2,p3 ) + triangle_area_2d(p1, p3, p4)

        if visualize:
            midpoint_tempx = (xx[1:,:]+xx[0:-1,:])/2
            self.midpointxx = (midpoint_tempx[:,1:]+midpoint_tempx[:,0:-1])/2
            midpoint_tempy = (yy[1:,:]+yy[0:-1,:])/2
            self.midpointyy = (midpoint_tempy[:,1:]+midpoint_tempy[:,0:-1])/2
            plt.scatter(self.midpointxx.flatten(),self.midpointyy.flatten(),c=areas.flatten())
            plt.colorbar()
            plt.show()
        self.AREA = np.einsum("ij,jkl->ikl",np.ones((4,1)),areas[np.newaxis,:,:])  
        return self.midpointxx,self.midpointyy
    
    def get_normal_directions(self,direction,UPWIND = False):

        if direction=="right":
            nx,ny = self.rightward_normal[self.unormal],self.rightward_normal[self.vnormal] 
        elif direction=="left":
           
            nx,ny = self.leftward_normal[self.unormal],self.leftward_normal[self.vnormal]
        elif direction=="down":
            nx,ny = self.downward_normal[self.unormal],self.downward_normal[self.vnormal]
        elif direction == "up":
            nx,ny = self.upward_normal[self.unormal],self.upward_normal[self.vnormal]
        return nx,ny
    
    
    
    
    

    def set_normals(self):
        xx = self.x
        yy = self.y

        A_top_bottom = np.sqrt((xx[:,1:]-xx[:,0:-1])**2+(yy[:,1:]-yy[:,0:-1])**2)
        xx_top_bottom = (xx[:,1:]+xx[:,0:-1])/2
        yy_top_bottom = (yy[:,1:]+yy[:,0:-1])/2
        nxi_top = -(yy[:,1:]-yy[:,0:-1])
        nyi_top = xx[:,1:]-xx[:,0:-1]
        
    
       
        A_left_right = np.sqrt((xx[1:,:]-xx[0:-1,:])**2+(yy[1:,:]-yy[0:-1,:])**2)
        
        self.A_left = np.einsum("ij,jkl->ikl",np.ones((4,1)),(A_left_right[np.newaxis,:,0:-1]))
        self.A_right = np.einsum("ij,jkl->ikl",np.ones((4,1)),(A_left_right[np.newaxis,:,1:]))
        self.A_top = np.einsum("ij,jkl->ikl",np.ones((4,1)),(A_top_bottom[np.newaxis,1:,:]))
        self.A_bottom = np.einsum("ij,jkl->ikl",np.ones((4,1)),(A_top_bottom[np.newaxis,0:-1,:]))

        self.A_LR = np.einsum("ij,jkl->ikl",np.ones((4,1)),(A_left_right[np.newaxis,:,:]))
        self.A_UD = np.einsum("ij,jkl->ikl",np.ones((4,1)),(A_top_bottom[np.newaxis,:,:]))

        xx_left_right = ((xx[1:,:]+xx[0:-1,:])/2)
        yy_left_right = ((yy[1:,:]+yy[0:-1,:])/2)
        nxi_left_right = -(yy[1:,:]-yy[0:-1,:])
        nyi_left_right = (xx[1:,:]-xx[0:-1,:])


        downward_normal = np.array([xx_top_bottom,yy_top_bottom,-(nxi_top/A_top_bottom),-(nyi_top/A_top_bottom)])
        upward_normal = np.array([xx_top_bottom,yy_top_bottom,(nxi_top/A_top_bottom),(nyi_top/A_top_bottom)])
        leftward_normal = np.array([xx_left_right,yy_left_right,(nxi_left_right/A_left_right),(nyi_left_right/A_left_right)])
        rightward_normal = np.array([xx_left_right,yy_left_right,-(nxi_left_right/A_left_right),-(nyi_left_right/A_left_right)])


        downward_S = np.array([xx_top_bottom,yy_top_bottom,-downward_normal[3],downward_normal[2]])
        upward_S = np.array([xx_top_bottom,yy_top_bottom,upward_normal[3],-upward_normal[2]])
        leftward_S = np.array([xx_left_right,yy_left_right,leftward_normal[3],-leftward_normal[2]])
        rightward_S = np.array([xx_left_right,yy_left_right,-rightward_normal[3],rightward_normal[2]])

        self.downward_normal = downward_normal
        self.upward_normal = upward_normal
        self.leftward_normal = leftward_normal
        self.rightward_normal = rightward_normal



        nx = rightward_normal #FOR NOW
        ny = upward_normal # FOR NOW
        self.downward_S = downward_S
        self.upward_S = upward_S
        self.leftward_S = leftward_S
        self.rightward_S = rightward_S

        self.A_top_bottom = A_top_bottom
        self.A_left_right = A_left_right
    
    
    
    
    
    
    
    
    
    
    
    
    def set_top_boundary_conditions(self,boundary_mach = .1):

        #self.mach= np.abs(self.V[:,1])/np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0]))
        #self.mach[0] = np.max([epsilon,2*self.mach[1]-self.mach[2]]) # Maybe dont need this line...
        self.mach[0,:] = boundary_mach*np.ones_like(self.mach[0,:])
        T = self.total_T(self.gamma,self.mach[0,:],self.T0)
        p_bndry = self.total_p(self.gamma,self.mach[0,:],self.p0)
        rho_bndry = self.total_density(p_bndry,self.R,T)
        u_bndry = 0*self.total_velocity(self.gamma,self.mach[0,:],self.R,T)
        v_bndry = -self.total_velocity(self.gamma,self.mach[0,:],self.R,T)
        
        self.V[:,-1,:] = np.array([rho_bndry,u_bndry,v_bndry,p_bndry])
        self.V[:,0,:] =self.V[:,1,:] #np.array([self.extrapolate1(self.V)[1]])
        
        #self.U,self.F = self.primitive_to_conserved(self.V)
    
    
    
    
    
    
    
    def set_boundary_conditions(self,boundary_mach = .1):

        #self.mach= np.abs(self.V[:,1])/np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0]))
        #self.mach[0] = np.max([epsilon,2*self.mach[1]-self.mach[2]]) # Maybe dont need this line...
        self.mach[:,0] = boundary_mach*np.ones_like(self.mach[:,0])
        T = self.total_T(self.gamma,self.mach[:,0],self.T0)
        p_bndry = self.total_p(self.gamma,self.mach[:,0],self.p0)
        rho_bndry = self.total_density(p_bndry,self.R,T)
        u_bndry = self.total_velocity(self.gamma,self.mach[:,0],self.R,T)
        v_bndry = 0*self.total_velocity(self.gamma,self.mach[:,0],self.R,T)
        
        self.V[:,:,0] = np.array([rho_bndry,u_bndry,v_bndry,p_bndry])
        self.V[:,:,-1] =self.V[:,:,-2] #np.array([self.extrapolate1(self.V)[1]])
        
        #self.U,self.F = self.primitive_to_conserved(self.V)


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
        #return np.ones_like(self.x)
        return np.ones_like(self.x)
    def Darea(self):
        x = (self.x[1:]+self.x[0:-1])/2
        return 0.4 * np.pi * np.cos(np.pi * (x - 0.5))

    def RUN_SIMULATION(self,output_quantity = 100,verbose=False ):
        self.set_arrays()
        self.set_geometry()
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
            ##print(self.V[-1,1])
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
            
        p_compute = self.V[:,2]
        u_compute = self.V[:,1]
        rho_compute = self.V[:,0]
        if not self.return_conserved:
            return p_compute,u_compute,rho_compute,convergence_history
        else:
            Mass_compute = self.U[1:-1,0]
            Momentum_compute = self.U[1:-1,1]
            Energy_compute = self.U[1:-1,2]
            #U,_ = self.primitive_to_conserved(np.array([rho_exact,u_exact,p_exact]).T)
            

            return Mass_compute,Momentum_compute,Energy_compute,convergence_history




            



    def c_plus_minus(self,alpha_plus_minus,beta_LR,mach_LR,M_plus_minus):
        return alpha_plus_minus*(1+beta_LR)*mach_LR-beta_LR*M_plus_minus



    def f_convective_cell(self,CELL_L,CELL_R,nx,ny):
        rho_L,rho_R = CELL_L[0],CELL_R[0]
        u_L,u_R = CELL_L[1],CELL_R[1]
        v_L,   v_R   = CELL_L[2], CELL_R[2]  # y-velocity
        p_L,   p_R   = CELL_L[3], CELL_R[3] 
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        
        U_L = u_L*nx +ny*v_L
        U_R = u_R*nx +ny*v_R



        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        M_L,M_R = (U_L/a_L,U_R/a_R)
        
        alpha_plus = .5*(1+np.sign(M_L))
        alpha_minus = .5*(1-np.sign(M_R))
        beta_L = -np.maximum(0,1-np.floor(np.abs(M_L)))
        beta_R = -np.maximum(0,1-np.floor(np.abs(M_R)))

        M_plus = .25*(M_L+1)**2
        M_minus = -.25*(M_R-1)**2

        
        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * (u_L**2+v_R**2)
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * (u_R**2+v_R**2)

        temp_L = rho_L*a_L*self.c_plus_minus(alpha_plus,beta_L,M_L,M_plus)
        temp_R = rho_R*a_R*self.c_plus_minus(alpha_minus,beta_R,M_R,M_minus)
        FC_i_half = np.array([temp_L,u_L*temp_L,v_L*temp_L,ht_L*temp_L])+np.array([temp_R,u_R*temp_R,v_R*temp_R,ht_R*temp_R])

        return FC_i_half 

    
    
    
    
    
    
    
    def f_convective(self,direction = "left",CELL_LR = None):
        #shift = self.FL_FR_FUNC(i_plus_half=i_plus_half)
        if CELL_LR is None:   
            shift = self.FLUXL_FLUXR_FUNC(direction)
            rho_L,rho_R =shift(self.V[0]) # Density
            if np.any(rho_L<=0):
                rho_L = np.maximum(rho_L, self.epsilon)
            if np.any(rho_R<=0):
                rho_R = np.maximum(rho_R, self.epsilon)
            
            nx,ny = self.get_normal_directions(direction = direction)
            u_L,u_R = shift(self.V[1])  # Velocity
            v_L,v_R = shift(self.V[2])
            p_L,p_R = shift(self.V[3])  # Pressure
        else:
            CELL_L,CELL_R = CELL_LR
            rho_L,rho_R = CELL_L[0],CELL_R[0]
            if np.any(rho_L<=0):
                rho_L = np.maximum(rho_L, self.epsilon)
            if np.any(rho_R<=0):
                rho_R = np.maximum(rho_R, self.epsilon)
            
            nx,ny = self.get_normal_directions(direction = direction)
            u_L,u_R = shift(self.V[1])  # Velocity
            v_L,v_R = shift(self.V[2])
            p_L,p_R = shift(self.V[3])  # Pressure
        
        
        U_L = u_L*nx +ny*v_L
        U_R = u_R*nx +ny*v_R



        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        M_L,M_R = (U_L/a_L,U_R/a_R)
        
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
        FC_i_half = np.array([temp_L,u_L*temp_L,v_L*temp_L,ht_L*temp_L])+np.array([temp_R,u_R*temp_R,v_R*temp_R,ht_R*temp_R])

        return FC_i_half 
    
    
    def FLUXL_FLUXR_FUNC(self,direction:str):
        

        def shift_LR(F,num_vals = 16,is_velocity = False): 
            

            F_temp = np.zeros((F.shape[0]+2, F.shape[1]-2))
            F_temp[1:-1, :] = F[:,1:-1]

            # Manual first-order extrapolation
            F_temp[0, :]    = 2*F_temp[1, :] - F_temp[2, :]
            F_temp[-1, :]   = 2*F_temp[-2, :] - F_temp[-3, :]

                        
            
            

            if self.is_foil:
                F_temp[0,0:num_vals] = (F_temp[3,self.NJ-num_vals-1:])[::-1]
                F_temp[1,0:num_vals] = (F_temp[2,self.NJ-num_vals-1:])[::-1]
                
                (F_temp[0,self.NJ-num_vals-1:]) = F_temp[3,0:num_vals][::-1]
                (F_temp[1,self.NJ-num_vals-1:]) = F_temp[2,0:num_vals][::-1]

            F = F_temp


            if direction=="up":
                FL = F[1:-2,:]
                FR = F[2:-1,:]
            elif direction=="down":
                FL = F[0:-3,:]
                FR = F[1:-2,:]
            return FL,FR
        def shift_UD(F,is_velocity =False): 
        
            G_temp = np.zeros((F.shape[0]-2,F.shape[1]+2))
            G_temp[:,1:-1] = F[1:-1,:]
            G_temp[:,0] = 2*G_temp[:,1]-G_temp[:,2]
            G_temp[:,-1] = 2*G_temp[:,-2]-G_temp[:,-3]



            
            G = G_temp
            
            if direction=="right": 
                GL = G[:,1:-2]
                GR = G[:,2:-1]
            elif direction=="left":
                GL = G[:,0:-3]
                GR = G[:,1:-2]
            return GL,GR
        
        

        
        if direction=="left" or direction=="right":  

            return shift_UD
        return shift_LR
    

    
    
    
    def shift_func(self,shift_indx):
        if(shift_indx==0):
            return  None
        return shift_indx
    def psi_minus(self,F,shift_indx,UP_DOWN = False):
        if self.flux_limiter_scheme==0:
            return 1
        if not UP_DOWN:
            NUM =  (F[2+shift_indx:self.shift_func(-2+shift_indx),:]-F[1+shift_indx:-3+shift_indx,:])
            DEN = self.min_func(F[3+shift_indx:self.shift_func(-1+shift_indx),:]-F[2+shift_indx:self.shift_func(-2+shift_indx),:])
        else:
            NUM =  (F[:,2+shift_indx:self.shift_func(-2+shift_indx)]-F[:,1+shift_indx:-3+shift_indx])
            DEN = self.min_func(F[:,3+shift_indx:self.shift_func(-1+shift_indx)]-F[:,2+shift_indx:self.shift_func(-2+shift_indx)])
        r_minus =NUM/DEN
        r = r_minus
        if self.flux_limiter_scheme==1:
            return (r+np.abs(r))/self.min_func(1+r)
        elif self.flux_limiter_scheme==2:
            return (r**2+r)/self.min_func(1+r**2)


    def psi_plus(self,F,shift_indx,UP_DOWN = False):
        if self.flux_limiter_scheme==0:
                return 1
        
        if not UP_DOWN:   
            NUM = (F[4+shift_indx:self.shift_func(shift_indx)]-F[3+shift_indx:self.shift_func(-1+shift_indx)])
            DEN = self.min_func(F[3+shift_indx:self.shift_func(-1+shift_indx)]-F[2+shift_indx:self.shift_func(-2+shift_indx)])
            
        else:
            NUM = (F[:,4+shift_indx:self.shift_func(shift_indx)]-F[:,3+shift_indx:self.shift_func(-1+shift_indx)])
            DEN = self.min_func(F[:,3+shift_indx:self.shift_func(-1+shift_indx)]-F[:,2+shift_indx:self.shift_func(-2+shift_indx)])
        r_plus = NUM/DEN
        r = r_plus
        if self.flux_limiter_scheme==1:
            return (r+np.abs(r))/self.min_func(1+r)
        elif self.flux_limiter_scheme ==2:
            return (r**2+r)/self.min_func(1+r**2)
        

    def min_func(self,s):
        indeces = np.where(s==0)[0]
        s[indeces] = self.epsilon
        return np.sign(s)*np.maximum(self.epsilon,np.abs(s))

    def vanleer_flux_cell(self,CELL_L,CELL_R,nx,ny):
        return self.f_convective_cell(CELL_L,CELL_R, nx,ny)+self.f_pressure_flux_cell(CELL_L,CELL_R, nx,ny)

    def f_pressure_flux_cell(self,CELL_L,CELL_R,nx,ny):

        rho_L,rho_R = CELL_L[0],CELL_R[0]
        u_L,u_R = CELL_L[1],CELL_R[1]
        v_L,   v_R   = CELL_L[2], CELL_R[2]  # y-velocity
        p_L,   p_R   = CELL_L[3], CELL_R[3] 
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)



        
        U_L = u_L*nx +ny*v_L
        U_R = u_R*nx+ny*v_R
        
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        M_L,M_R = (U_L/a_L,U_R/a_R)
        
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
        

        return np.array([temp_zero,nx*D_plus*p_L,ny*D_plus*p_L,temp_zero])+\
            np.array([temp_zero,nx*D_minus*p_R,ny*D_minus*p_R,temp_zero])

    def f_pressure_flux(self=True,direction = "left",first_order=False):
        
        #shift = self.FL_FR_FUNC(i_plus_half=i_plus_half)
        shift = self.FLUXL_FLUXR_FUNC(direction)
        rho_L,rho_R =shift(self.V[self.rho_idx]) # Density
        
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        p_L,p_R = shift(self.V[ self.p_idx])  # Pressure
        
        
        nx,ny = self.get_normal_directions(direction)
        u_L,u_R = shift(self.V[1])  # Velocity
        v_L,v_R = shift(self.V[2])
        U_L = u_L*nx +ny*v_L
        U_R = u_R*nx+ny*v_R
        
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        M_L,M_R = (U_L/a_L,U_R/a_R)
        
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
        

        return np.array([temp_zero,nx*D_plus*p_L,ny*D_plus*p_L,temp_zero])+\
            np.array([temp_zero,nx*D_minus*p_R,ny*D_minus*p_L,temp_zero])
    def plot_primitive(self,type_="pressure"):
        midpoint_tempx = (self.x[1:,:]+self.x[0:-1,:])/2
        midpointxx = (midpoint_tempx[:,1:]+midpoint_tempx[:,0:-1])/2
        midpoint_tempy = (self.y[1:,:]+self.y[0:-1,:])/2
        midpointyy = (midpoint_tempy[:,1:]+midpoint_tempy[:,0:-1])/2
        #print(self.V.shape,midpointxx.shape)
        if type_=="pressure":
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=(self.V[self.p_idx]).flatten())
            plt.colorbar()
        if type_=="velocity":
            plt.quiver(midpointxx,midpointyy,self.V[self.u_idx,1:-1,1:-1],self.V[self.v_idx,1:-1,1:-1])
        if type_=="density":
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=self.V[self.rho_idx].flatten())
            plt.colorbar()
        plt.show()
    def compute_timestep(self):
        rho =self.V[self.rho_idx,1:-1,1:-1]  # Density
        u = self.V[self.u_idx,1:-1,1:-1]
        v = self.V[self.v_idx,1:-1,1:-1]  # Velocity
        p = self.V[self.p_idx,1:-1,1:-1]  # Pressure
        a = np.sqrt(np.maximum(0,(self.gamma*(p/self.min_func(rho)))))


        nx_psi_p1,ny_psi_p1 = self.get_normal_directions("left",UPWIND=False)
        nx_psi_p2,ny_psi_p2 = self.get_normal_directions("right",UPWIND=False)
        nx_psi = (nx_psi_p2+nx_psi_p1)/2
        ny_psi = (ny_psi_p2+ny_psi_p1)/2

        nx_eta_p1,ny_eta_p1 = self.get_normal_directions("up",UPWIND=False)
        nx_eta_p2,ny_eta_p2 = self.get_normal_directions("down",UPWIND=False)
        nx_eta = (nx_eta_p2+nx_eta_p1)/2
        ny_eta = (ny_eta_p2+ny_eta_p1)/2
        
        
        Area_LR = .5*(self.A_left +self.A_right)
        Area_UD = .5*(self.A_top+self.A_bottom)
        
        lambda_LR = np.abs(u*nx_psi +v*ny_psi)+a
        lambda_UD = np.abs(u*nx_eta+v*ny_eta)+a

        delta_t = self.AREA/self.min_func(lambda_LR*Area_LR+lambda_UD*Area_UD)
        self.delta_t =self.CFL*delta_t


    def fortran_function(self,direction = "left",upwind_scheme = "roe"):
        Nx,Ny = self.get_normal_directions(direction=direction)
        shift = self.FLUXL_FLUXR_FUNC(direction = direction)
        V = np.array([shift(self.V[i]) for i in range(4)])
        VL,VR = V[:,0,:,:],V[:,1,:,:]
        temp_out = np.zeros_like(VL)
        if upwind_scheme=="roe":
            upwind = library.upwind_module.roe_flux
        elif upwind_scheme=="van leer":
            upwind = library.upwind_module.vanleer_flux

        for i in range(self.NI-1):
            for j in range(self.NJ-1):
                vl = VL[:,i,j]
                vr = VR[:,i,j]
                nx = Nx[i,j]
                ny = Ny[i,j]
                temp_out[:,i,j] = upwind(vl,vr,nx,ny)
        return temp_out


    def compute_residual(self):
        nx_L,ny_L = self.get_normal_directions(direction="left")
        nx_R,ny_R = self.get_normal_directions(direction="right")
        nx_U,ny_U = self.get_normal_directions(direction="up")
        nx_D,ny_D = self.get_normal_directions(direction="down")
        for i in range(1,self.NI-1):
            for j in range(1,self.NJ-1):
                
                FL_cell_L = self.V[:,i-1,j]
                FL_cell_R = self.V[:,i,j]
                FL = self.f_convective_cell(FL_cell_L,FL_cell_R,nx_L[i,j],ny_L[i,j]) + self.f_pressure_flux_cell(FL_cell_L,FL_cell_R,nx_L[i,j],ny_L[i,j])

                #Flux right
                FR_cell_L = self.V[:,i,j]
                FR_cell_R = self.V[:,i+1,j]
                FR = self.f_convective_cell(FR_cell_L,FR_cell_R,nx_R[i,j],ny_R[i,j]) + self.f_pressure_flux_cell(FR_cell_L,FR_cell_R,nx_R[i,j],ny_R[i,j])

                # Flux down
                FD_cell_L = self.V[:, i, j-1]
                FD_cell_R = self.V[:, i, j]
                FD = self.f_convective_cell(FD_cell_L, FD_cell_R, nx_D[i,j], ny_D[i,j]) \
                    + self.f_pressure_flux_cell(FD_cell_L, FD_cell_R, nx_D[i, j], ny_D[i, j])

                # Flux up
                FU_cell_L = self.V[:, i, j]
                FU_cell_R = self.V[:, i, j+1]
                FU = self.f_convective_cell(FU_cell_L, FU_cell_R, nx_U[i,j], ny_U[i,j]) \
                    + self.f_pressure_flux_cell(FU_cell_L, FU_cell_R, nx_U[i, j], ny_U[i, j])
                
                


                self.residual[:, i, j] = self.A_left[:,i, j]* FL + self.A_right[:,i, j]* FR + self.A_bottom[:,i, j] * FD+ self.A_top[:,i, j]   * FU

    def compute_residual_cell(self):


        nx_L,ny_L = self.get_normal_directions(direction="right")
        nx_R,ny_R = self.get_normal_directions(direction="right")
        nx_U,ny_U = self.get_normal_directions(direction="up")
        nx_D,ny_D = self.get_normal_directions(direction="up")
        V = self.V

        #nx_L[-1,:] = 1
        #nx_R[-1,:] = 1
        #ny_L[-1,:] = 0
        #ny_R[-1,:] = 0

        #nx_U[-1,:] = 0
        #nx_U[-1,:] = 0
        #ny_D[-1,:] = 0
        #ny_D[-1,:] = 0
        
        



        FL_cell_L = V[:,1:-1,0:-2]
        FL_cell_R = V[:,1:-1,1:-1]
        FR_cell_L = V[:,1:-1,1:-1]
        FR_cell_R = V[:,1:-1,2:]
        FD_cell_L = V[:, 0:-2, 1:-1]   
        FD_cell_R = V[:, 1:-1, 1:-1]  
        FU_cell_L = V[:, 1:-1, 1:-1]   
        FU_cell_R = V[:, 2:  , 1:-1]   

        self.FL[:,:,:-1] = self.vanleer_flux_cell(FL_cell_L,FL_cell_R,nx_L[:,0:-1],ny_L[:,0:-1])
        self.FR[:,:,1:] = self.vanleer_flux_cell(FR_cell_L,FR_cell_R,nx_R[:,1:],ny_R[:,1:])
        self.FD[:, 0:-1, :] = self.vanleer_flux_cell(FD_cell_L, FD_cell_R,nx_D[0:-1, :], ny_D[0:-1, :])
        self.FU[:, 1:, :] = self.vanleer_flux_cell(    FU_cell_L, FU_cell_R,nx_U[1:,:], ny_U[1:, :])

        residual = ((self.FU[:,1:,:]*self.A_top)-(self.FD[:,0:-1,:]*self.A_bottom))+((self.FR[:,:,1:]*self.A_right)-(self.FL[:,:,0:-1]*self.A_left))

        return residual

    def iteration_step(self,return_error=True):
        deltax = np.abs(self.x[2,:]-self.x[1,:])
        d_minus_half = 0
        d_plus_half = 0
        if self.damping_scheme==0:
            d_plus_half = -(self.d2(shift=1)-self.d4(shift=1))
            d_minus_half = -(self.d2(shift=-1)-self.d4(shift=-1))
        
        #self.compute_timestep()
        #if not self.local_timestep:
        #    self.delta_t = np.min(self.delta_t)
        A_plus_1_2 =  self.A_LR
        A_minus_1_2 = self.A_LR
        A_UP =  self.A_UD
        A_DOWN = (self.A_UD)
        
        if self.damping_scheme==0:
            F_plus_1_2 = (self.F[2:,:]+self.F[1:-1,:])/2 + d_plus_half
            F_minus_1_2 = (self.F[0:-2,:]+self.F[1:-1,:])/2 +d_minus_half
        elif self.damping_scheme==1:
            F_plus_1_2 = (A_plus_1_2*(self.f_convective(direction="right")+self.f_pressure_flux(direction="right")))
            F_minus_1_2 = (A_minus_1_2*(self.f_convective(direction="left")+self.f_pressure_flux(direction="left")))
            F_UP = (A_UP*(self.f_convective(direction="up")+self.f_pressure_flux(direction="up")))
            F_DOWN = (A_DOWN*(self.f_convective(direction="down")+self.f_pressure_flux(direction="down")))
        elif self.damping_scheme==2:
            F_plus_1_2 = self.compute_roe_flux(direction="right")
            F_minus_1_2 = self.compute_roe_flux(direction="left")
            F_UP = self.compute_roe_flux(direction="up")
            F_DOWN = self.compute_roe_flux(direction="down")
        elif self.damping_scheme==3:

            F_plus_1_2 = self.fortran_function(direction="right")
            F_minus_1_2 = self.fortran_function(direction="left")
            F_UP = self.fortran_function(direction="up")
            F_DOWN = self.fortran_function(direction="down")
        A_plus_1_2 =  self.A_right
        A_minus_1_2 = (self.A_left)
        A_UP =  self.A_top
        A_DOWN = (self.A_bottom)
        
        #print("Convective",self.f_pressure_flux(i_plus_half=True))
        
        
        #residual = ( F_plus_1_2*A_plus_1_2+F_minus_1_2*A_minus_1_2 ) + (F_UP*A_UP+F_DOWN*A_DOWN )
        
        if self.damping_scheme==4:

        
            self.residual[:,1:-1,1:-1] = self.compute_residual_cell()


        
                
        
        
        self.delta_t = self.CFL
        self.U[:,1:-1,1:-1] = self.U[:,1:-1,1:-1]-(self.residual[:,1:-1,1:-1]*self.delta_t/(self.AREA))
        
        #self.U[:,1:-1,1:] = self.residual[:,1:-1,1:]
        #self.U[:,1:-1,0:-1] = self.residual[:,1:-1,0:-1]
        #self.U[:,1:,1:-1] =self.residual[:,1:,1:-1]
        #self.U[:,0:-1,1:-1] =self.residual[:,0:-1,1:-1]
        self.U[:, 1:-1, 0] = 2 * self.U[:, 1:-1, 1] - self.U[:, 1:-1, 2]
        self.U[:, 1:-1, -1] = 2 * self.U[:, 1:-1, -2] - self.U[:, 1:-1, -3]

        # Extrapolate bottom/top faces (y = 0 and y = -1), excluding boundaries in x
        self.U[:, 0, 1:-1] = 2 * self.U[:, 1, 1:-1] - self.U[:, 2, 1:-1]
        self.U[:, -1, 1:-1] = 2 * self.U[:, -2, 1:-1] - self.U[:, -3, 1:-1]


        # Now do the corners explicitly
        # Top-left
        self.U[:, 0, 0] = 2 * self.U[:, 1, 1] - self.U[:, 2, 2]
        # Top-right
        self.U[:, 0, -1] = 2 * self.U[:, 1, -2] - self.U[:, 2, -3]
        # Bottom-left
        self.U[:, -1, 0] = 2 * self.U[:, -2, 1] - self.U[:, -3, 2]
        # Bottom-right
        self.U[:, -1, -1] = 2 * self.U[:, -2, -2] - self.U[:, -3, -3]
        
        self.V = self.conserved_to_primitive(self.U)
        #
        #self.update_source()
        if return_error:
            return self.residual
        #print("Inflow velocity:", self.V[0,1],self.V[1,1])




















    def get_boundary_conditions_NORMALS(self):
        
        
        # Top
        v2x = self.V[1,-2,1:-1]
        v2y = self.V[2,-2,1:-1]
        u_top,v_top = self.compute_slip_walls(self.upward_S[self.unormal,-1,:],\
                                              self.upward_S[self.vnormal,-1,:],self.upward_normal[self.unormal,-1,:],\
                                                self.upward_normal[self.vnormal,-1,:],v2x,v2y)
        top = [u_top,v_top]

        v2x = (self.V[1,1,1:-1])
        v2y = (self.V[2,1,1:-1])
        u_top,v_top = self.compute_slip_walls(self.downward_S[self.unormal,0,:],\
                                              self.downward_S[self.vnormal,0,:],self.downward_normal[self.unormal,0,:],\
                                                self.downward_normal[self.vnormal,0,:],v2x,v2y)
        bottom = [u_top,v_top]
        

        return top,bottom












    """    
    def get_boundary_conditions_NORMALS(self):
        
        
        # Top
        v2x = self.V[1,1:-1,-2]
        v2y = self.V[2,1:-1,-2]
        u_top,v_top = self.compute_slip_walls(self.leftward_S[self.unormal,:,-1],\
                                              self.leftward_S[self.vnormal,:,-1],self.leftward_normal[self.unormal,:,-1],\
                                                self.leftward_normal[self.vnormal,:,-1],v2x,v2y)
        top = [u_top,v_top]

        v2x = self.V[1,1:-1,1]
        v2y = self.V[2,1:-1,1]
        u_top,v_top = self.compute_slip_walls(self.rightward_S[self.unormal,:,0],\
                                              self.rightward_S[self.vnormal,:,0],self.rightward_normal[self.unormal,:,0],\
                                                self.rightward_normal[self.vnormal,:,0],v2x,v2y)
        bottom = [u_top,v_top]
        

        return top,bottom
    """
    def compute_slip_walls(self,s1,s2,n1,n2,v2x,v2y):
        # Compute dot products
        dot_vs = v2x * s1 + v2y * s2
        dot_vn = v2x * n1 + v2y * n2

        # Determinant of A
        detA = s1 * n2 - s2 * n1

        # Inverse of A (analytical)
        A_inv = (1 / detA) * np.array([
            [ n2, -s2],
            [-n1,  s1]
        ])

        # Right-hand side
        rhs = np.array([dot_vs, -dot_vn])
        v1 = np.einsum("ijk,jk->ik",A_inv,rhs)
        #temp = np.array([v2x,v2y]) - 2*(np.einsum("ij,ij->i",np.array([v2x,v2y]),np.array([n1,n2]))[:,np.newaxis]).T@np.array([n1,n2])
        return v1


    def compute_doubleBar_values(self,direction,compute_deltas = False):
        shift = self.FLUXL_FLUXR_FUNC(direction)
        rho_L,rho_R =shift(self.V[self.rho_idx]) # Density
        
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        p_L,p_R = shift(self.V[ self.p_idx])  # Pressure
        
        u_L,u_R = shift(self.V[1],is_velocity=True)  # Velocity
        v_L,v_R = shift(self.V[2],is_velocity = True)
        


        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * (u_L**2+v_L**2)
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * (u_R**2 +v_R**2)

        R = np.sqrt(np.maximum(0,rho_R/self.min_func(rho_L)))
        p_double_bar = np.sqrt(np.maximum(self.epsilon,p_L*p_R))
        rho_double_bar = R*rho_L
        u_double_bar = (R*u_R+u_L)/(R+1)
        v_double_bar = (R*v_R+v_L)/(R+1)
        ht_double_bar = (R*ht_R+ht_L)/(R+1)
        if not compute_deltas:
            return (p_double_bar,rho_double_bar,u_double_bar,v_double_bar,ht_double_bar)
        return (p_double_bar,rho_double_bar,u_double_bar,v_double_bar,ht_double_bar),(p_R-p_L,rho_R-rho_L,u_R-u_L,v_R-v_L)
    def compute_roe_eigs(self,direction):
        _,rho,u,v,ht = self.compute_doubleBar_values(direction)
        a = np.sqrt(np.maximum(0,(self.gamma-1)*(ht-(u**2+v**2)/2)))
        nx,ny = self.get_normal_directions(direction=direction)
        U = nx*u+ny*v
        ones = np.ones_like(u)
        lam1 = U
        lam2 = U
        lam3 = U+a
        lam4 = U-a
        

        r1 = np.array([ones,u,v,(u**2+v**2)/2])
        r2 = np.array([0*ones,ny*rho,-nx*rho,rho*(ny*u-nx*v)])
        r3 = (rho/np.maximum(self.epsilon,2*a))*np.array([ones,u+nx*a,v+ny*a,ht+U*a])
        r4 = (-rho/np.maximum(self.epsilon,2*a))*np.array([ones,u-nx*a,v-ny*a,ht-U*a])

        return (lam1,lam2,lam3,lam4), (r1,r2,r3,r4)
    

    
     

    def compute_roe_flux(self, direction):
        #self.set_boundary_conditions()
        shift = self.FLUXL_FLUXR_FUNC(direction)
        lams,eigvecs = self.compute_roe_eigs(direction)
        ws = self.compute_wave_amplitudes(direction)

        nx,ny = self.get_normal_directions(direction)
        rho_L,rho_R =shift(self.V[self.rho_idx]) # Density
        
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        p_L,p_R = shift(self.V[ self.p_idx])  # Pressure
        
        u_L,u_R = shift(self.V[1],is_velocity = True)  # Velocity
        v_L,v_R = shift(self.V[2],is_velocity = True)

 
        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * (u_L**2+v_L**2)
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * (u_R**2+v_R**2)

        vel_L = u_L*nx+v_L*ny
        vel_R = u_R*nx+v_R*ny

        F_L =  np.array([
            rho_L* vel_L,
            rho_L * u_L* vel_L,
            rho_L * v_L * vel_L,
            rho_L * ht_L * vel_L
        ])

        F_R = np.array([
            rho_R* vel_R,
            rho_R * u_R* vel_R,
            rho_R * v_R * vel_R,
            rho_R * ht_R * vel_R
        ])




        
        #if i_plus_half and not self.p_back== -1:
        #    F_R[-1,2] = self.p_back
        
        temp_sum = 0
        def modified_lambda(eigenvalue,lambda_eps = .1):
            _,_,u,v,ht = self.compute_doubleBar_values(direction)
            a = np.sqrt(np.maximum(self.epsilon,(self.gamma-1)*(ht-(u**2+v**2)/2)))
            indeces_LT = np.where(eigenvalue<=2*lambda_eps*a)
            eigenvalue[indeces_LT] = (eigenvalue[indeces_LT]**2)/(4*lambda_eps*a[indeces_LT])+lambda_eps*a[indeces_LT]
            
            return eigenvalue

        for i in range(4):
            temp_sum += modified_lambda(np.abs(lams[i]))*ws[i]*eigvecs[i] 
        flux_roe = .5*(F_L+F_R) - .5*temp_sum
        return flux_roe
    
         
    def compute_wave_amplitudes(self,direction):
        
        bars,deltas = self.compute_doubleBar_values(direction,compute_deltas=True)
        nx,ny = self.get_normal_directions(direction)
        _,rho,u,v,ht = bars
        delta_p,delta_rho,delta_u,delta_v = deltas
        a = np.sqrt(np.maximum(self.epsilon,(self.gamma-1)*(ht-(u**2+v**2)/2)))
        dw1 = delta_rho+(delta_p/self.min_func(a**2))
        dw2 = ny*delta_u-nx*delta_v
        dw3 = nx*delta_u+ny*delta_v + (delta_p/self.min_func(rho*a))
        dw4 = nx*delta_u+ny*delta_v - (delta_p/self.min_func(rho*a))
        
        return (dw1,dw2,dw3,dw4)
    


        

    def compute_doubleBar_values_cell(self,CELL_L,CELL_R,compute_deltas = False):
        rho_L,rho_R = CELL_L[0],CELL_R[0]
        u_L,u_R = CELL_L[1],CELL_R[1]
        v_L,   v_R   = CELL_L[2], CELL_R[2]  # y-velocity
        p_L,   p_R   = CELL_L[3], CELL_R[3] 
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        


        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * (u_L**2+v_L**2)
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * (u_R**2 +v_R**2)

        R = np.sqrt(np.maximum(0,rho_R/self.min_func(rho_L)))
        p_double_bar = np.sqrt(np.maximum(self.epsilon,p_L*p_R))
        rho_double_bar = R*rho_L
        u_double_bar = (R*u_R+u_L)/(R+1)
        v_double_bar = (R*v_R+v_L)/(R+1)
        ht_double_bar = (R*ht_R+ht_L)/(R+1)
        if not compute_deltas:
            return (p_double_bar,rho_double_bar,u_double_bar,v_double_bar,ht_double_bar)
        return (p_double_bar,rho_double_bar,u_double_bar,v_double_bar,ht_double_bar),(p_R-p_L,rho_R-rho_L,u_R-u_L,v_R-v_L)
    def compute_roe_eigs_cell(self,CELL_L,CELL_R,nx,ny):
        _,rho,u,v,ht = self.compute_doubleBar_values_cell(CELL_L,CELL_R)
        a = np.sqrt(np.maximum(0,(self.gamma-1)*(ht-(u**2+v**2)/2)))
        
        U = nx*u+ny*v
        ones = np.ones_like(u)
        lam1 = U
        lam2 = U
        lam3 = U+a
        lam4 = U-a
        

        r1 = np.array([ones,u,v,(u**2+v**2)/2])
        r2 = np.array([0*ones,ny*rho,-nx*rho,rho*(ny*u-nx*v)])
        r3 = (rho/np.maximum(self.epsilon,2*a))*np.array([ones,u+nx*a,v+ny*a,ht+U*a])
        r4 = (-rho/np.maximum(self.epsilon,2*a))*np.array([ones,u-nx*a,v-ny*a,ht-U*a])

        return (lam1,lam2,lam3,lam4), (r1,r2,r3,r4)



        

    def compute_roe_flux_cell(self,CELL_L,CELL_R, nx,ny):
        #self.set_boundary_conditions()
        
        lams,eigvecs = self.compute_roe_eigs_cell(CELL_L,CELL_R,nx,ny)
        ws = self.compute_wave_amplitudes_cell(CELL_L,CELL_R,nx,ny)

    
        rho_L,rho_R = CELL_L[0],CELL_R[0]
        u_L,u_R = CELL_L[1],CELL_R[1]
        v_L,   v_R   = CELL_L[2], CELL_R[2]  # y-velocity
        p_L,   p_R   = CELL_L[3], CELL_R[3] 
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)


        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * (u_L**2+v_L**2)
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * (u_R**2+v_R**2)

        vel_L = u_L*nx+v_L*ny
        vel_R = u_R*nx+v_R*ny

        F_L =  np.array([
            rho_L* vel_L,
            rho_L * u_L* vel_L,
            rho_L * v_L * vel_L,
            rho_L * ht_L * vel_L
        ])

        F_R = np.array([
            rho_R* vel_R,
            rho_R * u_R* vel_R,
            rho_R * v_R * vel_R,
            rho_R * ht_R * vel_R
        ])




        
        #if i_plus_half and not self.p_back== -1:
        #    F_R[-1,2] = self.p_back
        
        temp_sum = 0
        def modified_lambda(eigenvalue,lambda_eps = .0001):
            _,_,u,v,ht = self.compute_doubleBar_values_cell(CELL_L,CELL_R)
            a = np.sqrt(np.maximum(self.epsilon,(self.gamma-1)*(ht-(u**2+v**2)/2)))
            indeces_LT = np.where(eigenvalue<=2*lambda_eps*a)
            eigenvalue[indeces_LT] = (eigenvalue[indeces_LT]**2)/(4*lambda_eps*a[indeces_LT])+lambda_eps*a[indeces_LT]
            
            return eigenvalue

        for i in range(4):
            temp_sum += modified_lambda(np.abs(lams[i]))*ws[i]*eigvecs[i] 
        flux_roe = .5*(F_L+F_R) - .5*temp_sum
        return flux_roe

            
    def compute_wave_amplitudes_cell(self,CELL_L,CELL_R,nx,ny):
        
        bars,deltas = self.compute_doubleBar_values_cell(CELL_L,CELL_R,compute_deltas=True)
        
        _,rho,u,v,ht = bars
        delta_p,delta_rho,delta_u,delta_v = deltas
        a = np.sqrt(np.maximum(self.epsilon,(self.gamma-1)*(ht-(u**2+v**2)/2)))
        dw1 = delta_rho+(delta_p/self.min_func(a**2))
        dw2 = ny*delta_u-nx*delta_v
        dw3 = nx*delta_u+ny*delta_v + (delta_p/self.min_func(rho*a))
        dw4 = nx*delta_u+ny*delta_v - (delta_p/self.min_func(rho*a))
        
        return (dw1,dw2,dw3,dw4)





    
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

        return P/np.maximum((R*T),self.epsilon)
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


            
    def load_grid(self,filename):

        with open(filename, 'r') as f:
            # Read integers
            nzones = int(f.readline())
            imax, jmax, kmax = map(int, f.readline().split())

            size = imax * jmax * kmax

            # Read the flattened data for x, y, and zztemp
            data = []
            while len(data) < 3 * size:
                line = f.readline()
                data.extend(map(float, line.split()))

            # Split data into x, y, zztemp
            data = np.array(data)
            x = data[0:size].reshape((kmax, jmax, imax))
            y = data[size:2*size].reshape((kmax, jmax, imax))
            zztemp = data[2*size:].reshape((kmax, jmax, imax))

        self.x = x[0]
        self.y = y[0]
