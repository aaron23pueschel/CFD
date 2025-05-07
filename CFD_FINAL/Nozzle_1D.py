import f90nml
import numpy as np
import scipy
import ctypes
import matplotlib.pyplot as plt
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
            self.NJ = nml["inputs"]["NJ"]
            self.T0 = nml["inputs"]["T0"]
            self.Ru = nml["inputs"]["Ru"]
            self.M = nml["inputs"]["M"]
            self.R = self.Ru/self.M
            self.CFL = nml["inputs"]["CFL"]
            self.ghost_cells = nml["inputs"]["ghost_cells"]
            self.domainx = nml["inputs"]["domainx"]
            self.domainy = nml["inputs"]["domainy"]
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
        
        self.U =  None
        self.V = None
        self.A = None
        self.S = None
        self.residual = None
        self.mach = None
        self.delta_t = None
        self.nx_ny_top = None
        self.nx_ny_left = None
        self.nx_ny_right = None
        self.nx_ny_bottom = None
        self.AREA = None

                ## C functions
        #self.FM = functions().FM()
        #self.DfdM = functions().DfdM()
        #self.newton = functions().newton()


    def set_arrays(self):
        self.U = np.zeros((self.NJ+1,self.NI+1,4)) # Number of faces
        self.F = np.zeros((self.NJ+1,self.NI+1,4))
        self.V = np.zeros((self.NJ+1,self.NI+1,4))
        self.S = np.zeros((self.NJ-1,self.NI-1,4)) # No ghost cells for source

    def primitive_to_conserved(self,primitive):
        rho = primitive[:,:, 0]  # Density
        u = primitive[:,:, 1]  # Velocity
        v = primitive[:,:, 2] 
        p = primitive[:,:,3]

        # Compute total energy per unit mass (et)
        et = p / ((self.gamma - 1) * rho) + 0.5 * ((u)**2+v**2)
        ht = (self.gamma / (self.gamma - 1)) * (p / rho) + 0.5 * ((u)**2+v**2)
        
        U = np.einsum("ijk->jki",np.array([rho, rho * u,rho*v, rho * et]))

        F = np.column_stack((rho * u, rho * u**2 + p,rho*u*v, rho * u * ht))
        G = np.array([rho * v,rho*u*v, rho * v**2 + p, rho * v * ht])
        
        return U, F,G  # Return both conserved variables and fluxes

    def conserved_to_primitive(self,conserved):
        rho = conserved[:,:,0]  # Density
       
        u = conserved[:,:,1] / rho  # Velocity
        v = conserved[:,:,2]/rho
        #if np.any(u<=0):
        #    u = np.sign(u)*np.maximum(np.abs(u), self.epsilon)
        p = (self.gamma - 1) * (conserved[:,:,3] - 0.5 * rho * (u**2+v**2))  # Pressure
        V = np.array([rho,u,v,p])
        
        return np.einsum("ijk->jki",V) 

    def set_uniform_geometry(self):
        self.x = np.linspace(self.domainx[0],self.domainx[1],self.NI)
        self.y = np.linspace(self.domainy[0],self.domainy[1],self.NJ)
    def set_curved_geometry(self):
        x = np.linspace(-1,1,self.NI)
        y = np.linspace(0,1,self.NJ)
        xx_,yy_ = np.meshgrid(x,y)
        #xx = (xx_)
        #yy = (yy_+xx**2)
        xx = (xx_)
        yy = .5*(yy_+xx**2)
        self.xx = xx
        self.yy = yy
    def plot_Geometry(self):
        plt.plot(self.xx.flatten(), self.yy.flatten(), ".")
        #self.A = self.Area()
    def extrapolate1(self,a,extrapolate_y = False):

        temp0 = a[1]
        temp1 = a[-2]



        if not extrapolate_y:
            temp0 = 2*a[1]-a[2]
            temp1 = 2*a[-2]-a[-3]
            return temp0,temp1
        temp0 = 2*a[:,1]-a[:,2]
        temp1 = 2*a[:,-2]-a[:,-3]
        return temp0,temp1
        
    def set_initial_conditions(self):
        
        mach = .1*np.ones((self.NJ+1,self.NI+1))
        T = self.total_T(self.gamma,mach,self.T0)
        p = self.total_p(self.gamma, mach,self.p0) # Number of cells
        rho = self.total_density(p,self.R,T)
        u = self.total_velocity(self.gamma,mach,self.R,T)
        
        
        self.V[:,:,0] = rho
        self.V[1:-1,1:-1,1] = u[1:-1,1:-1]*self.upward_S[2]
        self.V[1:-1,1:-1,2] = u[1:-1,1:-1]*self.upward_S[3]

        self.V[:,0,1:3] = self.V[:,1,1:3]
        self.V[:,-1,1:3] = self.V[:,-2,1:3]
        self.V[0,:,1:3] = self.V[1,:,1:3]
        self.V[-1,:,1:3] = self.V[-2,:,1:3]
        
        
        self.V[:,:,3] = p
        self.U,self.F,self.G = self.primitive_to_conserved(self.V)
    def plot_primitive(self,type_="pressure"):
        midpoint_tempx = (self.xx[1:,:]+self.xx[0:-1,:])/2
        midpointxx = (midpoint_tempx[:,1:]+midpoint_tempx[:,0:-1])/2
        midpoint_tempy = (self.yy[1:,:]+self.yy[0:-1,:])/2
        midpointyy = (midpoint_tempy[:,1:]+midpoint_tempy[:,0:-1])/2
        #print(self.V.shape,midpointxx.shape)
        if type_=="pressure":
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=(self.V[1:-1,1:-1,3]).flatten())
            plt.colorbar()
        if type_=="velocity":
            plt.quiver(midpointxx,midpointyy,self.V[1:-1,1:-1,1],self.V[1:-1,1:-1,2])
        if type_=="density":
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=self.V[1:-1,1:-1,0].flatten())
            plt.colorbar()
        plt.show()
        
    def plot_boundary_primitive(self,type_ = "pressure"):
        midpoint_tempx = (self.xx[1:,:]+self.xx[0:-1,:])/2
        midpointxx = (midpoint_tempx[:,1:]+midpoint_tempx[:,0:-1])/2
        midpoint_tempy = (self.yy[1:,:]+self.yy[0:-1,:])/2
        midpointyy = (midpoint_tempy[:,1:]+midpoint_tempy[:,0:-1])/2
        
        if type_=="pressure":
            fig, ax = plt.subplots()
            vmin = self.V[:,:,3].min()
            vmax = self.V[:,:,3].max()

            # Plot each wall with consistent color mapping
            plt.scatter(midpointxx[0,:], midpointyy[0,:], c=self.V[0,1:-1,3], cmap='viridis', vmin=vmin, vmax=vmax)
            plt.scatter(midpointxx[-1,:], midpointyy[-1,:], c=self.V[-1,1:-1,3], cmap='viridis', vmin=vmin, vmax=vmax)
            plt.scatter(midpointxx[:,0], midpointyy[:,0], c=self.V[1:-1,0,3], cmap='viridis', vmin=vmin, vmax=vmax)
            plt.scatter(midpointxx[:,-1], midpointyy[:,-1], c=self.V[1:-1,-1,3], cmap='viridis', vmin=vmin, vmax=vmax)

        
            plt.colorbar()  # Use 'sc' here
            
            
        if type_=="velocity":
           
           
           
           
            plt.quiver(midpointxx[0,:],midpointyy[0,:],self.V[0,1:-1,1],self.V[0,1:-1,2])
            plt.quiver(midpointxx[-1,:],midpointyy[-1,:],self.V[-1,1:-1,1],self.V[-1,1:-1,2])
            plt.quiver(midpointxx[:,0],midpointyy[:,0],self.V[1:-1,0,1],self.V[1:-1,0,2])
            plt.quiver(midpointxx[:,-1],midpointyy[:,-1],self.V[1:-1,-1,1],self.V[1:-1,-1,2])
        if type_=="density":
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=self.V[1:-1,1:-1,0].flatten())
            plt.colorbar()
        plt.show()

    def compute_doubleBar_values(self,i_plus_half,compute_deltas = False):
        shift = self.FL_FR_FUNC(i_plus_half=i_plus_half)

        rho_L,rho_R =shift(self.V[:,0]) # Density
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
        u_L,u_R = shift(self.V[:,1])  # Velocity
        p_L,p_R = shift(self.V[:, 2])  # Pressure
        if i_plus_half and not self.p_back== -1:
            p_R[-1] = self.p_back
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
    def compute_roe_eigs(self,i_plus_half):
        p,rho,u,ht = self.compute_doubleBar_values(i_plus_half)
        a = np.sqrt(np.maximum(0,(self.gamma-1)*(ht-(u**2)/2)))
        ones = np.ones_like(u)
        lam1 = u
        lam2 = u+a
        lam3 = u-a
        

        r1 = np.array([ones,u,(u**2)/2])
        r2 = (rho/np.maximum(self.epsilon,2*a))*np.array([ones,u+a,ht+u*a])
        r3 = (-rho/np.maximum(self.epsilon,2*a))*np.array([ones,u-a,ht-u*a])

        return (lam1,lam2,lam3), (r1,r2,r3)
    def compute_roe_flux(self, i_plus_half):
        #self.set_boundary_conditions()
        shift = self.FL_FR_FUNC(i_plus_half=i_plus_half,vector = True)
        lams,eigvecs = self.compute_roe_eigs(i_plus_half=i_plus_half)
        #print(lams)
        ws = self.compute_wave_amplitudes(i_plus_half=i_plus_half)
        F_L,F_R = shift(self.F)
        #if i_plus_half and not self.p_back== -1:
        #    F_R[-1,2] = self.p_back
        
        temp_sum = 0
        def modified_lambda(eigenvalue,lambda_eps = .1):
            p,rho,u,ht = self.compute_doubleBar_values(i_plus_half)
            a = np.sqrt(np.maximum(self.epsilon,(self.gamma-1)*(ht-(u**2)/2)))
            indeces_LT = np.where(eigenvalue<=2*lambda_eps*a)
            eigenvalue[indeces_LT] = (eigenvalue[indeces_LT]**2)/(4*lambda_eps*a[indeces_LT])+lambda_eps*a[indeces_LT]
            
            return eigenvalue

        for i in range(3):
            temp_sum += modified_lambda(np.abs(lams[i]))*ws[i]*eigvecs[i] 
        flux_roe = .5*(F_L+F_R) - .5*temp_sum.T
        return flux_roe
    
    def set_normals_OLD(self):
        # Sets area normals and normals
        xx = self.xx
        yy = self.yy
        self.A_top_bottom = np.sqrt((xx[1:,:]-xx[0:-1,:])**2+(yy[1:,:]-yy[0:-1,:])**2)
        self.xx_top_bottom = (xx[1:,:]+xx[0:-1,:])/2
        self.yy_top_bottom = (yy[1:,:]+yy[0:-1,:])/2
        nxi_top = -(yy[1:,:]-yy[0:-1,:])
        nyi_top = xx[1:,:]-xx[0:-1,:]

        self.A_left_right = np.sqrt((xx[:,1:]-xx[:,0:-1])**2+(yy[:,1:]-yy[:,0:-1])**2)
        self.xx_left_right = ((xx[:,1:]+xx[:,0:-1])/2)
        self.yy_left_right = ((yy[:,1:]+yy[:,0:-1])/2)
        nxi_left_right = -(yy[:,1:]-yy[:,0:-1])
        nyi_left_right = (xx[:,1:]-xx[:,0:-1])




       
        self.nx_ny_right = [nxi_top/self.A_top_bottom,nyi_top/self.A_top_bottom]
        self.nx_ny_left = [-nxi_top/self.A_top_bottom,-nyi_top/self.A_top_bottom]
        self.nx_ny_top = [nxi_left_right/self.A_left_right,nyi_left_right/self.A_left_right]
        self.nx_ny_bottom = [-nxi_left_right/self.A_left_right,-nyi_left_right/self.A_left_right]
        self.sx_sy_left = [self.nx_ny_left[1],-self.nx_ny_left[0]]
        self.sx_sy_right = [self.nx_ny_right[1], -self.nx_ny_right[0]]
        self.sx_sy_top   = [self.nx_ny_top[1],   -self.nx_ny_top[0]]
        self.sx_sy_bottom= [self.nx_ny_bottom[1],-self.nx_ny_bottom[0]]

    def set_normals(self):
        
        xx = self.xx
        yy = self.yy

        A_top_bottom = np.sqrt((xx[:,1:]-xx[:,0:-1])**2+(yy[:,1:]-yy[:,0:-1])**2)
        xx_top_bottom = (xx[:,1:]+xx[:,0:-1])/2
        yy_top_bottom = (yy[:,1:]+yy[:,0:-1])/2
        nxi_top = -(yy[:,1:]-yy[:,0:-1])
        nyi_top = xx[:,1:]-xx[:,0:-1]



        A_left_right = np.sqrt((xx[1:,:]-xx[0:-1,:])**2+(yy[1:,:]-yy[0:-1,:])**2)
        self.A_left = A_left_right[:,0:-1]
        self.A_right = A_left_right[:,1:]
        self.A_top = A_top_bottom[1:,:]
        self.A_bottom = A_top_bottom[0:-1,:]



        xx_left_right = ((xx[1:,:]+xx[0:-1,:])/2)
        yy_left_right = ((yy[1:,:]+yy[0:-1,:])/2)
        nxi_left_right = -(yy[1:,:]-yy[0:-1,:])
        nyi_left_right = (xx[1:,:]-xx[0:-1,:])


        downward_normal = np.array([xx_top_bottom[0:-1,:],yy_top_bottom[0:-1,:],-(nxi_top/A_top_bottom)[0:-1,:],-(nyi_top/A_top_bottom)[0:-1,:]])
        upward_normal = np.array([xx_top_bottom[1:,:],yy_top_bottom[1:,:],(nxi_top/A_top_bottom)[1:,:],(nyi_top/A_top_bottom)[1:,:]])
        leftward_normal = np.array([xx_left_right[:,0:-1],yy_left_right[:,0:-1],(nxi_left_right/A_left_right)[:,0:-1],(nyi_left_right/A_left_right)[:,0:-1]])
        rightward_normal = np.array([xx_left_right[:,1:],yy_left_right[:,1:],-(nxi_left_right/A_left_right)[:,1:],-(nyi_left_right/A_left_right)[:,1:]])


        downward_S = np.array([xx_top_bottom[0:-1,:],yy_top_bottom[0:-1,:],-downward_normal[3],downward_normal[2]])
        upward_S = np.array([xx_top_bottom[1:,:],yy_top_bottom[1:,:],upward_normal[3],-upward_normal[2]])
        leftward_S = np.array([xx_left_right[:,0:-1],yy_left_right[:,0:-1],leftward_normal[3],-leftward_normal[2]])
        rightward_S = np.array([xx_left_right[:,1:],yy_left_right[:,1:],-rightward_normal[3],rightward_normal[2]])

        self.downward_normal = downward_normal
        self.upward_normal = upward_normal
        self.leftward_normal = leftward_normal
        self.rightward_normal = rightward_normal

        self.downward_S = downward_S
        self.upward_S = upward_S
        self.leftward_S = leftward_S
        self.rightward_S = rightward_S

        self.A_top_bottom = A_top_bottom
        self.A_left_right = A_left_right

                

         
    def compute_wave_amplitudes(self,i_plus_half):
        
        bars,deltas = self.compute_doubleBar_values(i_plus_half,compute_deltas=True)
        p,rho,u,ht = bars
        delta_p,delta_rho,delta_u = deltas
        a = np.sqrt(np.maximum(0,(self.gamma-1)*(ht-(u**2)/2)))
        dw1 = delta_rho-(delta_p/self.min_func(a**2))
        dw2 = delta_u+(delta_p/self.min_func(rho*a))
        dw3 = delta_u-(delta_p/self.min_func(rho*a))
        
        return (dw1,dw2,dw3)
    def set_boundary_conditions_OLD(self):

        

        mach = .1*np.ones(self.NJ-1)
        T = self.total_T(self.gamma,mach,self.T0)
        

        

        p_bndry = self.total_p(self.gamma,mach,self.p0)
        rho_bndry = self.total_density(p_bndry,self.R,T)
        U_bndry = self.total_velocity(self.gamma,mach,self.R,T)
        
        
        #print(self.V.shape,U_bndry.shape,self.nx_ny_right[0][:,0].shape,"shapes")
        self.V[1:-1,0,1] = U_bndry*(self.nx_ny_right[0][:,0])
        self.V[1:-1,0,2] = (U_bndry*self.nx_ny_right[1][:,0])
        self.V[1:-1,0,3] = p_bndry
        self.V[1:-1,0,0] = rho_bndry
        

        # top
        s1,s2 = self.sx_sy_bottom[0][0,:],self.sx_sy_bottom[1][0,:]
        n1,n2 = self.nx_ny_top[0][0,:],self.nx_ny_top[1][0,:]
        v2x = self.V[1,1:-1,1]
        v2y = self.V[1,1:-1,2]

        v1 = self.compute_slip_walls(s1,s2,n1,n2,v2x,v2y)
        self.V[0,1:-1,1] = v1[0]
        self.V[0,1:-1,2] = v1[1]

        # Bottom BC
        s1,s2 = self.sx_sy_bottom[0][-1,:],self.sx_sy_bottom[1][-1,:]
        n1,n2 = self.nx_ny_bottom[0][-1,:],self.nx_ny_bottom[1][-1,:]
        v2x = self.V[-2,1:-1,1]
        v2y = self.V[-2,1:-1,2]

        v1 = self.compute_slip_walls(s1,s2,n1,n2,v2x,v2y)
        self.V[-1,1:-1,1] = v1[0]
        self.V[-1,1:-1,2] = v1[1]

        #self.V[1:-1,-1,0] = self.V[1:-1,-2,0]
        #self.V[1:-1,-1,0] = self.V[1:-1,-2,0]


        #U_top = self.V[1:-1,-1,1]*self.nx_ny_top[0][:,-1] +self.V[1:-1,-1,2]*self.nx_ny_top[1][:,-1]
        #U_bottom = self.V[1:-1,0,1]*self.nx_ny_bottom[0][:,0] +self.V[1:-1,0,2]*self.nx_ny_bottom[1][:,0]
        
        
        #T_top = self.V[1:-1,-2,3]/(self.V[1:-1,-2,0]*self.R)
        #T_bottom = self.V[1:-1,1,3]/(self.V[1:-1,1,0] *self.R )


        #self.V[-1,:,3] = self.V[-2,:,3]
        #self.V[0,:,3] = self.V[1,:,3]
        #self.V[-1,:,0] = self.V[-2,:,0]
        #self.V[0,:,0] = self.V[1,:,0]
        #self.V[-1,:,0] = self.V[1:,-2,0]
        #self.V[0,:,0] = self.V[1:,1,0]

        #self.V[0,0,:] = self.V[1,0,:]
        #self.V[-1,0,:] = self.V[-2,0,:]
        #self.V[0,0,2] = self.V[1,0,2]
        #self.V[-1,0,2] = self.V[-2,0,2]
        #self.V[-1,0,:] = self.V[-1,1,:]
        #self.V[:,-1,3] = self.V[:,-2,3]
        #self.V[:,-1,0] = self.V[:,-2,0]
        self.U,self.F,self.G = self.primitive_to_conserved(self.V)
    
    
    
    
    
    
    def set_boundary_conditions_NORMALS(self):
        
        mach = .1*np.ones(self.NJ-1)
        T = self.total_T(self.gamma,mach,self.T0)
        
        p_bndry = self.total_p(self.gamma,mach,self.p0)
        rho_bndry = self.total_density(p_bndry,self.R,T)
        U_bndry = self.total_velocity(self.gamma,mach,self.R,T)
        
        u = self.total_velocity(self.gamma,mach,self.R,T)

        #print(self.V.shape,U_bndry.shape,self.nx_ny_right[0][:,0].shape,"shapes")
        self.V[1:-1,0,1] = (U_bndry*self.upward_S[2,:,0])
        self.V[1:-1,0,2] = (U_bndry*self.upward_S[3,:,0])
        self.V[1:-1,0,3] = p_bndry
        self.V[1:-1,0,0] = rho_bndry

        self.V[0,0,:] = self.V[1,0,:]
        self.V[-1,0,:] = self.V[-2,0,:]
        
        # Top
        v2x = self.V[-2,1:-1,1]
        v2y = self.V[-2,1:-1,2]
        u_top,v_top = self.compute_slip_walls(self.upward_S[2,-2,:],self.upward_S[3,-2,:],self.upward_normal[2,-2,:],\
                              self.upward_normal[3,-2,:],v2x,v2y)
        self.V[-1,1:-1,1] = u_top
        self.V[-1,1:-1,2] = v_top

        self.V[-1,0,1] = self.V[-1,1,1]
        self.V[-1,-1,1] = self.V[-1,-2,1]
        


       # Bottom
        v2x = self.V[1,1:-1,1]
        v2y = self.V[1,1:-1,2]
        u_bottom,v_bottom = self.compute_slip_walls(self.downward_S[2,0,:],self.downward_S[3,0,:],self.downward_normal[2,0,:],\
                              self.downward_normal[3,0,:],v2x,v2y)
        self.V[0,1:-1,1] = u_bottom
        self.V[0,1:-1,2] = v_bottom

        self.V[0,1:-1,3]= self.V[1,1:-1,3]
        self.V[-1,1:-1,3]= self.V[-2,1:-1,3]
        

        
        T = self.V[1,1:-1,3]/(self.V[1,1:-1,0]*self.R)
        rho0 = self.total_density(self.p0,self.R,T)
        self.V[0,1:-1,0]= rho0

        T = self.V[-2,1:-1,3]/(self.V[-2,1:-1,0]*self.R)
        rho1 = self.total_density(self.p0,self.R,T)
        self.V[-1,1:-1,0]=rho1

        self.U,_,_ = self.primitive_to_conserved(self.V)
        
    def set_boundary_conditions(self):
        
        mach = .1*np.ones(self.NJ-1)
        T = self.total_T(self.gamma,mach,self.T0)
        
        p_bndry = self.total_p(self.gamma,mach,self.p0)
        rho_bndry = self.total_density(p_bndry,self.R,T)
        U_bndry = self.total_velocity(self.gamma,mach,self.R,T)
        

        #print(self.V.shape,U_bndry.shape,self.nx_ny_right[0][:,0].shape,"shapes")
        self.V[1:-1,0,1] = (U_bndry*self.upward_S[2,:,0])
        self.V[1:-1,0,2] = (U_bndry*self.upward_S[3,:,0])
        self.V[1:-1,0,3] = p_bndry
        self.V[1:-1,0,0] = rho_bndry

        self.V[0,0,:] = self.V[1,0,:]
        self.V[-1,0,:] = self.V[-2,0,:]





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
        
    def set_conserved_variables(self): # Set F_{i+1/2} and F_{i-1/2}
        T = self.V[:,:,3]/(self.V[:,:,0]*self.R)
        et = (self.R/(self.gamma-1))*T+.5*(self.V[:,:,1]**2+self.V[:,:,2]**2)
        ht = (self.gamma*self.R/(self.gamma-1))*T+.5*(self.V[:,:,1]**2+self.V[:,:,2]**2)


        self.U = np.zeros((self.NI+1,self.NJ+1,3)) # Number of faces
        self.F = np.zeros((self.NI+1,self.NJ+1,3))
        self.G = np.zeros((self.NI+1,self.NJ+1,3))
        
       
        
        self.U = np.array([
        self.V[:,:, 0],  # Density
        self.V[:,:, 0] * self.V[:,:, 1],  # Momentum (rho * u)
        self.V[:,:, 0] * self.V[:,:, 2],
        self.V[:,:, 0] * et ]).T

       

    def Area(self):
        return .2+.4*(1+np.sin(np.pi*(self.x-.5)))
    def Darea(self):
        x = (self.x[1:]+self.x[0:-1])/2
        return 0.4 * np.pi * np.cos(np.pi * (x - 0.5))

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
            return p_compute,u_compute,rho_compute,np.array(p_exact),np.array(u_exact),np.array(rho_exact),convergence_history
        else:
            Mass_compute = self.U[1:-1,0]
            Momentum_compute = self.U[1:-1,1]
            Energy_compute = self.U[1:-1,2]
            U,_ = self.primitive_to_conserved(np.array([rho_exact,u_exact,p_exact]).T)
            
            Mass_exact = U[:,0]
            Momentum_exact = U[:,1]
            Energy_exact = U[:,2]

            return Mass_compute,Momentum_compute,Energy_compute,Mass_exact,Momentum_exact,Energy_exact,convergence_history




            



    def c_plus_minus(self,alpha_plus_minus,beta_LR,mach_LR,M_plus_minus):
        return alpha_plus_minus*(1+beta_LR)*mach_LR-beta_LR*M_plus_minus


    def get_normal_directions(self,direction):
        if direction=="right":
            nx,ny = self.rightward_normal[2],self.rightward_normal[3]
        elif direction=="left":
            nx,ny = self.leftward_normal[2],self.leftward_normal[3]
        elif direction=="down":
            nx,ny = self.downward_normal[2],self.downward_normal[3]
        elif direction == "up":
            nx,ny = self.upward_normal[2],self.upward_normal[3]
        
        return nx,ny
    def FG_convective(self,direction):
        shift = self.FLUXL_FLUXR_FUNC(direction=direction)
        
        nx,ny = self.get_normal_directions(direction)
        V_L,V_R =shift(self.V)


        
        rho_L,rho_R =V_L[:,:,0],V_R[:,:,0]
        u_L,u_R =V_L[:,:,1],V_R[:,:,1]
        v_L,v_R =V_L[:,:,2],V_R[:,:,2]
        p_L,p_R =V_L[:,:,3],V_R[:,:,3]
        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)
 
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        #print(u_L.shape,nx.shape,"u_l,nx",direction)

        vel_L = u_L*nx+v_L*ny
        vel_R = u_R*nx+v_R*ny
    
        M_L,M_R = (vel_L/a_L,vel_R/a_R)
        
        alpha_plus = .5*(1+np.sign(M_L))
        alpha_minus = .5*(1-np.sign(M_R))
        beta_L = -np.maximum(0,1-np.floor(np.abs(M_L)))
        beta_R = -np.maximum(0,1-np.floor(np.abs(M_R)))

        M_plus = .25*(M_L+1)**2
        M_minus = -.25*(M_R-1)**2

        
        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * (u_L**2+v_L**2)
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * (u_R**2+v_R**2)
       
        temp_L = rho_L*a_L*self.c_plus_minus(alpha_plus,beta_L,M_L,M_plus)
        
        temp_R = rho_R*a_R*self.c_plus_minus(alpha_minus,beta_R,M_R,M_minus)
        GFC_i_half = np.array([temp_L,u_L*temp_L,v_L*temp_L,ht_L*temp_L])+np.array([temp_R,u_R*temp_R,v_R*temp_R,ht_R*temp_R])
        return np.einsum("ijk->jki",GFC_i_half)
    def FLUXL_FLUXR_FUNC(self,direction:str):
        
        if direction=="right":
            shift_indx = 0
        elif direction=="left":
            shift_indx = -1
        elif direction=="down":
            shift_indx = -1
        elif direction == "up":
            shift_indx = 0
        
        def shift_LR(F): 
        
            F_temp = np.zeros((F.shape[0]-2,F.shape[1]+2,4))
            F_temp[:,1:-1,:] = F[1:-1,:,:]
            F_temp[:,0,:],F_temp[:,-1,:] = F_temp[:,1,:],F_temp[:,-2,:] #self.extrapolate1(F_temp,extrapolate_y=True)
            
            
            F = F_temp
            
            p1 = self.psi_plus(F,-1+shift_indx,F_flux=True)
            p2 = self.psi_minus(F,shift_indx,F_flux=True)
            p3 = self.psi_minus(F,shift_indx+1,F_flux=True)
            p4 = self.psi_plus(F,shift_indx,F_flux=True)
            #print(p1.shape,p2.shape,p3.shape,p4.shape,F[:,2+shift_indx:-2+shift_indx].shape)
            epsilon = self.upwind_order
            FL = F[:,2+shift_indx:-2+shift_indx]+(epsilon/4)*((1-self.kappa)*(p1*(F[:,2+shift_indx:-2+shift_indx]-F[:,1+shift_indx:-3+shift_indx]))+\
                                                                        (1+self.kappa)*p2*(F[:,3+shift_indx:-1+shift_indx]-F[:,2+shift_indx:-2+shift_indx]))
            FR = F[:,3+shift_indx:-1+shift_indx]-(epsilon/4)*((1-self.kappa)*(p3*(F[:,4+shift_indx:self.shift_func(shift_indx)]-F[:,3+shift_indx:-1+shift_indx]))\
                                                                       +(1+self.kappa)*p4*(F[:,3+shift_indx:-1+shift_indx]-F[:,2+shift_indx:-2+shift_indx]))
            return FL,FR
        
        def shift_UD(F): 
        
            G_temp = np.zeros((F.shape[0]+2,F.shape[1]-2,4))
            G_temp[1:-1,:,:] = F[:,1:-1,:]
            G_temp[0,:,:],G_temp[-1,:,:] =G_temp[1,:,:],G_temp[-2,:,:] #self.extrapolate1(G_temp,extrapolate_y = False)
            
            
            G = G_temp
            
            p1 = self.psi_plus(G,-1+shift_indx,F_flux=False)
            p2 = self.psi_minus(G,shift_indx,F_flux=False)
            p3 = self.psi_minus(G,shift_indx+1,F_flux=False)
            p4 = self.psi_plus(G,shift_indx,F_flux=False)
            
            epsilon = self.upwind_order
            GL = G[2+shift_indx:-2+shift_indx,:]+(epsilon/4)*((1-self.kappa)*(p1*(G[2+shift_indx:-2+shift_indx,:]-G[1+shift_indx:-3+shift_indx,:]))+\
                                                                       (1+self.kappa)*p2*(G[3+shift_indx:-1+shift_indx,:]-G[2+shift_indx:-2+shift_indx,:]))
            GR = G[3+shift_indx:-1+shift_indx,:]-(epsilon/4)*((1-self.kappa)*(p3*(G[4+shift_indx:self.shift_func(shift_indx),:]-G[3+shift_indx:-1+shift_indx,:]))\
                                                                       +(1+self.kappa)*p4*(G[3+shift_indx:-1+shift_indx,:]-G[2+shift_indx:-2+shift_indx,:]))
            return GL,GR
        
        

        
        if direction=="left" or direction=="right":  
            return shift_LR
        return shift_UD
    

    def shift_func(self,shift_indx):
        if(shift_indx==0):
            return  None
        return shift_indx
    def psi_minus(self,Flux:np.ndarray,shift_indx:int,F_flux:bool)-> float:
        if self.flux_limiter_scheme==0:
            return 1
        if not F_flux:
            NUM =  (Flux[2+shift_indx:self.shift_func(-2+shift_indx),:]-Flux[1+shift_indx:-3+shift_indx,:])
            DEN = self.min_func(Flux[3+shift_indx:self.shift_func(-1+shift_indx),:]-Flux[2+shift_indx:self.shift_func(-2+shift_indx),:])
        else:
            NUM =  (Flux[:,2+shift_indx:self.shift_func(-2+shift_indx)]-Flux[:,1+shift_indx:-3+shift_indx])
            DEN = self.min_func(Flux[:,3+shift_indx:self.shift_func(-1+shift_indx)]-Flux[:,2+shift_indx:self.shift_func(-2+shift_indx)])
        r_minus =NUM/DEN
        r = r_minus
        if self.flux_limiter_scheme==1:
            return (r+np.abs(r))/self.min_func(1+r)
        elif self.flux_limiter_scheme==2:
            return (r**2+r)/self.min_func(1+r**2)
        return 0.0

    def psi_plus(self,Flux:np.ndarray,shift_indx:int,F_flux:bool)-> float:
        if self.flux_limiter_scheme==0:
            return 1
        if F_flux:
            NUM = (Flux[:,4+shift_indx:self.shift_func(shift_indx)]-Flux[:,3+shift_indx:self.shift_func(-1+shift_indx)])
            DEN = self.min_func(Flux[:,3+shift_indx:self.shift_func(-1+shift_indx)]-Flux[:,2+shift_indx:self.shift_func(-2+shift_indx)])
        else:
            NUM = (Flux[4+shift_indx:self.shift_func(shift_indx),:]-Flux[3+shift_indx:self.shift_func(-1+shift_indx),:])
            DEN = self.min_func(Flux[3+shift_indx:self.shift_func(-1+shift_indx),:]-Flux[2+shift_indx:self.shift_func(-2+shift_indx),:])
        r_plus = NUM/DEN
        r = r_plus
        
        if self.flux_limiter_scheme==1:
            return (r+np.abs(r))/self.min_func(1+r)
        elif self.flux_limiter_scheme ==2:
            return (r**2+r)/self.min_func(1+r**2)
        return 0.0
        

    def min_func(self,s):
        indeces = np.where(s==0)[0]
        s[indeces] = self.epsilon
        return np.sign(s)*np.maximum(self.epsilon,np.abs(s))
    def FG_pressure_flux(self,direction):
        nx,ny = self.get_normal_directions(direction)
        
        shift = self.FLUXL_FLUXR_FUNC(direction)
        
        
        
        V_L,V_R =shift(self.V)
        rho_L,rho_R =V_L[:,:,0],V_R[:,:,0]
        u_L,u_R =V_L[:,:,1],V_R[:,:,1]
        v_L,v_R =V_L[:,:,2],V_R[:,:,2]
        p_L,p_R =V_L[:,:,3],V_R[:,:,3]

        if np.any(rho_L<=0):
            rho_L = np.maximum(rho_L, self.epsilon)
        if np.any(rho_R<=0):
            rho_R = np.maximum(rho_R, self.epsilon)

        
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        
        #U_L = u_L*nx[0:-1,:]+v_L*ny[0:-1,:]
        #U_R = u_R*nx[1:,:]*+v_R*ny[1:,:]

        #print(ny)


        U_L = u_L*nx+v_L*ny
        U_R = u_R*nx+v_R*ny

    


        M_L,M_R = (U_L/a_L,U_R/a_R)
        
        #print(U_L,U_R,"nx",direction)
        alpha_plus = .5*(1+np.sign(M_L))
        alpha_minus = .5*(1-np.sign(M_R))
        beta_L = -np.maximum(0,1-np.floor(np.abs(M_L)))
        beta_R = -np.maximum(0,1-np.floor(np.abs(M_R)))
        #print(beta_L-beta_R,"beta")
        M_plus = .25*(M_L+1)**2
        M_minus = -.25*(M_R-1)**2

        p_doubleBar_plus = M_plus*(-M_L+2)
        p_doubleBar_minus = M_minus*(-M_R-2)

        D_plus = alpha_plus*(1+beta_L)-beta_L*p_doubleBar_plus
        D_minus = alpha_minus*(1+beta_R)-beta_R*p_doubleBar_minus
        #print(D_minus-D_plus)
        temp_zero = np.zeros_like(p_L)
        #print(p_L+p_R,direction)
        G =  np.array([temp_zero,D_plus*nx*p_L,D_plus*ny*p_L,temp_zero])+np.array([temp_zero,D_minus*nx*p_R,D_minus*ny*p_R,temp_zero])

        return np.einsum("ijk->jki",G)
        
        


    def compute_timestep(self):
        delta_x = np.abs(self.x[1]-self.x[0])
        self.delta_t =self.CFL*delta_x/(np.abs(self.V[:,1]) + np.sqrt(np.abs(self.gamma*self.V[:,2]/self.V[:,0])))
    def compute_all_areas(self,visualize=False):
        def triangle_area_2d(a, b, c):
            u = np.array(b) - np.array(a)
            v = np.array(c) - np.array(a)
            return 0.5 * abs(u[0]*v[1] - u[1]*v[0])
        xx =self.xx
        yy = self.yy
        p1 = [xx[0:-1,0:-1],yy[0:-1,0:-1]]
        p2 = [xx[1:,0:-1],yy[1:,0:-1]] 
        p3 = [xx[0:-1,1:],yy[0:-1,1:]] 
        p4 = [xx[1:,1:],yy[1:,1:]]
        areas = triangle_area_2d(p1,p2,p3 ) + triangle_area_2d(p1, p3, p4)

        if visualize:
            midpoint_tempx = (xx[1:,:]+xx[0:-1,:])/2
            midpointxx = (midpoint_tempx[:,1:]+midpoint_tempx[:,0:-1])/2
            midpoint_tempy = (yy[1:,:]+yy[0:-1,:])/2
            midpointyy = (midpoint_tempy[:,1:]+midpoint_tempy[:,0:-1])/2
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=areas.flatten())
            plt.colorbar()
            plt.show()
        self.AREA = areas

    def F_normal_OLD(self):
        rho = self.V[:,:,0]
        u = self.V[:,:,1]
        v = self.V[:,:,2]
        p = self.V[:,:,3]
        ht = (self.gamma / (self.gamma - 1)) * (p / rho) + 0.5 * ((u)**2+v**2)
                # LEFT fluxes (at i-1/2 face)
        U_left = self.nx_ny_left[0][:, 0:-1] * u[1:-1, 0:-2] + self.nx_ny_left[1][:, 0:-1] * v[1:-1, 0:-2]
        F_left = np.array([
            rho[1:-1, 0:-2] * U_left,
            u[1:-1, 0:-2] * U_left + self.nx_ny_left[0][:, 0:-1] * p[1:-1, 0:-2],
            v[1:-1, 0:-2] * U_left + self.nx_ny_left[1][:, 0:-1] * p[1:-1, 0:-2],
            rho[1:-1, 0:-2] * U_left * ht[1:-1, 0:-2]
        ]).T

        # RIGHT fluxes (i+1/2)
        U_right = self.nx_ny_right[0][:, 1:] * u[1:-1, 2:] + self.nx_ny_right[1][:, 1:] * v[1:-1, 2:]
        F_right = np.array([
            rho[1:-1, 2:] * U_right,
            u[1:-1, 2:] * U_right + self.nx_ny_right[0][:, 1:] * p[1:-1, 2:],
            v[1:-1, 2:] * U_right + self.nx_ny_right[1][:, 1:] * p[1:-1, 2:],
            rho[1:-1, 2:] * U_right * ht[1:-1, 2:]
        ]).T

        U_bottom = (self.nx_ny_bottom[0].T)[:,0:-1] * u[0:-2, 1:-1] + (self.nx_ny_bottom[1].T)[:,0:-1] * v[0:-2, 1:-1]
        F_bottom = np.array([
            rho[0:-2, 1:-1] * U_bottom,
            u[0:-2, 1:-1] * U_bottom + (self.nx_ny_bottom[0].T)[:,0:-1] * p[0:-2, 1:-1],
            v[0:-2, 1:-1] * U_bottom + (self.nx_ny_bottom[1].T)[:,0:-1] * p[0:-2, 1:-1],
            rho[0:-2, 1:-1] * U_bottom * ht[0:-2, 1:-1]
        ]).T

        # TOP fluxes (j+1/2)
        
        U_top = (self.nx_ny_top[0].T[:,1:]) * u[2:,1:-1] + self.nx_ny_top[0][1:,:] * v[2:,1:-1]
        F_top = np.array([
            rho[2:, 1:-1] * U_top,
            u[2:, 1:-1] * U_top + (self.nx_ny_top[0].T[:,1:]) * p[2:, 1:-1],
            v[2:, 1:-1] * U_top + (self.nx_ny_top[1].T[:,1:]) * p[2:, 1:-1],
            rho[2:, 1:-1] * U_top * ht[2:, 1:-1]
        ]).T

        return np.einsum("ijk->jki",F_left),np.einsum("ijk->jki",F_right),np.einsum("ijk->jki",F_bottom),np.einsum("ijk->jki",F_top)
    



    def F_normal(self):
        rho = self.V[:,:,0]
        u = self.V[:,:,1]
        v = self.V[:,:,2]
        p = self.V[:,:,3]
        ht = (self.gamma / (self.gamma - 1)) * (p/rho) + 0.5 * ((u)**2+v**2)
                # LEFT fluxes (at i-1/2 face)
        U_left = (self.leftward_normal[2] * u[1:-1, 0:-2] + self.leftward_normal[3] * v[1:-1, 0:-2])
        F_left = np.array([
            rho[1:-1, 0:-2] * U_left,
            u[1:-1, 0:-2] * U_left + self.leftward_normal[2] * p[1:-1, 0:-2],
            v[1:-1, 0:-2] * U_left + self.leftward_normal[3] * p[1:-1, 0:-2],
            rho[1:-1, 0:-2] * U_left * ht[1:-1, 0:-2]
        ])

        # RGHT fluxes (i+1/2)
        U_right = (self.rightward_normal[2] * u[1:-1, 2:] + self.rightward_normal[3] * v[1:-1, 2:])
        F_right = np.array([
            rho[1:-1, 2:] * U_right,
            u[1:-1, 2:] * U_right + self.rightward_normal[2] * p[1:-1, 2:],
            v[1:-1, 2:] * U_right + self.rightward_normal[3] * p[1:-1, 2:],
            rho[1:-1, 2:] * U_right * ht[1:-1, 2:]
        ])
        
        U_bottom = (self.downward_normal[2] * u[0:-2, 1:-1] + self.downward_normal[3] * v[0:-2, 1:-1])
        F_bottom = np.array([
            rho[0:-2, 1:-1] * U_bottom,
            u[0:-2, 1:-1] * U_bottom + self.downward_normal[2] * p[0:-2, 1:-1],
            v[0:-2, 1:-1] * U_bottom + self.downward_normal[3] * p[0:-2, 1:-1],
            rho[0:-2, 1:-1] * U_bottom * ht[0:-2, 1:-1]
        ])

        # TOP fluxes (j+1/2)
        
        U_top = (self.upward_normal[2] * u[2:,1:-1] + self.upward_normal[3] * v[2:,1:-1])
        F_top = np.array([
            rho[2:, 1:-1] * U_top,
            u[2:, 1:-1] * U_top + self.upward_normal[2] * p[2:, 1:-1],
            v[2:, 1:-1] * U_top + self.upward_normal[3] * p[2:, 1:-1],
            rho[2:, 1:-1] * U_top * ht[2:, 1:-1]
        ])

        return np.einsum("ijk->jki",F_left),np.einsum("ijk->jki",F_right),np.einsum("ijk->jki",F_bottom),np.einsum("ijk->jki",F_top)
    def iteration_step(self,return_error=True):
        

        residual = np.zeros((self.NI-1,self.NJ-1,3))
        #self.compute_timestep()
        #if not self.local_timestep:
        #    self.delta_t = np.min(self.delta_t)
        
        self.delta_t = .000001
     
        #F_left = 
        Volume =self.AREA 
        self.F_left = (self.FG_convective(direction = "left")+self.FG_pressure_flux(direction="left"))
        self.F_top = (self.FG_convective(direction = "up")+self.FG_pressure_flux(direction="up"))
        self.F_right = (self.FG_convective(direction="right")+self.FG_pressure_flux(direction="right"))
        self.F_bottom = (self.FG_convective(direction="down")+self.FG_pressure_flux(direction="down"))
        
        
        #Fi_minus_1_2,Fi_plus_1_2,Fj_minus_1_2,Fj_plus_1_2 = self.F_normal()
        self.F_left,self.F_right,self.F_top,self.F_bottom = self.F_normal()
        A_left = (self.A_left)[:,:,np.newaxis]@(np.ones((1,4)))
        A_right = (self.A_right)[:,:,np.newaxis]@(np.ones((1,4)))
        A_top = (self.A_top)[:,:,np.newaxis]@(np.ones((1,4)))
        A_bottom = (self.A_bottom)[:,:,np.newaxis]@(np.ones((1,4)))
        
        #residual =  (Fi_plus_1_2*Ai_plus_1_2+Fi_minus_1_2*Ai_minus_1_2)+ \
        #            (Fj_plus_1_2*Aj_plus_1_2+Fj_minus_1_2*Aj_minus_1_2) #-self.S*deltax

    
        residual = self.F_left*A_left + self.F_right*A_right+self.F_top*A_top+self.F_bottom*A_bottom
        Volume = (Volume[:,:,np.newaxis]@np.ones((1,4)))
        self.residual = residual
        #if self.local_timestep:       
        #    self.U[1:-1] = self.U[1:-1]-(residual*np.tile(self.delta_t[1:-1],(3,1)).T/np.tile(Volume,(3,1)).T)
        #else:
        
        self.U[1:-1,1:-1] = self.U[1:-1,1:-1]-(residual*self.delta_t/Volume)
        # Update all 

        
        #self.U[0,:,:],self.U[-1,:,:] = self.extrapolate1(self.U)
        self.U[0,:,:] = self.U[1,:,:]
        self.U[-1,:,:] = self.U[-2,:,:]
        self.U[:,0,:] = self.U[:,1,:]
        self.U[:,-1,:] = self.U[:,-2,:]
        #self.U[]
        #_,self.U[:,-1,:] = self.extrapolate1(self.U,extrapolate_y=True)

        self.V = self.conserved_to_primitive(self.U)
        ##self.update_source()
        return residual
        if return_error:
            return np.linalg.norm(residual,axis=0)
        
        

    def update_source(self):
        #deltax = np.abs(self.x[2]-self.x[1])
        #self.S[:,1] = self.V[1:-1,2]*(self.A[1:]-self.A[0:-1])/deltax
        return



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