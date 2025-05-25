import numpy as np
from DataStructures import DataStructures


class Upwind(object):
    def __init__(self,Data,kappa,upwind_order,dampingscheme,flux_limiter):
        self.flux_limiter_scheme = flux_limiter 
        self.kappa = kappa
        self.upwind_order = upwind_order
        self.damping_scheme = dampingscheme
        self.Data = Data
        self.epsilon = self.Data.epsilon
        self.gamma = 1.4
    


    def c_plus_minus(self,alpha_plus_minus,beta_LR,mach_LR,M_plus_minus):
        return alpha_plus_minus*(1+beta_LR)*mach_LR-beta_LR*M_plus_minus


    def get_normal_directions(self,direction):
         
        if direction=="right":
            nx,ny = self.Data.rightward_normal[self.Data.unormal],self.Data.rightward_normal[self.Data.vnormal]
        elif direction=="left":
            nx,ny = self.Data.leftward_normal[self.Data.unormal],self.Data.leftward_normal[self.Data.vnormal]
        elif direction=="down":
            nx,ny = self.Data.downward_normal[self.Data.unormal],self.Data.downward_normal[self.Data.vnormal]
        elif direction == "up":
            nx,ny = self.Data.upward_normal[self.Data.unormal],self.Data.upward_normal[self.Data.vnormal]
        return nx,ny

    def FG_convective(self,direction):
        shift = self.FLUXL_FLUXR_FUNC(direction=direction)
        
        nx,ny = self.get_normal_directions(direction)
        V_L,V_R =shift(self.Data.V)


       


        
        rho_L,rho_R =V_L[self.Data.rho_idx],V_R[self.Data.rho_idx]
        u_L,u_R =V_L[self.Data.u_idx],V_R[self.Data.u_idx]
        v_L,v_R =V_L[self.Data.v_idx],V_R[self.Data.v_idx]
        p_L,p_R =V_L[self.Data.p_idx],V_R[self.Data.p_idx]
        rho_L = np.maximum(self.epsilon,rho_L)
        rho_R = np.maximum(self.epsilon,rho_R)
        p_L = np.maximum(self.epsilon,p_L)
        p_R = np.maximum(self.epsilon,p_R)

 
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        #print(u_L.shape,nx.shape,"u_l,nx",direction)
        # print(direction,nx.shape,u_L.shape)
        vel_L = u_L*nx+v_L*ny
        vel_R = u_R*nx+v_R*ny

        M_L,M_R = (vel_L/a_L,vel_R/a_R)
        if direction=="up" or direction=="down":
            alpha_plus = .5*(1+np.sign(v_L))
            alpha_minus = .5*(1-np.sign(v_R))
        else:
            alpha_plus = .5*(1+np.sign(u_L))
            alpha_minus = .5*(1-np.sign(u_R))
        #alpha_plus = .5*(1+np.sign(vel_L))
        #alpha_minus = .5*(1-np.sign(vel_R))

        beta_L = -np.maximum(0,1-np.floor(np.abs(M_L)))
        beta_R = -np.maximum(0,1-np.floor(np.abs(M_R)))
        
        M_plus = .25*(M_L+1)**2
        M_minus = -.25*(M_R-1)**2

        
        ht_L = (self.gamma / (self.gamma - 1)) * (p_L / rho_L) + 0.5 * (u_L**2+v_L**2)
        ht_R = (self.gamma / (self.gamma - 1)) * (p_R / rho_R) + 0.5 * (u_R**2+v_R**2)
       
        temp_L = rho_L*a_L*self.c_plus_minus(alpha_plus,beta_L,M_L,M_plus)
        
        temp_R = rho_R*a_R*self.c_plus_minus(alpha_minus,beta_R,M_R,M_minus)
        GFC_i_half = np.array([temp_L,u_L*temp_L,v_L*temp_L,ht_L*temp_L])+np.array([temp_R,u_R*temp_R,v_R*temp_R,ht_R*temp_R])
        return GFC_i_half
    

    
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
        
            F_temp = np.zeros((4,F.shape[1],F.shape[2]+4))
            F_temp[:,:,2:-2] = F
            
            F_temp[:,:,1] = self.Data.inflow
            F_temp[:,:,0] = 2*F_temp[:,:,1]-F_temp[:,:,2]
            F_temp[:,:,-2] = 2*F_temp[:,:,-3]-F_temp[:,:,-4]
            F_temp[:,:,-1] = 2*F_temp[:,:,-2]-F_temp[:,:,-3]
            
            
            F = F_temp
            
            p1 = self.psi_plus(F,-1+shift_indx,F_flux=True)
            p2 = self.psi_minus(F,shift_indx,F_flux=True)
            p3 = self.psi_minus(F,shift_indx+1,F_flux=True)
            p4 = self.psi_plus(F,shift_indx,F_flux=True)
            
            
            epsilon = self.upwind_order
            FL = F[:,:,2+shift_indx:-2+shift_indx]+(epsilon/4)*((1-self.kappa)*(p1*(F[:,:,2+shift_indx:-2+shift_indx]-F[:,:,1+shift_indx:-3+shift_indx]))+\
                                                                        (1+self.kappa)*p2*(F[:,:,3+shift_indx:-1+shift_indx]-F[:,:,2+shift_indx:-2+shift_indx]))
            FR = F[:,:,3+shift_indx:-1+shift_indx]-(epsilon/4)*((1-self.kappa)*(p3*(F[:,:,4+shift_indx:self.shift_func(shift_indx)]-F[:,:,3+shift_indx:-1+shift_indx]))\
                                                                       +(1+self.kappa)*p4*(F[:,:,3+shift_indx:-1+shift_indx]-F[:,:,2+shift_indx:-2+shift_indx]))
            return FL,FR
        
        def shift_UD(F): 
        
            G_temp = np.zeros((4,F.shape[1]+4,F.shape[2]))
            G_temp[:,2:-2,:] = F
            G_temp[:,1,:] = 2*G_temp[:,2,:]-G_temp[:,3,:]
            G_temp[:,-2,:] = 2*G_temp[:,-3,:]-G_temp[:,-4,:]
            
            
            G = G_temp
            
            #p1 = self.psi_plus(G,-1+shift_indx,F_flux=False)
            #p2 = self.psi_minus(G,shift_indx,F_flux=False)
            #p3 = self.psi_minus(G,shift_indx+1,F_flux=False)
            #p4 = self.psi_plus(G,shift_indx,F_flux=False)
            
            #epsilon = self.upwind_order
            GL = G[:,2+shift_indx:-2+shift_indx,:]#+(epsilon/4)*((1self.kappa)*(p1*(G[2+shift_indx:-2+shift_indx,:]-G[1+shift_indx:-3+shift_indx,:]))+\
                                                #                       (1+self.kappa)*p2*(G[3+shift_indx:-1+shift_indx,:]-G[2+shift_indx:-2+shift_indx,:]))
            GR = G[:,3+shift_indx:-1+shift_indx,:]#-(epsilon/4)*((1self.kappa)*(p3*(G[4+shift_indx:self.shift_func(shift_indx),:]-G[3+shift_indx:-1+shift_indx,:]))\
                                                #                       +(1+self.kappa)*p4*(G[3+shift_indx:-1+shift_indx,:]-G[2+shift_indx:-2+shift_indx,:]))
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
        
        
        
        V_L,V_R =shift(self.Data.V)
        rho_L,rho_R =V_L[self.Data.rho_idx],V_R[self.Data.rho_idx]
        u_L,u_R =V_L[self.Data.u_idx],V_R[self.Data.u_idx]
        v_L,v_R =V_L[self.Data.v_idx],V_R[self.Data.v_idx]
        p_L,p_R =V_L[self.Data.p_idx],V_R[self.Data.p_idx]

        rho_L = np.maximum(self.epsilon,rho_L)
        rho_R = np.maximum(self.epsilon,rho_R)
        p_L = np.maximum(self.epsilon,p_L)
        p_R = np.maximum(self.epsilon,p_R)

        
        a_L = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_L/rho_L))))
        a_R = np.sqrt(np.maximum(self.epsilon,(self.gamma*(p_R/rho_R))))
        
        #U_L = u_L*nx[0:-1,:]+v_L*ny[0:-1,:]
        #U_R = u_R*nx[1:,:]*+v_R*ny[1:,:]

        #print(ny)


        U_L = u_L*nx+v_L*ny
        U_R = u_R*nx+v_R*ny

    


        M_L,M_R = (U_L/a_L,U_R/a_R)
        
        #print(U_L,U_R,"nx",direction)
        if direction=="up" or direction=="down":
            alpha_plus = .5*(1+np.sign(v_L))
            alpha_minus = .5*(1-np.sign(v_R))
        else:
            alpha_plus = .5*(1+np.sign(u_L))
            alpha_minus = .5*(1-np.sign(u_R))
        #alpha_plus = .5*(1+np.sign(U_L))
        #alpha_minus = .5*(1-np.sign(U_R))

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

        return G

    
        

    def  GetFluxes(self,method = "Leer"):
        if method=="Leer":
            bc_top,bc_bottom = self.Data.bc_multiplier
            FL_conv = self.FG_convective(direction = "left")
            FT_conv = bc_top*self.FG_convective(direction = "up")
            FR_conv = self.FG_convective(direction="right")
            FD_conv = bc_bottom*self.FG_convective(direction="down")



            F_left = (FL_conv+self.FG_pressure_flux(direction="left"))
            F_top = (FT_conv+self.FG_pressure_flux(direction="up"))
            F_right = (FR_conv+self.FG_pressure_flux(direction="right"))
            F_bottom = (FD_conv+self.FG_pressure_flux(direction="down"))
            
    
        if method=="Roe":
            F_left = self.compute_roe_flux("left")
            F_right = self.compute_roe_flux("right")
            F_top = self.compute_roe_flux("up")
            F_bottom = self.compute_roe_flux("down")

        return F_left,F_right,F_top,F_bottom 
    def compute_doubleBar_values(self,direction,compute_deltas = False):
        shift = self.FLUXL_FLUXR_FUNC(direction)
        V_L,V_R =shift(self.Data.V)
        rho_L,rho_R =V_L[self.Data.rho_idx],V_R[self.Data.rho_idx]
        u_L,u_R =V_L[self.Data.u_idx],V_R[self.Data.u_idx]
        v_L,v_R =V_L[self.Data.v_idx],V_R[self.Data.v_idx]
        p_L,p_R =V_L[self.Data.p_idx],V_R[self.Data.p_idx]

        rho_L = np.maximum(self.epsilon,rho_L)
        rho_R = np.maximum(self.epsilon,rho_R)
        p_L = np.maximum(self.epsilon,p_L)
        p_R = np.maximum(self.epsilon,p_R)


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
        V_L,V_R =shift(self.Data.V)


       


        
        rho_L,rho_R =V_L[self.Data.rho_idx],V_R[self.Data.rho_idx]
        u_L,u_R =V_L[self.Data.u_idx],V_R[self.Data.u_idx]
        v_L,v_R =V_L[self.Data.v_idx],V_R[self.Data.v_idx]
        p_L,p_R =V_L[self.Data.p_idx],V_R[self.Data.p_idx]
        rho_L = np.maximum(self.epsilon,rho_L)
        rho_R = np.maximum(self.epsilon,rho_R)
        p_L = np.maximum(0,p_L)
        p_R = np.maximum(0,p_R)

 
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
            temp_sum += np.abs(lams[i])*ws[i]*eigvecs[i] 
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
    



    def F_normal(self):
        shape = self.Data.V.shape


        V = np.zeros((shape[0],shape[1]+2,shape[2]+2))
        V[:,1:-1,1:-1] = self.Data.V
        V[:,:,0] = V[:,:,1]
        V[:,:,-1] = V[:,:,-2]
        V[:,0,:] = V[:,1,:]
        V[:,-1,:] = V[:,-2,:]

        rho = V[self.Data.rho_idx]
        u = V[self.Data.u_idx]
        v = V[self.Data.v_idx]
        p= V[self.Data.p_idx]
        ht = (self.gamma / (self.gamma - 1)) * (p/rho) + 0.5 * ((u)**2+v**2)

        U_left = (
            self.Data.leftward_normal[self.Data.unormal] * u[1:-1, 0:-2] +
            self.Data.leftward_normal[self.Data.vnormal] * v[1:-1, 0:-2]
        )

        F_left_C = np.array([
            rho[1:-1, 0:-2] * U_left,
            rho[1:-1, 0:-2] * u[1:-1, 0:-2] * U_left,
            rho[1:-1, 0:-2] * v[1:-1, 0:-2] * U_left,
            rho[1:-1, 0:-2] * ht[1:-1, 0:-2] * U_left
        ])
        F_left_P = np.array([
            np.zeros_like(U_left),
            self.Data.leftward_normal[self.Data.unormal] * p[1:-1, 0:-2],
            self.Data.leftward_normal[self.Data.vnormal] * p[1:-1, 0:-2],
            np.zeros_like(U_left)
        ])

        # Right face fluxes (i+1/2)
        U_right = (
            self.Data.rightward_normal[self.Data.unormal] * u[1:-1, 2:] +
            self.Data.rightward_normal[self.Data.vnormal] * v[1:-1, 2:]
        )

        F_right_C = np.array([
            rho[1:-1, 2:] * U_right,
            rho[1:-1, 2:] * u[1:-1, 2:] * U_right,
            rho[1:-1, 2:] * v[1:-1, 2:] * U_right,
            rho[1:-1, 2:] * ht[1:-1, 2:] * U_right
        ])
        F_right_P = np.array([
            np.zeros_like(U_right),
            self.Data.rightward_normal[self.Data.unormal] * p[1:-1, 2:],
            self.Data.rightward_normal[self.Data.vnormal] * p[1:-1, 2:],
            np.zeros_like(U_right)
        ])

        # Bottom face fluxes (j-1/2)
        U_bottom = (
            self.Data.downward_normal[self.Data.unormal] * u[0:-2, 1:-1] +
            self.Data.downward_normal[self.Data.vnormal] * v[0:-2, 1:-1]
        )

        F_bottom_C = np.array([
            rho[0:-2, 1:-1] * U_bottom,
            rho[0:-2, 1:-1] * u[0:-2, 1:-1] * U_bottom,
            rho[0:-2, 1:-1] * v[0:-2, 1:-1] * U_bottom,
            rho[0:-2, 1:-1] * ht[0:-2, 1:-1] * U_bottom
        ])
        F_bottom_P = np.array([
            np.zeros_like(U_bottom),
            self.Data.downward_normal[self.Data.unormal] * p[0:-2, 1:-1],
            self.Data.downward_normal[self.Data.vnormal] * p[0:-2, 1:-1],
            np.zeros_like(U_bottom)
        ])

        # Top face fluxes (j+1/2)
        U_top = (
            self.Data.upward_normal[self.Data.unormal] * u[2:, 1:-1] +
            self.Data.upward_normal[self.Data.vnormal] * v[2:, 1:-1]
        )

        F_top_C = np.array([
            rho[2:, 1:-1] * U_top,
            rho[2:, 1:-1] * u[2:, 1:-1] * U_top,
            rho[2:, 1:-1] * v[2:, 1:-1] * U_top,
            rho[2:, 1:-1] * ht[2:, 1:-1] * U_top
        ])
        F_top_P = np.array([
            np.zeros_like(U_top),
            self.Data.upward_normal[self.Data.unormal] * p[2:, 1:-1],
            self.Data.upward_normal[self.Data.vnormal] * p[2:, 1:-1],
            np.zeros_like(U_top)
        ])

        return (
            self.Data.bc_multiplier * F_left_C + F_left_P,
            self.Data.bc_multiplier * F_right_C + F_right_P,
            self.Data.bc_multiplier * F_top_C + F_top_P,
            self.Data.bc_multiplier * F_bottom_C + F_bottom_P
        )
