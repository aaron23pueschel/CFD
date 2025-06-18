import numpy as np
import matplotlib.pyplot as plt
import MMS_FORTRAN
class DataStructures(object):
    
    def __init__(self,NI,NJ,grd_name):

        self.p_idx = 3
        self.u_idx = 1
        self.v_idx = 2
        self.rho_idx = 0

        
        self.xxnormal = 0
        self.yynormal = 1
        self.unormal = 2
        self.vnormal = 3
        self.epsilon = .0000001
        self.MMS_length = 1

        self.NI = NI
        self.NJ = NJ

        self.U = np.zeros((4,self.NI-1,self.NJ-1)) 
        self.F = np.zeros((4,self.NI-1,self.NJ-1))
        self.V = np.zeros((4,self.NI-1,self.NJ-1))
        self.S = np.zeros((4,self.NI-1,self.NJ-1)) 
        
        self.x = np.linspace(0,1,self.NJ) # Opposite
        self.y = np.linspace(0,1,self.NI)
        self.xx = None
        self.yy = None
        #self.set_curved_geometry()
        self.set_ramp_BC()
        #self.load_grid(grd_name)
        self.set_MMS()
        self.set_normals()
        self.compute_all_areas()
        
    def set_MMS(self,MMS_ON=False):
        MMS_FORTRAN.set_inputs.initialize_constants()

        xx= (self.xx[1:,:]+self.xx[0:-1,:])/2
        x = (xx[:,1:]+xx[:,0:-1])/2

        yy= (self.yy[1:,:]+self.yy[0:-1,:])/2
        y = (yy[:,1:]+yy[:,0:-1])/2
        

        
        pressure = np.zeros_like(x)
        vvel = np.zeros_like(x)
        uvel = np.zeros_like(x)
        density = np.zeros_like(x)

        cmass = np.zeros_like(x)
        xmom = np.zeros_like(x)
        ymom = np.zeros_like(x)
        energymom = np.zeros_like(x)
        length = self.MMS_length
        for i in range(len(x)):
            for j in range(len(x[0])):
                pressure[i,j] = MMS_FORTRAN.press_mms(length,x[i,j],y[i,j])
                density[i,j] = MMS_FORTRAN.rho_mms(length,x[i,j],y[i,j])
                uvel[i,j] = MMS_FORTRAN.uvel_mms(length,x[i,j],y[i,j])
                vvel[i,j] = MMS_FORTRAN.vvel_mms(length,x[i,j],y[i,j])

                cmass[i,j] = MMS_FORTRAN.rmassconv(length,x[i,j],y[i,j])
                xmom[i,j] = MMS_FORTRAN.xmtmconv(length,x[i,j],y[i,j])
                ymom[i,j] = MMS_FORTRAN.ymtmconv(length,x[i,j],y[i,j])
                energymom[i,j] = MMS_FORTRAN.energyconv(1.4,length,x[i,j],y[i,j])
        if MMS_ON:
            self.MMS_primitive = np.array([density,uvel,vvel,density])
            self.MMS_conserved = np.array([cmass,xmom,ymom,energymom])
        else:
            self.MMS_conserved = 0
            self.MMS_primitive = 0
    def set_ramp_BC(self):
        
        def generate_ramp_mesh(x_flat=1.0, x_ramp=2.0, height=1.0, angle_deg=30, 
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
        self.xx = X
        self.yy = Y


    def set_normals(self):
        
        xx = self.xx
        yy = self.yy

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

    def set_curved_geometry(self):
        
        xx_,yy_ = np.meshgrid(self.x,self.y)
        xx = (xx_)
        yy = yy_ +xx #-#.5*(yy_+xx**2)


        self.xx = xx
        self.yy = yy
    def plot_Geometry(self):
        plt.plot(self.xx.flatten(), self.yy.flatten(), ".")



    def plot_primitive(self,type_="pressure"):
        midpoint_tempx = (self.xx[1:,:]+self.xx[0:-1,:])/2
        midpointxx = (midpoint_tempx[:,1:]+midpoint_tempx[:,0:-1])/2
        midpoint_tempy = (self.yy[1:,:]+self.yy[0:-1,:])/2
        midpointyy = (midpoint_tempy[:,1:]+midpoint_tempy[:,0:-1])/2
        #print(self.V.shape,midpointxx.shape)
        if type_=="pressure":
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=(self.V[self.p_idx]).flatten())
            plt.colorbar()
        if type_=="velocity":
            plt.quiver(midpointxx,midpointyy,self.V[self.u_idx],self.V[self.v_idx])
        if type_=="density":
            plt.scatter(midpointxx.flatten(),midpointyy.flatten(),c=self.V[self.rho_idx].flatten())
            plt.colorbar()
        plt.show()
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
        top_wall = height_throat + (height_in - height_throat) * (1 + np.cos(theta)) / 2
        bottom_wall = -top_wall  # symmetric nozzle

        X, Y = np.meshgrid(x, np.linspace(0, 1, self.NI))

        # Stretch Y between bottom and top walls
        for i in range(self.NJ):
            Y[:, i] = bottom_wall[i] + Y[:, i] * (top_wall[i] - bottom_wall[i])
        self.xx = X
        self.yy = Y
    def compute_all_areas(self,visualize=True):
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
        self.AREA = np.einsum("ij,jkl->ikl",np.ones((4,1)),areas[np.newaxis,:,:])
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

        self.xx = x[0]
        self.yy = y[0]
