import numpy as np
import pyvista as pv
def load_grid(filename):


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

    return x[0],y[0]


def write_file(infile_name,outfile_name):

    x,y = load_grid(infile_name)


    # plt.scatter(x[::-1],y[::-1])
    # plt.plot((x[::-1])[0,0],(y[::-1])[0,0],"o",color = "red")


    x11 = x[0:-1,0:-1]
    x12 = x[0:-1,1:]
    x21 = x[1:,0:-1]
    x22 = x[1:,1:]

    y11 = y[0:-1,0:-1]
    y12 = y[0:-1,1:]
    y21 = y[1:,0:-1]
    y22 = y[1:,1:]

    x_cells= np.array([x21,x22,x11,x12])
    y_cells = np.array([y21,y22,y11,y12])
    x_cells = x_cells.reshape((4,x11.shape[0]*x11.shape[1]))
    y_cells = y_cells.reshape((4,x11.shape[0]*x11.shape[1]))

    np.savetxt(f"{outfile_name}_xx.csv",(x_cells.T).copy(),delimiter=",")
    np.savetxt(f"{outfile_name}_yy.csv",(y_cells.T).copy(),delimiter=",")


def write_vtk(grid_name,rho,u,v,p,output_name = "output.vtk"):
    x,y = load_grid(grid_name)
    midpoint_tempx = (x[1:,:]+x[0:-1,:])/2
    midpointxx = (midpoint_tempx[:,1:]+midpoint_tempx[:,0:-1])/2
    midpoint_tempy = (y[1:,:]+y[0:-1,:])/2
    midpointyy = (midpoint_tempy[:,1:]+midpoint_tempy[:,0:-1])/2


        
    grid = pv.StructuredGrid(midpointxx,midpointyy, np.zeros_like(p))

    grid["pressure"] =p.T.flatten()
    Ux = u.T
    Uy = v.T
    density = rho.T
    vel = np.stack([Ux, Uy, np.zeros_like(Ux)], axis=-1).reshape(-1, 3)
    grid["velocity"] = vel
    grid["density"] = density.flatten()
    grid.save(output_name) 