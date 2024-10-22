
import cupy as np
import matplotlib.pyplot as plt
import pathlib,sys,os,json
from cupy.fft import rfft2,irfft2
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
mpl.rc("text", usetex = True)


paramfile = '/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/parameters.json'
with open(paramfile,'r') as jsonFile: params = json.load(jsonFile)

d = params["d"] # Dimension
nu =params["nu"] # Viscosity
Re = 1/nu if nu > 0 else np.inf # Reynolds number
N = Nx = Ny = params["N"] # Grid size
dt = params["dt"] # Timestep
T = params["T"] # Final time
alph = params["alph"] # Density of the particles
eta = params["eta"]/(Nx//3) # Desired Kolmogorov length scale
Nprtcl = int(params["Nprtcl"]*Nx*Ny) # Number of particles
tf = params["tf"] # Kolmogorov timescale
st = params["st"]*tf # Particle Stokes number
linnu = params["linnu"] # Linear viscosity
order = params["order"] # Order of the scheme


savePlot = pathlib.Path(f"/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/Plots/Re_{np.round(Re,2)},dt_{dt},N_{N}/")
savePlot.mkdir(parents=True, exist_ok=True)
loadPath = pathlib.Path(f"/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/data/Re_{np.round(Re,2)},dt_{dt},N_{N}/")
loadPath.exists()



Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
x,y = np.meshgrid(X,Y,indexing="ij")
PI = np.pi
TWO_PI = 2*PI

## It is best to define the function which returns the real part of the iifted function as ifft. 
ifft2 = lambda x: irfft2(x,(Nx,Ny))

## Forming the 2D grid (k space)
Kx = 2*np.pi*np.linspace(-(Nx//2) , Nx//2 - 0.5*(1+ (-1)**(Nx%2)),Nx)/Lx
Ky = 2*np.pi*np.linspace(-(Ny//2) , Ny//2 - 0.5*(1+ (-1)**(Ny%2)),Ny)/Ly
Kx = np.append(Kx[Kx>=0], Kx[Kx<0])
Ky = np.append(Ky[Ky>=0], -Ky[0])
kx,ky = np.meshgrid(Kx,Ky,indexing="ij")
## Defining the inverese laplacian.
lap = -(kx**2 + ky**2)
# lap1 = lap.copy()
# lap1[lap1== 0] = np.inf
lapinv = 1.0/np.where(lap == 0., np.inf, lap)
kx.shape

poly = np.zeros((Nprtcl,d,order),dtype = np.float64)
Mmat = np.array([
    [0,-7/15,4/5,-1/3],
    [1,-1/5,-9/5,1],
    [0,4/5,6/5,-1],
    [0,-2/15,-1/5,1/3]    
]) # The matrix for the spline interpolation
def interp_spline(pos,u_field,A_field,deludelt,DADt):
    global umat,Amat,deludeltmat,DADtmat
    umat[:] = 0.0
    Amat[:] = 0.0
    deludeltmat[:] = 0.0
    DADtmat[:] = 0.0
    
    idx = (pos/dx).astype(int)
    delx = (pos%TWO_PI - X[idx%N])/dx
    
    for i in range(order): 
        poly[:,i] = delx**i
    for i in range(order):
        for j in range(order):
            # temparr[:] = u_field[((idx[:,0]-1 + i)%N),((idx[:,1]-1 + j)%N),...] #! Not giving for saving memories and variables
            umat  += u_field[((idx[:,0]-1 + i)%N),((idx[:,1]-1 + j)%N),...]*np.einsum('p,q,...p,...q->...',Mmat[i],Mmat[j],poly[...,0],poly[...,1])[:,None]
            
    return umat,Amat





savedir = savePlot/"4movies"
savedir.mkdir(parents=True, exist_ok=True)
Xplot,Yplot = np.linspace(0,Lx,Nx+1,endpoint= True), np.linspace(0,Ly,Ny+1,endpoint= True)
xnew = np.zeros((Nx+1,Ny+1))
psi = np.zeros((Nx,Ny//2+1),dtype = np.complex128)
u_field = np.zeros((Nx,Ny,2))
alphs = [0.7,0.75,0.8,0.95]
labels = ['a','b','c','d']
Ar = np.zeros((d,d,Nx,Ny),dtype = np.float64)
A = np.zeros((d,d,Nx,Ny//2+1),dtype = np.complex128)
xnew = np.zeros((Nx+1,Ny+1))
Qnew = np.zeros((Nx+1,Ny+1))
for t in np.arange(0,150.1,1):
    print(t,end = '\r')
    plt.figure(figsize = (16,14))
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$", rotation = 0)
    
    for ii in range(4):
    # alph = 0.66666666666666667
        alph = alphs[ii]
        pos= np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/pos.npy")
        vel= np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/vel.npy")
        # TrZ = np.einsum('...ii->...',np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/prtcl_Z.npy"))
        caus_count = np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/caus_count.npy")
        # print(TrZ.shape,Nprtcl*N**2)
        causidx = np.argwhere(caus_count>0)
        # print((caus_count>0).sum()/len(caus_count))
        if t%5 < 0.01:
            xi_last = np.load(loadPath/f"time_{t:.2f}/w.npy")
            # print(xi_last.shape,lapinv.shape)
            # psi[:] = -xi_last*lapinv
            # u_field[...,0] = ifft2(1j * ky*psi)
            # u_field[...,1] = ifft2(-1j * kx*psi) 
            # xi_last_r = ifft2(xi_last)
            # xnew[1:,1:] = np.roll(xi_last_r, -1,axis = (0,1))
            # xnew[:-1,:-1] = xi_last_r
            
            u =1j* ky*lapinv*xi_last
            v = -1j*kx*lapinv*xi_last
            A[:] = 0.0
            Ar[:] = 0.0
            A[0,0] = 1j*kx*u
            A[0,1] = 1j*ky*u
            A[1,0] = 1j*kx*v
            A[1,1] = 1j*ky*v
            # Q = 0.0
            for i in range(d):
                for j in range(d):
                    Ar[i,j] = ifft2(A[i,j])
                    
            Q = -0.5*st*np.einsum('ij...,ji...->...',Ar,Ar)
            Xplot,Yplot = np.linspace(0,Lx,Nx+1,endpoint= True), np.linspace(0,Ly,Ny+1,endpoint= True)
            # xnew[1:,1:] = np.roll(xi_last_r, -1,axis = (0,1))
            Qnew[1:,1:] = np.roll(Q, -1,axis = (0,1))
            Qnew[:-1,:-1] = Q
            Qnew[-1,:-1] = Q[0,:]
            Qnew[:,-1] = Qnew[:,0]
            # xnew[:-1,:-1] = xi_last_r    
            # Qnew[:-1,:-1] = Q    
            
        
        norm = TwoSlopeNorm(vcenter = 0,vmax = .1,vmin=-.1)
        # u = Linterp(pos,u_field)
        # q = np.linalg.norm(vel - u,axis = 1)/np.linalg.norm(u,axis = 1)
        # print(pos.shape, q.shape)
        plt.subplot(2,2,ii+1)
        p1 = plt.pcolor(Xplot.get(),Yplot.get(),(Qnew.T).get(),cmap = "RdBu_r",norm = norm)
        plt.colorbar(p1)
        plt.plot(pos[:,0].get(),pos[:,1].get(),'.',color='#000000',markersize  = 0.2)
        plt.plot(pos[causidx,0].get(),pos[causidx,1].get(),'x',color='#33d364',markersize  = 0.4)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$", rotation = 0)
        plt.title(fr"({labels[ii]}) $\alpha={alph}$ ",fontsize = 15)
        
    plt.tight_layout()
    plt.savefig(savedir/f"caustics_time_{t:.2f}.png",dpi = 100)
    # plt.show()
    plt.close()
         