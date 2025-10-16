"""
This script calculates the different flow gradients and savessnapshots of them within a specific time. 
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import pathlib,sys,os,json
from scipy.fft import rfft2,irfft2
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
mpl.rc("text", usetex = True)

idx = 8
alph_idx = idx%8
st_idx = idx//8
#%%

## ---------------- params ----------------------- ## 
paramfile = '/home/rajarshi.chattopadhyay/pokemon_data/codes/parameters.json'
with open(paramfile,'r') as jsonFile: params = json.load(jsonFile)
PI = np.pi
TWO_PI = 2*PI
d = params["d"] # Dimension
nu =params["nu"] # Viscosity
Re = 1/nu if nu > 0 else np.inf # Reynolds number
N = Nx = Ny = params["N"] # Grid size
dt = params["dt"] # Timestep
# T = params["T"] # Final time
T = 500# Final time
einit = params["einit"] # Initial energy
kinit = params["kinit"] # Initial energy is distributed among these wavenumbers
linnu = params["linnu"] # Linear drag co-efficient
lp = params["lp"] # Power of the laplacian in the hyperviscous term
shell_count = params["shell_count"] # The shells where the energy is injected
alphs = [0.70,0.72,0.75,0.77,0.80,0.85,0.90,1.00]
# alph = params["alpha"] # The density ratio
alph = alphs[alph_idx] # The density ratio
Nprtcl = int(params["Nprtcl"]*Nx*Ny) # Number of particles
tf = params["tf"] # Final time for the particles
sts = [0.2,0.3,0.4,0.47]
# st = params["st"]*tf # Time period for the particles
st = sts[st_idx]*tf # Time period for the particles
tp = st # Save the particle data after this many timesteps
st_old = st
st = 3./4.*st #* correcting for the modified tp in the problem
savestep = int(params["savestep"]/dt) # Save the data after this many timesteps
prtcl_savestep = int(params["prtcl_savestep"]/dt) # Save the particle data after this many timesteps
eta = params["eta"]/(Nx//3) # Desired Kolmogorov length scale
restart = params["restart"] # Restart from the last saved data
kf = params["kf"] # Forcing wavenumber
f0 = params["f0"] # Forcing amplitude
order = params["order"] # Order of the spline interpolation
# t = np.arange(0,1.1*dt + T,dt) # Time array
P = (nu**3/eta**4)/len(shell_count) # power per unit shell
print(f"Number of particles: {Nprtcl}, old stokes number {st_old/tf}, new stokes  number {st/tf}, and alpha {alph}")
## ----------------------------------------------- ##


#%%
curr_path = pathlib.Path("/home/rajarshi.chattopadhyay/pokemon_data")
iname = "bspline" if order >1 else "linear"

savePlot = curr_path/f"Plots/{iname}/Re_{np.round(Re,2)},dt_{dt},N_{N}/"
savePlot.mkdir(parents=True, exist_ok=True)


if st_old == 0.3*tf: 
    loadPath = curr_path/f"Re_{np.round(Re,2)},dt_{dt},N_{Nx}/"
else: 
    loadPath = curr_path/f"brenner/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/"
    
prtcl_loadPath = loadPath/f"alpha_{alph:.2f}_prtcl/St_{(st_old/tf):.2f}/"
print(f" Found loadpath {loadPath.exists()}")
print(f" Found particle loadpath {prtcl_loadPath.exists()}")

savePlotdataPath = prtcl_loadPath/f"savePlotdata_{iname}/"


try: savePlotdataPath.mkdir(parents = True, exist_ok = False)
except FileExistsError: pass

#%%


Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
x,y = np.meshgrid(X,Y,indexing="ij")
PI = np.pi
TWO_PI = 2*PI
#%%
## It is best to define the function which returns the real part of the iifted function as ifft. 
ifft2 = lambda x: irfft2(x,(Nx,Ny))
#%%
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
#%%
def mk_tl2pi(Q): 
    Qnew = np.zeros((Q.shape[0]+ 1, Q.shape[1]+ 1))
    Qnew[1:,1:] = np.roll(Q, -1,axis = (0,1))
    Qnew[:-1,:-1] = Q
    Qnew[-1,:-1] = Q[0,:]
    Qnew[:,-1] = Qnew[:,0]
            
    
    return Qnew
#%%

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
#%%
for t in np.arange(0,500.1,1):
    print(t,end = '\r')


    

    # for ii in range(4):
    # alph = 0.66666666666666667
        # alph = alphs[ii]
        # pos= np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/pos.npy")
        # vel= np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/vel.npy")
        # TrZ = np.einsum('...ii->...',np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/prtcl_Z.npy"))
        # caus_count = np.load(loadPath/f"alpha_{alph:.2}_prtcl/St_{st}/time_{t:.2f}/caus_count.npy")
        # print(TrZ.shape,Nprtcl*N**2)
        # causidx = np.argwhere(caus_count>0)
        # print((caus_count>0).sum()/len(caus_count))
    if t%1 < 0.01:
        xi_last = tp*np.load(loadPath/f"time_{t:.2f}/w.npz")["vorticity"]
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
        sig_s = Ar[0,1] + Ar[1,0]
        sig_n = Ar[0,0] - Ar[1,1]
        omg = Ar[1,0] - Ar[0,1]
        print((omg**3).mean())
        Q = -0.5*np.einsum('ij...,ji...->...',Ar,Ar)
        # Xplot,Yplot = np.linspace(0,Lx,Nx+1,endpoint= True), np.linspace(0,Ly,Ny+1,endpoint= True)
        # xnew[1:,1:] = np.roll(xi_last_r, -1,axis = (0,1))
        # Qnew = mk_tl2pi(Q.real)
        # sig_snew = mk_tl2pi(sig_s.real)
        # sig_nnew = mk_tl2pi(sig_n.real)
        # omgnew = mk_tl2pi(omg.real)
            
            
        
        # norm = TwoSlopeNorm(vcenter = 0,vmax = 0.8,vmin=-0.8)
        
        
        # fig,axs = plt.subplots(2, 2, sharex='col', sharey='row',constrained_layout=True,figsize = (6,6))
        # # fig.patch.set_facecolor('white')
        # # fig.patch.set_alpha(1.0)
        # p1 = axs[0,0].imshow(Qnew.T*10, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm)
        # axs[0,0].set_title(r"$10Q$", fontsize=15)
        
        # p2 = axs[0,1].imshow(sig_snew.T, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm)
        # axs[0,1].set_title(r"$\sigma_s$", fontsize=15)
        
        # p3 = axs[1,0].imshow(sig_nnew.T, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm)
        # axs[1,0].set_title(r"$\sigma_n$", fontsize=15)
        
        # p4 = axs[1,1].imshow(omgnew.T, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm)
        # axs[1,1].set_title(r"$\omega$", fontsize=15)
        
        # # plt.tight_layout(pad = 0,w_pad = 0, h_pad = 0)
        # cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03])  # [left, bottom, width, height]
        # cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal',shrink = 0.3,extend ="both")
        # # cbar.set_label(r"$Value$", fontsize=12)
        
        
        # # u = Linterp(pos,u_field)
        # # q = np.linalg.norm(vel - u,axis = 1)/np.linalg.norm(u,axis = 1)
        # # print(pos.shape, q.shape)
        # # plt.subplot(2,2,ii+1)
        # # p1 = plt.pcolor(Xplot.get(),Yplot.get(),(Qnew.T).get(),cmap = "RdBu_r",norm = norm)
        # # plt.colorbar(p1)
        # # plt.plot(pos[:,0].get(),pos[:,1].get(),'.',color='#000000',markersize  = 0.2)
        # # plt.plot(pos[causidx,0].get(),pos[causidx,1].get(),'x',color='#33d364',markersize  = 0.4)
        # # plt.gca().set_aspect('equal', adjustable='box')
        # # plt.xlabel(r"$x$")
        # # plt.ylabel(r"$y$", rotation = 0)
        # # plt.title(fr"({labels[ii]}) $\alpha={alph}$ ",fontsize = 15)

        # fig.supxlabel(r"$x$")
        # fig.supylabel(r"$y$")
        # # print(str(savedir))
        # fig.savefig(savedir/f"flow_gradients_time_{t:.0f}.png", bbox_inches='tight')
        # # plt.savefig(savedir/f"caustics_time_{t:.2f}.png")
        # # plt.show()
        # plt.close()
#%%

# %%
