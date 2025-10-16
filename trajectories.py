"""
This script calculates the different flow gradients and savessnapshots of them within a specific time. 
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import pathlib,sys,os,json,h5py
from scipy.fft import rfft2,irfft2
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
mpl.rc("text", usetex = True)

idx = int(float(sys.argv[-1]))
# idx = 15
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
with h5py.File("/home/rajarshi.chattopadhyay/pokemon_data/plot_data.hdf5", "r") as f:   

    Q_c = f[f"st_{round(0.75*sts[st_idx],3):.2f}"]["meanQ_min"][alph_idx]

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
def plot_prtcl(axs,nnoncausprtcl,pos,tcaus,tidx,colors):
    axs.scatter(pos[tidx,:,0],pos[tidx,:,1],s =20.0,marker = '.',c = colors)
    for prtcl in range(nnoncausprtcl):

        # print(tcaus[prtcl])
        if tcaus[prtcl]< tidx:
            axs.plot(pos[:tcaus[prtcl],prtcl,0],pos[:tcaus[prtcl],prtcl,1],'.',color = "green",markersize = 0.1)
            axs.plot(pos[tcaus[prtcl]:tidx,prtcl,0],pos[tcaus[prtcl]:tidx,prtcl,1],'.',color = "magenta",markersize = 0.1)
        else:
            axs.plot(pos[:tidx,prtcl,0],pos[:tidx,prtcl,1],'.',color = "green",markersize = 0.1)
    return None

def plot_noncaus_prtcl(axs,nnoncausprtcl,pos,tidx):
    axs.scatter(pos[tidx,:,0],pos[tidx,:,1],s =20.0,marker = '.',c = "brown")
    for prtcl in range(nnoncausprtcl):
        axs.plot(pos[:tidx,prtcl,0],pos[:tidx,prtcl,1],'.',color = "brown",markersize = 0.1)
    return None

#%%
ti = 100
tf = 200
dtload = 0.1
caus_counti =  np.round(np.load(prtcl_loadPath/f"time_{ti:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)
caus_countf =  np.round(np.load(prtcl_loadPath/f"time_{tf:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)
non_caus_cond = caus_countf == 0
prtcl_idx = [] 
#! Takes particles that go through high strain regions after halfway. 
for tidx, t in enumerate(np.arange((tf-ti)/2 + ti,tf + 0.5*dtload,dtload)):
    caus_count =  np.round(np.load(prtcl_loadPath/f"time_{t:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)
    pos = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npz")["pos"]
    Qmat = np.load(prtcl_loadPath/f"time_{t:.2f}/Q.npz")["Q"]
    cond = (Qmat < Q_c)*(non_caus_cond)
    if cond.sum() > 0:
        prtcl_idx = prtcl_idx + list(np.argwhere(cond).ravel())
        prtcl_idx = np.unique(prtcl_idx)
        if len(prtcl_idx)>= 10:
            break


causidx = caus_countf == caus_counti + 1
print(causidx.sum())
ncausprtcl = 10
nnoncausprtcl = 10
ntimes = int(round((tf-ti)/dtload) +1)
pos = np.zeros((ntimes,ncausprtcl,2))
pos_nc = np.zeros((ntimes,nnoncausprtcl,2))


prtcl_idx = np.random.choice(prtcl_idx, nnoncausprtcl)
cprtcl_idx = np.random.choice(causidx.sum(),ncausprtcl)
print(ntimes)
caus_counti = caus_counti[causidx][cprtcl_idx]
caus_countf = caus_countf[causidx][cprtcl_idx]

            

#%%
tcaus = np.ones_like(caus_counti)*65536
for tidx, t in enumerate(np.arange(ti,tf + 0.5*dtload,dtload)):
    print(t,end = '\r')


    

    # for ii in range(4):
    # alph = 0.66666666666666667
    # vel = np.load(prtcl_loadPath/f"time_{t:.2f}/vel.npz")["vel"][]
    pos[tidx] = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npz")["pos"][causidx,:][cprtcl_idx,:]
    pos_nc[tidx] = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npz")["pos"][cprtcl_idx,:]
    caus_count =  np.round(np.load(prtcl_loadPath/f"time_{t:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)[causidx][cprtcl_idx]
    
    caus = caus_count - caus_counti
    colors = np.array(["green","magenta"])[caus]
    tcaus = np.minimum(np.where(caus > 0, tidx, tcaus),tcaus).astype(np.int32)
    # print(tcaus)

    if np.abs(t - np.round(t)) < dtload/2:
    # if True:    
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
        Q = -0.5*np.einsum('ij...,ji...->...',Ar,Ar)
        Xplot,Yplot = np.linspace(0,Lx,Nx+1,endpoint= True), np.linspace(0,Ly,Ny+1,endpoint= True)
        # xnew[1:,1:] = np.roll(xi_last_r, -1,axis = (0,1))
        Qnew = mk_tl2pi(Q.real)
        sig_snew = mk_tl2pi(sig_s.real)
        sig_nnew = mk_tl2pi(sig_n.real)
        omgnew = mk_tl2pi(omg.real)
            
            
        
        
        
        
        
        
        
        # if tidx == ntimes-1:
        norm = TwoSlopeNorm(vcenter = 0,vmax = 0.8,vmin=-0.8)
        
        
        fig,axs = plt.subplots(2, 2, sharex='col', sharey='row',constrained_layout=True,figsize = (6,6))
        # fig.patch.set_facecolor('white')
        # fig.patch.set_alpha(1.0)
        p1 = axs[0,0].imshow(Qnew.T*10, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm,alpha = 0.5)
        axs[0,0].set_title(r"$10Q$", fontsize=15)
        
        p2 = axs[0,1].imshow(sig_snew.T, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm,alpha = 0.5)
        axs[0,1].set_title(r"$\sigma_s$", fontsize=15)
        
        p3 = axs[1,0].imshow(sig_nnew.T, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm,alpha = 0.5)
        axs[1,0].set_title(r"$\sigma_n$", fontsize=15)
        
        p4 = axs[1,1].imshow(omgnew.T, extent=(Xplot[0], Xplot[-1], Yplot[0], Yplot[-1]), origin='lower', cmap='RdBu_r', norm=norm,alpha = 0.5)
        # p4 = axs[1,1].contour(Xplot,Yplot,omgnew.T,levels = np.linspace(-0.8,0.8,10),cmap = "RdBu_r",norm=norm)
        axs[1,1].set_title(r"$\omega$", fontsize=15)
        
        for ax in axs.flatten():

            plot_prtcl(ax, ncausprtcl, pos, tcaus, tidx, colors)
            plot_noncaus_prtcl(ax, nnoncausprtcl, pos_nc, tidx)

        # axs[1,1].scatter(pos[tidx,:,0],pos[tidx,:,1],s =10.0,marker = '.',c = colors)
        # for prtcl in range(ncausprtcl):

        #     print(tcaus[prtcl])
        #     if tcaus[prtcl]< tidx:

        #         axs[1,1].plot(pos[:tcaus[prtcl],prtcl,0],pos[:tcaus[prtcl],prtcl,1],'.',color = "green",markersize = 0.1)
        #         axs[1,1].plot(pos[tcaus[prtcl]:tidx,prtcl,0],pos[tcaus[prtcl]:tidx,prtcl,1],'.',color = "magenta",markersize = 0.1)
        #     else:
        #         axs[1,1].plot(pos[:tidx,prtcl,0],pos[:tidx,prtcl,1],'.',color = "green")

        
        # plt.tight_layout(pad = 0,w_pad = 0, h_pad = 0)
        cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03])  # [left, bottom, width, height]
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal',shrink = 0.3,extend ="both")
        # cbar.set_label(r"$Value$", fontsize=12)
        
        
        # u = Linterp(pos,u_field)
        # q = np.linalg.norm(vel - u,axis = 1)/np.linalg.norm(u,axis = 1)
        # print(pos.shape, q.shape)
        # plt.subplot(2,2,ii+1)
        # p1 = plt.pcolor(Xplot.get(),Yplot.get(),(Qnew.T).get(),cmap = "RdBu_r",norm = norm)
        # plt.colorbar(p1)
        # plt.plot(pos[:,0].get(),pos[:,1].get(),'.',color='#000000',markersize  = 0.2)
        # plt.plot(pos[causidx,0].get(),pos[causidx,1].get(),'x',color='#33d364',markersize  = 0.4)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$y$", rotation = 0)
        # plt.title(fr"({labels[ii]}) $\alpha={alph}$ ",fontsize = 15)
        fig.suptitle(r"$\frac{t}{\tau_p} = $"+f"{t/tp:.2f}")
        fig.supxlabel(r"$x$")
        fig.supylabel(r"$y$")
        # print(str(savedir))
        # fig.savefig(savedir/f"flow_gradients_time_{t:.0f}.png", bbox_inches='tight')
        fig.savefig(savedir/f"prtcls_time_{t:.0f}_alph_{alph:.2f}.png",bbox_inches='tight')
        print(f"saved for time {t:.2f} in {savedir}")
        # plt.show()
        plt.close()
#%%

# %%
