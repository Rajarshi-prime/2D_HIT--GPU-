"""
This script modifies the trajectories script in the local computer. 
Written on 23.09.2025.
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import pathlib,sys,os,json,h5py
from scipy.fft import rfft2,irfft2
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
from scipy.interpolate import splrep, splev, splder,sproot,PPoly
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
Q_c
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
    
    
def compute_prev_minima(x,y,count_number):
    """Outputs the value of x for which the derivative of y is zero 
    With a syntax copied from Microsoft Copilot.
    """
    tck = splrep(x,y,k = 3)
    tck_der = splder(tck)
    tck_2der = splder(tck,n=2)
    scaled_y = y/ np.mean(y**2)**0.5
    # print(PPoly.from_spline(tck_der).roots(extrapolate = False))
    try: 
        roots = PPoly.from_spline(tck_der).roots(extrapolate = False)
        der2_at_roots = splev(roots, tck_2der) 
        last_root = roots[der2_at_roots >0][-1] #! time for the last root 

    except :
        idx = -1 if y[-1] < y[0] else 0
        count_number += 1
        return x[idx],y[idx],count_number
    return x[x>last_root][0],y[x>last_root][0],count_number


def compute_next_minima(x,y,count_number):
    """Outputs the value of x for which the derivative of y is zero 
    With a syntax copied from Microsoft Copilot.
    """
    # print(PPoly.from_spline(tck_der).roots(extrapolate = False))
    if len(x) > 5:
        tck = splrep(x,y,k = 5)
        tck_der = splder(tck)
        tck_2der = splder(tck,n=2)
        try: 
            roots = PPoly.from_spline(tck_der).roots(extrapolate = False)
            der2_at_roots = splev(roots,tck_2der)
            first_root = roots[der2_at_roots>0][0] #! time for the last root where the 2nd derivative is less than zero. 
        except :
                idx = -1 if y[-1] < y[0] else 0
                count_number += 1
                return x[idx],y[idx],count_number
        return x[x<first_root][-1],y[x<first_root][-1],count_number
    else: 
        idx = -1 if y[-1] < y[0] else 0
        count_number += 1
        return x[idx],y[idx],count_number

def rescale(traj,idx,vel):
    """
    Rescales the trajectory such that the particle at idx is at origin with y vel. 
    """
    traj -= traj[idx,:]
    # trajnew = traj
    velmin = vel[idx] 
    vmin = np.linalg.norm(velmin)
    cth = velmin[0]/vmin
    sth = velmin[1]/vmin
    R = np.array([[sth, -cth],[cth,sth]])
    trajnew =  np.einsum('ij,...j->...i', R, traj)
    # if trajnew[idx+1,0] < trajnew[idx,0]:
    #     trajnew[idx:,0] *= -1.0
    # if trajnew[idx-1,0] > trajnew[idx,0]:
    #     trajnew[:idx,0] *= -1.0
    return trajnew
    
def unmodulo(arr,base = TWO_PI):
    """
    Checks if the points have difference more than 'base'/2. Then it adds 'base' to the all the next points according to the sign of the crossing.
    """
    cond = np.zeros_like(arr)
    cond[1:] = np.cumsum(np.sign(arr[:-1] -arr[1:] )*(np.abs(arr[1:] - arr[:-1]) > base/2))* base
    return cond + arr    
    
def curvature(t,x,y):
    """calculates the curvature of x(t) and y(t)"""
    xspline = splrep(t,x,k = 5)
    yspline = splrep(t,y,k = 5)

    xdot_tck = splder(xspline)
    ydot_tck = splder(yspline)
    
    xddot_tck = splder(xspline,n=2)
    yddot_tck = splder(yspline,n=2)
    

    dxdt = splev(t,xdot_tck)
    dydt = splev(t,ydot_tck)
    d2xdt2 = splev(t,xddot_tck)
    d2ydt2 = splev(t,yddot_tck)
    v = (dxdt**2 + dydt**2)**0.5
    return np.abs(dxdt*d2ydt2 - dydt*d2xdt2)/(v**3 + 1e-16)
    
    
def curvature_v(t,vx,vy):
    """calculates the curvature of x(t) and y(t) when given v_x(t) and v_y(t)"""
    v = (vx**2 + vy**2)**0.5
    vhatx = vx/(v + 1e-16)
    vhaty = vy/(v + 1e-16)
    vhatdotx = np.gradient(vhatx,t)
    vhatdoty = np.gradient(vhaty,t)

    return (vhatdotx**2 + vhatdoty**2)**0.5/v
#%%
with h5py.File("/home/rajarshi.chattopadhyay/pokemon_data/plot_data.hdf5", "r") as f:   

    Q_c = f[f"st_{round(0.75*sts[st_idx],3):.2f}"]["meanQ_min"][alph_idx]

#%%
"""
1. Caustics particles. 
    a. Take particles that go through caustics between 235-265. 
    b. Rescale their trajectories from  200-300.
2. Non caustics particles. 
    a. Take non-caustics particles that cross the Q threshold between 235-265. 
    b. Rescale their trajectories from 200-300. 

Plot. 
"""   

#%%
ntraj = 4000
ti = 200
tf = 300
dtload = 0.1
caus_counti =  np.round(np.load(prtcl_loadPath/f"time_{ti:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)
caus_countf =  np.round(np.load(prtcl_loadPath/f"time_{tf:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)
non_caus_cond = caus_countf == 0
caus_idx = np.argwhere((caus_countf - caus_counti) == 1).ravel()
if len(caus_idx)< ntraj: 
    raise SystemExit("Not enough caus particles")
caus_idx = np.random.choice(caus_idx, size = min(caus_idx.size,ntraj,len(caus_idx)),replace = False)
prtcl_idx = [] 
print(len(caus_idx))


#! Takes particles that go through high strain regions after halfway. 

for tidx,t in enumerate(np.round(np.arange(ti,tf+0.05,0.1),1)):

    pos = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npz")["pos"]
    Qmat = np.load(prtcl_loadPath/f"time_{t:.2f}/Q.npz")["Q"]
    cond = (Qmat < Q_c)*(non_caus_cond)
    if cond.sum() > 0 :
        prtcl_idx = prtcl_idx + list(np.argwhere(cond).ravel())
        prtcl_idx = list(np.unique(prtcl_idx))
    
if len(prtcl_idx)< ntraj: 
    raise SystemExit("Not enough Qmin particles")
prtcl_idx = np.random.choice(prtcl_idx, size = ntraj, replace = False)



# %%
tstart  = 100
tend = 400
trange = np.round(np.arange(tstart, tend,0.1),1)
pos_caus = np.zeros((trange.size,ntraj, 2), dtype = np.float64)
vel_caus = np.zeros((trange.size,ntraj, 2), dtype = np.float64)
pos_nc = np.zeros((trange.size,ntraj, 2), dtype = np.float64)
vel_nc = np.zeros((trange.size,ntraj, 2), dtype = np.float64)
caus_count = np.zeros((trange.size,ntraj),dtype = np.int32)
Q_caus = np.zeros((trange.size,ntraj),dtype = np.float64)
Q_nc = np.zeros((trange.size,ntraj),dtype = np.float64)


#%%

for i,t in enumerate(trange):
    pos = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npz")["pos"]
    vel = np.load(prtcl_loadPath/f"time_{t:.2f}/vel.npz")["vel"]
    caus_count[i] = np.round(np.load(prtcl_loadPath/f"time_{t:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)[caus_idx]
    Qmat = np.load(prtcl_loadPath/f"time_{t:.2f}/Q.npz")["Q"]
    
    pos_caus[i] = pos[caus_idx]
    vel_caus[i] = vel[caus_idx]
    Q_caus[i] = Qmat[caus_idx]
    pos_nc[i] = pos[prtcl_idx]
    vel_nc[i] = vel[prtcl_idx]
    Q_nc[i] = Qmat[prtcl_idx]
    print(t,end='\r')
#%%


# %%
tcond = (trange >= ti)*(trange<=tf)
caus_minidx = np.zeros(ntraj, dtype = np.int32)
nc_minidx = np.zeros(ntraj,dtype = np.int32)
for i in range(ntraj):
    cidx = int((trange< ti).sum() + (np.argwhere((caus_count[tcond,i] - caus_count[tcond,i][0])>0).ravel())[0])

    tminim,Qminma, _  = compute_prev_minima(np.arange(trange[:cidx].size),Q_caus[:cidx,i],0)
    
    caus_minidx[i] = int(tminim)
    
    cross_idx = int((trange< ti).sum() + (np.argwhere(Q_nc[tcond,i] < Q_c).ravel())[0])
    tminim,Qminma, _ = compute_next_minima(np.arange(trange[cross_idx:].size),Q_nc[cross_idx:,i],0)
    
    nc_minidx[i] = cross_idx  + int(tminim)

# %%
curvature_caus = np.zeros((trange.size, ntraj))
curvature_nc = np.zeros((trange.size, ntraj))
for i in range(ntraj):
    for m in range(2):
        pos_caus[:,i,m] = unmodulo(pos_caus[:,i,m])
        pos_nc[:,i,m] = unmodulo(pos_nc[:,i,m])
        
    curvature_caus[:,i] = curvature(trange, pos_caus[:,i,0],pos_caus[:,i,1])
    curvature_nc[:,i] = curvature(trange, pos_nc[:,i,0],pos_nc[:,i,1])
        
    # curvature_caus[:,i] = curvature_v(trange, vel_caus[:,i,0],vel_caus[:,i,1])
    # curvature_nc[:,i] = curvature_v(trange, vel_nc[:,i,0],vel_nc[:,i,1])
    # print((pos_nc[:,i,0],pos_nc[:,i,1]))
    
    pos_caus[:,i,:] = rescale(pos_caus[:,i,:], caus_minidx[i],vel_caus[:,i,:])
    pos_nc[:,i,:] = rescale(pos_nc[:,i,:],nc_minidx[i],vel_nc[:,i,:])
    

#%%
ftimes = 301
fctimes = ftimes - ftimes//2
curvature_caus = np.zeros(ftimes)
curvature_nc = np.zeros(ftimes)
count_c = np.zeros(ftimes)
count_nc = np.zeros(ftimes)
for i in range(ntraj):
    minidx = caus_minidx[i] 

    endidx = minidx +  min( ftimes - fctimes, len(pos_caus[minidx:minidx + ftimes - fctimes,i,0]))
    startidx = max(minidx -fctimes,0)
    
    time = np.arange(startidx ,endidx)*0.1
    curvature_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature(time, pos_caus[startidx: endidx,i,0], pos_caus[startidx:endidx,i,1])
    # curvature_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature_v(time, vel_caus[startidx: endidx,i,0], vel_caus[startidx:endidx,i,1])
    count_c[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += 1
curvature_caus /= count_c + 1e-16
for i in range(ntraj):
    minidx = nc_minidx[i]
    endidx = minidx +  min( ftimes - fctimes, len(pos_nc[minidx:minidx + ftimes - fctimes,i,0]))
    startidx = max(minidx -fctimes,0)
    time = np.arange(startidx ,endidx)*0.1
    
    curvature_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature(time, pos_nc[startidx:endidx,i,0], pos_nc[startidx:endidx,i,1])
    # curvature_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature_v(time, vel_nc[startidx:endidx,i,0], vel_nc[startidx:endidx,i,1])
    count_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += 1
    
curvature_nc /= count_nc + 1e-16

#%%
plt.plot((np.arange(ftimes)- ftimes//2)*0.1/st,curvature_caus,color = "red", label = "Caus")
plt.plot((np.arange(ftimes)- ftimes//2)*0.1/st,curvature_nc,color = "blue", label = "NC")
plt.xlabel(r"$(t - t_{min})/\tau_p$")
plt.ylabel("Curvature")
# plt.ylim(None,10)
plt.yscale("log")
plt.legend()
plt.savefig(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/curvature_alph_{alph:.2f}.png",dpi = 300,bbox_inches = "tight")


#%%
# mean_curvature_caus = np.std(curvature_caus,axis = -1)
# mean_curvature_nc = np.std(curvature_nc,axis = -1)
# plt.plot(mean_curvature_caus,color ="red")
# plt.plot(mean_curvature_nc,color = "blue")
# mean_curvature_caus

# %%
colors = ["9a031e", "fb8b24","ffffff","00b4d8","0077b6"]
colors2 = ["81b29a","06d6a0","ffffff","ef476f","e07a5f"]
colors = ["#" + c for c in colors]
colors2 = ["#" + c for c in colors2]
positions = np.linspace(0,1,len(colors))
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
cmap_fut = mpl.colors.LinearSegmentedColormap.from_list('cmap_fut', colors[2:])
cmap_past = mpl.colors.LinearSegmentedColormap.from_list('cmap_past', colors[:3])
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors2)))
cmap_fut2 = mpl.colors.LinearSegmentedColormap.from_list('cmap_fut', colors2[2:])
cmap_past2 = mpl.colors.LinearSegmentedColormap.from_list('cmap_past', colors2[:3])

#%%
#* Both at once
ftimes = 501
curvature_caus = np.zeros((ftimes))
curvature_nc = np.zeros((ftimes))

movies = pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/4movies")
movies.mkdir(parents = True, exist_ok = True)
for t in range(ftimes - 1,ftimes): 

    # if t < ftimes//2: continue
    fig,ax = plt.subplots(1,2,figsize = (6,6))
    plt.style.use('dark_background')
    for axs in ax:
        axs.axis('off')
        axs.set_xlim(-0.5,0.5)
        axs.set_ylim(-1,1)
    for i in range(ntraj):
        #* pre Q_min behaviour
        minidx = caus_minidx[i]
        tact = min(t,ftimes//2)
        fctimes = ftimes - ftimes//2
        if minidx - (fctimes) + tact >0 : 
            if pos_caus[max( minidx - fctimes, 0) ,i,0] < 0 : 
                col_caus_past = cmap_past2
                col_caus_fut = cmap_fut2
            else: 
                col_caus_past = cmap_past
                col_caus_fut = cmap_fut
                
            
            ax[0].scatter(pos_caus[max(minidx - fctimes, 0) : minidx - (fctimes - tact) ,i,0],pos_caus[max(minidx - fctimes, 0) : minidx - (fctimes - tact),i,1],c = np.arange(tact + min(minidx - fctimes, 0)),cmap = col_caus_past,s = 0.1,alpha = 1.0)
        
        
        minidx = nc_minidx[i]
        tact = min(t,ftimes//2)
        # print(minidx - ftimes)
        if minidx - (fctimes) + tact >0 : 
            if pos_nc[max( minidx - fctimes, 0),i,0] < 0: 
                col_nc_past = cmap_past2
                col_nc_fut = cmap_fut2
            else: 
                col_nc_past = cmap_past
                col_nc_fut = cmap_fut
                   
               
            ax[1].scatter(pos_nc[max(minidx - fctimes, 0) : minidx - (fctimes - tact),i,0],pos_nc[max(minidx - fctimes, 0) : minidx - (fctimes - tact),i,1],c = np.arange(tact + min(minidx - fctimes, 0)),cmap = col_nc_past,s = 0.1,alpha = 1.0)
        
        if t > ftimes//2: #*  post Q_min behaviour.
            # raise SystemExit("t_min reached")
            minidx = caus_minidx[i]
            tact = t - ftimes//2
            if minidx + tact>= trange.size: 
                tact = -1 - minidx
            
        
            
            ax[0].scatter(pos_caus[minidx:minidx + tact,i,0],pos_caus[minidx:minidx + tact ,i,1],c = np.arange(tact),cmap = col_caus_fut,s = 0.1,alpha = 1.0)
            
            
            minidx = nc_minidx[i]
            tact = t - ftimes//2
            if minidx + tact>= trange.size: 
                tact = -1 - minidx

            ax[1].scatter(pos_nc[minidx:minidx + tact,i,0],pos_nc[minidx:minidx + tact,i,1],c = np.arange(tact),cmap = col_nc_fut,s = 0.1,alpha = 1.0)
        # if t == ftimes-1: 
        #     #! Only when minidx > ftimes//2 and so on
        #     minidx = caus_minidx[i]
        #     curvature_caus[:] += curvature(trange[:ftimes], pos_caus[minidx -fctimes :minidx + ftimes - fctimes,i,0], pos_caus[minidx -fctimes :minidx + ftimes - fctimes,i,1])/ntraj    
            
        #     minidx = nc_minidx[i]
        #     curvature_nc[:] += curvature(trange[:ftimes], pos_nc[minidx -fctimes :minidx + ftimes - fctimes,i,0], pos_nc[minidx -fctimes :minidx + ftimes - fctimes,i,1])/ntraj    
            
    
    ax[1].set_title("NC", color = "white")
    ax[0].set_title("Caus",color = "white")
    fig.suptitle(fr"$t-t_{{min}}= {((t - ftimes//2)/st*0.1):.1f} \tau_p$",color = "white")
    # Add colorbars for both subplots
    norm = mpl.colors.Normalize(vmin = -ftimes//2,vmax = ftimes//2)
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap2), ax=ax[0], orientation='horizontal',shrink = 0.5,norm = norm)
    cbar1.set_label(r'$t - t_{min}$', color='white')
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax[1], orientation='horizontal',shrink = 0.5,norm = norm)
    cbar2.set_label(r'$t - t_{min}$', color='white')
    plt.savefig(movies/f"traj_lr_{alph:.2f}_frame_{t}.png",dpi = 300,bbox_inches = "tight")
    fig.tight_layout()
    plt.close()
    print(t,end = '\r')

#%%
# for alph in alphs:
#     os.system(f'bash -c "source ~/.bashrc && cd {movies} && ffmpeg -framerate 8 -start_number 0 -i traj_lr_{alph:.2f}_frame_%d.png -vf \\"pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2\\" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart traj_tiny_alph_{alph:.2f}.mp4 -y && mv traj_tiny_alph_{alph:.2f}.mp4 ../ && rm -r *lr_{alph:.2f}_*"')
# %%

#%%
# plt.plot((np.arange(ftimes)- ftimes//2)*0.1/st,curvature_caus,color = "red", label = "Caus")
# plt.plot((np.arange(ftimes)- ftimes//2)*0.1/st,curvature_nc,color = "blue", label = "NC")
# plt.xlabel(r"$(t - t_{min})/\tau_p$")
# plt.ylabel("Curvature")
# # plt.ylim(None,10)
# plt.yscale("log")
# plt.legend()
# plt.savefig(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/curvature_alph_{alph:.2f}.png",dpi = 300,bbox_inches = "tight")

# %%
# with h5py.File(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/curvature_data.hdf5","a") as f:
#     if f[f'alph_{alph:.2f}'].exists(): del f[f'alph_{alph:.2f}']
#     grp = f.create_group(f"alph_{alph:.2f}")
#     grp = f[f"alph_{alph:.2f}"]
#     grp.create_dataset("curvature_caus",data = curvature_caus)
#     grp.create_dataset("curvature_nc",data = curvature_nc)
#     grp.attrs["tp"] = tp
# # %%
