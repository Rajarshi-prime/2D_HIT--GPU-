import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pathlib,sys,os,json
from cupyx.scipy.fft import rfft2,irfft2
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
mpl.rc("text", usetex = True)
import h5py
import cupy as cp

## ---------------- params ----------------------- ## 
paramfile = 'parameters.json'
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
alph = params["alph"] # The density ratio
Nprtcl = int(params["Nprtcl"]*Nx*Ny) # Number of particles
tf = params["tf"] # Final time for the particles
st = params["st"]*tf # Time period for the particles
tp = st # Save the particle data after this many timesteps
savestep = int(params["savestep"]/dt) # Save the data after this many timesteps
prtcl_savestep = int(params["prtcl_savestep"]/dt) # Save the particle data after this many timesteps
eta = params["eta"]/(Nx//3) # Desired Kolmogorov length scale
restart = params["restart"] # Restart from the last saved data
kf = params["kf"] # Forcing wavenumber
f0 = params["f0"] # Forcing amplitude
order = params["order"] # Order of the spline interpolation
# t = np.arange(0,1.1*dt + T,dt) # Time array
P = (nu**3/eta**4)/len(shell_count) # power per unit shell
print(f"Number of particles: {Nprtcl}, stokes number in simulation time {st}")
## ----------------------------------------------- ##

print(f"alpha {alph}")

## ---------- Grid and wavenumbers --------------- ##
Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
x,y = np.meshgrid(X,Y,indexing="ij")
PI = np.pi
TWO_PI = 2*PI

## It is best to define the function which returns the real part of the iifted function as ifft. 
ifft2 = lambda x,axis = None: irfft2(x,(Nx,Ny),axes = axis)

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
## ----------------------------------------------- ##


# %%
def find_range(x,idx): return np.sum(np.cumsum(x[:idx][::-1]) == range(1,len(x[:idx])+1))+np.sum(np.cumsum(x[idx:])== range(1,len(x[idx:])+1))

Mmat = np.array([
    [0,-7/15,4/5,-1/3],
    [1,-1/5,-9/5,1],
    [0,4/5,6/5,-1],
    [0,-2/15,-1/5,1/3]    
]) # The matrix for the spline interpolation
idx = np.zeros((Nprtcl,d),dtype = int)
delx = np.zeros((Nprtcl,d),dtype = np.float64)
poly = np.zeros((Nprtcl,order,d),dtype = np.float64)
Lx = Ly = TWO_PI
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
def interp_spline(pos,s_field,smat):
    smat[:] = 0.0
    idx = np.zeros(pos.shape,dtype = int)
    delx = np.zeros(pos.shape,dtype = float)
    poly = np.zeros((pos.shape[0],order,d),dtype = float)
    
    idx[:] = (pos/dx).astype(int)
    delx[:] = (pos%TWO_PI - X[idx%N])/dx
    # print(smat.shape,poly.shape,delx.shape)
    for i in range(order): 
        poly[:,i] = delx**i
    for i in range(order):
        for j in range(order):
            # temparr[:] = u_field[((idx[:,0]-1 + i)%N),((idx[:,1]-1 + j)%N),...] #! Not giving for saving memories and variables
            # print(s_field[((idx[:,0]-1 + i)%N),((idx[:,1]-1 + j)%N),...].shape)
            # print(s_field[((idx[:,0]-1 + i)%N),((idx[:,1]-1 + j)%N),...].shape, "hello")
            smat  += s_field[((idx[:,0]-1 + i)%N),((idx[:,1]-1 + j)%N),...]*np.einsum('p,q,...p,...q->...',Mmat[i],Mmat[j],poly[...,0],poly[...,1])
            
            
    return smat
# %%
curr_path = pathlib.Path(__file__).parent 
iname = "bspline" if order >1 else "linear"
savePlot = pathlib.Path(f"/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/Plots/iname/Re_{np.round(Re,2)},dt_{dt},N_{N}/")
savePlot.mkdir(parents=True, exist_ok=True)
loadPath = pathlib.Path(f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/")
prtcl_loadPath = loadPath/f"alpha_{alph:.2}_prtcl/St_{st/tf:.2f}/"
loadPath.exists()
# loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/Caustics3D/data_new/fluid/omg/alph_0.7000/st_0.00848/dt_8.48e-06")
try: savePlot.mkdir(parents=True, exist_ok=False)
except FileExistsError: pass
savePlotdataPath = savePlot.parent/f"plotdat"
try: savePlotdataPath.mkdir(parents = True, exist_ok = False)
except FileExistsError: pass

os.listdir(loadPath)

times = np.arange(0,T+ 0.1,params["savestep"])

#! Goal -- load files one by one, extract caustics and non-caustics particles.
"""Check and store statistics. And be done with it
Stats to store: 
The eigenvaules series for caustics particles.
The time of caustics

"""

print(times[-1],st,str(savePlot))
Ntimes = len(times)
TrZ = np.zeros((Ntimes, Nprtcl))
TrZ2 = np.zeros((Ntimes,Nprtcl))
Z = np.zeros((Nprtcl,d,d))
caus_count = np.zeros((Ntimes, Nprtcl))
vel = np.zeros((Ntimes, Nprtcl,d))
pos = np.zeros((Ntimes, Nprtcl,d))
Qmean = np.zeros((Ntimes))
Qstd = np.zeros((Ntimes))
Qinterp = np.zeros((Ntimes, Nprtcl))
siginterp = np.zeros((Ntimes, Nprtcl))
xiinterp = np.zeros((Ntimes, Nprtcl))
causQmean = np.zeros(Ntimes)
causQstd = np.zeros(Ntimes)
Qmax = 0.5
Qmin = -0.5
Qbins = np.linspace(Qmin,Qmax, 6001)
Q_field_pdf = np.zeros((Ntimes,len(Qbins)-1))
Q_particle_pdf = np.zeros((Ntimes,len(Qbins)-1))
Q_caus_pdf = np.zeros((Ntimes,len(Qbins)-1))
xi = np.zeros((Nx,Ny//2+1),dtype = np.complex128)
A = np.zeros((d,d,Nx,Ny//2+1),dtype = np.complex128)
u = xi.copy()
v = xi.copy()
xi_r = np.zeros((Nx,Ny),dtype = np.float64)
Q_field = xi_r.copy()
sig_field = xi_r.copy()
Ar = np.zeros((d,d,Nx,Ny),dtype = np.float64)
# eiga = np.zeros((Ntimes, Nprtcl,3)).astype(complex)
# eigz = np.zeros((Ntimes, Nprtcl,3)).astype(complex)
# causeiga = []
# causeigz = [] 
# causTrZ = []
# causvel = []
# causno = 0
# Q_llim = -0.3
# Q_ulim = 0.6
# R_llim = -0.03
# R_ulim = 0.03
# tbins = np.linspace(times[0],times[-1],2001)
# Qbins = np.linspace(Q_llim,Q_ulim, 201)
# Rbins = np.linspace(R_llim,R_ulim, 201)
# Qpdf = np.zeros((len(tbins) -1,len(Qbins) -1))
# Rpdf = np.zeros((len(tbins) -1,len(Rbins) -1))
# initQ = np.zeros((Nprtcl))
# initR = np.zeros((Nprtcl))
tip = np.zeros(Nprtcl)
tfp = np.zeros(Nprtcl)
tis = np.array([])
tfs = np.array([])
prtcls = np.array([])
xi = np.zeros((Nx,Ny//2+1),dtype = np.complex128)
xi_r = np.zeros((Nx,Ny),dtype = np.float64)
u = np.zeros((Nx,Ny//2+1),dtype = np.complex128)
v = np.zeros((Nx,Ny//2+1),dtype = np.complex128)
Q_field = np.zeros((Nx,Ny),dtype = np.float64)
for i,t in enumerate(times):
    print(f"loading time {t}",end='\r')
    Z[:] =  np.load(prtcl_loadPath/f"time_{t:.2f}/prtcl_Z.npz")["Zmatrix"]
    # Z[:] =  np.load(prtcl_loadPath/f"time_{t:.2f}/prtcl_Z.npy")
    TrZ[i] = np.einsum('...ii->...', Z)
    TrZ2[i] = np.einsum('...ij,...ji->...', Z,Z)
    caus_count[i] =  np.load(prtcl_loadPath/f"time_{t:.2f}/caus_count.npz")["caustics_count"]
    # caus_count[i] =  np.load(prtcl_loadPath/f"time_{t:.2f}/caus_count.npy")
    final_caus_count = np.load(prtcl_loadPath/f"time_{times[-1]:.2f}/caus_count.npz")["caustics_count"]
    # final_caus_count = np.load(prtcl_loadPath/f"time_{times[-1]:.2f}/caus_count.npy")
    # print(f"final_caus_count: {np.sum(final_caus_count>0)}")
    newcaus_idx = np.argwhere(caus_count[i] > caus_count[i-1]) if i>0 else np.array([])
    if newcaus_idx.size>0:
        # print(newcaus_idx.size)
        tfp[newcaus_idx] = t
        tis = np.append(tis,tip[newcaus_idx])
        tfs = np.append(tfs,tfp[newcaus_idx])
        prtcls = np.append(prtcls,newcaus_idx)
        tip[newcaus_idx] = t


    # vel[i] = np.load(prtcl_loadPath/f"time_{t:.2f}/vel.npy") 
    vel[i] = np.load(prtcl_loadPath/f"time_{t:.2f}/vel.npz")["vel"]
    # pos[i] = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npy")
    pos[i] = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npz")["pos"]
    
    xi[:] = np.load(loadPath/f"time_{t:.2f}/w.npz")["vorticity"]
    xi_r[:] = ifft2(cp.array(xi)).get()
    # print(xi_r.max(),np.sqrt(np.mean(xi_r**2)))
    u[:] = 1j* ky*lapinv*xi
    v[:] = -1j*kx*lapinv*xi
    # ur = ifft2(u) 
    # vr = ifft2(v)
    A[0,0] = 1j*kx*u
    A[0,1] = 1j*ky*u
    A[1,0] = 1j*kx*v
    A[1,1] = 1j*ky*v
    Q_field[:] = 0.0
    Ar[:] = ifft2(cp.array(A),axis = (2,3))
    # for ii in range(d):
    #     for jj in range(d):
    #         Ar[ii,jj] = ifft2(cp.array(A[ii,jj])).get()

    Q_field[:] = -0.5*np.einsum('ij...,ji...->...',Ar,Ar)
    sig_field[:] = (xi_r**2 - 4*Q_field)**0.5
    Q_field_pdf[i,:]  = np.histogram(st**2*Q_field.ravel(),bins = Qbins)[0]/(Nx*Ny)
    if i == 0:
        xi_rms = st*np.sqrt(np.mean(st*xi_r**2))
        sig_rms = st*np.sqrt(np.mean(sig_field**2))
        Q_rms = st**2*np.sqrt(np.mean(Q_field**2))
    
    # print(Q_field.shape)
    # print(np.max(Q_field),np.min(Q_field),np.mean(Q_field))
    # pos_test = np.array([(x.ravel())[::2],(y.ravel())[::2]]).T + np.random.random((Nx*Ny//2,2))*dx*0.1
    # Qinterp_test = np.zeros(pos_test.shape[0])
    # Qinterp_test[:] = interp_spline(pos_test,Q_field,Qinterp_test)
    # print(np.min(Qinterp_test),np.mean(Qinterp_test),np.max(Qinterp_test))
    Qinterp[i,:] = st**2*interp_spline(pos[i],Q_field,Qinterp[i,:]) 
    xiinterp[i,:] = st*interp_spline(pos[i],xi_r,xiinterp[i,:])
    
    Q_particle_pdf[i,:] = np.histogram(Qinterp[i,:],bins = Qbins)[0]/Nprtcl
    #print(np.min(Qinterp[i,:]),np.mean(Qinterp[i,:]),np.max(Qinterp[i,:]))
    #print(i)
    causidx = np.argwhere(caus_count[i] > 0)

    if caus_count[i].any() > 0:
        causQmean[i] = np.mean(Qinterp[i,causidx])
        causQstd[i] = np.std(Qinterp[i,causidx])
        Q_caus_pdf[i,:] = np.histogram(Qinterp[i,causidx],bins = Qbins)[0]/np.sum(caus_count[i] > 0)
    # Omean[i] = np.mean(Qinterp)
    # Qstd[i] = np.std(Qinterp)
# print(Qinterp)
# Q_caus = Qinterp[:,caus_count[-1] > 0]

del pos,vel
# %%
maxdt = np.max(tfs-tis)
maxsize = np.sum(times<= maxdt) + 1
Q_caus_shifted = np.ones((maxsize,int(np.sum(caus_count[-1]))))*65536.
omg_caus_shifted = np.ones((maxsize,int(np.sum(caus_count[-1]))))*65536.
TrZ_shifted = np.ones((maxsize,int(np.sum(caus_count[-1]))))*65536.
TrZ2_shifted = np.ones((maxsize,int(np.sum(caus_count[-1]))))*65536.
t_caus_len = np.zeros(int(np.sum(caus_count[-1])))
Q_initial = np.zeros(int(np.sum(caus_count[-1])))
omg_initial = np.zeros(int(np.sum(caus_count[-1])))

if (prtcl_loadPath/f"caus-details.hdf5").exists():
    os.remove(prtcl_loadPath/f"caus-details.hdf5")

with h5py.File(prtcl_loadPath/f"caus-details.hdf5",'w') as f:
    # try:
    caus_ratio = f.create_dataset("Caustics_ratio",data = (np.sum(caus_count>0,axis = 1)/Nprtcl),dtype = np.float64)
    times = f.create_dataset("times",data = times,dtype = np.float64)
    caus_new = np.sum(caus_count>0,axis = 1)/Nprtcl -np.roll(np.sum(caus_count>0,axis = 1)/Nprtcl ,1)
    caus_new[0] = 0
    newcaus = f.create_dataset("new_caus",data =  caus_new,dtype = np.float64)
    caus_same = np.sum((caus_count >1)*(caus_count > np.roll(caus_count,1,axis = 0)),axis = 1)/Nprtcl
    caus_same[0] = 0
    samecaus = f.create_dataset("same_caus",data = caus_same,dtype = np.float64)
    
    meanQ = np.mean(Qinterp,axis = 1)
    print(meanQ)
    stdQ = np.std(Qinterp,axis = 1)
    print(np.min(stdQ),np.max(stdQ))
    Qmean = f.create_dataset("Qmean",data = meanQ,dtype = np.float64)
    Qstd = f.create_dataset("Qstd",data = stdQ,dtype = np.float64)

    causQmean = f.create_dataset("causQmean",data = causQmean,dtype = np.float64)
    causQstd = f.create_dataset("causQstd",data = causQstd,dtype = np.float64)
    
    Q_field_pdf = f.create_dataset("Q_field_pdf",data = Q_field_pdf,dtype = np.float64)
    Q_particle_pdf = f.create_dataset("Q_particle_pdf",data = Q_particle_pdf,dtype = np.float64)
    Q_caus_pdf = f.create_dataset("Q_caus_pdf",data = Q_caus_pdf,dtype = np.float64)

del caus_ratio,caus_new, caus_same,causQmean,causQstd,Qmean,Qstd,Q_field_pdf,Q_particle_pdf,Q_caus_pdf
# del causQmean,causQstd,Qmean,Qstd,Q_field_pdf,Q_particle_pdf,Q_caus_pdf

causidx = np.argwhere(caus_count[-1] > 0)
# print(causidx)
ins = 0
print(f"Maxsize: {maxsize}")
for i in causidx.ravel():
    # print(i)
    # print(caus_count[-1,i])
    """
    For each particle, cau_count will give how many instances to split the particle into. 
    Then for each of the instances, add the Q in the Q_caus_pdf. 
    """
    # print(caus_count[-1,i])
    for j in range(int(caus_count[-1,i])):
        region = ((caus_count[:,i] > j-1)*(caus_count[:,i] <= j)    ).ravel() #region in time where this caustics happened
        regionlen = np.sum(region)
        try:
            t_caus_len[ins] = regionlen
            Q_caus_shifted[(maxsize-regionlen):,ins] = Qinterp[region,i] # Adding the trajectory of the particle and rest zero.
            Q_initial[ins] = Q_caus_shifted[(maxsize-regionlen),ins]
            omg_caus_shifted[(maxsize-regionlen):,ins] = xiinterp[region,i]
            omg_initial[ins] = omg_caus_shifted[(maxsize-regionlen),ins]
            TrZ_shifted[(maxsize-regionlen):,ins] = TrZ[region,i]
            TrZ2_shifted[(maxsize-regionlen):,ins] = TrZ2[region,i]
        except:
            print((maxsize-regionlen), regionlen,ins, Q_caus_shifted.shape)
            raise SystemExit

        ins = ins + 1
del Qinterp,xiinterp
print(caus_count.shape)
    
print(f"fraction of caustics: {np.sum(caus_count,axis = 1)/Nprtcl} at stokes number {st/tf}")
def first_value(x):
    """This returns the first non-zero value of the shifted Q_caus"""
    
    diff_gate = np.cumsum((x !=0)*1,axis = 0)
    return x[diff_gate == 1]
    
sig_initial = np.sqrt(omg_initial**2 - 4*Q_initial)**0.5
Q_caus_shifted_vort = Q_caus_shifted[:,Q_initial>0]
Q_caus_shifted_strain = Q_caus_shifted[:,Q_initial<0]
Q_caus_shifted_extreme_vort = Q_caus_shifted[:,np.abs(omg_initial)>2*xi_rms]
Q_caus_shifted_extreme_strain = Q_caus_shifted[:,sig_initial>2*sig_rms]
Q_caus_shifted_extreme_pos_Q = Q_caus_shifted[:,Q_initial>4*Q_rms]
Q_caus_shifted_extreme_neg_Q = Q_caus_shifted[:,Q_initial<-4*Q_rms]

print(Q_caus_shifted_strain.shape,Q_caus_shifted_vort.shape,Q_caus_shifted_extreme_vort.shape, Q_caus_shifted_extreme_strain.shape, Q_caus_shifted_extreme_pos_Q.shape,Q_caus_shifted_extreme_neg_Q.shape
)
with h5py.File(prtcl_loadPath/f"caus-details.hdf5",'a') as f:
    
    Q_caus_shifted = f.create_dataset("Q_caus_shifted",data = Q_caus_shifted,dtype = np.float64,compression = 'gzip')
    print(f"Saved the data in {str(prtcl_loadPath)}")
    omg_caus_shifted = f.create_dataset("omg_caus_shifted",data = omg_caus_shifted,dtype = np.float64,compression = 'gzip')
    TrZ_shifted = f.create_dataset("TrZ_shifted",data = TrZ_shifted,dtype = np.float64,compression = 'gzip')
    TrZ2_shifted = f.create_dataset("TrZ2_shifted",data = TrZ2_shifted,dtype = np.float64,compression = 'gzip')
    Q_caus_shifted_vort = f.create_dataset("Q_caus_shifted_vort",data = Q_caus_shifted_vort,dtype = np.float64,compression = 'gzip')
    Q_caus_shifted_strain = f.create_dataset("Q_caus_shifted_strain",data = Q_caus_shifted_strain,dtype = np.float64,compression = 'gzip')
    Q_caus_shifted_extreme_vort = f.create_dataset("Q_caus_shifted_extreme_vort",data = Q_caus_shifted_extreme_vort,dtype = np.float64,compression = 'gzip')
    Q_caus_shifted_extreme_strain = f.create_dataset("Q_caus_shifted_extreme_strain",data = Q_caus_shifted_extreme_strain,dtype = np.float64,compression = 'gzip')
    Q_caus_shifted_extreme_pos_Q = f.create_dataset("Q_caus_shifted_extreme_pos_Q",data = Q_caus_shifted_extreme_pos_Q,dtype = np.float64,compression = 'gzip')
    Q_caus_shifted_extreme_neg_Q = f.create_dataset("Q_caus_shifted_extreme_neg_Q",data = Q_caus_shifted_extreme_neg_Q,dtype = np.float64,compression = 'gzip')
    t_caus_len = f.create_dataset("t_caus_len",data = t_caus_len,dtype = np.float64,compression = 'gzip')
    print(f"Saved the data in {str(prtcl_loadPath)}")


# %%
