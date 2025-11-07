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
from scipy.interpolate import splrep, splev, splder,sproot,PPoly
from scipy.signal import find_peaks
mpl.rc("text", usetex = True)
from matplotlib.legend_handler import HandlerLine2D

aspect = 8/397.5
default_fontsize = round(16*72*aspect)
fraction_default_fontsize = 1.5*default_fontsize
mpl.rcParams['font.size'] = default_fontsize
mpl.rcParams['axes.labelsize'] = default_fontsize
mpl.rcParams['xtick.labelsize'] = default_fontsize
mpl.rcParams['ytick.labelsize'] = default_fontsize
mpl.rcParams['legend.fontsize'] = default_fontsize
mpl.rcParams['axes.formatter.use_locale'] = False  # Ensure consistent formatting
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = (-1, 2)
mpl.rcParams['lines.linewidth'] = 3.0
mpl.rc('axes', linewidth=0.5)
mpl.rc("text", usetex = True)
# %%

#%%
idx = int(float(sys.argv[-1]))
# idx = 8
alph_idx = idx%8
st_idx = idx//8
#%%

## ---------------- params ----------------------- ## 
paramfile ='/home/rajarshi.chattopadhyay/pokemon_data/codes/parameters.json'
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
curr_path = pathlib.Path("/home/rajarshi.chattopadhyay/pokemon_data/")
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
def unmodulo(arr,base = TWO_PI):
    """
    Checks if the points have difference more than PI. Then it adds two pi to the all the next points.
    """
    
    cond = np.zeros_like(arr)
    cond[1:] = np.nancumsum(np.sign(arr[:-1] -arr[1:] )*(np.abs(arr[1:] - arr[:-1]) > base/2))* base
    return cond + arr

#%%
def curvature(t,x,y):
    """calculates the curvature of x(t) and y(t)"""
    dxdt = np.gradient(x,t)
    dydt = np.gradient(y,t)
    v = (dxdt**2 + dydt**2)**0.5
    d2xdt2 = np.gradient(dxdt,t)
    d2ydt2 = np.gradient(dydt,t)
    return np.abs(dxdt*d2ydt2 - dydt*d2xdt2)/(v**3 + 1e-16)

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

#%%
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
#%%

#%%
with h5py.File(prtcl_loadPath/'caus-details.hdf5','r') as f:
    print(f.keys())

    pos_caus_shifted = f['pos_caus_shifted'][:]
    pos_nc_shifted = f['pos_nc_shifted'][:]
    vel_caus_shifted = f['vel_caus_shifted'][:]
    u_caus_shifted = f['u_caus_shifted'][:]
    vel_nc_shifted = f['vel_nc_shifted'][:]
    u_nc_shifted = f['u_nc_shifted'][:]
    Q_nc_shifted = f['Q_nc_shifted'][:]
    Q_caus_shifted = f['Q_caus_shifted'][:]
    TrZ_caus_shifted = f['TrZ_shifted'][:]
    TrZ_nc_shifted = f['TrZ_nc_shifted'][:]

    
    a_caus = f['a_caus_shifted'][:]
    b_caus = f['b_caus_shifted'][:]
    c_caus = f['c_caus_shifted'][:]
    a_nc = f['a_nc_shifted'][:]
    b_nc = f['b_nc_shifted'][:]
    c_nc = f['c_nc_shifted'][:]
    
    sig_s_caus = b_caus + c_caus
    sig_s_nc = b_nc + c_nc
    sig_n_caus = 2*a_caus
    sig_n_nc = 2*a_nc
    omg_caus = c_caus - b_caus
    omg_nc = c_nc - b_nc
    
    # t_mins = f['t_mins'][:]

#%%
t_mins = np.zeros(Q_caus_shifted.shape[1])
Q_mins = np.zeros(Q_caus_shifted.shape[1])
for ins in range(Q_caus_shifted.shape[1]):
    region = ~np.isnan(Q_caus_shifted[:,ins])
    t_local = np.arange(region.sum())*0.1
    t_local -= t_local[-1]
    t_mins[ins],Q_mins[ins],_ = compute_prev_minima(t_local,Q_caus_shifted[region,ins],0)
#%% 
Q_nc_shifted.shape,t_mins.shape,(np.isnan(Q_nc_shifted).sum(axis = (0))==800).sum()
#%%

idx_caus_min = (t_mins*10).astype(np.int32) - 1
idx_caus_min = np.where(idx_caus_min < -400, -1, idx_caus_min)
#* Take only those whose idx_caus_min is less than -20. 
cond = idx_caus_min < -1
pos_caus_shifted = pos_caus_shifted[:,cond,:]
vel_caus_shifted = vel_caus_shifted[:,cond,:]
u_caus_shifted = u_caus_shifted[:,cond,:]
Q_caus_shifted = Q_caus_shifted[:,cond]
TrZ_caus_shifted = TrZ_caus_shifted[:,cond]
idx_caus_min = idx_caus_min[cond]
sig_s_caus = sig_s_caus[:,cond] 
sig_n_caus = sig_n_caus[:,cond] 
omg_caus = omg_caus[:,cond] 
idx_nc_min = (np.ones(len(Q_nc_shifted[-1]))*400 - len(Q_nc_shifted)).astype(np.int32)
cond = np.isnan(Q_nc_shifted).sum(axis = (0))!=800
pos_nc_shifted = pos_nc_shifted[:,cond,:]
vel_nc_shifted = vel_nc_shifted[:,cond,:]
u_nc_shifted = u_nc_shifted[:,cond,:]
Q_nc_shifted = Q_nc_shifted[:,cond]
TrZ_nc_shifted = TrZ_nc_shifted[:,cond]
sig_s_nc = sig_s_nc[:,cond]
sig_n_nc = sig_n_nc[:,cond]
omg_nc = omg_nc[:,cond]
idx_nc_min = idx_nc_min[cond]
numcaus = len(idx_caus_min)
numnc = len(idx_nc_min)
#%%
Q_nc_shifted.shape,t_mins.shape,(np.isnan(Q_nc_shifted).sum(axis = (0))==800).sum()
#%%
eneg_caus = np.moveaxis(np.array([sig_n_caus - (sig_n_caus**2 + sig_s_caus**2)**0.5,sig_s_caus]),0,-1)
eneg_caus /= np.linalg.norm(eneg_caus,axis = -1)[...,None]

epos_caus = np.moveaxis(np.array([sig_n_caus + (sig_n_caus**2 + sig_s_caus**2)**0.5,sig_s_caus]),0,-1)
epos_caus /= np.linalg.norm(epos_caus,axis = -1)[...,None]
#%%
eneg_nc = np.moveaxis(np.array([sig_n_nc - (sig_n_nc**2 + sig_s_nc**2)**0.5,sig_s_nc]),0,-1)
eneg_nc /= np.linalg.norm(eneg_nc,axis = -1)[...,None]

epos_nc = np.moveaxis(np.array([sig_n_nc + (sig_n_nc**2 + sig_s_nc**2)**0.5,sig_s_nc]),0,-1)
epos_nc /= np.linalg.norm(epos_nc,axis = -1)[...,None]
#%%
Q_nc_shifted.shape,t_mins.shape,(np.isnan(Q_caus_shifted).sum(axis = (0))==400).sum()

#%%
def hat(vect):
    return vect/np.linalg.norm(vect,axis = -1)[...,None]
# %%
#%% 
#? Translating and rotating the co-ordinates. 

for i in range(numcaus):
    for m in range(2):
        pos_caus_shifted[:,i,m] = unmodulo(pos_caus_shifted[:,i,m])
        
    pos_caus_shifted[:,i,:] = rescale(pos_caus_shifted[:,i,:], idx_caus_min[i],vel_caus_shifted[:,i,:])
    
for i in range(numnc):
    for m in range(2):
        pos_nc_shifted[:,i,m] = unmodulo(pos_nc_shifted[:,i,m])
    pos_nc_shifted[:,i,:] = rescale(pos_nc_shifted[:,i,:],idx_nc_min[i],vel_nc_shifted[:,i,:])
#%%
ftimes = 350
fctimes = ftimes - ftimes//2
trange = np.arange(ftimes)
trange = (trange - ftimes//2)*0.1
veneg_caus = np.ones((ftimes,numcaus))*np.nan
veneg_nc = np.ones((ftimes,numnc))*np.nan
ueneg_caus = np.ones((ftimes,numcaus))*np.nan
ueneg_nc = np.ones((ftimes,numnc))*np.nan
vepos_caus = np.ones((ftimes,numcaus))*np.nan
vepos_nc = np.ones((ftimes,numnc))*np.nan
uepos_caus = np.ones((ftimes,numcaus))*np.nan
uepos_nc = np.ones((ftimes,numnc))*np.nan
Q_caus = np.ones((ftimes,numcaus))*np.nan
Q_nc = np.ones((ftimes,numnc))*np.nan
sigs_caus = np.ones((ftimes,numcaus))*np.nan
sigs_nc = np.ones((ftimes,numnc))*np.nan
sign_caus = np.ones((ftimes,numcaus))*np.nan
sign_nc = np.ones((ftimes,numnc))*np.nan
om_caus = np.ones((ftimes,numcaus))*np.nan
om_nc = np.ones((ftimes,numnc))*np.nan
TrZ_caus = np.ones((ftimes,numcaus))*np.nan
TrZ_nc = np.ones((ftimes,numnc))*np.nan
vtheta_caus = np.ones((ftimes,numcaus))*np.nan
vtheta_nc = np.ones((ftimes,numnc))*np.nan
utheta_caus = np.ones((ftimes,numcaus))*np.nan
utheta_nc = np.ones((ftimes,numnc))*np.nan


curvature_caus = np.zeros(ftimes)
curvature_nc = np.zeros(ftimes)
vrms_caus = np.zeros(ftimes)
vrms_nc = np.zeros(ftimes)
count_c = np.zeros(ftimes)
count_nc = np.zeros(ftimes)
#%%
"""
Take only the notnan part. Subtract from the minimum index. 
"""
caus_max = 0.
nc_max = 0.
std_caus = 0.
std_nc = 0.
for i in range(numcaus):
    
    minidx = idx_caus_min[i] + len(Q_caus_shifted)
    notnan = ~np.isnan(pos_caus_shifted[:,i,0])
    firstnotnan = np.argwhere(np.cumsum(1.0*notnan) == 1).ravel()[0]
    trajnotnan = pos_caus_shifted[notnan,i,:]
    minidx = minidx -firstnotnan
    endidx = minidx +  min(ftimes - fctimes, len(trajnotnan[minidx:minidx + ftimes - fctimes]))
    startidx = max(minidx -fctimes,0)
    time = np.arange(startidx ,endidx)
    
    velnotnan = vel_caus_shifted[notnan,i,:]
    unotnan = u_caus_shifted[notnan,i,:]
    Qnotnan = Q_caus_shifted[notnan,i]
    sig_snotnan = sig_s_caus[notnan,i]
    sig_nnotnan = sig_n_caus[notnan,i]
    omg_notnan = omg_caus[notnan,i]
    TrZ_notnan = TrZ_caus_shifted[notnan,i]
    
    Q_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = Qnotnan[startidx:endidx]
    sigs_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = sig_snotnan[startidx:endidx]
    sign_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = sig_nnotnan[startidx:endidx]
    om_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = omg_notnan[startidx:endidx]
    TrZ_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = TrZ_notnan[startidx:endidx]

  
    
    
    eneg_notnan = eneg_caus[notnan,i]
    
    veneg_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(velnotnan)[startidx:endidx],eneg_notnan[startidx:endidx])
    ueneg_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(unotnan)[startidx:endidx],eneg_notnan[startidx:endidx])
    
    epos_notnan = epos_caus[notnan,i]
    
    vepos_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(velnotnan)[startidx:endidx],epos_notnan[startidx:endidx])
    uepos_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(unotnan)[startidx:endidx],epos_notnan[startidx:endidx])
    
    
    vtheta_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.arctan2(vepos_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i],veneg_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i])
    utheta_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.arctan2(uepos_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i],ueneg_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i])
    
    caus_max = max(np.abs(np.einsum('...i,...i->...', eneg_notnan, epos_notnan)).max(),caus_max)
    
    std_caus += np.std(vepos_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i]**2 + veneg_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i]**2)
    time = np.arange(startidx ,endidx)
    curvature_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature(time, trajnotnan[startidx: endidx,0], trajnotnan[startidx:endidx,1])
    vrms_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += np.sum(velnotnan[startidx:endidx]**2, axis = -1)
    
    count_c[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += 1
curvature_caus /= count_c + 1e-16
vrms_caus = (vrms_caus/(count_c + 1e-16))**0.5
for i in range(numnc):
    minidx = idx_nc_min[i]+ len(Q_nc_shifted)
    notnan = ~np.isnan(pos_nc_shifted[:,i,0])
    firstnotnan = np.argwhere(np.cumsum(1.0*notnan) == 1).ravel()[0]
    trajnotnan = pos_nc_shifted[notnan,i,:]
    minidx = minidx -firstnotnan
    endidx = minidx +  min(ftimes - fctimes, len(trajnotnan[minidx:minidx + ftimes - fctimes]))
    startidx = max(minidx -fctimes,0)
    time = np.arange(startidx ,endidx)
    
    velnotnan = vel_nc_shifted[notnan,i,:]
    unotnan = u_nc_shifted[notnan,i,:]
    Qnotnan = Q_nc_shifted[notnan,i]
    sig_snotnan = sig_s_nc[notnan,i]
    sig_nnotnan = sig_n_nc[notnan,i]
    omg_notnan = omg_nc[notnan,i]
    TrZ_notnan = TrZ_nc_shifted[notnan,i]
    
    Q_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = Qnotnan[startidx:endidx]
    sigs_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = sig_snotnan[startidx:endidx]
    sign_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = sig_nnotnan[startidx:endidx]
    om_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = omg_notnan[startidx:endidx]
    TrZ_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = TrZ_notnan[startidx:endidx]
    
    eneg_notnan = eneg_nc[notnan,i]
    
    veneg_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(velnotnan)[startidx:endidx],eneg_notnan[startidx:endidx])
    ueneg_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(unotnan)[startidx:endidx],eneg_notnan[startidx:endidx])
    
    epos_notnan = epos_nc[notnan,i]
    
    vepos_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(velnotnan)[startidx:endidx],epos_notnan[startidx:endidx])
    uepos_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.einsum('...i,...i->...',hat(unotnan)[startidx:endidx],epos_notnan[startidx:endidx])
    
    vtheta_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.arctan2(vepos_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i],veneg_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i])
    
    utheta_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i] = np.arctan2(uepos_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i],ueneg_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i])
    
    
    nc_max = max(np.abs(np.einsum('...i,...i->...', eneg_notnan, epos_notnan)).max(),nc_max)
    
    std_nc += np.std(vepos_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i]**2 + veneg_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx),i]**2)
    
    time = np.arange(startidx ,endidx)
    curvature_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature(time, trajnotnan[startidx:endidx,0], trajnotnan[startidx:endidx,1])
    vrms_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += np.sum(velnotnan[startidx:endidx]**2,axis = -1)
    count_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += 1
curvature_nc /= count_nc + 1e-16
vrms_nc = (vrms_nc/(count_nc + 1e-16))**0.5
#%%
caus_max, nc_max, std_nc, std_caus
#%%
thtbins = np.linspace(0,TWO_PI,36)
thtvals = 0.5*(thtbins[1:]+ thtbins[:-1])
vbins = np.linspace(-0.05,1.05,51)
ubins = np.linspace(-0.05,1.05,51)
vvals = 0.5*(vbins[1:] + vbins[:-1])
uvals = 0.5*(ubins[1:] + ubins[:-1])

Q_caus_mean = np.nanmean(Q_caus,axis = -1)
Q_nc_mean = np.nanmean(Q_nc,axis = -1)
TrZ_caus_mean = np.nanmean(TrZ_caus,axis = -1)
TrZ_nc_mean = np.nanmean(TrZ_nc,axis = -1)
sigs_caus_mean = np.nanmean(np.abs(sigs_caus),axis = -1)
sigs_nc_mean = np.nanmean(np.abs(sigs_nc),axis = -1)
sign_caus_mean = np.nanmean(np.abs(sign_caus),axis = -1)
sign_nc_mean = np.nanmean(np.abs(sign_nc),axis = -1)
omg_caus_mean = np.nanmean(np.abs(om_caus),axis = -1)
omg_nc_mean = np.nanmean(np.abs(om_nc),axis = -1)
sig2_caus_mean = np.nanmean(sigs_caus**2 + sign_caus**2,axis = -1)
sig2_nc_mean = np.nanmean(sigs_nc**2 + sign_nc**2,axis = -1)
omg2_caus_mean = np.nanmean(om_caus**2 ,axis = -1)
omg2_nc_mean = np.nanmean(om_nc**2 ,axis = -1)

#%%
def save_dset(f,dname,data):
    if dname in f:
        del f[dname]
    
    f.create_dataset(dname, data = data, dtype = np.float64, compression = 'gzip')
    return f[dname]
#%%
with h5py.File(
    f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/tm_shifted_mean_data.hdf5",
    "a"
) as f:
    try:
        grp = f.create_group(
            f"Re_{Re}_dt_{dt}_N_{N}/alpha_{alph:.2f}_st_{st/tp:.2f}/"
        )
    except ValueError:
        grp = f[f"Re_{Re}_dt_{dt}_N_{N}/alpha_{alph:.2f}_st_{st/tp:.2f}/"]

    # save all the nanmean-averaged datasets
    save_dset(grp, "t",trange)
    save_dset(grp, "Q_caus_mean", Q_caus_mean)
    save_dset(grp, "Q_nc_mean", Q_nc_mean)
    save_dset(grp, "TrZ_caus_mean", TrZ_caus_mean)
    save_dset(grp, "TrZ_nc_mean", TrZ_nc_mean)
    save_dset(grp, "sigs_caus_mean", sigs_caus_mean)
    save_dset(grp, "sigs_nc_mean", sigs_nc_mean)
    save_dset(grp, "sign_caus_mean", sign_caus_mean)
    save_dset(grp, "sign_nc_mean", sign_nc_mean)
    save_dset(grp, "omg_caus_mean", omg_caus_mean)
    save_dset(grp, "omg_nc_mean", omg_nc_mean)
    save_dset(grp, "sig2_caus_mean", sig2_caus_mean)
    save_dset(grp, "sig2_nc_mean", sig2_nc_mean)
    save_dset(grp, "omg2_caus_mean", omg2_caus_mean)
    save_dset(grp, "omg2_nc_mean", omg2_nc_mean)
    

    
    
    
    
    
    
    
raise SystemExit("Saved into hdf5 file")
#%%
cls  = ["c1121f","003049"]
cls = ["#" + c for c in cls]
cls2 = ["e63946","669bbc"]
cls2 = ["#" + c for c in cls2]
periodic = lambda x: np.append(x,x[0])

#%%
mean_caus = 0.
mean_nc = 0.
std_caus = 0.
std_nc = 0.
savePath = pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/4movies_ultimate/st_{st/tp:.2f}")
savePath.mkdir(parents = True, exist_ok= True)
fig = plt.figure(figsize=(12, 16))
gs = mpl.gridspec.GridSpec(4, 6, figure=fig)

ax11 = fig.add_subplot(gs[0, 0:3], projection='polar')
ax12 = fig.add_subplot(gs[0, 3:6], projection='polar')
ax21 = fig.add_subplot(gs[1,0:3])
ax22 = fig.add_subplot(gs[1,3:6])
ax31 = fig.add_subplot(gs[2,0:3])
ax32 = fig.add_subplot(gs[2,3:6])
ax41 = fig.add_subplot(gs[3,0:3])
ax42 = fig.add_subplot(gs[3,3:6])


for ax in [ax21,ax22,ax31,ax32,ax41,ax42]:
    ax.set_xlim(trange[0]/st,trange[-1]/st)


ax21.set_ylim(-0.25,0)
ax22.set_ylim(-0.4,0)
ax31.set_ylim(0.1,0.5)
ax32.set_ylim(0.1,0.8)
ax41.set_ylim(0.15,0.3)
ax42.set_ylim(0.5,100)
ax42.set_yscale("log")

ax11.set_theta_zero_location('N')
ax11.set_theta_direction(-1)
ax11.set_rticks([(1/ PI)**0.5],[""])
ax11.set_thetagrids([0,90,180,270],[r"$0$",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$"])
ax12.set_theta_zero_location('N')
ax12.set_theta_direction(-1)
ax12.set_rticks([(1/ PI)**0.5],[""])
ax12.set_thetagrids([0,90,180,270],[r"$0$",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$"])
labels = ["Caus", "NC"]
for i in range(ftimes):
# for i in range(2):
    print(i,end = "\r")
    vtht_caus_notnan = vtheta_caus[i,~np.isnan(vtheta_caus[i])]%TWO_PI
    utht_caus_notnan = utheta_caus[i,~np.isnan(utheta_caus[i])]%TWO_PI
    vtht_nc_notnan = vtheta_nc[i,~np.isnan(vtheta_nc[i])]%TWO_PI
    utht_nc_notnan = utheta_nc[i,~np.isnan(utheta_nc[i])]%TWO_PI
    
    vtht_caus_pdf = np.histogram(vtht_caus_notnan,bins = thtbins, density = True)[0]
    utht_caus_pdf = np.histogram(utht_caus_notnan,bins = thtbins, density = True)[0]
    vtht_nc_pdf = np.histogram(vtht_nc_notnan,bins = thtbins, density = True)[0]
    utht_nc_pdf = np.histogram(utht_nc_notnan,bins = thtbins, density = True)[0]
    
    # vepos_caus_notnan = vepos_caus[i,~np.isnan(vepos_caus[i])]
    # uepos_caus_notnan = uepos_caus[i,~np.isnan(uepos_caus[i])]
    # updf_caus = np.histogram(uepos_caus_notnan, bins = ubins[::5],density = True)[0]
    # vpdf_caus = np.histogram(vepos_caus_notnan, bins = vbins[::5],density = True)[0]
    
    # vepos_nc_notnan = vepos_nc[i,~np.isnan(vepos_nc[i])]
    # uepos_nc_notnan = uepos_nc[i,~np.isnan(uepos_nc[i])]
    # updf_nc = np.histogram(uepos_nc_notnan, bins = ubins[::5],density = True)[0]
    # vpdf_nc = np.histogram(vepos_nc_notnan, bins = vbins[::5],density = True)[0]

    # veneg_caus_notnan = veneg_caus[i,~np.isnan(veneg_caus[i])]
    # ueneg_caus_notnan = ueneg_caus[i,~np.isnan(ueneg_caus[i])]
    # updf_caus = np.histogram(ueneg_caus_notnan, bins = ubins[::5],density = True)[0]
    # vpdf_caus = np.histogram(veneg_caus_notnan, bins = vbins[::5],density = True)[0]
    
    # veneg_nc_notnan = veneg_nc[i,~np.isnan(veneg_nc[i])]
    # ueneg_nc_notnan = ueneg_nc[i,~np.isnan(ueneg_nc[i])]
    # updf_nc = np.histogram(ueneg_nc_notnan, bins = ubins[::5],density = True)[0]
    # vpdf_nc = np.histogram(veneg_nc_notnan, bins = vbins[::5],density = True)[0]
    
    
    # mean_nc += np.mean(vepos_nc_notnan**2 + veneg_nc_notnan**2)/ftimes
    # std_nc += np.std(vepos_nc_notnan**2 + veneg_nc_notnan**2)
    
    # mean_caus += np.mean(vepos_caus_notnan**2 + veneg_caus_notnan**2)/ftimes
    # std_caus += np.std(vepos_caus_notnan**2 + veneg_caus_notnan**2)
    

    # continue
    
    # h1, = ax11.plot(uvals[::5],updf_caus, label = "Caus",color = cls[0],alpha = 0.3)
    # h2,= ax11.plot(uvals[::5],updf_nc, label = "NC",color = cls[1],alpha = 0.3)    
    # ax12.plot(vvals[::5],vpdf_caus, label = "Caus",color = cls[0],alpha = 0.3)
    # ax12.plot(vvals[::5],vpdf_nc, label = "NC",color = cls[1],alpha = 0.3)
    
    h1, = ax11.plot(periodic(thtvals),periodic((2*utht_caus_pdf)**0.5), label = "Caus",color = cls[0])
    h2,= ax11.plot(periodic(thtvals),periodic((2*utht_nc_pdf)**0.5), label = "NC",color = cls[1])
    ax11.fill_between(periodic(thtvals),periodic((2*utht_caus_pdf)**0.5), alpha=0.3, color=cls2[0],lw = 0.0)
    ax11.fill_between(periodic(thtvals),periodic((2*utht_nc_pdf)**0.5), alpha=0.3, color=cls2[1],lw = 0.0)
    ax11.set_title(r'$\theta_u$',pad = 30)
    
    ax12.plot(periodic(thtvals),periodic((2*vtht_caus_pdf)**0.5), label = "Caus",color = cls[0])
    ax12.plot(periodic(thtvals),periodic((2*vtht_nc_pdf)**0.5), label = "NC",color = cls[1])
    ax12.fill_between(periodic(thtvals),periodic((2*vtht_caus_pdf)**0.5), alpha=0.3, color=cls2[0],lw = 0.0)
    ax12.fill_between(periodic(thtvals),periodic((2*vtht_nc_pdf)**0.5), alpha=0.3, color=cls2[1],lw = 0.0)
    ax12.set_title(r'$\theta_v$',pad = 30)

    
    ax21.plot(trange[:]/st,Q_caus_mean[:],color = cls[0], label = "Caus")
    ax21.plot(trange[:]/st,Q_nc_mean[:],color = cls[1], label = "NC")
    ax21.plot(trange[i]/st,Q_caus_mean[i],'o',color = cls2[0])
    ax21.plot(trange[i]/st,Q_nc_mean[i],'o',color = cls2[1])
    ax21.set_xlabel(r'$(t-t_m)/St$')
    ax21.set_ylabel(r"$\langle Q \rangle$")
    
    ax22.plot(trange[:]/st,TrZ_caus_mean[:],color = cls[0], label = "Caus")
    ax22.plot(trange[:]/st,TrZ_nc_mean[:],color = cls[1], label = "NC")
    ax22.plot(trange[i]/st,TrZ_caus_mean[i],'o',color = cls2[0])
    ax22.plot(trange[i]/st,TrZ_nc_mean[i],'o',color = cls2[1])
    ax22.set_xlabel(r'$(t-t_m)/St$')
    ax22.set_ylabel(r"$\langle \delta \rangle$")
    fig.tight_layout()
    
    ax31.plot(trange[:]/st,sign_caus_mean[:],color = cls[0], label = "Caus")
    ax31.plot(trange[:]/st,sign_nc_mean[:],color = cls[1], label = "NC")
    ax31.plot(trange[i]/st,sign_caus_mean[i],'o',color = cls2[0])
    ax31.plot(trange[i]/st,sign_nc_mean[i],'o',color = cls2[1])
    ax31.set_xlabel(r'$(t-t_m)/St$')
    ax31.set_ylabel(r"$\langle |\sigma_n| \rangle$")
    
    ax32.plot(trange[:]/st,sigs_caus_mean[:],color = cls[0], label = "Caus")
    ax32.plot(trange[:]/st,sigs_nc_mean[:],color = cls[1], label = "NC")
    ax32.plot(trange[i]/st,sigs_caus_mean[i],'o',color = cls2[0])
    ax32.plot(trange[i]/st,sigs_nc_mean[i],'o',color = cls2[1])
    ax32.set_xlabel(r'$(t-t_m)/St$')
    ax32.set_ylabel(r"$\langle |\sigma_s| \rangle$")
    
    ax41.plot(trange[:]/st,omg_caus_mean[:],color = cls[0], label = "Caus")
    ax41.plot(trange[:]/st,omg_nc_mean[:],color = cls[1], label = "NC")
    ax41.plot(trange[i]/st,omg_caus_mean[i],'o',color = cls2[0])
    ax41.plot(trange[i]/st,omg_nc_mean[i],'o',color = cls2[1])
    ax41.set_xlabel(r'$(t-t_m)/St$')
    ax41.set_ylabel(r"$\langle |\omega| \rangle$")
    
    ax42.plot(trange[:]/st,curvature_caus[:],color = cls[0], label = "Caus")
    ax42.plot(trange[:]/st,curvature_nc[:],color = cls[1], label = "NC")
    ax42.plot(trange[i]/st,curvature_caus[i],'o',color = cls2[0])
    ax42.plot(trange[i]/st,curvature_nc[i],'o',color = cls2[1])
    ax42.set_xlabel(r'$(t-t_m)/St$')
    ax42.set_ylabel(r"$\langle$ Curvature $\rangle$")
    
    fig.suptitle(fr"$ (t - t_m)/St = {(trange[i]/st):.2f}$")
    fig.tight_layout()
    leg = fig.legend(handles = [h1,h2],ncols = 2, loc = 'lower center', bbox_to_anchor = (0.5,-0.01))
    for lh in leg.legend_handles: 
        lh.set_linewidth(3.0)
        lh.set_alpha(1.0)
    # fig.savefig(savePath/f"ultimate_alpha_{alph:.2f}_t_{i}.png",bbox_inches = 'tight',pad_inches = 0.0,dpi = 300)
    
    raise SystemExit
    for ax in fig.axes:
        for line in ax.get_lines(): line.remove()
        for col in ax.collections: col.remove()
        for patch in ax.patches: patch.remove()
plt.close()
# %%

#%%
print(std_caus,std_nc,mean_caus, mean_nc)
#%%
savePath = pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/4movies_ultimate/st_{st/tp:.2f}")
for alph in alphs:
    cmd = ( 
        f'bash -c \'source ~/.bashrc && '
        f'cd "{savePath}" && '
        f'ffmpeg -framerate 8 -start_number 0 -i ultimate_alpha_{alph:.2f}_t_%d.png '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2" '
        f'-c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart '
        f'ultimate_alpha_{alph:.2f}.mp4 -y && '
        f'mv ultimate_alpha_{alph:.2f}.mp4 ../\''
    )
    os.system(cmd)
#%%
# %%
savePath = pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/4movies_ultimate/st_{st/tp:.2f}")
savePath.mkdir(parents = True, exist_ok= True)

fig = plt.figure(figsize = (16,22))
gs = mpl.gridspec.GridSpec(6, 12, figure=fig)

ax11 = fig.add_subplot(gs[0, 0:3], projection='polar')
ax12 = fig.add_subplot(gs[0, 3:6], projection='polar')
ax13 = fig.add_subplot(gs[0, 6:9], projection='polar')
ax14 = fig.add_subplot(gs[0, 9:12], projection='polar')
axupol = [ax11,ax12,ax13,ax14]


ax2 = fig.add_subplot(gs[1:3,:])
ax31 = fig.add_subplot(gs[3, 0:3], projection='polar')
ax32 = fig.add_subplot(gs[3, 3:6], projection='polar')
ax33 = fig.add_subplot(gs[3, 6:9], projection='polar')
ax34 = fig.add_subplot(gs[3, 9:12],projection='polar')
axvpol = [ax31,ax32,ax33,ax34]

ax41 = fig.add_subplot(gs[4, 0:4])
ax42 = fig.add_subplot(gs[4, 4:8])
ax43 = fig.add_subplot(gs[4, 8:12])
ax51 = fig.add_subplot(gs[5, :6])
ax52 = fig.add_subplot(gs[5, 6:])
# Adjust the layout to remove vertical spaces between the two rows
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax43.set_ylim(-0.4,0)
ax52.set_yscale("log")

for axs in [ax11,ax12,ax13,ax14,ax31,ax32,ax33,ax34]:
    axs.set_theta_zero_location('N')
    axs.set_theta_direction(-1)
    axs.set_rticks([(1/ PI)**0.5],[""])
    axs.set_rlim(0,1.3)
    axs.set_thetagrids([0,90,180,270],[r"$0$",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$"])

labels = ["Caus", "NC"]

for ax in [ax2,ax41,ax42,ax43,ax51,ax52]:
    ax.set_xlim(trange[0]/st*0.75,trange[-1]/st/2)


ax2.plot(trange[:]/st,Q_caus_mean[:],color = cls[0], label = "Caus",lw = 8)
ax2.plot(trange[:]/st,Q_nc_mean[:],color = cls[1], label = "NC",lw = 8)
# ax2.axhline(0.0, ls = '--',color = 'black')
ax2.set_xlabel(r'$(t-t_m)/St$')
ax2.set_ylabel(r"$\langle Q \rangle$")

ax41.plot(trange[:]/st,sig2_caus_mean[:],color = cls[0], label = "Caus")
ax41.plot(trange[:]/st,sig2_nc_mean[:],color = cls[1], label = "NC")

ax41.set_xlabel(r'$(t-t_m)/St$')
ax41.set_ylabel(r"$\langle \sigma^2 \rangle$")

ax42.plot(trange[:]/st,omg2_caus_mean[:],color = cls[0], label = "Caus")
ax42.plot(trange[:]/st,omg2_nc_mean[:],color = cls[1], label = "NC")

ax42.set_xlabel(r'$(t-t_m)/St$')
ax42.set_ylabel(r"$\langle \omega^2 \rangle$")

ax43.plot(trange[:]/st,TrZ_caus_mean[:],color = cls[0], label = "Caus")
ax43.plot(trange[:]/st,TrZ_nc_mean[:],color = cls[1], label = "NC")

ax43.set_xlabel(r'$(t-t_m)/St$')
ax43.set_ylabel(r"$\langle \delta\rangle$")


ax51.plot(trange[:]/st,vrms_caus[:],color = cls[0], label = "Caus")
ax51.plot(trange[:]/st,vrms_nc[:],color = cls[1], label = "NC")
ax51.set_xlabel(r'$(t-t_m)/St$')
ax51.set_ylabel(r"$\langle v_{rms}\rangle$")

ax52.plot(trange[:]/st,curvature_caus[:],color = cls[0], label = "Caus")
ax52.plot(trange[:]/st,curvature_nc[:],color = cls[1], label = "NC")
ax52.set_xlabel(r'$(t-t_m)/St$')
ax52.set_ylabel(r"$\langle$ Curvature $\rangle$")

for count,i  in enumerate([87,120,151,180]):

    print(i,end = "\r")
    vtht_caus_notnan = vtheta_caus[i,~np.isnan(vtheta_caus[i])]%TWO_PI
    utht_caus_notnan = utheta_caus[i,~np.isnan(utheta_caus[i])]%TWO_PI
    vtht_nc_notnan = vtheta_nc[i,~np.isnan(vtheta_nc[i])]%TWO_PI
    utht_nc_notnan = utheta_nc[i,~np.isnan(utheta_nc[i])]%TWO_PI
    
    vtht_caus_pdf = np.histogram(vtht_caus_notnan,bins = thtbins, density = True)[0]
    utht_caus_pdf = np.histogram(utht_caus_notnan,bins = thtbins, density = True)[0]
    vtht_nc_pdf = np.histogram(vtht_nc_notnan,bins = thtbins, density = True)[0]
    utht_nc_pdf = np.histogram(utht_nc_notnan,bins = thtbins, density = True)[0]
    
    ax2.plot(trange[i]/st,Q_caus_mean[i],'o',color = cls2[0],markersize = 25)
    ax2.plot(trange[i]/st,Q_nc_mean[i],'o',color = cls2[1],markersize = 25)
    
    
    h1, = axupol[count].plot(periodic(thtvals),periodic((2*utht_caus_pdf)**0.5), label = "Caus",color = cls[0])
    h2,= axupol[count].plot(periodic(thtvals),periodic((2*utht_nc_pdf)**0.5), label = "NC",color = cls[1])
    axupol[count].fill_between(periodic(thtvals),periodic((2*utht_caus_pdf)**0.5), alpha=0.3, color=cls2[0],lw = 0.0)
    axupol[count].fill_between(periodic(thtvals),periodic((2*utht_nc_pdf)**0.5), alpha=0.3, color=cls2[1],lw = 0.0)
    # axupol[count].set_title(r'$\theta_u$',pad = 30)
    
    axvpol[count].plot(periodic(thtvals),periodic((2*vtht_caus_pdf)**0.5), label = "Caus",color = cls[0])
    axvpol[count].plot(periodic(thtvals),periodic((2*vtht_nc_pdf)**0.5), label = "NC",color = cls[1])
    axvpol[count].fill_between(periodic(thtvals),periodic((2*vtht_caus_pdf)**0.5), alpha=0.3, color=cls2[0],lw = 0.0)
    axvpol[count].fill_between(periodic(thtvals),periodic((2*vtht_nc_pdf)**0.5), alpha=0.3, color=cls2[1],lw = 0.0)
    # axvpol[count].set_title(r'$\theta_v$',pad = 30)
    
    

leg = fig.legend(handles = [h1,h2],ncols = 2, loc = 'lower center', bbox_to_anchor = (0.5,1.01))
fig.tight_layout()
fig.savefig(savePath/f"ultimate_alpha_{alph:.2f}_summary.png",bbox_inches = 'tight',pad_inches = 0.,dpi = 300,transparent = True)

# gs.update(hspace=0.2)  # Set the height space between the first and second row to 0
# %%

# %%

# %%

# %%

# %%

# %%
-0.9*PI%(TWO_PI)
# %%
    
