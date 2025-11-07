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

idx = int(float(sys.argv[-1]))
# idx = 15
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
    vel_nc_shifted = f['vel_nc_shifted'][:]
    Q_nc_shifted = f['Q_nc_shifted'][:]
    Q_caus_shifted = f['Q_caus_shifted'][:]
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
cond = idx_caus_min < -20
pos_caus_shifted = pos_caus_shifted[:,cond,:]
vel_caus_shifted = vel_caus_shifted[:,cond,:]
Q_caus_shifted = Q_caus_shifted[:,cond]
idx_caus_min = idx_caus_min[cond]
idx_nc_min = (np.ones(len(Q_nc_shifted[-1]))*400 - len(Q_nc_shifted)).astype(np.int32)
cond = np.isnan(Q_nc_shifted).sum(axis = (0))!=800
pos_nc_shifted = pos_nc_shifted[:,cond,:]
vel_nc_shifted = vel_nc_shifted[:,cond,:]
Q_nc_shifted = Q_nc_shifted[:,cond]
idx_nc_min = idx_nc_min[cond]
numcaus = len(idx_caus_min)
numnc = len(idx_nc_min)

#%%
Q_nc_shifted.shape,t_mins.shape,(np.isnan(Q_caus_shifted).sum(axis = (0))==400).sum()

#%%
# prtcl = 2030 #3795 , #2124, 3197 #! These trajectories without conditioning are special.
# print(idx_caus_min[prtcl])
# time,Qmin, count_number = compute_prev_minima(np.arange(len(Q_caus_shifted[:,prtcl])),Q_caus_shifted[:,prtcl],0)
# peaks, _ = find_peaks(-Q_caus_shifted[:,prtcl])
# time, Qmin, count_number,peaks
# #%%

# plt.plot(Q_caus_shifted[:,prtcl])
# plt.plot(time, Qmin,'o')
# plt.plot(time,Q_caus_shifted[idx_caus_min[prtcl],prtcl],'s')
# plt.plot(peaks,Q_caus_shifted[peaks,prtcl],'x')
# plt.xlim(350,400)
# # plt.plot(Q_nc_shifted[idx_nc_min[prtcl]-2:idx_nc_min[prtcl]+2,prtcl])




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
(np.isnan(pos_caus_shifted).sum(axis = (0))==800).sum()

#%%
ftimes = 501
fctimes = ftimes - ftimes//2
trange = np.arange(ftimes)
trange = (trange - ftimes//2)*0.1
curvature_caus = np.zeros(ftimes)
curvature_nc = np.zeros(ftimes)
count_c = np.zeros(ftimes)
count_nc = np.zeros(ftimes)

"""
Take only the notnan part. Subtract from the minimum index. 
"""
for i in range(numcaus):
    
    minidx = idx_caus_min[i] + len(Q_caus_shifted)
    notnan = ~np.isnan(pos_caus_shifted[:,i,0])
    firstnotnan = np.argwhere(np.cumsum(1.0*notnan) == 1).ravel()[0]
    trajnotnan = pos_caus_shifted[notnan,i,:]
    minidx = minidx -firstnotnan
    endidx = minidx +  min(ftimes - fctimes, len(trajnotnan[minidx:minidx + ftimes - fctimes]))
    startidx = max(minidx -fctimes,0)
    time = np.arange(startidx ,endidx)
    curvature_caus[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature(time, trajnotnan[startidx: endidx,0], trajnotnan[startidx:endidx,1])
    count_c[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += 1
curvature_caus /= count_c + 1e-16
for i in range(numnc):
    minidx = idx_nc_min[i]+ len(Q_nc_shifted)
    notnan = ~np.isnan(pos_nc_shifted[:,i,0])
    firstnotnan = np.argwhere(np.cumsum(1.0*notnan) == 1).ravel()[0]
    trajnotnan = pos_nc_shifted[notnan,i,:]
    minidx = minidx -firstnotnan
    endidx = minidx +  min(ftimes - fctimes, len(trajnotnan[minidx:minidx + ftimes - fctimes]))
    startidx = max(minidx -fctimes,0)
    time = np.arange(startidx ,endidx)
    
    curvature_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += curvature(time, trajnotnan[startidx:endidx,0], trajnotnan[startidx:endidx,1])
    count_nc[fctimes -(minidx - startidx): fctimes +(endidx - minidx)] += 1
curvature_nc /= count_nc + 1e-16
#%%
numcaus, numnc
#%%
plt.plot(trange,count_c,"b")
# plt.plot(trange,count_nc,"r")
# plt.yscale("log")
#%%
fig = plt.figure()
plt.plot(trange,curvature_caus,color = "red", label = "Caus")
plt.plot(trange,curvature_nc,color = "blue", label = "NC")
plt.xlabel(r"$(t - t_{m})/St$")
plt.ylabel("Curvature")
plt.yscale("log")
plt.legend()
fig.savefig(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/curvature_alph_{alph:.2f}.png",bbox_inches = "tight",format = "png",dpi = 300)
fig.savefig(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/curvature_alph_{alph:.2f}.pdf",bbox_inches = "tight",format = "pdf",dpi = 300)
plt.show()
#%% 
# colors = ["9a031e", "fb8b24","ffffff","00b4d8","0077b6"]
# colors2 = ["81b29a","06d6a0","ffffff","ef476f","e07a5f"]
# colors = ["#" + c for c in colors]
# colors2 = ["#" + c for c in colors2]
# positions = np.linspace(0,1,len(colors))
# cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
# cmap_fut = mpl.colors.LinearSegmentedColormap.from_list('cmap_fut', colors[2:])
# cmap_past = mpl.colors.LinearSegmentedColormap.from_list('cmap_past', colors[:3])
# cmap2 = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors2)))
# cmap_fut2 = mpl.colors.LinearSegmentedColormap.from_list('cmap_fut', colors2[2:])
# cmap_past2 = mpl.colors.LinearSegmentedColormap.from_list('cmap_past', colors2[:3])

# #%%
# # ntraj = 500
# # ftimes = 101
# # for t in range(ftimes): 

# #     fig,ax = plt.subplots(1,2,figsize = (12,6))
# #     plt.style.use('dark_background')
# #     for axs in ax:
# #         axs.axis('off')
# #         axs.set_xlim(0,2.0)
# #         axs.set_ylim(-0.5,1.5)
# #     for i in range(ntraj):
# #         minidx = idx_caus_min[i]
# #         tact = t
# #         if minidx + t> -1: 
# #             tact = -1 - minidx

# #         ax[0].scatter(pos_caus_shifted[minidx:minidx + tact,i,0],pos_caus_shifted[minidx:minidx + tact,i,1],c = np.arange(tact),cmap = cmap_fut,s = 0.1,alpha = 1.0)
        
        
# #         minidx = idx_nc_min[i]
# #         tact = t


# #         ax[1].scatter(pos_nc_shifted[minidx:minidx + tact,i,0],pos_nc_shifted[minidx:minidx + tact,i,1],c = np.arange(tact),cmap = cmap_fut,s = 0.1,alpha = 1.0)
# #     ax[1].set_title("NC", color = "white")
# #     ax[0].set_title("Caus",color = "white")
# #     # plt.show()
# #     plt.savefig(f"/Users/rajarshi/Library/CloudStorage/Nextcloud-rajarshi.chattopadhyay@cloud.icts.res.in/Fluids/Caustics/traj_viz/Plots/bspline/Re_125000.0,dt_0.005,N_1024/4movies/traj_fut_t_{t}.png",dpi = 300)
# #     plt.close()





# # #%%
# # #? Past trajectories    
# #  # %%
# # ntraj = 500
# # ftimes = 101
# # for t in range(ftimes): 

# #     fig,ax = plt.subplots(1,2,figsize = (12,6))
# #     plt.style.use('dark_background')
# #     for axs in ax:
# #         axs.axis('off')
# #         axs.set_xlim(-4.0,0)
# #         axs.set_ylim(-1.5,0.5)
# #     for i in range(ntraj):
# #         minidx = idx_caus_min[i]
# #         tact = t
# #         if minidx - (ftimes) + tact > -len(Q_caus_shifted): 
# #             ax[0].scatter(pos_caus_shifted[minidx - ftimes : minidx - (ftimes - tact) ,i,0],pos_caus_shifted[minidx - ftimes : minidx - (ftimes - tact),i,1],c = np.arange(tact),cmap = cmap_past,s = 0.1,alpha = 1.0)
        
        
# #         minidx = idx_nc_min[i]
# #         tact = t
# #         # print(minidx - ftimes)
# #         ax[1].scatter(pos_nc_shifted[minidx - ftimes : minidx - (ftimes - tact),i,0],pos_nc_shifted[minidx - ftimes : minidx - (ftimes - tact),i,1],c = np.arange(tact),cmap = cmap_past,s = 0.1,alpha = 1.0)
# #     ax[1].set_title("NC", color = "white")
# #     ax[0].set_title("Caus",color = "white")
# #     # plt.show()
# #     plt.savefig(f"/Users/rajarshi/Library/CloudStorage/Nextcloud-rajarshi.chattopadhyay@cloud.icts.res.in/Fluids/Caustics/traj_viz/Plots/bspline/Re_125000.0,dt_0.005,N_1024/4movies/traj_past_t_{t}.png",dpi = 300)
# #     plt.close()

# # %%

# #%%
# #* Both at once
# ntraj = np.random.randint(0,numcaus,500)

# movies = pathlib.Path(f"/Users/rajarshi/Library/CloudStorage/Nextcloud-rajarshi.chattopadhyay@cloud.icts.res.in/Fluids/Caustics/traj_viz/Plots/bspline/Re_125000.0,dt_0.005,N_1024/4movies")
# movies.mkdir(parents = True, exist_ok = True)
# for t in range( ftimes): 

#     # if t < ftimes//2: continue
#     fig,ax = plt.subplots(1,2,figsize = (10,6))
#     plt.style.use('dark_background')
#     for axs in ax:
#         axs.axis('off')
#         axs.set_xlim(-1,1)
#         axs.set_ylim(-2,1)
#     for i in ntraj:
#         #* pre Q_min behaviour
#         minidx = idx_caus_min[i]
#         tact = min(t,ftimes//2)
#         fctimes = ftimes - ftimes//2
#         if minidx - (fctimes) + tact > -len(Q_caus_shifted): 
#             if pos_caus_shifted[max(len(Q_caus_shifted) + minidx - fctimes, 0) ,i,0] < 0 : 
#                 col_caus_past = cmap_past2
#                 col_caus_fut = cmap_fut2
#             else: 
#                 col_caus_past = cmap_past
#                 col_caus_fut = cmap_fut
#             ax[0].scatter(pos_caus_shifted[max(len(Q_caus_shifted) + minidx - fctimes, 0) : minidx - (fctimes - tact) ,i,0],pos_caus_shifted[max(len(Q_caus_shifted) + minidx - fctimes, 0) : minidx - (fctimes - tact),i,1],c = np.arange(tact + min(len(Q_caus_shifted) + minidx - fctimes, 0)),cmap = col_caus_past,s = 0.1,alpha = 1.0)
        
        
#         minidx = idx_nc_min[i%numnc]
#         tact = min(t,ftimes//2)
#         # print(minidx - ftimes)
#         if pos_nc_shifted[minidx - fctimes,i%numnc,0] < 0: 
#             col_nc_past = cmap_past2
#             col_nc_fut = cmap_fut2
#         else: 
#             col_nc_past = cmap_past
#             col_nc_fut = cmap_fut
#         ax[1].scatter(pos_nc_shifted[minidx - fctimes : minidx - (fctimes - tact),i%numnc,0],pos_nc_shifted[minidx - fctimes : minidx - (fctimes - tact),i%numnc,1],c = np.arange(tact),cmap = col_nc_past,s = 0.1,alpha = 1.0)
        
#         if t > ftimes//2: #*  post Q_min behaviour.
#             # raise SystemExit("t_min reached")
#             minidx = idx_caus_min[i]
#             tact = t - ftimes//2
#             if minidx + tact> -1: 
#                 tact = -1 - minidx
            
#             ax[0].scatter(pos_caus_shifted[minidx:minidx + tact,i,0],pos_caus_shifted[minidx:minidx + tact ,i,1],c = np.arange(tact),cmap = col_caus_fut,s = 0.1,alpha = 1.0)
            
            
#             minidx = idx_nc_min[i%numnc]
#             tact = t - ftimes//2


#             ax[1].scatter(pos_nc_shifted[minidx:minidx + tact,i%numnc,0],pos_nc_shifted[minidx:minidx + tact,i%numnc,1],c = np.arange(tact),cmap = col_nc_fut,s = 0.1,alpha = 1.0)
            
#     ax[1].set_title("NC", color = "white")
#     ax[0].set_title("Caus",color = "white")
#     fig.suptitle(fr"$t-t_{{min}}= {((t - ftimes//2)/tp*0.1):.1f} \tau_p$",color = "white")
#     # Add colorbars for both subplots

#     cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap2), ax=ax[0], orientation='horizontal',shrink = 0.5)
#     cbar1.set_label(r'$x_0 < 0$', color='white')
#     cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax[1], orientation='horizontal',shrink = 0.5)
#     cbar2.set_label(r'$x_0 > 0$', color='white')
#     plt.savefig(movies/f"traj_lr_frame_{t}.png",dpi = 300,bbox_inches = "tight")
#     fig.tight_layout()
#     plt.close()
#     print(t,end = '\r')
#     # %%
