"""
This script rescales the position and velocity of every particle at the time they passed through the minimum of Q and then studies the past and the future for these particles. 
"""
#%%

#%%
import numpy as np
# from numba import njit
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
# import matplotlib as mpl
# mpl.rc("text", usetex = True)
from tqdm import tqdm
import pathlib,sys,os,json,time
import h5py,math
import numpy as cp
from scipy.interpolate import splrep, splev, splder,sproot,PPoly
from scipy.fft import rfft2, irfft2,fft
idx = int(float(sys.argv[-1]))
# idx = 8
alph_idx = idx%8
st_idx = idx//8
#%%
#
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

def ifft2(x):
    return irfft2(x,(Nx,Ny),axes = (0,1))
    
def fft2(x):
    return rfft2(x,axes = (0,1))


#
#%%
print(f"alpha {alph}")

## ---------- Grid and wavenumbers --------------- ##
Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
x,y = np.meshgrid(X,Y,indexing="ij")
PI = np.pi
TWO_PI = 2*PI


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
#%%

#


#
def find_range(x,idx): return np.sum(np.cumsum(x[:idx][::-1]) == range(1,len(x[:idx])+1))+np.sum(np.cumsum(x[idx:])== range(1,len(x[idx:])+1))


#

 
idx = cp.zeros((Nprtcl,d),dtype = int)
slx = idx[:,0].copy()
sly = idx[:,1].copy()
delx = cp.zeros((Nprtcl,d),dtype = cp.float64)
temparr = cp.zeros((Nprtcl),dtype=cp.float64)
poly = cp.zeros((Nprtcl,d,order),dtype = cp.float64)
Lx = Ly = TWO_PI
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]


#%%

def initialize_b_spline(order,X,dx,Nx):
    """
    Initializes the Matrix M and coefficient array ck in 2D according to hinsberg et al.
    ck is not computed using the formula. Rather doing 1/F(b_d) where F is the fourier transform of the b_d function given in (7.6).
    Mmat is compute using (7.4)
    """
    Mmat = cp.zeros((order,order))
    for i in range(order):
        for j in range(order):
            for s in range(j,order):
                Mmat[j,i] += (-1)**(s-j)*math.factorial(order)*(order - s-1)**(order-i-1)/(math.factorial(s-j)*math.factorial(order +j -s))
            
        Mmat[:,i] = Mmat[:,i]/(math.factorial(order-i-1)*math.factorial(i))
    Mmat = cp.array(Mmat)

    def bj(x,j): 
        bj = 0
        for i in range(order):
            bj += Mmat[j,i]*(x -order/2 + j+1)**(i)
        return bj*((x -order/2 + j+1) < 1)*(x -order/2 + j+1 >= 0)

    bm = 0. 
    xeval = cp.array(X/dx*((X//dx<Nx/2)) + (X/dx - Nx)*(X//dx>=Nx/2))
    for j in range(order):
        bm += bj(xeval,j)


    bk = fft(bm)
    ck = 1./bk
    ckxky = ck[:,None]*ck[None,:]
    cxy = ifft2(ckxky).real
    ck2d = fft2(cxy) #! This will be multiplied with all the fields to be interpolated. 
    
    del bm,bk,ck,ckxky,cxy
    nums = np.arange(order)
    
    return ck2d,Mmat,nums
ck2d, Mmat, nums = initialize_b_spline(order,X,dx,Nx)   


#%%
num_axes = 3
(None,) * num_axes + (slice(None),)
#%%

def interp_spline(pos,*args,shp = (Nprtcl)):
    """
    b-spline interpolation of the specified order given in Hinsberg et al.
    """
    global nums,idx,delx,poly,slx,sly,temparr,order,Mmat,dx,N,Nprtcl
    shapes = [(Nprtcl,*arg.shape[2:]) for arg in args]
    mats = [np.zeros(shp) for shp in shapes]
    idx[:] = (pos//dx).astype(np.int32)
    delx[:] = (pos%dx)/dx    
    poly[:] = delx[...,None]**nums

    #! This is the main loop for the interpolation. The loop is replaced by kernel code which makes it faster. 
    for i in range(order):
        for j in range(order):
            temparr[:] = np.einsum('p,q,...p,...q->...',Mmat[i],Mmat[j],poly[...,0,:],poly[...,1,:]) #! For saving computations
            slx[:] = (idx[:,0]-order//2 + 1 + i)%N #! For saving computations
            sly[:] = (idx[:,1]-order//2 + 1 + j)%N #! For saving computations
            
            for mat,arg,shp in zip(mats,args,shapes):
                tempshape =   (slice(None),) + (None,) *len(shp[1:])
                mat  += arg[slx,sly,...]*temparr[tempshape]    
                
    if len(args) <2: return mats[0]           
    else: return mats



#


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

times = np.arange(0,T+ 0.1,params["savestep"])

#! Goal -- load files one by one, extract caustics and non-caustics particles.
"""Check and store statistics. And be done with it
Stats to store: 
The eigenvaules series for caustics particles.
The time of caustics

"""


#%%

print(times[-1],st,str(savePlot))
Ntimes = len(times)
TrZ = np.zeros((Ntimes, Nprtcl))
TrZ2 = np.zeros((Ntimes,Nprtcl))
Z = np.zeros((Nprtcl,d,d))
caus_count = np.zeros((Ntimes, Nprtcl),dtype = np.int32)
diff = np.zeros((Ntimes, Nprtcl),dtype = np.float64) #! v - u values for each particle
vel = np.zeros((Ntimes, Nprtcl,d))
pos = np.zeros((Ntimes, Nprtcl,d))
Qmean = np.zeros((Ntimes))
Qstd = np.zeros((Ntimes))
uinterp = np.zeros((Ntimes,Nprtcl, d))
Qinterp = np.zeros((Ntimes, Nprtcl))
siginterp = np.zeros((Ntimes, Nprtcl))
xiinterp = np.zeros((Ntimes, Nprtcl))
ainterp = np.zeros((Ntimes, Nprtcl))
binterp = np.zeros((Ntimes, Nprtcl))
cinterp = np.zeros((Ntimes, Nprtcl))
Ainterp = np.zeros((Nprtcl,d,d))
causQmean = np.zeros(Ntimes)
causQstd = np.zeros(Ntimes)
Qmax = 1.0
Qmin = -1.0
Qbins = np.linspace(Qmin,Qmax, 6001)
Q_field_pdf = np.zeros((Ntimes,len(Qbins)-1))
Q_particle_pdf = np.zeros((Ntimes,len(Qbins)-1))
Q_caus_pdf = np.zeros((Ntimes,len(Qbins)-1))
xi = np.zeros((Nx,Ny//2+1),dtype = np.complex128)
A = np.zeros((Nx,Ny//2+1,d,d),dtype = np.complex128)
u = xi.copy()
v = xi.copy()
xi_r = np.zeros((Nx,Ny),dtype = np.float64)
u_field = np.zeros((Nx,Ny,d),dtype = np.float64)

Q_field = xi_r.copy()
sig_field = xi_r.copy()
Ar = np.zeros((Nx,Ny,d,d),dtype = np.float64)
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



glob_min_Q = 0.0




#%%

final_caus_count = np.load(prtcl_loadPath/f"time_{times[-1]:.2f}/caus_count.npz")["caustics_count"]
t1 = t2 = 0.0
for i,t in tqdm(enumerate(times)):
    t1 = time.time()
    # print(f"loading time {t} in time {t2 - t1 sec}",end='\r')
    Z[:] =  np.load(prtcl_loadPath/f"time_{t:.2f}/prtcl_Z.npz")["Zmatrix"]
    # Z[:] =  np.load(prtcl_loadPath/f"time_{t:.2f}/prtcl_Z.npy")
    TrZ[i] = np.einsum('...ii->...', Z)
    TrZ2[i] = np.einsum('...ij,...ji->...', Z,Z)
    caus_count[i] =  np.round(np.load(prtcl_loadPath/f"time_{t:.2f}/caus_count.npz")["caustics_count"]).astype(np.int32)



    if i> 0: newcaus_idx = caus_count[i] > caus_count[i-1] #! New caustics particles
    else : newcaus_idx = caus_count[i] > 0
    causptrcl = caus_count[i] > 0
    
    if newcaus_idx.size>0:
        tfp[newcaus_idx] = t
        tis = np.append(tis,tip[newcaus_idx])
        tfs = np.append(tfs,tfp[newcaus_idx])
        tip[newcaus_idx] = t


    vel[i] = np.load(prtcl_loadPath/f"time_{t:.2f}/vel.npz")["vel"]
    pos[i] = np.load(prtcl_loadPath/f"time_{t:.2f}/pos.npz")["pos"]
    
    xi[:] = np.load(loadPath/f"time_{t:.2f}/w.npz")["vorticity"]
    
    xi_r[:] = ifft2(xi)
    # print(xi_r.max(),np.sqrt(np.mean(xi_r**2)))
    u[:] = 1j* ky*lapinv*xi
    v[:] = -1j*kx*lapinv*xi
    
    A[...,0,0] = 1j*kx*u
    A[...,0,1] = 1j*ky*u
    A[...,1,0] = 1j*kx*v
    A[...,1,1] = 1j*ky*v
    Ar[:] = ifft2(A)
    
    u_field[...,0] = ifft2(u)
    u_field[...,1] = ifft2(v)
    
    Q_field[:] = -0.5*np.einsum('...ij,...ji->...',Ar,Ar)
    glob_min_Q = min(glob_min_Q,np.min(Q_field))
    sig_field[:] = (xi_r**2 - 4*Q_field)**0.5
    Q_field_pdf[i,:]  = np.histogram(tp**2*Q_field.ravel(),bins = Qbins)[0]/(Nx*Ny)


    if i == 0:
        xi_rms = tp*np.sqrt(np.mean(xi_r**2))
        sig_rms = tp*np.sqrt(np.mean(sig_field**2))
        Q_rms = tp**2*np.sqrt(np.mean(Q_field**2))
        print(f"rms omg, sig, Q: {xi_rms, sig_rms, Q_rms}")

    uinterp[i,:] = interp_spline(pos[i],u_field,shp = (Nprtcl,1))
    Ainterp[:] = tp*interp_spline(pos[i],Ar,shp = (Nprtcl,1,1))
    ainterp[i,:] = Ainterp[:,0,0]
    binterp[i,:] = Ainterp[:,0,1]
    cinterp[i,:] = Ainterp[:,1,0]
    Qinterp[i,:] = -ainterp[i,:]**2 - binterp[i,:]*cinterp[i,:]
    np.savez_compressed(prtcl_loadPath/f"time_{t:.2f}/Q.npz",Q = Qinterp[i,:])
    xiinterp[i,:] = cinterp[i,:] - binterp[i,:]
    siginterp[i,:] = (4*ainterp[i,:]**2 + (binterp[i,:] + cinterp[i,:])**2)**0.5
    if i == 0:
        factor = Qinterp[i]/(-0.5*TrZ2[i])
        print(f"Initial factor : {factor.mean(),factor.std()}")
        del factor
    
    Q_particle_pdf[i,:] = np.histogram(Qinterp[i,:],bins = Qbins)[0]/Nprtcl


    if causptrcl.sum() > 0:
        causQmean[i] = np.mean(Qinterp[i,causptrcl])
        causQstd[i] = np.std(Qinterp[i,causptrcl])
        Q_caus_pdf[i,:] = np.histogram(Qinterp[i,causptrcl],bins = Qbins)[0]/np.sum(causptrcl)
        
    t2 = time.time()
    
print(f"Data loaded for alpha {alph}, with global min Q {glob_min_Q}")
cond_new_caus = (caus_count == 1 )*((caus_count - np.roll(caus_count,1, axis = 0)) == 1) #! Condition clicks when particles formed caustics for the first time.
#%%

#%%

def save_dset(f,dname,data):
    if dname in f:
        del f[dname]
    
    f.create_dataset(dname, data = data, dtype = np.float64, compression = 'gzip')
    return f[dname]


with h5py.File(prtcl_loadPath/f"caus-details.hdf5",'w') as f:
    
    save_dset(f,"Caustics_ratio",(1.0*np.sum(caus_count>0,axis = 1))/Nprtcl)
    save_dset(f,'times',times)

    
    
    save_dset(f,"first_caus",data = (1.0*cond_new_caus.sum(axis = 1))/Nprtcl)
    
    
    meanQ = np.mean(Qinterp,axis = 1)
    stdQ = np.std(Qinterp,axis = 1)
    
    save_dset(f,"Qmean",data = meanQ)
    save_dset(f,"Qstd",data = stdQ)
    save_dset(f,"Q_field_pdf",data = Q_field_pdf)
    save_dset(f,"Q_particle_pdf",data = Q_particle_pdf)
    save_dset(f,"Q_caus_pdf",data = Q_caus_pdf)
    
del causQmean,causQstd,Qmean,Qstd,Q_field_pdf,Q_particle_pdf,Q_caus_pdf
    
#%%    
causidx = caus_count[-1] > 0
noncausidx = caus_count[-1] == 0
caus_count_c = caus_count[:,causidx]
caus_nums = caus_count_c[-1]
tot_caus = np.sum(caus_nums)
del caus_count

pos_c = pos[:, causidx]
pos_nc = pos[:,noncausidx]
del pos

vel_c = vel[:,causidx]
vel_nc = vel[:,noncausidx]
del vel

uinterp_c = uinterp[:,causidx]
uinterp_nc = uinterp[:,noncausidx]
del uinterp

Qinterp_c = Qinterp[:,causidx]
Qinterp_nc = Qinterp[:,noncausidx]
del Qinterp

xiinterp_c = xiinterp[:,causidx]
xiinterp_nc = xiinterp[:,noncausidx]
del xiinterp

siginterp_c = siginterp[:,causidx]
signterp_nc = siginterp[:,noncausidx]
del siginterp

TrZ_c = TrZ[:,causidx]
TrZ_nc = TrZ[:,noncausidx]
del TrZ

TrZ2_c = TrZ2[:,causidx]
TrZ2_nc = TrZ2[:,noncausidx]
del TrZ2

ainterp_c = ainterp[:,causidx]
ainterp_nc = ainterp[:,noncausidx]
del ainterp


binterp_c = binterp[:,causidx]
binterp_nc = binterp[:,noncausidx]
del binterp


cinterp_c = cinterp[:,causidx]
cinterp_nc = cinterp[:,noncausidx]
del cinterp

                    

#%%

maxdt = np.max(tfs-tis)
# maxsize = int(maxdt//dt + 1)
# maxsize = int( max(min(30*tp,maxdt//dt +1), 50) )
maxsize = int(max(min(30*tp, maxdt),40)//params["savestep"])  + 1

pos_caus_shifted = np.ones((maxsize, int(tot_caus),d))*np.nan
vel_caus_shifted = np.ones((maxsize, int(tot_caus),d))*np.nan
u_caus_shifted = np.ones((maxsize, int(tot_caus),d))*np.nan
Q_caus_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
omg_caus_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
sig_caus_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
TrZ_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
TrZ2_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
a_caus_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
b_caus_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
c_caus_shifted = np.ones((maxsize,int(tot_caus)))*np.nan
t_caus_len = np.zeros(int(tot_caus))
Q_initial = np.zeros(int(tot_caus))
omg_initial = np.zeros(int(tot_caus))
sig_initial = np.zeros(int(tot_caus))
t_mins = np.zeros(int(tot_caus))
Q_mins = np.zeros(int(tot_caus))


#%% 
def compute_prev_minima(x,y,count_number):
    """Outputs the value of x for which the derivative of y is zero 
    With a syntax copied from Microsoft Copilot.
    """
    tck = splrep(x,y,k = 5)
    tck_der = splder(tck)
    tck_2der = splder(tck,n=2)

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

#%% 
# # xtest = np.linspace(0,2*np.pi,50)
# # ytest = np.sin(xtest*4)

# # xroot,yroot,count_number  = compute_prev_minima(xtest,ytest,0)
# # print(xroot)

# # plt.plot(xtest,ytest,'.-')
# # # plt.plot(xtest,4*np.cos(xtest*4),color = 'red')
# # plt.plot(xroot,yroot,'o',color = 'black')
# # plt.grid()
#%%
ins = 0
meant_min = 0.0
meanQ_min = 0.0
count_number = 0
for i in range(len(caus_nums)):
    # print(i)
    # print(caus_count[-1,i])
    """
    For each particle, cau_count will give how many instances to split the particle into. 
    Then for each of the instances, add the Q in the Q_caus_pdf. 
    """
    # print(caus_count[-1,i])
    caus_count_c[:,i]
    
    for j in range(int(caus_count_c[-1,i])):
        region = ((caus_count_c[:,i] > j-1)*(caus_count_c[:,i] <= j)).ravel() #region in time where this caustics happened
        regionlen = np.sum(region)
        # try:
        t_caus_len[ins] = regionlen
        if regionlen > maxsize:
            pos_caus_shifted[:,ins,:] = pos_c[region,i][-maxsize:,:]
            vel_caus_shifted[:,ins,:] = vel_c[region,i][-maxsize:,:]
            u_caus_shifted[:,ins,:] = uinterp_c[region,i][-maxsize:,:]
            Q_caus_shifted[:,ins] = (Qinterp_c[region,i])[-maxsize:] # Adding the trajectory of the particle for last few times of its trajectory.
            omg_caus_shifted[:,ins] = (xiinterp_c[region,i])[-maxsize:]
            sig_caus_shifted[:,ins] = (siginterp_c[region,i])[-maxsize:]
            a_caus_shifted[:,ins] = (ainterp_c[region,i])[-maxsize:]
            b_caus_shifted[:,ins] = (binterp_c[region,i])[-maxsize:]
            c_caus_shifted[:,ins] = (cinterp_c[region,i])[-maxsize:]
            TrZ_shifted[:,ins] = (TrZ_c[region,i])[-maxsize:]
            TrZ2_shifted[:,ins] = (TrZ2_c[region,i])[-maxsize:]
        else: 
            pos_caus_shifted[(maxsize - regionlen):, ins,:] = pos_c[region,i,:]
            vel_caus_shifted[(maxsize - regionlen):, ins,:] = vel_c[region,i,:]
            u_caus_shifted[(maxsize - regionlen):, ins,:] = uinterp_c[region,i,:]
            Q_caus_shifted[(maxsize-regionlen):,ins] = Qinterp_c[region,i] # Adding the trajectory of the particle and rest nan.
            omg_caus_shifted[(maxsize-regionlen):,ins] = xiinterp_c[region,i]
            sig_caus_shifted[(maxsize-regionlen):,ins] = siginterp_c[region,i]
            a_caus_shifted[(maxsize-regionlen):,ins] = ainterp_c[region,i]
            b_caus_shifted[(maxsize-regionlen):,ins] = binterp_c[region,i]
            c_caus_shifted[(maxsize-regionlen):,ins] = cinterp_c[region,i]
            TrZ_shifted[(maxsize-regionlen):,ins] = TrZ_c[region,i]
            TrZ2_shifted[(maxsize-regionlen):,ins] = TrZ2_c[region,i]
            
        t_caus_local = times[:regionlen]
        t_caus_local = t_caus_local - t_caus_local[-1]
        t_mins[ins],Q_mins[ins],count_number = compute_prev_minima(t_caus_local,Qinterp_c[region,i],count_number)
        Q_initial[ins] = Qinterp_c[region,i][0]
        omg_initial[ins] = xiinterp_c[region,i][0]
        sig_initial[ins] = siginterp_c[region,i][0]
        # except:
        #     print("Exiting:",(maxsize-regionlen), regionlen,ins, Q_caus_shifted.shape)
        #     raise SystemExit

        ins = ins + 1
print(f"Ins caus_count_c {ins},{tot_caus}")

    
#%%

meant_min = np.mean(t_mins)
meanQ_min = np.mean(Q_mins)

stdQ_min = np.std(Q_mins)
minQ = meanQ_min 


minQarrays = np.linspace(minQ,glob_min_Q,1000)
caus_min_Qs = np.zeros_like(minQarrays)
noncaus_min_Qs = np.zeros_like(minQarrays)
for i,min_Q in enumerate(minQarrays):
    caus_min_Qs[i] = np.sum((Qinterp_c<min_Q).any(axis = 0))
    noncaus_min_Qs[i] = np.sum((Qinterp_nc<min_Q).any(axis =0))

del Qinterp_c,caus_count_c,TrZ_c,TrZ2_c,xiinterp_c,ainterp_c,binterp_c,cinterp_c,pos_c, vel_c,uinterp_c
#%%
tot_min_Qs = caus_min_Qs + noncaus_min_Qs
caus_min_Qs = caus_min_Qs/tot_min_Qs
noncaus_min_Qs = noncaus_min_Qs/tot_min_Qs

satisfying_conditionidx = (Qinterp_nc < minQ).any(axis = 0)

#%%

print(f"Number of non caustics particles ventured below the Q_min and survived are {satisfying_conditionidx.sum()}")
print(f"Number of non caustics particles that never ventured below the Q_min are {(noncausidx).sum()}")
print(f"Number of caustics particles {(causidx).sum()}")
    
#%%
random_indices = np.random.choice(satisfying_conditionidx.sum(),50000)
pos_nc = pos_nc[0:,satisfying_conditionidx,:][:,random_indices,:]
vel_nc = vel_nc[0:,satisfying_conditionidx,:][:,random_indices,:]
uinterp_nc = uinterp_nc[0:,satisfying_conditionidx,:][:,random_indices,:]
Qinterp_nc = Qinterp_nc[0:,satisfying_conditionidx][:,random_indices]
xiinterp_nc = xiinterp_nc[0:,satisfying_conditionidx][:,random_indices]
ainterp_nc = ainterp_nc[0:,satisfying_conditionidx][:,random_indices]
binterp_nc = binterp_nc[0:,satisfying_conditionidx][:,random_indices]
cinterp_nc = cinterp_nc[0:,satisfying_conditionidx][:,random_indices]
TrZ_nc = TrZ_nc[0:,satisfying_conditionidx][:,random_indices]
TrZ2_nc = TrZ2_nc[0:,satisfying_conditionidx][:,random_indices]


#%%


print(f"For alpha {alph} meant_min, meanQ_min,count_number : {meant_min, meanQ_min,count_number}")

#
# sig_caus_shifted = (omg_caus_shifted**2 - 4*Q_caus_shifted)**0.5
# sig_caus_shifted.imag.max()

#
meant_min, meanQ_min,count_number

#%%
meanQ_min_wrong = np.mean(np.nanmin(Q_caus_shifted,axis = 0))
minQ_mean = np.min(np.nanmean(Q_caus_shifted,axis = 1))

print("min1 : {meanQ_min_wrong}, min2 : {minQ_mean}" )
#
print(ins,tot_caus)


#%%

# [markdown]
# # <center> Trajectories of non-caustics particles
# 
# 1. Find the minima of Q for caustics particles. 
# 2. Check the particles that pass between the standard deviation of the trajectory. 
# 3. Find store the average trajectory 20*tp before and after entering the trajectory. 
# 

#

#


#

print(ins)

#%%
# sig_initial = np.sqrt(omg_initial**2 - 4*Q_initial)**0.5
Q_caus_shifted_vort = Q_caus_shifted[:,Q_initial>0]
Q_caus_shifted_strain = Q_caus_shifted[:,Q_initial<0]
Q_caus_shifted_extreme_vort = Q_caus_shifted[:,np.abs(omg_initial)>2*xi_rms]
Q_caus_shifted_extreme_strain = Q_caus_shifted[:,sig_initial>2*sig_rms]
Q_caus_shifted_extreme_pos_Q = Q_caus_shifted[:,Q_initial>4*Q_rms]
Q_caus_shifted_extreme_neg_Q = Q_caus_shifted[:,Q_initial<-4*Q_rms]


#%%



print(Q_caus_shifted_strain.shape,Q_caus_shifted_vort.shape,Q_caus_shifted_extreme_vort.shape, Q_caus_shifted_extreme_strain.shape, Q_caus_shifted_extreme_pos_Q.shape,Q_caus_shifted_extreme_neg_Q.shape
)
with h5py.File(prtcl_loadPath/f"caus-details.hdf5",'a') as f:
    
    print(f"Saved the data in {str(prtcl_loadPath)}")
    save_dset(f,"t_mins", data = t_mins)
    save_dset(f,"pos_caus_shifted",data = pos_caus_shifted)
    save_dset(f,"vel_caus_shifted",data = vel_caus_shifted)
    save_dset(f,"u_caus_shifted",data = u_caus_shifted)
    save_dset(f,"Q_caus_shifted",data = Q_caus_shifted)
    save_dset(f,"omg_caus_shifted",data = omg_caus_shifted)
    save_dset(f,"a_caus_shifted",data = a_caus_shifted)
    save_dset(f,"b_caus_shifted",data = b_caus_shifted)
    save_dset(f,"c_caus_shifted",data = c_caus_shifted)
    save_dset(f,"TrZ_shifted",data = TrZ_shifted)
    save_dset(f,"TrZ2_shifted",data = TrZ2_shifted)
    save_dset(f,"Q_caus_shifted_vort",data = Q_caus_shifted_vort)
    save_dset(f,"Q_caus_shifted_strain",data = Q_caus_shifted_strain)
    save_dset(f,"Q_caus_shifted_extreme_vort",data = Q_caus_shifted_extreme_vort)
    save_dset(f,"Q_caus_shifted_extreme_strain",data = Q_caus_shifted_extreme_strain)
    save_dset(f,"Q_caus_shifted_extreme_pos_Q",data = Q_caus_shifted_extreme_pos_Q)
    save_dset(f,"Q_caus_shifted_extreme_neg_Q",data = Q_caus_shifted_extreme_neg_Q)
    save_dset(f,"t_caus_len",data = t_caus_len)
    save_dset(f,"caus_min_Qs",data = caus_min_Qs)
    save_dset(f,"noncaus_min_Qs",data = noncaus_min_Qs)    
    print(f'Saved caustics data in {str(prtcl_loadPath)}')

del Q_caus_shifted,omg_caus_shifted,TrZ_shifted,TrZ2_shifted,Q_caus_shifted_vort,Q_caus_shifted_strain,Q_caus_shifted_extreme_vort,Q_caus_shifted_extreme_strain,Q_caus_shifted_extreme_pos_Q,Q_caus_shifted_extreme_neg_Q,t_caus_len,pos_caus_shifted,vel_caus_shifted,a_caus_shifted,b_caus_shifted,c_caus_shifted,u_caus_shifted
#%%
pos_nc_shifted = np.zeros((2*maxsize, len(random_indices),d))*np.nan
vel_nc_shifted = np.zeros((2*maxsize, len(random_indices),d))*np.nan
u_nc_shifted = np.zeros((2*maxsize, len(random_indices),d))*np.nan
Q_nc_shifted = np.zeros((2*maxsize,len(random_indices)))*np.nan
xi_nc_shifted = np.zeros((2*maxsize,len(random_indices)))*np.nan
a_nc_shifted = np.zeros((2*maxsize,len(random_indices)))*np.nan
b_nc_shifted = np.zeros((2*maxsize,len(random_indices)))*np.nan
c_nc_shifted = np.zeros((2*maxsize,len(random_indices)))*np.nan
TrZ_nc_shifted = np.zeros((2*maxsize,len(random_indices)))*np.nan
TrZ2_nc_shifted = np.zeros((2*maxsize,len(random_indices)))*np.nan
nc_sp_count_number = 0
for i in range(Qinterp_nc.shape[1]):
    tidx  = np.argwhere(Qinterp_nc[:,i] < minQ).ravel()[0]
    tshiftidx = tidx
    tminim,Qminma, nc_sp_count_number = compute_next_minima(times[tidx:],Qinterp_nc[tidx:,i],nc_sp_count_number)
    tshiftidx = np.argwhere(times <= tminim).ravel()[-1]
    if len(times) - tshiftidx < maxsize: 
        idxinit = max(0,tshiftidx-maxsize)
        idxstart = 0 if idxinit > 0 else maxsize - tshiftidx
        # print(len(times),tshiftidx,maxsize,idxinit)
        pos_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i,:] = pos_nc[tshiftidx:,i,:]
        pos_nc_shifted[idxstart:maxsize,i,:] = pos_nc[idxinit:tshiftidx,i,:]
        
        vel_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i,:] = vel_nc[tshiftidx:,i,:]
        vel_nc_shifted[idxstart:maxsize,i,:] = vel_nc[idxinit:tshiftidx,i,:]
        
        u_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i,:] = uinterp_nc[tshiftidx:,i,:]
        u_nc_shifted[idxstart:maxsize,i,:] =  uinterp_nc[idxinit:tshiftidx,i,:]
        
        Q_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i] = Qinterp_nc[tshiftidx:,i]
        Q_nc_shifted[idxstart:maxsize,i] = Qinterp_nc[idxinit:tshiftidx,i]
        
        xi_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i] = xiinterp_nc[tshiftidx:,i]
        xi_nc_shifted[idxstart:maxsize,i] = xiinterp_nc[idxinit:tshiftidx,i]
        
        a_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i] = ainterp_nc[tshiftidx:,i]
        a_nc_shifted[idxstart:maxsize,i] = ainterp_nc[idxinit:tshiftidx,i]
        
        b_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i] = cinterp_nc[tshiftidx:,i]
        b_nc_shifted[idxstart:maxsize,i] = xiinterp_nc[idxinit:tshiftidx,i]
        
        c_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i] = cinterp_nc[tshiftidx:,i]
        c_nc_shifted[idxstart:maxsize,i] = cinterp_nc[idxinit:tshiftidx,i]
        
        TrZ_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i] = TrZ_nc[tshiftidx:,i]
        TrZ_nc_shifted[idxstart:maxsize,i] = TrZ_nc[idxinit:tshiftidx,i]
        
        TrZ2_nc_shifted[maxsize:maxsize + len(times) -tshiftidx ,i] = TrZ2_nc[tshiftidx:,i]
        TrZ2_nc_shifted[idxstart:maxsize,i] = TrZ2_nc[idxinit:tshiftidx,i]
        
    elif tshiftidx < maxsize:
        idxfin = min(tshiftidx + maxsize, len(times))
        idxend = len(times)- tshiftidx if idxfin == len(times) else 2*maxsize
        
        pos_nc_shifted[maxsize:idxend,i,:] = pos_nc[tshiftidx:idxfin,i,:]
        pos_nc_shifted[maxsize - tshiftidx:maxsize,i,:] = pos_nc[:tshiftidx,i,:]
        
        vel_nc_shifted[maxsize:idxend,i,:] = vel_nc[tshiftidx:idxfin,i,:]
        vel_nc_shifted[maxsize - tshiftidx:maxsize,i,:] = vel_nc[:tshiftidx,i,:]

        u_nc_shifted[maxsize:idxend,i,:] = uinterp_nc[tshiftidx:idxfin,i,:]
        u_nc_shifted[maxsize - tshiftidx:maxsize,i,:] = uinterp_nc[:tshiftidx,i,:]
        
        Q_nc_shifted[maxsize:idxend,i] = Qinterp_nc[tshiftidx:idxfin,i]
        Q_nc_shifted[maxsize - tshiftidx:maxsize,i] = Qinterp_nc[:tshiftidx,i]
        
        xi_nc_shifted[maxsize:idxend,i] = xiinterp_nc[tshiftidx:idxfin,i]
        xi_nc_shifted[maxsize - tshiftidx:maxsize,i] = xiinterp_nc[:tshiftidx,i]
        
        a_nc_shifted[maxsize:idxend,i] = ainterp_nc[tshiftidx:idxfin,i]
        a_nc_shifted[maxsize - tshiftidx:maxsize,i] = ainterp_nc[:tshiftidx,i]
        
        b_nc_shifted[maxsize:idxend,i] = binterp_nc[tshiftidx:idxfin,i]
        b_nc_shifted[maxsize - tshiftidx:maxsize,i] = binterp_nc[:tshiftidx,i]
        
        c_nc_shifted[maxsize:idxend,i] = cinterp_nc[tshiftidx:idxfin,i]
        c_nc_shifted[maxsize - tshiftidx:maxsize,i] = cinterp_nc[:tshiftidx,i]
        
        TrZ_nc_shifted[maxsize:idxend,i] = TrZ_nc[tshiftidx:idxfin,i]
        TrZ_nc_shifted[maxsize - tshiftidx:maxsize,i] = TrZ_nc[:tshiftidx,i]
        
        TrZ2_nc_shifted[maxsize:idxend,i] = TrZ2_nc[tshiftidx:idxfin,i]
        TrZ2_nc_shifted[maxsize - tshiftidx:maxsize,i] = TrZ2_nc[:tshiftidx,i]
        
    else: 
        pos_nc_shifted[maxsize:,i,:] = pos_nc[tshiftidx:tshiftidx+maxsize,i,:]
        pos_nc_shifted[:maxsize,i,:] = pos_nc[tshiftidx-maxsize:tshiftidx,i,:]
        
        vel_nc_shifted[maxsize:,i,:] = vel_nc[tshiftidx:tshiftidx+maxsize,i,:]
        vel_nc_shifted[:maxsize,i,:] = vel_nc[tshiftidx-maxsize:tshiftidx,i,:]
 
        u_nc_shifted[maxsize:,i,:] = uinterp_nc[tshiftidx:tshiftidx+maxsize,i,:]
        u_nc_shifted[:maxsize,i,:] = uinterp_nc[tshiftidx-maxsize:tshiftidx,i,:]
        
        Q_nc_shifted[maxsize:,i] = Qinterp_nc[tshiftidx:tshiftidx+maxsize,i]
        Q_nc_shifted[:maxsize,i] = Qinterp_nc[tshiftidx-maxsize:tshiftidx,i]
        
        
        xi_nc_shifted[maxsize:,i] = xiinterp_nc[tshiftidx:tshiftidx+maxsize,i]
        xi_nc_shifted[:maxsize,i] = xiinterp_nc[tshiftidx-maxsize:tshiftidx,i]
        
        
        a_nc_shifted[maxsize:,i] = ainterp_nc[tshiftidx:tshiftidx+maxsize,i]
        a_nc_shifted[:maxsize,i] = ainterp_nc[tshiftidx-maxsize:tshiftidx,i]
        
        
        b_nc_shifted[maxsize:,i] = binterp_nc[tshiftidx:tshiftidx+maxsize,i]
        b_nc_shifted[:maxsize,i] = binterp_nc[tshiftidx-maxsize:tshiftidx,i]
        
        
        c_nc_shifted[maxsize:,i] = cinterp_nc[tshiftidx:tshiftidx+maxsize,i]
        c_nc_shifted[:maxsize,i] = cinterp_nc[tshiftidx-maxsize:tshiftidx,i]
        
        
        TrZ_nc_shifted[maxsize:,i] = TrZ_nc[tshiftidx:tshiftidx+maxsize,i]
        TrZ_nc_shifted[:maxsize,i] = TrZ_nc[tshiftidx-maxsize:tshiftidx,i]
        
        
        TrZ2_nc_shifted[maxsize:,i] = TrZ2_nc[tshiftidx:tshiftidx+maxsize,i]
        TrZ2_nc_shifted[:maxsize,i] = TrZ2_nc[tshiftidx-maxsize:tshiftidx,i]
        
        
        
            
del Qinterp_nc,xiinterp_nc,TrZ_nc,TrZ2_nc,ainterp_nc,binterp_nc,cinterp_nc,pos_nc,vel_nc,uinterp_nc

#%%


#

#

#

#

    
# print(f"fraction of caustics: {np.sum(caus_count,axis = 1)/Nprtcl} at stokes number {st/tf}")
def first_value(x):
    """This returns the first non-zero value of the shifted Q_caus"""
    
    diff_gate = np.cumsum((x !=0)*1,axis = 0)
    return x[diff_gate == 1]
    



#
with h5py.File(prtcl_loadPath/f"caus-details.hdf5",'a') as f:
    
    save_dset(f,"pos_nc_shifted",data = pos_nc_shifted)
    save_dset(f,"vel_nc_shifted",data = vel_nc_shifted)
    save_dset(f,"u_nc_shifted",data = u_nc_shifted)
    save_dset(f,"Q_nc_shifted",data = Q_nc_shifted)
    save_dset(f,"xi_nc_shifted",data = xi_nc_shifted)
    save_dset(f,"a_nc_shifted",data = a_nc_shifted)
    save_dset(f,"b_nc_shifted",data = b_nc_shifted)
    save_dset(f,"c_nc_shifted",data = c_nc_shifted)
    save_dset(f,"TrZ_nc_shifted",data = TrZ_nc_shifted)
    save_dset(f,"TrZ2_nc_shifted",data = TrZ2_nc_shifted)

    print(f"Saved non-caustics the data in {str(prtcl_loadPath)}")



#%%


# %%
#%%

# %%
