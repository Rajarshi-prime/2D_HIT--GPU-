# %%
import numpy as np
import matplotlib.pyplot as plt
import pathlib,sys,os,json,h5py
from scipy.fft import rfft2,irfft2,fftshift,ifftshift
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
# import seaborn as sns
aspect = 8/397.5
default_fontsize = round(12*72*aspect)
mpl.rc("text", usetex = True)
default_fontsize = round(aspect*12*72)
mpl.rcParams['font.size'] = default_fontsize
mpl.rcParams['figure.figsize'] = (12,8)
mpl.rcParams['legend.fontsize'] = default_fontsize
mpl.rcParams['xtick.labelsize'] = default_fontsize
mpl.rcParams['ytick.labelsize'] = default_fontsize
mpl.rcParams['axes.labelsize'] = default_fontsize
mpl.rcParams['axes.titlesize'] = default_fontsize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %%
# paramfile = '/home/rajarshi.chattopadhyay/pokemon_data/codes/parameters.json'
# with open(paramfile,'r') as jsonFile: params = json.load(jsonFile)

# d = params["d"] # Dimension
d = 2
# nu =params["nu"] # Viscosity
nu = 1/125000.0
Re = 1/nu if nu > 0 else np.inf # Reynolds number
# N = Nx = Ny = params["N"] # Grid size
N = Nx = Ny = 1024 # Grid size
# dt = params["dt"] # Timestep
dt = 0.005 # Timestep
# T = params["T"] # Final time
T = 500
# alph = params["alph"] # Density of the particles
alph = 1.0
# eta = params["eta"]/(Nx//3) # Desired Kolmogorov length scale
# kf = params["kf"] # Forcing wavenumber
kf = 4
# Nprtcl = int(params["Nprtcl"]*Nx*Ny) # Number of particles
Nprtcl = int(0.3*Nx*Ny) # Number of particles

# tf = params["tf"] # Kolmogorov timescale
tf = 3.0
# st = params["st"]*tf # Particle Stokes number
st = .3
tp = st*tf

# linnu = params["linnu"] # Linear viscosity
# order = params["order"] # Order of the scheme
order = 6 # Order of the scheme

# %%
iname = "_bspline" if order >1 else ""

# %%
Re= 125000.0
N = 1024
dt = 0.005
iname = "bspline"
savePlot = pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/Re_{np.round(Re,2)},dt_{dt},N_{N}/")
savePlot.mkdir(parents=True, exist_ok=True)
def loadPath(Re,dt = dt, N =N,st = st):
    if st ==0.3 : return pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/Re_{np.round(Re,2):.1f},dt_{dt},N_{N}/")
    else: return pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/brenner/Re_{np.round(Re,2):.1f},dt_{dt},N_{N}/")
loadPath(1000000.0).exists()
def prtcl_loadPath(alpha,st,Re=Re,dt = dt, N =N):
    return loadPath(Re,dt,N,st)/f"alpha_{alpha:.2f}_prtcl/St_{st:.2f}/"
# %%
Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
x,y = np.meshgrid(X,Y,indexing="ij")
PI = np.pi
TWO_PI = 2*PI

## It is best to define the function which returns the real part of the iifted function as ifft. 
def ifft2(x):
    return irfft2(x,(Nx,Ny),axes = (0,1))

def fft2(x):
    return rfft2(x,axes = (0,1))

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
normalize = np.where((ky== 0) + (ky == Ny//2) , 1/(Nx**4/TWO_PI**2),2/(Nx**4/TWO_PI**2))

# %%
xi_last = np.load(loadPath(Re)/f"time_400.00/w.npz")["vorticity"]
# xi_last = np.load(f"/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/data/Re_1000000.0,dt_0.005,N_1024/last/w.npy")
xi_last_r = ifft2(xi_last)
psi_last = -lapinv*xi_last
e = np.sum(0.5*(xi_last*np.conjugate(psi_last))*normalize).real
# print(xi_last_r.max(),np.sqrt(np.mean(xi_last_r**2)))
u =1j* ky*lapinv*xi_last
v = -1j*kx*lapinv*xi_last
ur = ifft2(u)
vr = ifft2(v)
A = np.zeros((d,d,Nx,Ny//2+1),dtype = np.complex128)
A[0,0] = 1j*kx*u
A[0,1] = 1j*ky*u
A[1,0] = 1j*kx*v
A[1,1] = 1j*ky*v
Ar = np.zeros((d,d,Nx,Ny),dtype = np.float64)
for i in range(d):
    for j in range(d):
        Ar[i,j] = ifft2(A[i,j])
        
Q_field = -0.5*st*np.einsum('ij...,ji...->...',Ar,Ar)
Xplot,Yplot = np.linspace(0,Lx,Nx+1,endpoint= True), np.linspace(0,Ly,Ny+1,endpoint= True)
xnew = np.zeros((Nx+1,Ny+1))
Qnew = np.zeros((Nx+1,Ny+1))
xnew[1:,1:] = np.roll(xi_last_r, -1,axis = (0,1))
Qnew[1:,1:] = np.roll(Q_field, -1,axis = (0,1))
xnew[:-1,:-1] = xi_last_r    
urms = (np.mean(ur**2 + vr**2))**0.5
t_l = 2*np.pi/kf/urms
Qnew[:-1,:-1] = Q_field
print(e,urms,t_l)

# %%
normalize = np.where((ky== 0) + (ky == Ny//2) , 1/(Nx**4/TWO_PI**2),2/(Nx**4/TWO_PI**2))
shells = np.arange(-0.5, round((N//2 + 1)*d**0.5) + 0.5)
shells[0] = 0. 
def e2d_to_1d(x):
    return (np.histogram(k.ravel(),bins = shells,weights=x.ravel() )[0]).real

# %%
k = np.sqrt(kx**2 + ky**2)
e_arr = e2d_to_1d(0.5*(np.abs(u)**2 + np.abs(v)**2)*normalize)

# %%
# # Energy Dissipation

# %%
nu =8e-6
Re = 1/nu
dissip = 0.0
count = 0
for time in np.arange(0,300):
    e_arr = np.load(loadPath(Re,st = .2)/f"time_{time:.2f}/e_arr.npz")["energy"]
    # print(e_arr.sum())
    kplot = np.arange(e_arr.size)
    dissip =  dissip + np.sum(2*(nu*kplot*2 )*(2*np.pi)**(-2.0)*e_arr )
    count+= 1
dissip = dissip/count
eta = (nu**3/dissip)**(1/4)
t_k = eta**2/nu 
print(t_k,eta*N/3,dissip,e_arr.sum()/(2*np.pi)**2)

# %%
# # Vorticity dissipation

# %%
e_arr = np.load(loadPath(Re)/f"time_300.00/e_arr.npz")["energy"]
# e_arr = np.load(f"/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/data/Re_1000000.0,dt_0.005,N_1024/last/e_arr.npy")
print(e_arr.sum())
kplot = np.arange(e_arr.size)
dissip = np.sum(2*(nu*kplot**4 )*(2*np.pi)**(-2.0)*e_arr )
# dissip = np.sum(2*(nu*kplot**2 )*e_arr )
eta = (nu**3/dissip)**(1/6.)
t_k = eta**2/nu
print(t_k,eta*N/3,dissip,e_arr.sum()/(2*np.pi)**2)

# %%
mpl.rc("text", usetex = False)
plt.figure(figsize=(12,6))
fig = plt.gcf()
ax = plt.gca()
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1.0)
mpl.rc("text", usetex = False)
kplot = np.arange(e_arr.size)
norm_val = lambda x: x/x[0] #! makes the value of the first element to be 1
ydat = norm_val(e_arr[1:N//2 + 1]*kplot[1:N//2 + 1]**(0/3))
plt.plot(kplot[1:N//2 + 1],ydat,'.-')
# plt.axvline(Nx//3*0.5,color = "black")
# plt.axvline(Nx//3*0.6,color = "black")
# plt.axvline(2,color = "black")
# plt.axvline(4,color = "black")
xline1 = np.array([1.,512.])
yline1 = xline1**(-9/3)
yline2 = xline1**(-12/3)
yline3 = xline1**(-5/3)
yline1[:] = yline1/yline1[0]*ydat[1]*10
yline2[:] = yline2/yline2[0]*ydat[1]*10
yline3[:] = yline3/yline3[0]*ydat[1]*10
# plt.plot(xline1,yline1,"--",linewidth = 2)
plt.plot(xline1,yline2,"--",linewidth = 2)
# plt.plot(xline1,yline3,"--",linewidth = 2)
# plt.axvline(4,linestyle = '--',color = "black")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-12,plt.ylim()[-1])
# plt.xlim(120,180)
plt.xlabel(r"$k$")
plt.ylabel(r"$E(k)$")
plt.grid()
np.sum(e_arr)
# plt.savefig(savePlot/"e_k_divya1.png",dpi = 300)
plt.show()
plt.close()
# print(t_k,eta*(N//3),dissip,e_arr.sum())


# %%
# PiYG
# PRGn
# BrBG
# mor
# RdGy
# RdBu
# RdYlBu
# RdYlGn
# Spectral
# coolwarm
# bwr
# seismic

# %%
dissip = (10*nu)**3/(3/N )**4
dissip

# %%
def initialize_b_spline(order):
    """
    Initializes the Matrix M and coefficient array ck in 2D according to hinsberg et al.
    ck is not computed using the formula. Rather doing 1/F(b_d) where F is the fourier transform of the b_d function given in (7.6).
    Mmat is compute using (7.4)
    """
    import scipy.fft as fft
    import numpy as npp
    import math
    Mmat = npp.zeros((order,order))
    for i in range(order):
        for j in range(order):
            for s in range(j,order):
                Mmat[j,i] += (-1)**(s-j)*math.factorial(order)*(order - s-1)**(order-i-1)/(math.factorial(s-j)*math.factorial(order +j -s))
            
        Mmat[:,i] = Mmat[:,i]/(math.factorial(order-i-1)*math.factorial(i))
    Mmat = np.array(Mmat)

    def bj(x,j): 
        bj = 0
        for i in range(order):
            bj += Mmat[j,i]*(x -order/2 + j+1)**(i)
        return bj*((x -order/2 + j+1) < 1)*(x -order/2 + j+1 >= 0)

    bm = 0. 
    xeval = X/dx*((X//dx<Nx/2)) + (X/dx - Nx)*(X//dx>=Nx/2)
    for j in range(order):
        bm += bj(xeval,j)


    bk = fft.fft(bm)
    ck = 1./bk
    ckxky = ck[:,None]*ck[None,:]
    cxy = fft.ifft2(ckxky).real
    ck2d = fft.rfft2(cxy) #! This will be multiplied with all the fields to be interpolated. 
    
    del bm,bk,ck,ckxky,cxy
    
    return ck2d,Mmat

# %%
ck2d,Mmat = initialize_b_spline(order)
Mmat.shape

# %%
temparr = np.zeros((Nprtcl))
slx = np.zeros((Nprtcl),dtype = int)
sly = np.zeros((Nprtcl),dtype = int)
idx  = np.zeros((Nprtcl,d),dtype = int)
delx = np.zeros((Nprtcl,d))
poly = np.zeros((Nprtcl,d,order))
umat = np.zeros((Nprtcl,d))
Amat = np.zeros((Nprtcl,d,d))


# %%
nums = np.arange(order)

def interp_spline(pos,u_field,A_field):
    """
    b-spline interpolation of the specified order given in Hinsberg et al.
    """
    global umat,Amat,deludeltmat,DADtmat,ugrdumat
    umat[:] = 0.0
    Amat[:] = 0.0

    
    idx[:] = (pos//dx).astype(int)
    delx[:] = (pos%dx)/dx    
    poly[:] = delx[...,None]**nums

    for i in range(order):
        for j in range(order):
            temparr[:] = np.einsum('p,q,...p,...q->...',Mmat[i],Mmat[j],poly[...,0,:],poly[...,1,:]) #! For saving computations
            slx[:] = (idx[:,0]-order//2 + 1 + i)%N #! For saving computations
            sly[:] = (idx[:,1]-order//2 + 1 + j)%N #! For saving computations
            
            umat  += u_field[slx,sly,...]*temparr[:,None]            
            
            Amat  += A_field[slx,sly]*temparr[:,None,None]
            
            
    return umat,Amat

# %%

# def interp_spline_vectorized(pos, u_field, A_field, order, dx, N):
#     """
#     Vectorized b-spline interpolation using CuPy.
    
#     Parameters:
#     -----------
#     pos : np.ndarray
#         Position array to interpolate at, shape (n_points, 2)
#     u_field : np.ndarray
#         Field values for u, shape (2, N, N)
#     A_field : np.ndarray
#         Field values for A, shape (2, 2, N, N)
#     order : int
#         Order of b-spline interpolation
#     dx : float
#         Grid spacing (float64)
#     N : int
#         Grid size
    
#     Returns:
#     --------
#     tuple(np.ndarray, np.ndarray)
#         Interpolated values for u (shape: (n_points, 2)) and 
#         A (shape: (n_points, 2, 2))
#     """
    
#     @np.vectorize('(),(),(),(),(),()')
#     def interpolate_point(x, y, u_field_flat, A_field_flat, dx_val, N_val):
#         # Initialize arrays for this point
#         idx = np.array([(x//dx_val).astype(int), (y//dx_val).astype(int)])
#         delx = np.array([x%dx_val/dx_val, y%dx_val/dx_val])
        
#         # Initialize result arrays
#         u_result = np.zeros(2)
#         A_result = np.zeros((2, 2))
        
#         # Generate polynomial terms
#         nums = np.arange(order)
#         poly = np.array([delx[0]**nums, delx[1]**nums])
        
#         # Perform interpolation
#         for i in range(order):
#             for j in range(order):
#                 # Compute indices with periodic boundary conditions
#                 slx = (idx[0]-1 + i)%N_val
#                 sly = (idx[1]-1 + j)%N_val
                
#                 # Get M-matrix coefficients
#                 Mi = Mmat[i]
#                 Mj = Mmat[j]
                
#                 # Compute basis function value
#                 temp = 0.0
#                 for p in range(order):
#                     for q in range(order):
#                         temp += Mi[p] * Mj[q] * poly[0,p] * poly[1,q]
                
#                 # Update u and A values
#                 idx_flat = slx * N_val + sly
#                 for k in range(2):
#                     u_result[k] += u_field_flat[k * N_val * N_val + idx_flat] * temp
                
#                 for k in range(2):
#                     for l in range(2):
#                         A_result[k,l] += A_field_flat[(k * 2 + l) * N_val * N_val + idx_flat] * temp
        
#         return (u_result[0], u_result[1], 
#                 A_result[0,0], A_result[0,1], 
#                 A_result[1,0], A_result[1,1])
    
#     # Flatten input arrays and prepare for vectorization
#     u_field_flat = u_field.reshape(2, -1).ravel()
#     A_field_flat = A_field.reshape(4, -1).ravel()
    
#     # Apply vectorized function with float64 dx
#     results = interpolate_point(
#         pos[:,0], pos[:,1],
#         u_field_flat, A_field_flat,
#         np.float64(dx), np.int32(N)  # Changed to float64 for dx
#     )
    
#     # Reshape results
#     n_points = pos.shape[0]
#     umat = np.stack([results[0], results[1]], axis=-1)
#     Amat = np.stack([
#         [results[2], results[3]],
#         [results[4], results[5]]
#     ], axis=-2)
    
#     return umat, Amat

# %%
u_field = np.zeros((Nx,Ny,2))
A = np.zeros((Nx,Ny//2+1,d,d),dtype = np.complex128)
Ar = np.zeros((Nx,Ny,d,d),dtype = np.float64)
#%%
Qmax,Qmin = 1,-1
Qbins = np.linspace(Qmin,Qmax, 6001)
Qvals = 0.5*(Qbins[1:] + Qbins[:-1])
dQval  = Qvals[1] - Qvals[0]
def Q_hists(st,alpha_values):

    Q_field_avg = np.zeros((alpha_values.size, Qvals.size))
    Q_particle_avg = np.zeros((alpha_values.size, Qvals.size))
    Q_caus_avg = np.zeros((alpha_values.size, Qvals.size))
    for ii,alph in enumerate(alpha_values[:]):
        with h5py.File(prtcl_loadPath(alph,st) / f"caus-details.hdf5",'r') as f:
            print(prtcl_loadPath(alph,st), f.keys())
            Q_field_avg[ii] = np.mean(f["Q_field_pdf"][:],axis = 0)
            Q_particle_avg[ii] = np.mean(f["Q_particle_pdf"][:],axis = 0)
            Q_caus_avg[ii] = np.mean(f["Q_caus_pdf"][:],axis = 0)
    
    return Q_field_avg,Q_particle_avg,Q_caus_avg
        
        

#%%

# %%
mpl.rc("text", usetex = True)

hex_colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

# Create the colormap
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("my_palette", hex_colors, N=256)
cls = custom_cmap(np.linspace(0,1,4))
ls = ['--','-']
mrkrs = ['o','s']


fig = plt.figure(figsize = (12,12))
gs = mpl.gridspec.GridSpec(3, 6, figure=fig)
ax11 = fig.add_subplot(gs[0, 0:2], )
ax12 = fig.add_subplot(gs[0, 2:4], )
ax13 = fig.add_subplot(gs[0, 4:6], )
ax21 = fig.add_subplot(gs[1, 0:2], )
ax22 = fig.add_subplot(gs[1, 2:4], )
ax23 = fig.add_subplot(gs[1, 4:6], )
ax3 = fig.add_subplot(gs[2, 1:4], )
inset_ax = inset_axes(ax3, width="30%", height="30%", loc="upper right",borderpad = 0) 
axs = [ax11, ax12, ax13,ax21,ax22,ax23, ax3]

axs[0].set_ylabel("$y$")
axs[3].set_ylabel("$y$")
for ax in axs[3:6]:
    ax.set_xlabel("$x$")
for ax in axs[:-1]:  ax.set_aspect('equal', adjustable='box')
ax3.set_yscale('log')
ax3.set_xlim(-0.2,0.3)
ax3.set_xlabel(r'$Q/St^2$')
ax3.set_ylabel(r'PDF')
ax3.set_ylim(1e-2,100)
norm = TwoSlopeNorm(vcenter = 0,vmax = 1.0,vmin=-1.0)
gs.update(wspace=0.3, hspace=0.4)

print(Re)
t = 300
idx = 0
nums = ["(a)", "(b)","(c)","(d)","(e)","(f)","(g)"]
for row, st in enumerate([0.3,0.4]):
    for colmn, alph in enumerate([0.7,0.77,1.0]):
        xi_last = np.load(loadPath(Re,st = st)/f"time_{t:.2f}/w.npz")["vorticity"]
        pos= np.load(loadPath(Re,st = st)/f"alpha_{alph:.2f}_prtcl/St_{st:.2f}/time_{t:.2f}/pos.npz")["pos"]
        caus_count = np.load(loadPath(Re,st = st)/f"alpha_{alph:.2f}_prtcl/St_{st:.2f}/time_{t:.2f}/caus_count.npz")['caustics_count']
        vel= np.load(loadPath(Re,st = st)/f"alpha_{alph:.2f}_prtcl/St_{st:.2f}/time_{t:.2f}/vel.npz")["vel"]
        TrZ = np.einsum('...ii->...',np.load(loadPath(Re,st = st)/f"alpha_{alph:.2f}_prtcl/St_{st:.2f}/time_{t:.2f}/prtcl_Z.npz")["Zmatrix"])
        causidx = np.argwhere(caus_count>0)
        print(f" Total number of caustics : {(caus_count>0).sum()/Nprtcl}",idx)

        psi = -xi_last*lapinv

        u_field[...,0] = ifft2(1j * ky*psi)
        u_field[...,1] = ifft2(-1j * kx*psi) 
        xi_last_r = ifft2(xi_last)

        u = 1j* ky*lapinv*xi_last
        v = -1j*kx*lapinv*xi_last
        A[...,0,0] = 1j*kx*u
        A[...,0,1] = 1j*ky*u
        A[...,1,0] = 1j*kx*v
        A[...,1,1] = 1j*ky*v

        Ar = ifft2(A)
                        
        Q = -0.5*tp**2*np.einsum('...ij,...ji->...',Ar,Ar)
        Xplot,Yplot = np.linspace(0,Lx,Nx+1,endpoint= True), np.linspace(0,Ly,Ny+1,endpoint= True)
        xnew = np.zeros((Nx+1,Ny+1))
        Qnew = np.zeros((Nx+1,Ny+1))
        xnew[:-1,:-1] = xi_last_r    
        xnew[-1,:-1]= xi_last_r[0,:]
        xnew[:,-1] = xnew[:,0]
        Qnew[:-1,:-1] = Q    
        Qnew[-1,:-1] = Q[0,:]    
        Qnew[:,-1] = Qnew[:,0]

        


        axs[idx].set_title(fr"${nums[idx]}~\alpha = {alph:.2f}, St = {st*0.75:.3f}$")
        axs[idx].plot(pos[...,0],pos[...,1],'.', color = "black", markersize = 0.1)
        axs[idx].plot(pos[causidx,0],pos[causidx,1],'.', color = "#01dcd6", markersize = 0.1,alpha = 1)
        p1 = axs[idx].imshow(xnew[:,::-1].T, cmap="RdBu", norm=norm,extent=(0, 2*np.pi, 0, 2*np.pi))
        
        idx += 1
        
    

    Q_field_avg,Q_particle_avg,Q_caus_avg = Q_hists(st,np.array([0.7,0.77,1.0]))
    Q_field_mean = np.sum(Q_field_avg*Qvals*dQval)
    Q_particle_mean = np.sum(Q_particle_avg*Qvals*dQval,axis = -1)
    xdat = Qvals/(st*4.0)**2
    dxdat = xdat[1] - xdat[0]

    ydat = Q_field_avg[-1]/( dxdat )
    if idx == 0: ax3.plot(xdat,ydat,label = fr"$\mathbf{{u}}$",color ='black',linewidth = 2)
    else: ax3.plot(xdat,ydat,color ='black',linewidth =2)


    for colmn, alph in enumerate([0.7,0.77,1.0]):
        ydat = Q_particle_avg[colmn]/dxdat  
        if row ==1:
            ax3.plot(xdat,ydat,label = rf"${alph:.2f}$",color = cls[colmn],ls = ls[row],linewidth = 2)
        else: 
            ax3.plot(xdat,ydat, color = cls[colmn],ls = ls[row],linewidth = 2)

        
        inset_ax.plot(st*0.75,Q_particle_mean[colmn], color=cls[colmn], marker = mrkrs[row])  
        
        
inset_ax.set_xlabel(r"$St$")
inset_ax.set_ylabel(r"$\langle Q/St^2 \rangle$")
inset_ax.set_xlim(0.2, 0.35)
inset_ax.set_ylim(-1e-5, -1e-6)
inset_ax.set_yticks([-2e-6, -8e-6])
ax3.set_title(f"$(g)$")
fig.tight_layout()
handles = [plt.Line2D([0], [0], lw=2,color="black", ls  =ls[0]),    plt.Line2D([0], [0], lw=2, color="black",ls = ls[1])]  
leg1 = ax3.legend(handles=handles, labels=[r'$0.225$',r'$0.3$'],handlelength = 1.,loc = "center right",frameon = True,bbox_to_anchor=(1.35, 0.5),ncol=1,title = r"$St$")
leg2 = ax3.legend(handlelength = 1.0,ncols =1,bbox_to_anchor=(-0.45, 0.5),loc = "center left",frameon = True,title = r"$\alpha$")
ax3.add_artist(leg1)
savePlot = pathlib.Path(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/Re_{Re:.1f},dt_{dt},N_{N}/st_{st:.2f}/alph_{alph:.2f}/")
savePlot.mkdir(exist_ok = True, parents= True)
cbar = fig.colorbar(p1, ax=axs[:-1], orientation="vertical", shrink=0.5, extend="both")
cbar.ax.set_title(r"$\omega$",rotation = 0)
plt.savefig(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/snapshot_{t:.2f}_PDF.pdf",bbox_inches = "tight", dpi = 300,pad_inches = 0.0,format = 'pdf')
plt.savefig(f"/home/rajarshi.chattopadhyay/pokemon_data/Plots/bspline/Re_125000.0,dt_0.005,N_1024/snapshot_{t:.2f}_PDF.png",bbox_inches = "tight", dpi = 300,pad_inches = 0.0)
plt.show()
plt.clf()
# plt.close()

# %%
