
# from google.colab import drive
# drive.mount('/content/drive')

import numpy as npp
import math,h5py,jax,pathlib,json,os
import jax.numpy as np
import jax.numpy.fft as fft
from tqdm import tqdm
from time import time
from sympy import LeviCivita

curr_path = pathlib.Path("/content/drive/My Drive/Collab")
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(9231472983719)

#!  ## <center> Details of the code
#     # This program solves the 2D vorticity equation without forcing using RK4 and FFT.
#     # The 2D vorticity equation is given by d_t \xi  + u.\grad \xi= \nu \laplacian \xi.
#     # Defining stream funciton and taking fourier transpose we get
#     # The dissipation term is (\laplace)^8.
#     # In the fourier space, the equation is
#     # d_t \psi_k = \int_k' (k-k')^2/k^2 [\hat{z} . {(k-k') x k'}] \psi_k' \psi_{k-k'}  - \nu k^2 \psi_k.
# 
#     # We have a finite number of grid points i.e. we have finite number of ks.
#     # We initiate a finite number of wavenumber. Then evolve and fourier the -k^2 \psi_k at every instant.
# 
# 
#     # Ignoring the k = 0 mode as this mode of psi does not contribute in the evolution equation of xi
# 
#     # Variables with r denotes quantities in position space.
#     # Underscore denotes derivative.
# 

## ---------------- params ----------------------- ##
paramfile = curr_path/'parameters.json'
with open(paramfile,'r') as jsonFile: params = json.load(jsonFile)

d = params["d"] # Dimension
nu =params["nu"] # Viscosity
Re = 1/nu if nu > 0 else np.inf # Reynolds number
N = Nx = Ny = params["N"] # Grid size

dt = params["dt"]
T = params["T"] # Final time
einit = params["einit"] # Initial energy
kinit = params["kinit"] # Initial energy is distributed among these wavenumbers
linnu = params["linnu"] # Linear drag co-efficient
lp = params["lp"] # Power of the laplacian in the hyperviscous term
shell_count = np.array(params["shell_count"]) # The shells where the energy is injected

alph = params["alph"] # The density ratio
Nprtcl = int(params["Nprtcl"]*Nx*Ny) # Number of particles
# Nprtcl = int(0.001*Nx*Ny)
tf = params["tf"] # Final time for the particles
st = params["st"]*tf # Time period for the particles #! Stokes number when non-dimensionalize by simulation time.
tp = st # Save the particle data after this many timesteps #!(in simulation time dimension, this is the tp)
savestep = int(params["savestep"]/dt) # Save the data after this many timesteps
prtcl_savestep = int(params["prtcl_savestep"]/dt) # Save the particle data after this many timesteps
save_spectra = prtcl_savestep # Save the spectra after this many timesteps
eta = params["eta"]/(Nx//3) # Desired Kolmogorov length scale
restart = params["restart"] # Restart from the last saved data
kf = params["kf"] # Forcing wavenumber
f0 = params["f0"] # Forcing amplitude
order = params["order"] # Order of the spline interpolation
Zthresh = params["Zthresh"] # Threshold for the Z matrix
t = np.arange(0,1.1*dt + T,dt) # Time array
P = (nu**3/eta**4)/len(shell_count) # power per unit shell
iname = "bspline" if order > 1 else "linear"
## ----------------------------------------------- ##

print(f"tp is {tp}")
## ------------ Grid and related operators ------------
## Forming the 2D grid (position space)
TWO_PI = 2* np.pi
Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
Nf = Nx//2 + 1
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
x,y = np.meshgrid(X,Y,indexing="ij")

## It is best to define the function which returns the real part of the iifted function as ifft.
@jax.jit
def ifft2(x):
    return fft.irfft2(x,(Nx,Ny),axes = (0,1))

@jax.jit
def fft2(x):
    return fft.rfft2(x,axes = (0,1))


## Forming the 2D grid (k space)
Kx = 2*np.pi*np.linspace(-(Nx//2) , Nx//2 - 0.5*(1+ (-1)**(Nx%2)),Nx)/Lx
Ky = 2*np.pi*np.linspace(-(Ny//2) , Ny//2 - 0.5*(1+ (-1)**(Ny%2)),Ny)/Ly
Kx = np.append(Kx[Kx>=0], Kx[Kx<0])
Ky = np.append(Ky[Ky>=0], -Ky[0])
kvec = np.moveaxis(np.array(np.meshgrid(Kx,Ky,indexing="ij")),[0,1,2],[2,0,1])



dalcutoff = ((2*np.pi*Nx)/Lx)//3,((2*np.pi*Ny)/Ly)//3
dealias = (abs(kvec[...,0])<dalcutoff[0])*(abs(kvec[...,1])<dalcutoff[1])
## Defining the inverese laplacian.
k = np.linalg.norm(kvec, axis = -1)
lap = -k**2
# lap1 = lap.copy()
# lap1[lap1== 0] = np.inf
lapinv = dealias/np.where(lap == 0., np.inf, lap)

kint = np.clip(np.round(k),None,Nf).astype(int)
xivis =  (nu *((-lap)**lp) + linnu)*dealias ## The viscous term
shells = np.arange(-0.5,Nf)
shells = shells.at[0].set(0.)
normalize = np.where((kvec[...,1]== 0) + (kvec[...,1] == Ny//2) , 1/(Nx**4/TWO_PI**2),2/(Nx**4/TWO_PI**2))

epsilon = np.array([[float(LeviCivita(i, j)) for j in range(d)] for i in range(d)])
## ----------------------------------------------------
epsilon


## -----------------  Parameters  ----------------------
print(f"Runnig for alpha = {alph}")
## ----------------------------------------------

## -------------- Initializing the zeros arrays -----------------

#! Like a dumbass, the dimensions are at the end.
psi = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
xi = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
x_old = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
x_new = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
k1 = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
k2 = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
k3 = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
k4 = np.zeros((Nx,Ny//2 + 1),dtype = np.complex128)
DuDtk = np.zeros((Nx,Ny//2 + 1,d),dtype = np.complex128)
grdDuDtk = np.zeros((Nx,Ny//2 + 1,d,d),dtype = np.complex128)


u_field = np.zeros((Nx,Ny,d))
ugrdu_field = np.zeros((Nx,Ny,d))
A_field = np.zeros((Nx,Ny,d,d))
DuDt_field = np.zeros((Nx,Ny,d))
grdDuDt_field = np.zeros((Nx,Ny,d,d))
xi_xr = np.zeros((Nx,Ny),dtype = np.float64)
xi_yr = np.zeros((Nx,Ny),dtype = np.float64)
advterm = np.zeros((Nx,Ny),dtype = np.float64)
factor = np.zeros(Nf)
factor2d = np.zeros((Nx,Nf))

Nt = len(t)
xprtcl = np.zeros((Nprtcl,2*d))
xprtcl_new = np.zeros((Nprtcl,2*d))
Z = np.zeros((Nprtcl,d,d))
Z_new = np.zeros((Nprtcl,d,d))
TrZ = np.zeros((Nprtcl))
xtemp = np.zeros((Nprtcl,2*d))
yprtcl = np.zeros((Nprtcl,2*d))

k1p = np.zeros((Nprtcl,2*d))
k2p = np.zeros((Nprtcl,2*d))
k3p = np.zeros((Nprtcl,2*d))
k4p = np.zeros((Nprtcl,2*d))

k1Z = np.zeros((Nprtcl,d,d))
k2Z = np.zeros((Nprtcl,d,d))
k3Z = np.zeros((Nprtcl,d,d))
k4Z = np.zeros((Nprtcl,d,d))


idx  = np.zeros((Nprtcl,d),dtype = int)
delx = np.zeros((Nprtcl,d))
poly = np.zeros((Nprtcl,d,order))
umat = np.zeros((Nprtcl,d))
Amat = np.zeros((Nprtcl,d,d))
DuDt_mat = np.zeros((Nprtcl,d))
grdDuDt_mat = np.zeros((Nprtcl,d,d))

temparr = np.zeros((Nprtcl))
slx = np.zeros((Nprtcl),dtype = int)
sly = np.zeros((Nprtcl),dtype = int)
caus_count = np.zeros((Nprtcl))

term1 = np.zeros((Nx,Ny))
term2 = np.zeros((Nx,Ny))
term3 = np.zeros((Nx,Ny))
term4 = np.zeros((Nx,Ny))

## --------------------------------------------------------------

@jax.jit
def e2d_to_1d(x,k = k ,shells = shells):
    return (np.histogram(k.ravel(),bins = shells,weights=x.ravel() )[0]).real



def field_save(i,x_old):

    psi= -lapinv*x_old
    e_arr = e2d_to_1d(0.5*(x_old*np.conjugate(psi))*normalize)

    savePath = curr_path/f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/time_{t[i]:.2f}/"
    savePath.mkdir(parents=True, exist_ok=True)
    np.savez(savePath/f"w.npz", vorticity =  x_old)
    np.savez(savePath/f"e_arr.npz", energy =  e_arr)

def particle_save(i, xprtcl,Z,caus_count):


    alpha_name = "tracer" if alph == 2/3 else f"alpha_{alph:.2}_prtcl"
    savePathprtcl = curr_path/f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/{alpha_name}/St_{st/tf:.2f}/time_{t[i]:.2f}"
    savePathprtcl.mkdir(parents=True, exist_ok=True)
    np.savez(savePathprtcl/f"pos.npz",pos =  xprtcl[:,:d])
    np.savez(savePathprtcl/f"vel.npz",vel =  xprtcl[:,d:2*d])
    np.savez(savePathprtcl/f"prtcl_Z.npz", Zmatrix=  Z)
    np.savez(savePathprtcl/f"caus_count.npz", caustics_count =  caus_count)
## --------------------------------------------------------------


f_r = -kf*f0*np.sin(kf*y)
f = fft2(f_r)
@jax.jit
def forcing(xi,psi,f = f):
    """
    Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
    """

    # e_arr[:] = e2d_to_1d(0.5*(xi*np.conjugate(psi))*normalize) #!
    # if np.sum(e_arr)>100: raise ValueError("Blowup")
    # # print("inside",e_arr[shell1])
    # """Change forcing starts here"""
    # ## Const Power Input
    # factor[:] = np.where(ek_arr0 > 0, P/2.,0)*np.where(e_arr<1e-10,1,1/e_arr)
    # factor2d[:] = factor[kint]
    # # Constant shell energy
    # # factor[:] = 0.

    # # factor[shell1] = 1/dt*((ek_arr0[shell1]/max(1e-5,e_arr[shell1]))**0.5-1) #! The factors for each shell is calculated
    # # factor2d[:] = factor[kint]

    # return  factor2d*dealias*xi
    return  f
    # return  factor2d*dealias/normalize
@jax.jit
def index(arg,shift,Nx = Nx):
    return ((arg + shift )%Nx).astype(np.int32)




## ---------------- pre req for b-spline interps ----------------- ##
def initialize_b_spline(order):
    """
    Initializes the Matrix M and coefficient array ck in 2D according to hinsberg et al.
    ck is not computed using the formula. Rather doing 1/F(b_d) where F is the fourier transform of the b_d function given in (7.6).
    Mmat is compute using (7.4)
    """
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
## -------------------------------------------------------------- ##
nums = np.arange(order)
ck2d,Mmat = initialize_b_spline(order)

@jax.jit
def calc_phys_fields(psi,u_field,kk,kvec = kvec, epsilon = epsilon,ck2d = ck2d,lapinv = lapinv):
    """
    Calculates the physical fields from the streamfunction field.
    However the physical space fields are computed in b-spline basis making them apt for interpolation.

    #? Seems like the non-linear terms are more accurate without dealiasing.
    """
    A_field = -ifft2(np.einsum('ik,...k,...j->...ij',epsilon,kvec,kvec)*psi[...,None,None])

    ugrdu_field = np.einsum('...ij,...j->...i',A_field,u_field)
    DuDtk = -1j* (lapinv*kk*ck2d)[...,None]*np.einsum('ij,...j->...i',epsilon,kvec ) + fft2(ugrdu_field)*ck2d[...,None]

    grdDuDtk = 1j*np.einsum('...i,...j->...ij',DuDtk, kvec)


    DuDt_field = ifft2(DuDtk)
    grdDuDt_field = ifft2(grdDuDtk)

    A_field = -ifft2(np.einsum('ik,...k,...j->...ij',epsilon,kvec,kvec)*(psi*ck2d)[...,None,None])

    u_field = ifft2(1j*np.einsum('ik,...k->...i',epsilon, kvec)*(psi*ck2d)[...,None])

    return u_field,A_field, DuDt_field,grdDuDt_field
## -------------- interpolation of any field ---------------------


@jax.jit
def interp_spline(pos,u_field,A_field,DuDt_field,grdDuDt_field,Mmat = Mmat,order = order, N = N,nums = nums,dx= dx,umat = umat, Amat = Amat, DuDt_mat = DuDt_mat, grdDuDt_mat = grdDuDt_mat):
    """
    b-spline interpolation of the specified order given in Hinsberg et al.
    """
    umat = umat*0.0
    Amat = Amat*0.0
    DuDt_mat = DuDt_mat*0.0
    grdDuDt_mat = grdDuDt_mat*0.0

    idx = (pos//dx).astype(np.int32)
    delx = (pos%dx)/dx
    poly = delx[...,None]**nums

    for i in range(order):
        for j in range(order):
            temparr = np.einsum('p,q,...p,...q->...',Mmat[i],Mmat[j],poly[...,0,:],poly[...,1,:]) #! For saving computations
            slx = (idx[:,0]-order/2 +1 + i)%N #! For saving computations
            sly = (idx[:,1]-order/2 +1 + j)%N #! For saving computations


            umat = umat + u_field[slx,sly,...]*temparr[...,None]

            Amat = Amat + A_field[slx,sly,...]*temparr[...,None,None]

            DuDt_mat = DuDt_mat + DuDt_field[slx,sly]*temparr[...,None]

            grdDuDt_mat = grdDuDt_mat + grdDuDt_field[slx,sly]*temparr[...,None,None]

    return umat,Amat,DuDt_mat,grdDuDt_mat

@jax.jit
def RHSp(t,pos,umat,DuDt_mat,yprtcl = yprtcl,alph = alph,tp = tp):
    yprtcl = yprtcl.at[:,:d].set(pos[:,d:2*d])
    yprtcl = yprtcl.at[:,d:2*d].set(alph*(umat - pos[:,d:2*d])/tp + 3*(1-alph)*DuDt_mat)
    return yprtcl

@jax.jit
def RHSZ(t,Z,A,grdDuDt_mat,alph = alph,tp = tp):
    return alph*(A - Z)/tp - Z@Z + 3*(1 - alph)*grdDuDt_mat
## ------------------- RHS w/o viscosity --------------------
"""psi = str fn ; xi = vorticity"""
@jax.jit
def adv(t,xi,i,xivis = xivis,dealias = dealias,lapinv = lapinv,kvec = kvec, epsilon = epsilon):
    psi = -lapinv*xi*dealias

    u_field = ifft2(1j*np.einsum('ik,...k->...i',epsilon, kvec)*psi[...,None])
    xi_r = ifft2(1j*kvec*xi[...,None])
    advterm = np.einsum('...i,...i->...',u_field,xi_r)
    return -1.0*dealias * (fft2(advterm)) + forcing(xi,psi) - xivis*xi*dealias, psi, u_field

@jax.jit
def rel_err(x,y):
    """ Calculates the relative error in velocity. y is the actual velocity """
    return np.linalg.norm(x-y)/np.linalg.norm(y)



## ----------------------------------------------------------

@jax.jit
def RK4(h,ti, x_old,i,xprtcl, Z,caus_count,Zthresh =Zthresh,tp = tp):

    k1,psi,u_field = adv(ti,x_old,i)

    u_field,A_field,DuDt_field,grdDuDt_field = calc_phys_fields(psi,u_field,k1)
    umat,Amat,DuDt_mat,grdDuDt_mat = interp_spline(xprtcl[:,:d],u_field,A_field,DuDt_field,grdDuDt_field)
    
    # count += 1
    # err += rel_err(Z,Amat)
    # print(f"Error in k1 vel == umat:" ,rel_err(xprtcl[:,d:2*d],umat))
    # eiga[i] = eigval(Amat.astype(complex),eiga[i])
    # eigz[i] = eigval(np.nan_to_num(Z[i], posinf = 0.0, neginf=0.0).astype(complex) ,eigz[i])

    k1p = RHSp(t[i],xprtcl,umat,DuDt_mat)
    k1Z = RHSZ(t[i],Z,Amat,grdDuDt_mat) #! Making the fluid gradient quantities non-dimensional in simulation time.
    
    # print(f"Max TrK1Z : {np.max(np.abs(k1Z[:,0,0] + k1Z[:,1,1] ))}")

    xtemp = (xprtcl+h*k1p/2)
    xtemp = xtemp.at[:,:d].set(xtemp[:,:d]%Lx)

    k2,psi,u_field = adv(ti + h/2, x_old + h/2*k1,i)
    
    u_field,A_field,DuDt_field,grdDuDt_field = calc_phys_fields(psi,u_field,k2)
    umat,Amat,DuDt_mat,grdDuDt_mat = interp_spline(xtemp[:,:d],u_field,A_field,DuDt_field,grdDuDt_field)
    
    # count += 1
    # err += rel_err(Z+ h/2.*k1Z,Amat)
    # print(f"Error in k2 vel == umat:" ,rel_err(xtemp[:,d:2*d],umat))
    # print(f"Amat at k2 == Z + h/2*k1Z : {np.allclose(Amat, Z + h/2*k1Z)}")
    
    k2p = RHSp(ti+h/2,xtemp,umat,DuDt_mat)
    k2Z = RHSZ(ti+ h/2.,Z+ h/2.*k1Z,Amat,grdDuDt_mat) #! Making the fluid gradient quantities non-dimensional in simulation time.
    
    # # print(f"Max TrK2Z : {np.max(np.abs(k2Z[:,0,0] + k2Z[:,1,1] ))}")

    xtemp = (xprtcl+h*k2p/2)
    xtemp = xtemp.at[:,:d].set(xtemp[:,:d]%Lx)

    k3,psi,u_field = adv(ti + h/2, x_old + h/2*k2,i)
    
    u_field,A_field,DuDt_field,grdDuDt_field = calc_phys_fields(psi,u_field,k3)
    umat,Amat,DuDt_mat,grdDuDt_mat = interp_spline(xtemp[:,:d],u_field,A_field,DuDt_field,grdDuDt_field)
    
    # # count += 1
    # # err += rel_err(Z+ h/2.*k2Z,Amat)
    # # print(f"Error in k3 vel == umat:" ,rel_err(xtemp[:,d:2*d],umat))
    
    k3p = RHSp(ti+h/2,xtemp,umat,DuDt_mat)
    k3Z = RHSZ(ti+ h/2.,Z+ h/2.*k2Z,Amat,grdDuDt_mat) #! Making the fluid gradient quantities non-dimensional in simulation time.
    
    # # print(f"Max TrK3Z : {np.max(np.abs(k3Z[:,0,0] + k3Z[:,1,1] ))}")

    xtemp = (xprtcl+h*k3p)
    xtemp = xtemp.at[:,:d].set(xtemp[:,:d]%Lx)

    k4,psi,u_field = adv(ti + h, x_old + h*k3,i)
    
    u_field,A_field,DuDt_field,grdDuDt_field = calc_phys_fields(psi,u_field,k4)
    umat,Amat,DuDt_mat,grdDuDt_mat = interp_spline(xtemp[:,:d],u_field,A_field,DuDt_field,grdDuDt_field)
    
    # # count += 1
    # # err += rel_err(Z+ h*k3Z,Amat)
    # # print(f"Error in k4 vel == umat:" ,rel_err(xtemp[:,d:2*d],umat))
    
    k4p = RHSp(t[i]+h,xtemp,umat,DuDt_mat)
    k4Z = RHSZ(t[i]+ h,Z+ h*k3Z,Amat,grdDuDt_mat) #! Making the fluid gradient quantities non-dimensional in simulation time.
    
    # print(f"Max TrK4Z : {np.max(np.abs(k4Z[:,0,0] + k4Z[:,1,1] ))}")

    x_new = (x_old + h/6.0*(k1 + 2*k2 + 2*k3 + k4))#/(1.0 + h*xivis)
    xprtcl_new = xprtcl + h*(k1p + 2*k2p + 2*k3p + k4p)/6
    xprtcl_new = xprtcl_new.at[:,:d].set(xprtcl_new[:,:d]%Lx)
    Z_new = Z + h/6.*(k1Z+ 2*k2Z + 2*k3Z + k4Z)

    caus_count = caus_count + np.where( np.einsum('...ii->...',Z_new)< Zthresh*tp,1,0)
    Z_new = Z_new*np.where( np.einsum('...ii->...',Z_new)< Zthresh*tp,-1,1)[...,None,None]

    return x_new, xprtcl_new, Z_new,caus_count
    # return x_new






## -------------The RK4 integration function -----------------
def evolve_and_save(f,t,x0,xprtcl, Z,caus_count = caus_count):
    # count = 0
    # err = 0
    tstart = time()
    # print()
    h = t[1] - t[0]
    x_old = x0
    etot = 1.0*np.zeros_like(t)

    for i,ti in tqdm(enumerate(t[:-1])):
        # print(np.round(t[i],2),end= '\r')
        #! Things to do every 1 second (dt = 0.1)
        if i%savestep ==0: field_save(i,x_old)

        if i%prtcl_savestep == 0: particle_save(i,xprtcl,Z,caus_count)

        # x_new = RK4(h,ti, x_old,i,xprtcl, Z,caus_count)
        x_new, xprtcl_new, Z_new,caus_count = RK4(h,ti, x_old,i,xprtcl, Z,caus_count)

        x_old = x_new
        xprtcl = xprtcl_new
        Z = Z_new


    ## Saving the last time evolution
    field_save(i+1,x_old)
    particle_save(i+1,xprtcl,Z,caus_count)
## ---------------------------------------------------------

## ---------------- Initial conditions ----------------
if restart == False:
    # """Starts on its own"""
    
    # r = jax.random.RandomState(key,309)
    # thinit = .uniform(0,2*np.pi,np.shape(k))
    # eprofile = np.exp(-np.arange(Nf))
    # eprofile[shell_count] = 1e-8
    # eprofile[:] = eprofile/np.histogram(k.ravel(),bins = shells)[0]

    # psi0  = (eprofile[kint])**0.5*np.exp(1j*thinit)*(kint>0)*(kint<kinit)/np.where(kint>0, kint**1,1)
    # xi0 = -lap*psi0
    # e_arr = e2d_to_1d(0.5*(xi0*np.conjugate(psi0))*normalize)
    # ek_arr0= 0*e_arr.copy()
    # ek_arr0[shell_count] = e_arr[shell_count]
    # e0 = np.sum(e_arr)
    # xi0[:] = xi0*(einit/e0)**0.5
    # psi0[:] = psi0*(einit/e0)**0.5
    
    xi0_r = -f0*kf*nu*(np.cos(kf*x) + np.cos(kf*y))
    xi0 = fft.rfft2(xi0_r)
    psi0 = -lapinv*xi0
    
else:
    """Starts from the last saved data"""
    loadPath = curr_path/f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/last/"
    if loadPath.exists() == False:
        dt_load = 0.005
        loadPath = curr_path/f"data/Re_{np.round(2e5,2)},dt_{dt_load},N_{Nx}/last/"
        xi0 = np.load(loadPath/f"w.npy")
    else: xi0 = np.load(loadPath/f"w.npz")["vorticity"]
    
    psi0 = -lapinv*xi0

u_field = ifft2(1j*np.einsum('ik,...k->...i',epsilon, kvec)*psi0[...,None])
e_arr = e2d_to_1d(0.5*(xi0*np.conjugate(psi0))*normalize)
e0 = np.sum(e_arr)
ek_arr0= 0*e_arr.copy()
ek_arr0 = ek_arr0.at[shell_count].set(e_arr[shell_count])

print(f"Initial energy : {np.sum(e_arr)}")
print(f"max dt allowed :{dx/np.max(np.abs(u_field))}")

xprtcl = xprtcl.at[:,:d].set(jax.random.uniform(key, shape=(Nprtcl, d), minval=0, maxval=Lx))
u_field,A_field,DuDt_field,grdDuDt_field = calc_phys_fields(psi0,u_field,0.0*k1) #! Initialized for interpolation
umat,Amat,DuDt_mat,grdDuDt_mat = interp_spline(xprtcl[:,:d],u_field,A_field,DuDt_field,grdDuDt_field)
xprtcl = xprtcl.at[:,d:2*d].set(umat.copy())
Z = Z.at[:].set(Amat.copy())
Amat_Start = Amat.copy()
A_field_start = A_field.copy()

## ----------------------------------------------------

RK4(dt,0, xi0,0,xprtcl, Z,caus_count)
jax.effects_barrier() #! First compilation 
# #! Testing the RK4
# for i in tqdm(range(100)):
#     # adv(0,xi0,0)
#     # jax.effects_barrier()
#     # calc_phys_fields(psi0,u_field,0.0*k1)
#     # jax.effects_barrier()
#     # interp_spline(xprtcl[:,:d],u_field,A_field,DuDt_field,grdDuDt_field)
#     # jax.effects_barrier()
#     # RHSp(0,xprtcl,umat,DuDt_mat)
#     # jax.effects_barrier()
#     # RHSZ(0,Z,Amat,grdDuDt_mat)
#     # jax.effects_barrier()
#     RK4(dt,0, xi0,0,xprtcl, Z,caus_count)
#     jax.effects_barrier()


t1 = time()
print(f"Initial zeta:{np.max(np.abs(xi0))}")
etot = evolve_and_save(adv,t,xi0,xprtcl,Z)
t2 = time()
print(f"Time taken to evolve for {T} secs in {dt} sec timesteps with gridsize {Nx}x{Nx} is \n {t2-t1} sec")






