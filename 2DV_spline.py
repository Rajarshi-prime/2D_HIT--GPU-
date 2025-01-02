import cupy as np
import numpy as npp
import math
import cupy.fft as fft
from time import time
# import h5py
import pathlib,json
curr_path = pathlib.Path(__file__).parent
# fft.config.use_multi_gpus = True
## --------------- Details of the code ---------------
    # This program solves the 2D vorticity equation without forcing using RK4 and FFT. 
    # The 2D vorticity equation is given by d_t \xi  + u.\grad \xi= \nu \laplacian \xi.
    # Defining stream funciton and taking fourier transpose we get 
    # The dissipation term is (\laplace)^8. 
    # In the fourier space, the equation is
    # d_t \psi_k = \int_k' (k-k')^2/k^2 [\hat{z} . {(k-k') x k'}] \psi_k' \psi_{k-k'}  - \nu k^2 \psi_k.

    # We have a finite number of grid points i.e. we have finite number of ks. 
    # We initiate a finite number of wavenumber. Then evolve and fourier the -k^2 \psi_k at every instant. 


    # Ignoring the k = 0 mode as this mode of psi does not contribute in the evolution equation of xi

    # Variables with r denotes quantities in position space.
    # Underscore denotes derivative.
## ----------------------------------------------------

## ---------------- params ----------------------- ## 
paramfile = 'parameters.json'
with open(paramfile,'r') as jsonFile: params = json.load(jsonFile)

d = params["d"] # Dimension
nu =params["nu"] # Viscosity
Re = 1/nu if nu > 0 else np.inf # Reynolds number
N = Nx = Ny = params["N"] # Grid size
dt = params["dt"] # Timestep
# T = params["T"] # Final time
T = 1.0 # Final time
einit = params["einit"] # Initial energy
kinit = params["kinit"] # Initial energy is distributed among these wavenumbers
linnu = params["linnu"] # Linear drag co-efficient
lp = params["lp"] # Power of the laplacian in the hyperviscous term
shell_count = params["shell_count"] # The shells where the energy is injected
alph = params["alph"] # The density ratio
Nprtcl = int(params["Nprtcl"]*Nx*Ny) # Number of particles
tf = params["tf"] # Final time for the particles
st = params["st"]*tf # Time period for the particles #! Stokes number when non-dimensionalize by simulation time.
tp = st # Save the particle data after this many timesteps #!(in simulation time dimension, this is the tp)
savestep = int(params["savestep"]/dt) # Save the data after this many timesteps
save_spectra = int(10//dt) # Save the spectra after this many timesteps
prtcl_savestep = int(params["prtcl_savestep"]/dt) # Save the particle data after this many timesteps
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
ifft2 = lambda x: fft.irfft2(x,(Nx,Ny))

## Forming the 2D grid (k space)
Kx = 2*np.pi*np.linspace(-(Nx//2) , Nx//2 - 0.5*(1+ (-1)**(Nx%2)),Nx)/Lx
Ky = 2*np.pi*np.linspace(-(Ny//2) , Ny//2 - 0.5*(1+ (-1)**(Ny%2)),Ny)/Ly
Kx = np.append(Kx[Kx>=0], Kx[Kx<0])
Ky = np.append(Ky[Ky>=0], -Ky[0])
kx,ky = np.meshgrid(Kx,Ky,indexing="ij")
# kvec = np.array([kx,ky])



dalcutoff = ((2*np.pi*Nx)/Lx)//3,((2*np.pi*Ny)/Ly)//3
dealias = (abs(kx)<dalcutoff[0])*(abs(ky)<dalcutoff[1])  
## Defining the inverese laplacian.
lap = -(kx**2 + ky**2)
# lap1 = lap.copy()
# lap1[lap1== 0] = np.inf
lapinv = dealias/np.where(lap == 0., np.inf, lap)

k = (kx**2 + ky**2)**0.5
kint = np.clip(np.round(k),None,Nf).astype(int)
xivis =  nu *((-lap)**lp) + linnu ## The viscous term 
shells = np.arange(-0.5,Nf)
shells[0] = 0. 
normalize = np.where((ky== 0) + (ky == Ny//2) , 1/(Nx**4/TWO_PI**2),2/(Nx**4/TWO_PI**2))
## ----------------------------------------------------


## -----------------  Parameters  ----------------------
print(f"Runnig for alpha = {alph}")
## ----------------------------------------------


## -------------- Initializing the empty arrays -----------------
psi = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
xi = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
x_old = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
x_new = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k1 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k2 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k3 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k4 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
tk1 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
tk2 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)


u_field = np.zeros((Nx,Ny,2))
ugrdu_field = np.zeros((Nx,Ny,2))
deludelt = np.zeros((Nx,Ny,2))
A_field = np.zeros((Nx,Ny,2,2))
delAdelt = np.zeros((Nx,Ny,2,2))
ugrdA_field = np.zeros((Nx,Ny,2,2))
xi_xr = np.empty((Nx,Ny),dtype = np.float64)
xi_yr = np.empty((Nx,Ny),dtype = np.float64)
advterm = np.empty((Nx,Ny),dtype = np.float64)
factor = np.empty(Nf)
factor2d = np.empty((Nx,Nf))

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

# xindex = np.zeros((Nprtcl))
# yindex = np.zeros((Nprtcl))
# xrem = np.zeros((Nprtcl))
# yrem = np.zeros((Nprtcl))
idx  = np.zeros((Nprtcl,d),dtype = int)
delx = np.zeros((Nprtcl,d))
poly = np.zeros((Nprtcl,d,order))
umat = np.zeros((Nprtcl,d))
deludeltmat = np.zeros((Nprtcl,d))
ugrdumat = np.zeros((Nprtcl,d))
DADtmat = np.zeros((Nprtcl,d,d))
Amat = np.zeros((Nprtcl,d,d))

temparr = np.zeros((Nprtcl))
slx = np.zeros((Nprtcl),dtype = int)
sly = np.zeros((Nprtcl),dtype = int)
caus_count = np.zeros((Nprtcl))

term1 = np.zeros((Nx,Ny))
term2 = np.zeros((Nx,Ny))
term3 = np.zeros((Nx,Ny))
term4 = np.zeros((Nx,Ny))

## --------------------------------------------------------------
def e2d_to_1d(x):
    return (np.histogram(k.ravel(),bins = shells,weights=x.ravel() )[0]).real

def field_save(i,x_old):
    
    psi[:] = -lapinv*x_old
    if i%save_spectra == 0:
        print(f"saving e_array")
        e_arr[:] = e2d_to_1d(0.5*(x_old*np.conjugate(psi))*normalize)
    print(f"Energy at time {np.round(t[i],2)} is {np.sum(e_arr):.4f}",end= '\r')
    # savePath = curr_path/f"data_nodadt_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/time_{t[i]:.2f}/"
    savePath = curr_path/f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/time_{t[i]:.2f}/"
    savePath.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(savePath/f"w.npz", vorticity =  x_old)
    np.savez_compressed(savePath/f"e_arr.npz", energy =  e_arr)
    
def particle_save(i, xprtcl,Z,caus_count):
    
    print(f"Saving at time {t[i]} with Max TrZ : {np.max(np.abs(Z[:,0,0] + Z[:,1,1] ))}")
    # savePathprtcl = curr_path/f"data_nodadt_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/{alpha_name}/St_{st/tf:.2f}/time_{t[i]:.2f}"
    alpha_name = "tracer" if alph == 2/3 else f"alpha_{alph:.2}_prtcl"
    savePathprtcl = curr_path/f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/{alpha_name}/St_{st/tf:.2f}/time_{t[i]:.2f}"
    savePathprtcl.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(savePathprtcl/f"pos.npz",pos =  xprtcl[:,:d])
    np.savez_compressed(savePathprtcl/f"vel.npz",vel =  xprtcl[:,d:2*d])
    np.savez_compressed(savePathprtcl/f"prtcl_Z.npz", Zmatrix=  Z)
    np.savez_compressed(savePathprtcl/f"caus_count.npz", caustics_count =  caus_count)
## --------------------------------------------------------------


f_r = -kf*f0*np.sin(kf*y)
f = fft.rfft2(f_r)
def forcing(xi,psi):
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

index = lambda arg,shift: ((arg + shift )%Nx).astype(int) # Calculating the index


def calc_phys_fields(psi,u_field,kk):
    """
    Calculates the physical fields from the streamfunction field.
    However the physical space fields are computed in b-spline basis making them apt for interpolation.
    """
    
    A_field[...,0,0] = -ifft2(kx* ky*psi*ck2d)
    A_field[...,0,1] = -ifft2(ky* ky*psi*ck2d)
    A_field[...,1,0] = ifft2(kx* kx*psi*ck2d)
    A_field[...,1,1] = ifft2(ky* kx*psi*ck2d)
    
    ugrdu_field[:] = u_field[...,None,0]*A_field[...,:,0] + u_field[...,None,1]*A_field[...,:,1]

    term1[:] = ifft2(1j*kx**2* ky*psi*ck2d)
    term2[:] = ifft2(1j*ky**2* kx*psi*ck2d)
    term3[:] = ifft2(1j*kx**3*psi*ck2d)
    term4[:] = ifft2(1j*ky**3*psi*ck2d)
    
    ugrdA_field[...,0,0] = ifft2(fft.rfft2(-term1*u_field[...,0 ] - term2*u_field[...,1 ])*dealias) 
    ugrdA_field[...,0,1] = ifft2(fft.rfft2(-term2*u_field[...,0 ] - term4*u_field[...,1 ])*dealias) 
    ugrdA_field[...,1,0] = ifft2(fft.rfft2(term3*u_field[...,0 ] +  term1*u_field[...,1 ])*dealias) 
    ugrdA_field[...,1,1] = ifft2(fft.rfft2(term1*u_field[...,0 ] +  term2*u_field[...,1 ])*dealias) 
    
    
    tk1[:] = -1j*lapinv*ky*(kk - xivis*(x_old ))
    tk2[:] = 1j*lapinv*kx*(kk - xivis*(x_old ))
    deludelt[...,0] = ifft2(tk1*ck2d)
    deludelt[...,1] = ifft2(tk2*ck2d)
        
    delAdelt[...,0,0] = ifft2(1j*kx*tk1*ck2d)
    delAdelt[...,0,1] = ifft2(1j*ky*tk1*ck2d)
    delAdelt[...,1,0] = ifft2(1j*kx*tk2*ck2d)
    delAdelt[...,1,1] = ifft2(1j*ky*tk2*ck2d)
    
    u_field[...,0] = ifft2(1j * ky*psi*ck2d)
    u_field[...,1] = ifft2(-1j * kx*psi*ck2d) 
    
    return u_field,ugrdu_field,deludelt,A_field,delAdelt,ugrdA_field
## -------------- interpolation of any field ---------------------

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

def interp_spline(pos,u_field,A_field,deludelt,DADt,ugrdu_field):
    """
    b-spline interpolation of the specified order given in Hinsberg et al.
    """
    global umat,Amat,deludeltmat,DADtmat,ugrdumat
    umat[:] = 0.0
    Amat[:] = 0.0
    deludeltmat[:] = 0.0
    DADtmat[:] = 0.0
    ugrdumat[:] = 0.0
    
    idx[:] = (pos//dx).astype(int)
    delx[:] = (pos%dx)/dx    
    poly[:] = delx[...,None]**nums

    for i in range(order):
        for j in range(order):
            temparr[:] = np.einsum('p,q,...p,...q->...',Mmat[i],Mmat[j],poly[...,0,:],poly[...,1,:]) #! For saving computations
            slx[:] = (idx[:,0]-1 + i)%N #! For saving computations
            sly[:] = (idx[:,1]-1 + j)%N #! For saving computations
            
            umat  += u_field[slx,sly,...]*temparr[:,None]
            
            ugrdumat  += ugrdu_field[slx,sly]*temparr[:,None]
            
            deludeltmat  += deludelt[slx,sly]*temparr[:,None]
            
            Amat  += A_field[slx,sly]*temparr[:,None,None]
            
            DADtmat  +=  DADt[slx,sly]*temparr[:,None,None]
            
    return umat,Amat,deludeltmat,DADtmat,ugrdumat

def RHSp(t,pos,umat,DuDtmat):
    yprtcl[:,:d] = pos[:,d:2*d]
    yprtcl[:,d:2*d] = (alph*(umat - pos[:,d:2*d])/st + 3*(1-alph)*DuDtmat)
    return yprtcl
    


def RHSZ(t,Z,A,ugrdA):
    return (-alph*(Z - A) - Z@Z + 3*(1 - alph)*A@A)/st + 3*(1 - alph)*ugrdA        
# def RHSZ(t,Z,A,DADt): #! DADt turned off
#     return (-alph*(Z - A) - Z@Z + 3*(1 - alph)*A@A)/st #+ 3*(1 - alph)*DADt
## ------------------- RHS w/o viscosity --------------------
"""psi = str fn ; xi = vorticity"""
def adv(t,xi,i):
    global psi,dealias
    psi[:] = -lapinv*xi*dealias
    enet = np.sum(0.5*(xi*np.conjugate(psi))*normalize)
    # e_arr[:] = e2d_to_1d(0.5*(xi*np.conjugate(psi))*normalize) #!
    if enet>100: raise ValueError("Blowup")
    u_field[...,0] = ifft2(1j * ky*psi)
    u_field[...,1] = ifft2(-1j * kx*psi) 
    xi_xr[:] = ifft2(1j * kx*xi)
    xi_yr[:] = ifft2(1j * ky*xi)
    advterm[:] = u_field[...,0]*xi_xr + u_field[...,1]*xi_yr
    return -1.0*dealias * (fft.rfft2(advterm)) + forcing(xi,psi)

# def conv(t,x,q):
#     return dq

## ----------------------------------------------------------
    
print(np.max(np.abs(xivis)))

## -------------The RK4 integration function -----------------
def evolve_and_save(f,t,x0):
    # print()
    h = t[1] - t[0]
    x_old[:] = x0
    etot = 1.0*np.zeros(len(t)//20) +1
    
    for i,ti in enumerate(t[:-1]):
        # print(np.round(t[i],2),end= '\r')
        #! Things to do every 1 second (dt = 0.1)
        if i%savestep ==0: field_save(i,x_old)
                    
        if i%prtcl_savestep == 0: particle_save(i,xprtcl,Z,caus_count)
            
            

        
        k1[:] = adv(ti,x_old,i)
        u_field[:],ugrdu_field[:],deludelt[:],A_field[:],delAdelt[:],ugrdA_field[:] = calc_phys_fields(psi,u_field,k1)
                
        umat[:],Amat[:],deludeltmat[:],DADtmat[:],ugrdumat[:] = interp_spline(xprtcl[:,:d],u_field,A_field,deludelt,delAdelt+ugrdA_field,ugrdu_field)    
        
        # eiga[i] = eigval(Amat.astype(complex),eiga[i])
        # eigz[i] = eigval(np.nan_to_num(Z[i], posinf = 0.0, neginf=0.0).astype(complex) ,eigz[i])
        
        k1p[:] = RHSp(t[i],xprtcl,umat,deludeltmat + ugrdumat)
        k1Z[:] = RHSZ(t[i],Z,Amat*tp,DADtmat*tp) #! Making the fluid gradient quantities non-dimensional in simulation time.
        
        xtemp[:] = (xprtcl+h*k1p/2)
        xtemp[:,:d] = xtemp[:,:d]%Lx
        
        k2[:] = adv(ti + h/2, x_old + h/2*k1,i)
        u_field[:],ugrdu_field[:],deludelt[:],A_field[:],delAdelt[:],ugrdA_field[:] = calc_phys_fields(psi,u_field,k2)
        
        umat[:],Amat[:],deludeltmat[:],DADtmat[:],ugrdumat[:] = interp_spline(xtemp[:,:d],u_field,A_field,deludelt,delAdelt+ugrdA_field,ugrdu_field) 
        
        k2p[:] = RHSp(t[i]+h/2,xtemp,umat,deludeltmat + ugrdumat)
        k2Z[:] = RHSZ(t[i]+ h/2.,Z+ h/2.*k1Z,Amat*tp,DADtmat*tp) #! Making the fluid gradient quantities non-dimensional in simulation time.

        xtemp[:] = (xprtcl+h*k2p/2)
        xtemp[:,:d] = xtemp[:,:d]%Lx
        
        k3[:] = adv(ti + h/2, x_old + h/2*k2,i)
        u_field[:],ugrdu_field[:],deludelt[:],A_field[:],delAdelt[:],ugrdA_field[:] = calc_phys_fields(psi,u_field,k3)
        
        umat[:],Amat[:],deludeltmat[:],DADtmat[:],ugrdumat[:] = interp_spline(xtemp[:,:d],u_field,A_field,deludelt,delAdelt+ugrdA_field,ugrdu_field) 

        k3p[:] = RHSp(t[i]+h/2,xtemp,umat,deludeltmat + ugrdumat)
        k3Z[:] = RHSZ(t[i]+ h/2.,Z+ h/2.*k2Z,Amat*tp,DADtmat*tp) #! Making the fluid gradient quantities non-dimensional in simulation time.

        xtemp[:] = (xprtcl+h*k3p)
        xtemp[:,:d] = xtemp[:,:d]%Lx
        
        k4[:] = adv(ti + h, x_old + h*k3,i)
        u_field[:],ugrdu_field[:],deludelt[:],A_field[:],delAdelt[:],ugrdA_field[:] = calc_phys_fields(psi,u_field,k4)
                
        umat[:],Amat[:],deludeltmat[:],DADtmat[:],ugrdumat[:] = interp_spline(xtemp[:,:d],u_field,A_field,deludelt,delAdelt+ugrdA_field,ugrdu_field) 

        k4p[:] = RHSp(t[i]+h,xtemp,umat,deludeltmat + ugrdumat)
        k4Z[:] = RHSZ(t[i]+ h,Z+ h*k3Z,Amat*tp,DADtmat*tp) #! Making the fluid gradient quantities non-dimensional in simulation time.

        x_new[:] = (x_old + h/6.0*(k1 + 2*k2 + 2*k3 + k4))/(1.0 + h*xivis)
        xprtcl_new[:] = xprtcl + h*(k1p + 2*k2p + 2*k3p + k4p)/6    
        xprtcl_new[:,:d] = xprtcl_new[:,:d]%Lx
        Z_new[:] = Z + h/6.*(k1Z+ 2*k2Z + 2*k3Z + k4Z)
        cond = np.argwhere(np.einsum('...ii->...',Z_new)< Zthresh)
        caus_count[cond] += 1
        Z_new[cond] = -1*Z_new[cond]

        
        
        
            
            
            
            
        x_old[:] = x_new
        psi[:] = -lapinv*x_old
        xprtcl[:] = xprtcl_new
        Z[:] = Z_new
        
        
    ## Saving the last time evolution
    field_save(i+1,x_old)
    particle_save(i+1,xprtcl,Z,caus_count)
## ---------------------------------------------------------   


    









## ---------------- Initial conditions ----------------
if restart == False:
    """Starts on its own"""
    r = np.random.RandomState(309)
    thinit = r.uniform(0,2*np.pi,np.shape(k))
    eprofile = np.exp(-np.arange(Nf))
    eprofile[shell_count] = 1e-8
    eprofile[:] = eprofile/np.histogram(k.ravel(),bins = shells)[0]

    psi0  = (eprofile[kint])**0.5*np.exp(1j*thinit)*(kint>0)*(kint<kinit)/np.where(kint>0, kint**1,1)
    xi0 = -lap*psi0
    e_arr = e2d_to_1d(0.5*(xi0*np.conjugate(psi0))*normalize)
    ek_arr0= 0*e_arr.copy()
    ek_arr0[shell_count] = e_arr[shell_count]
    e0 = np.sum(e_arr)

    xi0[:] = xi0*(einit/e0)**0.5
    psi0[:] = psi0*(einit/e0)**0.5
else:
    """Starts from the last saved data"""
    loadPath = curr_path/f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/last/"
    if loadPath.exists() == False: 
        loadPath = curr_path/f"data_spline/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/last/"
        xi0 = np.load(loadPath/f"w.npy")
    xi0 = np.load(loadPath/f"w.npz")["vorticity"]
    psi0 = -lapinv*xi0

u_field[...,0] = ifft2(1j * ky*psi0)
u_field[...,1] = ifft2(-1j * kx*psi0) 
e_arr = e2d_to_1d(0.5*(xi0*np.conjugate(psi0))*normalize)
e0 = np.sum(e_arr)
ek_arr0= 0*e_arr.copy()
ek_arr0[shell_count] = e_arr[shell_count]
print(f"Initial energy : {np.sum(e_arr)}")
print(f"max dt allowed :{dx/np.max(np.abs(u_field))}")

xprtcl[:,:d] = np.random.uniform(0,Lx,(Nprtcl,d))
ck2d,Mmat = initialize_b_spline(order)
u_field[:],ugrdu_field[:],deludelt[:],A_field[:],delAdelt[:],ugrdA_field[:] = calc_phys_fields(psi,u_field,0.0*k1) #! Initialized for interpolation
umat[:],Amat[:],deludeltmat[:],DADtmat[:],ugrdumat[:] = interp_spline(xtemp[:,:d],u_field,A_field,deludelt,delAdelt+ugrdA_field,ugrdu_field) 
xprtcl[:,d:2*d] = umat
Z[:] = Amat*tp

## ----------------------------------------------------




t1 = time()
print(f"Initial zeta:{np.max(np.abs(xi0))}")
etot = evolve_and_save(adv,t,xi0)
t2 = time()
print(f"Time taken to evolve for {T} secs in {dt} sec timesteps with gridsize {Nx}x{Nx} is \n {t2-t1} sec")

# Saving a copy of the params along with the data
with open(curr_path/f"data_{iname}/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/parameters.json","w") as jsonFile: json.dump(params, jsonFile,indent = 2)

"""
Run with something like:
nohup time python -u 2DV.py > simul_256.out &
"""