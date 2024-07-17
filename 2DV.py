import cupy as np
import cupy.fft as fft
from time import time
# import h5py
import pathlib,json
curr_path = pathlib.Path(__file__).parent
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
T = params["T"] # Final time
einit = params["einit"] # Initial energy
kinit = params["kinit"] # Initial energy is distributed among these wavenumbers
linnu = params["linnu"] # Linear drag co-efficient
lp = params["lp"] # Power of the laplacian in the hyperviscous term
shell_count = params["shell_count"] # The shells where the energy is injected
alph = params["alph"] # The density ratio
Nprtcl = int(params["Nprtcl"]*Nx*Ny) # Number of particles
tp = params["tp"] # Time period for the particles
tf = params["tf"] # Final time for the particles
st = params["st"] # Save the particle data after this many timesteps
savestep = int(params["savestep"]/dt) # Save the data after this many timesteps
prtcl_savestep = int(params["prtcl_savestep"]/dt) # Save the particle data after this many timesteps
eta = params["eta"]/(Nx//3) # Desired Kolmogorov length scale
restart = params["restart"] # Restart from the last saved data
t = np.arange(0,1.1*dt + T,dt) # Time array
P = (nu**3/eta**4)/len(shell_count) # power per unit shell

## ----------------------------------------------- ##

## ------------ Grid and related operators ------------
## Forming the 2D grid (position space)
TWO_PI = 2* np.pi
Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
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

## Defining the inverese laplacian.
lap = -(kx**2 + ky**2)
# lap1 = lap.copy()
# lap1[lap1== 0] = np.inf
lapinv = 1.0/np.where(lap == 0., np.inf, lap)
Nf = Nx//2 + 1
shells = np.arange(-0.5,Nf)
shells[0] = 0. 
normalize = np.where((ky== 0) + (ky == Ny//2) , 1/(Nx**4/TWO_PI**2),2/(Nx**4/TWO_PI**2))
## ----------------------------------------------------


## -----------------  Parameters  ----------------------
k = (kx**2 + ky**2)**0.5
kint = np.clip(np.round(k),None,Nf).astype(int)
xivis =  nu *((-lap)**lp) + linnu ## The viscous term 

dalcutoff = ((2*np.pi*Nx)/Lx)//3,((2*np.pi*Ny)/Ly)//3
dealias = (abs(kx)<dalcutoff[0])*(abs(ky)<dalcutoff[1])  
print(f"Power input : {P*len(shell_count):6f}")
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


u_field = np.zeros((Nx,Ny,2))
DuDt = np.zeros((Nx,Ny,2))
A_field = np.zeros((Nx,Ny,2,2))
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

xindex = np.zeros((Nprtcl))
yindex = np.zeros((Nprtcl))
xrem = np.zeros((Nprtcl))
yrem = np.zeros((Nprtcl))
umat = np.zeros((Nprtcl,d))
DuDtmat = np.zeros((Nprtcl,d))
Amat = np.zeros((Nprtcl,d,d))

## --------------------------------------------------------------
def e2d_to_1d(x):
    return (np.histogram(k.ravel(),bins = shells,weights=x.ravel() )[0]).real
## --------------------------------------------------------------



def forcing(xi,psi):
    """
    Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
    """
    
    e_arr[:] = e2d_to_1d(0.5*(xi*np.conjugate(psi))*normalize) #!
    if np.sum(e_arr)>100: raise ValueError("Blowup")
    # print("inside",e_arr[shell1])
    """Change forcing starts here"""
    ## Const Power Input
    factor[:] = np.where(ek_arr0 > 0, P/2.,0)*np.where(e_arr<1e-10,1,1/e_arr)
    factor2d[:] = factor[kint]
    # Constant shell energy
    # factor[:] = 0.
     
    # factor[shell1] = 1/dt*((ek_arr0[shell1]/max(1e-5,e_arr[shell1]))**0.5-1) #! The factors for each shell is calculated
    # factor2d[:] = factor[kint]
    
    return  factor2d*dealias*xi
    # return  factor2d*dealias/normalize

index = lambda arg,shift: ((arg + shift )%Nx).astype(int) # Calculating the index

def calcA(psi,A_field):
    A_field[...,0,0] = -ifft2(kx* ky*psi)
    A_field[...,0,1] = -ifft2(ky* ky*psi)
    A_field[...,1,0] = ifft2(kx* kx*psi)
    A_field[...,1,1] = ifft2(ky* kx*psi)
    return A_field

## -------------- interpolation of any field ---------------------
def Linterp(pos,u_field,A_field,DuDt):
# pos = np.array([[1,2]])
    xindex[:] = (pos[:,0]//dx)%Nx
    yindex[:] = (pos[:,1]//dy)%Ny
    xrem[:] = pos[:,0]%dx/dx
    yrem[:] = pos[:,1]%dy/dy


    ## Linear interpolation
    umat[:] = (((1 - xrem) * (1 - yrem))[:,None] *u_field[index(xindex,0 ),index(yindex, 0),...] + ((1 - xrem) * (yrem) )[:,None]*u_field[index(xindex,0 ),index(yindex, 1),...] + ((xrem)*(1-yrem))[:,None]*u_field[index(xindex, 1 ),index(yindex, 0),...]  + ((xrem)*(yrem))[:,None]*u_field[index(xindex, 1 ),index(yindex, 1),...])
    
    Amat[:] = st*(((1 - xrem) * (1 - yrem))[:,None,None] *A_field[index(xindex,0 ),index(yindex, 0),...] + ((1 - xrem) * (yrem) )[:,None,None]*A_field[index(xindex,0 ),index(yindex, 1),...] + ((xrem)*(1-yrem))[:,None,None]*A_field[index(xindex, 1 ),index(yindex, 0),...]  + ((xrem)*(yrem))[:,None,None]*A_field[index(xindex, 1 ),index(yindex, 1),...])
    
    DuDtmat[:] =  (((1 - xrem) * (1 - yrem))[:,None] *DuDt[index(xindex,0 ),index(yindex, 0),...] + ((1 - xrem) * (yrem) )[:,None]*DuDt[index(xindex,0 ),index(yindex, 1),...] + ((xrem)*(1-yrem))[:,None]*DuDt[index(xindex, 1 ),index(yindex, 0),...]  + ((xrem)*(yrem))[:,None]*DuDt[index(xindex, 1 ),index(yindex, 1),...])
    
    
    return umat,Amat,DuDtmat

def RHSp(t,pos,umat,DuDtmat):
    yprtcl[:,:d] = pos[:,d:2*d]
    yprtcl[:,d:2*d] = (alph*(umat - pos[:,d:2*d])/st + 3*(1-alph)*DuDtmat)
    return yprtcl
    


# def RHSZ(t,Z,A,ugrdA):
#     return (-alph*(Z - A) - Z@Z + 3*(1 - alph)*A@A)/st + 3*(1 - alph)*ugrdA        
def RHSZ(t,Z,A):
    return (-alph*(Z - A) - Z@Z + 3*(1 - alph)*A@A)/st #+ 3*(1 - alph)*ugrdA
## ------------------- RHS w/o viscosity --------------------
"""psi = str fn ; xi = vorticity"""
def adv(t,xi,i):
    global psi, ur, vr,xi_xr, xi_yr, advterm,dealias,st
    psi[:] = -lapinv*xi*dealias
    e_arr[:] = e2d_to_1d(0.5*(xi*np.conjugate(psi))*normalize) #!
    if np.sum(e_arr)>100: raise ValueError("Blowup")
    u_field[...,0] = ifft2(1j * ky*psi)
    u_field[...,1] = ifft2(-1j * kx*psi) 
    xi_xr[:] = ifft2(1j * kx*xi)
    xi_yr[:] = ifft2(1j * ky*xi)
    advterm[:] = u_field[...,0]*xi_xr + u_field[...,1]*xi_yr
    return -1.0*dealias * (fft.rfft2(advterm)) + forcing(xi,psi)

# def conv(t,x,q):
#     return dq

## ----------------------------------------------------------
    

## -------------The RK4 integration function -----------------
def evolve_and_save(f,t,x0):
    # print()
    h = t[1] - t[0]
    x_old[:] = x0
    etot = 1.0*np.zeros(len(t)//20) +1
    
    for i,ti in enumerate(t[:-1]):
        # print(np.round(t[i],2),end= '\r')
        ## Things to do every 1 second (dt = 0.1)
        A_field[:]  = calcA(psi,A_field)
        if i%savestep ==0:
            psi[:] = -lapinv*x_old
            e_arr[:] = e2d_to_1d(0.5*(x_old*np.conjugate(psi))*normalize)
            print(f"Energy at time {np.round(t[i],2)} is {np.sum(e_arr):.4f}",end= '\r')
            
            ## Saving the vorticity contour
            # np.save(f"data/vorticity {np.round(t[i]/t[-1]*100,2)}%",ifft2(x_old))
            # vorticity[i//st,:] = ifft2(x_old).get()
            savePath = curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/time_{t[i]:.2f}/"
            savePath.mkdir(parents=True, exist_ok=True)
            np.save(savePath/f"w", x_old)
            np.save(savePath/f"e_arr", e_arr)
            
        if i%prtcl_savestep==0:
            # print(np.round(t[i],2),end= '\r')
            
            ## Saving the vorticity contour
            # np.save(f"data/vorticity {np.round(t[i]/t[-1]*100,2)}%",ifft2(x_old))
            # vorticity[i//st,:] = ifft2(x_old).get()
            savePathprtcl = curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/alpha_{alph:.2}_prtcl/time_{t[i]:.2f}"
            savePathprtcl.mkdir(parents=True, exist_ok=True)
            np.save(savePathprtcl/f"pos", xprtcl[:,:d])
            np.save(savePathprtcl/f"vel", xprtcl[:,d:2*d])
            # np.save(savePathprtcl/f"prtcl_A", A_field)
            # TrZ[:] =  np.einsum('...ii->...', Z).ravel()
            # np.save(savePathprtcl/f"prtcl_Z", Z)
            # np.save(savePathprtcl/f"prtcl_TrZ", TrZ)
        
        
        k1[:] = adv(ti,x_old,i)
        DuDt[...,0] = ifft2(-1j*lapinv*ky*(k1 - xivis*(x_old )))
        DuDt[...,1] = ifft2(1j*lapinv*kx*(k1 - xivis*(x_old )))
        
        
        umat[:],Amat[:],DuDtmat[:] = Linterp(xprtcl[:,:d],u_field,A_field,DuDt)    
        
        # eiga[i] = eigval(Amat.astype(complex),eiga[i])
        # eigz[i] = eigval(np.nan_to_num(Z[i], posinf = 0.0, neginf=0.0).astype(complex) ,eigz[i])
        k1p[:] = RHSp(t[i],xprtcl,umat,DuDtmat)
        # k1Z[:] = RHSZ(t[i],Z,Amat)
        
        xtemp[:] = (xprtcl+h*k1p/2)
        xtemp[:,:d] = xtemp[:,:d]%Lx
        
        k2[:] = adv(ti + h/2, x_old + h/2*k1,i)
        A_field[:]  = calcA(psi,A_field)
        DuDt[...,0] = ifft2(-1j*lapinv*ky*(k2 - xivis*(x_old + h/2*k1)))
        DuDt[...,1] = ifft2(1j*lapinv*kx*(k2 - xivis*(x_old + h/2*k1)))
        
        umat[:],Amat[:],DuDtmat[:] = Linterp(xtemp[:,:d],u_field,A_field,DuDt)    
        # DuDtmat[:] = Amat[...,0]*umat[...,0].reshape(Nprtcl,1) + Amat[...,1]*umat[:,1].reshape(Nprtcl,1) 
        
        k2p[:] = RHSp(t[i]+h/2,xtemp,umat,DuDtmat)
        # k2Z[:] = RHSZ(t[i]+ h/2.,Z+ h/2.*k1Z,Amat)

        xtemp[:] = (xprtcl+h*k2p/2)
        xtemp[:,:d] = xtemp[:,:d]%Lx
        
        k3[:] = adv(ti + h/2, x_old + h/2*k2,i)
        A_field[:]  = calcA(psi,A_field)
        DuDt[...,0] = ifft2(-1j*lapinv*ky*(k3 - xivis*(x_old + h/2*k2)))
        DuDt[...,1] = ifft2(1j*lapinv*kx*(k3 - xivis*(x_old + h/2*k2)))
        
        umat[:],Amat[:],DuDtmat[:] = Linterp(xtemp[:,:d],u_field,A_field,DuDt)    
        # DuDtmat[:] = Amat[...,0]*umat[...,0].reshape(Nprtcl,1) + Amat[...,1]*umat[:,1].reshape(Nprtcl,1) 

        k3p[:] = RHSp(t[i]+h/2,xtemp,umat,DuDtmat)
        # k3Z[:] = RHSZ(t[i]+ h/2.,Z+ h/2.*k2Z,Amat)

        xtemp[:] = (xprtcl+h*k3p)
        xtemp[:,:d] = xtemp[:,:d]%Lx
        
        k4[:] = adv(ti + h, x_old + h*k3,i)
        A_field[:]  = calcA(psi,A_field)
        DuDt[...,0] = ifft2(-1j*lapinv*ky*(k4 - xivis*(x_old + h*k3)))
        DuDt[...,1] = ifft2(1j*lapinv*kx*(k4 - xivis*(x_old + h*k3)))
        
        umat[:],Amat[:],DuDtmat[:] = Linterp(xtemp[:,:d],u_field,A_field,DuDt)    
        # DuDtmat[:] = Amat[...,0]*umat[...,0].reshape(Nprtcl,1) + Amat[...,1]*umat[:,1].reshape(Nprtcl,1) 

        k4p[:] = RHSp(t[i]+h,xtemp,umat,DuDtmat)
        # k4Z[:] = RHSZ(t[i]+ h,Z+ h*k3Z,Amat)

        x_new[:] = (x_old + h/6.0*(k1 + 2*k2 + 2*k3 + k4))/(1.0 + h*xivis)
        xprtcl_new[:] = xprtcl + h*(k1p + 2*k2p + 2*k3p + k4p)/6    
        xprtcl_new[:,:d] = xprtcl_new[:,:d]%Lx
        # Z_new[:] = Z + h/6.*(k1Z+ 2*k2Z + 2*k3Z + k4Z)
        # print(f"Max TrZ at time {t[i+1]} : {np.max(np.abs(Z[i+1,:,0,0] + Z[i+1,:,1,1] + Z[i+1,:,2,2]))}")
        # print(f"Max TrZ at time {t[i+1]} : {np.max(np.abs(np.einsum('...ii->...', Z[i+1]).ravel()))}")
        
        
        
            
            
            
            
        x_old[:] = x_new
        psi[:] = -lapinv*x_old      
        xprtcl[:] = xprtcl_new
        # Z[:] = Z_new
        
        
    ## Saving the last vorticity        
    # vorticity[i//st+1,:] = ifft2(x_old).get()
    psi[:] = -lapinv*x_old
    e_arr[:] = e2d_to_1d(0.5*(x_old*np.conjugate(psi))*normalize)
    print(f"Energy at time {np.round(t[i],2)} is {np.sum(e_arr):.2f}",end= '\r')
    savePath = curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/time_{t[i]:.2f}/"
    savePath.mkdir(parents=True, exist_ok=True)
    np.save(savePath/f"w", x_old)
    np.save(savePath/f"e_arr", e_arr)
    savePathprtcl = curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/alpha_{alph:.2}_prtcl/time_{t[i]:.2f}"
    savePathprtcl.mkdir(parents=True, exist_ok=True)
    np.save(savePathprtcl/f"pos", xprtcl[:,:d])
    np.save(savePathprtcl/f"vel", xprtcl[:,d:2*d])
    # TrZ[:] =  np.einsum('...ii->...', Z).ravel()
    # np.save(savePathprtcl/f"prtcl_Z", Z)
    # np.save(savePathprtcl/f"prtcl_TrZ", TrZ)
    # np.save(f"data/Re_{np.round(1/nu,2)},dt_{dt},N_{Nx}/alpha_{alph:.2}_prtcl_Z", Z)
    # np.save(f"data/Re_{np.round(1/nu,2)},dt_{dt},N_{Nx}/alpha_{alph:.2}_prtcl_TrZ", TrZ)
    return etot
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
    loadPath = curr_path/f"data/Re_{np.round(Re,2)},dt_{0.002},N_{Nx}/last/"
    xi0 = np.load(loadPath/f"w.npy")
    psi0 = -lapinv*xi0

u_field[...,0] = ifft2(1j * ky*psi0)
u_field[...,1] = ifft2(-1j * kx*psi0) 
e_arr = e2d_to_1d(0.5*(xi0*np.conjugate(psi0))*normalize)
e0 = np.sum(e_arr)
ek_arr0= 0*e_arr.copy()
ek_arr0[shell_count] = e_arr[shell_count]
print(f"Initial energy : {np.sum(e_arr)}")
u_field[...,0] = ifft2(1j * ky*psi0)
u_field[...,1] = ifft2(-1j * kx*psi0) 
A_field[:]  = calcA(psi0,A_field)

xprtcl[:,:d] = np.random.uniform(0,Lx,(Nprtcl,d))
umat[:],Amat[:],DuDtmat[:] = Linterp(xtemp[:,:d],u_field,A_field,DuDt)    
xprtcl[:,d:2*d] = umat
Z[:] = Amat

## ----------------------------------------------------




t1 = time()
etot = evolve_and_save(adv,t,xi0)
t2 = time()
print(f"Time taken to evolve for {T} secs in {dt} sec timesteps with gridsize {Nx}x{Nx} is \n {t2-t1} sec")

# Saving a copy of the params along with the data
with open(curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/parameters.json","w") as jsonFile: json.dump(params, jsonFile,indent = 2)

"""
Run with something like:
nohup time python -u 2DV.py > simul_256.out &
"""