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

savestep = int(params["savestep"]/dt) # Save the data after this many timesteps
eta = params["eta"]/(Nx//3) # Desired Kolmogorov length scale
kf = params["kf"] # Forcing wavenumber
f0 = params["f0"] # Forcing amplitude
t = np.arange(0,1.1*dt + T,dt) # Time array
P = (2*np.pi)**2*(nu**3/eta**4)/len(shell_count) # power per unit shell

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


ur = np.empty((Nx,Ny),dtype = np.float64)
vr = np.empty((Nx,Ny),dtype = np.float64)
xi_xr = np.empty((Nx,Ny),dtype = np.float64)
xi_yr = np.empty((Nx,Ny),dtype = np.float64)
advterm = np.empty((Nx,Ny),dtype = np.float64)
factor = np.empty(Nf)
factor2d = np.empty((Nx,Nf))

def e2d_to_1d(x):
    return (np.histogram(k.ravel(),bins = shells,weights=x.ravel() )[0]).real
## --------------------------------------------------------------

k_forc = kf
f_r = -k_forc*f0*np.sin(kf*y)
f = fft.rfft2(f_r)
def forcing(xi,psi):
    # """
    # Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
    # """
    
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


## ------------------- RHS w/o viscosity --------------------
"""psi = str fn ; xi = vorticity"""
def adv(t,xi,i):
    global psi, ur, vr,xi_xr, xi_yr, advterm,dealias,savestep
    psi[:] = -lapinv*xi*dealias
    # e_arr[:] = e2d_to_1d(0.5*(xi*np.conjugate(psi))*normalize) #!
    # if np.sum(e_arr)>100: raise ValueError("Blowup")
    ur[:] = ifft2(1j * ky*psi)
    vr[:] = ifft2(-1j * kx*psi) 
    xi_xr[:] = ifft2(1j * kx*xi)
    xi_yr[:] = ifft2(1j * ky*xi)
    advterm[:] = ur*xi_xr + vr*xi_yr
    return -1.0*dealias * (fft.rfft2(advterm)) + forcing(xi,psi) - xivis*xi*dealias




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
        
        
        # print(np.abs(x_old).max())
        k1[:] = adv(ti,x_old,i)
        # print(np.max(np.abs(k1)))
        k2[:] = adv(ti + h/2, x_old + h/2*k1,i)
        k3[:] = adv(ti + h/2, x_old + h/2*k2,i)
        k4[:] = adv(ti + h, x_old + h*k3,i)

        x_new[:] = (x_old + h/6.0*(k1 + 2*k2 + 2*k3 + k4))#/(1.0 + h*xivis)
        ## Things to do every 1 second (dt = 0.1)
        if i%savestep ==0:
            psi[:] = -lapinv*x_old
            e_arr[:] = e2d_to_1d(0.5*(x_old*np.conjugate(psi))*normalize)
            print(f"Energy at time {np.round(t[i],2)} is {np.sum(e_arr):.4f}",end= '\r')
            
            ## Saving the vorticity contour
            # np.save(f"data/vorticity {np.round(t[i]/t[-1]*100,2)}%",ifft2(x_old))
            # vorticity[i//savestep,:] = ifft2(x_old).get()
            savePath = curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/last"
            savePath.mkdir(parents=True, exist_ok=True)
            np.save(savePath/f"w", x_old)
            np.save(savePath/f"e_arr", e_arr)
            
        x_old[:] = x_new
        psi[:] = -lapinv*x_old      
        
        
    ## Saving the last vorticity        
    psi[:] = -lapinv*x_old
    e_arr[:] = e2d_to_1d(0.5*(x_old*np.conjugate(psi))*normalize)
    print(f"Energy at time {np.round(t[i],2)} is {np.sum(e_arr):.4f}",end= '\r')
    savePath = curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/last"
    savePath.mkdir(parents=True, exist_ok=True)
    np.save(savePath/f"w", x_old)
    np.save(savePath/f"e_arr", e_arr)
    return etot
## ---------------------------------------------------------   


    









## ---------------- Initial conditions ----------------
# r = np.random.RandomState(309)
# thinit = r.uniform(0,2*np.pi,np.shape(k))
# eprofile = np.exp(-np.arange(Nf)/5)
# eprofile[shell_count] = 1e-8
# eprofile[:] = eprofile/np.histogram(k.ravel(),bins = shells)[0]
# psi0  = (eprofile[kint])**0.5*np.exp(1j*thinit)*(kint>0)*(kint<kinit)/np.where(kint>0, kint**1,1)
# xi0 = -lap*psi0
xi0_r = -f0*kf*nu*(np.cos(kf*x) + np.cos(kf*y))
xi0 = fft.rfft2(xi0_r)
# xi0 = np.load(curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/last/w.npy")
# xi0_r = ifft2(xi0)
psi0 = -lapinv*xi0
e_arr = e2d_to_1d(0.5*(xi0*np.conjugate(psi0))*normalize)
ek_arr0= 0*e_arr.copy()
ek_arr0[shell_count] = e_arr[shell_count]
e0 = np.sum(e_arr)

# xi0[:] = xi0*(einit/e0)**0.5
# psi0[:] = psi0*(einit/e0)**0.5
ur[:] = ifft2(1j * ky*psi0)
vr[:] = ifft2(-1j * kx*psi0) 
e_arr = e2d_to_1d(0.5*(xi0*np.conjugate(psi0))*normalize)
print(f"Initial energy : {np.sum(e_arr)},(dx/u_max) : {(dx/np.max(np.abs(np.array([ur,vr])))):.4f}")
## ----------------------------------------------------

initial_savePath = pathlib.Path(f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/initial")
initial_savePath.mkdir(parents=True, exist_ok=True)
np.savez_compressed(initial_savePath/f"init_fields.npz",xi0 = xi0,e_arr = e_arr)

t1 = time()
print(f"Initial zeta:{np.max(np.abs(xi0))}")
etot = evolve_and_save(adv,t,xi0)
t2 = time()
print(f"Time taken to evolve for {T} secs in {dt} sec timesteps with gridsize {Nx}x{Nx} is \n {t2-t1} sec")

# Saving a copy of the params along with the data
with open(curr_path/f"data/Re_{np.round(Re,2)},dt_{dt},N_{Nx}/parameters.json","w") as jsonFile: json.dump(params, jsonFile,indent = 2)


"""
Run with something like:
nohup time python -u 2DV_bare.py > simul_256.out &
"""
