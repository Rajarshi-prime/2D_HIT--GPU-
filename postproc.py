import numpy as np
import os,json,pathlib,sys
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rc('text', usetex = True)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'})

## ---------------- params ----------------------- ## 
paramfile = 'parameters.json'
with open(paramfile,'r') as jsonFile: params = json.load(jsonFile)

d = params["d"] # Dimension
nu =params["nu"] # Viscosity
Re = 1/nu if nu > 0 else np.inf # Reynolds number
N = Nx = Ny = params["N"] # Grid size
dt = params["dt"] # Timestep
# T = params["T"] # Final time
T = 52 # Final time
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
t = np.arange(0,1.1*dt + T,dt) # Time array
P = (nu**3/eta**4)/len(shell_count) # power per unit shell

## ----------------------------------------------- ##


d# %%
def find_range(x,idx): return np.sum(np.cumsum(x[:idx][::-1]) == range(1,len(x[:idx])+1))+np.sum(np.cumsum(x[idx:])== range(1,len(x[idx:])+1))
# %%
curr_path = pathlib.Path(__file__).parent 
savePlot = pathlib.Path(f"/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/Plots/Re_{np.round(Re,2)},dt_{dt},N_{N}/")
savePlot.mkdir(parents=True, exist_ok=True)
loadPath = pathlib.Path(f"/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/data/Re_{np.round(Re,2)},dt_{dt},N_{N}/")
loadPath.exists()
# loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/Caustics3D/data_new/fluid/omg/alph_0.7000/st_0.00848/dt_8.48e-06")
try: savePlot.mkdir(parents=True, exist_ok=False)
except FileExistsError: pass
savePlotdataPath = savePlot.parent/f"plotdat"
try: savePlotdataPath.mkdir(parents = True, exist_ok = False)
except FileExistsError: pass

os.listdir(loadPath)

times = np.arange(0,T+ 0.1,params["prtcl_savestep"])

#! Goal -- load files one by one, extract caustics and non-caustics particles.
"""Check and store statistics. And be done with it
Stats to store: 
The eigenvaules series for caustics particles.
The time of caustics

"""

print(times[-1],st,str(savePlot))
Ntimes = len(times)
TrZ = np.zeros((Ntimes, Nprtcl))
# vel = np.zeros((Ntimes, Nprtcl,d))
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
for i,t in enumerate(times):
    print(f"loading time {i}",end='\r')
    TrZ[i] = np.einsum('...ii->...', np.load(loadPath/f"alpha_{alph:.2}_prtcl/time_{t:.2f}/prtcl_Z.npy"))
    # vel[i] = np.load(loadPath/f"alpha_{alph:.2}_prtcl/time_{t:.2f}/vel.npy")
    # eiga[:] = np.load(loadPath/f"eiga_{i+1+exception}.npy")
    # eigz[:] = np.load(loadPath/f"eigz_{i+1+exception}.npy")
    # initQ[i*Nprtcl:(i+1)*Nprtcl] = -0.5*np.sum(eiga[0,:]**2,axis =1).real
    # initR[i*Nprtcl:(i+1)*Nprtcl] = -1/3*np.sum(eiga[0,:]**3,axis =1).real

#     ## ------------ data for caustics particles ------------ ##
#     """ Calculating the indices of the particles which had caustics"""
#     prtcls = np.arange(TrZ.shape[1])[np.isnan(TrZ).any(axis =0)] 
#     causno = causno + len(prtcls)
#     # print(np.moveaxis(TrZ[:,prtcls],0,1).shape)
#     """Picking out the TrZ of those particles but putting particles to be in axis 0
#     so that we can add row wise to the list"""
#     causTrZ = causTrZ + list(np.moveaxis(TrZ[:,prtcls],0,1))
#     # causeiga = causeiga + list(np.moveaxis(eiga[:,prtcls],0,1))
#     # causeigz = causeigz + list(np.moveaxis(eigz[:,prtcls],0,1))
#     causvel = causvel + list(np.moveaxis(vel[:,prtcls],0,1))
#     ## ------------------------------------------------------ ## 
    
# #     ## ------- distribution for non-caustics particles ------ ##
# #     Q_s1 = -0.5*np.sum(eiga[:,~prtcls]**2, axis = 2).real
# #     R_s1 = -1/3*np.sum(eiga[:,~prtcls]**3, axis = 2).real
# #     t_s1 = times[:,None]*np.ones_like(Q_s1)
# #     Qpdf[:] = Qpdf + np.histogram2d(t_s1.ravel(),Q_s1.ravel(), bins = (tbins, Qbins))[0] # type: ignore
# #     Rpdf[:] = Qpdf + np.histogram2d(t_s1.ravel(),R_s1.ravel(), bins = (tbins, Rbins))[0] # type: ignore
# #     ## ------------------------------------------------------ ##

# del eiga,eigz,TrZ,vel
# # np.save(loadPath/f"Qpdf_non_caustic.npy",Qpdf/(batches*Nprtcl-causno))
# # np.save(loadPath/f"Rpdf_non_caustic.npy",Rpdf/(batches*Nprtcl-causno))
# # del Qpdf,Rpdf,Q_s1,R_s1,t_s1
# raise SystemExit
causidx = np.isnan(TrZ).any(axis = 0)
print(causidx,causidx.sum())
causTrZ = TrZ[:,causidx]
# causvel = vel[:,causidx]
causidx = np.argmax(np.isnan(causTrZ),axis = 0) - 1
print(causidx.shape)
caustimes = times[causidx]
print(caustimes,caustimes.max())
# causTrZ = np.moveaxis(np.array(causTrZ),0,1) #Moving the axis back : axis 0 is time. 
# # causeigz = np.moveaxis(np.array(causeigz),0,1)

# print(f"# of caustics : {causno} and causTrZ has the shape {causTrZ.shape}")

# causidx = np.argmax(np.isnan(causTrZ), axis=0) - 1 #extracting the first instant where caustics happend.
# tcaus = times[causidx]
# causTrZ[:] = np.where(times[:,None] > tcaus,causTrZ[causidx],causTrZ)
# # causeigz[:] = np.where(times[:,None,None] > tcaus[:,None],causeigz[causidx],causeigz)
# # np.save(loadPath/f"tcaus",tcaus)
# # np.save(loadPath/f"causeigz.npy",causeigz)
# # np.save(loadPath/f"causTrZ.npy",causTrZ)
# del causeigz,causTrZ

# # causeiga = np.moveaxis(np.array(causeiga),0,1)
# causvel = np.moveaxis(np.array(causvel),0,1)
# # # eiga1 = eiga.copy()


# # causeiga[:] = np.where(times[:,None,None] > tcaus[:,None],causeiga[causidx],causeiga)
# # causvel[:] = np.where(times[:,None,None] > tcaus[:,None],np.nan,causvel)


# # # #%% 
# # Q_s = -0.5*np.sum(causeiga**2, axis = 2).real
# # np.save(loadPath/f"Q_s.npy",Q_s)
# # del Q_s
# # R_s = -1/3*np.sum(causeiga**3, axis = 2).real
# # np.save(loadPath/f"R_s.npy",R_s)
# # np.save(loadPath/f"causeiga.npy",causeiga)
# # del R_s,causeiga



# ## Loading Q and R 
# # Qpdf_nc = np.load(loadPath/f"Qpdf_non_caustic.npy")
# # Rpdf_nc = np.load(loadPath/f"Rpdf_non_caustic.npy")
# Q_s = np.load(loadPath/f"Q_s.npy")
# R_s = np.load(loadPath/f"R_s.npy")
# print(f"Q_s,R_s shape: {Q_s.shape},{R_s.shape}")
# tcaus = np.load(loadPath/f"tcaus.npy")
# tcaus_max = max(tcaus)
# causno = len(tcaus)

# # condition = (alph**2 * (alph**4 + 144 * (alph - 1)**2 * Q_s**2 + 8 * (3 * alph - 1) * alph**2 * Q_s) - 16 * alph * R_s * (alph**2 * (9 * alph - 5) + 36 * (alph - 1)**2 * Q_s) - 1728 * (alph - 1)**3 * R_s**2<0)

# # # condition1 = (alph**2 * (alph**4 + 144 * (alph - 1)**2 * Q_s1**2 + 8 * (3 * alph - 1) * alph**2 * Q_s1) - 16 * alph * R_s1 * (alph**2 * (9 * alph - 5) + 36 * (alph - 1)**2 * Q_s1) - 1728 * (alph - 1)**3 * R_s1**2<0)

# # conda =lambda alph,r,q:  (alph**2 * (alph**4 + 144 * (alph - 1)**2 * q**2 + 8 * (3 * alph - 1) * alph**2 * q) - 16 * alph * r * (alph**2 * (9 * alph - 5) + 36 * (alph - 1)**2 * q) - 1728 * (alph - 1)**3 * r**2<0)
# # conda(alph,.0,0)
# # q,r = 0.,0.

# # Qbins,Rbins = list(np.arange(-0.5,0.51,0.002)),list(np.arange(-0.2,0.21,0.002))
# # vline = lambda Rbins: -3*(np.array(Rbins)**2/4)**(1/3)
# # caus_cond = lambda Rbins : 4*np.array(Rbins) - 1/16


# # def c_reg(alpha, Q, R):
    
# #     # Qs,Rs = np.meshgrid(Q,R,indexing = 'ij')
# #     Rs,Qs = np.meshgrid(R,Q,indexing = 'ij')
# #     cond = np.where(alpha**2 * (alpha**4 + 144 * (alpha - 1)**2 * Qs**2 + 8 * (3 * alpha - 1) * alpha**2 * Qs) - 16 * alpha * Rs * (alpha**2 * (9 * alpha - 5) + 36 * (alpha - 1)**2 * Qs) - 1728 * (alpha - 1)**3 * Rs**2<0,1,np.nan).T
# #     # plt.pcolor(R,Q,cond,alpha = 0.05)
# #     # print(np.min(R), np.max(R), np.min(Q), np.max(Q),end ="\r")
# #     plt.imshow(cond, extent=(np.min(R), np.max(R), np.min(Q), np.max(Q)),origin="lower",cmap="viridis", alpha = 0.3,aspect=2/5)
    
# #     return None

# # # contours = np.load(f"/home/fluiddynamics_data/JHTD/isotropic1024coarse_zyx/QR_contour_{1024}_frac_{0.99}.npz")["arr_0"]
# # # contours.shape
# # # # plt.plot(contours[:,0],contours[:,1],":")
# # # p1 = plt.pcolor(Rbins[:-1],Qbins[:-1],np.log(hist),cmap = 'jet')
# # plt.ylabel(r'$Q$')
# # plt.xlabel(r'$R$')
# # tplot = times[times<=tcaus_max]
# # dt = times[-1] - times[-2]
# # plot_step = len(tplot)//200
# # print(f"starting to plot")

# # plt.figure(figsize=(8,8))
# # s = 1
# # for i in range(0,round(len(tplot)),round(0.2*plot_step)):
# #     if s>0:
# #         plt.plot(Rbins,vline(Rbins),'--',color= 'black')
# #         plt.plot(Rbins,caus_cond(Rbins),'-.',color= 'black')
# #         c_reg(alph, Qbins, Rbins)
# #         # for frac in [0.9,0.99,0.999,0.9999,0.99999]:
# #         #     contours = np.load(f"/home/fluiddynamics_data/JHTD/isotropic1024coarse_zyx/QR_contour_{1024}_frac_{frac}.npz")["arr_0"]
# #         #     plt.plot(contours[:,0],contours[:,1],':')
# #         plt.ylim(-0.5,0.5)
# #         plt.xlim(-0.2,0.2)
# #         #! For caustics particles
# #         for j in range(Q_s.shape[1]):
# #         # for j in range(condition.shape[1]):
# #         # for j in range(3:
        
# #             # print((times[condition[:,j]]))
# #             # print(i,times[-1],(times[~condition[:,j]])[-1] + (times[condition[:,j]])[0],end = "\r")
# #             if np.sum(condition[:,j]) == 0: cs = "red"
# #             elif times[i] < (times[condition[:,j]])[0]: cs = "red"
# #             else : cs = "blue"
# #             plt.plot(R_s[:i,j],Q_s[:i,j],'-',color = "black",linewidth = 0.2)
# #             plt.plot(R_s[i,j],Q_s[i,j],marker='.',color = cs)
        
# #         # #! For non - caustics particles    
# #         # for j in range(min(Q_s1.shape[1],50)):
# #         #     if np.sum(condition1[:,j]) == 0: cs = "red"
# #         #     elif times[i] < (times[condition1[:,j]])[0]: cs = "red"
# #         #     else : cs = "blue"
            
# #         #     plt.plot(R_s1[:i,j],Q_s1[:i,j],'-',color = "black",linewidth = 0.2)
# #         #     plt.plot(R_s1[i,j],Q_s1[i,j],marker='x',color = cs)

# #         plt.savefig(savePlot/f"index_{s}.png")
# #         print(s,end = "\r")
# #         plt.clf()
# #         # print(s,end = "\r")
# #     s = s+1
    
# tbins = np.linspace(times[0],times[-1],2001)
# # ## ---------------- For caustics particles ----------------- ##
# """
# Plan A: 
# 1. Find the maximum tcaus
# 2. Set the trajectory of all particles till that tcaus_max. For particles which goes in caustics faster, add an enormous number to exclude them from the histogram
# 3. Plot pdf. 

# """

# Qbins = np.linspace(Q_s.mean() - 2*Q_s.std(),Q_s.mean() + 2*Q_s.std(), 201)
# Rbins = np.linspace(R_s.mean() - 2*R_s.std(),R_s.mean() + 2*R_s.std(), 201)
# # Qbins = np.linspace(Q_s.min() ,Q_s.max(), 201)
# # Rbins = np.linspace(R_s.min() ,R_s.max(), 201)
# # Qpdf_c = np.histogram2d(times[:,None].ravel(),Q_s.ravel(), bins = (tbins, Qbins))[0]/(Q_s.shape[1])
# # Rpdf_c = np.histogram2d(times[:,None].ravel(),R_s.ravel(), bins = (tbins, Rbins))[0]/(R_s.shape[1])


# tcaus_max = max(tcaus)
# indices = times<=tcaus_max
# t_new = times[indices] -tcaus_max
# Nnew = indices.sum()
# Q_new = np.zeros((Nnew,causno))
# R_new = np.zeros((Nnew,causno))
# vel_new = np.zeros((Nnew,causno,3))
# print(f"maximum caustics time: {tcaus_max/st}")
# tbins = np.linspace(t_new[0],t_new[-1],2001) #! Only taking the time till the maximum caustics time


# """padding the data of Q and R to a large number so that the histogram is not affected by the padding."""
# for i in range(causno):
#     cond = times<= tcaus[i]
#     pad_length = np.sum(cond)
#     Q_new[:,i] = np.pad(Q_s[cond,i], (Nnew - pad_length, 0),constant_values=1e10)
#     R_new[:,i] = np.pad(R_s[cond,i], (Nnew - pad_length, 0),constant_values=1e10)
#     vel_new[:,i] = np.pad(causvel[cond,i], (Nnew - pad_length, 0),constant_values=np.nan)
# cond1 = t_new > -10*Tp
# np.save(savePlotdataPath/f"vel_new",np.where(vel_new > 1e8,np.nan,vel_new)[cond1])
# raise ValueError
# # Q_new = Q_s[indices]
# # R_new = R_s[indices]
# R_mean = np.nanmean(np.where(R_new >1e8,np.nan,R_new),axis = 1)
# Q_mean = np.nanmean(np.where(Q_new >1e8,np.nan,Q_new),axis = 1)
# tdat = np.ones_like(Q_new)*t_new[:,None]
# """extracting out the position for the minima of Q timeseries"""
# idxQ = np.argmin(Q_mean*cond1)
# Qmin = Q_new[idxQ,:]
# Qmin = Qmin[Qmin<1e8]
# Rmin = R_new[idxQ,:]
# Rmin = Rmin[Rmin<1e8]
# """extraction out the position for the maxima of R timeseries"""
# idxR = np.argmax(R_mean*cond1)
# Rmax = R_new[idxR,:]
# Rmax = Rmax[Rmax<1e8]
# Qmax = Q_new[idxR,:]
# Qmax = Qmax[Qmax<1e8]

# initQRpdf,init_Rbins,init_Qbins = np.histogram2d(initR.ravel(),initQ.ravel(), bins = 100)
# QRpdf_init,Rinit_bins,Qinit_bins = np.histogram2d(R_s[0].ravel(),Q_s[0].ravel(), bins = 100)
# QRpdf_min,Rmin_bins,Qmin_bins = np.histogram2d(Rmin.ravel(),Qmin.ravel(), bins = 100)
# QRpdf_max,Rmax_bins,Qmax_bins = np.histogram2d(Rmax.ravel(),Qmax.ravel(), bins = 100)
# QRpdf_caus,Rcaus_bins,Qcaus_bins = np.histogram2d(R_new[-1].ravel(),Q_new[-1].ravel(), bins = 100)

# QRpdf_init[:] = QRpdf_init/QRpdf_init.sum()
# QRpdf_min[:] = QRpdf_min/QRpdf_min.sum()
# QRpdf_max[:] = QRpdf_max/QRpdf_max.sum()
# QRpdf_caus[:] = QRpdf_caus/QRpdf_caus.sum() 

# """Making the 2D histogram"""  #? Ensure that sum along each time is 1.   
# # tpdf = np.histogram(t_new, bins = tbins)[0]
# Rdat = np.histogram2d((tdat).ravel(),R_new.ravel(), bins = (tbins, Rbins))[0]
# Qdat = np.histogram2d((tdat).ravel(),Q_new.ravel(), bins = (tbins, Qbins))[0]
# Rdat[:] = Rdat/np.where(Rdat.sum(axis = 1) == 0., 1, Rdat.sum(axis = 1))[:,None]
# Qdat[:] = Qdat/np.where(Qdat.sum(axis = 1) == 0., 1, Qdat.sum(axis = 1))[:,None]
# #! Have to divide each bins of t by the number of tpoints in the bin such that the along Q or R at any time should give 1.
# print(Qdat.sum(axis = 1),Rdat.sum(axis = 1))


# np.save(savePlotdataPath/f"Qdat_c.npy",Qdat)
# np.save(savePlotdataPath/f"Qdat_mean.npy",Q_mean)
# np.save(savePlotdataPath/f"Rdat_c.npy",Rdat)
# np.save(savePlotdataPath/f"Rdat_mean.npy",R_mean)
# np.save(savePlotdataPath/f"tbins",tbins/Tp)
# np.save(savePlotdataPath/f"Qbins.npy",Qbins)
# np.save(savePlotdataPath/f"Rbins.npy",Rbins)
# np.save(savePlotdataPath/f"initQRpdf.npy",initQRpdf)
# np.save(savePlotdataPath/f"init_Rbins.npy",init_Rbins)
# np.save(savePlotdataPath/f"init_Qbins.npy",init_Qbins)
# np.save(savePlotdataPath/f"QRpdf_init.npy",QRpdf_init)
# np.save(savePlotdataPath/f"Rinit_bins.npy",Rinit_bins)
# np.save(savePlotdataPath/f"Qinit_bins.npy",Qinit_bins)
# np.save(savePlotdataPath/f"QRpdf_min.npy",QRpdf_min)
# np.save(savePlotdataPath/f"Rmin_bins.npy",Rmin_bins)
# np.save(savePlotdataPath/f"Qmin_bins.npy",Qmin_bins)
# np.save(savePlotdataPath/f"QRpdf_max.npy",QRpdf_max)
# np.save(savePlotdataPath/f"Rmax_bins.npy",Rmax_bins)
# np.save(savePlotdataPath/f"Qmax_bins.npy",Qmax_bins)
# np.save(savePlotdataPath/f"QRpdf_caus.npy",QRpdf_caus)
# np.save(savePlotdataPath/f"Rcaus_bins.npy",Rcaus_bins)
# np.save(savePlotdataPath/f"Qcaus_bins.npy",Qcaus_bins)
# np.save(savePlotdataPath/f"tcond.npy",t_new[cond1]/Tp)
# np.save(savePlotdataPath/f"Q_new",np.where(Q_new > 1e8,np.nan,Q_new)[cond1])
# np.save(savePlotdataPath/f"R_new",np.where(R_new > 1e8,np.nan,R_new)[cond1])

# """ Writing to a file about the number of caustics """
# with open("/mnt/pfs/rajarshi.chattopadhyay/copy/Caustics-3D/caustics.txt","a") as f: 
#     f.write(f"Number of caustics strating from {code} for alph = {alph}, st = {Tp/tk:.2f} is = {causno}\n Min Q time {t_new[idxQ]/Tp:.2f} and max R time {t_new[idxR]/Tp:.2f}\n")
# # plt.figure(figsize=(12,6))
# # p1 = plt.pcolor((tbins-tcaus_max)/Tp, Qbins,Qdat.T,cmap = "inferno")
# # plt.colorbar(p1)
# # plt.xlabel(r"$t -t_c$")
# # plt.ylabel(r"$R$")
# # plt.savefig(savePlot.parent/f"Qheatmap_c.png")
# # plt.close()


# # plt.figure(figsize=(12,6))
# # p1 = plt.pcolor((tbins-tcaus_max)/Tp,Qbins,Rdat.T,cmap = "inferno")
# # plt.colorbar(p1)
# # plt.xlabel(r"$t -t_c$")
# # plt.ylabel(r"$R$")
# # plt.savefig(savePlot.parent/f"Rheatmap_c.png")
# # plt.close()


# # for i in range(Q_s.shape[1]):
# #     plt.plot(tpdf[:,i]/Tp, Q_s[:,i],'-',lw = 0.2,color = "#0b7a75")
# # plt.xlim(plt.xlim()[0],0.01)
# # plt.xlabel(r"$t -t_c$")
# # plt.ylabel(r"$Q$")
# # plt.savefig(savePlot.parent/f"Qtraj-caustics.png")
# # plt.close()

# # for i in range(R_s.shape[1]):
# #     plt.plot(tpdf[:,i]/Tp, R_s[:,i],'-',lw = 0.2,color = "#7b2d26")
# # plt.xlim(plt.xlim()[0],0.01)
# # plt.xlabel(r"$t -t_c$")
# # plt.ylabel(r"$R$")
# # plt.savefig(savePlot.parent/f"Rtraj-caustics.png")
# # plt.close()


# # ## --------------------------------------------------------- ##


# ## ---------------- For non-caustics particles ------------- ##
# # Qdat = np.clip(Rpdf_nc,1e-6,None)

# # np.save(savePlotdataPath/f"Qdat_nc.npy",Qdat)
# # plt.figure(figsize=(12,6))
# # p1 = plt.pcolor(tbins/Tp, Qbins,Qdat.T,cmap = "inferno")
# # plt.colorbar(p1)
# # plt.savefig(savePlot.parent/f"Qheatmap_nc.png")
# # plt.close()

# # Rdat = np.clip(Rpdf_nc,1e-6,None)

# # np.save(savePlotdataPath/f"Rdat_c_{code}_alph_{alph:.4f}_st_{st}.npy",Rdat)
# # plt.figure(figsize=(12,6))
# # p1 = plt.pcolor(tbins/Tp,Qbins,Rdat.T,cmap = "inferno")
# # plt.colorbar(p1)
# # plt.savefig(savePlot.parent/f"Rheatmap_nc.png")
# # plt.close()
# ## --------------------------------------------------------- ##

# # %%
