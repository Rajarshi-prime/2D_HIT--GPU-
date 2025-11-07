#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
#%%
def a0(alpha, Q, R): 
    return (1/256.) * alpha**2 * (
    3 * alpha**3
    + 16 * (3 * alpha + 1) * alpha * Q
    + 576 * (1 - alpha) * R)**2
def a2(alpha, Q, R):
    return (15 * alpha**4 / 8. + 144 * (1 - alpha)**2 * Q**2 - 2 * (19 - 15 * alpha) * alpha**2 * Q  - 72 * (1 - alpha) * alpha * R)
def a1(alpha,Q,R):
    return -(7 * alpha**6 / 16. + 24 * (1 - alpha) * (7 - 3 * alpha) * alpha**2 * Q**2 + 0.5 * (21 * alpha - 13) * alpha**4 * Q + 288 * (1 - alpha)**2 * alpha * Q * R + 1728 * (1 - alpha)**3 * R**2 - 4 * (9 * alpha + 7) * alpha**3 * R)

def a3(alpha, Q, R): 
    return 3*(8*Q*(1-alpha)- alpha**2)
    
def coeffs(alpha,Q,R):
    return np.array([1, a3(alpha,Q,R), a2(alpha,Q,R), a1(alpha,Q,R),a0(alpha,Q,R)])
#%%
Q = np.linspace(-2,2,1000)
R = np.linspace(-2,2,1000)
q,r = np.meshgrid(Q,R,indexing ='ij')
# %%
def nroots(alpha):
    global q,r
    roots = np.zeros((4, Q.size, R.size),dtype = np.complex128)
    count_real_pos_root = np.zeros((Q.size, R.size), dtype = np.int32)
    
    for i in range(Q.size):
        for j in range(R.size):
            roots[:,i,j] = np.roots(coeffs(alpha,q[i,j],r[i,j]))
            count_real_pos_root[i,j] = 2*((np.abs(roots[:,i,j].imag) < 1e-4)*(roots[:,i,j].real>0)).sum()
    return count_real_pos_root

#%%
from matplotlib.colors import LinearSegmentedColormap
cols = ["#d00000","#ffba08","#3f88c5","#a8c256","#000000"]
cols.reverse()
cmap= LinearSegmentedColormap.from_list("coolors", cols)
#%%
root_count = [nroots(alpha) for alpha in [0.7,0.75,0.85,1]]
#%%
# %%
mpl.rc("text", usetex = False)
fig , axs = plt.subplots(1,4,figsize = (12,5))
labels = ["$(a)$","$(b)$","$(c)$","$(d)$"]
for label,ax,alpha,roots in zip(labels,axs,[0.7,0.75,0.85,1],root_count):
    p1 = ax.imshow(roots, extent = (R.min(), R.max(), Q.min(), Q.max()), origin = 'lower', cmap = cmap, alpha = 0.9)
    ax.set_title(fr"{label} $\alpha = {alpha}$")
# plt.colorbar(p1, )

for ax in axs:
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel("$R$")
axs[0].set_ylabel("$Q$")
fig.tight_layout(pad = 0.2)
cbar = fig.colorbar(p1, ticks=[0, 2, 4, 6, 8], ax=axs, orientation="horizontal", shrink=0.5,location = "bottom",spacing = "uniform",pad = 0.2)
cbar.ax.set_title(r"# of real roots")
# %%
# %%
