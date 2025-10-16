# %%
import os,pathlib,json,sys,glob
import numpy as np

# %%
reg_fnames = ["w.npy","e_arr.npy","pos.npy","vel.npy","caus_count.npy","prtcl_Z.npy"]

# %%
def find_npy_files(root_folder):
    npy_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in glob.glob(os.path.join(dirpath, '*.npy')):
            npy_files.append(filename)
    return npy_files


# %%

# Example usage
root_folder = '/home/rajarshi.chattopadhyay/fluid/2DV_and_particles/data/'
npy_files = find_npy_files(root_folder)


# %%
for file in npy_files:
    fname = file[:-4].split('/')[-1] #! the filename without the path
    # print(fname,end = "\r")
    new_fname = file[:-4] + ".npz" #! The new filename with the path
    try: 
        data = np.load(file) 
    except ValueError:
        print(f"Error in file {file}")
        os.remove(file) #! remove the old file
        continue
    np.savez_compressed(new_fname,fname =  data)
    os.remove(file) #! remove the old file
    # if fname not in reg_fname: print(fname)

# %%



