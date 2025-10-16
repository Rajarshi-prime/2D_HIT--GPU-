import cupy as np
import numpy as npp
import math
import cupy.fft as fft


def initialize_b_spline(order,X,dx,Nx):
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
    nums = np.arange(order)
    
    return ck2d,Mmat,nums
    
def interp_spline(pos,u_field,A_field,DuDt_field,grdDuDt_field):
    """
    b-spline interpolation of the specified order given in Hinsberg et al.
    """
    global umat,Amat,DuDt_mat,grdDuDt_mat,nums,idx,delx,poly,slx,sly,temparr,order,Mmat,dx,N,Nprtcl
    umat[:] = 0.0
    Amat[:] = 0.0
    DuDt_mat[:] = 0.0
    grdDuDt_mat[:] = 0.0
    
    idx[:] = (pos//dx).astype(np.int32)
    delx[:] = (pos%dx)/dx    
    poly[:] = delx[...,None]**nums
    
    umat[:],Amat[:],DuDt_mat[:],grdDuDt_mat[:] = parallel_computation_cuda(order, Mmat, poly, idx, N,Nprtcl, u_field,DuDt_field,A_field,grdDuDt_field,umat,DuDt_mat,Amat,grdDuDt_mat)
    
    
    #! This is the main loop for the interpolation. The loop is replaced by kernel code which makes it faster. 
    # for i in range(order):
    #     for j in range(order):
    #         temparr[:] = np.einsum('p,q,...p,...q->...',Mmat[i],Mmat[j],poly[...,0,:],poly[...,1,:]) #! For saving computations
    #         slx[:] = (idx[:,0]-1 + i)%N #! For saving computations
    #         sly[:] = (idx[:,1]-1 + j)%N #! For saving computations
            
    #         umat  += u_field[slx,sly,...]*temparr[:,None]
            
    #         DuDt_mat  += DuDt_field[slx,sly]*temparr[:,None]
            
    #         Amat  += A_field[slx,sly]*temparr[:,None,None]
            
    #         grdDuDt_mat  +=  grdDuDt_field[slx,sly]*temparr[:,None,None]
            
    return umat,Amat,DuDt_mat,grdDuDt_mat




#* The structure of the kernel code is obtained from claude.ai and microsoft copilot. However it is modified to make it accurate and suit the need for the problem.


kernel_code = r'''
extern "C" __device__ double atomicAdd_new(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

extern "C" __global__ void process_combinations(
    const double* pos,
    const double* Mmat,
    const double* u_field,
    const double* DuDt_field,
    const double* A_field,
    const double* grdDuDt_field,
    double* umat,
    double* DuDt_mat,
    double* Amat,
    double* grdDuDt_mat,
    int order,
    int N,
    int Np,
    double dx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_combinations = order * order;
    
    if (tid < total_combinations) {
        int i = tid / order;
        int j = tid % order;
        
        // Process one i,j combination per thread
        for (int b = 0; b < Np; b++) {
            double temparr = 0.0;
            int idxx = pos[b*2 + 0]/dx;
            int idxy = pos[b*2 + 1]/dx;
            double delx = ((pos[b*2 + 0]%dx + dx)%dx)/dx;
            double dely = ((pos[b*2 + 1]%dx + dx)%dx)/dx;
            // Manual einsum computation
            for (int p = 0; p < order; p++) {
                for (int q = 0; q < order; q++) {
                    double polyx = delx^p;
                    double polyy = dely^q;
                    temparr += Mmat[i*order + p] * Mmat[j*order+q] * polyx * polyy;
                }
            }
            
            // Compute indices
            int slx = ((idxx - 1 + i)%N + N) % N;
            int sly = ((idxy - 1 + j)%N + N) % N;
 
            // Update umat using custom atomicAdd for double precision
            for (int comp =0 ; comp < 2; comp++) {
            atomicAdd_new(&umat[b*2 + comp], u_field[slx*2*N + sly*2 + comp] * temparr);
            atomicAdd_new(&DuDt_mat[b*2 + comp], DuDt_field[slx*2*N + sly*2 + comp] * temparr);
                for (int comp1 =0;comp1<2;comp1++){
                atomicAdd_new(&Amat[b*2*2 + comp*2 + comp1], A_field[slx*2*2*N + sly*2*2 + comp*2 + comp1] * temparr);
                atomicAdd_new(&grdDuDt_mat[b*2*2 + comp*2 + comp1], grdDuDt_field[slx*2*2*N + sly*2*2 + comp*2 + comp1] * temparr);
                }
            }
        }
    }
}
'''

def cuda_interp(order, Mmat, poly, idx, N,Np, u_field,DuDt_field,A_field,grdDuDt_field,umat,DuDt_mat,Amat,grdDuDt_mat,kernel_code = kernel_code):
    """
    Parallel computation of the interpolation using CUDA
    The kernel code does everything in C.
    """
    umat[:] = 0.0
    DuDt_mat[:] = 0.0
    Amat[:] = 0.0
    grdDuDt_mat[:] = 0.0
    
    # Compile the kernel
    threads_per_block = order*order
    module = np.RawModule(code=kernel_code)
    kernel = module.get_function('process_combinations')
    
    # Calculate grid dimensions
    total_threads = order * order
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    kernel((blocks,), (threads_per_block,),(Mmat, u_field,DuDt_field,A_field,grdDuDt_field, umat,DuDt_mat,Amat,grdDuDt_mat,np.int32(order), np.int32(N), np.int32(Np), np.float64(dx)))
    return umat,DuDt_mat,Amat,grdDuDt_mat