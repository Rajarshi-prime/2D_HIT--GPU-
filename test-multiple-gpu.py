import cupy as cp
import numpy as np
import time

def monitor_gpu_memory():
    """Print memory usage for each GPU"""
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(num_gpus)
    for i in range(num_gpus):
        # Set the current device before checking memory
        cp.cuda.Device(i).use()
        meminfo = cp.cuda.runtime.memGetInfo()
        free = meminfo[0] / 1024**3  # Convert to GB
        total = meminfo[1] / 1024**3
        used = total - free
        print(f"GPU {i}: Using {used:.2f}GB / {total:.2f}GB")

def verify_multi_gpu_fft():
    # Enable multi-GPU FFT
    cp.fft.config.use_multi_gpus = True
    print("Multi-GPU FFT enabled")
    
    # Create a large array to ensure work is distributed
    size = 2**26  # About 67 million elements
    print(f"\nCreating large array of size {size:,}")
    data = cp.random.random(size)
    
    # Print initial GPU memory usage
    print("\nGPU memory before FFT:")
    monitor_gpu_memory()
    
    # Perform FFT and time it
    print("\nPerforming FFT...")
    start_time = time.time()
    result = cp.fft.fft(data)
    cp.cuda.Stream.null.synchronize()  # Ensure FFT is complete
    end_time = time.time()
    
    # Print final GPU memory usage
    print("\nGPU memory after FFT:")
    monitor_gpu_memory()
    
    print(f"\nFFT completed in {end_time - start_time:.2f} seconds")
    
    # Verify result is valid
    print(f"FFT result shape: {result.shape}")
    print(f"First few values: {result[:5]}")

if __name__ == "__main__":
    verify_multi_gpu_fft()