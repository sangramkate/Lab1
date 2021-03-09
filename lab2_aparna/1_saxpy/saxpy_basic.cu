#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from dev_offsetition of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary  
  return 0;
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
    /*
    *xarray = (float*) malloc(size*sizeof(float));
    *yarray = (float*) malloc(size*sizeof(float));
    *resultarray = (float*) malloc(size*sizeof(float));*/
    gpuErrchk(cudaMallocHost(xarray, size*sizeof(float)));
    gpuErrchk(cudaMallocHost(yarray, size*sizeof(float)));
    gpuErrchk(cudaMallocHost(resultarray, size*sizeof(float)));

}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
    gpuErrchk(cudaFreeHost(xarray));
    gpuErrchk(cudaFreeHost(yarray));
    gpuErrchk(cudaFreeHost(resultarray));
}


void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 512; // change this if necessary

    float *device_x;
    float *device_y;
    float *device_result;

    gpuErrchk(cudaMalloc(&device_x, total_elems*sizeof(float)));
    gpuErrchk(cudaMalloc(&device_y, total_elems*sizeof(float)));
    gpuErrchk(cudaMalloc(&device_result, total_elems*sizeof(float)));


    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO: Compute number of thread blocks.
    // 


    double startGPUTime = CycleTimer::currentSeconds();
    //Copy to GPU
     gpuErrchk(cudaMemcpy(device_x, xarray, total_elems*sizeof(float), cudaMemcpyHostToDevice));
     gpuErrchk(cudaMemcpy(device_y, yarray, total_elems*sizeof(float), cudaMemcpyHostToDevice));
     //gpuErrchk(cudaMemcpy(device_result, resultarray, total_elems*sizeof(float), cudaMemcpyHostToDevice));
    double endMemCpyToDevice = CycleTimer::currentSeconds();
    timeCopyH2DAvg += (endMemCpyToDevice - startGPUTime); 

    double startKernelTime = CycleTimer::currentSeconds();

    // run saxpy_kernel on the GPU
    saxpy_kernel<<<(total_elems + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(total_elems, alpha, device_x, device_y, device_result);

    //
    // insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaDeviceSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaDeviceSynchronize();
    double endKernelTime = CycleTimer::currentSeconds();
    timeKernelAvg += (endKernelTime - startKernelTime);

    double endGPUTime = CycleTimer::currentSeconds();
    
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    
    // Copy result from GPU using cudaMemcpy
    double startMemCpyToHost = CycleTimer::currentSeconds();
     gpuErrchk(cudaMemcpy(resultarray, device_result, total_elems*sizeof(float), cudaMemcpyDeviceToHost));
    

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    timeCopyD2HAvg += (endTime - startMemCpyToHost);
    totalTimeAvg   += overallDuration;


    // free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
