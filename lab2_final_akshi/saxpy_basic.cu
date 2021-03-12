#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

__global__ void
saxpy_kernel(long N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    long index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

void 
getArrays(long size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
  *xarray = (float*)malloc(size*sizeof(float));
  *yarray = (float*)malloc(size*sizeof(float));
  *resultarray = (float*)malloc(size*sizeof(float));
  /*cudaMallocHost((void**)xarray, size * sizeof(float));
  cudaMallocHost((void**)yarray, size * sizeof(float));
  cudaMallocHost((void**)resultarray, size * sizeof(float));
  */

}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
  free(xarray);
  free(yarray);
  free(resultarray);
  /*
  cudaFreeHost(xarray);
  cudaFreeHost(yarray);
  cudaFreeHost(resultarray);
  */
}
void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 512; // change this if necessary

    float *device_x;
    float *device_y;
    float *device_result;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    long size = total_elems*sizeof(float);
    cudaError_t err_x =  cudaMalloc((void **)&device_x,size);
    if(err_x != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_x),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    cudaError_t err_y = cudaMalloc((void **)&device_y,size);
    if(err_y != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_y),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    cudaError_t err_res = cudaMalloc((void **)&device_result,size);
    if(err_res != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_res),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    
    


    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO: Compute number of thread blocks.
    // 
    const int numBlocks = (total_elems-1)/threadsPerBlock + 1;

    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    //
    
    double copyStart = CycleTimer::currentSeconds();
    cudaError_t err_xcpy = cudaMemcpy(device_x,xarray,size,cudaMemcpyHostToDevice);
    if(err_xcpy != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_xcpy),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    cudaError_t err_ycpy = cudaMemcpy(device_y,yarray,size,cudaMemcpyHostToDevice);
    if(err_ycpy != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_ycpy),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    
    timeCopyH2DAvg += CycleTimer::currentSeconds() - copyStart;
    
    //
    // TODO: insert time here to begin timing only the kernel
    //
    double startGPUTime = CycleTimer::currentSeconds();

    // run saxpy_kernel on the GPU
    saxpy_kernel<<<numBlocks,threadsPerBlock>>>(total_elems, alpha, device_x, device_y, device_result);    


    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaDeviceSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    
    cudaError_t err_sync = cudaDeviceSynchronize();
    if(err_sync != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_sync),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
   
    double endGPUTime = CycleTimer::currentSeconds();
    timeKernelAvg += endGPUTime - startGPUTime;
    
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    
    //
    // TODO: copy result from GPU using cudaMemcpy
    //
    copyStart = CycleTimer::currentSeconds();
    cudaError_t err_rescpy = cudaMemcpy(resultarray,device_result,size,cudaMemcpyDeviceToHost);  
    if(err_rescpy != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_rescpy),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    timeCopyD2HAvg += CycleTimer::currentSeconds() - copyStart;
    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;

    //
    // TODO free memory buffers on the GPU
    //
    
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
