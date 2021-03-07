#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary  
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
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
    cudaMalloc(&device_x, total_elems * sizeof(float));
    cudaMalloc(&device_y, total_elems * sizeof(float));
    cudaMalloc(&device_result, total_elems * sizeof(float));
    
    const long NumBlocks = ((total_elems + threadsPerBlock-1)/threadsPerBlock);

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    const long  size = (total_elems/partitions);
    double startGPUTime[partitions];
    double endGPUTime[partitions];
    double timeKernel[partitions];
    double endD2HTime[partitions];
    double startH2DTime[partitions];    
    cudaStream_t streams[partitions];

    for (int i=0; i<100; i++) {
  
        //
        // TODO: copy input arrays to the GPU using cudaMemcpy
        //
            startH2DTime[i]  = CycleTimer::currentSeconds();
       
            cudaMemcpyAsync(device_x+ i * size * sizeof(float),xarray + i * size * sizeof(float), size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(device_y+ i * size * sizeof(float),yarray + i * size * sizeof(float), size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
             
            //
            // TODO: insert time here to begin timing only the kernel
            //
            startGPUTime[i] = CycleTimer::currentSeconds();
    
            // compute number of blocks and threads per block

            // run saxpy_kernel on the GPU
            saxpy_kernel<<<NumBlocks,threadsPerBlock,0,streams[i]>>>(total_elems,alpha,device_x,device_y,device_result);
    
            //
            // TODO: insert timer here to time only the kernel.  Since the
            // kernel will run asynchronously with the calling CPU thread, you
            // need to call cudaDeviceSynchronize() before your timer to
            // ensure the kernel running on the GPU has completed.  (Otherwise
            // you will incorrectly observe that almost no time elapses!)
            //
            cudaStreamSynchronize(streams[i]);
            endGPUTime[i] = CycleTimer::currentSeconds();
            timeKernel[i] = endGPUTime[i] - startGPUTime[i];
    
            cudaError_t errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
            }
    
            //
            // TODO: copy result from GPU using cudaMemcpy
            //
            cudaMemcpyAsync(resultarray+ i * size * sizeof(float),device_result+ i * size * sizeof(float),size * sizeof(float), cudaMemcpyDeviceToHost,streams[i]);
    
            endD2HTime[i] = CycleTimer::currentSeconds();
    }

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    
    for(int j =0 ; j < partitions ; j++){
        timeKernelAvg  += timeKernel[j];
        timeCopyD2HAvg += endD2HTime[j] - endGPUTime[j];
        timeCopyH2DAvg += startGPUTime[j] - startH2DTime[j];
    }

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
