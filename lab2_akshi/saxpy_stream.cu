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
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

void getArrays(long size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
  cudaMallocHost((void**)xarray, size * sizeof(float));
  cudaMallocHost((void**)yarray, size * sizeof(float));
  cudaMallocHost((void**)resultarray, size * sizeof(float));

  //xarray = (float**)malloc(size*sizeof(float));
  //yarray = (float**)malloc(size*sizeof(float));
  //resultarray = (float**)malloc(size*sizeof(float));

}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
  cudaFreeHost(xarray);
  cudaFreeHost(yarray);
  cudaFreeHost(resultarray);
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
    cudaStream_t stream[partitions];

    long elemPerPartition = total_elems/partitions;
    for (int i=0; i<partitions; i++) { 
        cudaError_t err_str = cudaStreamCreate(&stream[i]);
        if(err_str != cudaSuccess) {
            fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_str),__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
    } 
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,0);
    for (int i=0; i<partitions; i++) {
  
        //
        // TODO: copy input arrays to the GPU using cudaMemcpy
        //
        
        double copyStart = CycleTimer::currentSeconds();
        cudaError_t err_xcpy = cudaMemcpyAsync(device_x + i*elemPerPartition,xarray + i*elemPerPartition,elemPerPartition*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
        if(err_xcpy != cudaSuccess) {
            fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_xcpy),__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }

        cudaError_t err_ycpy = cudaMemcpyAsync(device_y + i*elemPerPartition,yarray + i*elemPerPartition,elemPerPartition*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
        if(err_ycpy != cudaSuccess) {
            fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_ycpy),__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
        timeCopyH2DAvg += CycleTimer::currentSeconds() - copyStart;
        //
        // TODO: insert time here to begin timing only the kernel
        //
    
        // compute number of blocks and threads per block
        const int numBlocks = (elemPerPartition-1)/threadsPerBlock + 1; 

        double startGPUTime = CycleTimer::currentSeconds();
        saxpy_kernel<<<numBlocks,threadsPerBlock,0,stream[i]>>>(elemPerPartition, alpha, device_x + i*elemPerPartition, device_y + i*elemPerPartition, device_result + i*elemPerPartition);
    
        //
        // TODO: insert timer here to time only the kernel.  Since the
        // kernel will run asynchronously with the calling CPU thread, you
        // need to call cudaDeviceSynchronize() before your timer to
        // ensure the kernel running on the GPU has completed.  (Otherwise
        // you will incorrectly observe that almost no time elapses!)
        //
        // cudaDeviceSynchronize();
    
   
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
        cudaError_t err_rescpy = cudaMemcpyAsync(resultarray + i * elemPerPartition,device_result + i*elemPerPartition,elemPerPartition*sizeof(float),cudaMemcpyDeviceToHost,stream[i]);  
        if(err_rescpy != cudaSuccess) {
            fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_rescpy),__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
        timeCopyD2HAvg += CycleTimer::currentSeconds() - copyStart;
    }
    
   // for(int i = 0 ; i < partitions; i++) {
   //     cudaError_t err_sync = cudaStreamSynchronize(stream[i]);
   //     if(err_sync != cudaSuccess) {
   //         fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err_sync),__FILE__,__LINE__);
   //         exit(EXIT_FAILURE);
   //     }
   // }

    cudaDeviceSynchronize();

    float overallDuration;
    cudaEventRecord(end,0);  
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&overallDuration, start, end);

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    totalTimeAvg += overallDuration/1000.0;
    //double overallDuration = endTime - startTime;

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
