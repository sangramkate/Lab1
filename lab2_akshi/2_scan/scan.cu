#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
__global__ void
upsweep_kernel(int length, int twod, int twod1, int* device_result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < length)  && (index%twod1 == 0)) {
        device_result[index + twod1 - 1] += device_result[index + twod - 1];
    }
    if(index == length-1) {
        device_result[index] = 0;
    }
}

__global__ void
downsweep_kernel(int length, int twod, int twod1, int* device_result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((index < length) && (index%twod1 == 0)) {
        int t = device_result[index + twod - 1];
        device_result[index + twod - 1] = device_result[index + twod1 - 1];
        device_result[index + twod1 -1] += t;
    }
}
__global__ void
print_kernel(int length, int* device_result) {
    for(int i = 0; i < length; i++) {
        printf("out[%d] = %d\n",i,device_result[i]);
    }
}
void exclusive_scan(int* device_start, int length, int* device_result)
{
    //I am doing inplace update on device_result
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
    const int threadsPerBlock = 512;
    const int numBlocks = (length-1)/threadsPerBlock + 1;
    for (int twod = 1; twod < length ; twod *=2 ) {
        int twod1 = twod*2;
        upsweep_kernel<<<numBlocks,threadsPerBlock>>>(length, twod, twod1,  device_result);
        cudaError_t err1 = cudaDeviceSynchronize();
        if(err1 != cudaSuccess) {
            fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err1),__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
    }
    for (int twod = length/2; twod >= 1; twod /= 2) {
        int twod1 = twod*2;
        downsweep_kernel<<<numBlocks, threadsPerBlock>>>(length, twod, twod1, device_result);
        cudaError_t err2 = cudaDeviceSynchronize();
        if(err2 != cudaSuccess) {
            fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err2),__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    //exclusive_scan(device_input, end - inarray, device_result);
    exclusive_scan(device_input, rounded_length, device_result);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void
predicate_kernel(int* device_input, int length, int* predicate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        if(device_input[index] == device_input[index+1]) {
            predicate[index] = 1;
        } else {
            predicate[index] = 0;
        }
    }
}
__global__ void
repeat_kernel(int* predicate, int length, int* device_output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length) {
        if(predicate[index] != predicate[index+1]) {
            device_output[predicate[index]] = index;
        }
    }
}
int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */    

    
    const int threadsPerBlock = 512;
    const int numBlocks = (length-1)/threadsPerBlock + 1;
    int *device_predicate;

    int rounded_length = nextPow2(length);
    cudaError_t err3 = cudaMalloc((void **)&device_predicate, sizeof(int) * rounded_length);
    if(err3 != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err3),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    predicate_kernel<<<numBlocks,threadsPerBlock>>>(device_input, length, device_predicate);
    
    cudaError_t err4 = cudaDeviceSynchronize();
    if(err4 != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString(err4),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    int rounded_length = nextPow2(length);
    exclusive_scan(device_input, rounded_length, device_predicate);
    repeat_kernel<<<numBlocks,threadsPerBlock>>>(device_predicate, length, device_output);
    int ret_len = device_predicate[length-1];
    cudaFree(device_predicate);
    return ret_len;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);
    //int result = find_repeats(device_input, rounded_length, device_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
