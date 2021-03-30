#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "circleBoxTest.cu_inl"

#include "thrust/thrust/thrust/copy.h"
//#include "thrust/thrust/thrust/reduce.h"
//#include "thrust/thrust/thrust/execution_policy.h"
#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//Predicate functor
struct is_not_zero
{
    __host__  __device__
        bool operator()(const int x)
    {
        return (x != 0);
    }
};

__global__ void
update_result_arr(int N, int* output) {
    output[N-1] = 0; 
}
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
////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // This conditional is in the inner loop, but it evaluates the
    // same direction for all threads so it's cost is not so
    // bad. Attempting to hoist this conditional is not a required
    // student optimization in Assignment 2
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.

__global__ void  checkCircleBlockPair(int* circleBlockArray,int numCircles, int numBlocks,  int blocksPerRow){

  int circleId = blockIdx.x * blockDim.x + threadIdx.x;
  int block = blockIdx.y * blockDim.y + threadIdx.y;
  if((circleId < numCircles) && (block < numBlocks)){
      int index3 = circleId * 3;
      float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
      float rad = cuConstRendererParams.radius[circleId];
      circleBlockArray[(block * numCircles) + circleId] = circleInBoxConservative(p.x, p.y, rad, 
          static_cast<float>(1.f/blocksPerRow)*block, static_cast<float>(1.f/blocksPerRow)*(block+1), 
          static_cast<float>(1.f/blocksPerRow)*(blockIdx.y+1), static_cast<float>(1.f/blocksPerRow)*(blockIdx.y));
     // printf("circleBlockArray[%d]=%d\n",(block * numCircles) + circleId,circleBlockArray[(block * numCircles) + circleId]);
  }
}

__global__ void
upsweep_kernel(int N, int twod, int twod1, int* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((index < N) && ((index % twod1) == 0)) {
        output[index + twod1 - 1] += output[index + twod -1];    
    }
}


__global__ void
downsweep_kernel(int N, int twod, int twod1, int* output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < N) && ((index % twod1) == 0)) {
        int t = output[index + twod - 1];
         output[index+twod-1] = output[index+twod1-1];
         output[index+twod1-1] += t; // change twod1 to twod to reverse prefix sum.
    }
}

__global__ void
upsweep_small_kernel(int N, int* output) {
    int index = threadIdx.x;

    int num_threads = 1024;
    for(int i=N/1024; i<N; i*=2) {
        if(index < num_threads) {
            output[i*index + i - 1] += output[i*index + i/2 - 1];
        }
        num_threads /= 2;
        __syncthreads();
    }
}

__global__ void print_kernel(int* output, int N) {
	for(int i=0; i< N; i++) {
		printf("output[%d] = %d\n", i, output[i]);
	}
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    const int threadsPerBlock = 256; // change this if necessary
    int N = nextPow2(length);
    gpuErrchk(cudaDeviceSynchronize());
    //print_kernel<<<1,1>>>(device_result, N);
    for(int twod=1; twod<N; twod*=2)  {
        int twod1 = twod*2;
        //printf("twod=%d\n",twod);
        upsweep_kernel<<<(N + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(N, twod, twod1, device_result); 
        gpuErrchk(cudaDeviceSynchronize());
    }

    //upsweep_small_kernel<<<1, 1024>>>(N, device_result); 
    gpuErrchk(cudaDeviceSynchronize());

    //device_result[N-1] = 0;
    update_result_arr<<<1, 1>>>(N, device_result);
    gpuErrchk(cudaDeviceSynchronize());


    for(int twod=N/2; twod >=1; twod/=2) {
        int twod1 = twod*2 ;
        downsweep_kernel<<<(N + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(N, twod, twod1, device_result); 
        gpuErrchk(cudaDeviceSynchronize());
    }

    //gpuErrchk(cudaDeviceSynchronize());
    //print_kernel<<<1,1>>>(device_result, N);

}


__global__ void
process_repeat_kernel(int N, int* output, int* predicate) {

    // compute overall index from dev_offsetition of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
//	printf("predicate[%d]=%d\n", index, predicate[index]);
        if(predicate[index] != predicate[index + 1]) {
            output[predicate[index]] = index;
        }
    }
}

void cudaScan(int* inarray, int length, int* resultarray)
{
    int rounded_length = nextPow2(length);
    int threadsPerBlock = 256;
    int* device_out = NULL;
    int* in_array = NULL;

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMalloc(&in_array,rounded_length * sizeof(int)));
    gpuErrchk(cudaMemset(in_array,0,rounded_length * sizeof(int)));
    gpuErrchk(cudaMemcpy(in_array,inarray,sizeof(int)*length,cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMalloc(&device_out,rounded_length * sizeof(int)));
    gpuErrchk(cudaMemset(device_out,0,rounded_length * sizeof(int)));
    exclusive_scan(in_array, length, inarray);
    // Wait for any work left over to be completed.
    gpuErrchk(cudaDeviceSynchronize());
    process_repeat_kernel<<<(rounded_length + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(rounded_length, resultarray, inarray);
    cudaFree(in_array);
    cudaFree(device_out);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void sum(int* input)
{
	const int tid = threadIdx.x;

	auto step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
	}
}

__global__ void copyVal (int* outputArray, int* buffer, int length, int* pairs){
  
  *pairs = length;
  for(int i=0;i<length;i++){
    if(buffer[i]== -1){
      *pairs = i;
      break;
    }
    else{
      outputArray[i] = buffer[i];
    }
  }
  
  for(int i=0;i<length;i++){
      printf("--------------- pairs = %d\n",*pairs);
      printf("---------------buffer = %d\n",buffer[i]);
      printf("---------------output array = %d\n",outputArray[i]);
  }
}

int fillPairs ( int* predicateArray, int length, int* outputArray){
  int* buffer;
  int* input_buff;
  gpuErrchk(cudaDeviceSynchronize());
  cudaMalloc(&input_buff,length * sizeof(int));
  cudaMemcpy(input_buff,predicateArray,sizeof(int)*length,cudaMemcpyDeviceToDevice);
  cudaMalloc(&buffer,length * sizeof(int));
  cudaMemset(buffer,-1,length * sizeof(int));
  
  int* pairs;
  cudaMallocManaged(&pairs,sizeof(int));
  cudaScan(input_buff,length,buffer);
  gpuErrchk(cudaDeviceSynchronize());
  copyVal<<<1,1>>>(outputArray,buffer,length,pairs);
  cudaDeviceSynchronize();
  cudaFree(input_buff);
  cudaFree(buffer);
  gpuErrchk(cudaDeviceSynchronize());
  return *pairs; 
}

__global__ void  createIndexArray(int* circleBlockArray,int numCircles, int numBlocks){

  int circleId = blockIdx.x * blockDim.x + threadIdx.x;
  int block = blockIdx.y * blockDim.y + threadIdx.y;
  if((circleId < numCircles) && (block < numBlocks)){
        if(circleBlockArray[(block*numCircles)+ circleId] == 1)
            circleBlockArray[(block*numCircles)+ circleId] = circleId;
    } 
}

void 
CudaRenderer::circleInBoxTest(int numCircles, int numBlocks, int blocksPerRow){
   dim3 blockDim(blocksPerRow,blocksPerRow);
   dim3 gridDim(((numCircles +blockDim.x-1)/blockDim.x),((numBlocks +blockDim.y-1)/blockDim.y));

  //  thrust::device_ptr<int> d_circleBlockArray = thrust::device_malloc<int>(numBlocks * numCircles);
  //  thrust::device_ptr<int> d_circlePerBlock = thrust::device_malloc<int>(numBlocks);
  //  circleBlockArray = d_circleBlockArray.get();
   // circlePerBlock = d_circlePerBlock.get();
 
  // gpuErrchk(cudaMalloc(&circleBlockArray,sizeof(int)*numCircles*numBlocks));
    cudaMalloc(&circleBlockArray,sizeof(int)*(numBlocks*numCircles));
   
   int* host_circlePerBlock = (int*)malloc(sizeof(int)*(numBlocks+1));
   checkCircleBlockPair<<<gridDim,blockDim>>>(circleBlockArray, numCircles, numBlocks, blocksPerRow);
   gpuErrchk(cudaDeviceSynchronize());

   //thrust::device_ptr<int> t_circleBlockArray = thrust::device_pointer_cast(circleBlockArray);
   //thrust::device_ptr<int> t_circleBlockArray_end = thrust::device_pointer_cast(circleBlockArray );
   int * copy_array;
   cudaMalloc(&copy_array,sizeof(int)*(numBlocks*numCircles));
   cudaMemcpy(copy_array,circleBlockArray,sizeof(int)*(numBlocks*numCircles),cudaMemcpyDeviceToDevice);  
   sum<<< 1,numCircles*numBlocks/2 >>>(copy_array);
   gpuErrchk(cudaDeviceSynchronize());
   int* num = (int*) malloc(sizeof(int));
   cudaMemcpy(num,copy_array,sizeof(int),cudaMemcpyDeviceToHost);
   cudaFree(copy_array);
   
   int numPairs = *num;   //thrust::reduce(t_circleBlockArray, t_circleBlockArray+(numBlocks * numCircles),0);
   printf("numPairs=%d\n",numPairs);
  // int numPairs = thrust::reduce(d_circleBlockArray.get(), d_circleBlockArray.get()+(numBlocks * numCircles),0);
  // printf("numPairs= %d\n", numPairs); 
 
   //thrust::device_ptr<int> d_circleBlockIdx = thrust::device_malloc<int>(numPairs);
   
   cudaMalloc(&circleBlockIdx,sizeof(int)*numPairs);
   //cudaMalloc(&circleBlockIdx,sizeof(int)*numBlocks * numCircles);
    
   //createIndexArray<<<gridDim,blockDim>>>(circleBlockArray, numCircles, numBlocks);
   //gpuErrchk(cudaDeviceSynchronize());
  
 //  thrust::device_ptr<int> t_circlePerBlock(circlePerBlock);
  // thrust::device_vector<int> d_circlePerBlock(t_circlePerBlock, t_circlePerBlock + numBlocks+1);
  
   //thrust::device_ptr<int> d_res = thrust::device_malloc<int>(numCircles);

   for(int i=0; i < numBlocks; i++){
     if(i==0){

         //thrust::device_ptr<int> t_circleBlockArray = thrust::device_pointer_cast(circleBlockArray);
         //thrust::device_vector<int> d_src (t_circleBlockArray, t_circleBlockArray+ numCircles);

         gpuErrchk(cudaDeviceSynchronize());
         auto num = fillPairs(circleBlockArray,numCircles,circleBlockIdx);
         gpuErrchk(cudaDeviceSynchronize());
         //auto num = fillPairs(d_circleBlockArray.get(),numCircles,d_circleBlockIdx.get());
         //auto result_end = thrust::copy_if(d_circleBlockArray, d_circleBlockArray+numCircles, d_res , is_not_zero());
         //int* start_ptr = thrust::raw_pointer_cast(&d_res[0]);
         //int* end_ptr = thrust::raw_pointer_cast(&result_end[0]);
         //size_t num = (end_ptr - start_ptr);
         //size_t num = pairs; //(end_ptr - start_ptr+ 1);

         //thrust::device_ptr<int> t_circleBlockIdx(circleBlockIdx);
         //thrust::device_vector<int> d_circleBlockIdx(t_circleBlockIdx, t_circleBlockIdx + num);
         
         //thrust::copy(d_res,result_end,d_circleBlockIdx);
         printf("num=%d\n",num);
         host_circlePerBlock[0] =  0;
         host_circlePerBlock[1] =  num;
         printf("num=%d\n",num);
     }
     else{
         //thrust::device_ptr<int> t_circleBlockArray = thrust::device_pointer_cast(circleBlockArray+(i*numCircles));
         //thrust::device_vector<int> d_src(t_circleBlockArray, t_circleBlockArray+ numCircles);

         printf("goint into else\n");
         gpuErrchk(cudaDeviceSynchronize());
         auto num = fillPairs(circleBlockArray+(i*numCircles),numCircles,circleBlockIdx+host_circlePerBlock[i]);
         //auto result_end = thrust::copy_if(d_circleBlockArray+(i*numCircles), d_circleBlockArray+((i+1)*numCircles), d_res , is_not_zero());
         //int* start_ptr = thrust::raw_pointer_cast(&d_res[0]);
         //int* end_ptr = thrust::raw_pointer_cast(&result_end[0]);
         //size_t num = (end_ptr - start_ptr);
         //size_t num =  1; //(end_ptr - start_ptr+ 1);

         //thrust::device_ptr<int> t_circleBlockIdx(circleBlockIdx + host_circlePerBlock[i]);
         //thrust::device_vector<int> d_circleBlockIdx(t_circleBlockIdx, t_circleBlockIdx+ num);

         //thrust::copy(d_res,result_end,d_circleBlockIdx);
         printf("num=%d\n",num);
         host_circlePerBlock[i+1] = host_circlePerBlock[i] + num;
     }
   }
  // cudaMemcpy(host_circlePerBlock,circlePerBlock,sizeof(int)*(numBlocks+1),cudaMemcpyDeviceToHost);
    printf("number of circles in boxes:\n");
    for(int i=0; i<numBlocks+1;i++){
       printf(" box[%d]= %d \n",i,host_circlePerBlock[i]);
       //printf(" d_box[%d][%d]= %d \n",(i/blocksPerRow),(i%blocksPerRow),d_circlePerBlock[i]);
    }
}

__global__ void kernelRenderCircles(int imageWidth, int imageHeight, int* circleBlockIdx, int* device_circlePerBlock, int numBlocks) {
    //TODO: convert short to int
    //TODO: can direct get width from const params
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    int blockNum = blockIdx.y * blockDim.x + blockIdx.x;
    for (int index = device_circlePerBlock[blockNum]; index < device_circlePerBlock[blockNum+1]; index++) {
        int circleIndex = circleBlockIdx[index];
        int index3 = 3 * circleIndex;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    //    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;

        //float rad = cuConstRendererParams.radius[circleIndex];
        // BlockDim = 256 x1, gridDim = 4x4

        /*
        int circleInBox = circleInBoxConservative(p.x, p.y, rad, 
                static_cast<float>(1.f/gridDim.x)*blockIdx.x, static_cast<float>(1.f/gridDim.x)*(blockIdx.x+1), 
                static_cast<float>(1.f/gridDim.y)*(blockIdx.y+1), static_cast<float>(1.f/gridDim.y)*(blockIdx.y));
        */
        /*
        if((threadIdx.x + threadIdx.y)== 0) {
            printf("Blk : %dx%d, grid: %d %d\n", blockIdx.x, blockIdx.y, gridDim.x, circleInBox);
            printf("circleInBoxConservative p.x : %f, p.y : %f , rad : %f, %f, %f, %f, %f\n",
                p.x, p.y, rad, 
                static_cast<float>(1.f/gridDim.x)*blockIdx.x, static_cast<float>(1.f/gridDim.x)*(blockIdx.x+1), 
                static_cast<float>(1.f/gridDim.y)*(blockIdx.y+1), static_cast<float>(1.f/gridDim.y)*(blockIdx.y));
        }
        */
        /*
        __syncthreads(); //TODO: is this even needed? --- but why? 
        if(circleInBox == 0) { continue; }
        */
        /*
        if((threadIdx.x + threadIdx.y)== 0) {
            printf("Blk : %d, grid: %d\n", blockIdx.x, gridDim.x);
        }*/

        for(int tid_x = threadIdx.x ; tid_x < (imageWidth/gridDim.x); tid_x +=blockDim.x) {
            for(int tid_y = threadIdx.y ; tid_y < (imageHeight/gridDim.y); tid_y +=blockDim.y) {
                int x = blockIdx.x*(imageWidth/gridDim.x) + tid_x;  
                int y = blockIdx.y*(imageHeight/gridDim.y) + tid_y;  
                float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (y * imageWidth + x)]);
                float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(x) + 0.5f),
                                                             invHeight * (static_cast<float>(y) + 0.5f));
                shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        free(host_circlePerBlock);
        cudaFree(circleBlockIdx);
        cudaFree(device_circlePerBlock);
        cudaFree(circleBlockArray);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
    //dim3 blockDim(256, 1);
    //dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);


    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = image->width;
    short imageHeight = image->height;
    int blocksPerDim;
    if(numCircles < 10)
       blocksPerDim = 2;
    else if(numCircles < 20000)
       blocksPerDim = 4;
    else if(numCircles < 200000)
       blocksPerDim = 8;
    else 
       blocksPerDim = 16;

    int numBlocks = blocksPerDim * blocksPerDim;
   
    circleInBoxTest(numCircles,numBlocks,blocksPerDim);
    
    dim3 blockDim(16, 16);
   // dim3 gridDim((numPixels  + blockDim.x - 1) / blockDim.x);
    dim3 gridDim(blocksPerDim,blocksPerDim); //dividing it into block -- each block working on a portion of image
   
   
    cudaMalloc(&device_circlePerBlock,sizeof(int)*(numBlocks+1));
    cudaMemcpy(device_circlePerBlock,host_circlePerBlock,sizeof(int)*(numBlocks+1),cudaMemcpyHostToDevice);
   
    kernelRenderCircles<<<gridDim, blockDim>>>(imageWidth, imageHeight,circleBlockIdx, device_circlePerBlock,numBlocks);
    gpuErrchk(cudaDeviceSynchronize());
    //free(host_circlePerBlock);
    //cudaFree(circleBlockIdx);
    //cudaFree(device_circlePerBlock);
    //cudaFree(circleBlockArray);
}


