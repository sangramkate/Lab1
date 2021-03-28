#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"


class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRedColor;
    float* cudaDeviceBlueColor;
    float* cudaDeviceGreenColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    int* circleBlockArray;
    int* circlePerBlock;
    int* circleBlockIdx;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadeRedPixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);

    void shadeBluePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
    void shadeGreenPixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
    void shadeAlphaPixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
    void circleInBoxTest(int numCircles, int numBlocks, int blocksPerRow);

};


#endif
