#include "image.h"
#include "haar.h"
#include "cuda_detect.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define INPUT_FILENAME "Face.pgm"
#define OUTPUT_FILENAME "Output.pgm"

int main() {
    printf("-- entering main function --\n");

    // Check CUDA device availability.
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found or CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    printf("Found %d CUDA device(s).\n", deviceCount);

    // 1. Load the input image.
    MyImage imageObj;
    MyImage* image = &imageObj;
    if (readPgm(INPUT_FILENAME, image) == -1) {
        printf("Unable to open input image\n");
        return 1;
    }

    // (Optional) Save a test copy of the input image.
    printf("-- saving test copy of input image --\n");
    if (writePgm("TestOutput.pgm", image) == -1)
        printf("Unable to save test output image\n");
    else
        printf("-- test image saved as TestOutput.pgm --\n");

    // 2. Compute integral images. (CPU for now)
    printf("-- computing integral images (CPU) --\n");
    MyIntImage sumObj, sqsumObj;
    MyIntImage* sum = &sumObj, * sqsum = &sqsumObj;
    setSumImage(image->width, image->height, sum);
    setSumImage(image->width, image->height, sqsum);
    integralImages(image, sum, sqsum);
    printf("-- integral images computed --\n");


    // 3. Initialize the cascade classifier.
    printf("-- initializing cascade classifier (CPU) --\n");
    myCascade cascadeObj;
    myCascade* cascade = &cascadeObj;
    readTextClassifier(); // This reads classifier data into CPU global arrays
    loadCascadeClassifierCUDA(cascade); // Load classifier data to GPU
    printf("-- cascade classifier initialized and loaded to GPU --\n");


    // 4. Link integral images to the cascade. (CPU pointers for now)
    printf("-- linking integral images to cascade (CPU pointers) --\n");
    setImageForCascadeClassifier(cascade, sum, sqsum); // Original CPU function
    setImageForCascadeClassifierCUDA(cascade, sum, sqsum); // Call CUDA version (placeholder for now)
    printf("-- integral images linked to cascade --\n");


    // 5. Run CUDA detection. (Commented out for now)
    // printf("-- running CUDA detection --\n");
    // std::vector<MyRect> faces = detectObjectsCUDA(image, minSize, maxSize, cascade, scale_factor, min_neighbors);
    // printf("-- CUDA detection complete --\n");

    // 6. Draw detected face boxes on the image. (Commented out for now)
    // printf("-- drawing face boxes --\n");
    // for (const auto& faceRect : faces) {
    //     drawRectangle(image, faceRect);
    // }
    // printf("-- face boxes drawn --\n");


    // 7. Save the output image. (Commented out for now)
    printf("-- saving output image --\n");
    if (writePgm(OUTPUT_FILENAME, image) == -1)
        printf("Unable to save output image\n");
    else
        printf("-- output image saved as Output.pgm --\n");

    // 8. Clean up.
    printf("-- cleaning up --\n");
    freeImage(image);
    freeSumImage(sum);
    freeSumImage(sqsum);
    releaseTextClassifier();
    printf("-- cleanup complete --\n");

    printf("-- exiting main function --\n");
    return 0;
}