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


    // 3. Initialize the cascade classifier.


    // 4. Link integral images to the cascade. 


    // 5. Run CUDA detection. 

    // 6. Draw detected face boxes on the image. (Commented out for now)



    // 7. Save the output image. (Commented out for now)


    // 8. Clean up.

}

