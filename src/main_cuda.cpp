#include "image.h"
#include "haar.h"
#include "cuda_detect.h"  // Make sure this header declares detectObjects()
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define INPUT_FILENAME "Face.pgm"
#define OUTPUT_FILENAME "Output.pgm"

int main() {
    printf("-- entering main function --\n");

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

    // 2. Compute integral images.
    MyIntImage sumObj, sqsumObj;
    MyIntImage* sum = &sumObj;
    MyIntImage* sqsum = &sqsumObj;
    createSumImage(image->width, image->height, sum);
    createSumImage(image->width, image->height, sqsum);
    integralImages(image, sum, sqsum);

    // 3. Initialize the cascade classifier.
    myCascade cascadeObj;
    myCascade* cascade = &cascadeObj;
    // Set classifier properties:
    cascade->n_stages = 25;
    cascade->total_nodes = 2913;
    cascade->orig_window_size.width = 24;
    cascade->orig_window_size.height = 24;
    MySize minSize = { 20, 20 };
    MySize maxSize = { 0, 0 };

    printf("-- loading cascade classifier --\n");
    readTextClassifier();

    // 4. Link integral images to the cascade.
    setImageForCascadeClassifier(cascade, sum, sqsum);

    // 5. Run detection (using detectObjects which returns a vector of rectangles).
    float scaleFactor = 1.2f;
    int minNeighbours = 1;
    printf("-- detecting faces --\n");
    std::vector<MyRect> result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);
    printf("Number of detected faces: %zu\n", result.size());

    // 6. Draw detected face boxes on the image.
    for (size_t i = 0; i < result.size(); i++) {
        MyRect r = result[i];
        drawRectangle(image, r);
    }

    // 7. Save the output image.
    printf("-- saving output --\n");
    int flag = writePgm(OUTPUT_FILENAME, image);
    if (flag == -1)
        printf("Unable to save output image\n");
    else
        printf("-- image saved as %s --\n", OUTPUT_FILENAME);

    // 8. Clean up.
    releaseTextClassifier();
    freeImage(image);

    return 0;
}
