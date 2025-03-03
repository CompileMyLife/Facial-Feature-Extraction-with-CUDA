

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

    // 2. Compute integral images. (CPU computation)
    MyIntImage sumImage, sqsumImage;
    createSumImage(image->width, image->height, &sumImage);
    createSumImage(image->width, image->height, &sqsumImage);
    integralImages(image, &sumImage, &sqsumImage);

    // 3. Initialize the cascade classifier.
    myCascade cascade;
    readTextClassifier(&cascade); // Load classifier parameters from file

    // 4. Link integral images to the cascade.
    setImageForCascadeClassifier(&cascade, &sumImage, &sqsumImage);

    // (Optional) Load cascade classifier data to CUDA device memory.
    loadCascadeClassifierCUDA(&cascade);

    // 5. Precompute candidate windows on the host.
    printf("-- generating candidate windows on host --\n");
    float scaleFactor = 1.2f;
    // Assume a default detection window size of 20x20 (or use cascade.orig_window_size)
    MySize winSize = { 20, 20 };
    std::vector<MyRect> candidateWindows;

    // For each scale level, compute scaled window size and slide across the image.
    for (float scale = 1.0f; ; scale *= scaleFactor) {
        int scaledWidth = static_cast<int>(winSize.width * scale);
        int scaledHeight = static_cast<int>(winSize.height * scale);
        if (scaledWidth > image->width || scaledHeight > image->height)
            break;  // Stop when the scaled window exceeds image dimensions.

        // Use a stride of half the window dimensions for this scale.
        int stepX = scaledWidth / 2;
        int stepY = scaledHeight / 2;
        for (int y = 0; y <= image->height - scaledHeight; y += stepY) {
            for (int x = 0; x <= image->width - scaledWidth; x += stepX) {
                MyRect r;
                r.x = x;
                r.y = y;
                r.width = scaledWidth;
                r.height = scaledHeight;
                candidateWindows.push_back(r);
            }
        }
    }
    printf("Generated %lu candidate windows.\n", candidateWindows.size());

    // 6. Run CUDA detection.
    // The function runDetection will process the candidate windows on the GPU.
    std::vector<MyRect> detections = runDetection(candidateWindows);

    // 7. Draw detected face boxes on the image.
    for (const MyRect& r : detections) {
        // drawRectangle is assumed to modify 'image' in-place.
        drawRectangle(image, r);
    }

    // 8. Save the output image.
    if (writePgm(OUTPUT_FILENAME, image) == -1)
        printf("Unable to save output image\n");
    else
        printf("-- output image saved as %s --\n", OUTPUT_FILENAME);

    // 9. Clean up resources.
    // If you have a function to release the cascade classifier, call it.
    releaseTextClassifier();  // if defined

    // Optionally, reset the CUDA device.
    cudaDeviceReset();

    return 0;
}
