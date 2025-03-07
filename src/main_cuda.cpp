// main_cuda.cpp
#include "image_cuda.h"
#include "haar_cuda.h"
#include "cuda_detect.h"  // runDetection is declared here.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

//#define FINAL_DEBUG

extern int iter_counter;
inline int myRound(float value);  // Prototype for the inline function.
void ScaleImage_Invoker(myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);


#define INPUT_FILENAME "Face.pgm"
#define OUTPUT_FILENAME "Output.pgm"

extern void nearestNeighbor(MyImage* src, MyImage* dst);

// Helper function to scan classifier data and compute the extra offsets (in x and y)
// that account for all rectangle extents beyond the base detection window.
void computeExtraOffsets(const myCascade* cascade, int* extra_x, int* extra_y) {
    *extra_x = 0;
    *extra_y = 0;
    int totalRectElems = cascade->total_nodes * 12;
    for (int i = 0; i < totalRectElems; i += 4) {
        int rx = cascade->rectangles_array[i];
        int ry = cascade->rectangles_array[i + 1];
        int rw = cascade->rectangles_array[i + 2];
        int rh = cascade->rectangles_array[i + 3];
        if (rx == 0 && ry == 0 && rw == 0 && rh == 0)
            continue;
        int current_right = rx + rw;
        int current_bottom = ry + rh;
        if (current_right > *extra_x)
            *extra_x = current_right;
        if (current_bottom > *extra_y)
            *extra_y = current_bottom;
    }
}

// Helper function to print a subset of values from an integral image.
void debugPrintIntegralImageGPU(MyIntImage* img, int numSamples) {
    int width = img->width;
    int height = img->height;
    int total = width * height;

#ifdef Final_DEBUG
    printf("GPU Integral image summary: width = %d, height = %d, total values = %d\n", width, height, total);
#endif

#ifdef Final_DEBUG
    // Print the four corner values
    printf("Top-left (index 0): %d\n", img->data[0]);
    printf("Top-right (index %d): %d\n", width - 1, img->data[width - 1]);
    printf("Bottom-left (index %d): %d\n", (height - 1) * width, img->data[(height - 1) * width]);
    printf("Bottom-right (index %d): %d\n", total - 1, img->data[total - 1]);
#endif

    int step = total / numSamples;
    if (step < 1)
        step = 1;
#ifdef Final_DEBUG
    printf("Printing %d sample values (every %d-th value):\n", numSamples, step);
    for (int i = 0; i < total; i += step) {
        printf("Index %d: %d\n", i, img->data[i]);
    }
#endif
}



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

    // Save a test copy of the input image.
    printf("-- saving test copy of input image --\n");
    if (writePgm("TestOutput.pgm", image) == -1)
        printf("Unable to save test output image\n");
    else
        printf("-- test image saved as TestOutput.pgm --\n");

    // 2. Compute integral images for the original image.
    MyIntImage sumObj, sqsumObj;
    MyIntImage* sum = &sumObj;
    MyIntImage* sqsum = &sqsumObj;
    createSumImage(image->width, image->height, sum);
    createSumImage(image->width, image->height, sqsum);
    integralImages(image, sum, sqsum);

    // 3. Initialize the cascade classifier.
    myCascade cascadeObj;
    myCascade* cascade = &cascadeObj;
    cascade->n_stages = 25;
    cascade->total_nodes = 2913;
    cascade->orig_window_size.width = 24;
    cascade->orig_window_size.height = 24;
    MySize minSize = { 20, 20 };
    MySize maxSize = { 0, 0 };

    printf("-- loading cascade classifier --\n");
    readTextClassifier(cascade);
    printf("-- cascade classifier loaded --\n");

    if (cascade->scaled_rectangles_array == NULL) {
        printf("ERROR: cascade->scaled_rectangles_array is NULL after readTextClassifier!\n");
    }
    else {
        printf("cascade->scaled_rectangles_array is NOT NULL after readTextClassifier: %p\n", cascade->scaled_rectangles_array);
    }

    // 4. Link integral images to the cascade.
    printf("-- linking integral images to cascade --\n");
    setImageForCascadeClassifier(cascade, sum, sqsum);
    printf("-- integral images linked to cascade --\n");

    // Compute extra offsets.
    int extra_x = 0, extra_y = 0;
    computeExtraOffsets(cascade, &extra_x, &extra_y);
    printf("Computed extra offsets: extra_x = %d, extra_y = %d\n", extra_x, extra_y);

    // Adjust the detection window size.
    int adjusted_width = cascade->orig_window_size.width + extra_x;
    int adjusted_height = cascade->orig_window_size.height + extra_y;
    printf("Adjusted detection window size: (%d, %d)\n", adjusted_width, adjusted_height);

    // Allocate buffers
    MyImage scaledImg;
    createImage(image->width, image->height, &scaledImg);
    MyIntImage scaledSum, scaledSqSum;
    createSumImage(image->width, image->height, &scaledSum);
    createSumImage(image->width, image->height, &scaledSqSum);
    float factor = 1.0f;

// ***** Run CUDA detection at each scale in the pyramid *****
// Here we demonstrate running CUDA detection for each valid scale.
// We'll loop over the scales again, performing CUDA detection and merging results.
    std::vector<MyRect> allGpuCandidates;
    int iter_counter = 1;
    factor = 1.0f;

    // START TIMER
    while (true) {
        int newWidth = (int)(image->width / factor);
        int newHeight = (int)(image->height / factor);
        int winWidth = myRound(cascade->orig_window_size.width * factor);
        int winHeight = myRound(cascade->orig_window_size.height * factor);
        MySize sz = { newWidth, newHeight };
        MySize winSize = { winWidth, winHeight };
        MySize diff = { sz.width - cascade->orig_window_size.width, sz.height - cascade->orig_window_size.height };

        if (diff.width < 0 || diff.height < 0)
            break;
        if (winSize.width < minSize.width || winSize.height < minSize.height) {
            factor *= 1.2f;
            continue;
        }

        // Reallocate buffers for this scale in the CUDA loop.
        freeImage(&scaledImg);
        freeSumImage(&scaledSum);
        freeSumImage(&scaledSqSum);
        createImage(newWidth, newHeight, &scaledImg);
        createSumImage(newWidth, newHeight, &scaledSum);
        createSumImage(newWidth, newHeight, &scaledSqSum);

        nearestNeighbor(image, &scaledImg);
        integralImages(&scaledImg, &scaledSum, &scaledSqSum);
        setImageForCascadeClassifier(cascade, &scaledSum, &scaledSqSum);

        // Check if the detection window fits in the scaled integral image.
        if (factor * (cascade->orig_window_size.width + extra_x) < scaledSum.width &&
            factor * (cascade->orig_window_size.height + extra_y) < scaledSum.height) {
            std::vector<MyRect> gpuCandidates = runDetection(&scaledSum, &scaledSqSum, cascade, 10000000, factor, adjusted_width, adjusted_height, iter_counter);
            // Merge candidates from this scale.
            allGpuCandidates.insert(allGpuCandidates.end(), gpuCandidates.begin(), gpuCandidates.end());
        }
        else {
            //printf("Scale factor %f too high for valid detection window at this iteration.\n", factor);
        }
        factor *= 1.2f;
    }


    printf("CUDA detection detected %zu candidates.\n", allGpuCandidates.size());
    for (size_t i = 0; i < allGpuCandidates.size(); i++) {
        printf("[DEBUG] CUDA Candidate %zu: x=%d, y=%d, width=%d, height=%d\n",
            i, allGpuCandidates[i].x, allGpuCandidates[i].y, allGpuCandidates[i].width, allGpuCandidates[i].height);
    }

    // 8. Draw candidate face boxes on the original image.
    for (size_t i = 0; i < allGpuCandidates.size(); i++) {
        drawRectangle(image, allGpuCandidates[i]);
    }

    // 9. Save the output image.
    printf("-- saving output --\n");
    int flag = writePgm(OUTPUT_FILENAME, image);
    if (flag == -1)
        printf("Unable to save output image\n");
    else
        printf("-- image saved as %s --\n", OUTPUT_FILENAME);

    // 10. Clean up.
    releaseTextClassifier(cascade);
    freeImage(image);
    freeSumImage(sum);
    freeSumImage(sqsum);


    // Free temporary buffers.
    freeImage(&scaledImg);
    freeSumImage(&scaledSum);
    freeSumImage(&scaledSqSum);

    return 0;
}