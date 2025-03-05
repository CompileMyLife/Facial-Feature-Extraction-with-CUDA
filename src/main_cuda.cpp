// main_cuda.cpp
#include "image.h"
#include "haar.h"
#include "cuda_detect.h"  // runDetection is declared here.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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
    printf("GPU Integral image summary: width = %d, height = %d, total values = %d\n", width, height, total);

    // Print the four corner values
    printf("Top-left (index 0): %d\n", img->data[0]);
    printf("Top-right (index %d): %d\n", width - 1, img->data[width - 1]);
    printf("Bottom-left (index %d): %d\n", (height - 1) * width, img->data[(height - 1) * width]);
    printf("Bottom-right (index %d): %d\n", total - 1, img->data[total - 1]);

    int step = total / numSamples;
    if (step < 1)
        step = 1;
    printf("Printing %d sample values (every %d-th value):\n", numSamples, step);
    for (int i = 0; i < total; i += step) {
        printf("Index %d: %d\n", i, img->data[i]);
    }
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

    // ***** NEW STEP: Allocate temporary buffers once using full original image dimensions *****
    // This mimics the CPU code which calls:
    //   createImage(img->width, img->height, img1);
    //   createSumImage(img->width, img->height, sum1);
    //   createSumImage(img->width, img->height, sqsum1);
    // We allocate buffers here and then re-set their dimensions inside the scale loop.
    MyImage scaledImg;
    createImage(image->width, image->height, &scaledImg);
    MyIntImage scaledSum, scaledSqSum;
    createSumImage(image->width, image->height, &scaledSum);
    createSumImage(image->width, image->height, &scaledSqSum);

    // ***** Now iterate over scales using these temporary buffers *****
    float factor = 1.0f;
    std::vector<MyRect> candidates;
    while (true) {
        iter_counter++;
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

        // Reset the temporary buffers to the new scaled dimensions.
        setImage(newWidth, newHeight, &scaledImg);
        setSumImage(newWidth, newHeight, &scaledSum);
        setSumImage(newWidth, newHeight, &scaledSqSum);

        // Downsample the original image into our temporary buffer.
        nearestNeighbor(image, &scaledImg);

        // Compute the integral images for the downsampled image.
        integralImages(&scaledImg, &scaledSum, &scaledSqSum);

        // (Optional debug print for a particular iteration.)
        if (iter_counter == 2) {
            printf("DEBUG: Scale iteration %d, factor = %.3f\n", iter_counter, factor);
            debugPrintIntegralImageGPU(&scaledSum, 10);
        }

        // Update the cascade with these integral images.
        setImageForCascadeClassifier(cascade, &scaledSum, &scaledSqSum);
        printf("detecting faces, iter := %d\n", iter_counter);

        // Process this scale with the cascade filter.
        ScaleImage_Invoker(cascade, factor, scaledSum.height, scaledSum.width, candidates);

        factor *= 1.2f;
    }

    // Optionally, perform grouping on the candidates.
    if (!candidates.empty()) {
        groupRectangles(candidates, minSize.width, 0.4f);
    }

    // Free temporary buffers.
    freeImage(&scaledImg);
    freeSumImage(&scaledSum);
    freeSumImage(&scaledSqSum);

    // ***** Now run CUDA detection using the same cascade and (last computed) scaled integral images.
    printf("Detecting faces at fixed scale using CUDA...\n");
    std::vector<MyRect> gpuCandidates = runDetection(&scaledSum, &scaledSqSum, cascade, 10000000, factor, adjusted_width, adjusted_height);
    printf("CUDA detection detected %zu candidates.\n", gpuCandidates.size());
    for (size_t i = 0; i < gpuCandidates.size(); i++) {
        printf("[DEBUG] CUDA Candidate %zu: x=%d, y=%d, width=%d, height=%d\n",
            i, gpuCandidates[i].x, gpuCandidates[i].y, gpuCandidates[i].width, gpuCandidates[i].height);
    }

    // 8. Draw candidate face boxes on the original image.
    for (size_t i = 0; i < gpuCandidates.size(); i++) {
        drawRectangle(image, gpuCandidates[i]);
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

    return 0;
}