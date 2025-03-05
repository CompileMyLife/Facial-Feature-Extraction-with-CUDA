// main_cuda.cpp
#include "image.h"
#include "haar.h"
#include "cuda_detect.h"  // runDetection is declared here.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define INPUT_FILENAME "Face.pgm"
#define OUTPUT_FILENAME "Output.pgm"

extern void nearestNeighbor(MyImage* src, MyImage* dst);

// Helper function to scan classifier data and compute the extra offsets (in x and y)
// that account for all rectangle extents beyond the base detection window.
void computeExtraOffsets(const myCascade* cascade, int* extra_x, int* extra_y) {
    *extra_x = 0;
    *extra_y = 0;
    // Each feature consists of 3 rectangles, and each rectangle is represented by 4 integers:
    // [x_offset, y_offset, width, height]
    // Thus, the total number of rectangle entries is total_nodes * 12.
    int totalRectElems = cascade->total_nodes * 12;
    for (int i = 0; i < totalRectElems; i += 4) {
        int rx = cascade->rectangles_array[i];
        int ry = cascade->rectangles_array[i + 1];
        int rw = cascade->rectangles_array[i + 2];
        int rh = cascade->rectangles_array[i + 3];
        // Skip "unused" rectangles (all zero)
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

    // 2. Compute integral images for the original image (used only for cascade initialization).
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
    // (You can adjust minSize/maxSize if needed)
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

    // 4. Link integral images to the cascade (for initial setup).
    printf("-- linking integral images to cascade --\n");
    setImageForCascadeClassifier(cascade, sum, sqsum);
    printf("-- integral images linked to cascade --\n");

    // Compute extra offsets based on the classifier's rectangle data.
    int extra_x = 0, extra_y = 0;
    computeExtraOffsets(cascade, &extra_x, &extra_y);
    printf("Computed extra offsets: extra_x = %d, extra_y = %d\n", extra_x, extra_y);

    // Adjust the detection window size by adding the extra offsets.
    int adjusted_width = cascade->orig_window_size.width + extra_x;
    int adjusted_height = cascade->orig_window_size.height + extra_y;
    printf("Adjusted detection window size: (%d, %d)\n", adjusted_width, adjusted_height);

    // 5. Run multi-scale CUDA detection.
    // We'll accumulate candidates from each scale.
    std::vector<MyRect> allCandidates;

    float scaleFactor = 1.2f;  // Scale increment factor
    int maxCandidates = 1000000000;  // Adjust as needed

    // Start with factor = 1, then multiply by scaleFactor each iteration.
    // Loop until the downsampled image is too small for detection.
    float factor = 1.0f;
    int iter_counter = 0;
    while (true) {
        iter_counter++;
        // Compute new window size for current scale.
        int winWidth = (int)(cascade->orig_window_size.width * factor + 0.5f);
        int winHeight = (int)(cascade->orig_window_size.height * factor + 0.5f);

        // Compute scaled image size.
        int scaledWidth = (int)(image->width / factor);
        int scaledHeight = (int)(image->height / factor);

        // If the scaled image is too small for even one detection window, break.
        if (scaledWidth < cascade->orig_window_size.width || scaledHeight < cascade->orig_window_size.height)
            break;

        // Skip scales where the detection window is smaller than minSize.
        if (winWidth < minSize.width || winHeight < minSize.height) {
            factor *= scaleFactor;
            continue;
        }

        printf("Scale iter %d: factor = %.3f, scaled image size = (%d, %d), detection window = (%d, %d)\n",
            iter_counter, factor, scaledWidth, scaledHeight, winWidth, winHeight);

        // Allocate temporary scaled image.
        MyImage scaledImg;
        createImage(scaledWidth, scaledHeight, &scaledImg);

        // Allocate temporary integral images for the scaled image.
        MyIntImage scaledSum, scaledSqSum;
        createSumImage(scaledWidth, scaledHeight, &scaledSum);
        createSumImage(scaledWidth, scaledHeight, &scaledSqSum);

        // Downsample the original image to the current scale.
        nearestNeighbor(image, &scaledImg);

        // Compute integral images for the scaled image.
        integralImages(&scaledImg, &scaledSum, &scaledSqSum);

        // Link the scaled integral images to the cascade.
        setImageForCascadeClassifier(cascade, &scaledSum, &scaledSqSum);

        printf("Detecting faces at scale factor %.3f...\n", factor);
        // Run CUDA detection at this scale.
        std::vector<MyRect> candidates = runDetection(&scaledSum, &scaledSqSum, cascade, maxCandidates, factor, adjusted_width, adjusted_height);
        printf("Scale factor %.3f detected %zu candidates.\n", factor, candidates.size());

        // Accumulate candidate windows.
        allCandidates.insert(allCandidates.end(), candidates.begin(), candidates.end());

        // Free temporary scaled images and integral images.
        freeImage(&scaledImg);
        freeSumImage(&scaledSum);
        freeSumImage(&scaledSqSum);

        // Update scale factor.
        factor *= scaleFactor;
    }

    // Optionally, perform grouping of overlapping candidates.
    if (!allCandidates.empty()) {
        // groupRectangles modifies the vector in place.
        groupRectangles(allCandidates, 1, 0.4f);
    }

    printf("-- face detection using CUDA complete --\n");
    printf("Total number of candidate faces detected: %zu\n", allCandidates.size());

    // 6. Draw detected face boxes on the image.
    for (size_t i = 0; i < allCandidates.size(); i++) {
        drawRectangle(image, allCandidates[i]);
    }

    // 7. Save the output image.
    printf("-- saving output --\n");
    int flag = writePgm(OUTPUT_FILENAME, image);
    if (flag == -1)
        printf("Unable to save output image\n");
    else
        printf("-- image saved as %s --\n", OUTPUT_FILENAME);

    // 8. Clean up.
    releaseTextClassifier(cascade);
    freeImage(image);
    freeSumImage(sum);
    freeSumImage(sqsum);

    return 0;
}
