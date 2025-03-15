/**
 * @file main.cpp
 * @brief Main function for CUDA-based image detection.
 *
 * Date: 03.12.25
 * Authors: Ibrahim Binmahfood, Kunjan Vyas, Robert Wilcox
 *
 * This program demonstrates a CUDA-accelerated multi-scale face (or object) detection
 * algorithm using a Haar-like cascade classifier. The steps include:
 *   - Parsing command-line arguments for input/output image paths.
 *   - Checking for CUDA device availability.
 *   - Loading an input image and computing its integral images.
 *   - Initializing and linking a cascade classifier.
 *   - Computing additional offsets for the detection window.
 *   - Performing multi-scale detection using a CUDA detection routine.
 *   - Grouping and drawing detection results (candidate rectangles).
 *   - Saving the output image and cleaning up allocated resources.
 *
 * Usage: ./program -i [path/to/input/image] -o [path/to/output/image]
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#include "image_cuda.h"
#include "haar_cuda.h"
#include "cuda_detect.cuh"  // runDetection is declared here.


#define MINNEIGHBORS 1

//#define FINAL_DEBUG


extern int iter_counter;

inline int myRound(float value);  // Prototype for the inline function.

void ScaleImage_Invoker(myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);

extern void nearestNeighbor(MyImage* src, MyImage* dst);

// Helper function to scan classifier data and compute the extra offsets (in x and y)
// that account for all rectangle extents beyond the base detection window.
void computeExtraOffsets(const myCascade* cascade, int* extra_x, int* extra_y);

// Helper function to print a subset of values from an integral image.
void debugPrintIntegralImageGPU(MyIntImage* img, int numSamples);


int main(int argc, char** argv) {
    // Variable declarations for option parsing and runtime measurement.
    int opt;
    int rc;

    // Pointers to hold input and output file paths.
    char* input_file_path = NULL;
    char* output_file_path = NULL;

    // Variables to store start and end times for performance measurement.
    struct timespec t_start;
    struct timespec t_end;

    // Parse command-line options using getopt.
    while ((opt = getopt(argc, argv, "i:o:")) != -1) {
        switch (opt) {
            // Option 'i' for input image file path.
        case 'i':
            // Verify that the specified file exists.
            if (access(optarg, F_OK) != 0) {
                fprintf(stderr, "ERROR: path to file %s does not exist\n", optarg);
                fprintf(stderr, "Usage: %s -i [path/to/image] -o [path/to/output/image]\nExitting...\n", argv[0]);
                exit(1);
            }
            // Store valid input file path.
            input_file_path = optarg;
            break;

            // Option 'o' for output image file path.
        case 'o':
            output_file_path = optarg;
            break;

            // Default case if an unknown option is provided.
        default:
            fprintf(stderr, "Usage: %s -i [path/to/image] -o [path/to/output/image]\nExitting...\n", argv[0]);
            exit(1);
        }
    }

    // Indicate entry into the main function.
    printf("-- entering main function --\n");

    // Check for available CUDA devices.
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
    if (readPgm(input_file_path, image) == -1) {
        printf("Unable to open input image\n");
        return 1;
    }

    // 2. Compute integral images for fast feature computation.
    MyIntImage sumObj, sqsumObj;
    MyIntImage* sum = &sumObj;
    MyIntImage* sqsum = &sqsumObj;
    createSumImage(image->width, image->height, sum);
    createSumImage(image->width, image->height, sqsum);
    integralImages(image, sum, sqsum);

    // 3. Initialize the cascade classifier parameters.
    myCascade cascadeObj;
    myCascade* cascade = &cascadeObj;
    cascade->n_stages = 25;
    cascade->total_nodes = 2913;
    cascade->orig_window_size.width = 24;
    cascade->orig_window_size.height = 24;
    MySize minSize = { 20, 20 };
    MySize maxSize = { 0, 0 };

    // Load the cascade classifier data.
    printf("-- loading cascade classifier --\n");
    readTextClassifier(cascade);
    printf("-- cascade classifier loaded --\n");

    // Validate that the classifier rectangles have been loaded.
    if (cascade->scaled_rectangles_array == NULL) {
        printf("ERROR: cascade->scaled_rectangles_array is NULL after readTextClassifier!\n");
    }
    else {
        printf("cascade->scaled_rectangles_array is NOT NULL after readTextClassifier: %p\n", cascade->scaled_rectangles_array);
    }

    // 4. Link the computed integral images to the cascade classifier.
    printf("-- linking integral images to cascade --\n");
    setImageForCascadeClassifier(cascade, sum, sqsum);
    printf("-- After setImageForCascadeClassifier call --\n");
    printf("-- integral images linked to cascade --\n");

    // Compute extra offsets that adjust the detection window dimensions.
    int extra_x = 0, extra_y = 0;
    computeExtraOffsets(cascade, &extra_x, &extra_y);
    printf("Computed extra offsets: extra_x = %d, extra_y = %d\n", extra_x, extra_y);

    // Adjust the detection window size to include extra offsets.
    int adjusted_width = cascade->orig_window_size.width + extra_x;
    int adjusted_height = cascade->orig_window_size.height + extra_y;
    printf("Adjusted detection window size: (%d, %d)\n", adjusted_width, adjusted_height);

    // Allocate buffers for scaled image and its integral images.
    MyImage scaledImg;
    createImage(image->width, image->height, &scaledImg);
    MyIntImage scaledSum, scaledSqSum;
    createSumImage(image->width, image->height, &scaledSum);
    createSumImage(image->width, image->height, &scaledSqSum);
    float factor = 1.0f;

    // ***** Run CUDA detection at each scale in the image pyramid *****
    // Prepare a vector to store all candidate detections from GPU.
    std::vector<MyRect> allGpuCandidates;
    int iter_counter = 1;
    factor = 1.0f;

    // Start the timer for performance measurement.
    rc = clock_gettime(CLOCK_REALTIME, &t_start);
    assert(rc == 0);

    // Loop over different scales of the image.
    while (true) {
        // Calculate new dimensions based on the scaling factor.
        int newWidth = (int)(image->width / factor);
        int newHeight = (int)(image->height / factor);
        int winWidth = myRound(cascade->orig_window_size.width * factor);
        int winHeight = myRound(cascade->orig_window_size.height * factor);
        MySize sz = { newWidth, newHeight };
        MySize winSize = { winWidth, winHeight };
        // Compute the available difference in dimensions for placing the detection window.
        MySize diff = { sz.width - cascade->orig_window_size.width, sz.height - cascade->orig_window_size.height };

        // If the difference is negative, the window no longer fits; exit the loop.
        if (diff.width < 0 || diff.height < 0)
            break;
        // Skip scales that produce a detection window smaller than the minimum allowed size.
        if (winSize.width < minSize.width || winSize.height < minSize.height) {
            factor *= 1.2f;
            continue;
        }

        // Reallocate buffers for the current scale.
        freeImage(&scaledImg);
        freeSumImage(&scaledSum);
        freeSumImage(&scaledSqSum);
        createImage(newWidth, newHeight, &scaledImg);
        createSumImage(newWidth, newHeight, &scaledSum);
        createSumImage(newWidth, newHeight, &scaledSqSum);

        // Scale the input image using nearest neighbor interpolation.
        nearestNeighbor(image, &scaledImg);
        // Compute the integral images for the scaled image.
        integralImages(&scaledImg, &scaledSum, &scaledSqSum);
        // Link the scaled integral images to the cascade classifier.
        setImageForCascadeClassifier(cascade, &scaledSum, &scaledSqSum);

        // Check if the detection window fits within the scaled integral image dimensions.
        if (factor * (cascade->orig_window_size.width + extra_x) < scaledSum.width &&
            factor * (cascade->orig_window_size.height + extra_y) < scaledSum.height) {
            // Run the CUDA detection for the current scale.
            std::vector<MyRect> gpuCandidates = runDetection(&scaledSum, &scaledSqSum, cascade, 10000000, factor, adjusted_width, adjusted_height, iter_counter);
            // Merge the current scale's candidates into the overall candidate list.
            allGpuCandidates.insert(allGpuCandidates.end(), gpuCandidates.begin(), gpuCandidates.end());
        }
        else {
            // Scale factor too high; detection window does not fit. (Debug print commented out.)
        }
        // Increment the scale factor for the next iteration.
        factor *= 1.2f;
    }

    // Group nearby candidate rectangles to remove duplicates.
    groupRectangles(allGpuCandidates, MINNEIGHBORS, 0.4f);

    // Stop the timer and calculate the runtime.
    rc = clock_gettime(CLOCK_REALTIME, &t_end);
    assert(rc == 0);

    unsigned long long int runtime = 1000000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_nsec - t_start.tv_nsec;

    // Output the number of candidates and runtime details.
    printf("\nCUDA detection detected %zu candidates.\n", allGpuCandidates.size());
    printf("Time = %lld nanoseconds\t(%lld.%09lld sec)\n\n", runtime, runtime / 1000000000, runtime % 1000000000);

    // Debug output: print the coordinates and dimensions for each detected candidate.
    for (size_t i = 0; i < allGpuCandidates.size(); i++) {
        printf("[DEBUG] CUDA Candidate %zu: x=%d, y=%d, width=%d, height=%d\n",
            i, allGpuCandidates[i].x, allGpuCandidates[i].y, allGpuCandidates[i].width, allGpuCandidates[i].height);
    }

    // 8. Draw detection rectangles (candidate face boxes) on the original image.
    for (size_t i = 0; i < allGpuCandidates.size(); i++) {
        drawRectangle(image, allGpuCandidates[i]);
    }

    // 9. Save the output image with drawn candidate rectangles.
    printf("-- saving output --\n");
    int flag = writePgm(output_file_path, image);
    if (flag == -1)
        printf("Unable to save output image\n");
    else
        printf("-- image saved as %s --\n", output_file_path);

    // 10. Clean up and release resources.
    releaseTextClassifier(cascade);
    freeImage(image);
    freeSumImage(sum);
    freeSumImage(sqsum);

    // Free the temporary buffers allocated for scaled images.
    freeImage(&scaledImg);
    freeSumImage(&scaledSum);
    freeSumImage(&scaledSqSum);

    return 0;
}

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

#ifdef FINAL_DEBUG
    printf("GPU Integral image summary: width = %d, height = %d, total values = %d\n", width, height, total);
#endif

#ifdef FINAL_DEBUG
    // Print the four corner values
    printf("Top-left (index 0): %d\n", img->data[0]);
    printf("Top-right (index %d): %d\n", width - 1, img->data[width - 1]);
    printf("Bottom-left (index %d): %d\n", (height - 1) * width, img->data[(height - 1) * width]);
    printf("Bottom-right (index %d): %d\n", total - 1, img->data[total - 1]);
#endif

    int step = total / numSamples;
    if (step < 1)
        step = 1;

#ifdef FINAL_DEBUG
    printf("Printing %d sample values (every %d-th value):\n", numSamples, step);
    for (int i = 0; i < total; i += step) {
        printf("Index %d: %d\n", i, img->data[i]);
    }
#endif
}
