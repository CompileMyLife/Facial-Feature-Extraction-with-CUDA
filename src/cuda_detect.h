#ifndef CUDA_DETECT_H
#define CUDA_DETECT_H

#include "haar.h"
#include "image.h"

#ifdef __cplusplus
#include <vector>

// CUDA-accessible classifier parameters (Device memory pointers)
extern int* stages_array_d;
extern int* rectangles_array_d;
extern int* weights_array_d;
extern int* alpha1_array_d;
extern int* alpha2_array_d;
extern int* tree_thresh_array_d;
extern int* stages_thresh_array_d;
extern int** scaled_rectangles_array_d; // Array of pointers - needs careful handling in CUDA

// runDetection launches the CUDA detection kernel using the host‐side
// integral images and cascade classifier. It transfers data to the GPU,
// launches the kernel, retrieves results, cleans up device memory,
// and returns a std::vector<MyRect> containing candidate detections.
std::vector<MyRect> runDetection();

// Function to load classifier data to GPU memory
void loadCascadeClassifierCUDA(myCascade* cascade);

// Function to set up image pointers for CUDA cascade classifier
void setImageForCascadeClassifierCUDA(myCascade* cascade, MyIntImage* sum, MyIntImage* sqsum);


#endif // __cplusplus

#endif // CUDA_DETECT_H