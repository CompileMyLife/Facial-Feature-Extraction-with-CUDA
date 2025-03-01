#ifndef CUDA_DETECT_H
#define CUDA_DETECT_H

#include "haar.h"
#include "image.h"

// runDetection launches the CUDA detection kernel using the host‐side
// integral images and cascade classifier. It transfers data to the GPU,
// launches the kernel, retrieves results, and prints the number of candidate windows.
// Parameters:
//   h_sum       - pointer to the host MyIntImage for the integral image
//   h_sqsum     - pointer to the host MyIntImage for the squared integral image
//   cascade     - pointer to the host cascade classifier (after setImageForCascadeClassifier)
//   maxCandidates - maximum number of candidate detections allocated on the device
//   scaleFactor - current scale factor (e.g., 1.0f)
void runDetection(MyIntImage *h_sum, MyIntImage *h_sqsum, myCascade *cascade, int maxCandidates, float scaleFactor);

#endif // CUDA_DETECT_H
