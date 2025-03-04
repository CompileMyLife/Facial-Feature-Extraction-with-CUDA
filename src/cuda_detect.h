#ifndef CUDA_DETECT_H
#define CUDA_DETECT_H

#include "haar.h"
#include "image.h"

#ifdef __cplusplus
#include <vector>

// runDetection launches the CUDA detection kernel using the host‐side
// integral images and cascade classifier. It transfers data to the GPU,
// launches the kernel, retrieves results, cleans up device memory,
// and returns a std::vector<MyRect> containing candidate detections.
// Parameters:
//   h_sum         - pointer to the host MyIntImage for the integral image
//   h_sqsum       - pointer to the host MyIntImage for the squared integral image
//   cascade       - pointer to the host cascade classifier (after setImageForCascadeClassifier)
//   maxCandidates - maximum number of candidate detections allocated on the device
//   scaleFactor   - current scale factor (e.g., 1.0f)
//   adjusted_width  - detection window width after accounting for extra offsets
//   adjusted_height - detection window height after accounting for extra offsets
std::vector<MyRect> runDetection(MyIntImage* h_sum, MyIntImage* h_sqsum,
    myCascade* cascade,
    int maxCandidates,
    float scaleFactor,
    int adjusted_width,
    int adjusted_height);
#endif // __cplusplus

#endif // CUDA_DETECT_H
