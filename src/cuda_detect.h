#ifndef CUDA_DETECT_H
#define CUDA_DETECT_H

#include "image.h"
#include "haar.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

	// Declare device globals as extern __device__.
	extern __device__ int* d_stages_array;
	extern __device__ float* d_stages_thresh_array;
	extern __device__ int* d_rectangles_array;
	extern __device__ int* d_weights_array;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Prototypes for CUDA detection functions.
std::vector<MyRect> runDetection(const std::vector<MyRect>& candidateWindows);
void loadCascadeClassifierCUDA(myCascade* cascade);
void setImageForCascadeClassifierCUDA(myCascade* cascade, MyIntImage* sum, MyIntImage* sqsum);
#endif

#endif // CUDA_DETECT_H
