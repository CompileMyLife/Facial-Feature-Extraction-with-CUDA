// cuda_detect.cu
//
// This file implements a CUDA‐accelerated sliding window detector for the Viola‐Jones algorithm.
// This version implements the real weak classifier evaluation and uses device memory for classifier parameters.

#include "cuda_detect.cuh" // Include header file for CUDA detection
#include <stdio.h>      // Standard input/output library
#include <math.h>       // Math functions (e.g., sqrtf)
#include <cuda_runtime.h> // CUDA runtime library
#include <device_launch_parameters.h> // CUDA device launch parameters
#include <vector>   // Needed for std::vector to store detection results

// Uncomment the following line to enable extra CUDA debug prints.
#define DEBUG_CUDA_PRINTS

// Macro to restrict device prints to one thread for less output.
#ifdef DEBUG_CUDA_PRINTS
#define DEV_PRINT(...) do { \
    if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0) { \
        printf(__VA_ARGS__); \
    } \
} while(0)
#else
#define DEV_PRINT(...) 
#endif

// ---------------------------------------------------------------------
// Device globals for classifier parameters.
__device__ int* d_stages_array;
__device__ float* d_stages_thresh_array;
__device__ int* d_rectangles_array;
__device__ int* d_weights_array;
__device__ int* d_alpha1_array;
__device__ int* d_alpha2_array;
__device__ int* d_tree_thresh_array;

// Device function: Real weak classifier evaluation.
__device__ int evalWeakClassifier_device(int variance_norm_factor, int p_offset,
    int haar_counter, int w_index, int r_index)
{
#ifdef DEBUG_CUDA_PRINTS
    printf("[Device] Eval weak classifier: haar_counter=%d, w_index=%d, r_index=%d, p_offset=%d\n",
        haar_counter, w_index, r_index, p_offset);
#endif
    int t = d_tree_thresh_array[haar_counter] * variance_norm_factor;

    int sum = (*(d_rectangles_array + r_index * 12 + 0) + p_offset)
        - (*(d_rectangles_array + r_index * 12 + 1) + p_offset)
        - (*(d_rectangles_array + r_index * 12 + 2) + p_offset)
        + (*(d_rectangles_array + r_index * 12 + 3) + p_offset);
    sum = sum * (*(d_weights_array + w_index * 3 + 0));

    sum += (*(d_rectangles_array + r_index * 12 + 4) + p_offset)
        - (*(d_rectangles_array + r_index * 12 + 5) + p_offset)
        - (*(d_rectangles_array + r_index * 12 + 6) + p_offset)
        + (*(d_rectangles_array + r_index * 12 + 7) + p_offset);
    sum = sum + (*(d_weights_array + w_index * 3 + 1));

    if ((d_rectangles_array + r_index * 12 + 8) != NULL)
    {
        sum += (*(d_rectangles_array + r_index * 12 + 8) + p_offset)
            - (*(d_rectangles_array + r_index * 12 + 9) + p_offset)
            - (*(d_rectangles_array + r_index * 12 + 10) + p_offset)
            + (*(d_rectangles_array + r_index * 12 + 11) + p_offset);
        sum = sum + (*(d_weights_array + w_index * 3 + 2));
    }

#ifdef DEBUG_CUDA_PRINTS
    printf("[Device] Weak classifier: sum=%d, threshold=%d, returning %d\n",
        sum, t, (sum >= t ? d_alpha2_array[haar_counter] : d_alpha1_array[haar_counter]));
#endif

    if (sum >= t)
        return d_alpha2_array[haar_counter];
    else
        return d_alpha1_array[haar_counter];
}

// Device function: Run the cascade classifier on a candidate window.
__device__ int runCascadeClassifier_device(const myCascade* d_cascade, MyPoint p, int start_stage)
{
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        printf("[Device] runCascadeClassifier_device: window=(%d, %d)\n", p.x, p.y);
#endif
    int p_offset = p.y * (d_cascade->sum.width) + p.x;
    int pq_offset = p.y * (d_cascade->sqsum.width) + p.x;

    unsigned int variance_norm_factor = (d_cascade->pq0[pq_offset] - d_cascade->pq1[pq_offset]
        - d_cascade->pq2[pq_offset] + d_cascade->pq3[pq_offset]);
    unsigned int mean = (d_cascade->p0[p_offset] - d_cascade->p1[p_offset]
        - d_cascade->p2[p_offset] + d_cascade->p3[p_offset]);
    variance_norm_factor = (variance_norm_factor * d_cascade->inv_window_area) - mean * mean;
    if (variance_norm_factor > 0)
        variance_norm_factor = (unsigned int)sqrtf((float)variance_norm_factor);
    else
        variance_norm_factor = 1;

    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    float stage_sum = 0.0f;

    for (int i = start_stage; i < d_cascade->n_stages; i++) {
        stage_sum = 0.0f;
        int num_features = d_stages_array[i];
#ifdef DEBUG_CUDA_PRINTS
        printf("[Device] Stage %d: num_features=%d\n", i, num_features);
#endif
        for (int j = 0; j < num_features; j++) {
            stage_sum += evalWeakClassifier_device(variance_norm_factor, p_offset, haar_counter, w_index, r_index);
            haar_counter++;
            w_index++;
            r_index++;
        }
#ifdef DEBUG_CUDA_PRINTS
        printf("[Device] Stage %d: stage_sum=%f, threshold=%f\n", i, stage_sum, d_stages_thresh_array[i]);
#endif
        if (stage_sum < d_stages_thresh_array[i])
            return -i;
    }
    return 1;
}

// CUDA kernel: Each thread processes one candidate window.
__global__ void detectKernel(MyIntImage d_sum, MyIntImage d_sqsum,
    myCascade d_cascade, float factor,
    int x_max, int y_max,
    MyRect* d_candidates, int* d_candidateCount,
    int* d_stages, float* d_stages_thresh, int* d_rects,
    int* d_weights, int* d_alpha1, int* d_alpha2, int* d_tree_thresh)
{
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        printf("-- Entering detectKernel, block=(%d, %d), thread=(%d, %d) --\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
#endif
    d_stages_array = d_stages;
    d_stages_thresh_array = d_stages_thresh;
    d_rectangles_array = d_rects;
    d_weights_array = d_weights;
    d_alpha1_array = d_alpha1;
    d_alpha2_array = d_alpha2;
    d_tree_thresh_array = d_tree_thresh;
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        printf("-- Device globals set in detectKernel --\n");
#endif

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        printf("-- Thread position: x=%d, y=%d --\n", x, y);
#endif

    if (x > x_max || y > y_max) {
#ifdef DEBUG_CUDA_PRINTS
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
            printf("-- Thread out of bounds, exiting: x=%d, y=%d, x_max=%d, y_max=%d --\n", x, y, x_max, y_max);
#endif
        return;
    }

    MyPoint p;
    p.x = x;
    p.y = y;

#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        printf("-- Calling runCascadeClassifier_device for window (%d, %d) --\n", x, y);
#endif
    int result = runCascadeClassifier_device(&d_cascade, p, 0);
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        printf("-- runCascadeClassifier_device returned: %d for window (%d, %d) --\n", result, x, y);
#endif

    if (result > 0) {
        MyRect r;
        r.x = (int)roundf(x * factor);
        r.y = (int)roundf(y * factor);
        r.width = (int)roundf(d_cascade.orig_window_size.width * factor);
        r.height = (int)roundf(d_cascade.orig_window_size.height * factor);
#ifdef DEBUG_CUDA_PRINTS
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
            printf("-- Detection found at (%d, %d), scaled rect=(%d, %d, %d, %d) --\n", x, y, r.x, r.y, r.width, r.height);
#endif
        int idx = atomicAdd(d_candidateCount, 1);
        d_candidates[idx] = r;
#ifdef DEBUG_CUDA_PRINTS
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
            printf("-- Candidate added at index %d --\n", idx);
#endif
    }
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        printf("-- Exiting detectKernel, block=(%d, %d), thread=(%d, %d) --\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
#endif
}

// Host function: runDetection() manages data transfer, kernel launch, and result retrieval.
std::vector<MyRect> runDetection(MyIntImage* h_sum, MyIntImage* h_sqsum, myCascade* cascade, int maxCandidates, float scaleFactor)
{
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] runDetection() started.\n");
#endif
    std::vector<MyRect> candidates;

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Allocating and copying sum integral image data to device.\n");
#endif
    int dataSize = h_sum->width * h_sum->height * sizeof(int);
    int* d_sumData;
    cudaMalloc((void**)&d_sumData, dataSize);
    cudaMemcpy(d_sumData, h_sum->data, dataSize, cudaMemcpyHostToDevice);
    MyIntImage h_sumDevice = *h_sum;
    h_sumDevice.data = d_sumData;
    MyIntImage* d_sum;
    cudaMalloc((void**)&d_sum, sizeof(MyIntImage));
    cudaMemcpy(d_sum, &h_sumDevice, sizeof(MyIntImage), cudaMemcpyHostToDevice);

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Allocating and copying squared sum integral image data to device.\n");
#endif
    dataSize = h_sqsum->width * h_sqsum->height * sizeof(int);
    int* d_sqsumData;
    cudaMalloc((void**)&d_sqsumData, dataSize);
    cudaMemcpy(d_sqsumData, h_sqsum->data, dataSize, cudaMemcpyHostToDevice);
    MyIntImage h_sqsumDevice = *h_sqsum;
    h_sqsumDevice.data = d_sqsumData;
    MyIntImage* d_sqsum;
    cudaMalloc((void**)&d_sqsum, sizeof(MyIntImage));
    cudaMemcpy(d_sqsum, &h_sqsumDevice, sizeof(MyIntImage), cudaMemcpyHostToDevice);

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Updating cascade structure with device pointers for integral images.\n");
#endif
    cascade->p0 = d_sumData;
    cascade->p1 = d_sumData + (h_sumDevice.width - 1);
    cascade->p2 = d_sumData + (h_sumDevice.width * (h_sumDevice.height - 1));
    cascade->p3 = d_sumData + (h_sumDevice.width * (h_sumDevice.height - 1) + (h_sumDevice.width - 1));
    cascade->pq0 = d_sqsumData;
    cascade->pq1 = d_sqsumData + (h_sqsumDevice.width - 1);
    cascade->pq2 = d_sqsumData + (h_sqsumDevice.width * (h_sumDevice.height - 1));
    cascade->pq3 = d_sqsumData + (h_sqsumDevice.width * (h_sumDevice.height - 1) + (h_sqsumDevice.width - 1));

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Transferring classifier parameters to device memory.\n");
#endif
    int* d_stages_array_dev;
    float* d_stages_thresh_array_dev;
    int* d_rectangles_array_dev;
    int* d_weights_array_dev;
    int* d_alpha1_array_dev;
    int* d_alpha2_array_dev;
    int* d_tree_thresh_array_dev;

    cudaMalloc((void**)&d_stages_array_dev, cascade->n_stages * sizeof(int));
    cudaMemcpy(d_stages_array_dev, cascade->stages_array, cascade->n_stages * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_stages_thresh_array_dev, cascade->n_stages * sizeof(float));
    cudaMemcpy(d_stages_thresh_array_dev, cascade->stages_thresh_array, cascade->n_stages * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_rectangles_array_dev, cascade->total_nodes * 12 * sizeof(int));
    cudaMemcpy(d_rectangles_array_dev, cascade->rectangles_array, cascade->total_nodes * 12 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_weights_array_dev, cascade->total_nodes * 3 * sizeof(int));
    cudaMemcpy(d_weights_array_dev, cascade->weights_array, cascade->total_nodes * 3 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_alpha1_array_dev, cascade->total_nodes * sizeof(int));
    cudaMemcpy(d_alpha1_array_dev, cascade->alpha1_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_alpha2_array_dev, cascade->total_nodes * sizeof(int));
    cudaMemcpy(d_alpha2_array_dev, cascade->alpha2_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_tree_thresh_array_dev, cascade->total_nodes * sizeof(int));
    cudaMemcpy(d_tree_thresh_array_dev, cascade->tree_thresh_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice);

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Copying cascade structure to device memory.\n");
#endif
    myCascade* d_cascade;
    cudaMalloc((void**)&d_cascade, sizeof(myCascade));
    cudaMemcpy(d_cascade, cascade, sizeof(myCascade), cudaMemcpyHostToDevice);

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Allocating memory for detection results on device.\n");
#endif
    MyRect* d_candidates;
    cudaMalloc((void**)&d_candidates, maxCandidates * sizeof(MyRect));
    int* d_candidateCount;
    cudaMalloc((void**)&d_candidateCount, sizeof(int));
    cudaMemset(d_candidateCount, 0, sizeof(int));

    int x_max = 100;
    int y_max = 100;
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Search space dimensions: x_max=%d, y_max=%d\n", x_max, y_max);
#endif

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Launching detection kernel.\n");
#endif
    dim3 blockDim(16, 16);
    dim3 gridDim((x_max + blockDim.x - 1) / blockDim.x, (y_max + blockDim.y - 1) / blockDim.y);
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] gridDim=(%d, %d), blockDim=(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
#endif
    detectKernel << <gridDim, blockDim >> > (*d_sum, *d_sqsum, *d_cascade, scaleFactor,
        x_max, y_max, d_candidates, d_candidateCount,
        d_stages_array_dev, d_stages_thresh_array_dev,
        d_rectangles_array_dev, d_weights_array_dev,
        d_alpha1_array_dev, d_alpha2_array_dev, d_tree_thresh_array_dev);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[Host] Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[Host] Kernel execution error: %s\n", cudaGetErrorString(err));
    }

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Kernel execution completed.\n");
#endif

    int h_candidateCount = 0;
    cudaMemcpy(&h_candidateCount, d_candidateCount, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Detected %d candidate windows.\n", h_candidateCount);

    if (h_candidateCount > 0)
    {
        MyRect* h_candidates = (MyRect*)malloc(h_candidateCount * sizeof(MyRect));
        cudaMemcpy(h_candidates, d_candidates, h_candidateCount * sizeof(MyRect), cudaMemcpyDeviceToHost);
        for (int i = 0; i < h_candidateCount; i++) {
            candidates.push_back(h_candidates[i]);
        }
        free(h_candidates);
    }

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Cleaning up device memory.\n");
#endif
    cudaFree(d_candidates);
    cudaFree(d_candidateCount);
    cudaFree(d_cascade);
    cudaFree(d_sum);
    cudaFree(d_sqsum);
    cudaFree(d_sumData);
    cudaFree(d_sqsumData);
    cudaFree(d_stages_array_dev);
    cudaFree(d_stages_thresh_array_dev);
    cudaFree(d_rectangles_array_dev);
    cudaFree(d_weights_array_dev);
    cudaFree(d_alpha1_array_dev);
    cudaFree(d_alpha2_array_dev);
    cudaFree(d_tree_thresh_array_dev);

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] runDetection() completed.\n");
#endif
    return candidates;
}
