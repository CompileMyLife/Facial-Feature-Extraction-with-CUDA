// cuda_detect.cu
//
// This file implements a CUDA‐accelerated sliding window detector for the Viola‐Jones algorithm.
// It contains device functions for cascade evaluation, the kernel that processes candidate windows,
// and the runDetection() function that transfers host data to device memory, launches the kernel,
// and retrieves results.

#include "cuda_detect.h"
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ---------------------------------------------------------------------
// Device classifier globals (placeholders).
// In a full implementation, you'll allocate and copy these arrays.
// ---------------------------------------------------------------------
__device__ int *d_stages_array; 
__device__ int *d_stages_thresh_array;

// Define scaled_rectangles_array as a device pointer.
// For now, we leave it unallocated (NULL).
__device__ int **scaled_rectangles_array = 0;

// Device function: Evaluate one weak classifier for a candidate window.
// For demonstration, if scaled_rectangles_array is NULL, return a dummy value.
__device__ int evalWeakClassifier_device(int variance_norm_factor, int p_offset,
                                           int tree_index, int w_index, int r_index)
{
    // If scaled_rectangles_array is not set, return a dummy value.
    if (scaled_rectangles_array == 0) {
         return 1;
    }

    int t = d_stages_thresh_array[tree_index] * variance_norm_factor;
    int sum = ( *(scaled_rectangles_array[r_index] + p_offset)
              - *(scaled_rectangles_array[r_index + 1] + p_offset)
              - *(scaled_rectangles_array[r_index + 2] + p_offset)
              + *(scaled_rectangles_array[r_index + 3] + p_offset) )
              * d_stages_array[w_index];
    sum += ( *(scaled_rectangles_array[r_index+4] + p_offset)
           - *(scaled_rectangles_array[r_index + 5] + p_offset)
           - *(scaled_rectangles_array[r_index + 6] + p_offset)
           + *(scaled_rectangles_array[r_index + 7] + p_offset) )
           * d_stages_array[w_index + 1];
    if (scaled_rectangles_array[r_index+8] != NULL)
    {
        sum += ( *(scaled_rectangles_array[r_index+8] + p_offset)
               - *(scaled_rectangles_array[r_index + 9] + p_offset)
               - *(scaled_rectangles_array[r_index + 10] + p_offset)
               + *(scaled_rectangles_array[r_index + 11] + p_offset) )
               * d_stages_array[w_index + 2];
    }
    return (sum >= t) ? 1 : 0;
}

// Device function: Run the cascade classifier on a candidate window.
__device__ int runCascadeClassifier_device(const myCascade *d_cascade, MyPoint p, int start_stage)
{
    int p_offset = p.y * (d_cascade->sum.width) + p.x;
    int pq_offset = p.y * (d_cascade->sqsum.width) + p.x;
    unsigned int variance_norm_factor = (d_cascade->pq0[pq_offset] - d_cascade->pq1[pq_offset]
                                           - d_cascade->pq2[pq_offset] + d_cascade->pq3[pq_offset]);
    unsigned int mean = (d_cascade->p0[p_offset] - d_cascade->p1[p_offset]
                          - d_cascade->p2[p_offset] + d_cascade->p3[p_offset]);
    variance_norm_factor = (variance_norm_factor * d_cascade->inv_window_area) - mean * mean;
    if(variance_norm_factor > 0)
        variance_norm_factor = (unsigned int)sqrtf((float)variance_norm_factor);
    else
        variance_norm_factor = 1;

    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    int stage_sum = 0;
    for (int i = start_stage; i < d_cascade->n_stages; i++) {
        stage_sum = 0;
        int num_features = d_stages_array[i]; // assume each stage's feature count is stored here
        for (int j = 0; j < num_features; j++) {
            stage_sum += evalWeakClassifier_device(variance_norm_factor, p_offset,
                                                     haar_counter, w_index, r_index);
            haar_counter++;
            w_index += 3;
            r_index += 12;
        }
        if (stage_sum < 0.4f * d_stages_thresh_array[i]) {
            return -i; // Rejection at stage i.
        }
    }
    return 1; // Window passed all stages.
}

// CUDA kernel: Each thread processes one candidate window.
__global__ void detectKernel(MyIntImage d_sum, MyIntImage d_sqsum,
                               myCascade d_cascade, float factor,
                               int x_max, int y_max,
                               MyRect *d_candidates, int *d_candidateCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > x_max || y > y_max) return;
    MyPoint p;
    p.x = x;
    p.y = y;
    int result = runCascadeClassifier_device(&d_cascade, p, 0);
    if (result > 0) {
        MyRect r;
        r.x = (int)roundf(x * factor);
        r.y = (int)roundf(y * factor);
        r.width  = (int)roundf(d_cascade.orig_window_size.width * factor);
        r.height = (int)roundf(d_cascade.orig_window_size.height * factor);
        int idx = atomicAdd(d_candidateCount, 1);
        d_candidates[idx] = r;
    }
}

// runDetection() transfers host data to the device, launches the detection kernel,
// retrieves results, and prints the number of candidate windows.
void runDetection(MyIntImage *h_sum, MyIntImage *h_sqsum, myCascade *cascade, int maxCandidates, float scaleFactor)
{
    // --- 1. Transfer integral image (sum) data ---
    int dataSize = h_sum->width * h_sum->height * sizeof(int);
    int *d_sumData;
    cudaMalloc((void**)&d_sumData, dataSize);
    cudaMemcpy(d_sumData, h_sum->data, dataSize, cudaMemcpyHostToDevice);
    MyIntImage h_sumDevice = *h_sum;
    h_sumDevice.data = d_sumData;
    MyIntImage *d_sum;
    cudaMalloc((void**)&d_sum, sizeof(MyIntImage));
    cudaMemcpy(d_sum, &h_sumDevice, sizeof(MyIntImage), cudaMemcpyHostToDevice);

    // --- 2. Transfer squared integral image (sqsum) data ---
    dataSize = h_sqsum->width * h_sqsum->height * sizeof(int);
    int *d_sqsumData;
    cudaMalloc((void**)&d_sqsumData, dataSize);
    cudaMemcpy(d_sqsumData, h_sqsum->data, dataSize, cudaMemcpyHostToDevice);
    MyIntImage h_sqsumDevice = *h_sqsum;
    h_sqsumDevice.data = d_sqsumData;
    MyIntImage *d_sqsum;
    cudaMalloc((void**)&d_sqsum, sizeof(MyIntImage));
    cudaMemcpy(d_sqsum, &h_sqsumDevice, sizeof(MyIntImage), cudaMemcpyHostToDevice);

    // --- 3. Update cascade pointers to use device data ---
    cascade->p0 = d_sumData;
    cascade->p1 = d_sumData + (h_sumDevice.width - 1);
    cascade->p2 = d_sumData + (h_sumDevice.width * (h_sumDevice.height - 1));
    cascade->p3 = d_sumData + (h_sumDevice.width * (h_sumDevice.height - 1) + (h_sumDevice.width - 1));
    cascade->pq0 = d_sqsumData;
    cascade->pq1 = d_sqsumData + (h_sqsumDevice.width - 1);
    cascade->pq2 = d_sqsumData + (h_sqsumDevice.width * (h_sqsumDevice.height - 1));
    cascade->pq3 = d_sqsumData + (h_sqsumDevice.width * (h_sqsumDevice.height - 1) + (h_sqsumDevice.width - 1));

    // --- 4. Copy the cascade structure to the device ---
    myCascade *d_cascade;
    cudaMalloc((void**)&d_cascade, sizeof(myCascade));
    cudaMemcpy(d_cascade, cascade, sizeof(myCascade), cudaMemcpyHostToDevice);

    // --- 5. Allocate memory for detection results ---
    MyRect *d_candidates;
    cudaMalloc((void**)&d_candidates, maxCandidates * sizeof(MyRect));
    int *d_candidateCount;
    cudaMalloc((void**)&d_candidateCount, sizeof(int));
    cudaMemset(d_candidateCount, 0, sizeof(int));

    // --- 6. Determine search space dimensions ---
    int x_max = h_sumDevice.width - cascade->orig_window_size.width;
    int y_max = h_sumDevice.height - cascade->orig_window_size.height;

    // --- 7. Launch detection kernel ---
    dim3 blockDim(16, 16);
    dim3 gridDim((x_max + blockDim.x - 1) / blockDim.x, (y_max + blockDim.y - 1) / blockDim.y);
    detectKernel<<<gridDim, blockDim>>>(*d_sum, *d_sqsum, *d_cascade, scaleFactor,
                                          x_max, y_max, d_candidates, d_candidateCount);
    cudaDeviceSynchronize();

    // --- 8. Retrieve detection results ---
    int h_candidateCount = 0;
    cudaMemcpy(&h_candidateCount, d_candidateCount, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Detected %d candidate windows.\n", h_candidateCount);
    MyRect *h_candidates = (MyRect*)malloc(h_candidateCount * sizeof(MyRect));
    cudaMemcpy(h_candidates, d_candidates, h_candidateCount * sizeof(MyRect), cudaMemcpyDeviceToHost);

    // --- 9. Clean up device memory ---
    cudaFree(d_candidates);
    cudaFree(d_candidateCount);
    cudaFree(d_cascade);
    cudaFree(d_sum);
    cudaFree(d_sqsum);
    cudaFree(d_sumData);
    cudaFree(d_sqsumData);
    free(h_candidates);
}
