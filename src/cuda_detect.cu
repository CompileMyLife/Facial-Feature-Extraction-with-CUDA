// cuda_detect.cu
//
// This file implements a CUDA‐accelerated sliding window detector for the Viola‐Jones algorithm.
// This version implements the real weak classifier evaluation and uses Unified Memory for the integral images,
// cascade structure, and detection results.
#include "cuda_detect.h"    // Include header file for CUDA detection
#include <stdio.h>          // Standard I/O
#include <math.h>           // Math functions
#include <cuda_runtime.h>   // CUDA runtime
#include <device_launch_parameters.h>
#include <vector>           // For std::vector
#include <string.h>         // For memcpy
#include <assert.h>         // For device-side assertions

#define DEBUG_CANDIDATE_X 2015
#define DEBUG_CANDIDATE_Y 863

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return std::vector<MyRect>(); \
    } \
} while(0)

// ---------------------------------------------------------------------
// Constant memory for classifier parameters.
__constant__ int* d_stages_array;
__constant__ float* d_stages_thresh_array;
__constant__ int* d_rectangles_array;
__constant__ int* d_weights_array;
__constant__ int* d_alpha1_array;
__constant__ int* d_alpha2_array;
__constant__ int* d_tree_thresh_array;

// ---------------------------------------------------------------------
// Declaration of atomicCAS.
extern __device__ int atomicCAS(int* address, int compare, int val);

// ---------------------------------------------------------------------
// Device function: Integer square root for the GPU.
// This function replicates the behavior of the CPU's int_sqrt.
__device__ int int_sqrt_device(int value) {
    int i;
    unsigned int a = 0, b = 0, c = 0;
    for (i = 0; i < (32 >> 1); i++) {
        c <<= 2;
        c += (value >> 30); // get the upper 2 bits of value
        value <<= 2;
        a <<= 1;
        b = (a << 1) | 1;
        if (c >= b) {
            c -= b;
            a++;
        }
    }
    return a;
}

// ---------------------------------------------------------------------
// Device function: Rounding function mirroring CPU implementation
__device__ inline int myRound_device(float value) {
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}


// ---------------------------------------------------------------------
// Device function: Evaluate a weak classifier for candidate window p.
// Assumes that for each feature, d_rectangles_array stores 12 ints in the order:
// [x_offset1, y_offset1, width1, height1, x_offset2, y_offset2, width2, height2,
//  x_offset3, y_offset3, width3, height3]
__device__ float evalWeakClassifier_device(const myCascade* d_cascade, int variance_norm_factor, MyPoint p,
    int haar_counter, int w_index, int r_index, float scaleFactor)
{

    //printf("[Device] entered evalWeakClassifier_device\n");

    // Print candidate coordinates for every 100th candidate

    int* rect = d_rectangles_array + r_index;

    // --- First Rectangle ---
    int tl1_x = p.x + (int)myRound_device(rect[0] * scaleFactor);
    int tl1_y = p.y + (int)myRound_device(rect[1] * scaleFactor);
    int br1_x = tl1_x + (int)myRound_device(rect[2] * scaleFactor);
    int br1_y = tl1_y + (int)myRound_device(rect[3] * scaleFactor);

    if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y &&
        haar_counter == 0 && w_index == 0 && r_index == 0) {
        printf("[Device DEBUG] First rectangle: tl=(%d,%d), br=(%d,%d)\n", tl1_x, tl1_y, br1_x, br1_y);
    }


    // Check bounds
    assert(tl1_x >= 0 && tl1_x < d_cascade->sum.width);
    assert(tl1_y >= 0 && tl1_y < d_cascade->sum.height);
    assert(br1_x >= 0 && br1_x < d_cascade->sum.width);
    assert(br1_y >= 0 && br1_y < d_cascade->sum.height);

    int idx_tl1 = tl1_y * d_cascade->sum.width + tl1_x;
    int idx_tr1 = tl1_y * d_cascade->sum.width + br1_x;
    int idx_bl1 = br1_y * d_cascade->sum.width + tl1_x;
    int idx_br1 = br1_y * d_cascade->sum.width + br1_x;


    int sum1 = d_cascade->p0[idx_br1] - d_cascade->p0[idx_tr1] - d_cascade->p0[idx_bl1] + d_cascade->p0[idx_tl1];
    sum1 = sum1 * d_weights_array[w_index + 0];

    // --- Second Rectangle ---
    int tl2_x = p.x + (int)myRound_device(rect[4] * scaleFactor);
    int tl2_y = p.y + (int)myRound_device(rect[5] * scaleFactor);
    int br2_x = tl2_x + (int)myRound_device(rect[6] * scaleFactor);
    int br2_y = tl2_y + (int)myRound_device(rect[7] * scaleFactor);


    if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y &&
        haar_counter == 0 && w_index == 0 && r_index == 0) {
        printf("[Device DEBUG] Second rectangle: tl=(%d,%d), br=(%d,%d)\n", tl2_x, tl2_y, br2_x, br2_y);
    }

    assert(tl2_x >= 0 && tl2_x < d_cascade->sum.width);
    assert(tl2_y >= 0 && tl2_y < d_cascade->sum.height);
    assert(br2_x >= 0 && br2_x < d_cascade->sum.width);
    assert(br2_y >= 0 && br2_y < d_cascade->sum.height);

    int idx_tl2 = tl2_y * d_cascade->sum.width + tl2_x;
    int idx_tr2 = tl2_y * d_cascade->sum.width + br2_x;
    int idx_bl2 = br2_y * d_cascade->sum.width + tl2_x;
    int idx_br2 = br2_y * d_cascade->sum.width + br2_x;


    int sum2 = d_cascade->p0[idx_br2] - d_cascade->p0[idx_tr2] - d_cascade->p0[idx_bl2] + d_cascade->p0[idx_tl2];
    sum2 = sum2 * d_weights_array[w_index + 1];

    int total_sum = sum1 + sum2;

    int sum3 = 0;

    // --- Third Rectangle (if present) ---
    if (d_weights_array[w_index + 2] != 0)
    {
        int tl3_x = p.x + (int)myRound_device(rect[8] * scaleFactor);
        int tl3_y = p.y + (int)myRound_device(rect[9] * scaleFactor);
        int br3_x = tl3_x + (int)myRound_device(rect[10] * scaleFactor);
        int br3_y = tl3_y + (int)myRound_device(rect[11] * scaleFactor);

        if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y &&
            haar_counter == 0 && w_index == 0 && r_index == 0) {
            printf("[Device DEBUG] Third rectangle: tl=(%d,%d), br=(%d,%d)\n", tl3_x, tl3_y, br3_x, br3_y);
        }

        assert(tl3_x >= 0 && tl3_x < d_cascade->sum.width);
        assert(tl3_y >= 0 && tl3_y < d_cascade->sum.height);
        assert(br3_x >= 0 && br3_x < d_cascade->sum.width);
        assert(br3_y >= 0 && br3_y < d_cascade->sum.height);

        int idx_tl3 = tl3_y * d_cascade->sum.width + tl3_x;
        int idx_tr3 = tl3_y * d_cascade->sum.width + br3_x;
        int idx_bl3 = br3_y * d_cascade->sum.width + tl3_x;
        int idx_br3 = br3_y * d_cascade->sum.width + br3_x;
        sum3 = d_cascade->p0[idx_br3] - d_cascade->p0[idx_tr3] - d_cascade->p0[idx_bl3] + d_cascade->p0[idx_tl3];
        total_sum += sum3 * d_weights_array[w_index + 2];
    }

    // Debug print for a specific candidate at stage 0, feature 0.
    if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y && haar_counter == 0 && w_index == 0 && r_index == 0) {
        printf("[Device DEBUG] Candidate (%d,%d), Stage 0, Feature 0: sum1 = %d, sum2 = %d, sum3 = %d, final_sum = %d\n",
            p.x, p.y, sum1, sum2, sum3, total_sum);
    }


    int t =d_tree_thresh_array[haar_counter] * (variance_norm_factor);

    return (total_sum >= t) ? d_alpha2_array[haar_counter] : d_alpha1_array[haar_counter];
}


// ---------------------------------------------------------------------
// Device function: Run the cascade classifier on a candidate window.
__device__ int runCascadeClassifier_device(MyIntImage* d_sum, MyIntImage* d_sqsum, 
    const myCascade* d_cascade, MyPoint p, int start_stage, float scaleFactor)
{
    // Ensure candidate window is within bounds.
    assert(p.x >= 0 && p.x < d_cascade->sum.width);
    assert(p.y >= 0 && p.y < d_cascade->sum.height);

    int p_offset = p.y * d_cascade->sum.width + p.x;
    assert(p_offset < d_cascade->sum.width * d_cascade->sum.height);



    int pq_offset = p.y * d_cascade->sqsum.width + p.x;
    assert(pq_offset < d_cascade->sqsum.width * d_cascade->sqsum.height);

    // Compute the mean and variance from the integral images.
    unsigned int var_norm = (d_cascade->pq0[pq_offset] - d_cascade->pq1[pq_offset]
        - d_cascade->pq2[pq_offset] + d_cascade->pq3[pq_offset]);
    unsigned int mean = (d_cascade->p0[p_offset] - d_cascade->p1[p_offset]
        - d_cascade->p2[p_offset] + d_cascade->p3[p_offset]);

    var_norm = (var_norm * d_cascade->inv_window_area) - mean * mean;
    if (var_norm > 0)
		var_norm = int_sqrt_device(var_norm);
    else
        var_norm = 1;

    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    float stage_sum = 0.0f;

    for (int i = start_stage; i < d_cascade->n_stages; i++) {
        stage_sum = 0.0f;
        int num_features = d_stages_array[i];

        // Check that processing this stage won't overrun our classifier arrays.
        if (haar_counter + num_features > d_cascade->total_nodes) {
            return 1;
        }
        for (int j = 0; j < num_features; j++) {

            // Compute the feature response.
            int feature_result = evalWeakClassifier_device(d_cascade, (int)var_norm, p, 
                haar_counter, w_index, r_index, scaleFactor);
                stage_sum += feature_result;
                haar_counter++;
                w_index += 3;    // advance the weight index by 3 (since 3 weights per feature)
                r_index += 12;   // advance the rectangle index by 12 (since 12 ints per feature)
        }

        /* the number "0.4" is empirically chosen for 5kk73 */
        if (stage_sum < 0.4 * d_stages_thresh_array[i])
            return -i;
    }
    return 1;
}



// ---------------------------------------------------------------------
// CUDA kernel: Each thread processes one candidate window.
__global__ void detectKernel(MyIntImage* d_sum, MyIntImage* d_sqsum,
    myCascade* d_cascade, float scaleFactor,
    int x_max, int y_max,
    MyRect* d_candidates, int* d_candidateCount,
    int maxCandidates)
{
	//printf("[Device] entered detectKernel\n");

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= x_max || y >= y_max)
        return;

    MyPoint p;
    p.x = x;
    p.y = y;



    int result = runCascadeClassifier_device(d_sum, d_sqsum, d_cascade, p, 0, scaleFactor);

    if (result > 0) {
        MyRect r;
        r.x = (int)myRound_device(x * scaleFactor);
        r.y = (int)myRound_device(y * scaleFactor);

        r.width = (int)myRound_device(d_cascade->orig_window_size.width * scaleFactor);
        r.height = (int)myRound_device(d_cascade->orig_window_size.height * scaleFactor);
        int idx = atomicAdd(d_candidateCount, 1);
        if (idx < maxCandidates) {
            d_candidates[idx] = r;
        }
       
    }
}

// runDetection in cuda_detect.cu
std::vector<MyRect> runDetection(MyIntImage* h_sum, MyIntImage* h_sqsum,
    myCascade* cascade, int maxCandidates,
    float scaleFactor, int extra_x, int extra_y)
{
    std::vector<MyRect> candidates;

    // --- Step 1: Allocate Unified Memory for sum integral image ---
    int dataSize = h_sum->width * h_sum->height * sizeof(int);
    MyIntImage* d_sum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sum->data), dataSize));
    memcpy(d_sum->data, h_sum->data, dataSize);
    d_sum->width = h_sum->width;
    d_sum->height = h_sum->height;

    // --- Step 2: Allocate Unified Memory for squared sum integral image ---
    dataSize = h_sqsum->width * h_sqsum->height * sizeof(int);
    MyIntImage* d_sqsum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sqsum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sqsum->data), dataSize));
    memcpy(d_sqsum->data, h_sqsum->data, dataSize);
    d_sqsum->width = h_sqsum->width;
    d_sqsum->height = h_sqsum->height;

    // --- Step 3: Allocate Unified Memory for the cascade structure ---
    myCascade* d_cascade = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_cascade, sizeof(myCascade)));
    *d_cascade = *cascade;  // copy host cascade to unified memory

    // --- Step 4: Update the cascade with unified memory pointers for integral images ---
    d_cascade->sum = *d_sum;
    d_cascade->sqsum = *d_sqsum;
    d_cascade->sum.data = d_sum->data;   // Use unified memory data buffer of d_sum
    d_cascade->sqsum.data = d_sqsum->data; // Use unified memory data buffer of d_sqsum
    d_cascade->sum.width = d_sum->width;
    d_cascade->sum.height = d_sum->height;
    d_cascade->sqsum.width = d_sqsum->width;
    d_cascade->sqsum.height = d_sqsum->height;

    // Use the original window size for classification
    int winW = d_cascade->orig_window_size.width;
    int winH = d_cascade->orig_window_size.height;
    d_cascade->p0 = d_cascade->sum.data;
    d_cascade->p1 = d_cascade->sum.data + winW - 1;
    d_cascade->p2 = d_cascade->sum.data + d_cascade->sum.width * (winH - 1);
    d_cascade->p3 = d_cascade->sum.data + d_cascade->sum.width * (winH - 1) + (winW - 1);

    d_cascade->pq0 = d_cascade->sqsum.data;
    d_cascade->pq1 = d_cascade->sqsum.data + winW - 1;
    d_cascade->pq2 = d_cascade->sqsum.data + d_cascade->sqsum.width * (winH - 1);
    d_cascade->pq3 = d_cascade->sqsum.data + d_cascade->sqsum.width * (winH - 1) + (winW - 1);

    printf("Cascade corner pointers:\n");
    printf(" p0 = %p\n", (void*)d_cascade->p0);
    printf(" p1 = %p (offset: %td)\n", (void*)d_cascade->p1, d_cascade->p1 - d_cascade->sum.data);
    printf(" p2 = %p (offset: %td)\n", (void*)d_cascade->p2, d_cascade->p2 - d_cascade->sum.data);
    printf(" p3 = %p (offset: %td)\n", (void*)d_cascade->p3, d_cascade->p3 - d_cascade->sum.data);
    printf(" pq0 = %p\n", (void*)d_cascade->pq0);
    printf(" pq1 = %p (offset: %td)\n", (void*)d_cascade->pq1, d_cascade->pq1 - d_cascade->sqsum.data);
    printf(" pq2 = %p (offset: %td)\n", (void*)d_cascade->pq2, d_cascade->pq2 - d_cascade->sqsum.data);
    printf(" pq3 = %p (offset: %td)\n", (void*)d_cascade->pq3, d_cascade->pq3 - d_cascade->sqsum.data);

    // --- Step 5: Transfer classifier parameters to device constant memory ---
    // (Allocate device memory for the classifier arrays and copy them from host.)
    int* d_stages_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_stages_array_dev, cascade->n_stages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_stages_array_dev, cascade->stages_array, cascade->n_stages * sizeof(int), cudaMemcpyHostToDevice));

    float* d_stages_thresh_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_stages_thresh_array_dev, cascade->n_stages * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_stages_thresh_array_dev, cascade->stages_thresh_array, cascade->n_stages * sizeof(float), cudaMemcpyHostToDevice));

    int* d_rectangles_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_rectangles_array_dev, cascade->total_nodes * 12 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rectangles_array_dev, cascade->rectangles_array, cascade->total_nodes * 12 * sizeof(int), cudaMemcpyHostToDevice));

    int* d_weights_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_weights_array_dev, cascade->total_nodes * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_weights_array_dev, cascade->weights_array, cascade->total_nodes * 3 * sizeof(int), cudaMemcpyHostToDevice));

    int* d_alpha1_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_alpha1_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_alpha1_array_dev, cascade->alpha1_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    int* d_alpha2_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_alpha2_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_alpha2_array_dev, cascade->alpha2_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    int* d_tree_thresh_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_tree_thresh_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tree_thresh_array_dev, cascade->tree_thresh_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_array, &d_stages_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_thresh_array, &d_stages_thresh_array_dev, sizeof(float*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_rectangles_array, &d_rectangles_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_weights_array, &d_weights_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha1_array, &d_alpha1_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha2_array, &d_alpha2_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_tree_thresh_array, &d_tree_thresh_array_dev, sizeof(int*)));
    printf("[Host DEBUG] Transferred classifier parameters to device constant memory.\n");

    // --- Step 6: Allocate Unified Memory for detection results ---
    MyRect* d_candidates = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_candidates, maxCandidates * sizeof(MyRect)));
    int* d_candidateCount = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_candidateCount, sizeof(int)));
    *d_candidateCount = 0;
    printf("[Host DEBUG] d_candidates allocated at %p, d_candidateCount allocated at %p, initial candidate count = %d\n",
        (void*)d_candidates, (void*)d_candidateCount, *d_candidateCount);

    // --- Step 7: Determine search space dimensions and launch the detection kernel ---
    // Use extra_x and extra_y only to clip the search space so that the sliding window remains in bounds.
    // The classifier window size remains the original size (cascade->orig_window_size) * scaleFactor.
    int baseWidth = cascade->orig_window_size.width;
    int baseHeight = cascade->orig_window_size.height;

    int detectionWindowWidth = (baseWidth + extra_x) * scaleFactor;
    int detectionWindowHeight = (baseHeight + extra_y) * scaleFactor;

    // Compute maximum valid starting positions for the sliding window.
    int x_max = d_sum->width - detectionWindowWidth -1;
    int y_max = d_sum->height - detectionWindowHeight -1;
    if (x_max < 0) x_max = 0;
    if (y_max < 0) y_max = 0;
    printf("[Host DEBUG] Search space dimensions (with extra margins): x_max=%d, y_max=%d\n", x_max, y_max);

    dim3 blockDim(16, 16);
    dim3 gridDim((x_max + blockDim.x - 1) / blockDim.x,
        (y_max + blockDim.y - 1) / blockDim.y);
    printf("[Host] Launching kernel with gridDim=(%d, %d), blockDim=(%d, %d)\n",
        gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Launch the kernel with the original base window size for classification.
    CUDA_CHECK(cudaDeviceSynchronize());
    detectKernel << <gridDim, blockDim >> > (d_sum, d_sqsum, d_cascade, scaleFactor, x_max, y_max, d_candidates, d_candidateCount, maxCandidates);
    CUDA_CHECK(cudaGetLastError());
    printf("[Host DEBUG] Kernel launched.\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[Host DEBUG] Kernel execution completed.\n");

    int hostCandidateCount = 0;
    CUDA_CHECK(cudaMemcpy(&hostCandidateCount, d_candidateCount, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[Host] Detected %d candidate windows.\n", hostCandidateCount);
    for (int i = 0; i < hostCandidateCount; i++) {
        candidates.push_back(d_candidates[i]);
    }

    printf("[Host DEBUG] Cleaning up Unified Memory and device memory allocated with cudaMalloc.\n");
    cudaFree(d_candidates);
    cudaFree(d_candidateCount);
    cudaFree(d_cascade);
    cudaFree(d_sum->data);
    cudaFree(d_sum);
    cudaFree(d_sqsum->data);
    cudaFree(d_sqsum);
    cudaFree(d_stages_array_dev);
    cudaFree(d_stages_thresh_array_dev);
    cudaFree(d_rectangles_array_dev);
    cudaFree(d_weights_array_dev);
    cudaFree(d_alpha1_array_dev);
    cudaFree(d_alpha2_array_dev);
    cudaFree(d_tree_thresh_array_dev);

    printf("[Host] runDetection() completed.\n");
    return candidates;
}

