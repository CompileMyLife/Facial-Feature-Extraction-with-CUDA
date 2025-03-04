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

// Uncomment to enable extra CUDA debug prints.
//#define DEBUG_CUDA_PRINTS
//#define DEBUG_CUDA_PRINTS2
#define DEBUG_CUDA_PRINTS3
#define DEBUG_CUDA_PRINTS5
// #define DEBUG_INDEX
#define IDX_PRINT
#define DEBUG_CUDA_PRINTS6

#ifdef DEBUG_CUDA_PRINTS
#define DEV_PRINT(...) do { \
    if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0) { \
        printf(__VA_ARGS__); \
    } \
} while(0)
#else
#define DEV_PRINT(...)
#endif

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
__constant__ float* d_alpha1_array;
__constant__ float* d_alpha2_array;
__constant__ int* d_tree_thresh_array;

// ---------------------------------------------------------------------
// Declaration of atomicCAS.
extern __device__ int atomicCAS(int* address, int compare, int val);

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
#ifdef DEBUG_CUDA_PRINTS
    if ((p.x % 100 == 0) && (p.y % 100 == 0))
        printf("[Device] Candidate=(%d,%d): haar_counter=%d, w_index=%d, r_index=%d\n",
            p.x, p.y, haar_counter, w_index, r_index);
#endif

    int* rect = d_rectangles_array + r_index;

    // --- First Rectangle ---
    int tl1_x = p.x + (int)roundf(rect[0] * scaleFactor);
    int tl1_y = p.y + (int)roundf(rect[1] * scaleFactor);
    int br1_x = tl1_x + (int)roundf(rect[2] * scaleFactor);
    int br1_y = tl1_y + (int)roundf(rect[3] * scaleFactor);

#ifdef DEBUG_CUDA_PRINTS6
    // Print candidate coordinate, scale factor, and computed rectangle for debugging.
    printf("[DEBUG] scaleFactor = %f, candidate=(%d,%d)\n", scaleFactor, p.x, p.y);
    printf("[DEBUG] Rect1 computed: tl=(%d,%d), br=(%d,%d) | Integral dims: width=%d, height=%d\n",
        tl1_x, tl1_y, br1_x, br1_y, d_cascade->sum.width, d_cascade->sum.height);
#endif

#ifdef DEBUG_INDEX
    if (p.x == 2015 && p.y == 863) { 
        printf("[GPU DEBUG] Candidate=(%d,%d) Rect1: tl=(%d,%d), br=(%d,%d)\n",
            p.x, p.y, tl1_x, tl1_y, br1_x, br1_y);
    }
#endif

#ifdef DEBUG_CUDA_PRINTS
    if ((p.x % 100 == 0) && (p.y % 100 == 0)) {
        printf("[Device DEBUG] Rect1 raw: offset=(%d,%d), size=(%d,%d)\n",
            rect[0], rect[1], rect[2], rect[3]);
        printf("[Device DEBUG] Rect1 absolute: tl=(%d,%d), br=(%d,%d)\n",
            tl1_x, tl1_y, br1_x, br1_y);
        printf("[Device DEBUG] Integral image bounds: width=%d, height=%d\n",
            d_cascade->sum.width, d_cascade->sum.height);
    }
#endif

    // Check bounds
    assert(tl1_x >= 0 && tl1_x < d_cascade->sum.width);
    assert(tl1_y >= 0 && tl1_y < d_cascade->sum.height);
    assert(br1_x >= 0 && br1_x < d_cascade->sum.width);
    assert(br1_y >= 0 && br1_y < d_cascade->sum.height);

    int idx_tl1 = tl1_y * d_cascade->sum.width + tl1_x;
    int idx_tr1 = tl1_y * d_cascade->sum.width + br1_x;
    int idx_bl1 = br1_y * d_cascade->sum.width + tl1_x;
    int idx_br1 = br1_y * d_cascade->sum.width + br1_x;

#ifdef DEBUG_INDEX
    if (p.x == 2015 && p.y == 863) {
        printf("[GPU DEBUG] Rect1 indices: tl=%d, tr=%d, bl=%d, br=%d\n",
            idx_tl1, idx_tr1, idx_bl1, idx_br1);
        printf("[GPU DEBUG] Integral image dims: width=%d, height=%d\n",
            d_cascade->sum.width, d_cascade->sum.height);
    }
#endif

#ifdef DEBUG_CUDA_PRINTS
    if ((p.x % 100 == 0) && (p.y % 100 == 0))
        printf("[Device DEBUG] Rect1 indices: tl=%d, tr=%d, bl=%d, br=%d\n",
            idx_tl1, idx_tr1, idx_bl1, idx_br1);
#endif

    int sum1 = d_cascade->p0[idx_br1] - d_cascade->p0[idx_tr1] - d_cascade->p0[idx_bl1] + d_cascade->p0[idx_tl1];
    sum1 = sum1 * d_weights_array[w_index + 0];

    // --- Second Rectangle ---
    int tl2_x = p.x + (int)roundf(rect[4] * scaleFactor);
    int tl2_y = p.y + (int)roundf(rect[5] * scaleFactor);
    int br2_x = tl2_x + (int)roundf(rect[6] * scaleFactor);
    int br2_y = tl2_y + (int)roundf(rect[7] * scaleFactor);

#ifdef DEBUG_CUDA_PRINTS6
    printf("[DEBUG] Rect2 computed: tl=(%d,%d), br=(%d,%d)\n",
        tl2_x, tl2_y, br2_x, br2_y);
#endif

#ifdef DEBUG_CUDA_PRINTS
    if ((p.x % 100 == 0) && (p.y % 100 == 0)) {
        printf("[Device DEBUG] Rect2 raw: offset=(%d,%d), size=(%d,%d)\n",
            rect[4], rect[5], rect[6], rect[7]);
        printf("[Device DEBUG] Rect2 absolute: tl=(%d,%d), br=(%d,%d)\n",
            tl2_x, tl2_y, br2_x, br2_y);
    }
#endif

    assert(tl2_x >= 0 && tl2_x < d_cascade->sum.width);
    assert(tl2_y >= 0 && tl2_y < d_cascade->sum.height);
    assert(br2_x >= 0 && br2_x < d_cascade->sum.width);
    assert(br2_y >= 0 && br2_y < d_cascade->sum.height);

    int idx_tl2 = tl2_y * d_cascade->sum.width + tl2_x;
    int idx_tr2 = tl2_y * d_cascade->sum.width + br2_x;
    int idx_bl2 = br2_y * d_cascade->sum.width + tl2_x;
    int idx_br2 = br2_y * d_cascade->sum.width + br2_x;

#ifdef DEBUG_CUDA_PRINTS
    if ((p.x % 100 == 0) && (p.y % 100 == 0))
        printf("[Device DEBUG] Rect2 indices: tl=%d, tr=%d, bl=%d, br=%d\n",
            idx_tl2, idx_tr2, idx_bl2, idx_br2);
#endif

    int sum2 = d_cascade->p0[idx_br2] - d_cascade->p0[idx_tr2] - d_cascade->p0[idx_bl2] + d_cascade->p0[idx_tl2];
    sum2 = sum2 * d_weights_array[w_index + 1];

    int total_sum = sum1 + sum2;

    // --- Third Rectangle (if present) ---
    if (d_weights_array[w_index + 2] != 0)
    {
        int tl3_x = p.x + (int)roundf(rect[8] * scaleFactor);
        int tl3_y = p.y + (int)roundf(rect[9] * scaleFactor);
        int br3_x = tl3_x + (int)roundf(rect[10] * scaleFactor);
        int br3_y = tl3_y + (int)roundf(rect[11] * scaleFactor);

#ifdef DEBUG_CUDA_PRINTS6
        printf("[DEBUG] Rect3 computed: tl=(%d,%d), br=(%d,%d)\n",
            tl3_x, tl3_y, br3_x, br3_y);
#endif

#ifdef DEBUG_CUDA_PRINTS
        if ((p.x % 100 == 0) && (p.y % 100 == 0)) {
            printf("[Device DEBUG] Rect3 raw: offset=(%d,%d), size=(%d,%d)\n",
                rect[8], rect[9], rect[10], rect[11]);
            printf("[Device DEBUG] Rect3 absolute: tl=(%d,%d), br=(%d,%d)\n",
                tl3_x, tl3_y, br3_x, br3_y);
        }
#endif
        assert(tl3_x >= 0 && tl3_x < d_cascade->sum.width);
        assert(tl3_y >= 0 && tl3_y < d_cascade->sum.height);
        assert(br3_x >= 0 && br3_x < d_cascade->sum.width);
        assert(br3_y >= 0 && br3_y < d_cascade->sum.height);

        int idx_tl3 = tl3_y * d_cascade->sum.width + tl3_x;
        int idx_tr3 = tl3_y * d_cascade->sum.width + br3_x;
        int idx_bl3 = br3_y * d_cascade->sum.width + tl3_x;
        int idx_br3 = br3_y * d_cascade->sum.width + br3_x;
#ifdef DEBUG_CUDA_PRINTS
        if ((p.x % 100 == 0) && (p.y % 100 == 0))
            printf("[Device DEBUG] Rect3 indices: tl=%d, tr=%d, bl=%d, br=%d\n",
                idx_tl3, idx_tr3, idx_bl3, idx_br3);
#endif
        int sum3 = d_cascade->p0[idx_br3] - d_cascade->p0[idx_tr3] - d_cascade->p0[idx_bl3] + d_cascade->p0[idx_tl3];
        total_sum += sum3 * d_weights_array[w_index + 2];
    }

#ifdef DEBUG_CUDA_PRINTS5
    if (p.x == 2015 && p.y == 863 && haar_counter == 0) {
        int sum3 = 0;
        if (d_weights_array[w_index + 2] != 0) {
            int tl3_x = p.x + rect[8];
            int tl3_y = p.y + rect[9];
            int br3_x = tl3_x + rect[10];
            int br3_y = tl3_y + rect[11];
            int idx_tl3 = tl3_y * d_cascade->sum.width + tl3_x;
            int idx_tr3 = tl3_y * d_cascade->sum.width + br3_x;
            int idx_bl3 = br3_y * d_cascade->sum.width + tl3_x;
            int idx_br3 = br3_y * d_cascade->sum.width + br3_x;
            sum3 = d_cascade->p0[idx_br3] - d_cascade->p0[idx_tr3] - d_cascade->p0[idx_bl3] + d_cascade->p0[idx_tl3];
            sum3 = sum3 * d_weights_array[w_index + 2];
        }
        printf("[GPU DEBUG] Candidate (%d,%d), Stage 0, Feature 0: sum1 = %d, sum2 = %d, sum3 = %d, final_sum = %d\n",
            p.x, p.y, sum1, sum2, sum3, sum1 + sum2 + sum3);
    }
#endif

#ifdef DEBUG_CUDA_PRINTS5
    if (p.x == 4037 && p.y == 2397 && haar_counter == 0) {
        int sum3 = 0;
        if (d_weights_array[w_index + 2] != 0) {
            int tl3_x = p.x + rect[8];
            int tl3_y = p.y + rect[9];
            int br3_x = tl3_x + rect[10];
            int br3_y = tl3_y + rect[11];
            int idx_tl3 = tl3_y * d_cascade->sum.width + tl3_x;
            int idx_tr3 = tl3_y * d_cascade->sum.width + br3_x;
            int idx_bl3 = br3_y * d_cascade->sum.width + tl3_x;
            int idx_br3 = br3_y * d_cascade->sum.width + br3_x;
            sum3 = d_cascade->p0[idx_br3] - d_cascade->p0[idx_tr3] - d_cascade->p0[idx_bl3] + d_cascade->p0[idx_tl3];
            sum3 = sum3 * d_weights_array[w_index + 2];
        }
        printf("[GPU DEBUG] Candidate (%d,%d), Stage 0, Feature 0: sum1 = %d, sum2 = %d, sum3 = %d, final_sum = %d\n",
            p.x, p.y, sum1, sum2, sum3, sum1 + sum2 + sum3);
    }
#endif

#ifdef DEBUG_CUDA_PRINTS5
    if (p.x == 1154 && p.y == 883 && haar_counter == 0) {
        int sum3 = 0;
        if (d_weights_array[w_index + 2] != 0) {
            int tl3_x = p.x + rect[8];
            int tl3_y = p.y + rect[9];
            int br3_x = tl3_x + rect[10];
            int br3_y = tl3_y + rect[11];
            int idx_tl3 = tl3_y * d_cascade->sum.width + tl3_x;
            int idx_tr3 = tl3_y * d_cascade->sum.width + br3_x;
            int idx_bl3 = br3_y * d_cascade->sum.width + tl3_x;
            int idx_br3 = br3_y * d_cascade->sum.width + br3_x;
            sum3 = d_cascade->p0[idx_br3] - d_cascade->p0[idx_tr3] - d_cascade->p0[idx_bl3] + d_cascade->p0[idx_tl3];
            sum3 = sum3 * d_weights_array[w_index + 2];
        }
        printf("[GPU DEBUG] Candidate (%d,%d), Stage 0, Feature 0: sum1 = %d, sum2 = %d, sum3 = %d, final_sum = %d\n",
            p.x, p.y, sum1, sum2, sum3, sum1 + sum2 + sum3);
    }
#endif


    int t =d_tree_thresh_array[haar_counter] * (variance_norm_factor);

#ifdef DEBUG_CUDA_PRINTS6
    printf("[DEBUG] Weak classifier: total_sum=%d, threshold=%d\n", total_sum, t);
#endif

#ifdef DEBUG_CUDA_PRINTS
    if ((p.x % 100 == 0) && (p.y % 100 == 0))
        printf("[Device] Weak classifier: total_sum=%d, threshold=%d, candidate=(%d,%d) returns %d\n",
            total_sum, t, p.x, p.y,
            (total_sum >= t ? d_alpha2_array[haar_counter] : d_alpha1_array[haar_counter]));
#endif

    return (total_sum >= t) ? (float)d_alpha2_array[haar_counter] : (float)d_alpha1_array[haar_counter];
}


// ---------------------------------------------------------------------
// Device function: Run the cascade classifier on a candidate window.
__device__ int runCascadeClassifier_device(MyIntImage* d_sum, MyIntImage* d_sqsum, 
    const myCascade* d_cascade, MyPoint p, int start_stage, float scaleFactor)
{
    // Ensure candidate window is within bounds.
    assert(p.x >= 0 && p.x < d_cascade->sum.width);
    assert(p.y >= 0 && p.y < d_cascade->sum.height);

#ifdef DEBUG_CUDA_PRINTS6
    printf("[DEBUG] runCascadeClassifier_device: candidate=(%d,%d), scaleFactor=%f, sum dims=(%d,%d)\n",
        p.x, p.y, scaleFactor, d_cascade->sum.width, d_cascade->sum.height);
#endif

    int p_offset = p.y * d_cascade->sum.width + p.x;
    assert(p_offset < d_cascade->sum.width * d_cascade->sum.height);

    int pq_offset = p.y * d_cascade->sqsum.width + p.x;
    assert(pq_offset < d_cascade->sqsum.width * d_cascade->sqsum.height);
    
#ifdef IDX_PRINT
    printf("[GPU DEBUG: IDX] Candidate=(%d,%d): p_offset=%d, pq_offset=%d, sum dims=(%d,%d), sqsum dims=(%d,%d)\n",
        p.x, p.y, p_offset, pq_offset,
        d_cascade->sum.width, d_cascade->sum.height,
        d_cascade->sqsum.width, d_cascade->sqsum.height);
#endif

    // Compute the mean and variance from the integral images.
    unsigned int var_norm = (d_cascade->pq0[pq_offset] - d_cascade->pq1[pq_offset]
        - d_cascade->pq2[pq_offset] + d_cascade->pq3[pq_offset]);
    unsigned int mean = (d_cascade->p0[p_offset] - d_cascade->p1[p_offset]
        - d_cascade->p2[p_offset] + d_cascade->p3[p_offset]);

#ifdef IDX_PRINT
    printf("[GPU DEBUG: IDX] Candidate=(%d,%d): mean=%u, initial var_norm=%u\n", p.x, p.y, mean, var_norm);
#endif

    var_norm = (var_norm * d_cascade->inv_window_area) - mean * mean;
    if (var_norm > 0)
        var_norm = (unsigned int)sqrtf((float)var_norm);
    else
        var_norm = 1;

#ifdef IDX_PRINT
    printf("[GPU DEBUG: IDX] Candidate=(%d,%d): final var_norm=%u\n", p.x, p.y, var_norm);
#endif

    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    float stage_sum = 0.0f;

    for (int i = start_stage; i < d_cascade->n_stages; i++) {
        stage_sum = 0.0f;
        int num_features = d_stages_array[i];

#ifdef IDX_PRINT
        printf("[GPU DEBUG: IDX] Stage %d: Candidate=(%d,%d), num_features=%d\n", i, p.x, p.y, num_features);
#endif

#ifdef DEBUG_CUDA_PRINTS
        if ((p.x % 100 == 0) && (p.y % 100 == 0))
            printf("[Device DEBUG] Stage %d: num_features=%d\n", i, num_features);
#endif
        for (int j = 0; j < num_features; j++) {
            // Make sure we don't exceed the total number of nodes.

            if (haar_counter >= d_cascade->total_nodes) {
#ifdef DEBUG_CUDA_PRINTS6
                printf("[ERROR] haar_counter (%d) reached total_nodes (%d). Candidate accepted.\n",
                    haar_counter, d_cascade->total_nodes);
#endif
                return 1;
            }

#ifdef IDX_PRINT
            printf("[GPU DEBUG: IDX] Before eval: Candidate=(%d,%d), haar_counter=%d, w_index=%d, r_index=%d\n",
                p.x, p.y, haar_counter, w_index, r_index);
#endif

            // Compute the feature response.
            int feature_result = evalWeakClassifier_device(d_cascade, var_norm, p, 
                haar_counter, w_index, r_index, scaleFactor);

#ifdef IDX_PRINT
            printf("[GPU DEBUG] After eval: Candidate=(%d,%d), Stage %d, Feature %d, feature_result=%d\n",
                p.x, p.y, i, j, feature_result);
#endif

#ifdef DEBUG_CUDA_PRINTS2
            if ((p.x >= 2619 && p.x < 2619 + 316) &&
                (p.y >= 693 && p.y < 693 + 307)) {
                printf("[Device DEBUG] ROI Candidate (%d,%d): Stage %d, Feature %d: result = %d\n", p.x, p.y, i, j, feature_result);
            }
#endif

#ifdef DEBUG_CUDA_PRINTS4
            if (p.x == 2015 && p.y == 863) {  // Same location as CPU
                printf("[GPU DEBUG] Candidate (%d,%d), Stage 0, Feature 0: response = %d\n", p.x, p.y, feature_result);
            }

#endif
            stage_sum += feature_result;
            haar_counter++;
            w_index += 3;    // advance the weight index by 3 (since 3 weights per feature)
            r_index += 12;   // advance the rectangle index by 12 (since 12 ints per feature)
        }
#ifdef DEBUG_CUDA_PRINTS
        if ((p.x % 100 == 0) && (p.y % 100 == 0))
            printf("[Device DEBUG] Stage %d: stage_sum=%f, threshold=%f\n", i, stage_sum, d_stages_thresh_array[i]);
#endif
        if (stage_sum < d_stages_thresh_array[i])
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

#ifdef DEBUG_CUDA_PRINTS2
    if (x % 100 == 0 && y % 100 == 0)
        printf("[Device DEBUG] Processing candidate window at (%d,%d)\n", x, y);
#endif

    int result = runCascadeClassifier_device(d_sum, d_sqsum, d_cascade, p, 0, scaleFactor);

#ifdef DEBUG_CUDA_PRINTS2
    if (x % 100 == 0 && y % 100 == 0)
        printf("[Device DEBUG] runCascadeClassifier_device returned: %d for window (%d,%d)\n", result, x, y);
#endif

    if (result > 0) {
        MyRect r;
        r.x = (int)roundf(x * scaleFactor);
        r.y = (int)roundf(y * scaleFactor);
        r.width = (int)roundf(d_cascade->orig_window_size.width * scaleFactor);
        r.height = (int)roundf(d_cascade->orig_window_size.height * scaleFactor);
        int idx = atomicAdd(d_candidateCount, 1);
        if (idx < maxCandidates) {
            d_candidates[idx] = r;
        }
       
#ifdef DEBUG_CUDA_PRINTS
        if (x % 100 == 0 && y % 100 == 0)
            printf("[Device DEBUG] Candidate added at index %d for window (%d,%d)\n", idx, x, y);
#endif
    }
}

std::vector<MyRect> runDetection(MyIntImage* h_sum, MyIntImage* h_sqsum, myCascade* cascade, int maxCandidates, float scaleFactor)
{
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] runDetection() started.\n");
#endif
    std::vector<MyRect> candidates;

    // --- Step 1: Allocate Unified Memory for sum integral image ---
    int dataSize = h_sum->width * h_sum->height * sizeof(int);
    MyIntImage* d_sum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sum->data), dataSize));
    memcpy(d_sum->data, h_sum->data, dataSize);
    d_sum->width = h_sum->width;
    d_sum->height = h_sum->height;
    printf("[DEBUG] d_sum allocated at %p; dimensions: width=%d, height=%d\n", (void*)d_sum, d_sum->width, d_sum->height);

    // --- Step 2: Allocate Unified Memory for squared sum integral image ---
    dataSize = h_sqsum->width * h_sqsum->height * sizeof(int);
    MyIntImage* d_sqsum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sqsum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sqsum->data), dataSize));
    memcpy(d_sqsum->data, h_sqsum->data, dataSize);
    d_sqsum->width = h_sqsum->width;
    d_sqsum->height = h_sqsum->height;
    printf("[DEBUG] d_sqsum allocated at %p; dimensions: width=%d, height=%d\n", (void*)d_sqsum, d_sqsum->width, d_sqsum->height);

    // --- Step 5: Allocate Unified Memory for the cascade structure ---
    myCascade* d_cascade = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_cascade, sizeof(myCascade)));
    *d_cascade = *cascade;
    printf("[Host DEBUG] d_cascade allocated at %p, n_stages=%d, total_nodes=%d\n", (void*)d_cascade, d_cascade->n_stages, d_cascade->total_nodes);


    // --- Step 3: Update the cascade with Unified Memory pointers for integral images ---
        // Copy MyIntImage structs themselves
    d_cascade->sum = *d_sum;
    d_cascade->sqsum = *d_sqsum;

    // Correct the data pointers within d_cascade->sum and d_cascade->sqsum
    d_cascade->sum.data = d_sum->data;   // Point to the Unified Memory data buffer of d_sum
    d_cascade->sqsum.data = d_sqsum->data; // Point to the Unified Memory data buffer of d_sqsum

	// Update dimensions of the integral images in the cascade structure
    d_cascade->sum.width = d_sum->width;
    d_cascade->sum.height = d_sum->height;
    d_cascade->sqsum.width = d_sqsum->width;
    d_cascade->sqsum.height = d_sqsum->height;

    // Update cascade corner pointers to reflect the new unified memory pointers
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


    // --- Step 4: Transfer classifier parameters to device memory ---
    int* d_stages_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_stages_array_dev, cascade->n_stages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_stages_array_dev, cascade->stages_array, cascade->n_stages * sizeof(int), cudaMemcpyHostToDevice));
    printf("[DEBUG] d_stages_array_dev allocated at %p\n", (void*)d_stages_array_dev);

    float* d_stages_thresh_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_stages_thresh_array_dev, cascade->n_stages * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_stages_thresh_array_dev, cascade->stages_thresh_array, cascade->n_stages * sizeof(float), cudaMemcpyHostToDevice));
    printf("[DEBUG] d_stages_thresh_array_dev allocated at %p\n", (void*)d_stages_thresh_array_dev);

    int* d_rectangles_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_rectangles_array_dev, cascade->total_nodes * 12 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rectangles_array_dev, cascade->rectangles_array, cascade->total_nodes * 12 * sizeof(int), cudaMemcpyHostToDevice));
    printf("[DEBUG] d_rectangles_array_dev allocated at %p\n", (void*)d_rectangles_array_dev);

    int* d_weights_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_weights_array_dev, cascade->total_nodes * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_weights_array_dev, cascade->weights_array, cascade->total_nodes * 3 * sizeof(int), cudaMemcpyHostToDevice));
    printf("[DEBUG] d_weights_array_dev allocated at %p\n", (void*)d_weights_array_dev);

    float* d_alpha1_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_alpha1_array_dev, cascade->total_nodes * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_alpha1_array_dev, cascade->alpha1_array, cascade->total_nodes * sizeof(float), cudaMemcpyHostToDevice));
    printf("[DEBUG] d_alpha1_array_dev allocated at %p\n", (void*)d_alpha1_array_dev);

    float* d_alpha2_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_alpha2_array_dev, cascade->total_nodes * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_alpha2_array_dev, cascade->alpha2_array, cascade->total_nodes * sizeof(float), cudaMemcpyHostToDevice));
    printf("[DEBUG] d_alpha2_array_dev allocated at %p\n", (void*)d_alpha2_array_dev);

    int* d_tree_thresh_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_tree_thresh_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tree_thresh_array_dev, cascade->tree_thresh_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));
    printf("[DEBUG] d_tree_thresh_array_dev allocated at %p\n", (void*)d_tree_thresh_array_dev);

    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_array, &d_stages_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_thresh_array, &d_stages_thresh_array_dev, sizeof(float*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_rectangles_array, &d_rectangles_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_weights_array, &d_weights_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha1_array, &d_alpha1_array_dev, sizeof(float*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha2_array, &d_alpha2_array_dev, sizeof(float*)));
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
    // Define margin values
    const int margin_x = 0;  // maximum extra x offset from a rectangle feature
    const int margin_y = 0;  // maximum extra y offset from a rectangle feature

    // Compute search space with additional margins:
    int x_max = d_sum->width - cascade->orig_window_size.width - margin_x;
    int y_max = d_sum->height - cascade->orig_window_size.height - margin_y;
    printf("[Host DEBUG] Search space dimensions (with margin): x_max=%d, y_max=%d\n", x_max, y_max);

    dim3 blockDim(16, 16);
    dim3 gridDim((x_max + blockDim.x - 1) / blockDim.x,
        (y_max + blockDim.y - 1) / blockDim.y);
    printf("[Host] Launching kernel with gridDim=(%d, %d), blockDim=(%d, %d)\n",
        gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Prepare host copies (passed by value) of the unified memory structures.
    MyIntImage h_sumStruct = *d_sum;
    MyIntImage h_sqsumStruct = *d_sqsum;
    myCascade h_cascadeStruct = *d_cascade;
    printf("[Host] Cascade structure on device (Unified Memory):\n");
    printf("  n_stages = %d\n", h_cascadeStruct.n_stages);
    printf("  total_nodes = %d\n", h_cascadeStruct.total_nodes);
    printf("  orig_window_size = (%d, %d)\n", h_cascadeStruct.orig_window_size.width, h_cascadeStruct.orig_window_size.height);
    printf("  inv_window_area = %f\n", h_cascadeStruct.inv_window_area);

    // Create a stuct that contains all kernel arguments
    struct KernelArgs {
        MyIntImage* d_sum;
        MyIntImage* d_sqsum;
        myCascade* d_cascade;
        float scaleFactor;
        int x_max;
        int y_max;
        MyRect* d_candidates;
        int* d_candidateCount;
        int maxCandidates;
    };

    // Fill an instance of kernel args struct
    KernelArgs args;
    args.d_sum = d_sum;
    args.d_sqsum = d_sqsum;
    args.d_cascade = d_cascade;
    args.scaleFactor = scaleFactor;
    args.x_max = x_max;
    args.y_max = y_max;
    args.d_candidates = d_candidates;
    args.d_candidateCount = d_candidateCount;
    args.maxCandidates = maxCandidates;

    void* kernelArgs[] = { &args };


    // --- Debug prints to verify device pointers and kernel arguments ---
    printf("[DEBUG] Verifying device pointers and kernel arguments:\n");
    printf("    h_sumStruct.data = %p\n", (void*)h_sumStruct.data);
    printf("    h_sqsumStruct.data = %p\n", (void*)h_sqsumStruct.data);
    printf("    h_cascadeStruct.p0 = %p\n", (void*)h_cascadeStruct.p0);
    printf("    h_cascadeStruct.p1 = %p\n", (void*)h_cascadeStruct.p1);
    printf("    h_cascadeStruct.p2 = %p\n", (void*)h_cascadeStruct.p2);
    printf("    h_cascadeStruct.p3 = %p\n", (void*)h_cascadeStruct.p3);
    printf("    h_cascadeStruct.pq0 = %p\n", (void*)h_cascadeStruct.pq0);
    printf("    h_cascadeStruct.pq1 = %p\n", (void*)h_cascadeStruct.pq1);
    printf("    h_cascadeStruct.pq2 = %p\n", (void*)h_cascadeStruct.pq2);
    printf("    h_cascadeStruct.pq3 = %p\n", (void*)h_cascadeStruct.pq3);

    printf("    scaleFactor = %f\n", scaleFactor);
    printf("    x_max = %d, y_max = %d\n", x_max, y_max);

    printf("    d_candidates pointer = %p\n", (void*)d_candidates);
    printf("    d_candidateCount pointer = %p, initial value = %d\n", (void*)d_candidateCount, *d_candidateCount);

    printf("--- Debug: GPU stages_array (first 10 elements) ---\n");
    for (int i = 0; i < 10; i++) {
        printf("  stages_array[%d] = %d\n", i, d_cascade->stages_array[i]);
    }

    printf("--- Debug: GPU stages_thresh_array (first 10 elements) ---\n");
    for (int i = 0; i < 10; i++) {
        printf("  stages_thresh_array[%d] = %f\n", i, d_cascade->stages_thresh_array[i]);
    }

    printf("--- Debug: GPU rectangles_array (first 10 elements) ---\n");
    for (int i = 0; i < 10; i++) {
        printf("  rectangles_array[%d] = %d\n", i, d_cascade->rectangles_array[i]);
    }

    printf("--- Debug: GPU weights_array (first 10 elements) ---\n");
    for (int i = 0; i < 10; i++) {
        printf("  weights_array[%d] = %d\n", i, d_cascade->weights_array[i]);
    }

    printf("[Host DEBUG] Synchronizing before kernel launch...\n");
    fflush(stdout);

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("[Host] Kernel execution error: %s\n", cudaGetErrorString(syncErr));
    }

    detectKernel << <gridDim, blockDim >> > (d_sum, d_sqsum, d_cascade, scaleFactor, x_max, y_max, d_candidates, d_candidateCount, maxCandidates);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("[Host] cudaLaunchKernel error: %s\n", cudaGetErrorString(launchErr));
    }

    printf("[Host DEBUG] Kernel launched.\n");
    fflush(stdout);



    syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("[Host] Kernel execution error: %s\n", cudaGetErrorString(syncErr));
    }
    printf("[Host DEBUG] Kernel execution completed.\n");

    printf("[DEBUG] d_candidateCount pointer = %p\n", (void*)d_candidateCount);
    int hostCandidateCount = 0;
    cudaError_t memcpyErr = cudaMemcpy(&hostCandidateCount, d_candidateCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (memcpyErr != cudaSuccess) {
        printf("[DEBUG] cudaMemcpy error reading d_candidateCount: %s\n", cudaGetErrorString(memcpyErr));
    }
    else {
        printf("[DEBUG] Read candidate count: %d\n", hostCandidateCount);
    }
    printf("[Host] Detected %d candidate windows.\n", hostCandidateCount);
    if (hostCandidateCount > 0) {
        for (int i = 0; i < hostCandidateCount; i++) {
            candidates.push_back(d_candidates[i]);
        }
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
