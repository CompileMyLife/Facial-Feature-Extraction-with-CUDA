// cuda_detect.cu
//
// This file implements a CUDA‐accelerated sliding window detector for the Viola‐Jones algorithm.
// This version implements the real weak classifier evaluation and uses Unified Memory for the integral images,
// cascade structure, and detection results.

#include "cuda_detect.cuh"    // Include header file for CUDA detection

// Uncomment to enable extra CUDA debug prints.
#define DEBUG_CUDA_PRINTS

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
__constant__ int* d_alpha1_array;
__constant__ int* d_alpha2_array;
__constant__ int* d_tree_thresh_array;

// ---------------------------------------------------------------------
// Declaration of atomicCAS.
extern __device__ int atomicCAS(int* address, int compare, int val);

// ---------------------------------------------------------------------
// Custom atomic add.
__device__ int myAtomicAdd(int* address, int val) {
    int old = *address;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, assumed + val);
    } while (assumed != old);
    return old;
}

// ---------------------------------------------------------------------
// Device function: Evaluate a weak classifier for candidate window p.
// Assumes that for each feature, d_rectangles_array stores 12 ints in the order:
// [x_offset1, y_offset1, width1, height1, x_offset2, y_offset2, width2, height2, x_offset3, y_offset3, width3, height3]
__device__ int evalWeakClassifier_device(const myCascade *d_cascade, int variance_norm_factor, MyPoint p,
    int haar_counter, int w_index, int r_index)
{
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
        printf("[Device] Eval weak classifier: haar_counter=%d, w_index=%d, r_index=%d, candidate=(%d,%d)\n",
               haar_counter, w_index, r_index, p.x, p.y);
#endif

    int *rect = d_rectangles_array + r_index * 12;

    // First rectangle:
    int tl1_x = p.x + rect[0];
    int tl1_y = p.y + rect[1];
    int br1_x = tl1_x + rect[2];
    int br1_y = tl1_y + rect[3];
    // Check bounds:
    assert(tl1_x >= 0 && tl1_x < d_cascade->sum.width);
    assert(tl1_y >= 0 && tl1_y < d_cascade->sum.height);
    assert(br1_x >= 0 && br1_x < d_cascade->sum.width);
    assert(br1_y >= 0 && br1_y < d_cascade->sum.height);
    int idx_tl1 = tl1_y * d_cascade->sum.width + tl1_x;
    int idx_tr1 = tl1_y * d_cascade->sum.width + br1_x;
    int idx_bl1 = br1_y * d_cascade->sum.width + tl1_x;
    int idx_br1 = br1_y * d_cascade->sum.width + br1_x;
#ifdef DEBUG_CUDA_PRINTS
    printf("[Device DEBUG] Rect1: tl=(%d,%d) br=(%d,%d) => idx_tl=%d, idx_tr=%d, idx_bl=%d, idx_br=%d\n",
           tl1_x, tl1_y, br1_x, br1_y, idx_tl1, idx_tr1, idx_bl1, idx_br1);
#endif
    int sum1 = d_cascade->p0[idx_br1] - d_cascade->p0[idx_tr1] - d_cascade->p0[idx_bl1] + d_cascade->p0[idx_tl1];
    sum1 = sum1 * d_weights_array[w_index * 3 + 0];

    // Second rectangle:
    int tl2_x = p.x + rect[4];
    int tl2_y = p.y + rect[5];
    int br2_x = tl2_x + rect[6];
    int br2_y = tl2_y + rect[7];
    assert(tl2_x >= 0 && tl2_x < d_cascade->sum.width);
    assert(tl2_y >= 0 && tl2_y < d_cascade->sum.height);
    assert(br2_x >= 0 && br2_x < d_cascade->sum.width);
    assert(br2_y >= 0 && br2_y < d_cascade->sum.height);
    int idx_tl2 = tl2_y * d_cascade->sum.width + tl2_x;
    int idx_tr2 = tl2_y * d_cascade->sum.width + br2_x;
    int idx_bl2 = br2_y * d_cascade->sum.width + tl2_x;
    int idx_br2 = br2_y * d_cascade->sum.width + br2_x;
#ifdef DEBUG_CUDA_PRINTS
    printf("[Device DEBUG] Rect2: tl=(%d,%d) br=(%d,%d) => idx_tl=%d, idx_tr=%d, idx_bl=%d, idx_br=%d\n",
           tl2_x, tl2_y, br2_x, br2_y, idx_tl2, idx_tr2, idx_bl2, idx_br2);
#endif
    int sum2 = d_cascade->p0[idx_br2] - d_cascade->p0[idx_tr2] - d_cascade->p0[idx_bl2] + d_cascade->p0[idx_tl2];
    // In the original code, the second weight is added.
    sum2 = sum2 + d_weights_array[w_index * 3 + 1];

    int total_sum = sum1 + sum2;

    // Third rectangle (if present):
    if (d_weights_array[w_index * 3 + 2] != 0)
    {
        int tl3_x = p.x + rect[8];
        int tl3_y = p.y + rect[9];
        int br3_x = tl3_x + rect[10];
        int br3_y = tl3_y + rect[11];
        assert(tl3_x >= 0 && tl3_x < d_cascade->sum.width);
        assert(tl3_y >= 0 && tl3_y < d_cascade->sum.height);
        assert(br3_x >= 0 && br3_x < d_cascade->sum.width);
        assert(br3_y >= 0 && br3_y < d_cascade->sum.height);
        int idx_tl3 = tl3_y * d_cascade->sum.width + tl3_x;
        int idx_tr3 = tl3_y * d_cascade->sum.width + br3_x;
        int idx_bl3 = br3_y * d_cascade->sum.width + tl3_x;
        int idx_br3 = br3_y * d_cascade->sum.width + br3_x;
#ifdef DEBUG_CUDA_PRINTS
        printf("[Device DEBUG] Rect3: tl=(%d,%d) br=(%d,%d) => idx_tl=%d, idx_tr=%d, idx_bl=%d, idx_br=%d\n",
               tl3_x, tl3_y, br3_x, br3_y, idx_tl3, idx_tr3, idx_bl3, idx_br3);
#endif
        int sum3 = d_cascade->p0[idx_br3] - d_cascade->p0[idx_tr3] - d_cascade->p0[idx_bl3] + d_cascade->p0[idx_tl3];
        total_sum += sum3 * d_weights_array[w_index * 3 + 2];
    }

    int t = d_tree_thresh_array[haar_counter] * variance_norm_factor;
#ifdef DEBUG_CUDA_PRINTS
    if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
        printf("[Device] Weak classifier: total_sum=%d, threshold=%d, returning %d\n",
               total_sum, t, (total_sum >= t ? d_alpha2_array[haar_counter] : d_alpha1_array[haar_counter]));
#endif

    return (total_sum >= t ? d_alpha2_array[haar_counter] : d_alpha1_array[haar_counter]);
}

// ---------------------------------------------------------------------
// Device function: Run the cascade classifier on a candidate window.
__device__ int runCascadeClassifier_device(const myCascade* d_cascade, MyPoint p, int start_stage)
{
    assert(p.x >= 0 && p.x < d_cascade->sum.width);
    assert(p.y >= 0 && p.y < d_cascade->sum.height);

    int p_offset = p.y * d_cascade->sum.width + p.x;
    assert(p_offset < d_cascade->sum.width * d_cascade->sum.height);

    int pq_offset = p.y * d_cascade->sqsum.width + p.x;
    assert(pq_offset < d_cascade->sqsum.width * d_cascade->sqsum.height);

#ifdef DEBUG_CUDA_PRINTS
    if (p.x == 0 && p.y == 0)
        printf("[Device DEBUG] p_offset=%d, pq_offset=%d, sum.width=%d, sum.height=%d, sqsum.width=%d, sqsum.height=%d\n",
               p_offset, pq_offset, d_cascade->sum.width, d_cascade->sum.height, d_cascade->sqsum.width, d_cascade->sqsum.height);
#endif

    unsigned int var_norm = (d_cascade->pq0[pq_offset] - d_cascade->pq1[pq_offset]
                             - d_cascade->pq2[pq_offset] + d_cascade->pq3[pq_offset]);
    unsigned int mean = (d_cascade->p0[p_offset] - d_cascade->p1[p_offset]
                         - d_cascade->p2[p_offset] + d_cascade->p3[p_offset]);
    var_norm = (var_norm * d_cascade->inv_window_area) - mean * mean;
    if (var_norm > 0)
        var_norm = (unsigned int)sqrtf((float)var_norm);
    else
        var_norm = 1;

    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    float stage_sum = 0.0f;

    for (int i = start_stage; i < d_cascade->n_stages; i++) {
        stage_sum = 0.0f;
        int num_features = d_stages_array[i];
#ifdef DEBUG_CUDA_PRINTS
        if (p.x == 0 && p.y == 0)
            printf("[Device DEBUG] Stage %d: num_features=%d\n", i, num_features);
#endif
        for (int j = 0; j < num_features; j++) {
            assert(haar_counter < d_cascade->total_nodes);
            assert(w_index < d_cascade->total_nodes);
            assert(r_index < d_cascade->total_nodes);
            stage_sum += evalWeakClassifier_device(d_cascade, var_norm, p, haar_counter, w_index, r_index);
            haar_counter++;
            w_index++;
            r_index++;
        }
#ifdef DEBUG_CUDA_PRINTS
        if (p.x == 0 && p.y == 0)
            printf("[Device DEBUG] Stage %d: stage_sum=%f, threshold=%f\n", i, stage_sum, d_stages_thresh_array[i]);
#endif
        if (stage_sum < d_stages_thresh_array[i])
            return -i;
    }
    return 1;
}

// ---------------------------------------------------------------------
// CUDA kernel: Each thread processes one candidate window.
__global__ void detectKernel(MyIntImage d_sum, MyIntImage d_sqsum,
    myCascade d_cascade, float factor,
    int x_max, int y_max,
    MyRect* d_candidates, int* d_candidateCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= x_max || y >= y_max)
        return;

    MyPoint p;
    p.x = x;
    p.y = y;

#ifdef DEBUG_CUDA_PRINTS
    if (x == 0 && y == 0)
        printf("[Device DEBUG] Calling runCascadeClassifier_device for window (%d, %d)\n", x, y);
#endif

    int result = runCascadeClassifier_device(&d_cascade, p, 0);
#ifdef DEBUG_CUDA_PRINTS
    if (x == 0 && y == 0)
        printf("[Device DEBUG] runCascadeClassifier_device returned: %d for window (%d, %d)\n", result, x, y);
#endif

    if (result > 0) {
        MyRect r;
        r.x = (int)roundf(x * factor);
        r.y = (int)roundf(y * factor);
        r.width = (int)roundf(d_cascade.orig_window_size.width * factor);
        r.height = (int)roundf(d_cascade.orig_window_size.height * factor);
        int idx = myAtomicAdd(d_candidateCount, 1);
        d_candidates[idx] = r;
    }
}

// ---------------------------------------------------------------------
// Host function: runDetection() manages data transfer, kernel launch, and result retrieval.
std::vector<MyRect> runDetection(MyIntImage* h_sum, MyIntImage* h_sqsum, myCascade* cascade, int maxCandidates, float scaleFactor)
{
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] runDetection() started.\n");
#endif
    std::vector<MyRect> candidates;

    // --- Step 1: Allocate Unified Memory for sum integral image ---
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Allocating Unified Memory for sum integral image.\n");
#endif
    int dataSize = h_sum->width * h_sum->height * sizeof(int);
    MyIntImage* d_sum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sum->data), dataSize));
    memcpy(d_sum->data, h_sum->data, dataSize);
    d_sum->width = h_sum->width;
    d_sum->height = h_sum->height;

    printf("[DEBUG] d_sum dimensions: width=%d, height=%d\n", d_sum->width, d_sum->height);

    // --- Step 2: Allocate Unified Memory for squared sum integral image ---
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Allocating Unified Memory for squared sum integral image.\n");
#endif
    dataSize = h_sqsum->width * h_sqsum->height * sizeof(int);
    MyIntImage* d_sqsum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sqsum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sqsum->data), dataSize));
    memcpy(d_sqsum->data, h_sqsum->data, dataSize);
    d_sqsum->width = h_sqsum->width;
    d_sqsum->height = h_sqsum->height;

    printf("[DEBUG] d_sqsum dimensions: width=%d, height=%d\n", d_sqsum->width, d_sqsum->height);

    // --- Step 3: Update the cascade with Unified Memory pointers for integral images ---
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Updating cascade structure with Unified Memory pointers for integral images.\n");
#endif
    cascade->p0 = d_sum->data;
    cascade->p1 = d_sum->data + (d_sum->width - 1);
    cascade->p2 = d_sum->data + (d_sum->width * (d_sum->height - 1));
    cascade->p3 = d_sum->data + (d_sum->width * (d_sum->height - 1) + (d_sum->width - 1));
    cascade->pq0 = d_sqsum->data;
    cascade->pq1 = d_sqsum->data + (d_sqsum->width - 1);
    cascade->pq2 = d_sqsum->data + (d_sqsum->width * (d_sqsum->height - 1));
    cascade->pq3 = d_sqsum->data + (d_sqsum->width * (d_sqsum->height - 1) + (d_sqsum->width - 1));

    printf("[Host] Cascade pointers for integral images:\n");
    printf("  p0  = %p\n", (void*)cascade->p0);
    printf("  p1  = %p\n", (void*)cascade->p1);
    printf("  p2  = %p\n", (void*)cascade->p2);
    printf("  p3  = %p\n", (void*)cascade->p3);
    printf("  pq0 = %p\n", (void*)cascade->pq0);
    printf("  pq1 = %p\n", (void*)cascade->pq1);
    printf("  pq2 = %p\n", (void*)cascade->pq2);
    printf("  pq3 = %p\n", (void*)cascade->pq3);

    // --- Step 4: Transfer classifier parameters to device memory (unchanged) ---
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Transferring classifier parameters to device memory.\n");
#endif
    int* d_stages_array_dev = nullptr;
    float* d_stages_thresh_array_dev = nullptr;
    int* d_rectangles_array_dev = nullptr;
    int* d_weights_array_dev = nullptr;
    int* d_alpha1_array_dev = nullptr;
    int* d_alpha2_array_dev = nullptr;
    int* d_tree_thresh_array_dev = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_stages_array_dev, cascade->n_stages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_stages_array_dev, cascade->stages_array, cascade->n_stages * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_stages_thresh_array_dev, cascade->n_stages * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_stages_thresh_array_dev, cascade->stages_thresh_array, cascade->n_stages * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_rectangles_array_dev, cascade->total_nodes * 12 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rectangles_array_dev, cascade->rectangles_array, cascade->total_nodes * 12 * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_weights_array_dev, cascade->total_nodes * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_weights_array_dev, cascade->weights_array, cascade->total_nodes * 3 * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_alpha1_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_alpha1_array_dev, cascade->alpha1_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_alpha2_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_alpha2_array_dev, cascade->alpha2_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_tree_thresh_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tree_thresh_array_dev, cascade->tree_thresh_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_array, &d_stages_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_thresh_array, &d_stages_thresh_array_dev, sizeof(float*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_rectangles_array, &d_rectangles_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_weights_array, &d_weights_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha1_array, &d_alpha1_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha2_array, &d_alpha2_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_tree_thresh_array, &d_tree_thresh_array_dev, sizeof(int*)));

    // --- Step 5: Allocate Unified Memory for the cascade structure ---
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Allocating Unified Memory for cascade structure.\n");
#endif
    myCascade* d_cascade = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_cascade, sizeof(myCascade)));
    *d_cascade = *cascade;

    // --- Step 6: Allocate Unified Memory for detection results ---
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Allocating Unified Memory for detection results.\n");
#endif
    MyRect* d_candidates = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_candidates, maxCandidates * sizeof(MyRect)));
    int* d_candidateCount = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_candidateCount, sizeof(int)));
    *d_candidateCount = 0;

    int x_max = d_sum->width - cascade->orig_window_size.width;
    int y_max = d_sum->height - cascade->orig_window_size.height;
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Search space dimensions: x_max=%d, y_max=%d\n", x_max, y_max);
#endif

    // --- Step 7: Launch the detection kernel in full-grid mode ---
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Launching detection kernel (full-grid mode).\n");
#endif
    dim3 blockDim(16, 16);
    dim3 gridDim((x_max + blockDim.x - 1) / blockDim.x,
                 (y_max + blockDim.y - 1) / blockDim.y);
    printf("[Host] gridDim=(%d, %d), blockDim=(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    MyIntImage h_sumStruct = *d_sum;
    MyIntImage h_sqsumStruct = *d_sqsum;
    myCascade h_cascadeStruct = *d_cascade;
    printf("[Host] Cascade structure on device (Unified Memory):\n");
    printf("  n_stages = %d\n", h_cascadeStruct.n_stages);
    printf("  total_nodes = %d\n", h_cascadeStruct.total_nodes);
    printf("  orig_window_size = (%d, %d)\n", h_cascadeStruct.orig_window_size.width, h_cascadeStruct.orig_window_size.height);
    printf("  inv_window_area = %f\n", h_cascadeStruct.inv_window_area);

    MyRect* candidatePtr = d_candidates;
    int* candidateCountPtr = d_candidateCount;
    void* kernelArgs[] = {
        (void*)&h_sumStruct,
        (void*)&h_sqsumStruct,
        (void*)&h_cascadeStruct,
        (void*)&scaleFactor,
        (void*)&x_max,
        (void*)&y_max,
        (void*)&candidatePtr,
        (void*)&candidateCountPtr
    };

    cudaError_t launchErr = cudaLaunchKernel((const void*)detectKernel, gridDim, blockDim, kernelArgs, 0, 0);
    if (launchErr != cudaSuccess) {
        printf("[Host] cudaLaunchKernel error: %s\n", cudaGetErrorString(launchErr));
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("[Host] Kernel execution error: %s\n", cudaGetErrorString(syncErr));
    }
#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Kernel execution completed.\n");
#endif

    printf("[DEBUG] d_candidateCount pointer = %p\n", (void*)d_candidateCount);
    int hostCandidateCount = 0;
    cudaError_t memcpyErr = cudaMemcpy(&hostCandidateCount, d_candidateCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (memcpyErr != cudaSuccess) {
        printf("[DEBUG] cudaMemcpy error reading d_candidateCount: %s\n", cudaGetErrorString(memcpyErr));
    }
    else {
        printf("[DEBUG] Read candidate count: %d\n", hostCandidateCount);
    }
    int h_candidateCount = hostCandidateCount;
    printf("[Host] Detected %d candidate windows.\n", h_candidateCount);
    if (h_candidateCount > 0) {
        for (int i = 0; i < h_candidateCount; i++) {
            candidates.push_back(d_candidates[i]);
        }
    }

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] Cleaning up Unified Memory and device memory allocated with cudaMalloc.\n");
#endif
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

#ifdef DEBUG_CUDA_PRINTS
    printf("[Host] runDetection() completed.\n");
#endif
    return candidates;
}
