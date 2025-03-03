#include "cuda_detect.h"
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string.h>
#include <assert.h>

// Define device globals (only in this file).
__device__ int* d_stages_array = nullptr;
__device__ float* d_stages_thresh_array = nullptr;
__device__ int* d_rectangles_array = nullptr;
__device__ int* d_weights_array = nullptr;

// Stub for setImageForCascadeClassifierCUDA.
void setImageForCascadeClassifierCUDA(myCascade* cascade, MyIntImage* sum, MyIntImage* sqsum) {
    // Minimal stub: do nothing.
}

// Minimal stub for loadCascadeClassifierCUDA.
// This allocates device memory for cascade arrays and copies the dummy cascade data.
void loadCascadeClassifierCUDA(myCascade* cascade) {
    cudaError_t err;

    int stagesSize = (cascade->n_stages + 1) * sizeof(int);
    int* stages_array_dev = nullptr;
    err = cudaMalloc((void**)&stages_array_dev, stagesSize);
    if (err != cudaSuccess)
        printf("cudaMalloc for stages_array_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(stages_array_dev, cascade->stages_array, stagesSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        printf("cudaMemcpy for stages_array_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_stages_array, &stages_array_dev, sizeof(stages_array_dev));
    if (err != cudaSuccess)
        printf("cudaMemcpyToSymbol for d_stages_array failed: %s\n", cudaGetErrorString(err));

    int threshSize = cascade->n_stages * sizeof(float);
    float* stages_thresh_dev = nullptr;
    err = cudaMalloc((void**)&stages_thresh_dev, threshSize);
    if (err != cudaSuccess)
        printf("cudaMalloc for stages_thresh_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(stages_thresh_dev, cascade->stages_thresh_array, threshSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        printf("cudaMemcpy for stages_thresh_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_stages_thresh_array, &stages_thresh_dev, sizeof(stages_thresh_dev));
    if (err != cudaSuccess)
        printf("cudaMemcpyToSymbol for d_stages_thresh_array failed: %s\n", cudaGetErrorString(err));

    int numWeak = cascade->total_nodes;
    int rectSize = numWeak * 12 * sizeof(int);
    int* rectangles_dev = nullptr;
    err = cudaMalloc((void**)&rectangles_dev, rectSize);
    if (err != cudaSuccess)
        printf("cudaMalloc for rectangles_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(rectangles_dev, cascade->rectangles_array, rectSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        printf("cudaMemcpy for rectangles_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_rectangles_array, &rectangles_dev, sizeof(rectangles_dev));
    if (err != cudaSuccess)
        printf("cudaMemcpyToSymbol for d_rectangles_array failed: %s\n", cudaGetErrorString(err));

    int weightsSize = numWeak * sizeof(int);
    int* weights_dev = nullptr;
    err = cudaMalloc((void**)&weights_dev, weightsSize);
    if (err != cudaSuccess)
        printf("cudaMalloc for weights_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(weights_dev, cascade->weights_array, weightsSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        printf("cudaMemcpy for weights_dev failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_weights_array, &weights_dev, sizeof(weights_dev));
    if (err != cudaSuccess)
        printf("cudaMemcpyToSymbol for d_weights_array failed: %s\n", cudaGetErrorString(err));

    printf("loadCascadeClassifierCUDA: Cascade data loaded to device memory.\n");
}

// Dummy device function: Compute rectangle sum (placeholder)
__device__ float computeRectangleSum(int x, int y, int width, int height) {
    return 42.0f; // Dummy value
}

// Device function: Evaluate weak classifier.
__device__ float evalWeakClassifier_device(const MyRect candidate, int weakIndex) {
    int offset = weakIndex * 12; // 12 ints per weak classifier
    int rx = d_rectangles_array[offset + 0];
    int ry = d_rectangles_array[offset + 1];
    int rw = d_rectangles_array[offset + 2];
    int rh = d_rectangles_array[offset + 3];
    int absX = candidate.x + rx;
    int absY = candidate.y + ry;
    float rectSum = computeRectangleSum(absX, absY, rw, rh);
    float weightedResponse = rectSum * (float)d_weights_array[weakIndex];
    return weightedResponse;
}

// Device function: Run cascade classifier.
__device__ int runCascadeClassifier_device(const MyRect candidate) {
    int numStages = d_stages_array[0];
    int weakIndex = 0;
    for (int s = 0; s < numStages; s++) {
        int numWeak = d_stages_array[s + 1];
        float stageSum = 0.0f;
        for (int w = 0; w < numWeak; w++) {
            stageSum += evalWeakClassifier_device(candidate, weakIndex);
            weakIndex++;
        }
        if (stageSum < d_stages_thresh_array[s])
            return 0;
    }
    return 1;
}

// CUDA kernel: Each thread processes one candidate window.
__global__ void detectKernel(const MyRect* d_candidates, int* d_flags, int numCandidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCandidates) {
        MyRect candidate = d_candidates[idx];
        int result = runCascadeClassifier_device(candidate);
        d_flags[idx] = result;
    }
}

// Host function: Run detection.
std::vector<MyRect> runDetection(const std::vector<MyRect>& candidateWindows) {
    std::vector<MyRect> detections;
    int numCandidates = candidateWindows.size();
    if (numCandidates == 0)
        return detections;

    MyRect* d_candidates = nullptr;
    cudaMalloc((void**)&d_candidates, numCandidates * sizeof(MyRect));
    cudaMemcpy(d_candidates, candidateWindows.data(), numCandidates * sizeof(MyRect), cudaMemcpyHostToDevice);

    int* d_flags = nullptr;
    cudaMalloc((void**)&d_flags, numCandidates * sizeof(int));
    cudaMemset(d_flags, 0, numCandidates * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numCandidates + threadsPerBlock - 1) / threadsPerBlock;
    detectKernel << <blocksPerGrid, threadsPerBlock >> > (d_candidates, d_flags, numCandidates);
    cudaDeviceSynchronize();

    std::vector<int> flags(numCandidates, 0);
    cudaMemcpy(flags.data(), d_flags, numCandidates * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numCandidates; i++) {
        if (flags[i] == 1)
            detections.push_back(candidateWindows[i]);
    }

    cudaFree(d_candidates);
    cudaFree(d_flags);
    return detections;
}
