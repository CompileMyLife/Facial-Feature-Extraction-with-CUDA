#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel() {
    // do nothing
}

int main(){
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("Hello from CUDA!\n");
    return 0;
}
