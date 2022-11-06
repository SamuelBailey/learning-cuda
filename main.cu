#include <iostream>
#include <math.h>
#include <memory>
#include "print-hello.h"

__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i+= stride) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char** argv) {
    printHello();
    int deviceCount;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != 0) {
        std::cout << "Exiting with error: " << cudaResultCode << std::endl;
        exit(cudaResultCode);
    }

    // Ensure there is a cuda device on the system.
    if (deviceCount < 1) {
        std::cout << "No cuda devices found on the system. Exiting..." << std::endl;
        exit(1);
    }


    std::cout << "hello world" << std::endl;
    std::cout << "test" << std::endl;

    std::cout << "Number of cuda devices = " << deviceCount << std::endl;

    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    std::cout << "Multiprocessor count: " << properties.multiProcessorCount << std::endl;
    std::cout << "Threads per block: " << properties.maxThreadsPerBlock << std::endl;
    std::cout << "Threads per multiprocessor: " << properties.maxThreadsPerMultiProcessor << std::endl;

    const int N = 1<<20; // 1 million elements

    // auto x = std::make_unique<float[]>(N);
    // auto y = std::make_unique<float[]>(N);

    float *h_x, *h_y;
    cudaMallocHost(&h_x, N*sizeof(float));
    cudaMallocHost(&h_y, N*sizeof(float));
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(h_y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}

