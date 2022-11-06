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

    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    (void)cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}

