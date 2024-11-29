#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

__global__ void findMaxKernel(int* input, int* maxVal, int n) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sharedData[tid] = (idx < n) ? input[idx] : INT_MIN;
    __syncthreads();

    // Perform reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && (idx + stride) < n) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }

    // Write block maximum to global memory
    if (tid == 0) {
        maxVal[blockIdx.x] = sharedData[0];
    }
}

int getMaxHostParallel(int* arr, int n) {
    int *d_input, *d_max, *h_max;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    size_t bytes = n * sizeof(int);
    size_t maxBytes = blocks * sizeof(int);

    // Allocate memory
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_max, maxBytes);
    h_max = new int[blocks];

    cudaMemcpy(d_input, arr, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    findMaxKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_max, n);
    cudaMemcpy(h_max, d_max, maxBytes, cudaMemcpyDeviceToHost);

    // Find max of block-wise results
    int maxElement = h_max[0];
    for (int i = 1; i < blocks; i++) {
        maxElement = std::max(maxElement, h_max[i]);
    }

    // Free memory
    cudaFree(d_input);
    cudaFree(d_max);
    delete[] h_max;

    return maxElement;
}

__global__ void countSortKernel(int* input, int* output, int n, int exp) {
    __shared__ int count[10];

    // Initialize count array
    if (threadIdx.x < 10) {
        count[threadIdx.x] = 0;
    }
    __syncthreads();

    // Count occurrences of digits
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int digit = (input[i] / exp) % 10;
        atomicAdd(&count[digit], 1);
    }
    __syncthreads();

    // Compute prefix sum
    if (threadIdx.x < 10) {
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
    }
    __syncthreads();

    // Reorder elements
    for (int i = n - 1 - threadIdx.x; i >= 0; i -= blockDim.x) {
        int digit = (input[i] / exp) % 10;
        int idx = atomicSub(&count[digit], 1) - 1;
        output[idx] = input[i];
    }
    __syncthreads();

    // Copy back to input
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        input[i] = output[i];
    }
}

void radixSort(int* arr, int n) {
    int *d_input, *d_output;
    size_t bytes = n * sizeof(int);

    // Allocate unified memory
    cudaMallocManaged(&d_input, bytes);
    cudaMallocManaged(&d_output, bytes);

    // Copy input data to unified memory
    for (int i = 0; i < n; i++) {
        d_input[i] = arr[i];
    }

    // Find max element in parallel
    int max_element = getMaxHostParallel(arr, n);

    int threadsPerBlock = 256;

    // Radix sort for each decimal place
    for (int exp = 1; max_element / exp > 0; exp *= 10) {
        countSortKernel<<<1, threadsPerBlock>>>(d_input, d_output, n, exp);
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    for (int i = 0; i < n; i++) {
        arr[i] = d_input[i];
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int arr[] = {170, 45, 75, 90, 802, 24, 2, 66};
    int n = sizeof(arr) / sizeof(arr[0]);

    std::cout << "Original array: ";
    for (int i = 0; i < n; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Start timing

    radixSort(arr, n);

    cudaEventRecord(stop); // Stop timing
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Sorted array: ";
    for (int i = 0; i < n; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;

    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
