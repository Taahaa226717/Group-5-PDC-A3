#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

// return GB/sec
float GBPerSec(int bytes, float sec)
{
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void saxpy_kernel(int N, float alpha, float *x, float *y,
                             float *result)
{

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if (index < N)
        result[index] = alpha * x[index] + y[index];
}

// saxpyCuda --
//
// This function is regular C code running on the CPU.  It allocates
// memory on the GPU using CUDA API functions, uses CUDA API functions
// to transfer data from the CPU's memory address space to GPU memory
// address space, and launches the CUDA kernel function on the GPU.
void saxpyCuda(int N, float alpha, float *xarray, float *yarray,
               float *resultarray)
{

    // must read both input arrays (xarray and yarray) and write to
    // output array (resultarray)
    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block.  In this
    // application we've hardcoded thread blocks to contain 512 CUDA
    // threads.
    const int threadsPerBlock = 512;

    // Notice the round up here.  The code needs to compute the number
    // of threads blocks needed such that there is one thread per
    // element of the arrays.  This code is written to work for values
    // of N that are not multiples of threadPerBlock.
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // These are pointers that will be pointers to memory allocated
    // *one the GPU*.  You should allocate these pointers via
    // cudaMalloc.  You can access the resulting buffers from CUDA
    // device kernel code (see the kernel function saxpy_kernel()
    // above) but you cannot access the contents these buffers from
    // this thread. CPU threads cannot issue loads and stores from GPU
    // memory!
    float *device_x = nullptr;
    float *device_y = nullptr;
    float *device_result = nullptr;

    //
    // CS149 TODO: allocate device memory buffers on the GPU using cudaMalloc.
    //
    size_t inputSize = N * sizeof(float);
    size_t outputSize = N * sizeof(float);

    cudaError_t status;

    status = cudaMalloc(&device_x, inputSize);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error allocating device_x: %s\n", cudaGetErrorString(status));
        return;
    }

    status = cudaMalloc(&device_y, inputSize);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error allocating device_y: %s\n", cudaGetErrorString(status));
        cudaFree(device_x);
        return;
    }

    status = cudaMalloc(&device_result, outputSize);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error allocating device_result: %s\n", cudaGetErrorString(status));
        cudaFree(device_x);
        cudaFree(device_y);
        return;
    }

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    //
    // CS149 TODO: copy input arrays to the GPU using cudaMemcpy
    //

    status = cudaMemcpy(device_x, xarray, inputSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error copying to device_x: %s\n", cudaGetErrorString(status));
        goto cleanup;
    }

    status = cudaMemcpy(device_y, yarray, inputSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error copying to device_y: %s\n", cudaGetErrorString(status));
        goto cleanup;
    }

    cudaEvent_t kernelStart, kernelEnd;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);

    cudaEventRecord(kernelStart);

    // run CUDA kernel. (notice the <<< >>> brackets indicating a CUDA
    // kernel launch) Execution on the GPU occurs here.
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y,
                                              device_result);

    cudaEventRecord(kernelEnd);
    cudaEventSynchronize(kernelEnd);

    float kernelMs = 0.0f;
    cudaEventElapsedTime(&kernelMs, kernelStart, kernelEnd);

    //
    // CS149 TODO: copy result from GPU back to CPU using cudaMemcpy
    //

    status = cudaMemcpy(resultarray, device_result, outputSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error copying from device_result: %s\n", cudaGetErrorString(status));
        goto cleanup;
    }

    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess)
    {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode,
                cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Effective BW by CUDA saxpy: %.3f ms\t\t[%.3f GB/s]\n",
           1000.f * overallDuration, GBPerSec(totalBytes, overallDuration));
    printf("Kernel-only execution time (using events): %.3f ms\n", kernelMs);

    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelEnd);

cleanup:
    //
    // CS149 TODO: free memory buffers on the GPU using cudaFree
    //

    // Clean up GPU resources
    if (device_x)
        cudaFree(device_x);
    if (device_y)
        cudaFree(device_y);
    if (device_result)
        cudaFree(device_result);
}

void printCudaInfo()
{

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}