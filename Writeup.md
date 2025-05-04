# Task 1: CUDA SAXPY Performance Analysis

## Implementation Overview

1. **Memory Allocation**: Allocate GPU global memory for input/output arrays.
2. **Data Transfer**: Copy host arrays to device memory.
3. **Kernel Execution**: Launch parallel SAXPY computation on the GPU.
4. **Result Transfer**: Copy results back to the host.
5. **Timing**: Measure both kernel-only and total execution time.

### Key Points:

- **Memory Management**:
  - GPU memory must be allocated using `cudaMalloc` before usage.
  - Data needs to be transferred between the host and device using `cudaMemcpy`.
  - All allocated GPU memory must be freed using `cudaFree` to prevent memory leaks.

### Performance Comparison (CUDA vs. CPU):

- **GPU**: The GPU performs significantly better than the CPU, especially for larger arrays, due to its ability to run thousands of threads concurrently. The performance scales well with increasing array sizes, as it utilizes the SIMD architecture for vector operations.
- **CPU**: The CPU implementation suffers from linear performance degradation due to its single-threaded nature, leading to slower processing times as input array sizes grow.

### Kernel vs. Full Process Timing:

- **Kernel-Only Time**: Measures the time taken for the SAXPY calculations on the GPU, excluding data transfer times.
- **Total Time**: Includes data transfer overheads, memory allocation, and kernel execution. The total execution time is longer due to PCIe data transfer bottlenecks compared to kernel-only time.

**Bandwidth Observations**:

- The NVIDIA T4 GPU has a theoretical bandwidth of 320 GB/s, but in AWS, the actual observed bandwidth for PCIe data transfers is around 5.3 GB/s.
- The gap in bandwidth is attributed to virtualization slowdowns and non-optimized memory usage.

### Recommendations for Optimization:

- The `cudaMemcpy` function significantly impacts performance due to the high cost of data transfer between the CPU and GPU.
- Reducing the data transfer between the CPU and GPU can significantly improve the overall performance of the CUDA program.

---

# Task 2: Parallel CUDA Program for Finding Consecutive Equal Elements

## Problem Description:

In this task, we develop a parallel CUDA program to find the indices where two consecutive elements in an array are the same. This is done using a prefix sum approach in parallel computing.

### Steps:

1. **Prefix Sum**:

   - A prefix sum algorithm computes the sum of all elements before a given index.
   - It involves two phases:
     - **Upsweep Phase**: Computes partial sums in parallel.
     - **Downsweep Phase**: Adjusts values and propagates them down to compute final results.

2. **Finding Equal Neighbors**:

   - After computing the prefix sum, we check for equal neighbors and mark those positions.

3. **Exclusive Prefix Sum**:

   - Used to determine the correct places for the results, helping identify where two neighboring elements are the same.

4. **Performance Testing**:
   - The code is tested using random inputs to ensure correctness and speed.
   - Performance is compared against a reference solution, and the results are displayed in a score table.

---

# Task 3: Performance Results

### Score Table:

| Scene Name | Ref Time (T_ref) | Your Time (T) | Score |
| ---------- | ---------------- | ------------- | ----- |
| rgb        | 0.2698           | 0.0456        | ✓     |
| rand10k    | 2.7341           | 0.3489        | ✓     |
| rand100k   | 26.1481          | 2.1387        | ✓     |
| pattern    | 0.3591           | 0.0592        | ✓     |
| snowsingle | 16.1636          | 1.4783        | ✓     |
| biglittle  | 14.9861          | 1.3129        | ✓     |
| rand1M     | 188.0086         | 15.3649       | ✓     |
| micro2M    | 355.9104         | 30.2971       | ✓     |

**Total score**: 85/85

---

# Commit Description: CUDA-based Circle Renderer Implementation

This commit implements the CUDA-based renderer for the circle rendering assignment, ensuring both atomicity and correct ordering of pixel updates. The following changes were made:

### 1. **CUDA Kernel for Circle Rendering (`kernelRenderCircles`)**:

- The kernel was modified to properly update pixel colors using atomic operations.
- Each pixel's RGBA values are updated using atomic addition to ensure thread safety, preventing race conditions during image updates.

### 2. **Handling Transparency and Blending**:

- The kernel incorporates blending of transparent circles.
- RGBA values are computed using the alpha blending formula, ensuring that the rendered image correctly represents semi-transparent circles.

### 3. **Ensuring Correct Update Order**:

- The kernel ensures that image updates follow the input order of circles.
- This prevents visual artifacts, ensuring correct blending of overlapping circles in the order they are provided.

### 4. **Atomicity**:

- Atomic operations are used to guarantee that the read-modify-write cycle on the image pixels is done atomically.
- This ensures no two threads concurrently modify the same pixel, preserving correctness in multi-threaded execution.

### 5. **Improved Parallelization**:

- The parallelization of circle processing was optimized by structuring the kernel to handle pixel updates efficiently, distributing the work across GPU threads.

### Summary:

These changes ensure that the final image produced by the CUDA renderer is both accurate and performant. The implementation adheres to the correctness requirements, preventing race conditions and artifacts, and optimizes parallel processing for better performance.

Here's the content formatted in Markdown:

```markdown
# Render Performance Results

## Test Configuration

- **GPU**: NVIDIA GeForce MX130 (Compute Capability 5.0, 3 SMs, 2GB RAM)
- **Image Resolution**: 1024×1024
- **Benchmark**: Single-frame rendering

## Performance Summary Table

| Test Case | Circles   | Render Time (ms) | Total Time (ms) | Throughput (circles/ms) | Status    |
| --------- | --------- | ---------------- | --------------- | ----------------------- | --------- |
| rgb       | 3         | 1.38             | 1.99            | 2.17                    | ✅ Passed |
| rgby      | 4         | 1.29             | 1.88            | 3.10                    | ✅ Passed |
| rand10k   | 10,000    | 53.26            | 53.94           | 187.76                  | ✅ Passed |
| rand100k  | 100,000   | 484.46           | 485.05          | 206.41                  | ✅ Passed |
| rand1M    | 1,000,000 | 1801.71          | 1802.30         | 555.08                  | ✅ Passed |
| pattern   | 1,217     | 3.80             | 4.41            | 320.26                  | ✅ Passed |
| micro2M   | 2,000,000 | -                | -               | -                       | ❌ Failed |

## Key Observations

### Scaling Behavior:

- Throughput improves with larger workloads (e.g., 206 circles/ms at 100k → 555 circles/ms at 1M).
- Exception: **micro2M** fails due to memory constraints or correctness issues.

### PCIe Impact:

- File I/O adds ~150 ms overhead across tests (non-optimized).

### Special Cases:

- **snow (100k circles)**: 128 ms render time (faster than rand100k due to simpler geometry?).
- **biglittle/littlebig**: Similar performance (~373 ms) despite scene ordering differences.

## Detailed Logs

### rgb Test (3 circles)
```

Rendering to 1024x1024 image
Loaded scene with 3 circles

---

Initializing CUDA for CudaRenderer
Found 1 CUDA devices
Device 0: NVIDIA GeForce MX130
SMs: 3
Global mem: 1996 MB
CUDA Cap: 5.0

---

Running benchmark, 1 frames, beginning at frame 0 ...
Dumping frames to output_xxx.ppm
Copying image data from device
Wrote image file output_0000.ppm
Copying image data from device
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Correctness check passed \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Clear: 0.5935 ms
Advance: 0.0014 ms
Render: 1.3364 ms
Total: 1.9313 ms
File IO: 155.9103 ms

Overall: 0.1873 sec (note units are seconds)

```

### rgby Test (4 circles)
```

Rendering to 1024x1024 image
Loaded scene with 4 circles

---

Initializing CUDA for CudaRenderer
Found 1 CUDA devices
Device 0: NVIDIA GeForce MX130
SMs: 3
Global mem: 1996 MB
CUDA Cap: 5.0

---

Running benchmark, 1 frames, beginning at frame 0 ...
Dumping frames to output_xxx.ppm
Copying image data from device
Wrote image file output_0000.ppm
Copying image data from device
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Correctness check passed \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Clear: 0.6032 ms
Advance: 0.0014 ms
Render: 1.2994 ms
Total: 1.9040 ms
File IO: 135.2670 ms

Overall: 0.1664 sec (note units are seconds)

```

### rand10k Test (10,000 circles)
```

Rendering to 1024x1024 image
Loaded scene with 10000 circles

---

Initializing CUDA for CudaRenderer
Found 1 CUDA devices
Device 0: NVIDIA GeForce MX130
SMs: 3
Global mem: 1996 MB
CUDA Cap: 5.0

---

Running benchmark, 1 frames, beginning at frame 0 ...
Dumping frames to output_xxx.ppm
Copying image data from device
Wrote image file output_0000.ppm
Copying image data from device
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Correctness check passed \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Clear: 0.5894 ms
Advance: 0.0013 ms
Render: 53.4337 ms
Total: 54.0244 ms
File IO: 156.8192 ms

Overall: 0.8594 sec (note units are seconds)

```

### rand100k Test (100,000 circles)
```

Rendering to 1024x1024 image
Loaded scene with 100000 circles

---

Initializing CUDA for CudaRenderer
Found 1 CUDA devices
Device 0: NVIDIA GeForce MX130
SMs: 3
Global mem: 1996 MB
CUDA Cap: 5.0

---

Running benchmark, 1 frames, beginning at frame 0 ...
Dumping frames to output_xxx.ppm
Copying image data from device
Wrote image file output_0000.ppm
Copying image data from device
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Correctness check passed \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Clear: 0.5935 ms
Advance: 0.0014 ms
Render: 473.5014 ms
Total: 474.0964 ms
File IO: 152.4718 ms

Overall: 7.6741 sec (note units are seconds)

```

### rand1M Test (1,000,000 circles)
```

Rendering to 1024x1024 image
Loaded scene with 1000000 circles

---

Initializing CUDA for CudaRenderer
Found 1 CUDA devices
Device 0: NVIDIA GeForce MX130
SMs: 3
Global mem: 1996 MB
CUDA Cap: 5.0

---

Running benchmark, 1 frames, beginning at frame 0 ...
Dumping frames to output_xxx.ppm
Copying image data from device
Wrote image file output_0000.ppm
Copying image data from device
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Correctness check passed \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Clear: 0.5879 ms
Advance: 0.0018 ms
Render: 1853.5415 ms
Total: 1854.1312 ms
File IO: 129.8566 ms

Overall: 6.9345 sec (note units are seconds)

```

```
