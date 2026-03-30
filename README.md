# CUDA Tiled Matrix Multiplication

A high-performance square matrix multiplication implementation in CUDA using **shared memory tiling** to minimize global memory accesses and maximize throughput on NVIDIA GPUs.

---

## How It Works

Naive matrix multiplication reads from global memory repeatedly for each dot product computation, which is slow due to high memory latency. This implementation uses the **tiled algorithm**:

1. The output matrix `C = A × B` is divided into square tiles of size `TILE_WIDTH × TILE_WIDTH`.
2. Each thread block is responsible for computing one output tile.
3. Within each block, threads **collaboratively load** sub-tiles of `A` and `B` into fast shared memory (`__shared__`).
4. Threads then compute partial dot products from shared memory, repeating across all tiles along the shared dimension.
5. `__syncthreads()` barriers ensure all threads finish loading before computation begins, and finish computing before the next tile is loaded.

This reduces global memory bandwidth usage from `O(n³)` reads down to `O(n³ / TILE_WIDTH)`.

---

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability 3.0+)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with CUDA 11+)
- C++ compiler compatible with `nvcc` (e.g., MSVC on Windows, GCC on Linux)

---

## Building

### With `nvcc` directly

```bash
nvcc -O2 -DTILE_WIDTH=32 -o matmul matmul.cu
```

### Override tile width at compile time

```bash
nvcc -O2 -DTILE_WIDTH=16 -o matmul matmul.cu
```

> **Note:** `n` must be divisible by `TILE_WIDTH`. The default matrix size is `2048 × 2048` and the default tile width is `32`.

---

## Running

```bash
./matmul
```

**Sample output:**
```
GPU Runtime: 12.345678 ms
```

---

## Configuration

| Parameter     | Default | Description                                      |
|---------------|---------|--------------------------------------------------|
| `TILE_WIDTH`  | `32`    | Tile/block dimension (must divide `n` evenly)    |
| `n`           | `2048`  | Matrix dimension (`n × n` square matrices)       |

To change the matrix size, modify the `n` variable in `main()`.

---

## Code Structure

| Component                  | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| `matrix_multiply_tiled`   | CUDA kernel — tiled matrix multiplication using shared memory |
| `cuda_error_check()`      | Macro + inline function for CUDA API error checking          |
| `cuda_kernel_check()`     | Macro + inline function for post-kernel launch error checking |
| `print_matrix()`          | Debug utility to print a matrix to stdout                    |
| `main()`                  | Allocates memory, launches kernel, measures GPU time         |

---

## Performance Measurement

GPU execution time is measured using **CUDA Events** (`cudaEventRecord` / `cudaEventElapsedTime`), which provide accurate on-device timing independent of CPU overhead.

---

## Debug / Printing

Matrix printing is available but commented out in `main()` by default (printing a 2048×2048 matrix to stdout is not practical):

```cpp
// print_matrix("A", h_a, n);
// print_matrix("B", h_b, n);
// print_matrix("C (Result A x B)", h_c, n);
```

Uncomment these lines and reduce `n` to a small value (e.g., `4` or `8`) for functional verification.

---

## Limitations

- Only supports **square matrices** where `n` is a multiple of `TILE_WIDTH`.
- Uses `int` arithmetic — no floating-point support in this version.
- No CPU reference implementation is included for result validation.
