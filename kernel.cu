
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

#define cuda_error_check(code) {error_check(code, __FILE__, __LINE__); }
inline void error_check(cudaError_t err, const char* file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s at %d\n", cudaGetErrorName(err));
        if (abort) exit(EXIT_FAILURE);
    }
}

#define cuda_kernel_check() {kernel_check(__FILE__, __LINE__); }
inline void kernel_check(const char* file, int line, bool abort = true) {
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s at %d\n", cudaGetErrorName(err));
        if (abort) exit(EXIT_FAILURE);
    }
}

__global__ void matrix_multiply_tiled(int *d_c, const int * d_a, const int * d_b, int n)
{
    __shared__ int ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int cVal = 0;
    for (int m = 0; m < n / TILE_WIDTH; m++) {
        ds_a[ty][tx] = d_a[row * n + m * TILE_WIDTH + tx];
        ds_b[ty][tx] = d_b[n*(m * TILE_WIDTH + tx) + col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            cVal += ds_a[ty][k] * ds_b[k][tx];
        }

        __syncthreads();
    }

    d_c[row * n + col] = cVal;
}

void print_matrix(const char* name, const int* matrix, int n) {
    printf("--- Matrix %s ---\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}
int main()
{
    int n = 2048;
    size_t bytes = n * n * sizeof(int);

    int* h_a = (int*)malloc(bytes);
    int* h_b = (int*)malloc(bytes);
    int* h_c = (int*)malloc(bytes);

    float gpu_time;
    cudaEvent_t start, stop;
    cuda_error_check(cudaEventCreate(&start));
    cuda_error_check(cudaEventCreate(&stop));

    for (int i = 0; i < n * n; i++) {
        h_a[i] = i + 1;
        h_b[i] = 2;
    }

    int* d_a, * d_b, * d_c;
    cuda_error_check(cudaMalloc(&d_a, bytes));
    cuda_error_check(cudaMalloc(&d_b, bytes));
    cuda_error_check(cudaMalloc(&d_c, bytes));

    cuda_error_check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cuda_error_check(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(n / TILE_WIDTH, n / TILE_WIDTH);

    cudaEventRecord(start);
    matrix_multiply_tiled << <blocksPerGrid, threadsPerBlock >> > (d_c, d_a, d_b, n);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&gpu_time, start, stop);
    cuda_error_check(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    /*
    print_matrix("A", h_a, n);
    print_matrix("B", h_b, n);
    print_matrix("C (Result A x B)", h_c, n);*/

    cuda_error_check(cudaFree(d_a));
    cuda_error_check(cudaFree(d_b));
    cuda_error_check(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    printf("GPU Runtime: %f ms\n", gpu_time);

    return 0;
}

