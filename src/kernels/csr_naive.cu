#include "kernel.hpp"
#include "matrix_csr.hpp"


// using CSR format, naive
__global__ void csr_naive_kernel(
    int M, int N, 
    float *A_vals, int A_vals_size, 
    int *A_col_idxs, int A_col_idxs_size,
    int *A_row_ptrs, int A_row_ptrs_size,
    float *X, float *Y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = A_row_ptrs[idx];
    int end = (idx == N - 1) ? A_col_idxs_size : A_row_ptrs[idx + 1];
    
    float acc = 0.0f;
    for (int i = begin; i < end; i++) {
        acc += X[A_col_idxs[i]] * A_vals[i];
    }

    Y[idx] = acc;
}


void csr_naive_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
    dim3 block(32, 1, 1);
    dim3 grid(N / 32);

    CSRMatrix csr(M, N, A_host);

    const size_t size_A_row_ptrs = csr.RowPtrsSize() * sizeof(int);
    const size_t size_A_col_idxs = csr.ColIdxsSize() * sizeof(int);
    const size_t size_A_values = csr.ValuesSize() * sizeof(float);

    float *A_values_device; 
    int *A_col_idxs_device, *A_row_ptrs_device;
    CUDA_CHECK(cudaMalloc((void **)&A_values_device, size_A_values));
    CUDA_CHECK(cudaMalloc((void **)&A_col_idxs_device, size_A_col_idxs));
    CUDA_CHECK(cudaMalloc((void **)&A_row_ptrs_device, size_A_row_ptrs));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_values_device, csr.GetValues(), size_A_values, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_col_idxs_device, csr.GetColIdxs(), size_A_col_idxs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_row_ptrs_device, csr.GetRowPtrs(), size_A_row_ptrs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    // call the kernel
    TIME_KERNEL((csr_naive_kernel<<<grid, block>>>(
        M, N, 
        A_values_device, csr.ValuesSize(),
        A_col_idxs_device, csr.ColIdxsSize(),
        A_row_ptrs_device, csr.RowPtrsSize(),
        X_device, Y_device
    )));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_values_device));
    CUDA_CHECK(cudaFree(A_col_idxs_device));
    CUDA_CHECK(cudaFree(A_row_ptrs_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));

    return;
}