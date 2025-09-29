#include "awsp.hpp"
#include "kernel.hpp"
#include "ptx.hpp"

__global__ void awsp_kernel_v0(
    int M, int N,
    int nz_bk_max,
    uint32_t *bitmaps,
    float* A_vals,
    float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    uint32_t curr_mask = (1u << lane_id);
    uint32_t prev_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);

    float sum = 0;
    // X pointer: warp offset
    float *X_ptr = X + M/4*warp_id + lane_id;
    // A pointer: block offset
    float *A_ptr = A_vals + (blockIdx.x*M/32 + M/32*warp_id/4) * nz_bk_max;
    // bitmap pointer: same
    uint32_t *B_ptr = bitmaps + (blockIdx.x*M/32 + M/32*warp_id/4) * 32 + lane_id;

    for (int out_bk = 0; out_bk < M / 4; out_bk += 32) {
        uint32_t bitmap = *B_ptr;
        float x = *X_ptr;
        int A_val_bk_offset = 0;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float cur_x = __shfl_sync(0xffffffff, x, i);
            uint32_t cur_bitmap = __shfl_sync(0xffffffff, bitmap, i);
            // if (cur_x != 0.0f) {
                if (cur_bitmap & curr_mask) {
                    int A_val_row_offset = __popc(cur_bitmap & prev_mask);
                    float a = A_ptr[A_val_bk_offset + A_val_row_offset];
                    sum += cur_x * a;
                }
            // }
            A_val_bk_offset += __popc(cur_bitmap);
        }
        A_ptr += nz_bk_max;
        B_ptr += 32;
        X_ptr += 32;
    }

    // reduction
    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0; i < 3; i++) {
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

// version1: add one more pipeline stage
__global__ void awsp_kernel_v1(
    int M, int N,
    int nz_bk_max,
    uint32_t *bitmaps,
    float* A_vals,
    float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    uint32_t curr_mask = (1u << lane_id);
    uint32_t prev_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);

    float sum = 0;
    float *X_ptr = X + M/4*warp_id + lane_id;
    float *A_ptr = A_vals + (blockIdx.x*M/32 + M/32*warp_id/4) * nz_bk_max;
    uint32_t *B_ptr = bitmaps + (blockIdx.x*M/32 + M/32*warp_id/4) * 32 + lane_id;

    // pipeline buffers
    float X_buf[3]; // compute stage and load stage
    uint32_t B_buf[3]; // compute stage and load stage
    float A_buf[32]; // only use to compute

    // head stage 0: load the x and bitmap for compute
    LDG_F_CA_32(X_ptr, X_buf[0]); X_ptr += 32;
    LDG_U_CA_32(B_ptr, B_buf[0]); B_ptr += 32;
    // head stage 1: load x and bitmap for next load
    X_buf[1] = X_buf[0]; LDG_F_CA_32(X_ptr, X_buf[0]); X_ptr += 32;
    B_buf[1] = B_buf[0]; LDG_U_CA_32(B_ptr, B_buf[0]); B_ptr += 32;
    // head stage 2: load A for compute
    int first_bk_offset = 0;
    for (int i = 0; i < 32; i++) {
        // float cur_x = __shfl_sync(0xffffffff, X_buf[1], i);
        uint32_t cur_bitmap = __shfl_sync(0xffffffff, B_buf[1], i);
        // if (cur_x != 0.0f) {
            if (cur_bitmap & curr_mask) {
                int A_val_row_offset = __popc(cur_bitmap & prev_mask);
                A_buf[i] = A_ptr[first_bk_offset + A_val_row_offset];
            } else {
                A_buf[i] = 0.0f;
            }
        // }
        first_bk_offset += __popc(cur_bitmap);
    }
    A_ptr += nz_bk_max;
    
    // main pipeline
    for (int out_bk = 0; out_bk < (M/4 - 32*2); out_bk += 32) {
        // load the x and bitmap for next load
        X_buf[2] = X_buf[1]; X_buf[1] = X_buf[0]; LDG_F_CA_32(X_ptr, X_buf[0]); X_ptr += 32;
        B_buf[2] = B_buf[1]; B_buf[1] = B_buf[0]; LDG_U_CA_32(B_ptr, B_buf[0]); B_ptr += 32;

        int next_bk_offset = 0;
        for (int i = 0; i < 32; i++) {
            // consume bufferd A
            float x_calc = __shfl_sync(0xffffffff, X_buf[2], i);
            float a = A_buf[i];
            // todo: branch or not ?
            sum += a * x_calc;

            // load new A
            // float x_load = __shfl_sync(0xffffffff, X_buf[1], i);
            uint32_t b_load = __shfl_sync(0xffffffff, B_buf[1], i);
            // if (x_load != 0.0f) {
                if (b_load & curr_mask) {
                    int A_val_row_offset = __popc(b_load & prev_mask);
                    A_buf[i] = A_ptr[next_bk_offset + A_val_row_offset];
                } else {
                    A_buf[i] = 0.0f;
                }
            // }
            next_bk_offset += __popc(b_load);
        }
        A_ptr += nz_bk_max;
    }

    // tail stage
    // tail stage 0 (no need to load new x and bitmap)
    X_buf[2] = X_buf[1]; X_buf[1] = X_buf[0];
    B_buf[2] = B_buf[1]; B_buf[1] = B_buf[0];
    int last2_bk_offset = 0;
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_buf[2], i);
        float a = A_buf[i];
        sum += a * x_calc;

        // float x_load = __shfl_sync(0xffffffff, X_buf[1], i);
        uint32_t b_load = __shfl_sync(0xffffffff, B_buf[1], i);
        // if (x_load != 0.0f) {
            if (b_load & curr_mask) {
                int A_val_row_offset = __popc(b_load & prev_mask);
                A_buf[i] = A_ptr[last2_bk_offset + A_val_row_offset];
            } else {
                A_buf[i] = 0.0f;
            }
        // }
        last2_bk_offset += __popc(b_load);
    }
    // tail stage 1 (no need to load new x)
    X_buf[2] = X_buf[1];
    B_buf[2] = B_buf[1];
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_buf[2], i);
        float a = A_buf[i];
        sum += a * x_calc;
    }

    // reduction
    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0; i < 3; i++) {
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

// version2: pipeline deeper
__global__ void awsp_kernel_v2(
    int M, int N,
    int nz_bk_max,
    uint32_t *bitmaps,
    float* A_vals,
    float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    uint32_t curr_mask = (1u << lane_id);
    uint32_t prev_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);

    float sum = 0;
    float *X_ptr = X + M/4*warp_id + lane_id;
    float *A_ptr = A_vals + (blockIdx.x*M/32 + M/32*warp_id/4) * nz_bk_max;
    uint32_t *B_ptr = bitmaps + (blockIdx.x*M/32 + M/32*warp_id/4) * 32 + lane_id;

    // pipeline buffers
    float X_buf[2][3]; // compute stage and load stage
    uint32_t B_buf[2][3]; // compute stage and load stage
    float A_buf[2][32]; // only use to compute

    // head stage 0: load the x and bitmap for compute
    LDG_F_CA_32(X_ptr, X_buf[0][0]);
    LDG_F_CA_32(X_ptr + 32, X_buf[1][0]); X_ptr += 64;
    LDG_U_CA_32(B_ptr, B_buf[0][0]);
    LDG_U_CA_32(B_ptr + 32, B_buf[1][0]); B_ptr += 64;
    // head stage 1: load x and bitmap for next load
    X_buf[0][1] = X_buf[0][0]; LDG_F_CA_32(X_ptr, X_buf[0][0]);
    X_buf[1][1] = X_buf[1][0]; LDG_F_CA_32(X_ptr + 32, X_buf[1][0]); X_ptr += 64;
    B_buf[0][1] = B_buf[0][0]; LDG_U_CA_32(B_ptr, B_buf[0][0]);
    B_buf[1][1] = B_buf[1][0]; LDG_U_CA_32(B_ptr + 32, B_buf[1][0]); B_ptr += 64;
    // head stage 2: load A for compute
    #pragma unroll
    for (int idx = 0; idx < 2; idx++) {
        int first_bk_offset = 0;
        for (int i = 0; i < 32; i++) {
            // float cur_x = __shfl_sync(0xffffffff, X_buf[idx][1], i);
            uint32_t cur_bitmap = __shfl_sync(0xffffffff, B_buf[idx][1], i);
            // if (cur_x != 0.0f) {
                if (cur_bitmap & curr_mask) {
                    int A_val_row_offset = __popc(cur_bitmap & prev_mask);
                    A_buf[idx][i] = A_ptr[nz_bk_max*idx + first_bk_offset + A_val_row_offset];
                } else {
                    A_buf[idx][i] = 0.0f;
                }
            // }
            first_bk_offset += __popc(cur_bitmap);
        }
    }
    A_ptr += nz_bk_max * 2;
    
    // main pipeline
    for (int out_bk = 0; out_bk < (M/4 - 64*2); out_bk += 64) {
        // load the x and bitmap for next load
        X_buf[0][2] = X_buf[0][1]; X_buf[0][1] = X_buf[0][0]; LDG_F_CA_32(X_ptr, X_buf[0][0]);
        X_buf[1][2] = X_buf[1][1]; X_buf[1][1] = X_buf[1][0]; LDG_F_CA_32(X_ptr + 32, X_buf[1][0]); X_ptr += 64;
        B_buf[0][2] = B_buf[0][1]; B_buf[0][1] = B_buf[0][0]; LDG_U_CA_32(B_ptr, B_buf[0][0]);
        B_buf[1][2] = B_buf[1][1]; B_buf[1][1] = B_buf[1][0]; LDG_U_CA_32(B_ptr + 32, B_buf[1][0]); B_ptr += 64;
        #pragma unroll
        for (int idx = 0; idx < 2; idx++) {
            int next_bk_offset = 0;
            for (int i = 0; i < 32; i++) {
                // consume bufferd A
                float x_calc = __shfl_sync(0xffffffff, X_buf[idx][2], i);
                float a = A_buf[idx][i];
                // todo: branch or not ?
                sum += a * x_calc;

                // load new A
                // float x_load = __shfl_sync(0xffffffff, X_buf[idx][1], i);
                uint32_t b_load = __shfl_sync(0xffffffff, B_buf[idx][1], i);
                // if (x_load != 0.0f) {
                    if (b_load & curr_mask) {
                        int A_val_row_offset = __popc(b_load & prev_mask);
                        A_buf[idx][i] = A_ptr[nz_bk_max*idx + next_bk_offset + A_val_row_offset];
                    } else {
                        A_buf[idx][i] = 0.0f;
                    }
                // }
                next_bk_offset += __popc(b_load);
            }
        }  
        A_ptr += nz_bk_max * 2;
    }

    // tail stage
    // tail stage 0 (no need to load new x and bitmap)
    X_buf[0][2] = X_buf[0][1]; X_buf[0][1] = X_buf[0][0];
    X_buf[1][2] = X_buf[1][1]; X_buf[1][1] = X_buf[1][0];
    B_buf[0][2] = B_buf[0][1]; B_buf[0][1] = B_buf[0][0];
    B_buf[1][2] = B_buf[1][1]; B_buf[1][1] = B_buf[1][0];
    #pragma unroll
    for (int idx = 0; idx < 2; idx++) {
        int last_bk_offset = 0;
        for (int i = 0; i < 32; i++) {
            float x_calc = __shfl_sync(0xffffffff, X_buf[idx][2], i);
            float a = A_buf[idx][i];
            sum += a * x_calc;

            // float x_load = __shfl_sync(0xffffffff, X_buf[idx][1], i);
            uint32_t b_load = __shfl_sync(0xffffffff, B_buf[idx][1], i);
            // if (x_load != 0.0f) {
                if (b_load & curr_mask) {
                    int A_val_row_offset = __popc(b_load & prev_mask);
                    A_buf[idx][i] = A_ptr[nz_bk_max*idx + last_bk_offset + A_val_row_offset];
                } else {
                    A_buf[idx][i] = 0.0f;
                }
            // }
            last_bk_offset += __popc(b_load);
        }
    }
    
    // tail stage 1 (no need to load new x)
    X_buf[0][2] = X_buf[0][1];
    X_buf[1][2] = X_buf[1][1];
    B_buf[0][2] = B_buf[0][1];
    B_buf[1][2] = B_buf[1][1];
    for (int idx = 0; idx < 2; idx++) {
        for (int i = 0; i < 32; i++) {
            float x_calc = __shfl_sync(0xffffffff, X_buf[idx][2], i);
            float a = A_buf[idx][i];
            sum += a * x_calc;
        }
    }
    
    // reduction
    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0; i < 3; i++) {
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

void awsp_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host, int version) {
    dim3 block(128, 1, 1);
    dim3 grid(N / 32, 1, 1);

    AWSPMatrix awsp(M, N, A_host);

    const size_t size_A_val = awsp.ValuesSize() * sizeof(float);
    const size_t size_A_bmp = awsp.BitmapsSize() * sizeof(uint32_t);

    float *A_val_dev;
    uint32_t *A_bmp_dev;
    CUDA_CHECK(cudaMalloc((void **)&A_val_dev, size_A_val));
    CUDA_CHECK(cudaMalloc((void **)&A_bmp_dev, size_A_bmp));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_val_dev, awsp.GetValues(), size_A_val, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_bmp_dev, awsp.GetBitmaps(), size_A_bmp, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    switch (version)
    {
        case 0:
            TIME_KERNEL((awsp_kernel_v0<<<grid, block>>>(
                M, N, 
                awsp.nz_bk_max_,
                A_bmp_dev,
                A_val_dev,
                X_device, Y_device
            )));
            break;
        case 1:
            TIME_KERNEL((awsp_kernel_v1<<<grid, block>>>(
                M, N, 
                awsp.nz_bk_max_,
                A_bmp_dev,
                A_val_dev,
                X_device, Y_device
            )));
            break;
        case 2:
            TIME_KERNEL((awsp_kernel_v2<<<grid, block>>>(
                M, N, 
                awsp.nz_bk_max_,
                A_bmp_dev,
                A_val_dev,
                X_device, Y_device
            )));
            break;
        default:
            break;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_val_dev));
    CUDA_CHECK(cudaFree(A_bmp_dev));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));
    return;
}