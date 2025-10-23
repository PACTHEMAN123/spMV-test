#include "awsp_ref.hpp"
#include "kernel.hpp"
#include "ptx.hpp"


__global__ void wsp_sm_kernel(
    int M, int N,
    uint32_t *bitmaps,
    float* A_vals,
    float *X, float *Y,
    int* warp_offset
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    const uint32_t curr_mask = (1u << lane_id);
    const uint32_t prev_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);
    const uint32_t full_lane_mask = 0xffffffff;

    int block_nz_num = warp_offset[3];
    int block_warp_offset = warp_id == 0 ? 0 : warp_offset[warp_id - 1];

    float sum = 0;
    float *X_ptr = X + M/4*warp_id + lane_id;
    float *A_ptr = A_vals + block_nz_num * blockIdx.x + block_warp_offset;
    uint32_t *B_ptr = bitmaps + (blockIdx.x*M/32 + M/32*warp_id/4) * 32 + lane_id;

    // pipeline buffers
    float X_pf[2];
    float X_cc[2];
    uint32_t B_pf[2];
    uint32_t B_lw[2];
    __shared__ float A_smem[32*64*2*4];
    float *A_sm = A_smem + warp_id * 32*64*2;
    unsigned int A_sm_buf_offset = 0; // double buffer
    unsigned int A_sm_buf_length = (1 << 11);


    // head stage 0
    LDG_U_CA_32(B_ptr, B_pf[0]);
    LDG_U_CA_32(B_ptr + 32, B_pf[1]); B_ptr += 64;
    // head stage 1
    LDG_F_CA_32(X_ptr, X_pf[0]);
    LDG_F_CA_32(X_ptr + 32, X_pf[1]); X_ptr += 64;
    B_lw[0] = B_pf[0]; LDG_U_CA_32(B_ptr, B_pf[0]);
    B_lw[1] = B_pf[1]; LDG_U_CA_32(B_ptr + 32, B_pf[1]); B_ptr += 64;
    #pragma unroll
    for (int idx = 0; idx < 2; idx++) {
        for (int i = 0; i < 32; i++) {
            float *A_sm_ptr = A_sm + A_sm_buf_offset + (idx*32 + i) * 32 + lane_id;
            uint32_t cur_bitmap = __shfl_sync(full_lane_mask, B_lw[idx], i);
            int A_val_row_offset = __popc(cur_bitmap & prev_mask);
            unsigned int is_load = cur_bitmap & curr_mask;
            // CP_ASYNC_CA_32_PLEZ(A_sm_ptr, A_ptr + A_val_row_offset, is_load);
            if (is_load) {
                *A_sm_ptr = *(A_ptr + A_val_row_offset);
            } else {
                *A_sm_ptr = 0;
            }
            A_ptr += __popc(cur_bitmap);
        }
    }
    A_sm_buf_offset ^= A_sm_buf_length;
    // CP_ASYNC_COMMIT_GROUP();
    // head stage 2
    X_cc[0] = X_pf[0]; LDG_F_CA_32(X_ptr, X_pf[0]);
    X_cc[1] = X_pf[1]; LDG_F_CA_32(X_ptr + 32, X_pf[1]); 
    X_ptr += 64;
    B_lw[0] = B_pf[0]; LDG_U_CA_32(B_ptr, B_pf[0]);
    B_lw[1] = B_pf[1]; LDG_U_CA_32(B_ptr + 32, B_pf[1]);
    B_ptr += 64;
    // CP_ASYNC_WAIT_GROUP(1);
    #pragma unroll
    for (int idx = 0; idx < 2; idx++) {
        for (int i = 0; i < 32; i++) {
            float *A_sm_ptr = A_sm + A_sm_buf_offset + (idx*32 + i) * 32 + lane_id;
            uint32_t cur_bitmap = __shfl_sync(full_lane_mask, B_lw[idx], i);
            int A_val_row_offset = __popc(cur_bitmap & prev_mask);
            unsigned int is_load = cur_bitmap & curr_mask;
            // CP_ASYNC_CA_32_PLEZ(A_sm_ptr, A_ptr + A_val_row_offset, is_load);
            if (is_load) {
                *A_sm_ptr = *(A_ptr + A_val_row_offset);
            } else {
                *A_sm_ptr = 0;
            }
            A_ptr += __popc(cur_bitmap);
        }
    }
    A_sm_buf_offset ^= A_sm_buf_length;
    // CP_ASYNC_COMMIT_GROUP();

    
    // main pipeline
    for (int out_bk = 0; out_bk < (M/4 - 64*3); out_bk += 64) {
        // load the x and bitmap for next load
        B_lw[0] = B_pf[0]; 
        B_lw[1] = B_pf[1]; 
        LDG_U_CA_32(B_ptr, B_pf[0]);
        LDG_U_CA_32(B_ptr + 32, B_pf[1]); 
        B_ptr += 64;
        
        // CP_ASYNC_WAIT_GROUP(1);
        #pragma unroll
        for (int idx = 0; idx < 2; idx++) {
            #pragma unroll 32
            for (int i = 0; i < 32; i++) {
                // consume bufferd A
                float x_calc = __shfl_sync(full_lane_mask, X_cc[idx], i);
                float *A_sm_ptr = A_sm + A_sm_buf_offset + (idx*32 + i) * 32 + lane_id;
                
                float a = *A_sm_ptr;
                sum += a * x_calc;

                uint32_t b_load = __shfl_sync(full_lane_mask, B_lw[idx], i);
                int new_nz_num = __popc(b_load);
                int A_val_row_offset = __popc(b_load & prev_mask);
                unsigned int is_load = b_load & curr_mask;
                if (is_load) {
                    *A_sm_ptr = *(A_ptr + A_val_row_offset);
                } else {
                    *A_sm_ptr = 0;
                }
                // CP_ASYNC_CA_32_PLEZ(A_sm_ptr, A_ptr + A_val_row_offset, is_load);
                A_ptr += new_nz_num;
            }
        }
        A_sm_buf_offset ^= A_sm_buf_length;
        // CP_ASYNC_COMMIT_GROUP();
        X_cc[0] = X_pf[0]; 
        X_cc[1] = X_pf[1]; 
        LDG_F_CA_32(X_ptr, X_pf[0]);
        LDG_F_CA_32(X_ptr + 32, X_pf[1]);
        X_ptr += 64;
    }

    // tail stage
    // tail stage 0 (no need to load new x and bitmap)
    B_lw[0] = B_pf[0]; 
    B_lw[1] = B_pf[1];
    // CP_ASYNC_WAIT_GROUP(1);
    #pragma unroll
    for (int idx = 0; idx < 2; idx++) {
        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            float x_calc = __shfl_sync(full_lane_mask, X_cc[idx], i);
            float *A_sm_ptr = A_sm + A_sm_buf_offset + (idx*32 + i) * 32 + lane_id;
            float a = *A_sm_ptr;
            sum += a * x_calc;

            uint32_t b_load = __shfl_sync(full_lane_mask, B_lw[idx], i);
            int new_nz_num = __popc(b_load);
            int A_val_row_offset = __popc(b_load & prev_mask);
            unsigned int is_load = b_load & curr_mask;
            // CP_ASYNC_CA_32_PLEZ(A_sm_ptr, A_ptr + A_val_row_offset, is_load);
            if (is_load) {
                *A_sm_ptr = *(A_ptr + A_val_row_offset);
            } else {
                *A_sm_ptr = 0;
            }
            A_ptr += new_nz_num;
        }
    }
    A_sm_buf_offset ^= A_sm_buf_length;
    // CP_ASYNC_COMMIT_GROUP();
    X_cc[0] = X_pf[0]; 
    X_cc[1] = X_pf[1];
    LDG_F_CA_32(X_ptr, X_pf[0]);
    LDG_F_CA_32(X_ptr + 32, X_pf[1]);
    X_ptr += 64;

    // tail stage 1 (no need to load new x)
    // CP_ASYNC_WAIT_GROUP(1);
    #pragma unroll
    for (int idx = 0; idx < 2; idx++) {
        for (int i = 0; i < 32; i++) {
            float x_calc = __shfl_sync(0xffffffff, X_cc[idx], i);
            float *A_sm_ptr = A_sm + A_sm_buf_offset + (idx*32 + i) * 32 + lane_id;
            float a = *A_sm_ptr;
            sum += a * x_calc;
        }
    }
    A_sm_buf_offset ^= A_sm_buf_length;
    X_cc[0] = X_pf[0];
    X_cc[1] = X_pf[1];

    // tail stage 2 consume the last As
    // CP_ASYNC_WAIT_GROUP(0);
    #pragma unroll
    for (int idx = 0; idx < 2; idx++) {
        for (int i = 0; i < 32; i++) {
            float x_calc = __shfl_sync(0xffffffff, X_cc[idx], i);
            float *A_sm_ptr = A_sm + A_sm_buf_offset + (idx*32 + i) * 32 + lane_id;
            float a = *A_sm_ptr;
            sum += a * x_calc;
        }
    }
    A_sm_buf_offset ^= A_sm_buf_length;
    
    // reduction
    __shared__ __align__(128) float reduce_sum[3][32];
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

void wsp_sm_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
    dim3 block(128, 1, 1);
    dim3 grid(N / 32, 1, 1);

    AWSPRefMatrix awsp(M, N, A_host);

    const size_t size_A_val = awsp.ValuesSize() * sizeof(float);
    const size_t size_A_bmp = awsp.BitmapsSize() * sizeof(uint32_t);
    const size_t size_A_warp_offset = 4 * sizeof(int);

    float *A_val_dev;
    uint32_t *A_bmp_dev;
    int *A_warp_nz_offset;
    CUDA_CHECK(cudaMalloc((void **)&A_val_dev, size_A_val));
    CUDA_CHECK(cudaMalloc((void **)&A_bmp_dev, size_A_bmp));
    CUDA_CHECK(cudaMalloc((void **)&A_warp_nz_offset, size_A_warp_offset));


    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_val_dev, awsp.GetValues(), size_A_val, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_bmp_dev, awsp.GetBitmaps(), size_A_bmp, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_warp_nz_offset, awsp.GetWarpNZOffset(), size_A_warp_offset, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    TIME_KERNEL((wsp_sm_kernel<<<grid, block>>>(
        M, N, 
        A_bmp_dev,
        A_val_dev,
        X_device, Y_device,
        A_warp_nz_offset
    )));

    // TIME_KERNEL((gemv_aw_sp_r2<<<grid, block>>>(M, N, X_device, A_val_dev, Y_device, A_bmp_dev, A_warp_nz_offset)));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_val_dev));
    CUDA_CHECK(cudaFree(A_bmp_dev));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));
    return;
}