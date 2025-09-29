#include "kernel.hpp"
#include "wsp.hpp"

__global__ void wsp_kernel_v0(
    int M, int N,
    int nz_max_m,
    uint32_t *bitmaps,
    float* A_vals,
    float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int cur_col = blockIdx.x * 4 + warp_id;

    uint32_t curr_mask = (1u << lane_id);
    uint32_t prev_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);

    int A_val_row_cnt = 0;

    float sum = 0;

    // outer loop: each loop, 32 block
    for (int out_bk = 0; out_bk < M; out_bk += 32 * 32) {
        // for each warp, load 32 bitmaps for a big loop
        int bmp_start = cur_col * (M / 32) + out_bk / 32;
        uint32_t bitmap = bitmaps[bmp_start + lane_id];
        int nz_num = __popc(bitmap);

        // inner loop: each loop, 32 threads finish 1 block
        for (int i = 0; i < 32; i++) {
            uint32_t cur_bitmap = __shfl_sync(0xffffffff, bitmap, i);
            
            if (cur_bitmap & curr_mask) {
                int x_start = out_bk + i * 32;
                float x = X[x_start + lane_id];

                int A_val_in_blk_offset = __popc(cur_bitmap & prev_mask);
                int a_start = nz_max_m * cur_col + A_val_row_cnt;
                float a = A_vals[a_start + A_val_in_blk_offset];

                sum += x * a;
            }

            // update nz values counter
            int cur_nz_num = __shfl_sync(0xffffffff, nz_num, i);
            A_val_row_cnt += cur_nz_num;
        }
    }

    for (int i = 16; i >= 1; i >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, i);
    }

    if (lane_id == 0)
        Y[cur_col] = sum;
}

// weight sparse kernel: using pipeline
__global__ void wsp_kernel_v1(
    int M, int N,
    int nz_max_m,
    uint32_t *bitmaps,
    float* A_vals,
    float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int cur_col = blockIdx.x * 4 + warp_id;

    uint32_t curr_mask = (1u << lane_id);
    uint32_t prev_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);

    int A_val_row_cnt = 0;
    float sum = 0;
    float *X_ptr = X + lane_id;
    float *A_ptr = A_vals + nz_max_m * cur_col;

    float A_buf[2] = {0};
    float X_buf[2] = {0};

    

    for (int out_bk = 0; out_bk < M; out_bk += 32 * 32) {
        int bmp_start = cur_col * (M / 32) + out_bk / 32;
        uint32_t bitmap = bitmaps[bmp_start + lane_id];

        int stage = 0;

        // head stage: load the first 32 A and X
        uint32_t first_bitmap =  __shfl_sync(0xffffffff, bitmap, 0);
        if (first_bitmap & curr_mask) {
            X_buf[stage] = *X_ptr;
            int A_val_in_blk_offset = __popc(first_bitmap & prev_mask);
            A_buf[stage] = A_ptr[A_val_in_blk_offset];
        }
        X_ptr += 32;
        int cur_nz_num = __popc(first_bitmap);
        A_val_row_cnt += cur_nz_num; A_ptr += cur_nz_num;
        stage ^= 1;

        // main loop
        #pragma unroll
        for (int i = 0; i < 31; i++) {

            // compute
            uint32_t cur_bitmap = __shfl_sync(0xffffffff, bitmap, i); 
            if (cur_bitmap & curr_mask) {
                sum += X_buf[stage ^ 1] * A_buf[stage ^ 1];
            }

            // load next
            uint32_t next_bitmap = __shfl_sync(0xffffffff, bitmap, i + 1);
            if (next_bitmap & curr_mask) {
                X_buf[stage] = *X_ptr;
                int A_val_in_blk_offset = __popc(next_bitmap & prev_mask);
                A_buf[stage] = A_ptr[A_val_in_blk_offset];
            }
            X_ptr += 32;
            int cur_nz_num = __popc(next_bitmap);
            A_val_row_cnt += cur_nz_num; A_ptr += cur_nz_num;
            stage ^= 1;
        }

        // tail stage: no need to load the next
        uint32_t end_bitmap = __shfl_sync(0xffffffff, bitmap, 31);
        if (end_bitmap & curr_mask) {
            sum += X_buf[stage ^ 1] * A_buf[stage ^ 1];
        }
    }

    #pragma unroll
    for (int i = 16; i >= 1; i >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, i);
    }

    if (lane_id == 0)
        Y[cur_col] = sum;
}

void wsp_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host, int version) {
    dim3 block(128, 1, 1);
    dim3 grid(N / 4, 1, 1);

    WSPMatrix wsp(M, N, A_host);

    const size_t size_A_val = wsp.ValuesSize() * sizeof(float);
    const size_t size_A_bmp = wsp.BitmapsSize() * sizeof(uint32_t);

    float *A_val_dev;
    uint32_t *A_bmp_dev;
    CUDA_CHECK(cudaMalloc((void **)&A_val_dev, size_A_val));
    CUDA_CHECK(cudaMalloc((void **)&A_bmp_dev, size_A_bmp));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_val_dev, wsp.GetValues(), size_A_val, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_bmp_dev, wsp.GetBitmaps(), size_A_bmp, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    switch (version)
    {
        case 0:
            TIME_KERNEL((wsp_kernel_v0<<<grid, block>>>(
                M, N, 
                wsp.nz_max_m,
                A_bmp_dev,
                A_val_dev,
                X_device, Y_device
            )));
            break;
        case 1:
            TIME_KERNEL((wsp_kernel_v1<<<grid, block>>>(
                M, N, 
                wsp.nz_max_m,
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

