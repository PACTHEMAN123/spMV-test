#include "awsp_ref.hpp"
#include "kernel.hpp"
#include "ptx.hpp"


__global__ void awsp_ref_kernel(
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
    float X_lw[2];
    float X_cc[2];
    uint32_t B_pf[2];
    uint32_t B_lw[2];
    float A_buf[2][32]; // only use to compute

    // head stage 0: load the x and bitmap for compute
    LDG_F_CA_32(X_ptr, X_pf[0]);
    LDG_F_CA_32(X_ptr + 32, X_pf[1]); X_ptr += 64;
    LDG_U_CA_32(B_ptr, B_pf[0]);
    LDG_U_CA_32(B_ptr + 32, B_pf[1]); B_ptr += 64;
    // head stage 1: load x and bitmap for next load
    X_lw[0] = X_pf[0]; LDG_F_CA_32(X_ptr, X_pf[0]);
    X_lw[1] = X_pf[1]; LDG_F_CA_32(X_ptr + 32, X_pf[1]); X_ptr += 64;
    B_lw[0] = B_pf[0]; LDG_U_CA_32(B_ptr, B_pf[0]);
    B_lw[1] = B_pf[1]; LDG_U_CA_32(B_ptr + 32, B_pf[1]); B_ptr += 64;
    // head stage 2: load A for compute
    #pragma unroll 32
    for (int i = 0; i < 32; i++) {
        float cur_x = __shfl_sync(full_lane_mask, X_lw[0], i);
        uint32_t cur_bitmap = __shfl_sync(full_lane_mask, B_lw[0], i);
        int A_val_row_offset = __popc(cur_bitmap & prev_mask);
        bool is_load = (cur_bitmap & curr_mask) && (cur_x != 0.0f);
        A_buf[0][i] = is_load ? *(A_ptr + A_val_row_offset) : 0;
        A_ptr += __popc(cur_bitmap);
    }
    #pragma unroll 32
    for (int i = 0; i < 32; i++) {
        float cur_x = __shfl_sync(full_lane_mask, X_lw[1], i);
        uint32_t cur_bitmap = __shfl_sync(full_lane_mask, B_lw[1], i);
        int A_val_row_offset = __popc(cur_bitmap & prev_mask);
        bool is_load = (cur_bitmap & curr_mask) && (cur_x != 0.0f);
        A_buf[1][i] = is_load ? *(A_ptr + A_val_row_offset) : 0;
        A_ptr += __popc(cur_bitmap);
    }

    
    // main pipeline
    for (int out_bk = 0; out_bk < (M/4 - 64*2); out_bk += 64) {
        // load the x and bitmap for next load
        B_lw[0] = B_pf[0]; 
        B_lw[1] = B_pf[1]; 
        LDG_U_CA_32(B_ptr, B_pf[0]);
        LDG_U_CA_32(B_ptr + 32, B_pf[1]); 
        B_ptr += 64;
        X_cc[0] = X_lw[0]; 
        X_cc[1] = X_lw[1];
        X_lw[0] = X_pf[0]; 
        X_lw[1] = X_pf[1]; 
        LDG_F_CA_32(X_ptr, X_pf[0]);
        LDG_F_CA_32(X_ptr + 32, X_pf[1]);
        X_ptr += 64;

        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            // consume bufferd A
            float x_calc = __shfl_sync(full_lane_mask, X_cc[0], i);
            float a = A_buf[0][i];
            // todo: branch or not ?
            sum += a * x_calc;

            // load new A
            float x_load = __shfl_sync(full_lane_mask, X_lw[0], i);
            uint32_t b_load = __shfl_sync(full_lane_mask, B_lw[0], i);
            int new_nz_num = __popc(b_load);
            int A_val_row_offset = __popc(b_load & prev_mask);
            bool is_load = (b_load & curr_mask) && x_load != 0.0f;
            A_buf[0][i] = is_load ? *(A_ptr + A_val_row_offset) : 0;
            A_ptr += new_nz_num;
        }
        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            // consume bufferd A
            float x_calc = __shfl_sync(full_lane_mask, X_cc[1], i);
            float a = A_buf[1][i];
            // todo: branch or not ?
            sum += a * x_calc;

            // load new A
            float x_load = __shfl_sync(full_lane_mask, X_lw[1], i);
            uint32_t b_load = __shfl_sync(full_lane_mask, B_lw[1], i);
            int new_nz_num = __popc(b_load);
            int A_val_row_offset = __popc(b_load & prev_mask);
            bool is_load = (b_load & curr_mask) && x_load != 0.0f;
            A_buf[1][i] = is_load ? *(A_ptr + A_val_row_offset) : 0;
            A_ptr += new_nz_num;
        }
    }

    // tail stage
    // tail stage 0 (no need to load new x and bitmap)
    B_lw[0] = B_pf[0]; 
    B_lw[1] = B_pf[1];
    X_cc[0] = X_lw[0]; 
    X_lw[0] = X_pf[0];
    X_cc[1] = X_lw[1]; 
    X_lw[1] = X_pf[1];
    #pragma unroll 32
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(full_lane_mask, X_cc[0], i);
        float a = A_buf[0][i];
        sum += a * x_calc;

        float x_load = __shfl_sync(full_lane_mask, X_lw[0], i);
        uint32_t b_load = __shfl_sync(full_lane_mask, B_lw[0], i);
        int new_nz_num = __popc(b_load);
        int A_val_row_offset = __popc(b_load & prev_mask);
        bool is_load = (b_load & curr_mask) && x_load != 0.0f;
        A_buf[0][i] = is_load ? *(A_ptr + A_val_row_offset) : 0;
        A_ptr += new_nz_num;
    }
    #pragma unroll 32
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(full_lane_mask, X_cc[1], i);
        float a = A_buf[1][i];
        sum += a * x_calc;

        float x_load = __shfl_sync(full_lane_mask, X_lw[1], i);
        uint32_t b_load = __shfl_sync(full_lane_mask, B_lw[1], i);
        int new_nz_num = __popc(b_load);
        int A_val_row_offset = __popc(b_load & prev_mask);
        bool is_load = (b_load & curr_mask) && x_load != 0.0f;
        A_buf[1][i] = is_load ? *(A_ptr + A_val_row_offset) : 0;
        A_ptr += new_nz_num;
    }

    
    // tail stage 1 (no need to load new x)
    X_cc[0] = X_lw[0];
    X_cc[1] = X_lw[1];
    #pragma unroll 32
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_cc[0], i);
        float a = A_buf[0][i];
        sum += a * x_calc;
    }
    #pragma unroll 32
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_cc[1], i);
        float a = A_buf[1][i];
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

// using fill lane slot to avoid uncoalesced LDG
__global__ void gemv_aw_sp_r2(
    int m,int n,
    float* X,float* W,float* Y,
    unsigned int* Bitmap,
    int* warp_nz_offset 
){
    const int lane_id=threadIdx.x%32;
    const int warp_id=threadIdx.x/32;
    constexpr unsigned int full_lane_mask=0xffffffff;
    const unsigned prex_pos_mask=~(0xffffffff<<lane_id);
    const unsigned int curr_lane_mask=0x1<<lane_id;
    constexpr int pipeline_depth=64;
    constexpr int SZ_L=pipeline_depth/32;
    constexpr int SZ_G=pipeline_depth;
    // design 64+2+2 tier pipeline....
    // def pipline struct registers...
    float           x_pf[SZ_L];
    float           x_lw[SZ_L];
    float           x_cc[SZ_L];
    unsigned int    bm_pf[SZ_L];
    unsigned int    bm_lw[SZ_L];    
    float           w_pf[SZ_G];
    
    unsigned int block_nz_offset=warp_nz_offset[3];
    unsigned int curr_nz_offset=warp_id==0?0:warp_nz_offset[warp_id-1];
    
    float*          x_curr       =X      +(lane_id+m/4*warp_id);
    unsigned int*   bitmap_curr  =Bitmap +(lane_id+m/4*warp_id) + blockIdx.x*m;
    float*          w_curr       =W      +(curr_nz_offset)+(blockIdx.x)*(block_nz_offset);
    float sum=0;
    // head stage 0
    LDG_U_CA_32(bitmap_curr,bm_pf[0]);
    LDG_U_CA_32(bitmap_curr+32,bm_pf[1]);
    bitmap_curr+=64;
    LDG_F_CA_32(x_curr,x_pf[0]);
    LDG_F_CA_32(x_curr+32,x_pf[1]);
    x_curr+=64;
    // head stage 1
    bm_lw[0]=bm_pf[0];bm_lw[1]=bm_pf[1];
    LDG_U_CA_32(bitmap_curr,bm_pf[0]);
    LDG_U_CA_32(bitmap_curr+32,bm_pf[1]);
    bitmap_curr+=64;
    x_lw[0]=x_pf[0];x_lw[1]=x_pf[1];
    LDG_F_CA_32(x_curr,x_pf[0]);
    LDG_F_CA_32(x_curr+32,x_pf[1]);
    x_curr+=64;
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){
        float x_val_lw=__shfl_sync(full_lane_mask,x_lw[0],i_s);
        unsigned int bitmap_val=__shfl_sync(full_lane_mask,bm_lw[0],i_s);
        unsigned int new_nz_num=__popc(bitmap_val);
        unsigned int local_offset=__popc(bitmap_val&prex_pos_mask);
        if((x_val_lw!=0.0f) && (curr_lane_mask&bitmap_val)){
            LDG_F_CA_32(w_curr+local_offset,w_pf[i_s]);
        }else{
            w_pf[i_s]=0;
        }
        w_curr+=new_nz_num;
    }
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){
        float x_val_lw=__shfl_sync(full_lane_mask,x_lw[1],i_s);
        unsigned int bitmap_val=__shfl_sync(full_lane_mask,bm_lw[1],i_s);
        unsigned int new_nz_num=__popc(bitmap_val);
        unsigned int local_offset=__popc(bitmap_val&prex_pos_mask);
        if((x_val_lw!=0.0f) && (curr_lane_mask&bitmap_val)){
            LDG_F_CA_32(w_curr+local_offset,w_pf[i_s+32]);
        }else{
            w_pf[i_s+32]=0;
        }
        w_curr+=new_nz_num;
    }

    // main pipline loop
    for(int i_b=0;i_b<(m/4-64*2);i_b+=64){//@TODO
        // i+2 stage... load x, load bitmap
        bm_lw[0]=bm_pf[0];bm_lw[1]=bm_pf[1];
        LDG_U_CA_32(bitmap_curr,bm_pf[0]);
        LDG_U_CA_32(bitmap_curr+32,bm_pf[1]);
        bitmap_curr+=64;
        x_cc[0]=x_lw[0];x_cc[1]=x_lw[1];
        x_lw[0]=x_pf[0];x_lw[1]=x_pf[1];
        LDG_F_CA_32(x_curr,x_pf[0]);
        LDG_F_CA_32(x_curr+32,x_pf[1]);
        x_curr+=64;
        // 0-31
        #pragma unroll
        for(int i_s=0;i_s<32;i_s++){
            // i stage... calc
            float x_val=__shfl_sync(full_lane_mask,x_cc[0],i_s);
            float w_val=w_pf[i_s];
            // if(x_val!=0.0f){
                sum+=x_val*w_val;
            // }
            // i+1 stage... load w (nz value)
            float x_val_lw=__shfl_sync(full_lane_mask,x_lw[0],i_s);
            unsigned int bitmap_val=__shfl_sync(full_lane_mask,bm_lw[0],i_s);
            unsigned int new_nz_num=__popc(bitmap_val);
            unsigned int local_offset=__popc(bitmap_val&prex_pos_mask);
            if((x_val_lw!=0.0f) && (curr_lane_mask&bitmap_val)){
                LDG_F_CA_32(w_curr+local_offset,w_pf[i_s]);
            }else{
                w_pf[i_s]=0;
            }
            w_curr+=new_nz_num;
        }
        // 32-63
        #pragma unroll
        for(int i_s=0;i_s<32;i_s++){
            // i stage... calc
            float x_val=__shfl_sync(full_lane_mask,x_cc[1],i_s);
            float w_val=w_pf[i_s+32];
            // if(x_val!=0.0f){
                sum+=x_val*w_val;
            // }
            // i+1 stage... load w (nz value)
            float x_val_lw=__shfl_sync(full_lane_mask,x_lw[1],i_s);
            unsigned int bitmap_val=__shfl_sync(full_lane_mask,bm_lw[1],i_s);
            unsigned int new_nz_num=__popc(bitmap_val);
            unsigned int local_offset=__popc(bitmap_val&prex_pos_mask);
            if((x_val_lw!=0.0f) && (curr_lane_mask&bitmap_val)){
                LDG_F_CA_32(w_curr+local_offset,w_pf[i_s+32]);
            }else{
                w_pf[i_s+32]=0;
            }
            w_curr+=new_nz_num;
        }
    }
    // tail stage 0
    bm_lw[0]=bm_pf[0];bm_lw[1]=bm_pf[1];
    x_cc[0]=x_lw[0];x_cc[1]=x_lw[1];
    x_lw[0]=x_pf[0];x_lw[1]=x_pf[1];
   
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){
        // i stage... calc
        float x_val=__shfl_sync(full_lane_mask,x_cc[0],i_s);
        float w_val=w_pf[i_s];
        // if(x_val!=0.0f){
            sum+=x_val*w_val;
        // }
        // i+1 stage... load w (nz value)
        float x_val_lw=__shfl_sync(full_lane_mask,x_lw[0],i_s);
        unsigned int bitmap_val=__shfl_sync(full_lane_mask,bm_lw[0],i_s);
        unsigned int new_nz_num=__popc(bitmap_val);
        unsigned int local_offset=__popc(bitmap_val&prex_pos_mask);
        if((x_val_lw!=0.0f) && (curr_lane_mask&bitmap_val)){
            LDG_F_CA_32(w_curr+local_offset,w_pf[i_s]);
        }else{
            w_pf[i_s]=0;
        }
        w_curr+=new_nz_num;
    }
    // 32-63
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){
        // i stage... calc
        float x_val=__shfl_sync(full_lane_mask,x_cc[1],i_s);
        float w_val=w_pf[i_s+32];
        // if(x_val!=0.0f){
            sum+=x_val*w_val;
        // }
        // i+1 stage... load w (nz value)
        float x_val_lw=__shfl_sync(full_lane_mask,x_lw[1],i_s);
        unsigned int bitmap_val=__shfl_sync(full_lane_mask,bm_lw[1],i_s);
        unsigned int new_nz_num=__popc(bitmap_val);
        unsigned int local_offset=__popc(bitmap_val&prex_pos_mask);
        if((x_val_lw!=0.0f) && (curr_lane_mask&bitmap_val)){
            LDG_F_CA_32(w_curr+local_offset,w_pf[i_s+32]);
        }else{
            w_pf[i_s+32]=0;
        }
        w_curr+=new_nz_num;
    }
    // tail stage 1
    x_cc[0]=x_lw[0];x_cc[1]=x_lw[1];
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){
        // i stage... calc
        float x_val=__shfl_sync(full_lane_mask,x_cc[0],i_s);
        float w_val=w_pf[i_s];
        // if(x_val!=0.0f){
            sum+=x_val*w_val;
        // }
    }
    // 32-63
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){
        // i stage... calc
        float x_val=__shfl_sync(full_lane_mask,x_cc[1],i_s);
        float w_val=w_pf[i_s+32];
        // if(x_val!=0.0f){
            sum+=x_val*w_val;
        // }
    }

    __shared__ float reduce_sum_buf[3*32];
    if(warp_id!=0){
        reduce_sum_buf[(warp_id-1)*32+lane_id]=sum;
    }
    __syncthreads();
    if(warp_id==0){
        for(int i=0;i<3;i++){
            sum+=reduce_sum_buf[i*32+lane_id];
        }
        Y[blockIdx.x*32+lane_id]=sum;
    }
}

void awsp_ref_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
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

    TIME_KERNEL((awsp_ref_kernel<<<grid, block>>>(
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