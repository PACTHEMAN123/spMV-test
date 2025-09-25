

/**
 * / to be opt... increase reg usage..
// X:(m,1) W(n/32,m*32) Y(n,1) W@X=Y
__global__ void gemv_a_sp_pip(
    int m,int n,
    float* X,
    float* W,
    float* Y
){
    //m*32,n/32....
    const int lane_id=threadIdx.x%32;
    const int warp_id=threadIdx.x/32;
    constexpr unsigned int full_lane_mask=0xffffffff;
    // design 32 tier pipeline....

    float x_buf[3]={0};
    x_buf[0]=0;x_buf[1]=0;
    float w_buf[32]={0};
    float sum=0;
    float* X_curr=X+(lane_id+m/4*warp_id);
    float* W_curr=W+(lane_id+m*32/4*warp_id)+(blockIdx.x)*(m*32);

    // head stage 1
    x_buf[0]=*X_curr;
    X_curr+=32;
    // head stage 2
    x_buf[1]=x_buf[0];
    x_buf[0]=*X_curr;
    X_curr+=32;
    for(int i_s=0;i_s<32;i_s++){
        float x_load_w=__shfl_sync(full_lane_mask,x_buf[1],i_s);
        if(x_load_w!=0){
            w_buf[i_s]=*W_curr;
        }
        W_curr+=32;
    }
    // main pipeline
    //main piple line stage num
    for(int i_b=0;i_b<(m/4-32*2);i_b+=32){
        x_buf[2]=x_buf[1];
        x_buf[1]=x_buf[0];
        x_buf[0]=*X_curr;
        X_curr+=32;
        for(int i_s=0;i_s<32;i_s++){//load w
            float x_calc=__shfl_sync(full_lane_mask,x_buf[2],i_s);
            if(x_calc!=0){
                sum+=w_buf[i_s]*x_calc;
            }
            float x_load_w=__shfl_sync(full_lane_mask,x_buf[1],i_s);
            if(x_load_w!=0){
                w_buf[i_s]=*W_curr;
            }
            W_curr+=32;
        }
    }
    // tail stage 0
    x_buf[2]=x_buf[1];
    x_buf[1]=x_buf[0];
    for(int i_s=0;i_s<32;i_s++){//load w
        float x_calc=__shfl_sync(full_lane_mask,x_buf[2],i_s);
        if(x_calc!=0){
            sum+=w_buf[i_s]*x_calc;
        }
        float x_load_w=__shfl_sync(full_lane_mask,x_buf[1],i_s);
        if(x_load_w!=0){
            w_buf[i_s]=*W_curr;
        }
        W_curr+=32;
    }
    // tail stage 1
    x_buf[2]=x_buf[1];
    for(int i_s=0;i_s<32;i_s++){//load w
        float x_calc=__shfl_sync(full_lane_mask,x_buf[2],i_s);
        if(x_calc!=0){
            sum+=w_buf[i_s]*x_calc;
        }
    }
    // prepare to write back...
    // block level reduce sum...
    __shared__ float reduce_sum_buf[3*32];
    if(warp_id!=0){
        reduce_sum_buf[(warp_id-1)*32+lane_id]=sum;
    }
    __syncthreads();
    if(warp_id==0){
        for(int i=0;i<3;i++){
            sum+=reduce_sum_buf[i*32+lane_id];
        }
    }
    Y[blockIdx.x*32+lane_id]=sum;
}

// double pipeline depth...
__global__ void gemv_a_sp_pip_r2(
    int m,int n,
    float* X,
    float* W,
    float* Y
){
    //m*32,n/32....
    const int lane_id=threadIdx.x%32;
    const int warp_id=threadIdx.x/32;
    constexpr unsigned int full_lane_mask=0xffffffff;
    // design 32 tier pipeline....

    // 2 reg 1 stage, total 32(lane)*1*2=64 x val 1 stage...
    float x_buf[3*2]={0};
    // 32 reg 1 stage, total 32*2=64 x val 1 stage...
    float w_buf[32*2]={0};
    float sum=0;
    float* X_curr=X+(lane_id+m/4*warp_id);
    float* W_curr=W+(lane_id+m*32/4*warp_id)+(blockIdx.x)*(m*32);

    // head stage 1
    LDG_F_CA_32(X_curr,x_buf[0]);
    LDG_F_CA_32(X_curr+32,x_buf[3]);
    X_curr+=64;
    // head stage 2
    x_buf[1]=x_buf[0];x_buf[4]=x_buf[3];
    LDG_F_CA_32(X_curr,x_buf[0]);
    LDG_F_CA_32(X_curr+32,x_buf[3]);
    X_curr+=64;
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){
        float x_load_w=__shfl_sync(full_lane_mask,x_buf[1],i_s);
        if(x_load_w!=0){
            LDG_F_CG_32(W_curr,w_buf[i_s]);
        }
        x_load_w=__shfl_sync(full_lane_mask,x_buf[4],i_s);
        if(x_load_w!=0){
            LDG_F_CG_32(W_curr+32,w_buf[i_s+32]); 
        }
        W_curr+=64;
    }
    // main pipeline
    //main piple line stage num
    #pragma unroll
    for(int i_b=0;i_b<(m/4-64*2);i_b+=64){
        x_buf[2]=x_buf[1];x_buf[5]=x_buf[4];
        x_buf[1]=x_buf[0];x_buf[4]=x_buf[3];
        LDG_F_CA_32(X_curr,x_buf[0]);
        LDG_F_CA_32(X_curr+32,x_buf[3]);
        X_curr+=64;
        #pragma unroll
        for(int i_s=0;i_s<32;i_s++){//load w
            float x_calc=__shfl_sync(full_lane_mask,x_buf[2],i_s);
            if(x_calc!=0){
                sum+=w_buf[i_s]*x_calc;
            }
            float x_load_w=__shfl_sync(full_lane_mask,x_buf[1],i_s);
            if(x_load_w!=0){
                LDG_F_CG_32(W_curr,w_buf[i_s]);
            }
            x_calc=__shfl_sync(full_lane_mask,x_buf[5],i_s);
            if(x_calc!=0){
                sum+=w_buf[i_s+32]*x_calc;
            }
            x_load_w=__shfl_sync(full_lane_mask,x_buf[4],i_s);
            if(x_load_w!=0){
                LDG_F_CG_32(W_curr+32,w_buf[i_s+32]); 
            }
            W_curr+=64;
        }
    }
    // tail stage 0
    x_buf[2]=x_buf[1];x_buf[5]=x_buf[4];
    x_buf[1]=x_buf[0];x_buf[4]=x_buf[3];
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){//load w
        float x_calc=__shfl_sync(full_lane_mask,x_buf[2],i_s);
        if(x_calc!=0){
            sum+=w_buf[i_s]*x_calc;
        }
        float x_load_w=__shfl_sync(full_lane_mask,x_buf[1],i_s);
        if(x_load_w!=0){
            LDG_F_CG_32(W_curr,w_buf[i_s]);
        }
        x_calc=__shfl_sync(full_lane_mask,x_buf[5],i_s);
        if(x_calc!=0){
            sum+=w_buf[i_s+32]*x_calc;
        }
        x_load_w=__shfl_sync(full_lane_mask,x_buf[4],i_s);
        if(x_load_w!=0){
            LDG_F_CG_32(W_curr+32,w_buf[i_s+32]); 
        }
        W_curr+=64;
    }
    // tail stage 1
    x_buf[2]=x_buf[1];x_buf[5]=x_buf[4];
    #pragma unroll
    for(int i_s=0;i_s<32;i_s++){//load w
        float x_calc=__shfl_sync(full_lane_mask,x_buf[2],i_s);
        if(x_calc!=0){
            sum+=w_buf[i_s]*x_calc;
        }
        x_calc=__shfl_sync(full_lane_mask,x_buf[5],i_s);
        if(x_calc!=0){
            sum+=w_buf[i_s+32]*x_calc;
        }
    }
    // prepare to write back...
    // block level reduce sum...
    __shared__ float reduce_sum_buf[3*32];
    if(warp_id!=0){
        reduce_sum_buf[(warp_id-1)*32+lane_id]=sum;
    }
    __syncthreads();
    if(warp_id==0){
        for(int i=0;i<3;i++){
            sum+=reduce_sum_buf[i*32+lane_id];
        }
    }
    Y[blockIdx.x*32+lane_id]=sum;
}

 */

#include "kernel.hpp"
#include "asp.hpp"

#include <iostream>

__global__ void asp_kernel_v0(
    int M, int N,
    float *A, float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    float sum = 0.0f;
    float *A_ptr = A + blockIdx.x * (M * 32) + (32 * M/4) * warp_id + lane_id;
    float *X_ptr = X + M/4 * warp_id + lane_id;

    for (int bk = 0; bk < M / 4; bk += 32) {
        // load x
        float x = *X_ptr;

        for (int i = 0; i < 32; i++) {
            // shuffle current x
            float cur_x = __shfl_sync(0xffffffff, x, i);
            if (cur_x != 0.0f)
                sum += *A_ptr * cur_x;
            A_ptr += 32;
        }
        X_ptr += 32;
    }

    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0;i < 3; i++){
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

__global__ void asp_kernel_v1(
    int M, int N,
    float *A, float *X, float *Y
) {
    
}

void asp_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
    dim3 block(128, 1, 1);
    dim3 grid(N / 32, 1, 1);

    ASPMatrix asp(M, N, A_host);

    const size_t size_A = asp.ValuesSize() * sizeof(float);
    float *A_device;
    CUDA_CHECK(cudaMalloc((void **)&A_device, size_A));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_device, asp.GetValues(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    TIME_KERNEL((asp_kernel_v0<<<grid, block>>>(
        M, N, 
        A_device, X_device, Y_device
    )));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));
    return;
}