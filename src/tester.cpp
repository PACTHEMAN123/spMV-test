#include "tester.hpp"
#include "kernel.hpp"

SparseSgemvTester::SparseSgemvTester(int m, int n)
    : m_(m), n_(n) {
        // now we only test for 32-aligned sgemv
        assert(m % 32 == 0);
        assert(n % 32 == 0);
}



auto SparseSgemvTester::RunTest() -> void {
    std::cout << "=== Sparse SGEMV Test ===\n";

    GetRandomMatrix();
    GetRandomVector();
    GetCompressedMatrix();
    GenerateBitMap();
    // Print();

    std::cout << "======== CPU start ======\n";
    SgemvCPU();
    

    std::cout << "======== GPU start ======\n";
    SgemvGPU();
    
    CompareY();
    
    std::cout << "========== OK ===========\n";
}

auto SparseSgemvTester::SgemvCPU() -> void {
    Y_cpu_host = (float *)malloc(1 * n_ * sizeof(float));
    for (int i = 0; i < n_; i++) {
        float acc = 0.0f;
        for (int j = 0; j < m_; j++) {
            acc += X_host[j] * A_host[j * n_ + i];
        }
        Y_cpu_host[i] = acc;
    }
}

auto SparseSgemvTester::SgemvGPU() -> void {
    // register host and run kernels

    // naive
    float *naive_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(naive_kernel_Y_host);
    std::cout << "start to launch naive kernel" << std::endl;
    naive_gemv_gpu(m_, n_, A_host, X_host, naive_kernel_Y_host);

    // cublas
    float *cublas_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(cublas_kernel_Y_host);
    std::cout << "start to launch cublas kernel" << std::endl;
    cublas_gemv_gpu(m_, n_, A_host, X_host, cublas_kernel_Y_host);

    // tiling + share mem
    float *tiling_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(tiling_kernel_Y_host);
    std::cout << "start to launch tiling kernel" << std::endl;
    tiling_gemv_gpu(m_, n_, A_host, X_host, tiling_kernel_Y_host);

    // naive csr
    float *csr_naive_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(csr_naive_kernel_Y_host);
    std::cout << "start to launch csr naive kernel" << std::endl;
    csr_naive_gemv_gpu(m_, n_, A_host, X_host, csr_naive_kernel_Y_host);

    // csr + tiling
    float *csr_tiling_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(csr_tiling_kernel_Y_host);
    std::cout << "start to launch csr tiling kernel" << std::endl;
    csr_tiling_gemv_gpu(m_, n_, A_host, X_host, csr_tiling_kernel_Y_host);

    // wsp
    float *wsp_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(wsp_kernel_Y_host);
    std::cout << "start to launch wsp kernel" << std::endl;
    wsp_gemv_gpu(m_, n_, A_host, X_host, wsp_kernel_Y_host);

    // asp v0
    float *asp0_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(asp0_kernel_Y_host);
    std::cout << "start to launch asp v0 kernel" << std::endl;
    asp_gemv_gpu(m_, n_, A_host, X_host, asp0_kernel_Y_host, 0);

    // asp v1
    float *asp1_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(asp1_kernel_Y_host);
    std::cout << "start to launch asp v1 kernel" << std::endl;
    asp_gemv_gpu(m_, n_, A_host, X_host, asp1_kernel_Y_host, 1);

    // asp v2
    float *asp2_kernel_Y_host = (float *)malloc(1 * n_ * sizeof(float));
    Y_gpu_hosts.push_back(asp2_kernel_Y_host);
    std::cout << "start to launch asp v2 kernel" << std::endl;
    asp_gemv_gpu(m_, n_, A_host, X_host, asp2_kernel_Y_host, 2);
}

auto SparseSgemvTester::CompareY() -> void {
    float max_diff = 0.001f;

    int idx = 0;
    for (auto host: Y_gpu_hosts) {
        for (int i = 0; i < n_; i++) {
            float diff = Y_cpu_host[i] - host[i];
            if (abs(diff) > max_diff) {
                fprintf(stderr, "[GPU kernel %d] at [%d], cpu: %f, gpu: %f\n", idx, i, Y_cpu_host[i], host[i]);
                //exit(EXIT_FAILURE);
            }
        }
        idx += 1;
    } 
}

auto SparseSgemvTester::PrintCPU() -> void {

    std::cout << "=========================\n";

    for (int i = 0; i < n_; i++) {
        float val = Y_cpu_host[i];
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "\n";

    std::cout << "=========================\n";
}

auto SparseSgemvTester::GetRandomMatrix() -> void {
    A_host = (float *)malloc(m_ * n_ * sizeof(float));

    double sparsity_ratio = 0.5;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> valueDist(-1.0f, 1.0f);   // 非零元素的值分布
    std::uniform_real_distribution<double> probDist(0.0, 1.0);         // 控制稀疏率

    for (int i = 0; i < m_; i++) {
        for (int j = 0; j < n_; j++) {
            if (probDist(gen) > sparsity_ratio) {
                A_host[i * n_ + j] = valueDist(gen); // 生成非零值
            } else {
                A_host[i * n_ + j] = 0.0f;           // 留空为零
            }
        }
    }
}

auto SparseSgemvTester::GetCompressedMatrix() -> void {
    int compressed_size = 0;
    A_host_compressed = (float *)malloc(m_ * n_ * sizeof(float));
    for (int i = 0; i < m_; i++) {
        for (int j = 0; j < n_; j++) {
            if (A_host[i * n_ + j] != 0.0f) {
                A_host_compressed[compressed_size] = A_host[i * n_ + j];
                compressed_size += 1;
            }
        }
    }
    A_host_compressed_size_ = compressed_size;
}

auto SparseSgemvTester::GenerateBitMap() -> void {
    int num_of_u32 = m_ * n_ / 32;
    bitmap = (uint32_t *)malloc(num_of_u32 * sizeof(uint32_t));
    for (int idx = 0; idx < num_of_u32; idx++) {
        // for each u32
        uint32_t u32_bitmap = 0;
        for (int i = 0; i < 32; i++) {
            if (A_host[idx * 32 + i])
                u32_bitmap |= 1u << i;
        }
        bitmap[idx] = u32_bitmap;
    }
}

auto SparseSgemvTester::GetRandomVector() -> void {
    X_host = (float *)malloc(m_ * 1 * sizeof(float));

    double sparsity_ratio = 0.5;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> valueDist(-1.0f, 1.0f);   // 非零元素的值分布
    std::uniform_real_distribution<double> probDist(0.0, 1.0);         // 控制稀疏率

    for (int i = 0; i < m_; i++) {
        if (probDist(gen) > sparsity_ratio) {
            X_host[i] = valueDist(gen); // 生成非零值
        } else {
            X_host[i] = 0.0f;           // 留空为零
        }
    }
}

auto SparseSgemvTester::Print() -> void {

    std::cout << "=========================\n";

    // print the matrix A

    for (int i = 0; i < m_; i++) {
        for (int j = 0; j < n_; j++) {
            float val = A_host[i * n_ + j];
            // 宽度 8，保留 4 位小数
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "\n";
    }

    std::cout << "=========================\n";

    // print the vector x

    for (int i = 0; i < m_; i++) {
        float val = X_host[i];
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "\n";

    std::cout << "=========================\n";

    // print the compressed matrix

    for (int i = 0; i < A_host_compressed_size_; i++) {
        float val = A_host_compressed[i];
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
        if ((i + 1) % m_ == 0)
            std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "=========================\n";

    // print the bitmap
    for (int row = 0; row < m_; row++) {
        for (int col = 0; col < n_; col++) {
            int flatIdx = row * n_ + col;
            int wordIdx = flatIdx / 32;
            int bitIdx  = flatIdx % 32;

            bool isNonZero = (bitmap[wordIdx] >> bitIdx) & 1u;
            std::cout << (isNonZero ? 1 : 0) << " ";
        }
        std::cout << "\n";
    }

}




