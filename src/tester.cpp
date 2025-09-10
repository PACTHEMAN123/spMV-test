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
    // Print();

    std::cout << "======== CPU start ======\n";
    SgemvCPU();
    SgemvGPU();

    std::cout << "======== GPU start ======\n";
    
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
    Y_gpu_host = (float *)malloc(1 * n_ * sizeof(float));
    spmv_gpu(m_, n_, A_host, X_host, Y_gpu_host);
}

auto SparseSgemvTester::CompareY() -> void {
    float max_diff = 0.0001f;
    for (int i = 0; i < n_; i++) {
        float diff = Y_cpu_host[i] - Y_gpu_host[i];
        if (abs(diff) > max_diff) {
            fprintf(stderr, "at [%d], cpu: %f, gpu: %f", i, Y_cpu_host[i], Y_gpu_host[i]);
            exit(EXIT_FAILURE);
        }
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

    double sparsity_ratio = 0.7;
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

auto SparseSgemvTester::GetRandomVector() -> void {
    X_host = (float *)malloc(m_ * 1 * sizeof(float));

    double sparsity_ratio = 0.2;
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

    for (int i = 0; i < m_; i++) {
        for (int j = 0; j < n_; j++) {
            float val = A_host[i * n_ + j];
            // 宽度 8，保留 4 位小数
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "\n";
    }

    std::cout << "=========================\n";

    for (int i = 0; i < m_; i++) {
        float val = X_host[i];
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "\n";

    std::cout << "=========================\n";

    for (int i = 0; i < A_host_compressed_size_; i++) {
        float val = A_host_compressed[i];
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
        if ((i + 1) % m_ == 0)
            std::cout << "\n";
    }
    std::cout << "\n";

}




