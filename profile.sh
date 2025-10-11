#!/bin/bash

EXE_PATH=./build/sparse_sgemv

# 报告保存目录
KERNEL_NAME="gemv"
REPORT_DIR="./profile"
mkdir -p ${REPORT_DIR}

# 输出文件名
REPORT_FILE="${REPORT_DIR}/${KERNEL_NAME}.ncu-rep"

echo "运行 Nsight Compute..."
echo "  可执行文件: ${EXE_PATH}"
echo "  报告文件: ${REPORT_FILE}"

# 调用 ncu
rm ${REPORT_FILE}
sudo -v
sudo -E CUDA_VISIBLE_DEVICES=2 /usr/local/cuda/nsight-compute-2025.1.0/ncu --import-source yes -o ${REPORT_FILE} -f --set full ${EXE_PATH}

echo "完成 ✅ 报告已保存到 ${REPORT_FILE}"