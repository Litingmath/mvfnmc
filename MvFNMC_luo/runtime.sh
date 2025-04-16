#!/bin/bash

# 定义日志文件
LOG_FILE="runtime_luo.txt"

# 清空原有日志文件内容（可选）
> "$LOG_FILE"

echo "运行 dataset: luo" | tee -a "$LOG_FILE"
python PyDTI.py --method="mvfnmc1new1" --dataset="luo" >> "$LOG_FILE" 2>&1
python PyDTI.py --method="mvfnmc2" --dataset="luo" >> "$LOG_FILE" 2>&1


echo "所有任务运行完毕" | tee -a "$LOG_FILE"
