#!/bin/bash
# ==============================================================================
# HyperGKAN 多数据集并行训练脚本
# 
# 最优策略：每张GPU独立运行一个数据集，4张卡并行跑4个实验
# 比DataParallel多卡跑单个实验效率高得多（无通信开销，GPU利用率100%）
#
# 用法：
#   chmod +x run_all_datasets.sh
#   ./run_all_datasets.sh
#
# 说明：
#   - GPU 0: Temperature (温度)
#   - GPU 1: Cloud (云量)
#   - GPU 2: Humidity (湿度)
#   - GPU 3: Wind (风速)
#   - GPU 4-5: 空闲，可用于其他实验或消融实验
# ==============================================================================

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
BASE_CONFIG="${CONFIG_DIR}/config.yaml"

echo "=============================================="
echo "  HyperGKAN 多数据集并行训练"
echo "=============================================="
echo "  基础配置: ${BASE_CONFIG}"
echo "  GPU 数量: $(nvidia-smi -L | wc -l)"
echo ""

# 为每个数据集生成独立的配置文件
generate_config() {
    local dataset=$1
    local config_out="${CONFIG_DIR}/config_${dataset}.yaml"
    
    # 复制基础配置
    cp "${BASE_CONFIG}" "${config_out}"
    
    # 使用 sed 修改 dataset_selection（将所有设为false，再将目标设为true）
    sed -i "s/Temperature: true/Temperature: false/" "${config_out}"
    sed -i "s/Cloud: true/Cloud: false/" "${config_out}"
    sed -i "s/Humidity: true/Humidity: false/" "${config_out}"
    sed -i "s/Wind: true/Wind: false/" "${config_out}"
    
    # 将目标数据集设为 true
    sed -i "s/${dataset}: false/${dataset}: true/" "${config_out}"
    
    echo "  ✓ 生成配置: ${config_out} (${dataset}=true)"
}

# 生成4个独立配置
echo "生成数据集专用配置文件..."
generate_config "Temperature"
generate_config "Cloud"
generate_config "Humidity"
generate_config "Wind"
echo ""

# 并行启动训练
echo "启动并行训练..."
echo "=============================================="

# GPU 0: Temperature
echo "[GPU 0] 启动 Temperature 训练..."
CUDA_VISIBLE_DEVICES=0 nohup python "${SCRIPT_DIR}/train.py" \
    --config "configs/config_Temperature.yaml" \
    --gpu 0 \
    > "${SCRIPT_DIR}/logs/train_Temperature.log" 2>&1 &
PID_TEMP=$!
echo "  PID: ${PID_TEMP}"

# GPU 1: Cloud
echo "[GPU 1] 启动 Cloud 训练..."
CUDA_VISIBLE_DEVICES=1 nohup python "${SCRIPT_DIR}/train.py" \
    --config "configs/config_Cloud.yaml" \
    --gpu 0 \
    > "${SCRIPT_DIR}/logs/train_Cloud.log" 2>&1 &
PID_CLOUD=$!
echo "  PID: ${PID_CLOUD}"

# GPU 2: Humidity
echo "[GPU 2] 启动 Humidity 训练..."
CUDA_VISIBLE_DEVICES=2 nohup python "${SCRIPT_DIR}/train.py" \
    --config "configs/config_Humidity.yaml" \
    --gpu 0 \
    > "${SCRIPT_DIR}/logs/train_Humidity.log" 2>&1 &
PID_HUMIDITY=$!
echo "  PID: ${PID_HUMIDITY}"

# GPU 3: Wind
echo "[GPU 3] 启动 Wind 训练..."
CUDA_VISIBLE_DEVICES=3 nohup python "${SCRIPT_DIR}/train.py" \
    --config "configs/config_Wind.yaml" \
    --gpu 0 \
    > "${SCRIPT_DIR}/logs/train_Wind.log" 2>&1 &
PID_WIND=$!
echo "  PID: ${PID_WIND}"

echo ""
echo "=============================================="
echo "  所有训练已在后台启动！"
echo "=============================================="
echo ""
echo "  进程PID:"
echo "    Temperature (GPU 0): ${PID_TEMP}"
echo "    Cloud      (GPU 1): ${PID_CLOUD}"
echo "    Humidity   (GPU 2): ${PID_HUMIDITY}"
echo "    Wind       (GPU 3): ${PID_WIND}"
echo ""
echo "  查看实时日志:"
echo "    tail -f ${SCRIPT_DIR}/logs/train_Temperature.log"
echo "    tail -f ${SCRIPT_DIR}/logs/train_Cloud.log"
echo "    tail -f ${SCRIPT_DIR}/logs/train_Humidity.log"
echo "    tail -f ${SCRIPT_DIR}/logs/train_Wind.log"
echo ""
echo "  查看GPU使用情况:"
echo "    watch -n 2 nvidia-smi"
echo ""
echo "  终止所有训练:"
echo "    kill ${PID_TEMP} ${PID_CLOUD} ${PID_HUMIDITY} ${PID_WIND}"
echo ""

# 创建logs目录（如果不存在）
mkdir -p "${SCRIPT_DIR}/logs"

# 等待所有任务完成
echo "等待所有训练完成 (Ctrl+C 不会终止后台进程)..."
wait ${PID_TEMP} ${PID_CLOUD} ${PID_HUMIDITY} ${PID_WIND}

echo ""
echo "=============================================="
echo "  ✅ 所有训练已完成！"
echo "=============================================="
