#!/bin/bash

# 数据源、分类方式和采样比例的组合
DATA_SOURCES=("reddit" "sina" "tweet")
CLASSIFICATION_TYPES=("ac-1" "ac-2" "ac-3" "ac-4" "ac-5" "ac-6" "ac-7" "ac-8" "ac-9" "ac-10" "grouping" "hc-1" "hc-2" "hc-3" "hc-4" "hc-5")
SAMPLING_RATES=("10percent")
VSLoss_tau=("3.0" "4.0")
# 获取当前目录的绝对路径
CURRENT_DIR=$(pwd)
DATA_DIR="${CURRENT_DIR}/data"
for tau in "${VSLoss_tau[@]}"; do
  # 确保日志目录存在
  mkdir -p logs_VSLoss_DRW_tau_${tau}
  for source in "${DATA_SOURCES[@]}"; do
    for class_type in "${CLASSIFICATION_TYPES[@]}"; do
      for rate in "${SAMPLING_RATES[@]}"; do

        # 相对路径用于命令参数，绝对路径用于目录检查
        relative_prefix="data/${source}-${class_type}-onlyends_with_isolated_bi_${rate}_hop1"
        absolute_data_dir="${DATA_DIR}/${source}-${class_type}-onlyends_with_isolated_bi_${rate}_hop1"
        base_cmd="python main.py --data_prefix ${relative_prefix} --train_config ./train_config/gat_192_8_with_graph.yml --repeat_time 1 --gpu 0 --dir_log /root/autodl-tmp/test --loss_type VSLoss --train_rule DRW --DRW_epoch 20 --VSLoss_tau ${tau}"

        # 创建日志文件名
        log_file="logs_VSLoss_DRW_tau_${tau}/${source}_${class_type}_${rate}_loss_type_VSLoss_DRW_tau_${tau}.log"
        echo "Running experiment: ${source}-${class_type}-${rate}" > "${log_file}"
        echo "Command: ${base_cmd}" >> "${log_file}"
        echo "-----------------------------" >> "${log_file}"

        # 检查数据目录是否存在
        if [ ! -d "${absolute_data_dir}" ]; then
          echo "WARNING: Data directory does not exist: ${absolute_data_dir}" | tee -a "${log_file}"
          echo "Skipping this experiment" | tee -a "${log_file}"
          echo "----------------------------------------" | tee -a "${log_file}"
          continue
        fi

        f1_total=0

        for i in {1..5}; do
          echo "Run $i:" >> "${log_file}"
          tmp_output=$(mktemp)
          
          eval "${base_cmd}" > "${tmp_output}" 2>&1

          # 取最后3行写入日志
          tail -n 3 "${tmp_output}" >> "${log_file}"

          # 提取 F1 分数（匹配 `F1 = 0.xxxx`，提取数值）
          f1=$(grep -oP 'F1 = \K[0-9.]+' "${tmp_output}" | tail -n 1)
          echo "Extracted F1: $f1" >> "${log_file}"

          # 累加 F1
          f1_total=$(awk -v total="$f1_total" -v val="$f1" 'BEGIN {printf "%.4f", total + val}')
          
          rm "${tmp_output}"
          echo "-----------------------------" >> "${log_file}"
        done

        # 计算平均值
        f1_avg=$(awk -v total="$f1_total" 'BEGIN {printf "%.4f", total / 5}')
        echo "Average F1 over 5 runs: $f1_avg" >> "${log_file}"

        echo "Experiment completed for ${source}-${class_type}-${rate}-loss_type_VSLoss_DRW_tau_${tau}"
        echo "----------------------------------------"
      done
    done
  done
done
echo "All experiments completed!"
shutdown -h +1