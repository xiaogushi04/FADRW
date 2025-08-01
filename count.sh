#!/bin/bash

cd logs_LDAM_%1

# 定义 CSV 文件名
CSV_FILE="f1_scores.csv"

# 清空或创建 CSV 文件并写入表头
echo "日志文件名,原始F1分数（5个）,有效F1分数（去掉最小值后4个）,F1平均值,F1标准差" > "$CSV_FILE"

# 颜色定义（用于终端输出）
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 恢复默认颜色

total_avg_sum=0
log_count=0

echo -e "\n${YELLOW}==== 开始处理日志文件 ====${NC}\n"

for logfile in *.log; do
    echo -e "${GREEN}处理文件: $logfile${NC}"
    
    # 提取 F1 分数（最多5个，且小于1）
    mapfile -t f1s < <(grep "Extracted F1:" "$logfile" | grep -oP '[0-9.]+' | awk '$1 < 1' | head -n 5)
    
    # 显示原始F1分数
    echo -e "  原始F1分数: ${RED}${f1s[*]}${NC}"
    
    # 检查是否足够5个分数
    if [[ ${#f1s[@]} -lt 5 ]]; then
        echo -e "  ${RED}跳过: 有效F1分数不足5个${NC}\n"
        continue
    fi
    
    # 找到最小值（使用awk比较浮点数）
    min_val=${f1s[0]}
    min_idx=0
    for i in {1..4}; do
        if awk -v a="${f1s[$i]}" -v b="$min_val" 'BEGIN {exit (a >= b)}'; then
            min_val=${f1s[$i]}
            min_idx=$i
        fi
    done
    
    # 构建去掉最小值后的数组
    four_f1s=()
    for i in {0..4}; do
        [[ $i -ne $min_idx ]] && four_f1s+=(${f1s[$i]})
    done
    
    # 计算平均值（使用awk）
    sum=$(awk -v a="${four_f1s[0]}" -v b="${four_f1s[1]}" -v c="${four_f1s[2]}" -v d="${four_f1s[3]}" 'BEGIN {printf "%.4f", a + b + c + d}')
    avg=$(awk -v sum="$sum" 'BEGIN {printf "%.4f", sum / 4}')
    
    # 计算标准差
    variance_sum=0
    for score in "${four_f1s[@]}"; do
        diff=$(awk -v score="$score" -v avg="$avg" 'BEGIN {printf "%.4f", score - avg}')
        squared_diff=$(awk -v diff="$diff" 'BEGIN {printf "%.8f", diff * diff}')
        variance_sum=$(awk -v sum="$variance_sum" -v squared="$squared_diff" 'BEGIN {printf "%.8f", sum + squared}')
    done
    variance=$(awk -v sum="$variance_sum" 'BEGIN {printf "%.8f", sum / 4}')
    stddev=$(awk -v variance="$variance" 'BEGIN {printf "%.8f", sqrt(variance)}')
    
    # 格式化输出为百分比（保留2位小数）
    avg_percent=$(awk -v avg="$avg" 'BEGIN {printf "%.2f", avg * 100}')
    stddev_percent=$(awk -v stddev="$stddev" 'BEGIN {printf "%.2f", stddev * 100}')
    
    # 终端输出
    echo -e "  有效F1分数: ${GREEN}${four_f1s[*]}${NC}"
    echo -e "  平均值: ${YELLOW}$avg_percent${NC}"
    echo -e "  标准差: ${YELLOW}$stddev_percent${NC}\n"
    
    # 写入CSV（字段用双引号包裹）
    echo "\"$logfile\",\"${f1s[*]}\",\"${four_f1s[*]}\",$avg_percent,$stddev_percent" >> "$CSV_FILE"
    
    # 更新统计（使用awk累加浮点数）
    total_avg_sum=$(awk -v sum="$total_avg_sum" -v avg="$avg_percent" 'BEGIN {printf "%.2f", sum + avg}')
    ((log_count++))
done

# 计算总体平均值
overall_avg=$(awk -v sum="$total_avg_sum" -v count="$log_count" 'BEGIN {printf "%.2f", sum / count}')

# 终端输出统计信息
echo -e "${YELLOW}==== 处理完成 ====${NC}"
echo -e "有效日志文件数: ${GREEN}$log_count${NC}"
echo -e "F1平均值总和: ${YELLOW}$total_avg_sum${NC}"
echo -e "总体F1平均值: ${YELLOW}$overall_avg${NC}"
echo -e "详细结果已保存到: ${GREEN}$CSV_FILE${NC}"