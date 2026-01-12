#!/bin/bash
# ============================================================================
# ENHANCED REAL-TIME DRL TRAINING MONITOR
# ============================================================================
# Purpose: Comprehensive monitoring of federated DRL training with detailed
#          metrics tracking, accuracy analysis, and results visualization
# Usage: ./monitor_drl.sh [output_dir]
# Features: Training progress, per-agent accuracy, results comparison, 
#           live metrics, resource usage, performance trends
# ============================================================================

# Configuration
OUTPUT_DIR="${1:-results}"
LOG_FILE="$OUTPUT_DIR/training.log"
RESULTS_DIR="$OUTPUT_DIR"
REFRESH_INTERVAL=2  # seconds

# CSV Files - check multiple possible locations
PPO_CSV="$RESULTS_DIR/PPO_metrics.csv"
SAC_CSV="$RESULTS_DIR/SAC_metrics.csv"
TD3_CSV="$RESULTS_DIR/TD3_metrics.csv"
RANDOM_CSV="$RESULTS_DIR/Random_metrics.csv"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Unicode symbols
CHECK="✓"
CROSS="✗"
ARROW="→"
STAR="★"
PROGRESS="▶"

# Function to check and create directory structure
init_monitoring() {
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo -e "${YELLOW}Warning: Directory '$OUTPUT_DIR' not found${NC}"
        echo -e "${YELLOW}Creating results directory...${NC}"
        mkdir -p "$OUTPUT_DIR"
    fi
    
    # Check for results folder as fallback
    if [ ! -d "$RESULTS_DIR" ] && [ -d "results" ]; then
        RESULTS_DIR="results"
        echo -e "${GREEN}Found results directory${NC}"
    fi
}

# Function to get file modification time
get_last_update() {
    local file=$1
    if [ -f "$file" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            stat -f "%Sm" -t "%H:%M:%S" "$file" 2>/dev/null
        else
            stat -c "%y" "$file" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1
        fi
    else
        echo "N/A"
    fi
}

# Function to display comprehensive header
display_header() {
    clear
    local uptime=$((SECONDS / 60))
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}FEDERATED DRL TRAINING MONITOR - Real-Time Analytics${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} ${BLUE}Output Dir:${NC} %-50s ${CYAN}║${NC}" "$OUTPUT_DIR"
    echo -e "${CYAN}║${NC} ${BLUE}Monitor Uptime:${NC} %-38s ${CYAN}║${NC}" "${uptime} minutes"
    echo -e "${CYAN}║${NC} ${BLUE}Current Time:${NC} %-40s ${CYAN}║${NC}" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${CYAN}║${NC} ${BLUE}Refresh Rate:${NC} %-42s ${CYAN}║${NC}" "Every ${REFRESH_INTERVAL}s"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
}

# Function to extract latest metrics from CSV
get_latest_metric() {
    local csv_file=$1
    local column=$2
    
    if [ -f "$csv_file" ] && [ $(wc -l < "$csv_file") -gt 1 ]; then
        tail -n 1 "$csv_file" | cut -d',' -f"$column"
    else
        echo "N/A"
    fi
}

# Function to calculate average from CSV column
calc_average() {
    local csv_file=$1
    local column=$2
    
    if [ -f "$csv_file" ] && [ $(wc -l < "$csv_file") -gt 1 ]; then
        tail -n +2 "$csv_file" | cut -d',' -f"$column" | awk '{sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}'
    else
        echo "N/A"
    fi
}

# Function to get row count
count_rows() {
    local csv_file=$1
    if [ -f "$csv_file" ]; then
        echo $(($(wc -l < "$csv_file") - 1))
    else
        echo "0"
    fi
}

# Function to display training status
display_training_status() {
    echo -e "\n${GREEN}${PROGRESS} TRAINING STATUS:${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────${NC}"
    
    local ppo_rounds=$(count_rows "$PPO_CSV")
    local sac_rounds=$(count_rows "$SAC_CSV")
    local td3_rounds=$(count_rows "$TD3_CSV")
    local random_rounds=$(count_rows "$RANDOM_CSV")
    
    if [ "$ppo_rounds" -gt 0 ] || [ "$sac_rounds" -gt 0 ] || [ "$td3_rounds" -gt 0 ]; then
        echo -e "${BLUE}Active Training Sessions:${NC}"
        [ "$ppo_rounds" -gt 0 ] && echo -e "  ${CHECK} ${GREEN}PPO${NC}    - Round $ppo_rounds completed"
        [ "$sac_rounds" -gt 0 ] && echo -e "  ${CHECK} ${GREEN}SAC${NC}    - Round $sac_rounds completed"
        [ "$td3_rounds" -gt 0 ] && echo -e "  ${CHECK} ${GREEN}TD3${NC}    - Round $td3_rounds completed"
        [ "$random_rounds" -gt 0 ] && echo -e "  ${CHECK} ${YELLOW}Random${NC} - Round $random_rounds completed"
        
        echo -e "\n${BLUE}Training Progress:${NC}"
        local max_rounds=$(printf "%s\n" "$ppo_rounds" "$sac_rounds" "$td3_rounds" | sort -rn | head -n1)
        local total_expected=10
        local percent=$((max_rounds * 100 / total_expected))
        
        # Progress bar
        local filled=$((percent / 5))
        local empty=$((20 - filled))
        printf "  ["
        for i in $(seq 1 $filled); do printf "█"; done
        for i in $(seq 1 $empty); do printf "░"; done
        printf "] %3d%% (%d/%d rounds)\n" "$percent" "$max_rounds" "$total_expected"
    else
        echo -e "${YELLOW}${CROSS} No training data available yet${NC}"
        echo -e "${YELLOW}Waiting for training to start...${NC}"
    fi
}

# Function to display per-agent accuracy
display_agent_accuracy() {
    echo -e "\n${GREEN}🎯 PER-AGENT ACCURACY (Latest Round):${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────${NC}"
    
    printf "${BOLD}%-12s %-15s %-15s %-15s${NC}\n" "Agent" "Latest Acc" "Mean Acc" "Best Acc"
    echo "────────────────────────────────────────────────────────────────"
    
    # PPO metrics
    if [ -f "$PPO_CSV" ]; then
        local ppo_latest=$(get_latest_metric "$PPO_CSV" 2)
        local ppo_mean=$(calc_average "$PPO_CSV" 2)
        local ppo_best=$(tail -n +2 "$PPO_CSV" 2>/dev/null | cut -d',' -f2 | sort -rn | head -n1)
        [ -z "$ppo_best" ] && ppo_best="N/A"
        printf "%-12s ${GREEN}%-15s${NC} %-15s ${YELLOW}%-15s${NC}\n" "PPO" "$ppo_latest" "$ppo_mean" "$ppo_best"
    fi
    
    # SAC metrics
    if [ -f "$SAC_CSV" ]; then
        local sac_latest=$(get_latest_metric "$SAC_CSV" 2)
        local sac_mean=$(calc_average "$SAC_CSV" 2)
        local sac_best=$(tail -n +2 "$SAC_CSV" 2>/dev/null | cut -d',' -f2 | sort -rn | head -n1)
        [ -z "$sac_best" ] && sac_best="N/A"
        printf "%-12s ${GREEN}%-15s${NC} %-15s ${YELLOW}%-15s${NC}\n" "SAC" "$sac_latest" "$sac_mean" "$sac_best"
    fi
    
    # TD3 metrics
    if [ -f "$TD3_CSV" ]; then
        local td3_latest=$(get_latest_metric "$TD3_CSV" 2)
        local td3_mean=$(calc_average "$TD3_CSV" 2)
        local td3_best=$(tail -n +2 "$TD3_CSV" 2>/dev/null | cut -d',' -f2 | sort -rn | head -n1)
        [ -z "$td3_best" ] && td3_best="N/A"
        printf "%-12s ${GREEN}%-15s${NC} %-15s ${YELLOW}%-15s${NC}\n" "TD3" "$td3_latest" "$td3_mean" "$td3_best"
    fi
    
    # Random baseline
    if [ -f "$RANDOM_CSV" ]; then
        local rand_latest=$(get_latest_metric "$RANDOM_CSV" 2)
        local rand_mean=$(calc_average "$RANDOM_CSV" 2)
        local rand_best=$(tail -n +2 "$RANDOM_CSV" 2>/dev/null | cut -d',' -f2 | sort -rn | head -n1)
        [ -z "$rand_best" ] && rand_best="N/A"
        printf "%-12s ${RED}%-15s${NC} %-15s ${YELLOW}%-15s${NC}\n" "Random" "$rand_latest" "$rand_mean" "$rand_best"
    fi
}

# Function to display accuracy trends
display_accuracy_trends() {
    echo -e "\n${GREEN}📈 ACCURACY TRENDS (Last 5 Rounds):${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────${NC}"
    
    for agent_csv in "$PPO_CSV" "$SAC_CSV" "$TD3_CSV"; do
        if [ -f "$agent_csv" ]; then
            local agent_name=$(basename "$agent_csv" | cut -d'_' -f1)
            echo -e "${BLUE}${agent_name}:${NC}"
            
            # Extract last 5 rounds
            tail -n 5 "$agent_csv" | while IFS=',' read round acc reward; do
                if [ "$round" != "round" ]; then
                    # Create simple bar chart
                    local bar_length=$(printf "%.0f" $(echo "$acc * 50" | bc))
                    printf "  Round %-2s [" "$round"
                    for i in $(seq 1 $bar_length 2>/dev/null); do printf "█"; done
                    printf "] %.4f\n" "$acc"
                fi
            done
            echo ""
        fi
    done
}

# Function to display results comparison
display_results_comparison() {
    echo -e "\n${GREEN}🏆 RESULTS COMPARISON:${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────${NC}"
    
    if [ -f "$SUMMARY_CSV" ]; then
        echo -e "${BLUE}Final Results:${NC}"
        cat "$SUMMARY_CSV" | column -t -s ',' | tail -n +2
        
        # Highlight best performer
        echo -e "\n${YELLOW}${STAR} Best Performer:${NC}"
        tail -n +2 "$SUMMARY_CSV" | sort -t',' -k3 -rn | head -n1 | while IFS=',' read agent final_acc mean_acc std train_time; do
            echo -e "  Agent: ${GREEN}${BOLD}${agent}${NC}"
            echo -e "  Final Accuracy: ${GREEN}${final_acc}${NC}"
            echo -e "  Mean Accuracy: ${mean_acc}"
            echo -e "  Training Time: ${train_time}"
        done
    else
        echo -e "${YELLOW}Summary file not available yet${NC}"
    fi
}

# Function to display resource usage
display_resources() {
    echo -e "\n${GREEN}💻 RESOURCE USAGE:${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────${NC}"
    
    # Disk usage
    local disk_usage=$(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1)
    local file_count=$(find "$RESULTS_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    
    echo -e "${BLUE}Storage:${NC}"
    echo -e "  Disk Usage: ${disk_usage}"
    echo -e "  Files: ${file_count}"
    
    # Memory usage (if available)
    if command -v free &> /dev/null; then
        local mem_usage=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
        echo -e "  Memory: ${mem_usage}"
    fi
    
    # File timestamps
    echo -e "\n${BLUE}Last Updates:${NC}"
    [ -f "$PPO_CSV" ] && echo -e "  PPO: $(get_last_update "$PPO_CSV")"
    [ -f "$SAC_CSV" ] && echo -e "  SAC: $(get_last_update "$SAC_CSV")"
    [ -f "$TD3_CSV" ] && echo -e "  TD3: $(get_last_update "$TD3_CSV")"
}

# Function to display live training log
display_live_log() {
    echo -e "\n${GREEN}📝 RECENT ACTIVITY (Last 8 lines):${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────${NC}"
    
    if [ -f "$LOG_FILE" ]; then
        tail -n 8 "$LOG_FILE" | sed 's/^/  /'
    else
        echo -e "${YELLOW}  No log file available${NC}"
    fi
}

# Function to display statistics
display_statistics() {
    echo -e "\n${GREEN}📊 AGGREGATE STATISTICS:${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────${NC}"
    
    local total_rounds=0
    local avg_acc=0
    local agent_count=0
    
    for csv in "$PPO_CSV" "$SAC_CSV" "$TD3_CSV"; do
        if [ -f "$csv" ]; then
            local rounds=$(count_rows "$csv")
            local acc=$(calc_average "$csv" 2)
            
            total_rounds=$((total_rounds + rounds))
            if [ "$acc" != "N/A" ]; then
                avg_acc=$(echo "$avg_acc + $acc" | bc)
                agent_count=$((agent_count + 1))
            fi
        fi
    done
    
    if [ $agent_count -gt 0 ]; then
        avg_acc=$(echo "scale=4; $avg_acc / $agent_count" | bc)
        echo -e "${BLUE}Total Training Rounds:${NC} $total_rounds"
        echo -e "${BLUE}Average Accuracy (All Agents):${NC} ${GREEN}$avg_acc${NC}"
        echo -e "${BLUE}Active Agents:${NC} $agent_count"
        
        # Training efficiency
        if [ -f "$SUMMARY_CSV" ]; then
            local total_time=$(tail -n +2 "$SUMMARY_CSV" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.1f", sum}')
            echo -e "${BLUE}Total Training Time:${NC} ${total_time}s (${CYAN}$(echo "scale=1; $total_time/60" | bc)m${NC})"
        fi
    else
        echo -e "${YELLOW}No statistics available yet${NC}"
    fi
}

# Function to display footer
display_footer() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}${ARROW} Press Ctrl+C to exit | Refreshing in ${REFRESH_INTERVAL}s...${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
}

# Main monitoring loop
monitor_training() {
    while true; do
        display_header
        display_training_status
        display_agent_accuracy
        display_accuracy_trends
        display_results_comparison
        display_statistics
        display_resources
        display_live_log
        display_footer
        sleep $REFRESH_INTERVAL
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${GREEN}${CHECK} Monitoring stopped${NC}"; exit 0' INT TERM

# Initialize and start
init_monitoring

echo -e "${GREEN}${PROGRESS} Starting Enhanced DRL Training Monitor...${NC}"
echo -e "${CYAN}Looking for training data in: $OUTPUT_DIR${NC}"
sleep 2

monitor_training
