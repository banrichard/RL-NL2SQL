#!/usr/bin/env bash
set -euo pipefail

# Auto-run GRPO with dynamic GPU selection and Copilot fallback for non-resource errors.
# Automatically runs in screen/tmux to prevent accidental termination.

# Check if running inside screen or tmux session
if [[ -z "${STY:-}" && -z "${TMUX:-}" && "${SKIP_SCREEN_CHECK:-}" != "1" ]]; then
    echo "[INFO] Not running in screen/tmux. Auto-starting in screen..."
    
    # Generate session name
    SESSION_NAME="grpo_$(date +%Y%m%d_%H%M%S)"
    
    # Try screen first (more commonly available)
    if command -v screen >/dev/null 2>&1; then
        echo "[INFO] Launching in screen session: $SESSION_NAME"
        echo "[INFO] To attach: screen -r $SESSION_NAME"
        echo "[INFO] To detach: Ctrl+A then D"
        
        # Export all current environment variables and run in screen
        export SKIP_SCREEN_CHECK=1
        exec screen -dmS "$SESSION_NAME" bash "$0" "$@"
        
    # Fallback to tmux
    elif command -v tmux >/dev/null 2>&1; then
        echo "[INFO] Launching in tmux session: $SESSION_NAME"
        echo "[INFO] To attach: tmux attach -t $SESSION_NAME"
        echo "[INFO] To detach: Ctrl+B then D"
        
        export SKIP_SCREEN_CHECK=1
        exec tmux new-session -d -s "$SESSION_NAME" "bash '$0' $*"
        
    else
        echo "[WARN] Neither screen nor tmux found. Running in foreground (vulnerable to disconnection)."
        echo "[WARN] Install screen or tmux for better reliability: sudo apt install screen"
    fi
fi

GRPO_SCRIPT="${GRPO_SCRIPT:-/home/banlujie/RL-NL2SQL/scripts/grpo.sh}"
REQUIRED_GPUS="${REQUIRED_GPUS:-4}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-43008}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-100}"
MAX_RETRIES="${MAX_RETRIES:-120}"
RETRY_WAIT_SEC="${RETRY_WAIT_SEC:-45}"
LOG_DIR="${LOG_DIR:-/home/banlujie/RL-NL2SQL/logs/grpo_auto}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/banlujie/RL-NL2SQL}"
# Set to 1 to stop on non-resource errors, 0 to keep retrying all errors
STOP_ON_NON_RESOURCE_ERROR="${STOP_ON_NON_RESOURCE_ERROR:-0}"

mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_${RUN_ID}.log"

echo "[INFO] run_id=$RUN_ID" | tee -a "$MASTER_LOG"
echo "[INFO] grpo_script=$GRPO_SCRIPT" | tee -a "$MASTER_LOG"
if [[ -n "${STY:-}" ]]; then
    echo "[INFO] Running in screen session: $STY" | tee -a "$MASTER_LOG"
elif [[ -n "${TMUX:-}" ]]; then
    echo "[INFO] Running in tmux session" | tee -a "$MASTER_LOG"
fi

if [[ ! -f "$GRPO_SCRIPT" ]]; then
    echo "[ERROR] GRPO script not found: $GRPO_SCRIPT" | tee -a "$MASTER_LOG"
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found, cannot detect GPU resources." | tee -a "$MASTER_LOG"
    exit 2
fi

pick_available_gpus() {
    local required="$1"
    local min_free="$2"
    local selected=()

    while IFS=',' read -r idx free_mb; do
        idx="${idx//[[:space:]]/}"
        free_mb="${free_mb//[[:space:]]/}"

        if [[ -z "$idx" || -z "$free_mb" ]]; then
            continue
        fi

        if (( free_mb >= min_free )); then
            selected+=("${idx},${free_mb}")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    if (( ${#selected[@]} < required )); then
        return 1
    fi

    IFS=$'\n' read -r -d '' -a selected_sorted < <(
        printf '%s\n' "${selected[@]}" | sort -t',' -k2,2nr && printf '\0'
    )

    local out=""
    local i
    for ((i = 0; i < required; i++)); do
        local picked_idx
        picked_idx="${selected_sorted[$i]%%,*}"
        if [[ -z "$out" ]]; then
            out="$picked_idx"
        else
            out+=";$picked_idx"
        fi
    done

    echo "$out" | tr ';' ','
}

is_resource_error() {
    local log_file="$1"

    if [[ ! -f "$log_file" ]]; then
        return 1
    fi

    local pattern='CUDA out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED|resource temporarily unavailable|No CUDA GPUs are available|all CUDA-capable devices are busy|CUDA error: invalid device ordinal|RayOutOfMemoryError|NCCL.*(timeout|unhandled system error)|failed to allocate|cannot allocate memory'
    if grep -Eqi "$pattern" "$log_file"; then
        return 0
    fi

    return 1
}

invoke_copilot_cli() {
    local fail_log="$1"
    local attempt="$2"
    local summary

    summary="$(tail -n 200 "$fail_log")"

    echo "[INFO] Non-resource error detected at attempt=$attempt, invoking Copilot CLI..." | tee -a "$MASTER_LOG"

    local prompt
    prompt="Please analyze and fix the bug in /home/banlujie/RL-NL2SQL that causes scripts/grpo.sh to fail.\n\nExecution context:\n- CUDA_VISIBLE_DEVICES was auto-selected\n- Attempt: $attempt\n- Failing script: $GRPO_SCRIPT\n\nLast 200 lines of error log:\n$summary\n\nRequirements:\n1) Identify root cause\n2) Apply minimal safe code fix\n3) Explain what changed and why\n4) Provide verification command"

    if command -v copilot >/dev/null 2>&1; then
        if copilot -p "$prompt" \
            --add-dir "$WORKSPACE_DIR" \
            --allow-all-tools \
            --allow-all-paths \
            --no-ask-user; then
            echo "[INFO] Copilot CLI executed with write-enabled permissions." | tee -a "$MASTER_LOG"
            return 0
        fi
    fi

    if command -v gh >/dev/null 2>&1; then
        if gh copilot suggest -t shell "$prompt"; then
            echo "[INFO] Copilot CLI executed via 'gh copilot suggest -t shell'." | tee -a "$MASTER_LOG"
            return 0
        fi
        if gh copilot suggest "$prompt"; then
            echo "[INFO] Copilot CLI executed via 'gh copilot suggest'." | tee -a "$MASTER_LOG"
            return 0
        fi
    fi

    echo "[WARN] Copilot CLI not available or invocation failed." | tee -a "$MASTER_LOG"
    return 1
}

attempt=1
while (( attempt <= MAX_RETRIES )); do
    echo "[INFO] attempt=$attempt/$MAX_RETRIES checking GPU resources..." | tee -a "$MASTER_LOG"

    gpu_list="$(pick_available_gpus "$REQUIRED_GPUS" "$MIN_FREE_MEM_MB" || true)"

    if [[ -z "$gpu_list" ]]; then
        echo "[WARN] Not enough free GPUs. need=$REQUIRED_GPUS min_free_mb=$MIN_FREE_MEM_MB, retry in ${RETRY_WAIT_SEC}s" | tee -a "$MASTER_LOG"
        sleep "$RETRY_WAIT_SEC"
        ((attempt++))
        continue
    fi

    export CUDA_VISIBLE_DEVICES="$gpu_list"
    echo "[INFO] selected CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$MASTER_LOG"

    attempt_log="$LOG_DIR/attempt_${RUN_ID}_${attempt}.log"

    set +e
    bash "$GRPO_SCRIPT" 2>&1 | tee "$attempt_log"
    exit_code=${PIPESTATUS[0]}
    set -e

    echo "[INFO] attempt=$attempt exit_code=$exit_code" | tee -a "$MASTER_LOG"

    if (( exit_code == 0 )); then
        echo "[INFO] GRPO finished successfully." | tee -a "$MASTER_LOG"
        exit 0
    fi

    if is_resource_error "$attempt_log"; then
        echo "[WARN] Resource-related failure detected, will retry in ${RETRY_WAIT_SEC}s." | tee -a "$MASTER_LOG"
        sleep "$RETRY_WAIT_SEC"
        ((attempt++))
        continue
    fi

    # Non-resource error detected
    echo "[WARN] Non-resource failure detected at attempt=$attempt" | tee -a "$MASTER_LOG"
    invoke_copilot_cli "$attempt_log" "$attempt" || true
    
    if (( STOP_ON_NON_RESOURCE_ERROR == 1 )); then
        echo "[ERROR] STOP_ON_NON_RESOURCE_ERROR=1, stopping after Copilot invocation." | tee -a "$MASTER_LOG"
        exit "$exit_code"
    else
        echo "[INFO] STOP_ON_NON_RESOURCE_ERROR=0, will retry after ${RETRY_WAIT_SEC}s..." | tee -a "$MASTER_LOG"
        sleep "$RETRY_WAIT_SEC"
        ((attempt++))
        continue
    fi
done

echo "[ERROR] Exhausted retries ($MAX_RETRIES) due to resource constraints." | tee -a "$MASTER_LOG"
exit 1
