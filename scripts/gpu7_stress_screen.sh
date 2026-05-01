#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-gpu7_stress}"
TARGET_GPU="${TARGET_GPU:-7}"
TARGET_GPUS="${TARGET_GPUS:-}"
ALLOC_GB="${ALLOC_GB:-0}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/gpu7_stress_screen.sh start   # 启动 screen 后台占卡
  bash scripts/gpu7_stress_screen.sh stop    # 停止占卡
  bash scripts/gpu7_stress_screen.sh status  # 查看状态
  bash scripts/gpu7_stress_screen.sh attach  # 进入 screen 会话

Env:
  TARGET_GPU   默认 7（兼容旧参数，单卡）
  TARGET_GPUS  逗号分隔多卡，如 6,7（优先于 TARGET_GPU）
  SESSION_NAME 默认 gpu7_stress
  ALLOC_GB     >0 时尝试固定申请指定显存(GB)；0 时自动尽量吃满
  FREE_MEM_USED_MB  认为“空卡”的显存占用上限(MB)，默认 512
  FREE_UTIL_PCT     认为“空卡”的利用率上限(%)，默认 5
  POLL_SEC          轮询间隔(秒)，默认 2
EOF
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "缺少命令: $1"
        exit 1
    }
}

start_session() {
    require_cmd screen
    require_cmd python3

    if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
        echo "screen 会话已存在: ${SESSION_NAME}"
        exit 0
    fi

    local gpu_list="${TARGET_GPUS}"
    if [[ -z "${gpu_list}" ]]; then
        gpu_list="${TARGET_GPU}"
    fi

    screen -dmS "${SESSION_NAME}" bash -lc "CUDA_VISIBLE_DEVICES=${gpu_list} ALLOC_GB=${ALLOC_GB} FREE_MEM_USED_MB=${FREE_MEM_USED_MB:-512} FREE_UTIL_PCT=${FREE_UTIL_PCT:-5} POLL_SEC=${POLL_SEC:-2} python3 -u - <<'PY'
import os
import subprocess
import time

import torch

if not torch.cuda.is_available():
    raise SystemExit('CUDA 不可用，无法占卡')

target_gb = float(os.environ.get('ALLOC_GB', '0') or '0')
num_devices = torch.cuda.device_count()
free_mem_used_mb = int(float(os.environ.get('FREE_MEM_USED_MB', '512') or '512'))
free_util_pct = int(float(os.environ.get('FREE_UTIL_PCT', '5') or '5'))
poll_sec = float(os.environ.get('POLL_SEC', '2') or '2')
visible_raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')

print(f'Using physical GPUs from CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\")}')
print(f'Visible device count: {num_devices}')
print(f'Free thresholds: memory_used<={free_mem_used_mb}MB, util<={free_util_pct}%')
print(f'Poll interval: {poll_sec}s')

elem_size = torch.tensor([], dtype=torch.float16).element_size()
safety_bytes = int(256 * 1024 * 1024)
chunk_bytes = 256 * 1024 * 1024
chunk_elems = chunk_bytes // elem_size

physical_ids = []
if visible_raw.strip():
    for x in visible_raw.split(','):
        x = x.strip()
        if x.isdigit():
            physical_ids.append(int(x))
if len(physical_ids) < num_devices:
    physical_ids = list(range(num_devices))

slots = []
for idx in range(num_devices):
    device = torch.device(f'cuda:{idx}')
    phys = physical_ids[idx] if idx < len(physical_ids) else idx
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(f'[cuda:{idx} -> gpu:{phys}] {torch.cuda.get_device_name(device)} total={props.total_memory / 1024**3:.2f} GB')
    slots.append(
        {
            'idx': idx,
            'phys': phys,
            'device': device,
            'occupied': False,
            'chunks': [],
            'a': None,
            'b': None,
        }
    )


def query_gpu_stats():
    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
        )
    except Exception:
        return {}
    stats = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[0])
            mem_used = int(parts[1])
            util = int(parts[2])
            stats[idx] = (mem_used, util)
        except Exception:
            continue
    return stats


def is_slot_free(slot, gpu_stats):
    info = gpu_stats.get(slot['phys'])
    if info is not None:
        mem_used, util = info
        return mem_used <= free_mem_used_mb and util <= free_util_pct

    # fallback when nvidia-smi unavailable
    torch.cuda.set_device(slot['device'])
    free_b, total_b = torch.cuda.mem_get_info(slot['device'])
    used_mb = (total_b - free_b) / 1024**2
    free_ratio = free_b / max(total_b, 1)
    return used_mb <= free_mem_used_mb and free_ratio >= 0.9


def release_slot(slot):
    slot['chunks'].clear()
    slot['a'] = None
    slot['b'] = None
    slot['occupied'] = False
    torch.cuda.set_device(slot['device'])
    torch.cuda.empty_cache()


def occupy_slot(slot):
    device = slot['device']
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    total_bytes = props.total_memory
    chunks = []

    if target_gb > 0:
        target_bytes = int(target_gb * 1024**3)
    else:
        target_bytes = max(0, total_bytes - safety_bytes)

    allocated = 0
    while allocated < target_bytes:
        try:
            t = torch.empty((chunk_elems,), dtype=torch.float16, device=device)
            chunks.append(t)
            allocated += chunk_bytes
        except RuntimeError:
            break

    n = 4096
    a = torch.randn((n, n), device=device, dtype=torch.float16)
    b = torch.randn((n, n), device=device, dtype=torch.float16)
    slot['chunks'] = chunks
    slot['a'] = a
    slot['b'] = b
    slot['occupied'] = True
    print(f'[cuda:{slot[\"idx\"]} -> gpu:{slot[\"phys\"]}] occupied, allocated={allocated / 1024**3:.2f} GB')


while True:
    stats = query_gpu_stats()

    for slot in slots:
        if not slot['occupied']:
            if is_slot_free(slot, stats):
                try:
                    occupy_slot(slot)
                except RuntimeError as e:
                    release_slot(slot)
                    print(f'[cuda:{slot[\"idx\"]} -> gpu:{slot[\"phys\"]}] occupy failed: {type(e).__name__}')
            continue

        try:
            torch.cuda.set_device(slot['device'])
            c = slot['a'] @ slot['b']
            c = torch.relu(c)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[cuda:{slot[\"idx\"]} -> gpu:{slot[\"phys\"]}] runtime error, releasing: {type(e).__name__}')
            release_slot(slot)

    time.sleep(poll_sec)
PY"

    echo "已启动: screen -S ${SESSION_NAME} (GPU ${gpu_list})"
    echo "查看: screen -r ${SESSION_NAME}"
}

stop_session() {
    require_cmd screen
    if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
        screen -S "${SESSION_NAME}" -X quit
        echo "已停止: ${SESSION_NAME}"
    else
        echo "未找到会话: ${SESSION_NAME}"
    fi
}

status_session() {
    require_cmd screen
    screen -list | grep "${SESSION_NAME}" || true
}

attach_session() {
    require_cmd screen
    screen -r "${SESSION_NAME}"
}

cmd="${1:-}"
case "${cmd}" in
    start) start_session ;;
    stop) stop_session ;;
    status) status_session ;;
    attach) attach_session ;;
    *) usage; exit 1 ;;
esac
