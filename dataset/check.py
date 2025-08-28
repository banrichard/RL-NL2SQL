import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from types import SimpleNamespace
import os
# --- 關鍵：從您環境的 OpenRLHF 原始碼中引入真實的類別 ---
# 執行此腳本前，請確保您位於專案的根目錄下，且已啟用虛擬環境
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.datasets import SFTDataset
from datasets import load_dataset


# --- 1. 參數設定 (請確保與您的 .sh 腳本一致) ---
# 這些參數將用來模擬 strategy.args
class MockArgs:
    # 資料相關
    dataset = "/home/banlujie/data/bird_train/train/train_sft.json"
    input_key = "prompt"
    output_key = "SQL"
    pretrain = "microsoft/Phi-4-mini-reasoning" # 用於載入 Tokenizer
    max_len = 8192
    
    # DeepspeedStrategy 需要的參數
    # 我們不實際訓練，所以這裡的值大部分不重要，但需要存在
    local_rank = -1 # 在單程序測試中設為 -1
    zero_stage = 2
    bf16 = True
    adam_offload = False
    zpg = 1
    grad_accum_dtype = None
    overlap_comm = False
    use_ds_universal_ckpt = False
    deepcompile = False
    ds_tensor_parallel_size = 1 # 設為 1，因為我們不測試張量並行
    ring_attn_size = 1 # 設為 1
    micro_train_batch_size = 4
    # SFTDataset 需要的參數
    # 確保這些與您的 .sh 腳本設定一致！
    apply_chat_template = True
    packing_samples = True

# --- 2. 主測試函式 ---
def main():
    print("="*50)
    print("正在使用真實的 OpenRLHF 元件進行 DataLoader 除錯...")
    print("="*50)

    # --- 模擬環境 ---
    args = MockArgs()
    strategy = DeepspeedStrategy(args=args, zero_stage=args.zero_stage)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)

    # --- 建立 SFT Dataset ---
    # 這是 OpenRLHF 的標準流程
    raw_dataset = load_dataset("json", data_files=args.dataset, split="train")
    sft_dataset = SFTDataset(
        raw_dataset,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=False
    )
    print(f"SFTDataset 實例化完成，樣本數: {len(sft_dataset)}")

    # --- 建立 DataLoader (我們要測試的核心目標) ---
    # 我們在這裡明確設定 drop_last=False，以強制復現邊界問題
    print("\n正在呼叫 strategy.setup_dataloader (drop_last=False)...")
    train_dataloader = strategy.setup_dataloader(
        sft_dataset,
        args.micro_train_batch_size, # 假設您想測試這個
        collate_fn=sft_dataset.collate_fn,
        drop_last=False  # <--- 明確設定為 False 來觸發邊界問題
    )
    print("DataLoader 建立完畢。")

    # --- 遍歷並檢查每個批次 ---
    print("\n開始遍歷所有批次，檢查是否存在無效批次...")
    found_problem = False
    # DataLoader 的長度由 sampler 決定
    total_batches = len(train_dataloader.sampler) // args.micro_train_batch_size
    
    for i, batch in enumerate(tqdm(train_dataloader, total=total_batches, desc="檢查批次")):
        if batch is None:
            print(f"\n❌ 發現問題！第 {i} 個批次由 collate_fn 回傳為 None！")
            found_problem = True
            break
        
        # OpenRLHF 的 DataLoader 回傳一個元組
        input_ids, attention_mask, loss_mask = batch
        
        if input_ids.numel() == 0:
            print(f"\n❌ 發現問題！第 {i} 個批次的 input_ids 為空！ Shape: {input_ids.shape}")
            found_problem = True
            break
        
        if attention_mask is None or not attention_mask.any():
            print(f"\n❌ 發現問題！第 {i} 個批次的 attention_mask 無效 (None 或全為零)！")
            if attention_mask is not None:
                print(f"   Shape: {attention_mask.shape}")
            found_problem = True
            break
            
    print("\n" + "="*50)
    if not found_problem:
        print("✅ 檢查完成，所有批次均有效！")
        print("這表示問題只在與 DeepSpeed 完全整合的環境下才出現。")
        print("即便如此，為 DataLoader 設定 drop_last=True 依然是解決這類問題最穩健的方法。")
    else:
        print("🎉 問題已成功復現！根源就在 DataLoader 對邊界樣本的處理上。")
        print("➡️ 最終解決方案：請修改 `openrlhf/cli/train_sft.py`，在建立 DataLoader 時手動加上 `drop_last=True`。")
    print("="*50)

if __name__ == "__main__":
    # 由於 DeepspeedStrategy 內部可能依賴 distributed，我們需要一個 mock 的環境
    # 我們可以手動設定環境變數來模擬單程序運行的情況
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500' # 隨機指定一個未被佔用的埠
    
    main()