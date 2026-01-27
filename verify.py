#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np

def verify_flash_attention_gqa(q_path, k_path, v_path, o_path, is_causal=True, threshold=1e-4):
    # 1. 加载数据 [Batch, Seq, Head, Dim]
    q_np = np.load(q_path)
    k_np = np.load(k_path)
    v_np = np.load(v_path)
    o_expected_np = np.load(o_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 

    # 2. 转换为 Tensor
    t_q = torch.from_numpy(q_np).to(device).to(dtype)
    t_k = torch.from_numpy(k_np).to(device).to(dtype)
    t_v = torch.from_numpy(v_np).to(device).to(dtype)
    t_o_exp = torch.from_numpy(o_expected_np).to(device).to(dtype)

    # 3. 维度转换 [B, S, H, D] -> [B, H, S, D]
    t_q = t_q.transpose(1, 2)
    t_k = t_k.transpose(1, 2)
    t_v = t_v.transpose(1, 2)

    q_heads = t_q.shape[1]
    kv_heads = t_k.shape[1]
    
    if q_heads % kv_heads != 0:
        raise ValueError(f"Query heads ({q_heads}) 必须能被 KV heads ({kv_heads}) 整除")

    # 4. 执行计算
    try:
        # 注意：某些版本的 PyTorch SDPA 不需要 enable_gqa 参数，它会自动处理
        t_o_actual = F.scaled_dot_product_attention(
            t_q, t_k, t_v, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=is_causal,
            enable_gqa=True
        )
    except Exception as e:
        print(f"Flash Attention 调用失败: {e}")
        return

    # 5. 转回原始布局 [B, H, S, D] -> [B, S, H, D]
    t_o_actual = t_o_actual.transpose(1, 2).contiguous()

    # 6. 误差分析
    diff = torch.abs(t_o_actual - t_o_exp)
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    print(f"\n--- 验证结果 (is_causal={is_causal}) ---")
    print(f"最大误差: {max_err:.6e}")
    print(f"平均误差: {mean_err:.6e}")

    if max_err < threshold:
        print("✅ 匹配成功！")
    else:
        print(f"❌ 匹配失败！正在定位误差坐标 (阈值 > {threshold})...")
        
        # --- 新增：定位功能 ---
        # 找出所有超过阈值的索引 [N, 4] -> (Batch, Seq, Head, Dim)
        error_indices = torch.nonzero(diff > threshold)
        
        num_errors = error_indices.size(0)
        print(f"共有 {num_errors} 个位置误差超标。")

        # 打印前 10 个误差最大的位置
        # 我们按误差值大小排序
        flat_diff = diff.view(-1)
        topk_values, topk_indices = torch.topk(flat_diff, min(100, num_errors))
        
        print("\n[前 10 个最大误差坐标详情]:")
        print(f"{'Index (B, S, H, D)':<25} | {'Actual':<10} | {'Expected':<10} | {'Diff':<10}")
        print("-" * 65)
        
        # 这里的 index 需要转换回 4D 坐标
        shape = t_o_actual.shape
        for i in range(len(topk_values)):
            idx = topk_indices[i].item()
            # 这里的坐标对应 [B, S, H, D]
            d = idx % shape[3]
            h = (idx // shape[3]) % shape[2]
            s = (idx // (shape[3] * shape[2])) % shape[1]
            b = (idx // (shape[3] * shape[2] * shape[1]))
            
            act_val = t_o_actual[b, s, h, d].item()
            exp_val = t_o_exp[b, s, h, d].item()
            err_val = topk_values[i].item()
            
            print(f"({b}, {s}, {h}, {d})".ljust(25) + f"| {act_val:<10.4f} | {exp_val:<10.4f} | {err_val:<10.4f}")

        # 统计学分析：误差是否集中在特定的 Seq 或 Head？
        if num_errors > 0:
            bad_seq_indices = error_indices[:, 1]
            print(f"\n误差最集中的 Token Index: {torch.mode(bad_seq_indices).values.item()}")

if __name__ == "__main__":
    import sys 
    index = int(sys.argv[1])
    try_count = int(sys.argv[2])
    x = np.load(f"tmp/{index}.c.{try_count}.npy")

    is_causal = bool(x[0] == 1)
    verify_flash_attention_gqa(
        f"tmp/{index}.q.{try_count}.npy", 
        f"tmp/{index}.k.{try_count}.npy", 
        f"tmp/{index}.v.{try_count}.npy", 
        f"tmp/{index}.o.{try_count}.npy",
        is_causal
    )
