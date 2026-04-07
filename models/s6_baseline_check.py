"""
零扰动基线验证：确认 SleepEnv 在原始 patient_params 下
能否重现 Session 2-B 拟合的 N3 特征（delta_ratio ≈ 0.88）。

判定标准：
  delta_ratio >= 0.80  → 解释 A，SleepEnv 基线正常，可进入 Session 3-B
  delta_ratio  < 0.80  → 解释 B，SleepEnv 仿真未重现 N3，需修复再进 3-B
"""

import sys
import time
sys.path.insert(0, ".")   # 确保从项目根目录可以 import

from models.s6_rl_env import SleepEnv


def run_baseline_check():
    print("=" * 60)
    print("零扰动基线验证")
    print("=" * 60)

    # --- 加载参数 ---
    import json
    with open("data/patient_params_SC4001.json") as f:
        pp = json.load(f)
    print(f"\n病人参数：mue={pp['mue']}, mui={pp['mui']}, "
          f"g_LK={pp['g_LK']}, g_h={pp['g_h']}")
    print(f"Session 2-B 拟合质量：psd_corr={pp.get('psd_corr', 'N/A')}")
    print(f"期望 delta_ratio：≥ 0.80（Session 2-B N3 delta 占比约 0.88）\n")

    # --- 初始化环境 ---
    t0 = time.time()
    env = SleepEnv()

    # --- 零扰动：直接用原始 patient_params，跳过 reset() 的随机扰动 ---
    env._base_mue = float(pp["mue"])
    env._base_mui = float(pp["mui"])
    env._apply_patient_params(mue=env._base_mue, mui=env._base_mui)

    # --- 热身仿真（与 reset() 一致，5 秒） ---
    print("运行热身仿真（5000 ms）...")
    env._run_sim(5000)
    print(f"完成，耗时 {time.time()-t0:.1f}s")

    # --- 计算观测 ---
    obs = env._get_obs()
    delta_ratio = float(obs[0]) * 0.5 + 0.5
    sigma_ratio = max(0.0, (1.0 - float(obs[1])) / 20.0)
    mean_rate_raw = (float(obs[2]) + 1.0) * 30.0   # 反归一化

    print(f"\n--- 基线观测结果 ---")
    print(f"  delta_ratio (0.5–4 Hz)  = {delta_ratio:.4f}  （目标 ≥ 0.80）")
    print(f"  sigma_ratio (10–15 Hz)  = {sigma_ratio:.4f}  （目标 ≤ 0.05）")
    print(f"  mean_rate               = {mean_rate_raw:.2f} Hz")
    print(f"  obs (归一化)            = {obs}")

    # --- 扩展诊断：不同仿真时长（确认是否需要更长热身） ---
    print(f"\n--- 扩展诊断：仿真时长对 delta_ratio 的影响 ---")
    # 注意：MVP 限制下每次 run() 都从初始条件重启，三次是独立测试，不是接续仿真
    # _apply_patient_params 在这里是预防性重设，不影响结果
    for duration in [5000, 10000, 20000]:
        env._apply_patient_params(mue=env._base_mue, mui=env._base_mui)
        env._run_sim(duration)
        obs_d = env._get_obs()
        dr = float(obs_d[0]) * 0.5 + 0.5
        print(f"  duration={duration:>6} ms  →  delta_ratio={dr:.4f}")

    # --- 对比 Session 2-B 目标 PSD ---
    print(f"\n--- 与 Session 2-B 目标对比 ---")
    try:
        import numpy as np
        target_psd  = np.load("data/target_psd_SC4001.npy")
        target_freq = np.load("data/target_freqs.npy")
        total_mask   = (target_freq >= 0.5) & (target_freq <= 30.0)
        f_total      = target_freq[total_mask]        # 0.5–30 Hz 的频率轴
        p_rel        = target_psd[total_mask] / (target_psd[total_mask].sum() + 1e-30)
        delta_mask_t = f_total <= 4.0                  # delta 区域（0.5–4 Hz）
        target_delta = p_rel[delta_mask_t].sum()
        print(f"  Session 2-B 目标 delta_ratio = {target_delta:.4f}")
        print(f"  当前仿真 delta_ratio          = {delta_ratio:.4f}")
        print(f"  差值                          = {abs(delta_ratio - target_delta):.4f}")
    except FileNotFoundError:
        print("  （target_psd_SC4001.npy 未找到，跳过对比）")

    # --- 最终判定 ---
    print(f"\n{'='*60}")
    if delta_ratio >= 0.80:
        print(f"✓ 判定：解释 A — 基线 delta_ratio={delta_ratio:.4f} ≥ 0.80")
        print(f"  SleepEnv 仿真正常重现 N3 特征，reset() 的扰动是偏低的原因")
        print(f"  可以进入 Session 3-B")
    elif delta_ratio >= 0.60:
        print(f"△ 判定：中间状态 — delta_ratio={delta_ratio:.4f}，在 0.60–0.80 之间")
        print(f"  仿真有 N3 趋势但未达 Session 2-B 水平，建议检查热身时长")
        print(f"  参考：把 WARMUP_MS 从 5000 增加到 20000 后重跑")
    else:
        print(f"✗ 判定：解释 B — 基线 delta_ratio={delta_ratio:.4f} < 0.60")
        print(f"  SleepEnv 未重现 N3，不能直接进入 Session 3-B")
        print(f"  需要检查：MultiModel 包装后参数是否正确传入（见调试规则）")
    print("=" * 60)

    return delta_ratio


if __name__ == "__main__":
    run_baseline_check()
