"""
Session 3-B 决策点确认实验
决策点①：stim 范围有效性
决策点②：STEP_DURATION_MS=2000 vs 5000 的速度与质量对比
"""

import sys
import time
import numpy as np
sys.path.insert(0, ".")

from models.s6_rl_env import SleepEnv


# -----------------------------------------------------------------------
# 实验①：stim 范围有效性
# -----------------------------------------------------------------------

def experiment_stim_range():
    print("=" * 60)
    print("实验①：stim 范围有效性")
    print("=" * 60)
    print("方法：固定零扰动初始状态，对每个 stim 值重复 5 次取均值")
    print("      每次独立仿真 5000ms（与 SleepEnv.step 一致）\n")

    import json
    with open("data/patient_params_SC4001.json") as f:
        pp = json.load(f)

    env = SleepEnv()

    # 先热身到稳定 up-state（零扰动）
    env._base_mue = float(pp["mue"])
    env._base_mui = float(pp["mui"])
    env._apply_patient_params(mue=env._base_mue, mui=env._base_mui)

    # 确保初始在 up-state
    for _ in range(5):
        env._run_sim(5000)
        obs_init = env._get_obs()
        if (float(obs_init[2]) + 1.0) * 30.0 > 1.0:
            break
    baseline_delta = float(obs_init[0]) * 0.5 + 0.5
    print(f"基线 delta_ratio（stim=0）= {baseline_delta:.4f}\n")

    # 测试不同 stim 值
    stim_values = [-0.20, -0.10, -0.05, 0.00, +0.05, +0.10, +0.20]
    N_REPEAT = 5

    print(f"{'stim':>8}  {'delta均值':>10}  {'delta标准差':>12}  {'vs基线':>10}  {'结论':>20}")
    print("-" * 70)

    results = {}
    for stim in stim_values:
        deltas = []
        for _ in range(N_REPEAT):
            # 每次都从相同基础参数开始（独立仿真）
            current_mue = env._base_mue + stim
            from models.s6_rl_env import set_params_glob
            set_params_glob(env.model, "*ALNMassEXC*.input_0.mu", current_mue)
            env._run_sim(5000)
            obs = env._get_obs()
            deltas.append(float(obs[0]) * 0.5 + 0.5)

        mean_d = np.mean(deltas)
        std_d  = np.std(deltas)
        diff   = mean_d - baseline_delta
        noise_snr = abs(diff) / (std_d + 1e-6)

        # 判断信号是否超过噪声
        if abs(diff) >= 0.05:
            verdict = "✓ 有效信号"
        elif abs(diff) >= 0.02:
            verdict = "△ 弱信号"
        else:
            verdict = "✗ 淹没在噪声里"

        results[stim] = {"mean": mean_d, "std": std_d, "diff": diff}
        print(f"{stim:>+8.3f}  {mean_d:>10.4f}  {std_d:>12.4f}  "
              f"{diff:>+10.4f}  {verdict:>20}")

    # 判定
    print(f"\n--- 决策点① 判定 ---")
    diff_pos = results[+0.05]["diff"]
    diff_neg = results[-0.05]["diff"]
    max_effect_005 = max(abs(diff_pos), abs(diff_neg))

    if max_effect_005 >= 0.05:
        print(f"✓ ±0.05 有效：最大 delta 变化 = {max_effect_005:.4f} ≥ 0.05")
        print(f"  → Session 3-B 保持 action_space = ±0.05")
    elif max_effect_005 >= 0.02:
        print(f"△ ±0.05 信号弱：最大 delta 变化 = {max_effect_005:.4f}（0.02–0.05）")
        diff_02 = max(abs(results[+0.20]["diff"]), abs(results[-0.20]["diff"]))
        print(f"  ±0.20 的效果 = {diff_02:.4f}")
        print(f"  → Session 3-B 建议扩大 action_space 到 ±0.20")
    else:
        print(f"✗ ±0.05 无效：最大 delta 变化 = {max_effect_005:.4f} < 0.02（淹没在噪声）")
        print(f"  → 必须扩大 action_space，建议 ±0.20 或更大")

    return results


# -----------------------------------------------------------------------
# 实验②：STEP_DURATION_MS 速度与质量对比
# -----------------------------------------------------------------------

def experiment_step_duration():
    print("\n" + "=" * 60)
    print("实验②：STEP_DURATION_MS=2000 vs 5000 对比")
    print("=" * 60)
    print("方法：对每个时长各跑 5 步，记录耗时和 delta_ratio\n")

    import json
    with open("data/patient_params_SC4001.json") as f:
        pp = json.load(f)

    results = {}

    for duration in [2000, 5000]:
        print(f"--- duration = {duration} ms ---")
        env = SleepEnv()

        # 热身到 up-state
        env._base_mue = float(pp["mue"])
        env._base_mui = float(pp["mui"])
        env._apply_patient_params(mue=env._base_mue, mui=env._base_mui)
        for _ in range(5):
            env._run_sim(5000)
            obs_w = env._get_obs()
            if (float(obs_w[2]) + 1.0) * 30.0 > 1.0:
                break

        # 测量 5 步
        N_STEPS = 5
        step_times  = []
        step_deltas = []

        for i in range(N_STEPS):
            t0 = time.time()
            env.model.params["duration"] = duration
            env.model.run()
            elapsed = time.time() - t0

            obs = env._get_obs()
            dr  = float(obs[0]) * 0.5 + 0.5
            mr  = (float(obs[2]) + 1.0) * 30.0

            step_times.append(elapsed)
            step_deltas.append(dr)
            print(f"  Step {i+1}: {elapsed:.2f}s  "
                  f"delta_ratio={dr:.4f}  mean_rate={mr:.2f}Hz")

        mean_time  = np.mean(step_times)
        mean_delta = np.mean(step_deltas)
        std_delta  = np.std(step_deltas)
        results[duration] = {
            "mean_time":  mean_time,
            "mean_delta": mean_delta,
            "std_delta":  std_delta,
        }
        print(f"  均值: {mean_time:.2f}s/步  "
              f"delta={mean_delta:.4f}±{std_delta:.4f}\n")

    # 判定
    print("--- 决策点② 判定 ---")
    t2000 = results[2000]["mean_time"]
    t5000 = results[5000]["mean_time"]
    d2000 = results[2000]["mean_delta"]
    d5000 = results[5000]["mean_delta"]
    speedup = t5000 / t2000

    print(f"  2000ms: {t2000:.2f}s/步  delta={d2000:.4f}")
    print(f"  5000ms: {t5000:.2f}s/步  delta={d5000:.4f}")
    print(f"  加速比: {speedup:.1f}x")

    if speedup >= 1.5 and abs(d2000 - d5000) < 0.10:
        print(f"✓ 推荐用 2000ms：速度提升 {speedup:.1f}x，delta 差异仅 {abs(d2000-d5000):.4f}")
        print(f"  → Session 3-B 设 STEP_DURATION_MS=2000")
    elif speedup >= 1.5 and abs(d2000 - d5000) >= 0.10:
        print(f"△ 2000ms 更快（{speedup:.1f}x），但 delta 差异较大（{abs(d2000-d5000):.4f}）")
        print(f"  → 需权衡：快速实验用 2000ms，正式训练用 5000ms")
    else:
        print(f"△ 加速不明显（{speedup:.1f}x），建议保持 5000ms")

    # 训练时间估算
    print(f"\n--- 训练时间估算（100k 步）---")
    for dur, r in results.items():
        hours = r["mean_time"] * 100000 / 3600
        print(f"  {dur}ms/步: {r['mean_time']:.2f}s × 100k = {hours:.1f} 小时")

    return results


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    t_total = time.time()

    r1 = experiment_stim_range()
    r2 = experiment_step_duration()

    print("\n" + "=" * 60)
    print("两个决策点确认完成")
    print(f"总耗时：{time.time()-t_total:.1f}s")
    print("把完整输出贴回给 Yukai，等待 Session 3-B 设计决策。")
    print("=" * 60)
