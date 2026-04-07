"""
Session 3-B: SAC 强化学习训练
使用 stable-baselines3 SAC 训练 SleepEnv agent。

目标：agent 学习通过施加皮层刺激电流维持 N3 睡眠状态
      （防止系统跌入 down-state，delta_ratio 维持在 ≈0.91）

训练规模：100k 步，约 4.4 小时
日志：TensorBoard + CSV（同时输出）
"""

import os
import sys
import time
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from models.s6_rl_env import SleepEnv

# -----------------------------------------------------------------------
# 超参数（不要修改）
# -----------------------------------------------------------------------

TOTAL_TIMESTEPS  = 100_000
LOG_INTERVAL     = 500       # 每 500 步记录一次 CSV
EVAL_INTERVAL    = 5_000     # 每 5000 步评估一次（不扰动参数的纯贪心策略）
EVAL_EPISODES    = 3         # 每次评估运行 3 个 episode
MODEL_SAVE_PATH  = "outputs/sac_sleep_model"
TENSORBOARD_DIR  = "outputs/tb_logs"
CSV_LOG_PATH     = "outputs/sac_training_log.csv"

# SAC 超参数（stable-baselines3 默认值对连续控制通常有效）
SAC_KWARGS = dict(
    learning_rate   = 3e-4,
    buffer_size     = 10_000,
    learning_starts = 500,     # 前 500 步随机探索，之后开始学习
    batch_size      = 64,
    tau             = 0.005,
    gamma           = 0.99,
    train_freq      = 1,
    gradient_steps  = 1,
    verbose         = 1,
    tensorboard_log = TENSORBOARD_DIR,
)


# -----------------------------------------------------------------------
# 自定义 Callback：CSV 日志 + 定期评估 + 定期保存
# -----------------------------------------------------------------------

class SleepTrainingCallback(BaseCallback):
    """
    每 LOG_INTERVAL 步记录一次训练指标到 CSV。
    每 EVAL_INTERVAL 步运行贪心评估，记录评估 delta_ratio。
    每 EVAL_INTERVAL 步保存一次模型 checkpoint。
    """

    def __init__(self, eval_env: SleepEnv, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env    = eval_env
        self.csv_records = []
        self.t_start     = time.time()

    def _on_step(self) -> bool:
        # --- CSV 日志（每 LOG_INTERVAL 步）---
        if self.num_timesteps % LOG_INTERVAL == 0:
            # 从 rollout buffer 取最近的 reward 均值
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                mean_reward    = float(np.mean(recent_rewards))
                mean_ep_len    = float(np.mean(
                    [ep["l"] for ep in self.model.ep_info_buffer]
                ))
            else:
                mean_reward = float("nan")
                mean_ep_len = float("nan")

            elapsed = time.time() - self.t_start
            self.csv_records.append({
                "timestep":       self.num_timesteps,
                "mean_reward":    mean_reward,
                "mean_ep_length": mean_ep_len,
                "elapsed_s":      elapsed,
            })

            if self.verbose >= 1:
                print(f"  [CSV] step={self.num_timesteps:>7d}  "
                      f"mean_reward={mean_reward:.4f}  "
                      f"elapsed={elapsed:.0f}s")

        # --- 评估 + 模型保存（每 EVAL_INTERVAL 步）---
        if self.num_timesteps % EVAL_INTERVAL == 0 and self.num_timesteps > 0:
            eval_rewards  = []
            eval_deltas   = []

            for _ in range(EVAL_EPISODES):
                obs, _ = self.eval_env.reset()
                ep_reward = 0.0
                ep_deltas = []
                done = False

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = \
                        self.eval_env.step(action)
                    ep_reward += reward
                    ep_deltas.append(info["delta_ratio"])
                    done = terminated or truncated

                eval_rewards.append(ep_reward)
                eval_deltas.append(float(np.mean(ep_deltas)))

            mean_eval_reward = float(np.mean(eval_rewards))
            mean_eval_delta  = float(np.mean(eval_deltas))

            # 写入 CSV
            self.csv_records.append({
                "timestep":         self.num_timesteps,
                "eval_reward":      mean_eval_reward,
                "eval_delta_ratio": mean_eval_delta,
                "elapsed_s":        time.time() - self.t_start,
            })

            print(f"\n  [EVAL] step={self.num_timesteps:>7d}  "
                  f"eval_reward={mean_eval_reward:.4f}  "
                  f"eval_delta_ratio={mean_eval_delta:.4f}\n")

            # 保存 checkpoint
            ckpt_path = f"{MODEL_SAVE_PATH}_step{self.num_timesteps}"
            self.model.save(ckpt_path)
            print(f"  [SAVE] checkpoint → {ckpt_path}.zip")

        return True   # 返回 False 会提前终止训练

    def on_training_end(self):
        """训练结束时把所有 CSV 记录写入文件。"""
        if self.csv_records:
            df = pd.DataFrame(self.csv_records)
            df.to_csv(CSV_LOG_PATH, index=False)
            print(f"\n[CSV] 训练日志已保存：{CSV_LOG_PATH}")
            print(f"      共 {len(df)} 条记录")


# -----------------------------------------------------------------------
# 主训练流程
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Session 3-B — SAC 训练")
    print("=" * 60)
    print(f"  总步数     : {TOTAL_TIMESTEPS:,}")
    print(f"  预计耗时   : {TOTAL_TIMESTEPS * 0.16 / 3600:.1f} 小时")
    print(f"  TensorBoard: {TENSORBOARD_DIR}")
    print(f"  CSV 日志   : {CSV_LOG_PATH}")
    print(f"  模型保存   : {MODEL_SAVE_PATH}_stepXXXXX.zip")
    print()

    os.makedirs("outputs", exist_ok=True)

    # --- 训练环境 ---
    print("[1/4] 初始化训练环境...")
    t0 = time.time()
    train_env = SleepEnv()
    print(f"      完成，耗时 {time.time()-t0:.1f}s")

    # --- 评估环境（独立实例，避免状态干扰训练环境）---
    print("[2/4] 初始化评估环境...")
    t0 = time.time()
    eval_env = SleepEnv()
    print(f"      完成，耗时 {time.time()-t0:.1f}s")

    # --- SAC 模型 ---
    print("[3/4] 初始化 SAC 模型...")
    model = SAC("MlpPolicy", train_env, **SAC_KWARGS)

    # 配置 TensorBoard + CSV 双路日志
    new_logger = configure(TENSORBOARD_DIR, ["stdout", "tensorboard", "csv"])
    model.set_logger(new_logger)
    print(f"      SAC 参数：lr={SAC_KWARGS['learning_rate']}, "
          f"buffer={SAC_KWARGS['buffer_size']}, "
          f"batch={SAC_KWARGS['batch_size']}")

    # --- 训练 ---
    print(f"\n[4/4] 开始训练（{TOTAL_TIMESTEPS:,} 步）...")
    print(f"      TensorBoard 实时监控：")
    print(f"        tensorboard --logdir {TENSORBOARD_DIR}")
    print()

    callback = SleepTrainingCallback(eval_env=eval_env, verbose=1)
    t_train = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        log_interval=10,    # sb3 内置日志每 10 个 episode 打印一次
        reset_num_timesteps=True,
        progress_bar=False,  # 避免进度条干扰日志输出
    )

    elapsed = time.time() - t_train
    print(f"\n训练完成，实际耗时：{elapsed/3600:.2f} 小时 ({elapsed:.0f}s)")

    # --- 保存最终模型 ---
    model.save(MODEL_SAVE_PATH)
    print(f"最终模型已保存：{MODEL_SAVE_PATH}.zip")

    # --- 最终评估 ---
    print("\n=== 最终评估（10 episodes，贪心策略）===")
    final_rewards = []
    final_deltas  = []

    for ep in range(10):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        ep_deltas = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            ep_deltas.append(info["delta_ratio"])
            done = terminated or truncated

        final_rewards.append(ep_reward)
        final_deltas.append(float(np.mean(ep_deltas)))
        print(f"  Episode {ep+1:>2d}: reward={ep_reward:.4f}  "
              f"mean_delta={final_deltas[-1]:.4f}")

    print(f"\n最终评估均值：")
    print(f"  mean reward      = {np.mean(final_rewards):.4f} ± {np.std(final_rewards):.4f}")
    print(f"  mean delta_ratio = {np.mean(final_deltas):.4f} ± {np.std(final_deltas):.4f}")
    print(f"  目标 delta_ratio = 0.91")
    print(f"  与目标差距       = {abs(np.mean(final_deltas) - 0.91):.4f}")

    # 成功判定
    mean_delta_final = np.mean(final_deltas)
    if mean_delta_final >= 0.85:
        print(f"\n✓ Session 3-B 训练成功：mean_delta={mean_delta_final:.4f} ≥ 0.85")
    elif mean_delta_final >= 0.75:
        print(f"\n△ 部分成功：mean_delta={mean_delta_final:.4f}（0.75–0.85），"
              f"可调参后继续训练")
    else:
        print(f"\n✗ 训练未收敛：mean_delta={mean_delta_final:.4f} < 0.75，"
              f"需检查 reward 函数或超参数")

    print("=" * 60)


if __name__ == "__main__":
    main()
