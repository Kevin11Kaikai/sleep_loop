"""
Session 3-B 修复版：autoresearch 式迭代训练
每轮 10k 步，从最优 checkpoint 继续，记录 eval 结果。

迭代历史：
  Round 1：rescue_bonus=0.5，action_penalty=0.5，从 step5000 继续
  Round 2：（根据 Round 1 结果填写）
  Round 3：（根据 Round 2 结果填写）
"""

import os
import sys
import time
import json
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from models.s6_rl_env import SleepEnv

# -----------------------------------------------------------------------
# 当前轮次配置（每轮只改这里）
# -----------------------------------------------------------------------

ROUND_ID         = 5                                    # 当前轮次编号
CHECKPOINT_IN    = "outputs/sac_round3_best"            # 从哪个 checkpoint 继续（Round 3 最优）
ROUND_TIMESTEPS  = 20_000                               # 每轮步数（验证轮）
EVAL_EPISODES    = 5                                    # 每次评估 episode 数
EVAL_INTERVAL    = 2_000                                # 每 2000 步评估一次

MODEL_SAVE_PATH  = f"outputs/sac_round{ROUND_ID}"
TENSORBOARD_DIR  = f"outputs/tb_logs_round{ROUND_ID}"
CSV_LOG_PATH     = f"outputs/sac_round{ROUND_ID}_log.csv"
HISTORY_PATH     = "outputs/iteration_history.json"

# -----------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------

class IterationCallback(BaseCallback):

    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env    = eval_env
        self.csv_records = []
        self.t_start     = time.time()
        self.best_delta  = 0.0
        self.best_step   = 0

    def _on_step(self):
        if self.num_timesteps % EVAL_INTERVAL == 0 and self.num_timesteps > 0:
            rewards, deltas = [], []

            for _ in range(EVAL_EPISODES):
                obs, _ = self.eval_env.reset()
                ep_reward, ep_deltas, done = 0.0, [], False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = \
                        self.eval_env.step(action)
                    ep_reward += reward
                    ep_deltas.append(info["delta_ratio"])
                    done = terminated or truncated
                rewards.append(ep_reward)
                deltas.append(float(np.mean(ep_deltas)))

            mean_reward = float(np.mean(rewards))
            mean_delta  = float(np.mean(deltas))
            elapsed     = time.time() - self.t_start

            self.csv_records.append({
                "round":      ROUND_ID,
                "timestep":   self.num_timesteps,
                "eval_reward":      mean_reward,
                "eval_delta_ratio": mean_delta,
                "elapsed_s":  elapsed,
            })

            # 追踪本轮最优
            if mean_delta > self.best_delta:
                self.best_delta = mean_delta
                self.best_step  = self.num_timesteps
                best_path = f"{MODEL_SAVE_PATH}_best"
                self.model.save(best_path)

            print(f"  [Round {ROUND_ID}] step={self.num_timesteps:>6d}  "
                  f"eval_delta={mean_delta:.4f}  "
                  f"eval_reward={mean_reward:.4f}  "
                  f"elapsed={elapsed:.0f}s")

        return True

    def on_training_end(self):
        if self.csv_records:
            df = pd.DataFrame(self.csv_records)
            df.to_csv(CSV_LOG_PATH, index=False)
            print(f"\n[Round {ROUND_ID}] CSV 已保存：{CSV_LOG_PATH}")

# -----------------------------------------------------------------------
# 判定函数
# -----------------------------------------------------------------------

def judge_round(eval_deltas):
    """
    根据本轮所有 eval 点判断结果，决定下一轮策略。
    返回：verdict（✓/△/✗），recommendation（建议操作）
    """
    if not eval_deltas:
        return "✗", "无评估数据，检查环境"

    best  = max(eval_deltas)
    final = eval_deltas[-1]
    trend = eval_deltas[-1] - eval_deltas[0]   # 正 = 改善

    if best >= 0.80:
        verdict = "PASS"
        rec = "Round effective. Next: raise RESCUE_BONUS from 0.5 to 1.0"
    elif best >= 0.72 and trend > 0:
        verdict = "PARTIAL"
        rec = "Improving but below target. Next: reduce action penalty coeff 0.5->0.1"
    elif best >= 0.72 and trend <= 0:
        verdict = "PARTIAL"
        rec = "Peak OK but declining. Next: resume from best ckpt, raise RESCUE_BONUS to 1.0"
    else:
        verdict = "FAIL"
        rec = "Rescue bonus ineffective. Next: restart from step5000, loosen action penalty 0.5->0.1"

    return verdict, rec

# -----------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"Session 3-B 迭代训练 — Round {ROUND_ID}")
    print("=" * 60)
    print(f"  从 checkpoint : {CHECKPOINT_IN}.zip")
    print(f"  本轮步数      : {ROUND_TIMESTEPS:,}")
    print(f"  保存至        : {MODEL_SAVE_PATH}")
    print()

    os.makedirs("outputs", exist_ok=True)

    # 环境
    print("[1/4] 初始化训练环境...")
    train_env = SleepEnv()
    print("[2/4] 初始化评估环境...")
    eval_env  = SleepEnv()

    # 加载 checkpoint
    print(f"[3/4] 加载 checkpoint：{CHECKPOINT_IN}.zip ...")
    model = SAC.load(CHECKPOINT_IN, env=train_env)
    new_logger = configure(TENSORBOARD_DIR, ["stdout", "tensorboard", "csv"])
    model.set_logger(new_logger)
    print(f"      SAC 已加载，继续训练 {ROUND_TIMESTEPS:,} 步")

    # 训练
    print(f"\n[4/4] 开始训练...")
    callback = IterationCallback(eval_env=eval_env, verbose=1)
    t0 = time.time()

    model.learn(
        total_timesteps=ROUND_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=False,   # 保留原有步数计数
        log_interval=10,
        progress_bar=False,
    )

    elapsed = time.time() - t0
    print(f"\n本轮训练完成，耗时 {elapsed/60:.1f} 分钟")

    # 保存最终模型
    model.save(MODEL_SAVE_PATH)
    print(f"最终模型已保存：{MODEL_SAVE_PATH}.zip")

    # 最终评估（10 episodes）
    print(f"\n=== Round {ROUND_ID} 最终评估（10 episodes）===")
    final_rewards, final_deltas = [], []
    up_count, down_count = 0, 0

    for ep in range(10):
        obs, _ = eval_env.reset()
        ep_reward, ep_deltas, done = 0.0, [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            ep_deltas.append(info["delta_ratio"])
            done = terminated or truncated
        mean_d = float(np.mean(ep_deltas))
        final_rewards.append(ep_reward)
        final_deltas.append(mean_d)
        state = "up" if mean_d >= 0.75 else "down"
        if state == "up": up_count += 1
        else: down_count += 1
        print(f"  Episode {ep+1:>2d}: reward={ep_reward:.4f}  "
              f"mean_delta={mean_d:.4f}  [{state}]")

    mean_delta_final = float(np.mean(final_deltas))
    mean_reward_final = float(np.mean(final_rewards))

    print(f"\n--- Round {ROUND_ID} 结果汇总 ---")
    print(f"  mean delta_ratio = {mean_delta_final:.4f} ± {np.std(final_deltas):.4f}")
    print(f"  mean reward      = {mean_reward_final:.4f}")
    print(f"  up-state         = {up_count}/10")
    print(f"  down-state       = {down_count}/10")
    print(f"  上轮对比         = 0.684 -> {mean_delta_final:.4f}  "
          f"({'+ improve' if mean_delta_final > 0.684 else '- regress'})")

    # 判定与下轮建议
    eval_deltas_history = [r["eval_delta_ratio"]
                           for r in callback.csv_records
                           if "eval_delta_ratio" in r]
    verdict, recommendation = judge_round(eval_deltas_history)

    print(f"\n{'='*60}")
    print(f"Round {ROUND_ID} 判定：{verdict}")
    print(f"下轮建议：{recommendation}")
    print(f"最优 checkpoint：{MODEL_SAVE_PATH}_best.zip  "
          f"(eval_delta={callback.best_delta:.4f} @ step {callback.best_step})")
    print(f"{'='*60}")

    # 保存迭代历史
    history_entry = {
        "round":              ROUND_ID,
        "checkpoint_in":      CHECKPOINT_IN,
        "mean_delta_final":   mean_delta_final,
        "mean_reward_final":  mean_reward_final,
        "up_count":           up_count,
        "down_count":         down_count,
        "best_delta":         callback.best_delta,
        "best_step":          callback.best_step,
        "verdict":            verdict,
        "recommendation":     recommendation,
        "elapsed_min":        elapsed / 60,
    }

    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)
    history.append(history_entry)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)
    print(f"\n迭代历史已追加：{HISTORY_PATH}")


if __name__ == "__main__":
    main()
