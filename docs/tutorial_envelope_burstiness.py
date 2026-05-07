#!/usr/bin/env python3
"""
脚本：理解 Butter Filter、sosfiltfilt、Hilbert 变换与 Spindle Envelope Burstiness

目标：通过具体的数值演示，逐步说明 T7 约束中 spindle envelope burstiness 的原理。
- butter: 设计带通滤波器
- sosfiltfilt: 零相位双向滤波
- hilbert: 提取解析信号与包络
- 包络变异系数 (CV): 衡量成簇特性

基础概念：Fourier Transform & 频域滤波
- 任何实信号可分解为不同频率的复指数和（Fourier 级数/变换）
- 带通滤波器在频域中"选择"特定频段（10-14 Hz），衰减其他频段
- 工作原理都可通过频域分析理解

依赖：numpy, scipy, matplotlib
在 neurolib 环境中可直接运行，或在任何包含 scipy 的环境中运行。

用法：
    python tutorial_envelope_burstiness.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# 参数设置
# ============================================================================
FS = 1000.0                  # 采样率 [Hz]
DURATION_S = 5.0             # 信号长度 [s]
SPINDLE_LO = 10.0            # Spindle 带通滤波器下限 [Hz]
SPINDLE_HI = 14.0            # Spindle 带通滤波器上限 [Hz]
SPINDLE_CV_MIN = 0.7         # T7 约束：最小包络 CV
SPINDLE_PEAK_FREQ = 13.0     # 示例信号中的 spindle 峰值频率 [Hz]

# ============================================================================
# 第 1 部分：构建示例信号
# ============================================================================
print("=" * 70)
print("第 1 部分：构建带有具体数值的示例信号")
print("=" * 70)

# 时间轴
t = np.arange(0, DURATION_S, 1.0 / FS)
n_samples = len(t)
print(f"时间轴：{DURATION_S} 秒，采样率 {FS} Hz，总样本数 {n_samples}")
print(f"时间点示例：t[0]={t[0]:.4f}s, t[100]={t[100]:.4f}s, t[-1]={t[-1]:.4f}s\n")

# 信号 A：连续的 13 Hz 正弦波 + 低频背景 + 噪声（连续模式）
print("--- 信号 A：连续 spindle 模式（始终存在 13 Hz 振荡）---")
spindle_continuous = 2.0 * np.sin(2 * np.pi * SPINDLE_PEAK_FREQ * t)
# 低频背景（模拟 SO，0.5 Hz）
background = 1.0 * np.sin(2 * np.pi * 0.5 * t)
# 高斯白噪声
noise_continuous = 0.5 * np.random.RandomState(42).randn(n_samples)
# 合成信号 A
signal_A = spindle_continuous + background + noise_continuous

print(f"  - Spindle 分量：振幅 2.0 @ {SPINDLE_PEAK_FREQ} Hz")
print(f"  - 背景分量：振幅 1.0 @ 0.5 Hz")
print(f"  - 噪声分量：σ = 0.5")
print(f"  - 信号 A 时间示例：")
print(f"    t=0.0s: signal_A={signal_A[0]:.4f}")
print(f"    t=1.0s: signal_A={signal_A[1000]:.4f}")
print(f"    t=2.0s: signal_A={signal_A[2000]:.4f}")
print(f"  - 信号 A 统计：mean={signal_A.mean():.4f}, std={signal_A.std():.4f}\n")

# 信号 B：成簇的 13 Hz 振荡（只在 [1.0-3.0] s 和 [3.5-4.5] s 存在）
print("--- 信号 B：成簇 spindle 模式（13 Hz 在特定窗口内出现）---")
spindle_bursty = np.zeros_like(t)
# 第一个 burst：1.0-3.0 s
burst1_mask = (t >= 1.0) & (t < 3.0)
spindle_bursty[burst1_mask] = 2.0 * np.sin(2 * np.pi * SPINDLE_PEAK_FREQ * t[burst1_mask])
# 第二个 burst：3.5-4.5 s
burst2_mask = (t >= 3.5) & (t < 4.5)
spindle_bursty[burst2_mask] = 2.0 * np.sin(2 * np.pi * SPINDLE_PEAK_FREQ * t[burst2_mask])

noise_bursty = 0.5 * np.random.RandomState(42).randn(n_samples)
signal_B = spindle_bursty + background + noise_bursty

print(f"  - Burst 1: [1.0, 3.0] s")
print(f"  - Burst 2: [3.5, 4.5] s")
print(f"  - 非 burst 期间：仅有背景 + 噪声")
print(f"  - 信号 B 时间示例：")
print(f"    t=0.5s (无burst): signal_B={signal_B[500]:.4f}  (仅背景+噪声)")
print(f"    t=2.0s (burst内): signal_B={signal_B[2000]:.4f}  (有13Hz)")
print(f"    t=3.2s (无burst): signal_B={signal_B[3200]:.4f}  (仅背景+噪声)")
print(f"  - 信号 B 统计：mean={signal_B.mean():.4f}, std={signal_B.std():.4f}\n")

# ============================================================================
# 第 2 部分：用 butter 设计 10–14 Hz 带通滤波器
# ============================================================================
print("=" * 70)
print("第 2 部分：设计 10-14 Hz 带通滤波器 (Butterworth 4 阶)")
print("=" * 70)

# 设计 4 阶 Butterworth 带通滤波器
sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype='band', fs=FS, output='sos')

print(f"\nButterworth 滤波器参数：")
print(f"  - 阶数：4")
print(f"  - 通带：[{SPINDLE_LO}, {SPINDLE_HI}] Hz")
print(f"  - 采样率：{FS} Hz")
print(f"  - 输出格式：SOS (second-order sections)")
print(f"\nSOS 系数矩阵形状：{sos.shape}")
print(f"SOS 矩阵（每行是一个二阶段）：")
for i, row in enumerate(sos):
    b0, b1, b2, a0, a1, a2 = row
    print(f"  段 {i}: b=[{b0:.6f}, {b1:.6f}, {b2:.6f}], a=[{a0:.6f}, {a1:.6f}, {a2:.6f}]")
print(f"\n说明：")
print(f"  - 4 阶滤波器由 2 个二阶段构成（2 个二阶 = 1 个四阶）")
print(f"  - 每个二阶段都是一个稳定的 IIR 滤波器")
print(f"  - 级联使用时，确保数值稳定性（相比直接形式）")

# 计算频率响应（用于绘图）
freqs = np.linspace(0, 30, 300)
w = 2 * np.pi * freqs / FS
# SOS 形式的频率响应
H = np.ones_like(w, dtype=complex)
for b0, b1, b2, a0, a1, a2 in sos:
    num = b0 + b1 * np.exp(-1j * w) + b2 * np.exp(-2j * w)
    den = a0 + a1 * np.exp(-1j * w) + a2 * np.exp(-2j * w)
    H *= num / den
mag_db = 20 * np.log10(np.abs(H) + 1e-20)

print(f"\n频率响应关键点：")
print(f"  - 5 Hz 处增益：{mag_db[np.argmin(np.abs(freqs - 5))]:.2f} dB (带外衰减)")
print(f"  - 10 Hz 处增益：{mag_db[np.argmin(np.abs(freqs - 10))]:.2f} dB (通带下限)")
print(f"  - 12 Hz 处增益：{mag_db[np.argmin(np.abs(freqs - 12))]:.2f} dB (通带中心)")
print(f"  - 14 Hz 处增益：{mag_db[np.argmin(np.abs(freqs - 14))]:.2f} dB (通带上限)")
print(f"  - 20 Hz 处增益：{mag_db[np.argmin(np.abs(freqs - 20))]:.2f} dB (带外衰减)\n")

# ============================================================================
# 第 3 部分：用 sosfiltfilt 做零相位滤波
# ============================================================================
print("=" * 70)
print("第 3 部分：用 sosfiltfilt 进行零相位双向滤波")
print("=" * 70)

# 对信号 A 和 B 应用 sosfiltfilt（零相位滤波）
filtered_A = sosfiltfilt(sos, signal_A)
filtered_B = sosfiltfilt(sos, signal_B)

print(f"\nsosfiltfilt 工作原理：")
print(f"  1. 正向滤波：从 t=0 至末尾")
print(f"  2. 反向滤波：翻转后从末尾至 t=0")
print(f"  3. 结果：零相位延迟（不改变信号中各分量的相位关系）")
print(f"\n时域对比（信号 A，连续模式）：")
print(f"  原始信号 (t=0-0.5s)：")
for idx in range(0, 501, 100):
    print(f"    t={t[idx]:.3f}s: raw={signal_A[idx]:+.4f}, filtered={filtered_A[idx]:+.4f}")
print(f"\n时域对比（信号 B，成簇模式）：")
print(f"  t=0.5s (无 burst): raw={signal_B[500]:+.4f} -> filtered={filtered_B[500]:+.4f}")
print(f"  t=2.0s (burst中): raw={signal_B[2000]:+.4f} -> filtered={filtered_B[2000]:+.4f}")
print(f"  t=3.2s (无 burst): raw={signal_B[3200]:+.4f} -> filtered={filtered_B[3200]:+.4f}\n")

print(f"关键观察：")
print(f"  - filtered_A 保留了 13 Hz 分量，衰减了其他高频和低频")
print(f"  - filtered_B 只在 burst 窗口有明显的 13 Hz 振荡")
print(f"  - 滤波器不改变相位，所以振荡形状保持一致\n")

# ============================================================================
# 第 4 部分：用 hilbert 提取解析信号与包络
# ============================================================================
print("=" * 70)
print("第 4 部分：用 Hilbert 变换提取包络 (Envelope)")
print("=" * 70)

# 对筛选后的信号应用 Hilbert 变换
analytic_A = hilbert(filtered_A)
analytic_B = hilbert(filtered_B)

# 提取包络（幅值）
envelope_A = np.abs(analytic_A)
envelope_B = np.abs(analytic_B)

print(f"\nHilbert 变换工作原理：")
print(f"  - 输入：实信号 x(t)")
print(f"  - 输出：解析信号 z(t) = x(t) + j*H[x(t)]")
print(f"    其中 H[x(t)] 是 x(t) 的 Hilbert 变换（虚部）")
print(f"  - 幅值：|z(t)| = sqrt(x(t)^2 + H[x(t)]^2)")
print(f"    这就是瞬时包络 (instantaneous envelope)")
print(f"  - 相位：arg(z(t)) = atan2(H[x(t)], x(t))")

print(f"\n解析信号示例（信号 A，t=1.0-1.1s）：")
for idx in range(1000, 1101, 20):
    real_part = np.real(analytic_A[idx])
    imag_part = np.imag(analytic_A[idx])
    mag = np.abs(analytic_A[idx])
    phase = np.angle(analytic_A[idx])
    print(f"  t={t[idx]:.3f}s: real={real_part:+.4f}, imag={imag_part:+.4f}, |z|={mag:.4f}, arg={phase:+.4f}")

print(f"\n包络统计（信号 A，连续模式）：")
print(f"  envelope_A 的统计量：")
print(f"    mean = {envelope_A.mean():.4f}")
print(f"    std  = {envelope_A.std():.4f}")
print(f"    min  = {envelope_A.min():.4f}")
print(f"    max  = {envelope_A.max():.4f}")

print(f"\n包络统计（信号 B，成簇模式）：")
print(f"  envelope_B 的统计量：")
print(f"    mean = {envelope_B.mean():.4f}")
print(f"    std  = {envelope_B.std():.4f}")
print(f"    min  = {envelope_B.min():.4f}")
print(f"    max  = {envelope_B.max():.4f}\n")

print(f"关键观察：")
print(f"  - 信号 A 的包络相对平整（连续振荡）")
print(f"  - 信号 B 的包络在 burst 窗口内高，窗口外低（成簇振荡）\n")

# ============================================================================
# 第 5 部分：用包络的变异系数解释 Spindle Burstiness
# ============================================================================
print("=" * 70)
print("第 5 部分：包络变异系数 (CV) 与 Spindle Burstiness")
print("=" * 70)

# 计算包络的变异系数（CV）
cv_A = envelope_A.std() / (envelope_A.mean() + 1e-12)
cv_B = envelope_B.std() / (envelope_B.mean() + 1e-12)

print(f"\n变异系数定义：")
print(f"  CV = sigma(envelope) / mu(envelope)")
print(f"  即：包络的标准差 / 包络的均值")
print(f"\nCV 的含义：")
print(f"  - CV 小 (< 0.7): 包络相对平稳，振幅波动小 -> 连续振荡")
print(f"  - CV 大 (> 0.7): 包络波动大，呈明显的高-低起伏 -> 成簇振荡")

print(f"\n--- 信号 A (连续模式) ---")
print(f"  envelope_A 均值: mu = {envelope_A.mean():.4f}")
print(f"  envelope_A 标准差: sigma = {envelope_A.std():.4f}")
print(f"  变异系数: CV_A = {cv_A:.4f}")
result_a = '<' if cv_A < SPINDLE_CV_MIN else '>'
status_a = '通过' if cv_A >= SPINDLE_CV_MIN else '未通过'
print(f"  判断: CV_A = {cv_A:.4f} {result_a} {SPINDLE_CV_MIN}")
print(f"  -> T7 约束: {status_a}")
print(f"     (连续振荡不够成簇)")

print(f"\n--- 信号 B (成簇模式) ---")
print(f"  envelope_B 均值: mu = {envelope_B.mean():.4f}")
print(f"  envelope_B 标准差: sigma = {envelope_B.std():.4f}")
print(f"  变异系数: CV_B = {cv_B:.4f}")
result_b = '<' if cv_B < SPINDLE_CV_MIN else '>'
status_b = '通过' if cv_B >= SPINDLE_CV_MIN else '未通过'
print(f"  判断: CV_B = {cv_B:.4f} {result_b} {SPINDLE_CV_MIN}")
print(f"  -> T7 约束: {status_b}")
print(f"     (成簇振荡充分的起伏，标志着真实 spindle 事件)\n")

print(f"物理解释：")
print(f"  - 生理学中，spindle 是由间歇性的 thalamic 脉冲驱动")
print(f"  - 这导致皮层响应呈现成簇模式：高-低-高-低的周期性")
print(f"  - 包络 CV 大意味着在能量度上，存在明显的亮-暗对比")
print(f"  - 这是健康 spindle 的特征，与连续或平稳的虚假振荡区分开\n")

# ============================================================================
# 第 6 部分：把包络指标映射回 Spindle 检测逻辑
# ============================================================================
print("=" * 70)
print("第 6 部分：在 Spindle 检测中使用包络与变异系数")
print("=" * 70)

# 在包络上设置阈值来检测 spindle 事件
print(f"\nSpindle 事件检测步骤 (T8 约束)：")
print(f"  1. 计算包络的平滑版本 (高斯模糊)")
print(f"  2. 设置阈值：第 75 百分位数")
print(f"  3. 找出包络超过阈值的时间段")
print(f"  4. 过滤出符合持续时间要求的事件")
print(f"  5. 计算总事件数\n")

# 平滑包络
sigma_samples = 200 * FS / 1000.0  # 200 ms 高斯平滑
envelope_A_smooth = gaussian_filter1d(envelope_A, sigma=sigma_samples)
envelope_B_smooth = gaussian_filter1d(envelope_B, sigma=sigma_samples)

thresh_A = np.percentile(envelope_A_smooth, 75)
thresh_B = np.percentile(envelope_B_smooth, 75)

print(f"信号 A (连续) 事件检测：")
print(f"  平滑 sigma = {sigma_samples:.0f} 样本 ({200} ms)")
print(f"  75 百分位阈值 = {thresh_A:.4f}")
above_A = (envelope_A_smooth > thresh_A).astype(int)
print(f"  超过阈值的样本数 = {above_A.sum()}")
print(f"  占比 = {100 * above_A.sum() / len(above_A):.1f}%")

print(f"\n信号 B (成簇) 事件检测：")
print(f"  平滑 sigma = {sigma_samples:.0f} 样本 ({200} ms)")
print(f"  75 百分位阈值 = {thresh_B:.4f}")
above_B = (envelope_B_smooth > thresh_B).astype(int)
print(f"  超过阈值的样本数 = {above_B.sum()}")
print(f"  占比 = {100 * above_B.sum() / len(above_B):.1f}%\n")

# 检测事件的开始和结束（通过差分）
diff_A = np.diff(np.concatenate(([0], above_A, [0])))
starts_A = np.where(diff_A == 1)[0]
ends_A   = np.where(diff_A == -1)[0]
n_events_A = len(starts_A)

diff_B = np.diff(np.concatenate(([0], above_B, [0])))
starts_B = np.where(diff_B == 1)[0]
ends_B   = np.where(diff_B == -1)[0]
n_events_B = len(starts_B)

print(f"信号 A：检测到 {n_events_A} 个事件")
if n_events_A > 0:
    durations_A = (ends_A - starts_A) / FS
    print(f"  事件持续时间范围：{durations_A.min():.3f}-{durations_A.max():.3f} s")
    print(f"  平均持续时间：{durations_A.mean():.3f} s")

print(f"\n信号 B：检测到 {n_events_B} 个事件")
if n_events_B > 0:
    durations_B = (ends_B - starts_B) / FS
    print(f"  事件持续时间范围：{durations_B.min():.3f}-{durations_B.max():.3f} s")
    print(f"  平均持续时间：{durations_B.mean():.3f} s")
    if n_events_B >= 2:
        print(f"  事件详情：")
        for i in range(min(n_events_B, 5)):
            print(f"    事件 {i+1}: [{t[starts_B[i]]:.3f}, {t[ends_B[i]]:.3f}] s, "
                  f"持续 {durations_B[i]:.3f} s")

print(f"\nT7 与 T8 约束的联系：")
print(f"  - T7 (包络 CV > 0.7): 衡量成簇特性 -> 信号 A 不通过，信号 B 通过")
print(f"  - T8 (>= 5 个事件，0.3-2.0 s): 衡量事件数量和长度")
print(f"  - 两者结合检查 spindle 的真实性和连贯性\n")

# ============================================================================
# 生成结论与可视化
# ============================================================================
print("=" * 70)
print("结论")
print("=" * 70)
conclusion = """
1. Butterworth 滤波器 (butter):
   - 通过设计二阶级联段，稳定地提取 10-14 Hz 频段
   - 4 阶 = 2 个 SOS 段，数值稳定性好

2. 零相位滤波 (sosfiltfilt):
   - 双向应用滤波器，消除相位延迟
   - 保留信号中各分量的相对相位关系

3. Hilbert 变换与包络 (hilbert -> abs):
   - 从实信号提取瞬时包络（振幅外轮廓）
   - 包络平滑，反映能量的时间变化

4. 变异系数 CV = sigma/mu:
   - 小 CV: 连续平稳振荡（不符合短暂 spindle 的生理学）
   - 大 CV: 成簇、高-低起伏（真实 spindle 特征）

5. T7 约束 (包络 CV > 0.7):
   - 检查 spindle 是否呈现成簇而非连续
   - 结合 T8 (事件计数) 进一步验证多个独立的 spindle 个体

在 s4_personalize_fig7_v7.py 中，这个流程用于检查模拟的 thalamic 活动
是否产生生理学合理的 spindle （不是虚假的连续振荡）。
"""
print(conclusion)

print("生成可视化图表...\n")
# 创建绘图
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

# 原始信号
axes[0, 0].plot(t, signal_A, label='Original', alpha=0.7)
axes[0, 0].set_title('Signal A: Raw (Continuous Spindle Mode)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid()

axes[0, 1].plot(t, signal_B, label='Original', alpha=0.7)
axes[0, 1].set_title('Signal B: Raw (Bursty Spindle Mode)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].legend()
axes[0, 1].grid()

# 滤波后信号
axes[1, 0].plot(t, filtered_A, label='Filtered (10-14 Hz)', alpha=0.7, color='orange')
axes[1, 0].set_title('Signal A: After sosfiltfilt')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].legend()
axes[1, 0].grid()

axes[1, 1].plot(t, filtered_B, label='Filtered (10-14 Hz)', alpha=0.7, color='orange')
axes[1, 1].set_title('Signal B: After sosfiltfilt')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].legend()
axes[1, 1].grid()

# 包络
axes[2, 0].plot(t, envelope_A, label='Envelope', color='red', alpha=0.7)
axes[2, 0].axhline(envelope_A.mean(), color='green', linestyle='--', label='Mean')
axes[2, 0].set_title(f'Signal A: Envelope (CV={cv_A:.3f})')
axes[2, 0].set_ylabel('Envelope Magnitude')
axes[2, 0].legend()
axes[2, 0].grid()

axes[2, 1].plot(t, envelope_B, label='Envelope', color='red', alpha=0.7)
axes[2, 1].axhline(envelope_B.mean(), color='green', linestyle='--', label='Mean')
axes[2, 1].set_title(f'Signal B: Envelope (CV={cv_B:.3f})')
axes[2, 1].set_ylabel('Envelope Magnitude')
axes[2, 1].legend()
axes[2, 1].grid()

# 平滑包络和阈值
axes[3, 0].plot(t, envelope_A_smooth, label='Smoothed Envelope', color='purple', alpha=0.7)
axes[3, 0].axhline(thresh_A, color='red', linestyle='--', label='Threshold (75 percentile)')
axes[3, 0].fill_between(t, 0, above_A * 0.5, alpha=0.3, label='Above Threshold')
axes[3, 0].set_title(f'Signal A: Event Detection ({n_events_A} events)')
axes[3, 0].set_xlabel('Time (s)')
axes[3, 0].set_ylabel('Envelope')
axes[3, 0].legend()
axes[3, 0].grid()

axes[3, 1].plot(t, envelope_B_smooth, label='Smoothed Envelope', color='purple', alpha=0.7)
axes[3, 1].axhline(thresh_B, color='red', linestyle='--', label='Threshold (75 percentile)')
axes[3, 1].fill_between(t, 0, above_B * 0.5, alpha=0.3, label='Above Threshold')
axes[3, 1].set_title(f'Signal B: Event Detection ({n_events_B} events)')
axes[3, 1].set_xlabel('Time (s)')
axes[3, 1].set_ylabel('Envelope')
axes[3, 1].legend()
axes[3, 1].grid()

plt.tight_layout()
plt.savefig('tutorial_envelope_burstiness.png', dpi=100, bbox_inches='tight')
print("✓ 图表已保存为 tutorial_envelope_burstiness.png\n")

print("=" * 70)
print("脚本执行完毕！")
print("=" * 70)
