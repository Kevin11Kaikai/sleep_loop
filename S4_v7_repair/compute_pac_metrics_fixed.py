"""
compute_pac_metrics_fixed.py
=============================
Drop-in replacement for v7's compute_pac_metrics.

PURPOSE
-------
Quantify SO–spindle phase-amplitude coupling in a way that is robust to
the non-sinusoidal cortical SO produced by the ALN/ThalamicNode model.

WHAT WAS WRONG WITH V7's VERSION
--------------------------------
v7 computed SO phase by:
    so_filt   = sosfiltfilt(butter(4, [0.5, 1.5], ...), r_ctx)
    so_phase  = np.angle(hilbert(so_filt))

This works for sinusoidal SO (Bedrosian's theorem). It FAILS for the
narrow-pulse cortex r_E that the model actually produces because:

  1. Bandpass smears each pulse into a smooth wave whose peak is ~50 ms
     LATER than the original pulse. Hilbert phase = 0 corresponds to
     the smeared peak, not the real UP peak.

  2. Long DOWN states (~1100 ms flat baseline) cause Hilbert phase to
     "linger" near ±π, giving 3.3× more samples to those bins than
     the others. This biases MVL angle away from the histogram peak.

  3. The "preferred phase" v7 reported (+159°) was therefore NEITHER
     the real UP peak NOR the real DOWN trough — it was a vector
     compromise distorted by uneven phase sampling.

THE FIX
-------
Replace the bandpass-Hilbert phase with cycle-by-cycle phase
(Cole & Voytek 2017, J Neurosci Methods, "Cycle-by-cycle analysis"):

    Detect r_ctx peaks → these are phase = 0 by definition.
    Linear interpolation gives phase ∈ [-π, π] between consecutive peaks.

This is exact at UP peaks (no Bedrosian violation) and produces
uniformly-distributed phase samples across [-π, π] (no DOWN-trough
over-sampling).

WHAT GETS REPORTED
------------------
The function returns SEVERAL phase indicators because the spindle
amplitude distribution can be unimodal OR bimodal, and a single number
("preferred phase") can mislead:

  mi                   : Tort 2010 KL Modulation Index
                         (strength of coupling, regardless of pattern)

  preferred_phase      : MVL angle on histogram
                         (canonical, but UNRELIABLE if bimodal)

  phase_argmax         : Center of bin with max amplitude
                         (robust to bimodality; report this when
                          bimodality_flag is True)

  phase_concentration  : |MVL on histogram|, in [0, 1]
                         (Helfrich 2018 "phase precision")

  up_down_ratio        : sum(amp at |phi| <= pi/2) / sum(amp at |phi| > pi/2)
                         (>1 = UP-locked, <1 = DOWN-locked)

  bimodality_flag      : True if histogram has 2+ well-separated peaks
                         (informational only, NOT a feasibility criterion)

  n_so_cycles          : Number of detected SO cycles
                         (sanity check; <3 -> ok=False)

  ok                   : Whether computation succeeded
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks


def compute_pac_metrics(r_ctx, r_thal, fs,
                        SO_FREQ_LO=0.5, SO_FREQ_HI=1.5,
                        SPINDLE_LO=10.0, SPINDLE_HI=14.0,
                        PAC_N_BINS=18,
                        SO_MIN_PERIOD_S=0.7,
                        SO_PEAK_PROMINENCE_FRAC=0.3):
    """
    Compute PAC metrics with cycle-by-cycle phase definition.

    Parameters
    ----------
    r_ctx : 1-D array
        Cortical excitatory firing rate r_E (NOT the bandpassed signal).
    r_thal : 1-D array
        Thalamic firing rate (e.g., r_TCR).
    fs : float
        Sampling rate in Hz.
    SO_FREQ_LO, SO_FREQ_HI : float
        Used for r_ctx peak detection bounds, not for filtering r_ctx.
        Kept as parameters for backward compatibility with v7's constants.
    SPINDLE_LO, SPINDLE_HI : float
        Spindle band edges (used to extract spindle envelope from r_thal).
    PAC_N_BINS : int
        Number of phase bins for KL-MI. Tort 2010 standard: 18.
    SO_MIN_PERIOD_S : float
        Minimum spacing between r_ctx peaks (sec). Default 0.7s = 1.4 Hz max.
    SO_PEAK_PROMINENCE_FRAC : float
        Min peak prominence as fraction of r_ctx range (0-1).
        0.3 means each peak must rise at least 30% of total range above
        its surroundings.

    Returns
    -------
    dict with keys: mi, preferred_phase, phase_argmax, phase_concentration,
                    up_down_ratio, bimodality_flag, n_so_cycles, ok
    """
    out = {
        "mi": 0.0,
        "preferred_phase": np.pi,
        "phase_argmax": np.pi,
        "phase_concentration": 0.0,
        "up_down_ratio": 0.0,
        "bimodality_flag": False,
        "n_so_cycles": 0,
        "ok": False,
    }

    # Sanity checks
    if len(r_ctx) < int(2 * fs) or len(r_thal) < int(2 * fs):
        return out
    if r_ctx.std() < 1e-6 or r_thal.std() < 1e-6:
        return out

    try:
        # ===============================================================
        # 第一步 - 从r_ctx中提取SO相位
        # ===============================================================
        ctx_range = r_ctx.max() - r_ctx.min() # range of r_ctx values, used to set peak prominence threshold
        if ctx_range < 1e-6: #如果ctx_range太小，说明r_ctx信号没有足够的变化，无法可靠地检测峰值，因此直接返回默认输出
            return out

        prominence = SO_PEAK_PROMINENCE_FRAC * ctx_range 
        # prominence是峰值检测的一个参数，表示每个峰值必须比其周围的值高出至少prominence才能被认为是一个有效的峰值。
        # 这里将prominence设置为r_ctx范围的一定比例，以适应不同信号的幅度。
        ctx_peaks, _ = find_peaks(
            r_ctx,
            distance=int(SO_MIN_PERIOD_S * fs),
            prominence=prominence,
        ) # find_peaks函数用于检测r_ctx中的峰值，返回峰值的索引ctx_peaks和相关属性（这里用_表示不使用）。distance参数确保峰值之间的最小距离，prominence参数确保峰值的显著性。

        if len(ctx_peaks) < 3:
            return out

        # Reject implausible SO rates
        mean_isi_s = (ctx_peaks[-1] - ctx_peaks[0]) / fs / (len(ctx_peaks) - 1)
        # mean_isi_s是平均峰间隔时间，单位为秒。通过计算最后一个峰和第一个峰之间的时间差，并除以峰的数量减一，得到平均每个SO周期的持续时间。
        if not (0.5 <= mean_isi_s <= 4.0): #如果平均峰间隔时间不在0.5秒到4秒之间，说明检测到的SO周期不符合生理范围，因此直接返回默认输出。
            return out

        # Build phase: 0 at each peak, ±π midway, wrapped to [-π, π]
        so_phase = np.full(len(r_ctx), np.nan) # so_phase是一个与r_ctx长度相同的数组，用于存储每个时间点对应的SO相位。
        #初始值为NaN，表示未定义。
        for i in range(len(ctx_peaks) - 1): # 循环遍历每对相邻的峰值索引，计算它们之间的相位。
            p0 = ctx_peaks[i] # p0是当前峰值的索引
            p1 = ctx_peaks[i + 1] # p1是下一个峰值的索引
            cycle_len = p1 - p0 # cycle_len是当前SO周期的长度，以样本数为单位。它等于下一个峰值索引减去当前峰值索引。
            if cycle_len <= 0: # 如果cycle_len小于等于0，说明峰值索引不正确（例如重复或逆序），无法计算相位，因此跳过这个周期。
                continue
            rel = (np.arange(p0, p1) - p0) / cycle_len # rel是一个数组，表示当前SO周期内每个样本相对于当前峰值的位置，归一化为[0, 1]。

            phase = np.where(rel < 0.5,
                             2 * np.pi * rel,
                             2 * np.pi * (rel - 1)) # phase是一个数组，表示当前SO周期内每个样本的相位。
            #对于rel < 0.5的样本，相位从0线性增加到π；对于rel >= 0.5的样本，相位从-π线性增加到0。这种定义确保了相位在每个峰值处为0，在两个峰值之间平滑变化，
            # 并且在下一个峰值处再次回到0。
            so_phase[p0:p1] = phase # so_phase数组的p0到p1范围内被赋值为计算得到的phase数组，表示这个SO周期内每个样本的相位。

        # ===============================================================
        # 第二步 - 从r_thal中提取纺锤体振幅包络
        # ===============================================================
        sos_sp = butter(4, [SPINDLE_LO, SPINDLE_HI],
                        btype="band", fs=fs, output="sos")
        # sos_sp是一个二阶节系数数组，用于对r_thal进行带通滤波，以提取纺锤体频段的信号。
        # 这里使用4阶Butterworth滤波器，频率范围由SPINDLE_LO和SPINDLE_HI定义，采样率为fs。
        sp_filt = sosfiltfilt(sos_sp, r_thal)
        # sp_filt是经过带通滤波后的r_thal信号，保留了纺锤体频段的成分。sosfiltfilt函数实现了零相位滤波，避免了相位失真。
        sp_amp = np.abs(hilbert(sp_filt))
        # sp_amp是纺锤体频段信号的包络，表示纺锤体活动的瞬时幅度。通过对滤波后的信号进行Hilbert变换，并取其绝对值来计算包络(envelope)。

        # ===============================================================
        # 第三步 - 仅保留中间部分，避免边缘效应和NaN值
        # ===============================================================
        edge = int(0.5 * fs) # edge代表边缘效应的样本数，这里设置为0.5秒对应的样本数。
        #因为滤波和Hilbert变换在信号的开始和结束部分可能会产生不可靠的结果，所以我们只保留中间部分的数据。
        so_phase = so_phase[edge:-edge] # so_phase被切片，仅保留从edge到-edge的部分，去掉两端的edge样本，以避免边缘效应。
        sp_amp = sp_amp[edge:-edge]# sp_amp同样被切片，保留中间部分的数据。
        valid = ~np.isnan(so_phase) # valid是一个布尔数组，表示so_phase中哪些样本是有效的（即非NaN）。只有这些样本才会被用于后续的PAC计算。
        if valid.sum() < int(5 * fs): # 如果有效样本数少于5秒的数据量，说明数据不足以可靠地计算PAC，因此直接返回默认输出。
            return out
        so_phase_v = so_phase[valid] # so_phase_v是一个数组，仅包含有效样本对应的SO相位值。
        sp_amp_v = sp_amp[valid] # sp_amp_v是一个数组，仅包含有效样本对应的纺锤体振幅包络值。这两个数组将用于后续的PAC指标计算。

        # ===============================================================
        # STEP 4 - Tort 2010 KL Modulation Index
        # ===============================================================
        bin_edges = np.linspace(-np.pi, np.pi, PAC_N_BINS + 1) # bin_edges是一个数组，定义了SO相位的分箱边界，
        # 从-π到π均匀分成PAC_N_BINS个箱子。每个箱子对应一个相位范围，用于统计纺锤体振幅在不同SO相位下的分布。
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin_centers是bin_edges的中心点，表示每个相位箱的代表相位值。
        # 这些中心点将用于计算MVL和其他基于相位的指标。 
        mean_amp = np.zeros(PAC_N_BINS) # mean_amp是一个数组，用于存储每个相位箱中纺锤体振幅的平均值。初始值为0。
        for i in range(PAC_N_BINS):
            mask = (so_phase_v >= bin_edges[i]) & (so_phase_v < bin_edges[i + 1]) 
            # mask是一个布尔数组，表示so_phase_v中哪些样本的相位值落在当前箱子的边界内。通过这个掩码，我们可以选择对应相位范围内的纺锤体振幅值。
            if mask.any(): # 如果mask中有至少一个True，说明当前箱子内有样本，那么我们计算这些样本对应的纺锤体振幅的平均值，并存储在mean_amp[i]中。
                mean_amp[i] = sp_amp_v[mask].mean()
                # mean_amp[i]是当前相位箱内纺锤体振幅的平均值。这个过程完成后，mean_amp数组将包含每个相位箱的平均纺锤体振幅，用于后续的KL-MI计算。

        total = mean_amp.sum() # total是所有相位箱中平均纺锤体振幅的总和，用于归一化mean_amp以得到概率分布p。
        if total <= 0 or not np.isfinite(total):
            return out

        p = mean_amp / total # p是一个数组，表示每个相位箱的概率
        p_safe = np.where(p > 0, p, 1.0) # p_safe是一个数组，将p中小于等于0的值替换为1.0，以避免在计算对数时出现log(0)的问题。对于p中大于0的值，p_safe保持不变。
        H = -np.sum(p * np.log(p_safe)) # H是Tort 2010 KL Modulation Index计算中的Shannon熵，表示纺锤体振幅分布p的熵值。通过对每个相位箱的概率p乘以其对数，并取负值求和得到。
        mi = float(np.clip((np.log(PAC_N_BINS) - H) / np.log(PAC_N_BINS),
                           0.0, 1.0)) # mi是Tort 2010 KL Modulation Index，表示纺锤体振幅分布p相对于均匀分布的偏离程度。
        #通过计算log(N) - H，并除以log(N)进行归一化，使得mi的值在0到1之间，其中0表示完全均匀分布（无耦合），1表示所有振幅集中在一个相位箱（最大耦合）。使用np.clip确保mi的值在0到1的范围内。

        # ===============================================================
        # STEP 5 - Three independent phase indicators
        # ===============================================================
        mvl_hist = (p * np.exp(1j * bin_centers)).sum()
        # mvl_hist是一个复数，表示基于相位箱中心和对应概率的平均向量长度。它的模长表示相位集中程度，角度表示平均相位方向。
        preferred_phase = float(np.angle(mvl_hist))
        # preferred_phase是mvl_hist的角度，表示纺锤体振幅分布的平均相位方向。这个值在[-π, π]范围内，指示了纺锤体活动最强的SO相位。
        phase_concentration = float(np.abs(mvl_hist))
        # phase_concentration是mvl_hist的模长，表示纺锤体振幅分布的平均强度。

        phase_argmax = float(bin_centers[int(np.argmax(mean_amp))])
        # phase_argmax是mean_amp数组中最大值对应的bin_centers的值，表示纺锤体振幅最大的SO相位箱的中心相位。

        up_mask = np.abs(bin_centers) <= (np.pi / 2)
        # up_mask是一个布尔数组，表示bin_centers中哪些相位值的绝对值小于等于π/2，即哪些相位箱被认为是UP状态相关的。
        up_w = float(mean_amp[up_mask].sum())
        # up_w是所有UP相关相位箱中平均纺锤体振幅的总和，表示纺锤体活动在UP状态相关相位上的强度。
        down_w = float(mean_amp[~up_mask].sum())
        # down_w是所有非UP相关相位箱中平均纺锤体振幅的总和，表示纺锤体活动在DOWN状态相关相位上的强度。
        up_down_ratio = up_w / max(down_w, 1e-12)
        # up_down_ratio是UP相关相位箱的总振幅与DOWN相关相位箱的总振幅的比值。通过将down_w与一个非常小的数（1e-12）取最大值，避免了除以零的情况。这个比值大于1表示纺锤体活动更倾向于UP状态相关相位，小于1表示更倾向于DOWN状态相关相位。

        # ===============================================================
        # STEP 6 - Bimodality detection
        # ===============================================================
        bimodality_flag = _detect_bimodality(mean_amp, bin_centers)
        # bimodality_flag是一个布尔值，表示纺锤体振幅分布是否具有双峰特征。通过调用_detect_bimodality函数
        # ，根据mean_amp和bin_centers的分布情况来判断是否存在两个以上的显著峰值。
        out.update({
            "mi": mi,
            "preferred_phase": preferred_phase,
            "phase_argmax": phase_argmax,
            "phase_concentration": phase_concentration,
            "up_down_ratio": up_down_ratio,
            "bimodality_flag": bimodality_flag,
            "n_so_cycles": int(len(ctx_peaks) - 1),
            "ok": True,
        })

    except Exception:
        pass

    return out


def _detect_bimodality(mean_amp, bin_centers,
                       peak_min_height_frac=0.7,
                       trough_depth_frac=0.7,
                       min_separation_rad=np.pi / 2):
    """Heuristic bimodality detection on circular 18-bin histogram."""
    n = len(mean_amp)

    is_peak = np.zeros(n, dtype=bool)
    for i in range(n):
        prev_amp = mean_amp[(i - 1) % n]
        next_amp = mean_amp[(i + 1) % n]
        if mean_amp[i] >= prev_amp and mean_amp[i] >= next_amp \
                and mean_amp[i] > 0:
            is_peak[i] = True

    peak_indices = np.where(is_peak)[0]
    if len(peak_indices) < 2:
        return False

    peak_heights = mean_amp[peak_indices]
    sort_order = np.argsort(peak_heights)[::-1]
    top1, top2 = peak_indices[sort_order[0]], peak_indices[sort_order[1]]
    h1, h2 = peak_heights[sort_order[0]], peak_heights[sort_order[1]]

    if h2 < peak_min_height_frac * h1:
        return False

    sep = abs(bin_centers[top1] - bin_centers[top2])
    sep = min(sep, 2 * np.pi - sep)
    if sep < min_separation_rad:
        return False

    def _arc_indices(start, end, n):
        idxs = []
        i = start
        while i != end:
            i = (i + 1) % n
            idxs.append(i)
            if len(idxs) > n:
                break
        return idxs

    arc_a = _arc_indices(top1, top2, n)
    arc_b = _arc_indices(top2, top1, n)
    if not arc_a or not arc_b:
        return False
    trough_a = mean_amp[arc_a[:-1]].min() if len(arc_a) > 1 else h2
    trough_b = mean_amp[arc_b[:-1]].min() if len(arc_b) > 1 else h2
    deeper_trough = min(trough_a, trough_b)

    return deeper_trough <= trough_depth_frac * h2