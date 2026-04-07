"""
s3_sleep_kernel.py
Thalamocortical motif: ALNNode (cortex) ↔ ThalamicNode, 2-node network.
Parameters from neurolib paper Table 3 (Cakan et al. 2021).
Runs 60 s simulation; saves cortex and thalamus firing-rate time series.

正文阅读顺序（搜索「# ── 步骤」跳转）：
  步骤 1 — set_params_glob：按 fnmatch 批量写 MultiModel.params（键为长名）
  步骤 2 — ThalamoCorticalNetwork：节点、噪声下标、连接矩阵、_sync 网络耦合
  步骤 3 — MultiModel + dt / duration(ms) / sampling_dt / backend
  步骤 4 — Table 3 + 噪声与丘脑电导（mue/mui 用 MultiModel 有效尺度）
  步骤 5 — tc_model.run()，numba 失败则回退 jitcdde
  步骤 6 — 读取 t 与 r_mean_EXC，kHz→Hz，拆成皮层/丘脑
  步骤 7 — 保存 outputs/*.npy，打印摘要（t 可能为秒，见文末判断）

API notes (discovered empirically):
  - Network.__init__(nodes, connectivity_matrix, delay_matrix=None)
  - ALNNode(exc_seed=..., inh_seed=...)  [no `seed` kwarg]
  - ThalamicNode()
  - noise_input_idx must be set on each mass BEFORE super().__init__()
  - sync_variables / _sync() must cover every required_coupling of all masses
    ALN EXC  needs: network_exc_exc, network_exc_exc_sq
    TCR      needs: network_exc_exc
    TRN      needs: network_inh_exc
  - model.params keys are fully-qualified dotted strings; no wildcard support
"""

import os
import fnmatch
import numpy as np

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

# ── 步骤 1：通配参数写入 — MultiModel.params 键名很长，用 pattern 批量匹配 ─────

def set_params_glob(model, pattern, value):
    matched = [k for k in model.params if fnmatch.fnmatch(k, pattern)]
    if not matched:
        print(f"  [warn] no params matched: {pattern!r}")
    for k in matched:
        model.params[k] = value
    return matched


# ── 步骤 2：双节点网络 — ALN=皮层(0)，ThalamicNode=丘脑(1)；Table 3 耦合强度 ──

class ThalamoCorticalNetwork(Network):
    """
    2-node motif: ALNNode (cortex) ↔ ThalamicNode.
    Connectivity: cTh→ALN=0.15, cALN→Th=0.02  (Table 3).

    __init__: 建节点 → 设 index / idx_state_var → 各 mass.noise_input_idx（须先于 super）
              → connectivity[to,from]、delay → Network(...)
    _sync:    声明的 sync_variables 在此接到各节点 r_mean_EXC（见 _EXC_WITHIN_IDX）
    """

    name  = "Thalamocortical Motif"
    label = "TCNet"

    # Network-level coupling variables needed by the masses:
    #   ALN ExcitatoryMass : network_exc_exc, network_exc_exc_sq
    #   ThalamocorticalMass: network_exc_exc
    #   ThalamicReticularMass: network_inh_exc
    sync_variables = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]

    # We override _sync() fully; no default generic coupling.
    default_coupling = {}

    default_output = f"r_mean_{EXC}"
    output_vars    = [f"r_mean_{EXC}", f"r_mean_{INH}"]

    # Within-node indices of the coupling output variable for each node:
    #   ALN: r_mean_EXC at index 6 (ExcitatoryALNMass has 7 vars, r_mean_EXC last)
    #   Th : r_mean_EXC at index 9 (ThalamocorticalMass has 10 vars, r_mean_EXC last)
    _EXC_WITHIN_IDX = [6, 9]   # one per node

    def __init__(self):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()

        aln.index = 0
        aln.idx_state_var = 0
        th.index  = 1
        th.idx_state_var = aln.num_state_variables   # = 13

        # Set noise_input_idx for every mass BEFORE super().__init__()
        # (super calls init_network → init_node → init_mass; init_mass skips
        #  setting noise_input_idx when already assigned.)
        # Layout: [ALN_EXC=0, ALN_INH=1, TCR=2, TRN=3]
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]

        # Connectivity matrix [to, from]:
        #   C[0,1] = 0.02  → Thalamus → Cortex
        #   C[1,0] = 0.15  → Cortex   → Thalamus
        connectivity = np.array([
            [0.0,  0.02],
            [0.15, 0.0 ],
        ])
        delay = np.zeros_like(connectivity)

        super().__init__(
            nodes=              [aln, th],
            connectivity_matrix=connectivity,
            delay_matrix=       delay,
        )

    def _sync(self):
        """
        Build coupling helpers for the numba / jitcdde backend.

        network_exc_exc   : additive weighted sum of r_mean_EXC from each node
        network_exc_exc_sq: same using squared connectivity (for ALN adaptation)
        network_inh_exc   : excitatory drive to TRN (also r_mean_EXC source)
        """
        # Nodal (within-node) coupling first
        all_couplings = sum([node._sync() for node in self], [])

        wi = self._EXC_WITHIN_IDX   # [ALN_idx=6, Th_idx=9]

        # 2a network_exc_exc — 跨节点兴奋率加权和
        all_couplings += self._additive_coupling(
            within_node_idx=wi,
            symbol="network_exc_exc",
        )
        # 2b network_exc_exc_sq — 权重平方（ALN 平均场适应项用）
        all_couplings += self._additive_coupling(
            within_node_idx=wi,
            symbol="network_exc_exc_sq",
            connectivity=self.connectivity * self.connectivity,
        )
        # 2c network_inh_exc — 兴奋驱动至 TRN 抑制 mass
        all_couplings += self._additive_coupling(
            within_node_idx=wi,
            symbol="network_inh_exc",
        )

        return all_couplings


# ── 步骤 3：封装为 MultiModel，并设积分步长、时长、采样与后端 ─────────────────

network  = ThalamoCorticalNetwork()
tc_model = MultiModel(network)

tc_model.params["dt"]          = 0.1
tc_model.params["sampling_dt"] = 1.0
tc_model.params["duration"]    = 60 * 1000   # 60 s，此处单位为 ms
tc_model.params["backend"]     = "numba"     # 失败则步骤 5 改 jitcdde

# ── 步骤 4：物理参数 — Table 3 + 噪声；mue/mui 为 MultiModel 标定值非核心 ALN 2.30 ──
# ALN background input — MultiModel units (neurolib example-4 reference values)
# Note: Table 3 values (mue=2.30) are for core ALNModel in mV/ms;
#       MultiModel ALNNode uses a different effective range (~3.2–4.2)
set_params_glob(tc_model, "*ALNMassEXC*.input_0.mu",    3.20)   # ← was 2.30
set_params_glob(tc_model, "*ALNMassINH*.input_0.mu",    3.50)   # ← was 3.44

# Adaptation parameters — keep Table 3 values
set_params_glob(tc_model, "*ALNMassEXC*.tauA",         1040.0)
set_params_glob(tc_model, "*ALNMassEXC*.b",              19.5)
set_params_glob(tc_model, "*ALNMassEXC*.a",               0.0)
# noise strength
set_params_glob(tc_model, "*ALNMass*.input_0.sigma",    0.05)
set_params_glob(tc_model, "*TCR*.input_0.sigma",        0.005)
# noise correlation time
set_params_glob(tc_model, "*.input_0.tau",              5.0)
# thalamic conductances
set_params_glob(tc_model, "*TCR*.g_LK",                 0.1)
set_params_glob(tc_model, "*TRN*.g_LK",                 0.1)
set_params_glob(tc_model, "*TCR*.g_h",                  0.1)

# ── 步骤 5：数值积分 — 优先 numba；异常则换 jitcdde 重跑 ─────────────────────

print("Running 60 s thalamocortical simulation...")
try:
    tc_model.run()
except Exception as exc_err:
    print(f"[warn] numba failed ({exc_err}), retrying with jitcdde backend...")
    tc_model.params["backend"] = "jitcdde"
    tc_model.run()
print("Simulation complete.")

# ── 步骤 6：取时间轴与兴奋性群体发放率；neurolib 常为 kHz，乘 1000 → Hz ───────

t_ms  = tc_model["t"]                    # (n_points,) — 可能是 ms 或 s，见步骤 7
r_exc = tc_model[f"r_mean_{EXC}"]       # (2, n_points)，行 0=皮层 1=丘脑

if r_exc.ndim == 2 and r_exc.shape[0] == 2:
    r_cortex   = r_exc[0, :] * 1000     # kHz → Hz
    r_thalamus = r_exc[1, :] * 1000
elif r_exc.ndim == 1:
    # single trace (isolated node fallback)
    r_cortex   = r_exc * 1000
    r_thalamus = np.zeros_like(r_cortex)
else:
    # unexpected shape — take first two rows
    r_cortex   = r_exc[0, :] * 1000
    r_thalamus = r_exc[1, :] * 1000

# ── 步骤 7：落盘 + 控制台摘要；t 末尾若 >1000 粗判为 ms 再换算成秒用于显示 ──

os.makedirs("outputs", exist_ok=True)
np.save("outputs/t_ms.npy",       t_ms)
np.save("outputs/r_cortex.npy",   r_cortex)
np.save("outputs/r_thalamus.npy", r_thalamus)

# tc_model["t"] 也可能是秒（neurolib 版本差异）；用 t[-1] 与 1000 粗分
t_end_s = t_ms[-1] if t_ms[-1] < 1000 else t_ms[-1] / 1000

# 核对关键参数是否写入
mue_key = next((k for k in tc_model.params if "ALNMassEXC" in k and "input_0.mu" in k), None)
b_key   = next((k for k in tc_model.params if "ALNMassEXC" in k and k.endswith(".b")), None)
print(f"Key params set:")
print(f"  mue = {tc_model.params.get(mue_key, 'not found')}")
print(f"  b   = {tc_model.params.get(b_key,   'not found')}")

print(f"Time axis      : {len(t_ms)} points, {t_end_s:.1f} s  (raw t[-1]={t_ms[-1]:.3f})")
print(f"Cortex  r_E    : {r_cortex.min():.2f} – {r_cortex.max():.2f} Hz")
print(f"Thalamus r_TCR : {r_thalamus.min():.2f} – {r_thalamus.max():.2f} Hz")
print("Saved: outputs/t_ms.npy, outputs/r_cortex.npy, outputs/r_thalamus.npy")
