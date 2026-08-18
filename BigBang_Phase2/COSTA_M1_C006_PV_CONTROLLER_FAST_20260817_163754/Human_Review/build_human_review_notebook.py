from __future__ import annotations

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "COSTA_C006_THREAD_HUMAN_REVIEW.ipynb"
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (neurolib)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# COSTA Phase 2A → C006 Controller: Human Review Notebook

> **Evidence ceiling: `EXPLORATORY_ONLY`**  
> 本 notebook 只读汇总当前 thread 已生成的工件，不读取 raw data、不重新运行模型、不重新拟合 C006，也不构成正式 C1/C2 或患者疗效证据。

组织方式参考 [sbi-practical-guide Figure 10 assembly notebook](https://github.com/sbi-dev/sbi-practical-guide/blob/main/paper/fig10_pyloric/notebooks/02_assemble_figure.ipynb)：保留原始 panel，同时从机器可读工件重新组装统一的审阅图和决策表。

**审阅问题：** 当前结果支持什么？哪些结果已失效？为什么没有达到 C1？下一步应该改变 target、actuator，还是停止？
""")

code(r"""
from pathlib import Path
import json, platform, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from IPython.display import display, Markdown, Image, HTML

REVIEW_DIR = Path.cwd()
C006_PV = REVIEW_DIR.parent
PHASE2 = C006_PV.parent
P2A = PHASE2 / "COSTA_PHASE2A_MVP_FAST_20260817_104026"
P2AB = PHASE2 / "COSTA_PHASE2AB_MVP_FAST_20260817_112545"
C1 = PHASE2 / "COSTA_C1_PROXY_MVP_FAST_20260817_120709"
C006_OLD = PHASE2 / "COSTA_M1_C006_CONTROLLER_FAST_20260817_161959"

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def show_png(path, width=1050):
    path = Path(path)
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        display(Markdown(f"⚠ Missing image: `{path}`"))

print("Human review directory:", REVIEW_DIR)
print("Final C006 result root:", C006_PV)
print("Python:", sys.executable)
print("Version:", platform.python_version())
""")

md(r"""
## 1. Executive decision trail

这张表区分“计算执行成功”和“科学门槛通过”。Phase 2A 的探索性关闭不等于 C1；最终 C006 controller bakeoff 完成但没有 controller 达到冻结门槛。
""")

code(r"""
p2a_status = read_json(P2A / "mvp_phase2a_status.json")
p2a_close = read_json(P2AB / "PHASE2A_CLOSEOUT_DECISION.json")
pid_dec = read_json(P2AB / "PHASE2B_PID_DECISION.json")
anchor_dec = read_json(C1 / "PERSONALIZATION_ANCHOR_DECISION.json")
old_dec = read_json(C006_OLD / "CONTROLLER_SELECTION_DECISION.json")
pv_dec = read_json(C006_PV / "CONTROLLER_SELECTION_DECISION.json")

trail = pd.DataFrame([
    ["Phase 2A coarse screen", "27/27 stable", p2a_status.get("status", "COMPLETE"), "Exploratory sensitivity map"],
    ["Phase 2A closeout", "9/9 stable", p2a_close["decision"], "Permitted exploratory Phase 2B only"],
    ["Earlier neurolib PID", "3/3 stable", pid_dec["decision"], "V1/neurolib plant; not valid R4/C006 baseline"],
    ["C1 personalization anchor", "6/6 stable", anchor_dec["decision"], "Control phase not executed"],
    ["First C006 controller", "superseded", old_dec["decision"], "Stopped: strict phase-control match failed"],
    ["Final C006 PLL/PV bakeoff", "24/24 stable", pv_dec["decision"], "No controller met T2 and R gates"],
], columns=["Stage", "Execution", "Decision", "Interpretation"])
display(trail.style.hide(axis="index"))

fig, ax = plt.subplots(figsize=(12, 2.8))
colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#b07aa1", "#e15759"]
ax.scatter(range(len(trail)), np.zeros(len(trail)), s=500, c=colors, zorder=3)
ax.plot(range(len(trail)), np.zeros(len(trail)), color="0.75", lw=3, zorder=1)
for i, row in trail.iterrows():
    ax.text(i, .14 if i%2==0 else -.14, row.Stage.replace(" ", "\n"), ha="center",
            va="bottom" if i%2==0 else "top", fontsize=9)
ax.set_ylim(-.55,.55); ax.set_xlim(-.4,len(trail)-.6); ax.axis("off")
ax.set_title("Thread-level evidence and decision trail")
plt.show()
""")

md("## 2. Phase 2A coarse T1–T8 control screen\n\n原始 quicklook 与统一重绘并列保留。数值是相对 sham 的探索性响应，不是临床效应。")
code(r"""
show_png(P2A / "mvp_phase2a_quicklook.png")
coarse = pd.read_csv(P2A / "mvp_phase2a_quicklook.csv").set_index("target")
dose_cols = [c for c in coarse.columns if c.startswith("dose_")]
matrix = coarse[dose_cols].astype(float)
fig, ax = plt.subplots(figsize=(12, 5))
lim = np.nanpercentile(np.abs(matrix.values), 95)
im = ax.imshow(matrix.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
ax.set_yticks(range(len(matrix)), matrix.index)
ax.set_xticks(range(len(dose_cols)), [c.replace("dose_","") for c in dose_cols], rotation=40, ha="right")
ax.set_xlabel("Dose (development grid)"); ax.set_title("Phase 2A relative response heatmap")
fig.colorbar(im, ax=ax, label="Relative change vs sham")
plt.show()
""")

md("## 3. Minimal Phase 2A replication and exploratory PID\n\nPhase 2A closeout passed its exploratory exit gate. The PID result below came from the earlier V1/neurolib plant and must not be treated as an R4/C006 PID comparator.")
code(r"""
rep = pd.read_csv(P2AB / "phase2a_8target_evaluation.csv")
fig, ax = plt.subplots(figsize=(10,4))
x=np.arange(len(rep)); w=.36
ax.bar(x-w/2, rep.negative_effect_median, w, label="−0.035")
ax.bar(x+w/2, rep.positive_effect_median, w, label="+0.035")
ax.axhline(0,color="black",lw=.8); ax.set_xticks(x,rep.target); ax.set_ylabel("Median paired relative effect")
ax.set_title("Phase 2A replication: all eight targets"); ax.legend(); plt.show()
show_png(P2AB / "pid_quicklook.png")
pid = pd.read_csv(P2AB / "pid_summary.csv")
display(pid[["seed","target","final_target_value","command_range","saturation_fraction","numerically_stable"]].style.hide(axis="index"))
""")

md("## 4. Attempted C1 proxy anchor\n\nThe extra SC4001 PSD anchor failed: the accessible V1 candidate was not better than the generic reference. This does **not** reverse the frozen R4/M1 C006 mechanistic result; it only invalidates that particular C1 proxy route.")
code(r"""
show_png(C1 / "c1_quicklook.png")
anchor = pd.read_csv(C1 / "personalization_anchor.csv")
bands = ["so_0p5_1p25_hz_log_psd_mae","delta_0p5_4_hz_log_psd_mae","sigma_10_15_hz_log_psd_mae"]
long=[]
for model,g in anchor.groupby("model"):
    for b in bands:
        for value in g[b]: long.append({"model":model,"band":b.split("_log")[0],"error":value})
ald=pd.DataFrame(long)
fig,ax=plt.subplots(figsize=(9,4))
for j,(model,g) in enumerate(ald.groupby("model")):
    means=g.groupby("band").error.mean().reindex([b.split("_log")[0] for b in bands])
    ax.plot(range(3),means.values,marker="o",label=model)
ax.set_xticks(range(3),["SO","Delta","Sigma"]);ax.set_ylabel("log-PSD MAE (lower is better)")
ax.set_title("Personalization anchor comparison");ax.legend();plt.show()
display(pd.DataFrame(anchor_dec["gates"].items(), columns=["Anchor gate","Passed"]).style.hide(axis="index"))
""")

md(r"""
## 5. Superseded first C006 controller attempt

> **SUPERSEDED / INVALID FOR CONTROLLER SELECTION**  
> 该图仅用于审阅失败历史。其 phase-shifted negative control 没有严格匹配总能量；随后一次修复仍在第三个种子触发 `PHASE_CONTROL_MATCH_FAILED`。不得引用图中的 controller 排名。
""")
code(r"""
show_png(C006_OLD / "controller_quicklook.png")
display(pd.DataFrame([read_json(C006_OLD / "SUPERSEDED_ARTIFACTS.json")]).style.hide(axis="index"))
""")

md("## 6. Final C006 literature-driven PLL/PV controller bakeoff\n\n这是当前有效 controller comparison：24/24 stable，预冻结门槛未被任何 controller 达到。")
code(r"""
show_png(C006_PV / "controller_quicklook.png")
runs = pd.read_csv(C006_PV / "controller_runs.csv")
active = runs[runs.condition != "SHAM"].copy()
order=["FIXED","PLL_PHASE_LOCKED","PV_PHASE_LOCKED","PV_PHASE_LOCKED_ADAPTIVE","PV_PHASE_SHIFTED_YOKED"]
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for j,metric in enumerate(["R","control_energy"]):
    for i,c in enumerate(order):
        vals=active.loc[active.condition==c,metric].values
        axes[j].scatter(np.full(len(vals),i),vals,s=45,zorder=3)
        axes[j].plot([i-.22,i+.22],[np.median(vals)]*2,color="black",lw=3)
    axes[j].set_xticks(range(len(order)),order,rotation=35,ha="right")
    axes[j].set_title("Per-seed score" if metric=="R" else "Per-seed control energy")
    axes[j].set_ylabel(metric)
axes[0].axhline(.10,color="#e15759",ls="--",label="MVP gate");axes[0].axhline(0,color="0.5",lw=.8);axes[0].legend()
plt.tight_layout();plt.show()
display(active[["seed","condition","R","dT2","dT5","dT6","pulse_count","control_energy"]].style.format(precision=4).hide(axis="index"))
""")

md("## 7. Final T1–T8 response map")
code(r"""
eight = pd.read_csv(C006_PV / "controller_8target_evaluation.csv")
pivot=eight.pivot(index="condition",columns="target",values="median_relative_effect").reindex(order)
fig,ax=plt.subplots(figsize=(12,4.8));lim=max(.02,np.nanpercentile(np.abs(pivot.values),95))
im=ax.imshow(pivot.values,aspect="auto",cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
ax.set_yticks(range(len(pivot)),pivot.index);ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=35,ha="right")
ax.set_title("Median relative T1–T8 effects vs paired sham")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]): ax.text(j,i,f"{pivot.iloc[i,j]:+.2f}",ha="center",va="center",fontsize=8)
fig.colorbar(im,ax=ax,label="Relative change");plt.show()

gates = pd.DataFrame(pv_dec["controller_gates"]).T
display(gates.style.applymap(lambda v:"background-color:#d9ead3" if v is True else "background-color:#f4cccc" if v is False else ""))
""")

md("## 8. Strict yoked negative-control validation\n\nUnlike the superseded attempt, the final negative control is a circular roll of the complete adaptive command vector. It therefore preserves the exact amplitude sample multiset, non-zero samples and total energy.")
code(r"""
yoked=read_json(C006_PV/"yoked_control_validation.json")
yc=pd.DataFrame(yoked["checks"]);yp=pd.DataFrame(yoked["phase_pairs"])
fig,axes=plt.subplots(1,3,figsize=(13,3.6))
axes[0].bar(yc.seed.astype(str),yc.energy_abs_error);axes[0].set_title("Energy mismatch");axes[0].set_ylabel("absolute error")
axes[1].bar(yc.seed.astype(str),yc.pulse_count_source);axes[1].scatter(yc.seed.astype(str),yc.pulse_count_shifted,color="black",label="shifted");axes[1].set_title("Matched logical pulse count");axes[1].legend()
axes[2].bar(yp.seed.astype(str),yp.separation_deg);axes[2].axhspan(120,240,color="#59a14f",alpha=.15);axes[2].set_title("Realized phase separation");axes[2].set_ylabel("degrees")
plt.tight_layout();plt.show()
display(yc.style.format(precision=12).hide(axis="index"));display(yp.style.format(precision=2).hide(axis="index"))
""")

md("## 9. C1 readiness audit")
code(r"""
c1_audit=pd.DataFrame([
 ["Frozen mechanistic candidate available", True, "R4/M1 C006; discrete frozen candidate"],
 ["Controller pipeline numerically executable", True, "24/24 final simulations stable"],
 ["Strict phase negative control valid", True, "3/3 exact energy match; valid phase separation"],
 ["Valid R4 PID baseline", False, "NO_VALID_R4_PID_BASELINE"],
 ["At least one controller median T2 gain ≥10%", False, "All controllers failed T2 gate"],
 ["At least one controller median R ≥10%", False, "Best median R was PLL at about +0.26%"],
 ["Adaptive phase controller beats yoked control by ≥5 pp", False, "Adaptive was lower by about 0.90 pp"],
 ["Exploratory C1 proxy MVP met", False, "NO_CONTROLLER_MET_MVP_GATE"],
 ["Formal C1 certified", False, "No real intervention pre/post evidence"],
],columns=["Criterion","Pass","Evidence"])
display(c1_audit.style.apply(lambda row:["background-color:#d9ead3" if row.Pass else "background-color:#f4cccc"]*3,axis=1).hide(axis="index"))
display(Markdown("### Review conclusion\n**当前没有达到探索性 C1 proxy，也没有达到正式 C1。** 已证明的是模型和 controller infrastructure 可执行，而不是睡眠质量指标得到了充分改善。"))
""")

md("## 10. Human-review decision matrix\n\n这不是自动实验计划；它把需要专家判断的科学分叉显式化。继续计算前应选择一个方向。")
code(r"""
review=pd.DataFrame([
 ["A. Preserve C006 and T2", "Keep u_E and T2; accept current negative result", "Highest comparability; low probability that more tuning changes conclusion", "Stop or archive"],
 ["B. Preserve C006, reconsider target", "Use T6/coupling or a predeclared multimetric endpoint", "Matches observed response better; changes C1 story", "Sleep-physiology review required"],
 ["C. Preserve C006, reconsider actuator", "Test another mechanistically justified control input/polarity", "May improve controllability; requires model-author justification", "Mechanistic review required"],
 ["D. Change both target and actuator", "New scientific hypothesis", "Highest flexibility; no longer a minimal continuation", "New exploratory protocol"],
],columns=["Option","Scientific choice","Trade-off","Required decision"])
display(review.style.hide(axis="index"))
""")

md("## 11. Artifact registry")
code(r"""
artifacts = [
 ("Phase 2A coarse quicklook", P2A/"mvp_phase2a_quicklook.png", "valid exploratory"),
 ("Phase 2A closeout", P2AB/"PHASE2A_CLOSEOUT_DECISION.json", "valid exploratory"),
 ("Earlier PID quicklook", P2AB/"pid_quicklook.png", "non-C006 contextual result"),
 ("C1 anchor quicklook", C1/"c1_quicklook.png", "failed anchor"),
 ("First C006 quicklook", C006_OLD/"controller_quicklook.png", "SUPERSEDED"),
 ("Final PV controller quicklook", C006_PV/"controller_quicklook.png", "valid current result"),
 ("Final decision", C006_PV/"CONTROLLER_SELECTION_DECISION.json", "valid current decision"),
 ("Yoked validation", C006_PV/"yoked_control_validation.json", "valid current validation"),
]
registry=pd.DataFrame([(name,str(path.relative_to(PHASE2)),status,path.exists()) for name,path,status in artifacts],columns=["Artifact","Relative path from BigBang_Phase2","Status","Exists"])
display(registry.style.hide(axis="index"))
""")

md(r"""
---

### Evidence boundary

- C006 是冻结的离散 mechanistic candidate，不是生物参数真值或正式患者数字孪生。
- 模型输出为 `mau` proxy，不是 scalp EEG voltage。
- 当前结果不支持真实患者睡眠改善、临床疗效、正式 C1/C2 或论文终案结论。
- Human review 的目的，是决定下一项科学假设，而不是通过追加 tuning 覆盖负面结果。
""")

nb["cells"] = cells
nbf.write(nb, OUT)
print(OUT)
