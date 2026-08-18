from __future__ import annotations

import argparse, csv, hashlib, importlib.metadata, importlib.util, json, os, platform, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurolib, numpy as np, scipy, pandas, numba

from m1_streaming_control import C006, CausalPLL, DT, make_drive, metrics, score, simulate, simulate_open

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
R4 = WORKSPACE / "BigBang_V4D1" / "COSTA_MVP_FRESH_R4_20260816_023933"
PROTOCOL_PATH = R4 / "CODE" / "FROZEN_V2" / "protocol_v2.json"
MODEL_PATH = R4 / "CODE" / "MECHANISTIC_V2" / "model.py"
SELECTION_PATH = R4 / "RUNS" / "FIT_WORKER" / "SELECTION_FREEZE.json"
EXPECTED = {
    str(PROTOCOL_PATH): "338366593A2A4C8626349D70AA38306C5931F0FF4D22FD96DB546DD41505E3C3",
    str(MODEL_PATH): "775A3961093C386A1E7DF8F0D033963EAB684F2566AE206E5BB6DB4A6CD03869",
    str(SELECTION_PATH): "3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF",
}
PID_CANDIDATES = {
    "PID_025": {"Kp": .0125, "Ki": .00125, "Kd": .00025},
    "PID_050": {"Kp": .025, "Ki": .0025, "Kd": .0005},
    "PID_100": {"Kp": .05, "Ki": .005, "Kd": .001},
}
DEV = (44017, 55021, 66029); EVAL = (77131, 88141, 99149)


def utc(): return datetime.now(timezone.utc).isoformat()
def log(s):
    line=f"[{utc()}] {s}"; print(line, flush=True)
    with (ROOT/"RUN_LOG.txt").open("a",encoding="utf-8") as f: f.write(line+"\n")
def dump(name, obj): (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False),encoding="utf-8")
def rows_csv(name, rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with (ROOT/name).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def package_version(name, mod):
    return getattr(mod,"__version__",None) or importlib.metadata.version(name)


class Stop:
    def __init__(self, seconds): self.started=time.monotonic(); self.deadline=self.started+seconds
    def check(self):
        if time.monotonic() >= self.deadline: raise TimeoutError("900_SECOND_HARD_STOP")


def manifest():
    return {"created_utc":utc(),"conda_default_env":os.getenv("CONDA_DEFAULT_ENV"),
            "python_executable":sys.executable,"python_version":platform.python_version(),
            "versions":{"neurolib":package_version("neurolib",neurolib),"numpy":np.__version__,
                        "scipy":scipy.__version__,"pandas":pandas.__version__,
                        "matplotlib":matplotlib.__version__,"numba":numba.__version__},
            "evidence_ceiling":"EXPLORATORY_ONLY"}


def preflight():
    env=manifest(); dump("environment_manifest.json",env)
    exact=str(Path(r"C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe").resolve()).lower()
    if env["conda_default_env"] != "neurolib" or str(Path(sys.executable).resolve()).lower()!=exact or env["versions"]["neurolib"]!="0.6.1":
        raise RuntimeError("NEUROLIB_ENVIRONMENT_MISMATCH")
    hashes={p:sha(Path(p)) for p in EXPECTED}
    if hashes != EXPECTED: raise RuntimeError("M1_FROZEN_INPUT_MISMATCH")
    protocol=json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")); selection=json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    cand=next(x for x in protocol["candidate_bank"] if x["candidate_id"]=="C006")
    if any(abs(float(cand[k])-C006[k])>1e-15 for k in C006) or selection["selected_candidate_id"]!="C006":
        raise RuntimeError("M1_FROZEN_INPUT_MISMATCH")
    spec=importlib.util.spec_from_file_location("frozen_r4_model",MODEL_PATH); mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    candidate=mod.Candidate("C006",**C006); de,dt=make_drive(26081602,4.096)
    frozen=mod.simulate(candidate,de,dt,protocol["model"],protocol["observation_mapping"])
    states,proxy=simulate_open(de,dt)
    state_err=float(np.max(np.abs(states-frozen.states))); proxy_err=float(np.max(np.abs(proxy-frozen.proxy)))
    eq={"state_max_abs_error":state_err,"proxy_max_abs_error":proxy_err,"threshold":1e-12,"passed":state_err<=1e-12 and proxy_err<=1e-12}
    dump("streaming_equivalence.json",eq)
    if not eq["passed"]: raise RuntimeError("STREAMING_EQUIVALENCE_FAILED")
    pll=CausalPLL(); phases=[]; truth=[]
    for i in range(int(20/DT)):
        truth_phase=2*np.pi*.73*i*DT
        phases.append(pll.update(np.sin(truth_phase))); truth.append((truth_phase+np.pi)%(2*np.pi)-np.pi)
    err=np.angle(np.exp(1j*(np.asarray(phases[-2500:])-np.asarray(truth[-2500:]))))
    pll_error=float(abs(np.angle(np.mean(np.exp(1j*err))))*180/np.pi)
    eq["known_sine_pll_error_deg"]=pll_error; dump("streaming_equivalence.json",eq)
    if pll_error>15: raise RuntimeError("PHASE_TRACKER_PREFLIGHT_FAILED")
    input_manifest={"candidate_id":"C006","candidate":cand,"selected_calibration":selection["selected_calibration"],
                    "calibration_use":"diagnostic metadata only; not used in dynamics or feedback","frozen_hashes":hashes,
                    "actuator":"dimensionless additive u_ctrl -> u_E","raw_data_opened":False,
                    "T8_name":"T8_MVP_cortical_state_crossing_rate",
                    "T8_classification":"SAFETY_PROXY_ONLY — NOT a validated dynamical-regime-transition metric"}
    dump("M1_INPUT_MANIFEST.json",input_manifest)


def record(result, extra=None):
    R,d=score(result.control_metrics,result.baseline_metrics)
    row={"seed":result.seed,"condition":result.condition,"stable":result.stable,"R":R,
         "dT2":d["T2"],"dT5":d["T5"],"dT6":d["T6"],"saturation_fraction":result.saturation_fraction,
         "command_range":float(np.ptp(result.command)),"control_energy":float(np.sum(result.command**2)*DT),
         "pulse_count":len(result.pulse_starts)}
    for k,v in result.control_metrics.items(): row[k]=v
    if extra: row.update(extra)
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",default="phase_screen_then_bakeoff"); ap.add_argument("--hard-stop-min",type=float,default=15)
    args=ap.parse_args(); stop=Stop(min(args.hard_stop_min*60,900)); run_count=0
    status={"status":"STARTING","execution_started":True,"pid":os.getpid(),"started_utc":utc(),"run_count":0,"max_runs":33,"evidence_ceiling":"EXPLORATORY_ONLY"}; dump("controller_status.json",status)
    log(f"EXECUTION_STARTED=true PID={os.getpid()} command={' '.join(sys.argv)} planned_simulations_max=33")
    preflight(); log("PREFLIGHT_OK frozen_C006=true streaming_equivalent=true")
    # One-pass provenance check: earlier gains came from a different neurolib/V1 plant and physical ext-current actuator.
    provenance={"decision":"LEGACY_TRANSFER_PID","same_R4_C006_four_state_plant":False,
                "same_dimensionless_u_ctrl_to_u_E_actuator":False,
                "evidence":"Prior frozen gains are located in Phase2AB neurolib/V1 ext_exc_current implementation; no R4/C006 provenance found in authorized code scope.",
                "action":"ONE_LIGHTWEIGHT_COARSE_TUNING_ONLY"}; dump("PID_PROVENANCE_DECISION.json",provenance)
    pid_rows=[]; selected_pid=None
    for name,gains in PID_CANDIDATES.items():
        for seed in DEV[:2]:
            stop.check(); log(f"PID_COARSE_START config={name} seed={seed}")
            rr=simulate(seed,"BASELINE_ADAPTIVE_PID",pid=gains,deadline=stop.check); run_count+=1
            pid_rows.append(record(rr,{"pid_config":name})); log(f"PID_COARSE_DONE config={name} seed={seed} R={pid_rows[-1]['R']:+.4f}")
    valid=[]
    for name,gains in PID_CANDIDATES.items():
        x=[r for r in pid_rows if r["pid_config"]==name]
        if all(r["stable"] for r in x) and np.median([r["R"] for r in x])>0 and sum(r["dT2"]>0 for r in x)>=1 and np.median([r["saturation_fraction"] for r in x])<.5:
            valid.append((float(np.median([r["R"] for r in x])),float(np.median([r["control_energy"] for r in x])),name,gains))
    if valid:
        valid.sort(key=lambda z:(-z[0],z[1],sum(z[3].values()))); _,_,pid_name,selected_pid=valid[0]
        dump("R4_TUNED_PID.json",{"decision":"R4_TUNED_PID","source":"single coarse development-seed pass","name":pid_name,"gains":selected_pid})
        provenance["final_decision"]="R4_TUNED_PID"; provenance["selected_config"]=pid_name
    else:
        provenance["final_decision"]="NO_VALID_R4_PID_BASELINE"; provenance["pid_excluded_from_winner_competition"]=True
    dump("PID_PROVENANCE_DECISION.json",provenance); rows_csv("pid_coarse_tuning.csv",pid_rows)
    # Phase response screen: four phases on one seed, then two frozen-phase confirmations.
    phase_rows=[]; phase_results=[]
    for phase in (-np.pi,-np.pi/2,0.0,np.pi/2):
        stop.check(); rr=simulate(DEV[2],"PHASE_LOCKED_FIXED",phase_target=float(phase),deadline=stop.check); run_count+=1
        phase_results.append(rr); phase_rows.append(record(rr,{"phase_target_rad":float(phase),"stage":"SCREEN"}))
    eligible=[r for r in phase_rows if r["stable"] and r["pulse_count"]>0]
    if not eligible: raise RuntimeError("NO_VALID_PHASE_RESPONSE")
    chosen=max(eligible,key=lambda r:r["R"])["phase_target_rad"]
    for seed in DEV[:2]:
        stop.check(); rr=simulate(seed,"PHASE_LOCKED_FIXED",phase_target=chosen,deadline=stop.check); run_count+=1
        phase_results.append(rr); phase_rows.append(record(rr,{"phase_target_rad":chosen,"stage":"CONFIRM"}))
    chosen_rows=[r for r in phase_rows if r["phase_target_rad"]==chosen]
    phase_ok=sum(r["R"]>0 for r in chosen_rows)>=2
    phase_decision={"selected_phase_rad":chosen,"positive_R_seeds":sum(r["R"]>0 for r in chosen_rows),"development_seed_count":3,
                    "decision":"PHASE_FROZEN_FOR_BAKEOFF" if phase_ok else "NO_VALID_PHASE_RESPONSE"}
    rows_csv("phase_response_screen.csv",phase_rows); dump("PHASE_SELECTION_DECISION.json",phase_decision)
    if not phase_ok: raise RuntimeError("NO_VALID_PHASE_RESPONSE")
    generic_target=1.2*float(np.median([rr.baseline_metrics["T2"] for rr in phase_results]))
    # Final paired bakeoff. Adaptive run precedes energy-matched phase-shifted replay.
    final_rows=[]; traces=[]
    by_seed={}
    for seed in EVAL:
        by_seed[seed]={}
        conditions=["SHAM","FIXED","PHASE_LOCKED_FIXED","PHASE_LOCKED_ADAPTIVE"]
        if selected_pid: conditions += ["GENERIC_PID","BASELINE_ADAPTIVE_PID"]
        for cond in conditions:
            stop.check(); log(f"BAKEOFF_START seed={seed} condition={cond}")
            rr=simulate(seed,cond,phase_target=chosen,pid=selected_pid,generic_target=generic_target,deadline=stop.check); run_count+=1; by_seed[seed][cond]=rr
            log(f"BAKEOFF_DONE seed={seed} condition={cond} stable={rr.stable}")
        adaptive=by_seed[seed]["PHASE_LOCKED_ADAPTIVE"]
        rr=simulate(seed,"PHASE_SHIFTED",phase_target=chosen+np.pi,replay=(adaptive.pulse_starts,adaptive.pulse_amplitudes),deadline=stop.check); run_count+=1; by_seed[seed]["PHASE_SHIFTED"]=rr
        energy_a=float(np.sum(adaptive.command**2)*DT); energy_b=float(np.sum(rr.command**2)*DT)
        if (len(rr.pulse_starts)!=len(adaptive.pulse_starts)
                or not np.allclose(sorted(rr.pulse_amplitudes),sorted(adaptive.pulse_amplitudes),rtol=0,atol=0)
                or abs(energy_a-energy_b)>1e-12):
            raise RuntimeError("PHASE_CONTROL_MATCH_FAILED")
        for cond,result in by_seed[seed].items():
            sham=by_seed[seed]["SHAM"].control_metrics
            R,d=score(result.control_metrics,sham)
            row={"seed":seed,"condition":cond,"stable":result.stable,"R":R,"dT2":d["T2"],"dT5":d["T5"],"dT6":d["T6"],
                 "saturation_fraction":result.saturation_fraction,"command_range":float(np.ptp(result.command)),
                 "control_energy":float(np.sum(result.command**2)*DT),"pulse_count":len(result.pulse_starts)}
            row.update(result.control_metrics); final_rows.append(row)
            stride=int(1/DT)
            for i in range(int(15/DT),len(result.proxy),stride):
                traces.append({"seed":seed,"condition":cond,"time_s":i*DT,"proxy":result.proxy[i],"u_ctrl":result.command[i]})
    if run_count>33: raise RuntimeError("RUN_BUDGET_EXCEEDED")
    rows_csv("controller_runs.csv",final_rows); rows_csv("controller_traces.csv",traces)
    # 8-target table and candidate eligibility.
    eval_rows=[]; candidates=sorted({r["condition"] for r in final_rows if r["condition"]!="SHAM"}); eligible_names=[]
    for cond in candidates:
        rs=[r for r in final_rows if r["condition"]==cond]
        gate={"stable_3of3":sum(r["stable"] for r in rs)==3,"T2":np.median([r["dT2"] for r in rs])>=.10 and sum(r["dT2"]>0 for r in rs)>=2,
              "T5_T6":np.median([r["dT5"] for r in rs])>=0 and np.median([r["dT6"] for r in rs])>=0,
              "R":np.median([r["R"] for r in rs])>=.10,"T1":all(.5<=r["T1"]<=1.25 for r in rs),
              "T8":all(r["T8_MVP_cortical_state_crossing_rate"] < 2*next(x["T8_MVP_cortical_state_crossing_rate"] for x in final_rows if x["seed"]==r["seed"] and x["condition"]=="SHAM") for r in rs)}
        if "PID" in cond or "ADAPTIVE" in cond: gate["saturation"]=sum(r["saturation_fraction"]<.5 for r in rs)>=2
        if all(gate.values()): eligible_names.append(cond)
        for target in ("T1","T2","T3","T4","T5","T6","T7","T8_MVP_cortical_state_crossing_rate"):
            sh=[next(x[target] for x in final_rows if x["seed"]==r["seed"] and x["condition"]=="SHAM") for r in rs]
            effects=[(r[target]-s)/max(abs(s),1e-4 if target=="T6" else 1e-12) for r,s in zip(rs,sh)]
            eval_rows.append({"condition":cond,"target":target,"median_relative_effect":float(np.median(effects)),"gate_passed":all(gate.values()),
                              "classification":"SAFETY_PROXY_ONLY — NOT a validated dynamical-regime-transition metric" if target.startswith("T8") else "EXPLORATORY_TARGET"})
    rows_csv("controller_8target_evaluation.csv",eval_rows)
    med={c:float(np.median([r["R"] for r in final_rows if r["condition"]==c])) for c in candidates}
    energy={c:float(np.median([r["control_energy"] for r in final_rows if r["condition"]==c])) for c in candidates}
    pool=[c for c in eligible_names if selected_pid or "PID" not in c]
    winner=max(pool,key=lambda c:(med[c],-energy[c])) if pool else None
    phase_adv=med.get("PHASE_LOCKED_ADAPTIVE",-99)-med.get("PHASE_SHIFTED",-99)
    if winner and winner.startswith("PHASE_LOCKED"):
        decision="PHASE_LOCKED_CONTROLLER_PROVISIONALLY_SELECTED" if selected_pid else "PHASE_LOCKED_CONTROLLER_PROVISIONALLY_SELECTED_WITHOUT_VALID_PID_COMPARATOR"
    elif winner and "PID" in winner: decision="PID_REMAINS_PROVISIONAL_CONTROLLER"
    elif winner=="FIXED": decision="FIXED_STIMULATION_PROVISIONALLY_SELECTED"
    elif winner: decision="MODULATION_WITHOUT_CONTROLLER_SPECIFIC_ADVANTAGE"
    else: decision="NO_CONTROLLER_MET_MVP_GATE"
    out={"decision":decision,"winner":winner,"eligible_controllers":eligible_names,"median_R":med,"median_energy":energy,
         "phase_specific_advantage_ge_5pp":phase_adv>=.05,"phase_advantage_R":phase_adv,
         "pid_baseline":provenance["final_decision"],"run_count":run_count,"max_runs":33,"evidence_ceiling":"EXPLORATORY_ONLY"}
    if not selected_pid: out["pid_comparison_limitation"]="本MVP未获得有效PID baseline，因此无法完成与有效PID的正式性能比较。"
    dump("CONTROLLER_SELECTION_DECISION.json",out)
    # Compact quicklook.
    fig,ax=plt.subplots(1,2,figsize=(11,4)); order=candidates
    ax[0].bar(range(len(order)),[med[c] for c in order]); ax[0].axhline(.1,color="k",ls="--",lw=1); ax[0].set_xticks(range(len(order)),order,rotation=35,ha="right"); ax[0].set_ylabel("Median R")
    ax[1].bar(range(len(order)),[energy[c] for c in order]); ax[1].set_xticks(range(len(order)),order,rotation=35,ha="right"); ax[1].set_ylabel("Median control energy")
    fig.suptitle("C006 Controller MVP — EXPLORATORY_ONLY"); fig.tight_layout(); fig.savefig(ROOT/"controller_quicklook.png",dpi=170); plt.close(fig)
    summary=f"""# C006 Controller MVP Summary\n\n- Decision: `{decision}`\n- Provisional winner: `{winner}`\n- Simulations: `{run_count}/33`; all within the 900-second hard stop.\n- PID baseline: `{provenance['final_decision']}`\n- Phase-specific adaptive advantage over matched shifted control: `{phase_adv:+.1%}`.\n- Evidence ceiling: `EXPLORATORY_ONLY`.\n\n{out.get('pid_comparison_limitation','A valid R4 PID baseline was available for exploratory comparison.')}\n\nT8 is `T8_MVP_cortical_state_crossing_rate`: **SAFETY_PROXY_ONLY — NOT a validated dynamical-regime-transition metric**. C006 remains a frozen discrete mechanistic candidate and this result is not clinical, confirmatory, C1/C2-certified, or evidence that real patient sleep improved.\n"""
    (ROOT/"CONTROLLER_MVP_SUMMARY.md").write_text(summary,encoding="utf-8")
    status.update({"status":"COMPLETE","decision":decision,"run_count":run_count,"elapsed_seconds":time.monotonic()-stop.started}); dump("controller_status.json",status)
    log(f"RUN_COMPLETE decision={decision} runs={run_count} elapsed_s={status['elapsed_seconds']:.2f}")


if __name__=="__main__":
    try: main()
    except Exception as exc:
        log(f"RUN_FAILED blocker={exc}"); dump("CONTROLLER_SELECTION_DECISION.json",{"decision":"STOPPED","single_blocker":str(exc),"evidence_ceiling":"EXPLORATORY_ONLY"})
        traceback.print_exc(); raise
