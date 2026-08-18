from __future__ import annotations

import argparse, csv, hashlib, importlib.metadata, importlib.util, json, os, platform, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurolib, numba, numpy as np, pandas, scipy

from c006_phase_controller import (C006, CONTROL_SAMPLES, CONTROL_START, CausalPLL, DT, FS,
    PhaseVocoder, build_yoked, make_drive, metrics, score, simulate, simulate_open, validate_yoked)

ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parents[1]
R4=WORKSPACE/"BigBang_V4D1"/"COSTA_MVP_FRESH_R4_20260816_023933"
PROTOCOL=R4/"CODE"/"FROZEN_V2"/"protocol_v2.json"; MODEL=R4/"CODE"/"MECHANISTIC_V2"/"model.py"
SELECTION=R4/"RUNS"/"FIT_WORKER"/"SELECTION_FREEZE.json"
EXPECTED={str(PROTOCOL):"338366593A2A4C8626349D70AA38306C5931F0FF4D22FD96DB546DD41505E3C3",
          str(MODEL):"775A3961093C386A1E7DF8F0D033963EAB684F2566AE206E5BB6DB4A6CD03869",
          str(SELECTION):"3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF"}
DEV=(44017,55021,66029); EVAL=(77131,88141,99149)


def utc(): return datetime.now(timezone.utc).isoformat()
def log(s):
    line=f"[{utc()}] {s}"; print(line,flush=True)
    with (ROOT/"RUN_LOG.txt").open("a",encoding="utf-8") as f: f.write(line+"\n")
def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False,default=_json_default),encoding="utf-8")
def write_csv(name,rows):
    if not rows:return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys:keys.append(k)
    with (ROOT/name).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def ver(name,mod):return getattr(mod,"__version__",None) or importlib.metadata.version(name)


class HardStop:
    def __init__(self,seconds):self.started=time.monotonic();self.deadline=self.started+seconds
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError("900_SECOND_HARD_STOP")


def environment_manifest():
    return {"created_utc":utc(),"conda_default_env":os.getenv("CONDA_DEFAULT_ENV"),"conda_prefix":os.getenv("CONDA_PREFIX"),
      "python_executable":sys.executable,"python_version":platform.python_version(),
      "versions":{"neurolib":ver("neurolib",neurolib),"numpy":np.__version__,"scipy":scipy.__version__,
                  "pandas":pandas.__version__,"matplotlib":matplotlib.__version__,"numba":numba.__version__},
      "evidence_ceiling":"EXPLORATORY_ONLY"}


def circular_error(est,truth):
    return np.angle(np.exp(1j*(np.asarray(est)-np.asarray(truth))))


def estimator_preflight():
    duration=20.0; n=int(duration/DT); t=np.arange(n)*DT; truth=2*np.pi*.73*t
    pll=CausalPLL(); pv=PhaseVocoder(); pe=[];ve=[];valid=[]
    for i in range(n):
        pe.append(pll.update(np.sin(truth[i]))[0]); out=pv.update(np.sin(truth[i]));ve.append(out[0]);valid.append(out[4])
    keep=t>=10
    pll_err=float(abs(np.angle(np.mean(np.exp(1j*circular_error(np.asarray(pe)[keep],truth[keep])))))*180/np.pi)
    pv_err=float(abs(np.angle(np.mean(np.exp(1j*circular_error(np.asarray(ve)[keep],truth[keep])))))*180/np.pi)
    duration=30.; n=int(duration/DT); t=np.arange(n)*DT; rate=(1.25-.5)/duration
    chirp_truth=2*np.pi*(.5*t+.5*rate*t*t); pv2=PhaseVocoder(); ce=[]
    for i in range(n):ce.append(pv2.update(np.sin(chirp_truth[i]))[0])
    err=np.abs(circular_error(np.asarray(ce)[t>=8],chirp_truth[t>=8]))*180/np.pi
    chirp_median=float(np.median(err))
    pv3=PhaseVocoder(); low_valid=[]
    for i in range(int(12/DT)):low_valid.append(pv3.update(.001*np.sin(2*np.pi*.73*i*DT))[4])
    low_ratio=float(np.mean(low_valid[int(4/DT):]))
    # Worst-case circular-boundary scheduler invariant, including a pulse split at the window edge.
    source=np.zeros(CONTROL_SAMPLES); width=int(.11/DT)
    starts=[25,CONTROL_SAMPLES-width-3]
    for s,a in zip(starts,[.017,.043]):source[s:s+width]=a
    shifted=build_yoked(source,347); schedule=validate_yoked(source,shifted,len(starts))
    checks={"pll_fixed_sine_error_deg":pll_err,"pv_fixed_sine_error_deg":pv_err,"pv_chirp_median_error_deg":chirp_median,
            "pv_low_amplitude_valid_ratio":low_ratio,"scheduler_boundary_test":schedule,
            "limits":{"fixed_sine_deg":15,"pv_chirp_median_deg":20,"low_amplitude_valid_ratio":.9}}
    checks["pll_passed"]=pll_err<=15; checks["pv_passed"]=pv_err<=15 and chirp_median<=20 and low_ratio>=.9
    checks["scheduler_passed"]=schedule["passed"]
    dump("phase_estimator_preflight.json",checks)
    if not checks["pll_passed"]:raise RuntimeError("PLL_PREFLIGHT_FAILED")
    if not checks["pv_passed"]:raise RuntimeError("PHASE_VOCODER_PREFLIGHT_FAILED")
    if not checks["scheduler_passed"]:raise RuntimeError("PHASE_CONTROL_MATCH_FAILED")


def preflight():
    env=environment_manifest();dump("environment_manifest.json",env)
    exact=str(Path(r"C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe").resolve()).lower()
    if env["conda_default_env"]!="neurolib" or str(Path(sys.executable).resolve()).lower()!=exact or env["versions"]["neurolib"]!="0.6.1":
        raise RuntimeError("NEUROLIB_ENVIRONMENT_MISMATCH")
    hashes={p:sha(Path(p)) for p in EXPECTED}
    if hashes!=EXPECTED:raise RuntimeError("M1_FROZEN_INPUT_MISMATCH")
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"));selection=json.loads(SELECTION.read_text(encoding="utf-8"))
    cand=next(x for x in protocol["candidate_bank"] if x["candidate_id"]=="C006")
    if selection["selected_candidate_id"]!="C006" or any(abs(float(cand[k])-C006[k])>1e-15 for k in C006):raise RuntimeError("M1_FROZEN_INPUT_MISMATCH")
    spec=importlib.util.spec_from_file_location("r4_frozen_model",MODEL);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    de,dt=make_drive(26081602,4.096);frozen=mod.simulate(mod.Candidate("C006",**C006),de,dt,protocol["model"],protocol["observation_mapping"])
    states,proxy=simulate_open(de,dt);se=float(np.max(np.abs(states-frozen.states)));pe=float(np.max(np.abs(proxy-frozen.proxy)))
    eq={"state_max_abs_error":se,"proxy_max_abs_error":pe,"threshold":1e-12,"passed":se<=1e-12 and pe<=1e-12};dump("streaming_equivalence.json",eq)
    if not eq["passed"]:raise RuntimeError("STREAMING_EQUIVALENCE_FAILED")
    dump("M1_INPUT_MANIFEST.json",{"candidate_id":"C006","candidate":cand,"selected_calibration":selection["selected_calibration"],
      "calibration_use":"diagnostic metadata only; not used in dynamics or feedback","frozen_hashes":hashes,
      "actuator":"dimensionless additive u_ctrl -> cortical u_E","raw_data_opened":False,
      "T8_name":"T8_MVP_cortical_state_crossing_rate","T8_classification":"SAFETY_PROXY_ONLY — NOT a validated dynamical-regime-transition metric"})
    estimator_preflight()


def internal_row(rr,extra=None):
    R,d=score(rr.control_metrics,rr.baseline_metrics)
    row={"seed":rr.seed,"condition":rr.condition,"stable":rr.stable,"R":R,"dT2":d["T2"],"dT5":d["T5"],"dT6":d["T6"],
         "pulse_count":rr.logical_pulse_count,"saturation_fraction":rr.saturation_fraction,
         "command_range":float(np.ptp(rr.command[CONTROL_START:])),"control_energy":float(np.sum(rr.command[CONTROL_START:]**2)*DT),
         "realized_pulse_phase_rad":rr.realized_pulse_phase_rad,**rr.control_metrics}
    if extra:row.update(extra)
    return row


def paired_row(rr,sham):
    R,d=score(rr.control_metrics,sham.control_metrics)
    return {"seed":rr.seed,"condition":rr.condition,"stable":rr.stable,"R":R,"dT2":d["T2"],"dT5":d["T5"],"dT6":d["T6"],
      "pulse_count":rr.logical_pulse_count,"saturation_fraction":rr.saturation_fraction,
      "command_range":float(np.ptp(rr.command[CONTROL_START:])),"control_energy":float(np.sum(rr.command[CONTROL_START:]**2)*DT),
      "realized_pulse_phase_rad":rr.realized_pulse_phase_rad,**rr.control_metrics}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",default="pv_phase_screen_then_bakeoff",choices=["pv_phase_screen_then_bakeoff"]);ap.add_argument("--hard-stop-min",type=float,default=15)
    args=ap.parse_args();stop=HardStop(min(900,args.hard_stop_min*60));run_count=0
    status={"status":"STARTING","execution_started":True,"pid":os.getpid(),"started_utc":utc(),"planned_runs":24,"evidence_ceiling":"EXPLORATORY_ONLY"};dump("controller_status.json",status)
    log(f"EXECUTION_STARTED=true PID={os.getpid()} command={' '.join(sys.argv)} planned_simulations=24")
    preflight();log("PREFLIGHT_OK C006=true streaming=true PLL=true PV=true yoked_scheduler=true")
    dump("PID_BASELINE_STATUS.json",{"decision":"NO_VALID_R4_PID_BASELINE","source":"prior C006 controller MVP negative result","pid_rerun":False,
      "limitation":"本MVP未获得有效PID baseline，因此无法完成与有效PID的正式性能比较。"})
    # Six-run prospective PV phase screen.
    phase_rows=[];phase_results=[]
    for phase in (-np.pi,-np.pi/2,0.0,np.pi/2):
        stop.check();log(f"PHASE_SCREEN_START seed={DEV[2]} phase={phase:+.6f}")
        rr=simulate(DEV[2],"PV_PHASE_LOCKED",estimator_kind="PV",phase_target=float(phase),deadline=stop.check);run_count+=1
        phase_results.append(rr);phase_rows.append(internal_row(rr,{"phase_target_rad":float(phase),"stage":"SCREEN"}))
    eligible=[r for r in phase_rows if r["stable"] and r["pulse_count"]>=5]
    if not eligible:raise RuntimeError("NO_VALID_PV_PHASE_RESPONSE")
    chosen=float(max(eligible,key=lambda r:r["R"])["phase_target_rad"])
    for seed in DEV[:2]:
        stop.check();log(f"PHASE_CONFIRM_START seed={seed} phase={chosen:+.6f}")
        rr=simulate(seed,"PV_PHASE_LOCKED",estimator_kind="PV",phase_target=chosen,deadline=stop.check);run_count+=1
        phase_results.append(rr);phase_rows.append(internal_row(rr,{"phase_target_rad":chosen,"stage":"CONFIRM"}))
    chosen_rows=[r for r in phase_rows if r["phase_target_rad"]==chosen]
    phase_ok=all(r["stable"] and r["pulse_count"]>=5 for r in chosen_rows) and sum(r["R"]>0 for r in chosen_rows)>=2
    phase_dec={"decision":"PV_PHASE_FROZEN_FOR_BAKEOFF" if phase_ok else "NO_VALID_PV_PHASE_RESPONSE","selected_phase_rad":chosen,
               "positive_R_seeds":sum(r["R"]>0 for r in chosen_rows),"development_seed_count":3}
    write_csv("phase_response_screen.csv",phase_rows);dump("PHASE_SELECTION_DECISION.json",phase_dec)
    if not phase_ok:raise RuntimeError("NO_VALID_PV_PHASE_RESPONSE")
    # Eighteen-run evaluation; sham first freezes per-seed half-cycle shift.
    all_results={};final_rows=[];traces=[];yoke_checks=[];phase_pairs=[]
    for seed in EVAL:
        stop.check();all_results[seed]={};log(f"BAKEOFF_START seed={seed} condition=SHAM")
        sham=simulate(seed,"SHAM",deadline=stop.check);run_count+=1;all_results[seed]["SHAM"]=sham
        half=int(round(FS/(2*sham.control_metrics["T1"])))
        for cond,kind in (("FIXED","PV"),("PLL_PHASE_LOCKED","PLL"),("PV_PHASE_LOCKED","PV"),("PV_PHASE_LOCKED_ADAPTIVE","PV")):
            stop.check();log(f"BAKEOFF_START seed={seed} condition={cond}")
            rr=simulate(seed,cond,estimator_kind=kind,phase_target=chosen,deadline=stop.check);run_count+=1;all_results[seed][cond]=rr
        adaptive=all_results[seed]["PV_PHASE_LOCKED_ADAPTIVE"];source=adaptive.command[CONTROL_START:].copy();shifted=build_yoked(source,half)
        check=validate_yoked(source,shifted,adaptive.logical_pulse_count);check.update({"seed":seed,"half_cycle_samples":half,"sham_T1_hz":sham.control_metrics["T1"]})
        yoke_checks.append(check);dump("yoked_control_validation.json",{"checks":yoke_checks,"all_passed":all(x["passed"] for x in yoke_checks)})
        if not check["passed"]:raise RuntimeError("PHASE_CONTROL_MATCH_FAILED")
        shifted_starts=[(s-CONTROL_START+half)%CONTROL_SAMPLES for s in adaptive.pulse_starts]
        stop.check();log(f"BAKEOFF_START seed={seed} condition=PV_PHASE_SHIFTED_YOKED")
        yoked=simulate(seed,"PV_PHASE_SHIFTED_YOKED",replay_command=shifted,replay_pulse_count=adaptive.logical_pulse_count,
                       replay_event_starts=shifted_starts,deadline=stop.check);run_count+=1;all_results[seed]["PV_PHASE_SHIFTED_YOKED"]=yoked
        if not np.array_equal(yoked.command[CONTROL_START:],shifted):raise RuntimeError("PHASE_CONTROL_MATCH_FAILED")
        sep=float(abs(np.angle(np.exp(1j*(yoked.realized_pulse_phase_rad-adaptive.realized_pulse_phase_rad))))*180/np.pi)
        phase_pairs.append({"seed":seed,"adaptive_phase_rad":adaptive.realized_pulse_phase_rad,"shifted_phase_rad":yoked.realized_pulse_phase_rad,"separation_deg":sep,"phase_separation_valid":120<=sep<=240})
        for cond,rr in all_results[seed].items():
            final_rows.append(paired_row(rr,sham))
            for i in range(CONTROL_START,len(rr.proxy),int(FS)):
                traces.append({"seed":seed,"condition":cond,"time_s":i*DT,"proxy":rr.proxy[i],"u_ctrl":rr.command[i]})
    if run_count!=24:raise RuntimeError(f"RUN_COUNT_MISMATCH_{run_count}")
    write_csv("controller_runs.csv",final_rows);write_csv("controller_traces.csv",traces)
    dump("yoked_control_validation.json",{"checks":yoke_checks,"phase_pairs":phase_pairs,"all_passed":all(x["passed"] for x in yoke_checks)})
    # Fixed gates and winner selection.
    candidates=["FIXED","PLL_PHASE_LOCKED","PV_PHASE_LOCKED","PV_PHASE_LOCKED_ADAPTIVE","PV_PHASE_SHIFTED_YOKED"]
    gates={};eval_rows=[]
    for cond in candidates:
        rs=[r for r in final_rows if r["condition"]==cond]
        gate={"stable_3of3":sum(r["stable"] for r in rs)==3,
          "T2":np.median([r["dT2"] for r in rs])>=.10 and sum(r["dT2"]>0 for r in rs)>=2,
          "T5_T6":np.median([r["dT5"] for r in rs])>=0 and np.median([r["dT6"] for r in rs])>=0,
          "R":np.median([r["R"] for r in rs])>=.10,"T1":all(.5<=r["T1"]<=1.25 for r in rs),
          "T8":all(r["T8_MVP_cortical_state_crossing_rate"]<2*next(x["T8_MVP_cortical_state_crossing_rate"] for x in final_rows if x["seed"]==r["seed"] and x["condition"]=="SHAM") for r in rs)}
        if cond=="PV_PHASE_LOCKED_ADAPTIVE":gate["saturation"]=sum(r["saturation_fraction"]<.5 for r in rs)>=2
        gates[cond]=gate
        for target in ("T1","T2","T3","T4","T5","T6","T7","T8_MVP_cortical_state_crossing_rate"):
            effects=[]
            for r in rs:
                s=next(x[target] for x in final_rows if x["seed"]==r["seed"] and x["condition"]=="SHAM")
                effects.append((r[target]-s)/max(abs(s),1e-4 if target=="T6" else 1e-12))
            eval_rows.append({"condition":cond,"target":target,"median_relative_effect":float(np.median(effects)),"controller_gate_passed":all(gate.values()),
              "classification":"SAFETY_PROXY_ONLY — NOT a validated dynamical-regime-transition metric" if target.startswith("T8") else "EXPLORATORY_TARGET"})
    write_csv("controller_8target_evaluation.csv",eval_rows)
    med={c:float(np.median([r["R"] for r in final_rows if r["condition"]==c])) for c in candidates}
    energy={c:float(np.median([r["control_energy"] for r in final_rows if r["condition"]==c])) for c in candidates}
    eligible=[c for c in candidates if all(gates[c].values())];winner=max(eligible,key=lambda c:(med[c],-energy[c])) if eligible else None
    if winner=="PV_PHASE_LOCKED_ADAPTIVE" and "PLL_PHASE_LOCKED" in eligible:
        replace=(med[winner]>=med["PLL_PHASE_LOCKED"]+.05) or (abs(med[winner]-med["PLL_PHASE_LOCKED"])<=.05 and energy[winner]<=.7*energy["PLL_PHASE_LOCKED"])
        if not replace:winner="PLL_PHASE_LOCKED"
    phase_adv=med["PV_PHASE_LOCKED_ADAPTIVE"]-med["PV_PHASE_SHIFTED_YOKED"]
    phase_sep_ok=sum(p["phase_separation_valid"] for p in phase_pairs)>=2
    phase_claim=bool(phase_adv>=.05 and phase_sep_ok)
    if winner=="PV_PHASE_LOCKED_ADAPTIVE":decision="PV_ADAPTIVE_CONTROLLER_PROVISIONALLY_SELECTED" if phase_claim else "MODULATION_WITHOUT_PHASE_SPECIFIC_ADVANTAGE"
    elif winner=="PV_PHASE_LOCKED":decision="PV_FIXED_PHASE_CONTROLLER_PROVISIONALLY_SELECTED"
    elif winner=="PLL_PHASE_LOCKED":decision="PLL_REMAINS_PROVISIONAL_PHASE_CONTROLLER"
    elif winner=="FIXED":decision="FIXED_STIMULATION_PROVISIONALLY_SELECTED"
    elif winner:decision="MODULATION_WITHOUT_PHASE_SPECIFIC_ADVANTAGE"
    else:decision="NO_CONTROLLER_MET_MVP_GATE"
    result={"decision":decision,"winner":winner,"eligible_controllers":eligible,"controller_gates":gates,"median_R":med,"median_energy":energy,
      "phase_adaptive_vs_yoked_R_advantage":phase_adv,"phase_separation_valid_2of3":phase_sep_ok,"phase_specific_advantage_claim_allowed":phase_claim,
      "pid_baseline":"NO_VALID_R4_PID_BASELINE","pid_comparison_limitation":"本MVP未获得有效PID baseline，因此无法完成与有效PID的正式性能比较。",
      "run_count":run_count,"planned_runs":24,"evidence_ceiling":"EXPLORATORY_ONLY"};dump("CONTROLLER_SELECTION_DECISION.json",result)
    fig,ax=plt.subplots(1,2,figsize=(12,4));x=np.arange(len(candidates))
    ax[0].bar(x,[med[c] for c in candidates]);ax[0].axhline(.1,color="k",ls="--");ax[0].set_ylabel("Median R")
    ax[1].bar(x,[energy[c] for c in candidates]);ax[1].set_ylabel("Median control energy")
    for a in ax:a.set_xticks(x,candidates,rotation=35,ha="right")
    fig.suptitle("C006 literature-driven phase controller MVP — EXPLORATORY_ONLY");fig.tight_layout();fig.savefig(ROOT/"controller_quicklook.png",dpi=170);plt.close(fig)
    summary=f"""# C006 PV Controller MVP Summary\n\n- Decision: `{decision}`\n- Provisional winner: `{winner}`\n- Completed simulations: `{run_count}/24`\n- Selected PV phase: `{chosen:+.6f} rad`\n- PV adaptive vs exactly yoked phase-shifted median R advantage: `{phase_adv:+.1%}`\n- Phase-specific claim allowed: `{phase_claim}`\n- PID baseline: `NO_VALID_R4_PID_BASELINE`\n- Evidence ceiling: `EXPLORATORY_ONLY`\n\n本MVP未获得有效PID baseline，因此无法完成与有效PID的正式性能比较。\n\nT8 is `T8_MVP_cortical_state_crossing_rate`: **SAFETY_PROXY_ONLY — NOT a validated dynamical-regime-transition metric**. C006 is a frozen discrete mechanistic candidate; these model-internal results are not clinical, confirmatory, C1/C2-certified, or evidence of improved real-patient sleep.\n"""
    (ROOT/"CONTROLLER_MVP_SUMMARY.md").write_text(summary,encoding="utf-8")
    status.update({"status":"COMPLETE","decision":decision,"run_count":run_count,"elapsed_seconds":time.monotonic()-stop.started});dump("controller_status.json",status)
    log(f"RUN_COMPLETE decision={decision} winner={winner} runs={run_count} elapsed_s={status['elapsed_seconds']:.2f}")


if __name__=="__main__":
    try:main()
    except Exception as exc:
        log(f"RUN_FAILED blocker={exc}");dump("CONTROLLER_SELECTION_DECISION.json",{"decision":"STOPPED","single_blocker":str(exc),"evidence_ceiling":"EXPLORATORY_ONLY"})
        dump("controller_status.json",{"status":"STOPPED","single_blocker":str(exc),"evidence_ceiling":"EXPLORATORY_ONLY"})
        traceback.print_exc();raise
