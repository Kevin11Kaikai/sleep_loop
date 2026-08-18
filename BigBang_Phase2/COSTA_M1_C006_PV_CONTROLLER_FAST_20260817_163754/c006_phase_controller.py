from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch

DT = 0.002
FS = 500.0
STATE0 = np.array([0.18, 0.16, 0.14, 0.12], dtype=float)
TAUS = np.array([0.018, 0.012, 0.028, 0.016], dtype=float)
WEIGHTS = np.array([1.0, -0.55, 0.35, -0.2], dtype=float)
C006 = dict(cortical_gain=0.82, cortex_to_thalamus=0.95,
            thalamus_to_cortex=1.10, relay_reticular_inhibition=1.25)
CONTROL_START = int(15.0 / DT)
CONTROL_SAMPLES = int(60.0 / DT)


def make_drive(seed: int, duration_s: float = 75.0) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(duration_s / DT)); t = np.arange(n) * DT
    rng = np.random.default_rng(seed); pe, pt, pc = rng.uniform(0, 2*np.pi, 3)
    e = 0.34 + 0.18*np.sin(2*np.pi*.73*t + pe)
    e += 0.32*(np.mod(t + pe/(2*np.pi*.73), .91) < .11)
    gate = np.mod(t + pt/(2*np.pi*1.17), 1.37) < .42
    th = .23 + .13*np.sin(2*np.pi*1.17*t + pt) + .20*np.sin(2*np.pi*11*t + pc)*gate
    return e.astype(float), th.astype(float)


class CausalPLL:
    def __init__(self, nominal_hz: float = .73):
        self.phase = 0.0; self.freq = nominal_hz; self.amp = 1e-3
        self.kp = .035; self.ki = .00008; self.confidence = 0.0; self.valid = False

    def update(self, x: float) -> tuple[float, float, float, float, bool]:
        self.amp = .999*self.amp + .001*abs(x)
        xn = float(np.clip(x/max(2*self.amp, 1e-6), -2, 2))
        err = xn*math.cos(self.phase)
        omega = float(np.clip(2*np.pi*self.freq + self.ki*err, 2*np.pi*.5, 2*np.pi*1.25))
        self.freq = omega/(2*np.pi)
        self.phase = (self.phase + omega*DT + self.kp*err + np.pi)%(2*np.pi)-np.pi
        self.confidence = float(np.clip(self.amp/.01, 0, 1)); self.valid = True
        return self.phase, self.freq, self.confidence, 2*self.amp, self.valid


class PhaseVocoder:
    """Causal short-time spectral phase estimator with peak interpolation."""
    def __init__(self, window_s: float = 4.0, update_s: float = .1):
        self.n = int(window_s*FS); self.hop = int(update_s*FS)
        self.buffer = np.zeros(self.n); self.pos = 0; self.count = 0; self.since = 0
        self.window = np.hanning(self.n); self.freq = .73; self.phase = 0.0
        self.confidence = 0.0; self.amplitude = 0.0; self.valid = False

    def update(self, x: float) -> tuple[float, float, float, float, bool]:
        self.buffer[self.pos] = x; self.pos = (self.pos + 1) % self.n
        self.count += 1; self.since += 1
        self.phase = (self.phase + 2*np.pi*self.freq*DT + np.pi)%(2*np.pi)-np.pi
        if self.count >= self.n and self.since >= self.hop:
            self.since = 0
            ordered = np.r_[self.buffer[self.pos:], self.buffer[:self.pos]]
            y = (ordered - np.mean(ordered))*self.window
            spec = np.fft.rfft(y); freqs = np.fft.rfftfreq(self.n, DT)
            ids = np.flatnonzero((freqs >= .5) & (freqs <= 1.25))
            mags = np.abs(spec[ids]); local = int(np.argmax(mags)); k = int(ids[local])
            delta = 0.0
            if 0 < k < len(spec)-1:
                a,b,c = np.log(np.abs(spec[[k-1,k,k+1]]) + 1e-18)
                den = a - 2*b + c
                if abs(den) > 1e-12: delta = float(np.clip(.5*(a-c)/den, -.5, .5))
            self.freq = float(np.clip((k+delta)*FS/self.n, .5, 1.25))
            t = np.arange(self.n)/FS
            coeff = np.sum(y*np.exp(-1j*2*np.pi*self.freq*t))
            self.phase = float((np.angle(coeff)+np.pi/2+2*np.pi*self.freq*t[-1]+np.pi)%(2*np.pi)-np.pi)
            self.amplitude = float(2*abs(coeff)/max(np.sum(self.window), 1e-12))
            self.confidence = float(mags[local]/max(np.sqrt(np.sum(mags*mags)), 1e-12))
            self.valid = bool(np.isfinite(self.phase) and np.isfinite(self.freq))
        return self.phase, self.freq, self.confidence, self.amplitude, self.valid


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-np.clip(3*np.asarray(x), -60, 60)))


def simulate_open(drive_e: np.ndarray, drive_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n=len(drive_e); states=np.empty((n,4)); states[0]=STATE0
    source=np.empty(n); source[0]=states[0]@WEIGHTS; base=source[0]; proxy=np.zeros(n)
    for i in range(1,n):
        e,inh,th,ret=states[i-1]; ue=drive_e[i-1]; ut=drive_t[i-1]
        inp=np.array([.82*(2.8*e-2.2*inh)+1.10*1.8*th+ue-.8,
                      .82*(2.2*e-.6*inh)+1.10*.7*th+.7*ue-.65,
                      .95*1.6*e-1.25*ret+ut-.4,
                      .95*e+1.8*th-.3*ret+.5*ut-.55])
        states[i]=np.clip(states[i-1]+DT/TAUS*(sigmoid(inp)-states[i-1]),0,1)
        source[i]=states[i]@WEIGHTS; base+=(DT/.160)*(source[i]-base); proxy[i]=100*(source[i]-base)
    return states,proxy


def metrics(proxy: np.ndarray, cortical_e: np.ndarray) -> dict:
    x=np.asarray(proxy,float); centered=x-np.mean(x); nperseg=min(len(x),int(8*FS))
    f,p=welch(centered,fs=FS,window="hann",nperseg=nperseg,noverlap=min(int(4*FS),nperseg-1))
    sm=(f>=.5)&(f<=1.25); sigm=(f>=10)&(f<=15)
    so=sosfiltfilt(butter(4,[.5,1.25],btype="bandpass",fs=FS,output="sos"),centered)
    sigma=sosfiltfilt(butter(4,[10,15],btype="bandpass",fs=FS,output="sos"),centered)
    phase=np.angle(hilbert(so)); env=np.abs(hilbert(sigma)); vec=np.sum(env*np.exp(1j*phase))/max(float(np.sum(env)),1e-12)
    duration=len(x)/FS/60; troughs,_=find_peaks(-so,distance=int(.5*FS),prominence=max(.25*np.std(so),1e-12))
    rms=np.sqrt(uniform_filter1d(sigma*sigma,size=int(.2*FS),mode="nearest")); threshold=np.mean(rms)+1.5*np.std(rms)
    changes=np.diff(np.r_[False,rms>threshold,False].astype(np.int8)); starts=np.flatnonzero(changes==1); ends=np.flatnonzero(changes==-1)
    spindles=sum(.5 <= (b-a)/FS <= 3 for a,b in zip(starts,ends))
    smooth=uniform_filter1d(np.asarray(cortical_e,float),size=int(.25*FS),mode="nearest"); lo,hi=np.quantile(smooth,[.35,.65])
    state=0; crossings=0
    for val in smooth:
        nxt=1 if val>hi else (-1 if val<lo else state)
        if state and nxt!=state: crossings+=1
        state=nxt
    return {"T1":float(f[sm][np.argmax(p[sm])]),"T2":float(np.sqrt(np.mean(so*so))),
            "T3":float(len(troughs)/duration),"T4":float(spindles/duration),"T5":float(np.trapezoid(p[sigm],f[sigm])),
            "T6":float(abs(vec)),"T7":float(np.angle(vec)),"T8_MVP_cortical_state_crossing_rate":float(crossings/duration)}


def score(active: dict, sham: dict) -> tuple[float, dict]:
    eps={"T2":1e-9,"T5":1e-12,"T6":1e-4}
    d={k:(active[k]-sham[k])/max(abs(sham[k]),eps[k]) for k in eps}
    return float(.5*d["T2"]+.25*d["T5"]+.25*d["T6"]),d


def build_yoked(source: np.ndarray, shift_samples: int) -> np.ndarray:
    return np.roll(np.asarray(source,float),int(shift_samples)).copy()


def validate_yoked(source: np.ndarray, shifted: np.ndarray, pulse_count: int) -> dict:
    a=np.asarray(source,float); b=np.asarray(shifted,float)
    multiset=bool(np.array_equal(np.sort(a),np.sort(b)))
    nonzero=int(np.count_nonzero(a))==int(np.count_nonzero(b))
    ea=float(np.sum(np.sort(a)**2)*DT); eb=float(np.sum(np.sort(b)**2)*DT)
    return {"pulse_count_source":int(pulse_count),"pulse_count_shifted":int(pulse_count),
            "amplitude_multiset_equal":multiset,"nonzero_samples_equal":nonzero,
            "source_energy":ea,"shifted_energy":eb,"energy_abs_error":abs(ea-eb),
            "passed":bool(multiset and nonzero and abs(ea-eb)<=1e-15)}


@dataclass
class RunResult:
    seed:int; condition:str; states:np.ndarray; proxy:np.ndarray; observed_proxy:np.ndarray; command:np.ndarray
    pulse_starts:list[int]; pulse_amplitudes:list[float]; baseline_metrics:dict; control_metrics:dict
    stable:bool; saturation_fraction:float; realized_pulse_phase_rad:float


def simulate(seed:int, condition:str, *, estimator_kind:str="PV", phase_target:float=0.0,
             replay_command:np.ndarray|None=None, replay_pulse_count:int=0,
             replay_event_starts:list[int]|None=None,
             deadline:Callable[[],None]|None=None) -> RunResult:
    de,dt=make_drive(seed); n=len(de); states=np.empty((n,4)); states[0]=STATE0
    source=np.empty(n); source[0]=states[0]@WEIGHTS; base=source[0]
    proxy=np.zeros(n); obs=np.zeros(n); command=np.zeros(n); rng=np.random.default_rng(seed+26081603)
    noise=rng.normal(0,.0015,n); estimator=CausalPLL() if estimator_kind=="PLL" else PhaseVocoder()
    baseline_start=int(5/DT); width=int(.11/DT); refractory=int(.8/DT); quiet=int(2.5/DT)
    last_pulse=-refractory; quiet_until=0; train_count=0; pulse_remaining=0; active_amp=.035; next_amp=.035
    starts:list[int]=[]; amps:list[float]=[]; baseline_amps=[]; amplitude_threshold=0.0
    integral=0.0; pulses_since_update=0; prev_phase=0.0; base_metrics=None
    for i in range(1,n):
        k=i-1
        if deadline and i%5000==0: deadline()
        phase,freq,conf,amp,valid=estimator.update(obs[k])
        if baseline_start<=k<CONTROL_START and valid: baseline_amps.append(amp)
        if k==CONTROL_START:
            amplitude_threshold=.5*float(np.median(baseline_amps)) if baseline_amps else 0.0
            base_metrics=metrics(proxy[baseline_start:CONTROL_START],states[baseline_start:CONTROL_START,0])
        u=0.0
        if k>=CONTROL_START:
            j=k-CONTROL_START
            if condition=="FIXED": u=.035
            elif condition=="PV_PHASE_SHIFTED_YOKED": u=float(replay_command[j]) if j<len(replay_command) else 0.0
            elif condition in ("PLL_PHASE_LOCKED","PV_PHASE_LOCKED","PV_PHASE_LOCKED_ADAPTIVE"):
                rel_prev=(prev_phase-phase_target)%(2*np.pi); rel=(phase-phase_target)%(2*np.pi)
                crossed=rel_prev-rel>np.pi
                gate=valid and .5<=freq<=1.25 and conf>=.60 and amp>=amplitude_threshold and k>=quiet_until
                if crossed and gate and k-last_pulse>=refractory:
                    active_amp=next_amp; pulse_remaining=width; last_pulse=k; starts.append(k); amps.append(active_amp)
                    pulses_since_update+=1; train_count+=1
                    if train_count>=3: quiet_until=k+quiet; train_count=0
                if condition=="PV_PHASE_LOCKED_ADAPTIVE" and (k-CONTROL_START)>0 and (k-CONTROL_START)%int(10/DT)==0 and pulses_since_update>=5:
                    recent=metrics(proxy[k-int(10/DT):k],states[k-int(10/DT):k,0]); bm=base_metrics
                    e2=(1.10*bm["T2"]-recent["T2"])/max(1.10*bm["T2"],1e-9)
                    p5=max(0.0,(bm["T5"]-recent["T5"])/max(abs(bm["T5"]),1e-12))
                    p6=max(0.0,(bm["T6"]-recent["T6"])/max(abs(bm["T6"]),1e-4))
                    err=.5*e2-.25*p5-.25*p6; integral=float(np.clip(integral+err,-4,4))
                    delta=float(np.clip(.005*err+.001*integral,-.005,.005)); next_amp=float(np.clip(next_amp+delta,.005,.05))
                    pulses_since_update=0
                if pulse_remaining>0: u=active_amp; pulse_remaining-=1
        command[k]=float(np.clip(u,0,.05))
        e,inh,th,ret=states[k]; ue=de[k]+command[k]; ut=dt[k]
        inp=np.array([.82*(2.8*e-2.2*inh)+1.10*1.8*th+ue-.8,
                      .82*(2.2*e-.6*inh)+1.10*.7*th+.7*ue-.65,
                      .95*1.6*e-1.25*ret+ut-.4,
                      .95*e+1.8*th-.3*ret+.5*ut-.55])
        states[i]=np.clip(states[k]+DT/TAUS*(sigmoid(inp)-states[k]),0,1)
        source[i]=states[i]@WEIGHTS; base+=(DT/.160)*(source[i]-base); proxy[i]=100*(source[i]-base); obs[i]=proxy[i]+noise[i]
        prev_phase=phase
    if replay_command is not None: command[CONTROL_START:]=replay_command
    else: command[-1]=command[-2]
    bm=metrics(proxy[baseline_start:CONTROL_START],states[baseline_start:CONTROL_START,0]); cm=metrics(proxy[CONTROL_START:],states[CONTROL_START:,0])
    stable=bool(np.isfinite(states).all() and np.isfinite(proxy).all() and states.min()>=0 and states.max()<=1)
    logical_count=replay_pulse_count if condition=="PV_PHASE_SHIFTED_YOKED" else len(starts)
    if condition=="PV_PHASE_SHIFTED_YOKED":
        sat=0.0; starts=[CONTROL_START+j for j in (replay_event_starts or [])]; amps=[]
    elif "PHASE_LOCKED" in condition: sat=float(np.mean([(a<=.005+1e-12 or a>=.05-1e-12) for a in amps])) if amps else 1.0
    else: sat=0.0
    analytic_phase=np.angle(hilbert(sosfiltfilt(butter(4,[.5,1.25],btype="bandpass",fs=FS,output="sos"),proxy[CONTROL_START:])))
    event_idx=[(s-CONTROL_START)%CONTROL_SAMPLES for s in starts]
    realized=float(np.angle(np.mean(np.exp(1j*analytic_phase[event_idx])))) if event_idx else float("nan")
    result=RunResult(seed,condition,states,proxy,obs,command,starts,amps,bm,cm,stable,sat,realized)
    result.logical_pulse_count=logical_count
    return result
