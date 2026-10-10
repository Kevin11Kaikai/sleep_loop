import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json, time, hashlib
from pathlib import Path
import numpy as np
from experiments.ft_upgrade_20261010.utils import read, write, digest, now, scores, verify_baseline
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = HERE.parent / 'ft_upgrade_20261010'
CACHE = OLD / 'cache/rich_all'
DEADLINE = 1791607859.1377695 + 86400
TRAIN_STOP = DEADLINE - 2700
def fingerprints():
    return {p.name: digest(p) for p in sorted(HERE.glob('*.py'))} | {'protocol.json': digest(HERE/'protocol.json'), 'common_scheduler.py': digest(HERE.parent/'resource_coordination/scheduler.py')}
def load_data():
    meta = read(CACHE/'manifest.json')
    arrays = {}
    for split in ['train','validation']:
        with np.load(CACHE/f'{split}.npz',allow_pickle=False) as f:
            arrays[split] = {k:f[k] for k in ['x_num','x_cat','y','ids','dt']}
    return arrays,meta
def prepare():
    HERE.joinpath('results').mkdir(exist_ok=True)
    baseline=verify_baseline()
    ft=read(OLD/'runs/ft_023/result.json')
    if digest(OLD/'runs/ft_023/best.pt')!=ft['checkpoint_sha256']: raise RuntimeError('Original FT checkpoint changed')
    meta=read(CACHE/'manifest.json')
    for name,h in meta['files_sha256'].items():
        if digest(CACHE/name)!=h: raise RuntimeError('Cache fingerprint mismatch: '+name)
    cfg=read(OLD/'runs/ft_023/config.json') | dict(max_epochs=40,min_epochs=8,patience=8,max_seconds=3600,spa=False,iai=False,microbatch=128,sampling='bootstrap')
    protocol=dict(name=HERE.name,created_utc=now(),seed=42,original_started_timestamp=1791607859.1377695,absolute_deadline_timestamp=DEADLINE,training_stop_timestamp=TRAIN_STOP,max_neural_seconds=21600,minimum_12h_revoked=True,base_config=cfg,cache_manifest_sha256=digest(CACHE/'manifest.json'),baseline_preservation=baseline,primary_ap=0.6338591015757505,minimum_recall=0.2705456936226167,independent_test=False,optional_trials_maximum=3,soap=dict(version='2.0.1',lr=1e-5,weight_decay=1e-6,gamma=.9,margin=1,max_epochs=3,max_seconds_per_arm=1800,positive_fraction=.25),resource_policy='FIFO exclusive GPU lease; CPU threads <=4; RAM free >=6 GiB',test_access='only once after selection lock; previously viewed test is exploratory')
    if (HERE/'resource_policy.json').exists():
        protocol['resource_policy']='Common exclusive FIFO trial GPU lease; memory_fraction .80; CPU threads <=4; free RAM >=6 GiB; unknown and legacy GPU jobs block'
        protocol['resource_policy_amendment_sha256']=digest(HERE/'resource_policy.json')
    if (HERE/'protocol.json').exists():
        prior=read(HERE/'protocol.json'); protocol['created_utc']=prior['created_utc']
        if prior!=protocol: raise RuntimeError('Frozen protocol differs')
    else: write(HERE/'protocol.json',protocol)
    write(HERE/'preparation.json',dict(status='PASS',created_utc=now(),cache_sha256=digest(CACHE/'manifest.json'),baseline=baseline))
    return protocol
def completed():
    return [read(p) for p in sorted((HERE/'runs').glob('*/result.json'))] if (HERE/'runs').exists() else []
def spent():
    value=sum(r['training_seconds'] for r in completed())
    for p in (HERE/'runs').glob('*/last.pt'):
        if not (p.parent/'result.json').exists():
            import torch
            value+=float(torch.load(p,map_location='cpu',weights_only=False).get('active_seconds',0))
    return value
