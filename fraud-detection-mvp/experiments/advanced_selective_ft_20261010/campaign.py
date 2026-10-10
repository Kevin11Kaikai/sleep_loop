import argparse, gc, os, time, traceback
import numpy as np
import torch
from .common import HERE,OLD,CACHE,TRAIN_STOP,prepare,load_data,read,write,now,digest,completed,spent,verify_baseline,fingerprints
from .checks import run as checks
from .importance import prepare as importance_prepare
from .train import run_trial,predict
from .models import build
from .diagnostics import run as diagnose
from .report import generate,exploratory_test
def state(status,**extra):
    write(HERE/'campaign_state.json',dict(status=status,updated_utc=now(),pid=os.getpid(),neural_seconds=spent(),absolute_training_stop=TRAIN_STOP,**extra))
def configs(base):
    return [('M0_compressed',base|dict(compression=.25,spa=False,iai=False)),('M1_full',base|dict(compression=None,spa=False,iai=False)),('M2_full_spa',base|dict(compression=None,spa=True,iai=False)),('M3_full_iai',base|dict(compression=None,spa=False,iai=True)),('M4_full_spa_iai',base|dict(compression=None,spa=True,iai=True))]
def smoke(arrays,meta,importance,base):
    if (HERE/'results/smoke.json').exists(): return read(HERE/'results/smoke.json')['microbatch']
    subset={name:{k:v[:8192 if name=='train' else 2048] for k,v in part.items()} for name,part in arrays.items()}
    for micro in [128,64,32]:
        try:
            results=[]
            for trial,cfg in configs(base)[:3]:
                reserve=sum(arrays['train'][k].nbytes-subset['train'][k].nbytes for k in ['x_num','x_cat'])+3*len(arrays['train']['y'])
                cfg=cfg|dict(max_epochs=1,min_epochs=1,patience=1,max_seconds=600,microbatch=micro,smoke_rows=[8192,2048],memory_reserve_bytes=reserve)
                results.append(run_trial(f'smoke_{trial}_micro{micro}',cfg,subset,meta,importance))
            write(HERE/'results/smoke.json',dict(status='PASS',microbatch=micro,effective_batch=128,rows=[8192,2048],real_dataset=True,models=results))
            return micro
        except torch.cuda.OutOfMemoryError:
            write(HERE/f'results/smoke_oom_{micro}.json',dict(status='OOM',microbatch=micro,traceback=traceback.format_exc()))
            gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError('Smoke OOM at microbatch32; no compression or CPU fallback')
def best_bce():
    items=[r for r in completed() if not r['trial_id'].startswith('smoke') and not r['config'].get('fine')]
    old=read(OLD/'runs/ft_023/result.json'); old['trial_id']='Original FT'; old['path']=str(OLD/'runs/ft_023'); old['training_seconds']=0
    for r in items: r['path']=str(HERE/'runs'/r['trial_id'])
    return max(items+[old],key=lambda r:r['validation_scores']['average_precision'])
def execute():
    protocol=prepare(); checks(); arrays,meta=load_data(); importance=importance_prepare(arrays,meta)
    state('DIAGNOSING'); diagnose(arrays,meta)
    state('SMOKE'); micro=smoke(arrays,meta,importance,protocol['base_config'])
    base=protocol['base_config']|dict(microbatch=micro)
    for trial,cfg in configs(base):
        if time.time()>=TRAIN_STOP or spent()>=21600-3600: break
        state('TRAINING',current_trial=trial); run_trial(trial,cfg,arrays,meta,importance)
        generate(arrays,meta)
    best=best_bce(); enough=best['validation_scores']['average_precision']>protocol['primary_ap'] and best['validation_scores']['recall_top_1pct']>=protocol['minimum_recall']
    if not enough:
        for trial,sampling in [('E1_complete_temporal_loss','temporal_loss'),('E2_complete_uniform','uniform')]:
            if spent()>21600-7200 or time.time()>TRAIN_STOP-7200: break
            cfg=best['config']|dict(max_epochs=40,min_epochs=8,patience=8,max_seconds=3600,microbatch=micro,sampling=sampling,spa=best['config'].get('spa',False),iai=best['config'].get('iai',False))
            state('TRAINING',current_trial=trial); run_trial(trial,cfg,arrays,meta,importance)
            generate(arrays,meta)
        records={r['trial_id']:r for r in completed()}
        pairs=[('M2_full_spa','M1_full'),('M4_full_spa_iai','M3_full_iai')]
        supporting=[records[a] for a,b in pairs if a in records and b in records and records[a]['validation_scores']['average_precision']-records[b]['validation_scores']['average_precision']>=.005 and records[a]['validation_scores']['recall_top_1pct']-records[b]['validation_scores']['recall_top_1pct']>=-.005]
        if supporting and spent()<21600-7200 and time.time()<TRAIN_STOP-7200:
            stable=importance_prepare(arrays,meta,stable=True); cfg=max(supporting,key=lambda r:r['validation_scores']['average_precision'])['config']
            state('TRAINING',current_trial='E3_stable_spa'); run_trial('E3_stable_spa',cfg,arrays,meta,stable); generate(arrays,meta)
    best=best_bce(); source=__import__('pathlib').Path(best['path'])/'best.pt'
    cp=torch.load(source,map_location='cpu',weights_only=False); source_importance=cp.get('importance',importance if best['config'].get('spa') else None); del cp
    fine_base=base|{k:best['config'][k] for k in ['compression','width','depth','heads','dropout','embedding','frequency_scale','ffn_multiplier']}|dict(spa=best['config'].get('spa',False),iai=best['config'].get('iai',False),lr=1e-5,weight_decay=1e-6,max_epochs=3,min_epochs=1,patience=3,max_seconds=1800)
    pair_stop=3
    for epoch in [1,2,3]:
        if spent()>=21600 or time.time()>=TRAIN_STOP: pair_stop=epoch-1; break
        prior=[read(HERE/'runs'/name/'progress.json') for name in ['C0_continue_bce','C1_official_soap'] if (HERE/'runs'/name/'progress.json').exists()]
        if any(p['training_seconds']>=1800 for p in prior): pair_stop=epoch-1; break
        for trial,loss in [('C0_continue_bce','bce'),('C1_official_soap','soap')]:
            state('FINE_TUNING',current_trial=trial,paired_epoch=epoch)
            run_trial(trial,fine_base|dict(fine=loss),arrays,meta,source_importance,source,epoch_target=epoch)
        soap=read(HERE/'runs/C1_official_soap/progress.json')
        if soap.get('metrics',{}).get('average_precision',soap['best_ap'])<best['validation_scores']['average_precision']-.005: pair_stop=epoch; break
    for trial,loss in [('C0_continue_bce','bce'),('C1_official_soap','soap')]:
        if (HERE/'runs'/trial/'last.pt').exists() and not (HERE/'runs'/trial/'result.json').exists():
            # Finalize existing best without further optimizer steps.
            cfg=fine_base|dict(fine=loss)
            run_trial(trial,cfg,arrays,meta,source_importance,source,epoch_target=pair_stop,finalize_only=True)
    generate(arrays,meta,status='TRAINING_FINISHED')
    if time.time()<TRAIN_STOP: exploratory_test(arrays,meta)
    actual={r['trial_id'] for r in completed()}
    required=[name for name,_ in configs(base)]+['C0_continue_bce','C1_official_soap']
    missing=[name for name in required if name not in actual]
    status='BUDGET_PARTIAL' if missing else 'COMPLETED'
    state(status,baseline_preservation=verify_baseline(),missing_required_trials=missing)
    generate(arrays,meta,status=status)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--run',action='store_true'); parser.add_argument('--resume',action='store_true'); parser.add_argument('--report',action='store_true'); args=parser.parse_args()
    if args.prepare:
        prepare(); checks(); arrays,meta=load_data(); importance_prepare(arrays,meta); state('PREPARED'); generate(arrays,meta,status='PREPARED')
    elif args.report:
        arrays,meta=load_data(); generate(arrays,meta,status=read(HERE/'campaign_state.json')['status'] if (HERE/'campaign_state.json').exists() else 'NOT_STARTED')
    elif args.run or args.resume:
        from .resources import alive
        import psutil, json
        worker_lock=HERE/'worker_lock.json'
        if worker_lock.exists():
            if alive(read(worker_lock)): raise RuntimeError('This group already has an active worker')
            worker_lock.unlink()
        fd=os.open(worker_lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        with os.fdopen(fd,'w') as f: json.dump({'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()},f)
        try:
            try: execute()
            except Exception:
                write(HERE/'failure.json',dict(status='FAILED',updated_utc=now(),traceback=traceback.format_exc(),fingerprints=fingerprints()))
                state('FAILED'); generate(status='FAILED'); raise
        finally:
            if worker_lock.exists() and read(worker_lock).get('pid')==os.getpid(): worker_lock.unlink()
    else: parser.print_help()
if __name__=='__main__': main()
