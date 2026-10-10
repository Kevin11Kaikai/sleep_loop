import gc, time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from .common import HERE,OLD,CACHE,read,write,now,scores,digest
from .models import build
from .train import predict
from .resources import lease, configure_torch
from experiments.ft_upgrade_20261010.train import cuda
def metrics(y,z,ids):
    p=expit(z.astype('float64')); loss=np.logaddexp(0,z)-y*z
    return scores(y,p,ids)|dict(bce=float(loss.mean()),weighted_bce=float((loss*np.where(y==1,3.,1.)).mean()),mean_fraud_probability=float(p[y==1].mean()),mean_nonfraud_probability=float(p[y==0].mean()))
def groups(predictions,part,meta):
    rows=[]; n=len(part['y']); masks={}
    for i,index in enumerate(np.array_split(np.arange(n),6)):
        m=np.zeros(n,bool); m[index]=True; masks[f'time_{i+1}']=m
    for name in ['card1','card1_addr1','card1_card2_addr1']:
        if name not in meta['categorical']: continue
        c=part['x_cat'][:,meta['categorical'].index(name)]
        masks[name+'_unknown_rare_missing']=c==0; masks[name+'_known']=c!=0
    for model,p in predictions.items():
        ranked=np.lexsort((part['ids'],-p)); global_top=np.zeros(n,bool); global_top[ranked[:int(np.ceil(.01*n))]]=True
        for name,m in masks.items():
            if len(np.unique(part['y'][m]))<2: continue
            row=scores(part['y'][m],p[m],part['ids'][m])
            rows.append(dict(model=model,group=name,fraud_rate=float(part['y'][m].mean()),mean_prediction=float(p[m].mean()),global_top1_captured=int(part['y'][m&global_top].sum()),global_top1_selected=int((m&global_top).sum()),**row))
    return rows
def run(arrays,meta):
    dest=HERE/'results/train_vs_validation_diagnostics.csv'
    if dest.exists(): return
    with lease('best_last_diagnostics') as resource:
        rows,preds,audit=_diagnose_cuda(arrays,meta,resource)
    with np.load(OLD/'runs/lgb_rich_all_stronger/predictions_validation.npz') as f:
        assert np.array_equal(f['ids'],arrays['validation']['ids']) and np.array_equal(f['y'],arrays['validation']['y'])
        preds['LightGBM']=f['p']
    pd.DataFrame(rows).to_csv(dest,index=False)
    pd.DataFrame(groups(preds,arrays['validation'],meta)).to_csv(HERE/'results/baseline_validation_groups.csv',index=False)
    write(HERE/'results/diagnostic_provenance.json',dict(status='PASS',updated_utc=now(),checkpoints=audit,test_labels_read=False,cache_sha256=digest(CACHE/'manifest.json')))
    print('FULL TRAIN/VALID DIAGNOSTICS PASS',flush=True)

def _diagnose_cuda(arrays,meta,resource):
    rows=[]; preds={}; audit=[]
    device=cuda(42); configure_torch(resource,device)
    for checkpoint in ['best','last']:
        path=OLD/f'runs/ft_023/{checkpoint}.pt'; cp=torch.load(path,map_location='cpu',weights_only=False)
        model=build(meta,cp['config']).cuda(); model.load_state_dict(cp['model'])
        start=time.perf_counter()
        for split in ['train','validation']:
            z=predict(model,arrays[split],128,logits=True)
            rows.append(dict(model='ft023_'+checkpoint,split=split,**metrics(arrays[split]['y'],z,arrays[split]['ids'])))
            if split=='validation': preds['ft023_'+checkpoint]=expit(z.astype('float64'))
        audit.append(dict(checkpoint=str(path),sha256=digest(path),inference_seconds=time.perf_counter()-start,mode='fresh CUDA eval inference; complete fixed train and validation',training_updates=0))
        del model,cp; gc.collect(); torch.cuda.empty_cache()
    return rows,preds,audit
