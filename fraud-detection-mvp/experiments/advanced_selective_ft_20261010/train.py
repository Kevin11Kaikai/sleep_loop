import gc, random, time
import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.nn import functional as F
from .common import HERE,CACHE,OLD,TRAIN_STOP,read,write,now,digest,scores,fingerprints,spent
from .models import build
from .resources import lease, configure_torch
from experiments.ft_upgrade_20261010.train import cuda
@torch.inference_mode()
def predict(model,part,batch=128,logits=False):
    model.eval(); output=[]; dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for start in range(0,len(part['y']),batch):
        num=torch.as_tensor(part['x_num'][start:start+batch],device='cuda'); cat=torch.as_tensor(part['x_cat'][start:start+batch],device='cuda')
        with torch.autocast('cuda',dtype=dtype): z=model(num,cat).squeeze(-1)
        output.append(z.float().cpu().numpy() if logits else z.float().sigmoid().cpu().numpy())
    result=np.concatenate(output)
    if not np.isfinite(result).all(): raise RuntimeError('Nonfinite predictions')
    return result
def rng_state():
    return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
def restore_rng(cp):
    torch.set_rng_state(cp['torch_rng']); torch.cuda.set_rng_state_all(cp['cuda_rng']); np.random.set_state(cp['numpy_rng']); random.setstate(cp['python_rng'])
def save_checkpoint(path,value):
    tmp=path.with_suffix('.tmp'); torch.save(value,tmp); tmp.replace(path)
def run_trial(trial,cfg,arrays,meta,importance=None,source=None,epoch_target=None,finalize_only=False):
    path=HERE/'runs'/trial; path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').exists(): return read(path/'result.json')
    if (path/'config.json').exists() and read(path/'config.json')!=cfg: raise RuntimeError('Trial config changed')
    write(path/'config.json',cfg)
    with lease(trial) as resource:
        return _run_trial_cuda(path,trial,cfg,arrays,meta,importance,source,epoch_target,finalize_only,resource)

def _run_trial_cuda(path,trial,cfg,arrays,meta,importance,source,epoch_target,finalize_only,resource):
    device=cuda(42); configure_torch(resource,device); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    train,valid=arrays['train'],arrays['validation']; dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=build(meta,cfg,importance,matched_init=True).to(device)
    optimizer=torch.optim.AdamW(model.make_parameter_groups(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4,min_lr=1e-6)
    aploss=None
    if cfg.get('fine'):
        from libauc.optimizers import SOAP
        optimizer=SOAP(model.make_parameter_groups(),lr=cfg['lr'],mode='adam',weight_decay=cfg['weight_decay'],epoch_decay=0,verbose=False,device=device)
        scheduler=None
        if cfg['fine']=='soap':
            from libauc.losses import APLoss
            aploss=APLoss(data_len=len(train['y']),gamma=.9,margin=1,surr_loss='squared_hinge',device=device)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    history=[]; best=-1.; bad=0; elapsed=0.; start_epoch=1; best_epoch=0
    if (path/'last.pt').exists():
        cp=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        if cp['config']!=cfg or cp['fingerprints']!=fingerprints(): raise RuntimeError('Resume code/config fingerprint mismatch')
        if cp['cache_sha256']!=digest(CACHE/'manifest.json'): raise RuntimeError('Resume input fingerprint mismatch')
        if source and cp.get('source_sha256')!=digest(source): raise RuntimeError('Fine-tuning source changed')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        if scheduler: scheduler.load_state_dict(cp['scheduler'])
        if aploss:
            aploss.u_all=cp['ap_u_all'].to(device); aploss.u_pos=cp['ap_u_pos'].to(device)
        scaler.load_state_dict(cp['scaler']); restore_rng(cp)
        if cfg.get('fine'):
            optimizer.T=cp.get('soap_T',0); optimizer.steps=cp.get('soap_steps',0)
        history=cp['history']; elapsed=cp['active_seconds']; best=cp['best_ap']; bad=cp['bad']; start_epoch=cp['epoch']+1; best_epoch=cp['best_epoch']
    elif source:
        cp=torch.load(source,map_location='cpu',weights_only=False); model.load_state_dict(cp['model'])
        p=predict(model,valid,cfg['microbatch']); best=scores(valid['y'],p,valid['ids'])['average_precision']
        save_checkpoint(path/'best.pt',dict(model=model.state_dict(),config=cfg,epoch=0,importance=importance,cache_sha256=digest(CACHE/'manifest.json')))
    xn=torch.as_tensor(train['x_num'],device=device); xc=torch.as_tensor(train['x_cat'],device=device); y=torch.as_tensor(train['y'],device=device,dtype=torch.float32)
    reserve=torch.empty(int(cfg.get('memory_reserve_bytes',0)),device=device,dtype=torch.uint8)
    if not next(model.parameters()).is_cuda: raise RuntimeError('CUDA model required')
    maximum=start_epoch-1 if finalize_only else min(cfg['max_epochs'],epoch_target or cfg['max_epochs']); stop='PAIRED_FINALIZATION' if finalize_only else 'MAX_EPOCHS'
    for epoch in range(start_epoch,maximum+1):
        if time.time()>=TRAIN_STOP or spent()>=21600: stop='CAMPAIGN_BUDGET'; break
        if elapsed>=cfg['max_seconds']: stop='TRIAL_TIME_BUDGET'; break
        t=time.perf_counter(); model.train(); g=torch.Generator(device='cpu').manual_seed(42000+epoch)
        if cfg.get('fine'):
            from libauc.sampler import DualSampler
            sampler=DualSampler(np.arange(len(y)),batch_size=128,labels=train['y'],num_pos=32,sampling_rate=None,random_seed=42000+epoch)
            order=torch.as_tensor(list(iter(sampler)),dtype=torch.long)
        elif cfg.get('sampling','bootstrap')=='bootstrap':
            order=torch.multinomial(torch.linspace(1.,4.,len(y)),len(y),replacement=True,generator=g)
        else: order=torch.randperm(len(y),generator=g)
        temporal=torch.linspace(1.,4.,len(y),device=device); temporal/=temporal.mean()
        total=0.; count=0; steps=0
        for offset in range(0,len(order),128):
            idx=order[offset:offset+128].to(device); optimizer.zero_grad(set_to_none=True)
            if cfg.get('fine'):
                pieces=[]
                for chunk in idx.split(cfg['microbatch']):
                    with torch.autocast('cuda',dtype=dtype):
                        pieces.append(activation_checkpoint(model,xn[chunk],xc[chunk],use_reentrant=False).squeeze(-1))
                logits=torch.cat(pieces).float()
                loss=aploss(logits.sigmoid(),y[idx],idx) if aploss else F.binary_cross_entropy_with_logits(logits,y[idx],pos_weight=torch.tensor(3.,device=device))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite fine-tuning loss')
                scaler.scale(loss).backward(); total+=float(loss.detach())*len(idx); count+=len(idx)
            for chunk in ([] if cfg.get('fine') else idx.split(cfg['microbatch'])):
                with torch.autocast('cuda',dtype=dtype): z=model(xn[chunk],xc[chunk]).squeeze(-1)
                if aploss:
                    loss=aploss(z.float().sigmoid(),y[chunk],chunk)
                else:
                    losses=F.binary_cross_entropy_with_logits(z.float(),y[chunk],pos_weight=torch.tensor(3.,device=device),reduction='none')
                    if cfg.get('sampling')=='temporal_loss': losses=losses*temporal[chunk]
                    loss=losses.mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                fraction=len(chunk)/len(idx); scaler.scale(loss*fraction).backward(); total+=float(loss.detach())*len(chunk); count+=len(chunk)
            scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradients')
            scaler.step(optimizer); scaler.update(); steps+=1
        torch.cuda.synchronize(); p=predict(model,valid,cfg['microbatch']); metric=scores(valid['y'],p,valid['ids'])
        seconds=time.perf_counter()-t; elapsed+=seconds
        if scheduler: scheduler.step(metric['average_precision'])
        if metric['average_precision']>best+1e-6:
            best=metric['average_precision']; best_epoch=epoch; bad=0
            save_checkpoint(path/'best.pt',dict(model=model.state_dict(),config=cfg,epoch=epoch,importance=importance,cache_sha256=digest(CACHE/'manifest.json')))
        else: bad+=1
        row=dict(epoch=epoch,training_loss=total/count,validation_ap=metric['average_precision'],validation_recall_1pct=metric['recall_top_1pct'],epoch_seconds=seconds,training_samples=count,unique_samples=int(order.unique().numel()),sampler_order_sha256=__import__('hashlib').sha256(order.numpy().tobytes()).hexdigest(),optimizer_steps=steps,learning_rate=optimizer.param_groups[0]['lr'])
        history.append(row); pd.DataFrame(history).to_csv(path/'history.csv',index=False)
        cp=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,scaler=scaler.state_dict(),epoch=epoch,best_epoch=best_epoch,best_ap=best,bad=bad,history=history,active_seconds=elapsed,config=cfg,importance=importance,fingerprints=fingerprints(),cache_sha256=digest(CACHE/'manifest.json'),source_sha256=digest(source) if source else None,soap_T=getattr(optimizer,'T',None),soap_steps=getattr(optimizer,'steps',None),sampler_state='deterministically reconstructed complete-epoch DualSampler; no mid-epoch resume',**rng_state())
        if aploss: cp.update(ap_u_all=aploss.u_all,ap_u_pos=aploss.u_pos)
        save_checkpoint(path/'last.pt',cp)
        write(path/'progress.json',dict(status='RUNNING',updated_utc=now(),epoch=epoch,best_ap=best,training_seconds=elapsed,metrics=metric,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576))
        print(f'{trial} epoch={epoch} AP={metric["average_precision"]:.6f} best={best:.6f} seconds={seconds:.1f}',flush=True)
        if not cfg.get('fine') and epoch>=cfg['min_epochs'] and bad>=cfg['patience']: stop='EARLY_STOPPING'; break
    if not finalize_only and epoch_target and maximum<cfg['max_epochs'] and stop=='MAX_EPOCHS':
        del model,optimizer,xn,xc,y,reserve; gc.collect(); torch.cuda.empty_cache(); return read(path/'progress.json')
    if not (path/'best.pt').exists(): raise RuntimeError('No valid checkpoint before budget expiry')
    cp=torch.load(path/'best.pt',map_location='cpu',weights_only=False); model.load_state_dict(cp['model'])
    t=time.perf_counter(); p=predict(model,valid,cfg['microbatch']); torch.cuda.synchronize(); inference=time.perf_counter()-t
    metric=scores(valid['y'],p,valid['ids']); np.savez_compressed(path/'predictions_validation.npz',ids=valid['ids'],y=valid['y'],p=p)
    if cfg.get('smoke_rows'):
        replay=build(meta,cfg,importance).to(device); replay.load_state_dict(cp['model'])
        replay_p=predict(replay,valid,cfg['microbatch'])
        if not np.allclose(p,replay_p,atol=1e-7,rtol=1e-6): raise RuntimeError('Fresh checkpoint reload prediction mismatch')
        del replay
    result=dict(status='COMPLETED',trial_id=trial,config=cfg,validation_scores=metric,best_epoch=cp['epoch'],epochs_executed=len(history),training_seconds=elapsed,inference_seconds=inference,waiting_seconds=resource['waiting_seconds'],stop_reason=stop,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_sha256=digest(path/'best.pt'),cache_sha256=digest(CACHE/'manifest.json'),fingerprints=fingerprints(),gpu=torch.cuda.get_device_name(0),finished_utc=now(),source_checkpoint=str(source) if source else None)
    write(path/'result.json',result); write(path/'progress.json',dict(status='COMPLETED',updated_utc=now(),best_ap=best,training_seconds=elapsed))
    del model,optimizer,xn,xc,y,reserve; gc.collect(); torch.cuda.empty_cache()
    return result
