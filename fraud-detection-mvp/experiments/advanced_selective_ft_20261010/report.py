import json, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
from .common import HERE,OLD,PROJECT,read,write,digest,now,scores,completed,verify_baseline
from .diagnostics import groups
def rows():
    items=[]
    for trial,label in [('lgb_rich_all_stronger','LightGBM'),('ft_023','Original FT')]:
        r=read(OLD/f'runs/{trial}/result.json'); items.append(dict(model=label,path=str(OLD/f'runs/{trial}'),**r['validation_scores'],training_seconds=r.get('training_seconds',r.get('active_neural_seconds')),best_epoch=r.get('best_epoch',r.get('best_iteration')),inference_seconds=r.get('inference_seconds'),peak_allocated_mib=r.get('peak_gpu_allocated_mib'),peak_reserved_mib=r.get('peak_gpu_reserved_mib'),parameter_count=r.get('parameter_count'),seed=42))
    for r in completed():
        if r['trial_id'].startswith('smoke'): continue
        items.append(dict(model=r['trial_id'],path=str(HERE/'runs'/r['trial_id']),seed=42,**r['validation_scores'],**{k:r[k] for k in ['training_seconds','best_epoch','inference_seconds','peak_allocated_mib','peak_reserved_mib','parameter_count']}))
    return items
def draw(preds,y,ids,destination):
    for kind in ['precision_recall','roc','recall_budget']:
        fig,ax=plt.subplots(figsize=(7,5))
        for name,p in preds.items():
            if kind=='precision_recall':
                precision,recall,_=precision_recall_curve(y,p); ax.plot(recall,precision,label=name); ax.set(xlabel='Recall',ylabel='Precision')
            elif kind=='roc':
                fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,label=name); ax.set(xlabel='False positive rate',ylabel='True positive rate')
            else:
                order=np.lexsort((ids,-p)); budgets=np.linspace(.001,.10,100); captured=np.cumsum(y[order]); ax.plot(100*budgets,[captured[int(np.ceil(b*len(y)))-1]/y.sum() for b in budgets],label=name); ax.set(xlabel='Investigation budget (%)',ylabel='Fraud recall')
        ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(destination/f'{kind}_curves.png',dpi=160); plt.close(fig)
def generate(arrays=None,meta=None,status='IN_PROGRESS'):
    data=rows(); result=HERE/'results'; result.mkdir(exist_ok=True); table=pd.DataFrame(data)
    table.to_csv(result/'experiment_comparison.csv',index=False); table.to_csv(result/'metrics.csv',index=False)
    table[['model','training_seconds','inference_seconds','peak_allocated_mib','peak_reserved_mib','parameter_count']].to_csv(result/'resource_usage.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4)); ax.bar(table.model,table.average_precision); ax.axhline(.6338591015757505,color='red',linestyle='--',label='Strong LightGBM AP'); ax.set(ylabel='Validation Average Precision'); ax.tick_params(axis='x',rotation=35); ax.legend(); fig.tight_layout(); fig.savefig(result/'model_comparison.png',dpi=160); plt.close(fig)
    preds={}; y=ids=None; checks=[]
    for r in data:
        with np.load(r['path']+'/predictions_validation.npz') as f:
            if y is None: y,ids=f['y'].copy(),f['ids'].copy()
            assert np.array_equal(f['y'],y) and np.array_equal(f['ids'],ids),'Prediction ID/label mismatch'
            p=f['p'].copy(); m=scores(y,p,ids)
            for k,value in m.items():
                if not np.isclose(value,r[k],atol=1e-10,rtol=1e-10): raise RuntimeError('Metric recomputation differs: '+r['model']+':'+k)
            preds[r['model']]=p; checks.append(r['model'])
    draw(preds,y,ids,result)
    if arrays is not None:
        slices=pd.DataFrame(groups(preds,arrays['validation'],meta)); slices.to_csv(result/'validation_groups.csv',index=False)
        times=slices[slices.group.str.startswith('time_')]; times.to_csv(result/'validation_time_slices.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,4))
        for name,g in times.groupby('model'): ax.plot(g.group,g.average_precision,marker='o',label=name)
        ax.set(ylabel='Average Precision'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(result/'temporal_ap.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for path in sorted((HERE/'runs').glob('*/history.csv')):
        if path.parent.name.startswith('smoke'): continue
        h=pd.read_csv(path); ax.plot(h.epoch,h.validation_ap,label=path.parent.name)
    ax.set(xlabel='Epoch',ylabel='Validation AP');
    if ax.lines: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(result/'training_curves.png',dpi=160); plt.close(fig)
    baseline=verify_baseline(); write(result/'verification.json',dict(status='PASS',verified_utc=now(),metric_recomputation='self-check, not independent validation',models=checks,baseline=baseline))
    lines=['# Selective FT: 当前实验报告','',f'状态：{status}；更新：{now()}。所有主结果是固定验证集、seed=42 的开发结果。','',table[['model','average_precision','recall_top_1pct','training_seconds']].to_string(index=False),'','## 解释与证据边界','原test已查看；本轮训练/选型不读取test标签。不做fresh seeds或rolling folds，不能声称跨seed稳定或独立测试胜出。','诊断是观察；压缩、SPA、IAI的匹配差异用于检验假设；SOAP必须与同预算继续BCE比较。负结果与未执行项目保留。']
    valid=[r for r in data if r['model'] not in ['LightGBM','Original FT']]
    if valid:
        qualifying=[r for r in valid if r['recall_top_1pct']>=.2705456936226167]
        best=max(valid,key=lambda r:r['average_precision']); selected=max(qualifying or valid,key=lambda r:r['average_precision'])
        win=selected['average_precision']>.6338591015757505 and selected['recall_top_1pct']>=.2705456936226167
        lines+=['',f'不受约束最高AP：{best["model"]}；AP={best["average_precision"]:.6f}。按R1约束选中：{selected["model"]}，AP={selected["average_precision"]:.6f}、R1={selected["recall_top_1pct"]:.6f}；双目标达到：{win}。',f'最高AP候选相对原FT改变{best["average_precision"]-.578390593063534:+.6f}；原AP差距缩小比例{(best["average_precision"]-.578390593063534)/(.6338591015757505-.578390593063534):.2%}。该比例不是独立泛化证明。']
    lookup={r['model']:r for r in data}
    lines+=['','## 受控实验逐项回答']
    for left,right,question in [('M1_full','M0_compressed','去掉压缩是否有帮助'),('M2_full_spa','M1_full','SPA是否有帮助'),('M3_full_iai','M1_full','IAI是否有帮助'),('M4_full_spa_iai','M2_full_spa','SPA之上加入IAI是否有帮助'),('C1_official_soap','C0_continue_bce','SOAP是否优于继续BCE')]:
        if left in lookup and right in lookup:
            a,b=lookup[left],lookup[right]; lines.append(f'- {question}：实测AP差{a["average_precision"]-b["average_precision"]:+.6f}，R1差{a["recall_top_1pct"]-b["recall_top_1pct"]:+.6f}；这是单seed开发观察，不是机制因果证明。')
        else: lines.append(f'- {question}：尚无完整匹配结果，不能判断。')
    required=['M0_compressed','M1_full','M2_full_spa','M3_full_iai','M4_full_spa_iai','C0_continue_bce','C1_official_soap']
    pd.DataFrame([dict(model=name,status='COMPLETED' if name in lookup else 'NOT_COMPLETED') for name in required]).to_csv(result/'planned_trial_status.csv',index=False)
    diag=result/'train_vs_validation_diagnostics.csv'
    if diag.exists():
        d=pd.read_csv(diag); lines+=['','## 拟合与后期退化观察',d[['model','split','average_precision','bce','weighted_bce']].to_string(index=False),'训练与验证差距、时间/OOV分层提示可能机制，但不能证明attention是唯一瓶颈。']
    lines+=['','## 局限与下一步','无fresh-seed确认；开发集已被多次选择；原test已查看；训练统计不保证匿名V/D字段在线可得；weighted BCE的概率可能失准；离线指标不等于减少损失，也不代表production readiness。','若出现可信增益，最有价值的下一步是冻结方法后用新的独立时间数据验证；本轮不额外增加这项验收门槛。','',f'English: Controlled, seed-42 development benchmark on IEEE-CIS with a preserved strong LightGBM control. Current status: {status}. Results above are measured; no independent-test superiority or production readiness is claimed.']
    if (HERE/'failure.json').exists(): lines+=['','## 未解决问题',json.dumps(read(HERE/'failure.json'),ensure_ascii=False,indent=2)]
    (HERE/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (HERE/'README.md').write_text('# Selective FT — IEEE-CIS\n\n状态：'+status+'。本目录独立保存全部代码与结果，不修改另一组。\n\n- [完整计划](PLAN_ZH.md)\n- [实际结果和解释](report.md)\n- [实验总表](results/experiment_comparison.csv)\n- [拟合诊断](results/train_vs_validation_diagnostics.csv)\n- [时间/OOV群体](results/validation_groups.csv)\n- [资源用量](results/resource_usage.csv)\n- [当前状态](campaign_state.json)\n- [固定协议](protocol.json)\n\n从 fraud-detection-mvp 运行：\n```powershell\n.\\.venv\\Scripts\\python.exe -m experiments.advanced_selective_ft_20261010.campaign --prepare\n.\\.venv\\Scripts\\python.exe -m experiments.advanced_selective_ft_20261010.campaign --run\n.\\.venv\\Scripts\\python.exe -m experiments.advanced_selective_ft_20261010.campaign --resume\n.\\.venv\\Scripts\\python.exe -m experiments.advanced_selective_ft_20261010.campaign --report\n```\n\nCUDA必需；依赖复用项目环境，额外包固定libauc==2.0.1（--no-deps）。源数据、row-level predictions和checkpoint仅本地保存。\n',encoding='utf-8')
    return table
def exploratory_test(arrays,meta):
    import torch, lightgbm as lgb
    from .models import build
    from .train import predict
    from .resources import lease
    from experiments.ft_upgrade_20261010.train import cuda
    from experiments.ft_upgrade_20261010.data import test_features
    from experiments.ft_upgrade_20261010.controls import frame
    dest=HERE/'results/test_metrics.csv'
    if dest.exists(): return
    data=[r for r in rows() if r['model']!='LightGBM']
    qualifying=[r for r in data if r['recall_top_1pct']>=.2705456936226167]
    chosen=max(qualifying or data,key=lambda r:r['average_precision']); unconstrained=max(data,key=lambda r:r['average_precision'])
    lock=HERE/'selection_lock.json'
    if lock.exists():
        chosen=read(lock)['selected']
    else: write(lock,dict(status='LOCKED_BEFORE_TEST',locked_utc=now(),selected=chosen,highest_ap=unconstrained,checkpoint_sha256=digest(Path(chosen['path'])/'best.pt'),test='previously viewed; exploratory only'))
    with lease('locked_exploratory_test') as resource:
        features,p=_test_cuda(chosen,meta,lock,resource)
    tree=lgb.Booster(model_file=str(OLD/'runs/lgb_rich_all_stronger/model.txt'))
    tp=tree.predict(frame(features,meta),num_threads=2)
    write(HERE/'test_access.json',dict(status='EXPLORATORY_PREVIOUSLY_VIEWED_TEST',access_utc=now(),selection_sha256=digest(lock),used_for_selection=False))
    with np.load(PROJECT/'results/prepared/full/test_labels.npz') as f:
        assert np.array_equal(f['transaction_id'],features['ids']); y=f['y'].copy()
    np.savez_compressed(HERE/'results/exploratory_test_predictions.npz',ids=features['ids'],y=y,selected=p,lightgbm=tp)
    pd.DataFrame([dict(model=chosen['model'],evaluation='exploratory previously viewed test',**scores(y,p,features['ids'])),dict(model='LightGBM',evaluation='exploratory previously viewed test',**scores(y,tp,features['ids']))]).to_csv(dest,index=False)

def _test_cuda(chosen,meta,lock,resource):
    import torch
    from .resources import configure_torch
    from .models import build
    from .train import predict
    from experiments.ft_upgrade_20261010.train import cuda
    from experiments.ft_upgrade_20261010.data import test_features
    device=cuda(42); configure_torch(resource,device); source=Path(chosen['path'])/'best.pt'; cp=torch.load(source,map_location='cpu',weights_only=False)
    assert digest(source)==read(lock)['checkpoint_sha256']
    model=build(meta,cp['config'],cp.get('importance')).cuda(); model.load_state_dict(cp['model'])
    features=test_features('rich_all'); features['y']=np.zeros(len(features['ids']),np.int8)
    p=predict(model,features,128); del model; gc.collect(); torch.cuda.empty_cache()
    return features,p
