import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from .common import HERE,write,read,digest,CACHE
def estimate(train,meta,indices):
    x=np.concatenate([train['x_num'][indices],train['x_cat'][indices]],axis=1).astype('float64')
    discrete=np.array([len(np.unique(x[:,i]))<=2 for i in range(train['x_num'].shape[1])]+[True]*train['x_cat'].shape[1])
    y=train['y'][indices]
    actual=mutual_info_classif(x,y,discrete_features=discrete,random_state=42)
    null=[]
    for seed in [101,102,103]: null.append(mutual_info_classif(x,np.random.RandomState(seed).permutation(y),discrete_features=discrete,random_state=42))
    return np.maximum(actual-np.mean(null,axis=0),0)
def prepare(arrays,meta,stable=False):
    target=HERE/'results'/('stable_importance.json' if stable else 'importance.json')
    if target.exists():
        state=read(target)
        if state['cache_sha256']!=digest(CACHE/'manifest.json'): raise RuntimeError('Importance cache changed')
        return state['scores']
    train=arrays['train']; names=meta['numeric_output_names']+meta['categorical']; rng=np.random.RandomState(42)
    if stable:
        rows=np.array_split(np.arange(len(train['y'])),4)
        values=np.array([estimate(train,meta,rng.choice(r,min(25000,len(r)),replace=False)) for r in rows])
        value=np.median(values,axis=0)-.5*(np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0))
        pd.DataFrame(values.T,index=names,columns=['window1','window2','window3','window4']).to_csv(HERE/'results/feature_importance_by_window.csv')
    else:
        indices=rng.choice(len(train['y']),min(50000,len(train['y'])),replace=False)
        value=estimate(train,meta,indices)
    state=dict(scores=value.tolist(),names=names,fit_split='train',fit_rows=len(train['y']),stable=stable,permutation_seeds=[101,102,103],cache_sha256=digest(CACHE/'manifest.json'))
    write(target,state)
    pd.DataFrame({'feature':names,'score':value}).to_csv(HERE/'results'/('stable_feature_scores.csv' if stable else 'feature_scores.csv'),index=False)
    return value.tolist()
