import copy, numpy as np, torch
from sklearn.metrics import average_precision_score
from experiments.ft_upgrade_20261010.train import build as original_build
from .common import HERE,read,write,scores
from .models import build,mask
def run():
    torch.set_num_threads(4); torch.manual_seed(42)
    meta={'numeric_output_names':['a','b','c'],'categorical':['d','e'],'cat_cardinalities':[4,5]}
    cfg=read(HERE/'protocol.json')['base_config']|dict(width=16,heads=4,dropout=0.,depth=2,embedding='linear')
    num=torch.randn(4,3); cat=torch.tensor([[0,1],[1,2],[2,3],[3,4]])
    checks=[]
    for compression in [None,.25]:
        official=original_build(meta,cfg|dict(compression=compression,sdpa=False)).eval()
        patched=original_build(meta,cfg|dict(compression=compression,sdpa=True)).eval(); patched.load_state_dict(official.state_dict())
        a,b=official(num,cat),patched(num,cat); torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        a.sum().backward(); b.sum().backward()
        for pa,pb in zip(official.parameters(),patched.parameters()):
            if pa.grad is not None: torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-4)
        checks.append(dict(compression=compression,maximum_forward_error=float((a-b).abs().max())))
    allowed=mask([3,1,2,0,0]); assert allowed[0].all() and not allowed[1:,0].any() and allowed.diag().all()
    assert not allowed[1,2] and allowed[2,1]
    spa=build(meta,cfg|dict(compression=None,spa=True,dropout=.2),[3,1,2,0,0],True)
    spa.train(); spa(num,cat).sum().backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in spa.parameters())
    spa.eval(); torch.testing.assert_close(spa(num,cat),spa(num,cat),rtol=0,atol=0)
    y=np.array([1,0,0,1,0,0,0,0,0,0]*10); p=np.full(100,.5); ids=np.arange(100)[::-1]
    metric=scores(y,p,ids); assert metric['average_precision']==average_precision_score(y,p)
    assert metric['captured_top_1pct']==int(y[np.argmin(ids)]) and metric['top_1pct_count']==1
    write(HERE/'results/correctness.json',dict(status='PASS',attention_equivalence=checks,spa_direction_cls_last_query=True,finite_dropout_gradients=True,metric_equivalence=True,deterministic_replay=True))
    print('CPU CORRECTNESS PASS',flush=True)
if __name__=='__main__': run()
