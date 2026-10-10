import types
import torch
from torch.nn import functional as F
from rtdl_revisiting_models import MultiheadAttention
from experiments.ft_upgrade_20261010.train import build as original_build
def mask(scores):
    s=torch.cat([torch.tensor([-float('inf')],dtype=torch.float64),torch.as_tensor(scores,dtype=torch.float64)])
    if not torch.isfinite(s[1:]).all(): raise ValueError('Nonfinite importance')
    return s[:,None]<=s[None,:]
def spa_forward(self,x_q,x_kv):
    if self.key_compression is not None: raise ValueError('SPA incompatible with compression')
    q,k,v=self.W_q(x_q),self.W_k(x_kv),self.W_v(x_kv)
    b,n,d=q.shape; h=self._n_heads
    q=q.reshape(b,n,h,-1).transpose(1,2); k=k.reshape(b,k.shape[1],h,-1).transpose(1,2); v=v.reshape(b,v.shape[1],h,-1).transpose(1,2)
    if n not in [1,len(self.spa_allowed)]: raise ValueError('Unexpected CLS layout')
    z=F.scaled_dot_product_attention(q,k,v,attn_mask=self.spa_allowed[:n],dropout_p=self.dropout.p if self.training and self.dropout is not None else 0.)
    z=z.transpose(1,2).reshape(b,n,d)
    return z if self.W_out is None else self.W_out(z)
def build(meta,cfg,importance=None,matched_init=False):
    torch.manual_seed(42)
    canonical=original_build(meta,cfg|{'compression':None,'spa':False,'iai':False}) if matched_init else None
    torch.manual_seed(42)
    model=original_build(meta,cfg)
    if canonical is not None:
        base=canonical.state_dict(); state=model.state_dict()
        for k in state:
            if k in base and state[k].shape==base[k].shape: state[k]=base[k]
        model.load_state_dict(state)
        del canonical
    if cfg.get('spa'):
        if importance is None or len(importance)!=len(meta['numeric_output_names'])+len(meta['categorical']): raise ValueError('Wrong MI/token order')
        allowed=mask(importance)
        for m in model.modules():
            if isinstance(m,MultiheadAttention):
                m.register_buffer('spa_allowed',allowed.clone()); m.forward=types.MethodType(spa_forward,m)
    if cfg.get('iai') and matched_init:
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m,MultiheadAttention):
                    for w in [m.W_q,m.W_k,m.W_v]:
                        w.weight.mul_(.1)
                        if w.bias is not None: w.bias.zero_()
    return model
