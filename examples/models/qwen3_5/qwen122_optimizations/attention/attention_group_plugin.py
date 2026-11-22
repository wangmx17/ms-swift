"""A1 strict 65..256 packed sequences, group4; current-stack numerical proof required."""
import hashlib, importlib.metadata, json, os
from pathlib import Path
import torch
from mate import flash_attn_varlen_func as fn
from training_musa_adaptor.patches.megatron import attention as ap
from training_musa_adaptor.ops import attention as ao
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map
original = ap._packed_forward
GROUP_SIZE = 4
counts = {'fast':0,'fallback':0,'native_calls':0,'sequences':0}
def chunked_packed_forward(m,q,k,v,mask,p,upstream):
 def fallback():
  counts['fallback']+=1
  return original(m,q,k,v,mask,p,upstream)
 if (ap._frozen_selection()[0].policy!='auto' or q.device.type!='musa' or any(x.dtype!=torch.bfloat16 for x in [q,k,v]) or any(x.ndim!=3 for x in [q,k,v]) or tuple(q.shape[1:])!=(32,256) or tuple(k.shape[1:])!=(2,256) or tuple(v.shape[1:])!=(2,256) or k.shape!=v.shape or mask not in ['causal','padding_causal','causal_bottom_right','padding_causal_bottom_right'] or ap._effective_dropout(m)!=0 or bool(getattr(m,'deterministic',False))):return fallback()
 qs=ap._packed_spans(p.cu_seqlens_q,getattr(p,'cu_seqlens_q_padded',None),q.shape[0]);ks=ap._packed_spans(p.cu_seqlens_kv,getattr(p,'cu_seqlens_kv_padded',None),k.shape[0])
 if qs!=ks or not qs or any(not(65<=n<=256) or n!=cap for _,n,cap in qs):return fallback()
 counts['fast']+=1;counts['sequences']+=len(qs)
 outputs=[]
 for i in range(0,len(qs),GROUP_SIZE):
  group=qs[i:i+GROUP_SIZE];start=group[0][0];end=group[-1][0]+group[-1][1]
  qq,kk,vv=[x[start:end].clone() for x in [q,k,v]]
  cu=[0]
  for _,n,_ in group:cu.append(cu[-1]+n)
  cu=torch.tensor(cu,device=q.device,dtype=torch.int32);maxlen=max(n for _,n,_ in group)
  o=fn(qq,kk,vv,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=maxlen,max_seqlen_k=maxlen,softmax_scale=getattr(m,'softmax_scale',None),causal=True,deterministic=False,**ao._mate_explicit_backends())
  if o.requires_grad:o.register_hook(lambda g:g.contiguous())
  outputs.append(o);counts['native_calls']+=1
 return torch.cat(outputs).reshape(q.shape[0],-1)

class AttentionGroupAudit(MegatronCallback):
    def on_train_begin(self):
        mode = os.environ['BWY_ATTENTION_GROUP_MODE']
        assert mode in ['native','group4']
        assert importlib.metadata.version('mate') == '0.2.8.post2'
        assert hashlib.sha256(Path(ap.__file__).read_bytes()).hexdigest() == '8e16341f5726b0a79048300a9f1817745c1c65d969f500d8af621f220588bbad'
        assert hashlib.sha256(Path(ao.__file__).read_bytes()).hexdigest() == '80fcb0188856c30e04f75625f18abd82634f79dd4b972f22cc4484790d21924f'
        assert ap._packed_forward is original
        if mode == 'group4':ap._packed_forward = chunked_packed_forward
        self.rank = int(os.environ['RANK']); self.root = Path(self.args.output_dir)
        record = dict(rank=self.rank,mode=mode,group_size=4,mate=importlib.metadata.version('mate'),
                      plugin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(Path(ap.__file__).read_bytes()).hexdigest(),
                      launch_blocking=[os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING','CUDA_LAUNCH_BLOCKING']])
        assert record['launch_blocking'] == [None,None]
        (self.root/f'attention-group-rank{self.rank}.json').write_text(json.dumps(record,indent=2))
    def on_step_begin(self):
        for k in counts:counts[k]=0
    def on_step_end(self):
        with (self.root/f'attention-group-calls-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.state.iteration,**counts))+'\n')
megatron_callbacks_map['attention_group_audit'] = AttentionGroupAudit
