import os
from pathlib import Path
R=Path(os.environ['BWY_RUNTIME_ROOT'])
# Each process gets an independent writable TileLang cache.
os.environ['TILELANG_CACHE_DIR']=str(R/'cache-ranks'/('rank-'+os.environ.get('RANK','launcher')))
import functools,json,hashlib,collections
import torch
from swift.megatron.callbacks import MegatronCallback,megatron_callbacks_map

class GDNTrace(MegatronCallback):
 def on_train_begin(self):
  self.rank=int(os.environ['RANK']);self.root=Path(self.args.output_dir);self.prof=None
  self.counts=collections.Counter();self.samples=[];self.seen=set()
  self.trace=os.environ.get('OPT_TRACE')=='1';bindings=[]
  import mcore_bridge.model.modules.gated_delta_net as bridge
  import megatron.core.ssm.gated_delta_net as mcore
  for module in [bridge,mcore]:
   fn=getattr(module,'chunk_gated_delta_rule',None)
   if fn is None:continue
   bindings.append({'module':module.__name__,'function_module':fn.__module__,'tilelang_dispatcher':bool(getattr(fn,'_training_musa_adaptor_tk_gdn',False))})
   if self.trace:
    def wrap(original,module_name):
     @functools.wraps(original)
     def observed(*args,**kwargs):
      self.counts['calls:'+module_name]+=1
      # Sample metadata during the first warmup step, outside the trace window.
      cu=kwargs.get('cu_seqlens');q=args[0] if args else kwargs['q']
      key=(cu.data_ptr(),tuple(q.shape)) if cu is not None else (None,tuple(q.shape))
      if self.rank==0 and self.state.iteration==0 and key not in self.seen and len(self.samples)<8:
       self.seen.add(key)
       offsets=cu.detach().cpu().tolist() if cu is not None else None
       self.samples.append({'module':module_name,'q_shape':list(q.shape),'q_dtype':str(q.dtype),'g_dtype':str(kwargs.get('g',args[3] if len(args)>3 else None).dtype),'beta_dtype':str(kwargs.get('beta',args[4] if len(args)>4 else None).dtype),'cu_seqlens':offsets})
      with torch.profiler.record_function('gdn_binding/'+module_name):return original(*args,**kwargs)
     return observed
    setattr(module,'chunk_gated_delta_rule',wrap(fn,module.__name__))
  if self.trace:
   import training_musa_adaptor.ops.gated_delta_rule as op
   original_check=op._supported_by_tilelang
   @functools.wraps(original_check)
   def checked(*args,**kwargs):
    with torch.profiler.record_function('gdn_dispatch_shape_check'):
     ok=original_check(*args,**kwargs)
    self.counts['tilelang_eligible' if ok else 'fla_fallback']+=1
    return ok
   op._supported_by_tilelang=checked
  cfg={k:getattr(self.trainer.config,k,None) for k in ['recompute_granularity','recompute_method','recompute_num_layers','moe_shared_expert_overlap','cross_entropy_fusion_impl','moe_router_fusion','tensor_model_parallel_size','pipeline_model_parallel_size','expert_model_parallel_size']}
  import torch_musa_op_extension as tmoe
  extension_root=Path(tmoe.__file__).parent
  cfg['extension_root']=str(extension_root)
  cfg['fix_sources']={name:hashlib.sha256((extension_root/'attention/tilelang/flash_linear_attention'/name).read_bytes()).hexdigest() for name in ['gdn_chunk_bwd.py','gdn_chunk_bwd_scan.py','gdn_zero_beta.py']}
  cfg['gdn_classes']=[{'name':name,'class_module':type(module).__module__,'forward_module':getattr(module.forward,'__module__',None)} for model in self.trainer.unwrapped_models for name,module in model.named_modules() if 'GatedDelta' in type(module).__name__]
  cfg['allocator_config']=os.environ.get('PYTORCH_MUSA_ALLOC_CONF')
  cfg['allocator_runtime_settings']=torch.musa.memory._snapshot()['allocator_settings']
  cfg.update(rank=self.rank,case=os.environ['GDN_CASE'],bindings=bindings,trace=self.trace,with_stack=True,record_shapes=False,manual_gc=self.args.manual_gc,manual_gc_steps=self.args.manual_gc_steps,global_batch_size=self.args.global_batch_size,micro_batch_size=self.args.micro_batch_size,disabled=os.environ.get('TRAINING_MUSA_ADAPTOR_DISABLE'),cache=os.environ.get('TILELANG_CACHE_DIR'),launch_blocking=[os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING','CUDA_LAUNCH_BLOCKING']])
  (self.root/f'gdn-effective-rank{self.rank}.json').write_text(json.dumps(cfg,indent=2))
 def on_step_begin(self):
  self.counts.clear()
  if self.rank==0 and self.trace and self.state.iteration==4:
   self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.PrivateUse1],with_stack=True,record_shapes=False,profile_memory=False)
   self.prof.__enter__()
 def on_step_end(self):
  if self.trace:
   with (self.root/f'gdn-calls-rank{self.rank}.jsonl').open('a') as f:f.write(json.dumps({'step':self.state.iteration,**self.counts})+'\n')
   if self.rank==0 and self.state.iteration==1:(self.root/'gdn-input-samples.json').write_text(json.dumps(self.samples,indent=2))
  if self.prof is not None and self.state.iteration==5:
   self.prof.__exit__(None,None,None)
   self.prof.export_chrome_trace(str(self.root/'rank0-step5.trace.json'))
   self.prof=None
megatron_callbacks_map['gdn_trace']=GDNTrace
