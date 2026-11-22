"""T1 isolated process-local integration. Numerical and fullmodel tests still required."""
import hashlib
import inspect
import json
import os
import torch
import types
from chunked_head_function import ChunkedHeadLoss
from pathlib import Path

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


class ChunkedHeadAudit(MegatronCallback):

    def on_train_begin(self):
        from mcore_bridge.model.gpt_model import GPTModel
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
        mode = os.environ['BWY_CHUNKED_HEAD_MODE']
        assert mode in ['native', 'chunk512']
        assert all(os.environ.get(k) is None for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING'])
        models = [m for root in self.trainer.unwrapped_models for m in root.modules() if isinstance(m, GPTModel)]
        assert len(models) == 1
        rows = []
        for model in models:
            c = model.config
            head = model.output_layer
            assert type(head) is ColumnParallelLinear, type(head)
            assert not any(p.requires_grad for p in head.parameters())
            assert head.bias is None and not head.sequence_parallel
            assert head.weight.shape == (248320, 3072) and head.weight.dtype == torch.bfloat16
            assert c.tensor_model_parallel_size == 1 and not c.sequence_parallel
            assert c.task_type == 'causal_lm' and c.cross_entropy_loss_fusion
            assert c.cross_entropy_fusion_impl == 'te' and not c.mtp_num_layers
            assert model.post_process and not model.mtp_process
            assert not model.share_embeddings_and_output_weights
            original = model._postprocess
            head_call = model._forward_output_layer
            loss_call = model.compute_language_model_loss
            assert not original.__func__.__closure__
            logger_enabled = original.__func__.__globals__['has_config_logger_enabled']
            assert not logger_enabled(c)
            signature = inspect.signature(original)
            counts = {'chunk_calls': 0, 'native_calls': 0}

            def postprocess(obj,
                            *args,
                            _original=original,
                            _head=head_call,
                            _loss=loss_call,
                            _signature=signature,
                            _counts=counts,
                            _logger=logger_enabled,
                            **kwargs):
                bound = _signature.bind(*args, **kwargs)
                bound.apply_defaults()
                v = bound.arguments
                # Preserve original inference/evaluation behavior.
                if not obj.training or v['labels'] is None:
                    _counts['native_calls'] += 1
                    return _original(*args, **kwargs)
                assert v.get('inference_context') is None and v.get('inference_params') is None
                assert not _logger(obj.config) and not obj.mtp_process
                x, labels = v['hidden_states'], v['labels']
                assert x.ndim == 3 and x.shape[1:] == (1, 3072)
                assert x.dtype == torch.bfloat16 and x.requires_grad
                assert labels.shape == (1, x.shape[0])
                weight = obj.output_layer.weight
                assert not weight.requires_grad

                def original_chunk_loss(part, frozen, target):
                    assert frozen is obj.output_layer.weight
                    logits = _head(part, weight=None, runtime_gather_output=v['runtime_gather_output'])
                    return _loss(target.transpose(0, 1).contiguous(), logits).transpose(0, 1).contiguous()

                _counts['chunk_calls'] += 1
                with torch.profiler.record_function('T1_frozen_head_ce_chunk512'):
                    return ChunkedHeadLoss.apply(x, weight,
                                                 labels.transpose(0, 1).contiguous(), 512,
                                                 original_chunk_loss).transpose(0, 1).contiguous()

            if mode == 'chunk512':
                model._postprocess = types.MethodType(postprocess, model)
            rows.append(
                dict(
                    head_type=type(head).__name__,
                    weight_shape=list(head.weight.shape),
                    weight_requires_grad=head.weight.requires_grad,
                    original_postprocess_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
                    mode=mode))
        self._counts = counts
        self._root = Path(self.args.output_dir)
        self._rank = int(os.environ['RANK'])
        payload = dict(
            rank=self._rank,
            mode=mode,
            models=rows,
            plugin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            function_sha256=hashlib.sha256(Path(inspect.getfile(ChunkedHeadLoss)).read_bytes()).hexdigest(),
            launch_blocking=[os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']])
        (self._root / f'chunked-head-rank{self._rank}.json').write_text(json.dumps(payload, indent=2))

    def on_step_end(self):
        with (self._root / f'chunked-head-calls-rank{self._rank}.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.state.iteration, **self._counts)) + '\n')


megatron_callbacks_map['chunked_head_audit'] = ChunkedHeadAudit
