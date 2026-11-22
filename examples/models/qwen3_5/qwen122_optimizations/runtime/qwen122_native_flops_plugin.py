"""Log installed Megatron's unchanged default sequence-length FLOPs formula."""
import hashlib
import inspect
import json
import time
from megatron.training.training import num_floating_point_operations
from pathlib import Path
from types import SimpleNamespace

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


class NativeFlops(MegatronCallback):

    def on_train_begin(self):
        a = dict(vars(self.args))
        a.update(vars(self.trainer.config))
        a.update(
            num_experts=a['num_moe_experts'],
            swiglu=a['gated_linear_unit'],
            group_query_attention=a['num_query_groups'] != a['num_attention_heads'],
            seq_length=self.args.max_length)
        a.setdefault('padded_vocab_size', self.trainer.config.padded_vocab_size)
        self.flops = num_floating_point_operations(SimpleNamespace(**a), self.args.global_batch_size)
        self.last_time = time.perf_counter()
        self.last_iteration = self.state.iteration
        self.in_eval = False
        if self.args.rank == 0:
            source = inspect.getsource(num_floating_point_operations)
            (Path(self.args.output_dir) / 'native-flops.json').write_text(
                json.dumps(
                    {
                        'source': inspect.getsourcefile(num_floating_point_operations),
                        'sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'flops_per_step': self.flops,
                        'convention': 'Megatron native formula, default seq_length=4096, no LoRA correction; '
                        'not actual packed token accounting',
                        'timing': 'training log interval wall time; no added GPU synchronization',
                    },
                    indent=2))

    def on_eval_begin(self):
        self.in_eval = True

    def on_eval_end(self):
        self.in_eval = False

    def on_log(self, logs):
        if self.in_eval:
            return
        now = time.perf_counter()
        steps = self.state.iteration - self.last_iteration
        if steps <= 0:
            return
        seconds = now - self.last_time
        logs['megatron_tflops_per_gpu'] = self.flops * steps / seconds / self.args.world_size / 1e12
        logs['perf_interval_seconds'] = seconds
        self.last_time, self.last_iteration = now, self.state.iteration


megatron_callbacks_map['native_flops'] = NativeFlops
