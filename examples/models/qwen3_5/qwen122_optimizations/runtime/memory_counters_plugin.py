"""Paired-run allocator counters and exception capture; no memory-history recording."""
import json
import os
import pickle
import time
import traceback
from pathlib import Path
import torch
import torch_musa
from torch_musa.core import memory
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


class MemoryDiagnostic(MegatronCallback):
    def stats(self, tag):
        row = {'time': time.time(), 'rank': self.rank, 'tag': tag,
               'iteration': self.state.iteration}
        try:
            row['stats'] = memory.memory_stats(self.device)
        except Exception as e:
            row['stats_error'] = repr(e)
        with (self.root/f'memory-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')

    def snapshot(self, tag):
        try:
            with (self.root/f'memory-{tag}-rank{self.rank}.pickle').open('wb') as f:
                pickle.dump(memory._snapshot(), f)
        except Exception:
            (self.root/f'memory-{tag}-rank{self.rank}.error').write_text(traceback.format_exc())

    def on_train_begin(self):
        self.root = Path(self.args.output_dir)
        self.rank = int(os.environ['RANK'])
        self.device = int(os.environ['LOCAL_RANK'])
        original = self.trainer.train_step

        def observed(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except BaseException:
                # Preserve the first Python exception before distributed cleanup.
                (self.root/f'exception-rank{self.rank}.txt').write_text(traceback.format_exc())
                self.stats('exception')
                self.snapshot('exception')
                raise

        self.trainer.train_step = observed
        self.stats('train_begin')

    def on_step_begin(self):
        memory.reset_peak_memory_stats(self.device)
        self.stats('step_begin')

    def on_step_end(self):
        self.stats('step_end')



megatron_callbacks_map['memory_counters'] = MemoryDiagnostic
