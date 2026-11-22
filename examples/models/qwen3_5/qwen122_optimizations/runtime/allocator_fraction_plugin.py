"""Isolated native allocator limit; no model, tensor, or arithmetic changes."""
import hashlib
import json
import os
import torch
import torch_musa
from pathlib import Path
from torch_musa.core import memory

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


class AllocatorFraction(MegatronCallback):

    def on_train_begin(self):
        rank = int(os.environ['RANK'])
        device = int(os.environ['LOCAL_RANK'])
        fraction = float(os.environ['GDN_ALLOCATOR_FRACTION'])
        assert fraction == 0.90
        assert torch_musa.current_device() == device
        assert not os.environ.get('MUSA_LAUNCH_BLOCKING')
        assert not os.environ.get('CUDA_LAUNCH_BLOCKING')
        settings = memory._snapshot()['allocator_settings']
        assert settings['expandable_segments'] is True
        total = torch_musa.get_device_properties(device).total_memory
        before = memory.memory_stats(device)
        # Official API returns only after the allocator sets its fraction flag.
        memory.set_per_process_memory_fraction(fraction, device)
        result = {
            'rank': rank,
            'device': device,
            'fraction': fraction,
            'total_memory': total,
            'allocator_limit_bytes': int(total * fraction),
            'set_returned': True,
            'allocator_settings': settings,
            'reserved_before': before['reserved_bytes.all.current'],
            'allocated_before': before['allocated_bytes.all.current'],
            'plugin_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'api_file': memory.__file__,
            'api_sha256': hashlib.sha256(Path(memory.__file__).read_bytes()).hexdigest(),
            'launch_blocking': [os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']],
        }
        assert result['reserved_before'] < result['allocator_limit_bytes']
        root = Path(self.args.output_dir)
        (root / f'allocator-fraction-rank{rank}.json').write_text(json.dumps(result, indent=2))


megatron_callbacks_map['allocator_fraction'] = AllocatorFraction
