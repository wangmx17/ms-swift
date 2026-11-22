"""A2 opt-in packed-boundary reuse; installed packages remain unchanged."""
import hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from packed_spans_cache import PackedSpansCache
from training_musa_adaptor.patches.megatron import attention as ap
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map
original = ap._packed_spans
cache = PackedSpansCache(original)
EXPECTED = '8e16341f5726b0a79048300a9f1817745c1c65d969f500d8af621f220588bbad'
class AttentionMetadataAudit(MegatronCallback):
    def on_train_begin(self):
        self.mode = os.environ['BWY_ATTENTION_METADATA_MODE']
        assert self.mode in ('baseline','cached')
        assert hashlib.sha256(Path(ap.__file__).read_bytes()).hexdigest() == EXPECTED
        assert ap._packed_spans is original
        if self.mode == 'cached': ap._packed_spans = cache
        self.root = Path(self.args.output_dir)
        self.rank = int(os.environ['RANK'])
        blocking = [os.environ.get(x) for x in ['MUSA_LAUNCH_BLOCKING','CUDA_LAUNCH_BLOCKING']]
        assert blocking == [None,None]
        record = dict(rank=self.rank, mode=self.mode, launch_blocking=blocking,
                      source_sha256=EXPECTED, max_entries=cache.max_entries,
                      plugin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      cache_sha256=hashlib.sha256(Path(__file__).with_name('packed_spans_cache.py').read_bytes()).hexdigest())
        (self.root/f'attention-metadata-rank{self.rank}.json').write_text(json.dumps(record,indent=2))
    def on_step_begin(self): cache.counts.clear()
    def on_step_end(self):
        with (self.root/f'attention-metadata-calls-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.state.iteration,entries=len(cache.entries),**cache.counts))+chr(10))
megatron_callbacks_map['attention_metadata_audit'] = AttentionMetadataAudit
