"""Extract the pinned original extension without modifying its source."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--runtime', type=Path, required=True)
args = parser.parse_args()
package = Path(__file__).resolve().parent
assert hashlib.sha256(
    args.archive.read_bytes()).hexdigest() == '2d3c65f34d05bf35bb50364a4c5d0af09538750f05adf39afe5c61102d64f727'
runtime = args.runtime.resolve()
runtime.mkdir(parents=True, exist_ok=True)
source = runtime / 'extension'
if not source.exists():
    unpack = runtime / 'unpack'
    unpack.mkdir(exist_ok=False)
    with tarfile.open(args.archive) as archive:
        archive.extractall(unpack, filter='data')
    shutil.move(str(unpack / 'torch_musa_op_extension-master'), source)
    unpack.rmdir()
rel = 'torch_musa_op_extension/attention/tilelang/flash_linear_attention/'
assert hashlib.sha256(
    (source / rel /
     'gdn_chunk_bwd.py').read_bytes()).hexdigest() == '57b7297f992bb87db3404915b13314be13175861854c140d6c2caaed07acec31'
assert hashlib.sha256((source / rel / 'gdn_chunk_bwd_scan.py'
                       ).read_bytes()).hexdigest() == 'ee694553273e8192277ec389e65397a35811bf5ccab7757d82b4169353311fb0'
assert not (source / rel / 'gdn_zero_beta.py').exists()
# Captured choices contain version/source fingerprints and real tuning choices.
config_path = package / 'fla-configs.json'
if config_path.exists():
    configs = json.loads(config_path.read_text())
    for name, content in configs.items():
        path = runtime / 'fla-configs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for path in (runtime / 'fla-configs').glob('*/manifest.json'):
        spec = json.loads(path.read_text())
        for name, sha in spec['files'].items():
            assert hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == sha
print('Verified isolated runtime:', runtime)
