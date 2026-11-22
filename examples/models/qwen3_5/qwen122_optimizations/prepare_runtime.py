"""Install pinned patches in a separate runtime; never edit site-packages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--runtime', type=Path, required=True)
args = parser.parse_args()
package = Path(__file__).resolve().parent
assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == '2d3c65f34d05bf35bb50364a4c5d0af09538750f05adf39afe5c61102d64f727'
runtime = args.runtime.resolve()
runtime.mkdir(parents=True, exist_ok=True)
source = runtime / 'extension'
retained = runtime / 'retained'
patches = package / 'patches'
if not source.exists():
    unpack = runtime / 'unpack'
    unpack.mkdir(exist_ok=False)
    with tarfile.open(args.archive) as archive:
        archive.extractall(unpack, filter='data')
    shutil.move(str(unpack / 'torch_musa_op_extension-master'), source)
    unpack.rmdir()
    for label in ['zero-beta', 'saved-state-recompute']:
        patch = patches / (label + '.patch')
        if not patch.exists():
            continue
        expected = json.loads((patches / (label + '-manifest.json')).read_text())
        for name, hashes in expected.items():
            path = source / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            assert actual == hashes['before'], (name, actual)
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)], cwd=source, check=True)
        for name, hashes in expected.items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == hashes['after'], name
        if label == 'zero-beta':
            shutil.copytree(source, retained)
expected = {}
for label in ['zero-beta', 'saved-state-recompute']:
    path = patches / (label + '-manifest.json')
    if path.exists():
        expected.update(json.loads(path.read_text()))
for name, hashes in expected.items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == hashes['after'], name
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
