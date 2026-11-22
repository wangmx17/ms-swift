"""Emit shell-quoted case plugin options from separate opt-in manifests."""
import json
from pathlib import Path
import shlex
root = Path(__file__).resolve().parent
plugins, callbacks = [], []
for group in ('lora', 'head', 'attention'):
    path = root / group / 'enable.json'
    if not path.exists():
        continue
    config = json.loads(path.read_text())
    for name in config['plugins']:
        target = root / name
        if not target.is_file():
            raise FileNotFoundError(target)
        plugins.append(str(target))
    callbacks.extend(config['callbacks'])
    for key, value in config['env'].items():
        if not key.isidentifier():
            raise ValueError(key)
        print('export ' + key + '=' + shlex.quote(value))
print('optional_plugins=(' + ' '.join(map(shlex.quote, plugins)) + ')')
print('optional_callbacks=(' + ' '.join(map(shlex.quote, callbacks)) + ')')
