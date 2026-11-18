import argparse
import json
import math
import torch
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Compare two optimizer_audit directories; no acceptance threshold is implied.')
parser.add_argument('reference', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--output', type=Path, required=True)
cli = parser.parse_args()
roots = {'plain': cli.reference, 'fused': cli.candidate}


def read(case, name):
    return json.loads((roots[case] / (name + '.rank0.json')).read_text())


def params(stage):
    a = torch.load(roots['plain'] / (stage + '.pt'), map_location='cpu', weights_only=True)
    b = torch.load(roots['fused'] / (stage + '.pt'), map_location='cpu', weights_only=True)
    if not a or a.keys() != b.keys():
        raise ValueError('Parameter dictionaries must be nonempty and have matching names.')
    max_abs = 0.
    sq = 0.
    base = 0.
    exact = True
    finite = True
    n = 0
    worst = ''
    for k in a:
        x = a[k].double()
        y = b[k].double()
        if x.shape != y.shape or a[k].dtype != b[k].dtype:
            raise ValueError(f'Parameter shape/dtype mismatch: {k}')
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError(f'Nonfinite parameter: {stage}/{k}')
        d = x - y
        v = d.abs().max().item()
        if v > max_abs:
            max_abs = v
            worst = k
        sq += d.square().sum().item()
        base += x.square().sum().item()
        n += x.numel()
        exact &= torch.equal(a[k], b[k])
        finite &= bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    return {
        'tensor_count': len(a),
        'numel': n,
        'bitwise_equal': exact,
        'finite': finite,
        'max_abs': max_abs,
        'rms': math.sqrt(sq / n),
        'relative_l2': math.sqrt(sq / base) if base else None,
        'worst_tensor': worst
    }


begins = {c: read(c, 'begin') for c in ['plain', 'fused']}
ends = {c: read(c, 'end') for c in ['plain', 'fused']}
argsdiff = {
    k: [begins['plain']['args'].get(k), begins['fused']['args'].get(k)]
    for k in set(begins['plain']['args']) | set(begins['fused']['args'])
    if begins['plain']['args'].get(k) != begins['fused']['args'].get(k)
}
logs = {c: {r['step']: r for r in ends[c]['log_history'] if 'loss' in r} for c in ends}
if not logs['plain'] or logs['plain'].keys() != logs['fused'].keys():
    raise ValueError('Nonempty matching logged steps are required.')
if ends['plain']['global_step'] != ends['fused']['global_step']:
    raise ValueError('Final optimizer steps differ.')
metrics = {}
for k in ['loss', 'grad_norm', 'learning_rate']:
    pairs = [(logs['plain'][s][k], logs['fused'][s][k]) for s in logs['plain']
             if k in logs['plain'][s] and k in logs['fused'][s]]
    if len(pairs) != len(logs['plain']):
        raise ValueError(f'Missing metric: {k}')
    if not all(math.isfinite(float(x)) and math.isfinite(float(y)) for x, y in pairs):
        raise ValueError(f'Nonfinite metric: {k}')
    metrics[k] = {
        'count': len(pairs),
        'finite': all(math.isfinite(float(x)) and math.isfinite(float(y)) for x, y in pairs),
        'max_abs': max((abs(float(x) - float(y)) for x, y in pairs), default=None)
    }
r = {
    'initial': params('initial'),
    'final': params('final'),
    'steps': {
        c: ends[c]['global_step']
        for c in ends
    },
    'logged_metrics': metrics,
    'args_diff': argsdiff,
    'begin': {
        c: {
            k: v
            for k, v in begins[c].items() if k != 'args'
        }
        for c in begins
    },
    'state': {
        c: read(c, 'state')
        for c in begins
    }
}
cli.output.write_text(json.dumps(r, indent=2))
print(json.dumps({k: v for k, v in r.items() if k not in ['begin']}, indent=2))
