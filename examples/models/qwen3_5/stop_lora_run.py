import os
import signal
import sys
from pathlib import Path

output = os.fsencode(sys.argv[1])
for entry in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = entry.read_bytes().split(b'\0')
        pid = int(entry.parent.name)
        is_training = (b'torch.distributed.run' in args or b'swift.cli._megatron.main' in args
                       or any(arg.endswith(b'/swift/cli/_megatron/sft.py') for arg in args))
        if pid != os.getpid() and output in args and is_training:
            os.kill(int(entry.parent.name), signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
