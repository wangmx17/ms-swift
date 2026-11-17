"""CPU-only checks for the Qwen3.8 example scripts."""
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples/models/qwen3_5'
spec = importlib.util.spec_from_file_location('qwen38_preprocess', EXAMPLES / 'preprocess_llava_qwen38.py')
preprocess_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_module)


class Qwen38ExamplesTest(unittest.TestCase):

    def test_preprocess_preserves_content_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            (images / '1.jpg').touch()
            (images / 'COCO_train2014_2.jpg').touch()
            records = [{
                'image':
                name,
                'id':
                name,
                'conversations': [{
                    'from': 'human',
                    'value': '<image>\n描述图片'
                }, {
                    'from': 'gpt',
                    'value': 'answer'
                }]
            } for name in ['1.jpg', '2.jpg']]
            source, output = root / 'input.json', root / 'output.jsonl'
            source.write_text(json.dumps(records))
            count, unique, _ = preprocess_module.preprocess(source, images, output)
            self.assertEqual((count, unique), (2, 2))
            actual = [json.loads(line) for line in output.read_text().splitlines()]
            for original, converted in zip(records, actual):
                self.assertEqual(original['conversations'], converted['conversations'])
                self.assertEqual(original['id'], converted['id'])
                self.assertTrue(Path(converted['image']).is_file())
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                preprocess_module.preprocess(source, images, output)
            self.assertEqual(output.read_bytes(), before)
            records.append({**records[0], 'image': 'missing.jpg'})
            source.write_text(json.dumps(records))
            failed = root / 'failed.jsonl'
            with self.assertRaises(FileNotFoundError):
                preprocess_module.preprocess(source, images, failed)
            self.assertFalse(failed.exists())
            self.assertFalse(list(root.glob('.failed.jsonl.*.tmp')))

    def test_training_defaults_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = root / 'megatron'
            command.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            command.chmod(0o755)
            env = {**os.environ, 'PATH': str(root) + ':' + os.environ['PATH'], 'OUTPUT_DIR': str(root / 'output')}
            for key in ['TP', 'PP', 'CP', 'MICRO_BATCH_SIZE', 'GLOBAL_BATCH_SIZE', 'TRAIN_ITERS']:
                env.pop(key, None)
            result = subprocess.check_output(['bash', str(EXAMPLES / 'mcore_qwen3_8_27b_full.sh')], env=env, text=True)
            args = json.loads(result)
            self.assertEqual(args[0], 'sft')
            flags = dict(zip(args[1::2], args[2::2]))
            expected = {
                '--tensor_model_parallel_size': '4',
                '--pipeline_model_parallel_size': '1',
                '--context_parallel_size': '1',
                '--sequence_parallel': 'true',
                '--micro_batch_size': '8',
                '--global_batch_size': '64',
                '--train_iters': '1000',
                '--max_length': '8192',
                '--freeze_vit': 'true',
                '--freeze_aligner': 'true',
                '--apply_rope_fusion': 'true',
                '--manual_gc': 'true',
                '--manual_gc_steps': '100'
            }
            for key, value in expected.items():
                self.assertEqual(flags[key], value)

    def test_stop_does_not_signal_another_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / 'examples/models/qwen3_5'
            scripts.mkdir(parents=True)
            launcher = scripts / 'mcore_qwen3_8_27b_4node_launch.sh'
            shutil.copyfile(EXAMPLES / launcher.name, launcher)
            train = scripts / 'mcore_qwen3_8_27b_full.sh'
            train.write_text('#!/usr/bin/env bash\nwhile true; do sleep 1; done\n')
            (scripts / 'hostfile').write_text('node0.example.invalid slots=8\n')
            ssh = root / 'ssh'
            ssh.write_text('#!/usr/bin/env python3\nimport subprocess,sys\n'
                           "sys.exit(subprocess.call(sys.argv[-1], shell=True, executable='/bin/bash'))\n")
            ssh.chmod(0o755)
            processes = []
            try:
                for name in ['target', 'other']:
                    processes.append(
                        subprocess.Popen(['bash', str(train)],
                                         start_new_session=True,
                                         env={
                                             **os.environ, 'OUTPUT_DIR': str(root / name)
                                         }))
                time.sleep(0.1)
                env = {
                    **os.environ, 'PATH': str(root) + ':' + os.environ['PATH'],
                    'OUTPUT_DIR': str(root / 'target'),
                    'HOSTFILE': str(scripts / 'hostfile')
                }
                subprocess.run(['bash', str(launcher), 'stop'], env=env, check=True, capture_output=True, timeout=10)
                self.assertEqual(processes[0].wait(timeout=5), -signal.SIGTERM)
                self.assertIsNone(processes[1].poll())
            finally:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
