#!/usr/bin/env python3
"""Convert LLaVA-Instruct-150K JSON to JSONL with absolute image paths."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def convert_sample(sample, image_dir, image_names, number):
    if not isinstance(sample, dict):
        raise ValueError(f'第 {number} 条样本不是 JSON 对象')
    image = sample.get('image')
    if not isinstance(image, str) or not image or Path(image).name != image:
        raise ValueError(f'第 {number} 条样本的 image 必须是原始图片文件名')
    turns = sample.get('conversations')
    if not isinstance(turns, list) or not turns:
        raise ValueError(f'第 {number} 条样本缺少 conversations')
    for turn in turns:
        if (not isinstance(turn, dict) or turn.get('from') not in ('human', 'gpt')
                or not isinstance(turn.get('value'), str)):
            raise ValueError(f'第 {number} 条样本的对话格式无效')

    if image in image_names:
        image_path = image_dir / image
    elif f'COCO_train2014_{image}' in image_names:
        image_path = image_dir / f'COCO_train2014_{image}'
    else:
        raise FileNotFoundError(f'第 {number} 条样本缺少图片：{image}')
    return {**sample, 'image': str(image_path)}


def preprocess(input_path, image_dir, output_path):
    input_path = Path(input_path).expanduser().resolve()
    image_dir = Path(image_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f'输出已存在：{output_path}')
    if not image_dir.is_dir():
        raise NotADirectoryError(f'图片目录不存在：{image_dir}')
    with input_path.open(encoding='utf-8') as stream:
        samples = json.load(stream)
    if not isinstance(samples, list) or not samples:
        raise ValueError('输入必须是非空的 LLaVA JSON 数组')
    with os.scandir(image_dir) as entries:
        image_names = {entry.name for entry in entries if entry.is_file()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    used_images = set()
    try:
        with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=output_path.parent,
                prefix=f'.{output_path.name}.',
                suffix='.tmp',
                delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for number, sample in enumerate(samples, 1):
                converted = convert_sample(sample, image_dir, image_names, number)
                stream.write(json.dumps(converted, ensure_ascii=False) + '\n')
                used_images.add(converted['image'])
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(samples), len(used_images), output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        default='/mnt_moer/wangmx/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json',
    )
    parser.add_argument(
        '--image-dir',
        default='/mnt_moer/wangmx/llava_instruct_150k_preprocessed/images',
    )
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        samples, images, output = preprocess(args.input, args.image_dir, args.output)
    except (OSError, ValueError) as error:
        print(f'错误：{error}', file=sys.stderr)
        return 1
    print(f'样本数：{samples}；图片数：{images}；输出：{output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
