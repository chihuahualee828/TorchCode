"""Convert manually downloaded small corpora into the capstone JSONL schemas.

Run `python scripts/prepare_llm_data.py {pretrain,sft,preference}` from repo root.
Downloads and optional pyarrow installation are described in datasets/llm/README.md.
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets' / 'llm'
SOURCES = {
    'pretrain': ROOT / 'pretrain/raw/TinyStoriesV2-GPT4-valid.txt',
    'sft': ROOT / 'sft/raw/train-00000-of-00001.parquet',
    'preference': ROOT / 'preference/raw/test_prefs-00000-of-00001.parquet',
}


def write_atomic(path, content):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.converted-')
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def records(stage, path):
    if stage == 'pretrain':
        text = path.read_text(encoding='utf-8')
        separator = '<|endoftext|>'
        if separator not in text:
            raise ValueError('Expected TinyStories story separators')
        for story in text.split(separator):
            story = story.strip()
            if story:
                yield {'text': story}, story
        return
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Install pyarrow to convert the Parquet files: pip install pyarrow') from exc
    for row in pq.read_table(path).to_pylist():
        if stage == 'sft':
            messages = row.get('messages')
            if not messages or messages[-1]['role'] != 'assistant':
                continue
            if any(m['role'] not in ('system', 'user', 'assistant') or not m['content'] for m in messages):
                continue
            yield {'messages': messages}, row.get('prompt_id') or json.dumps(messages, sort_keys=True)
        else:
            chosen, rejected = row.get('chosen'), row.get('rejected')
            if not chosen or not rejected or chosen[:-1] != rejected[:-1]:
                continue
            if chosen[-1]['role'] != rejected[-1]['role'] or chosen[-1]['role'] != 'assistant':
                continue
            prompt = chosen[:-1]
            if not prompt or prompt[-1]['role'] != 'user':
                continue
            positive, negative = chosen[-1]['content'], rejected[-1]['content']
            if not positive or not negative or positive == negative:
                continue
            yield {'prompt': prompt, 'chosen': positive, 'rejected': negative}, (row.get('prompt_id') or json.dumps(prompt, sort_keys=True))


def convert(stage, source=None, limit=0, force=False):
    source = Path(source) if source else SOURCES[stage]
    if not source.is_file():
        raise FileNotFoundError(f'Manually download the dataset to {source}; see datasets/llm/README.md')
    directory = ROOT / stage
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f'{name}.jsonl' for name in ('train', 'validation')}
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError('Converted files already exist; pass --force to replace them')
    output = {'train': [], 'validation': []}
    seen = set()
    for record, group in records(stage, source):
        key = hashlib.sha256(str(group).encode('utf-8')).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        split = 'validation' if int(key[:8], 16) % 20 == 0 else 'train'
        output[split].append(record)
        if limit and sum(map(len, output.values())) >= limit:
            break
    if not all(output.values()):
        raise ValueError('Both splits need records; raise --limit or inspect the source file')
    for split, path in paths.items():
        write_atomic(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output[split]))
    manifest = {'stage': stage, 'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'train': len(output['train']), 'validation': len(output['validation'])}
    write_atomic(directory / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
    print(f"{stage}: {manifest['train']} train, {manifest['validation']} validation -> {directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=SOURCES)
    parser.add_argument('--source', type=Path, help='Override the documented raw file path')
    parser.add_argument('--limit', type=int, default=0, help='Maximum source records; 0 means all')
    parser.add_argument('--force', action='store_true', help='Replace converted output')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    convert(args.stage, args.source, args.limit, args.force)
