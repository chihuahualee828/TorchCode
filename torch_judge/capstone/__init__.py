"""Provided infrastructure for the two final assignments (no model solution)."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


@dataclass
class LLMConfig:
    vocab_size: int = 260
    d_model: int = 64
    num_heads: int = 4
    num_kv_heads: int = 2
    num_layers: int = 2
    hidden_dim: int = 128
    max_seq_len: int = 256
    rope_base: float = 10000.0
    eps: float = 1e-5

    def __post_init__(self):
        values = (self.vocab_size, self.d_model, self.num_heads, self.num_kv_heads,
                  self.num_layers, self.hidden_dim, self.max_seq_len)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError('Dimensions must be positive integers')
        if self.d_model % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError('d_model divisible by heads; heads divisible by kv heads')
        if (self.d_model // self.num_heads) % 2:
            raise ValueError('RoPE needs an even head dimension')
        if self.eps <= 0 or self.rope_base <= 0:
            raise ValueError('eps and rope_base must be positive')


@contextmanager
def deterministic(seed=0):
    """CPU grading scope; restore caller RNG, thread count and algorithm settings."""
    threads = torch.get_num_threads()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.default_generator.manual_seed(seed)
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn)
            torch.set_num_threads(threads)


class ByteTokenizer:
    """Fixed UTF-8 vocabulary: PAD=0, BOS=1, EOS=2, SEP=3, bytes=4..259."""
    pad_id, bos_id, eos_id, sep_id, vocab_size = 0, 1, 2, 3, 260

    def encode(self, text):
        return [b + 4 for b in text.encode('utf-8')]

    def decode(self, ids):
        return bytes(int(i) - 4 for i in ids if 4 <= int(i) < 260).decode('utf-8', errors='replace')

    def prompt(self, messages):
        # Complete prior turns end in SEP; the next assistant role is supplied.
        ids = [self.bos_id]
        for m in messages:
            ids += self.encode(m['role'] + ': ' + m['content']) + [self.sep_id]
        return ids + self.encode('assistant: ')


def encode_record(record, tokenizer, assistant_only=True):
    """Unshifted ids/labels; mask roles and non-assistant text for SFT."""
    if 'text' in record:
        ids = [tokenizer.bos_id] + tokenizer.encode(record['text']) + [tokenizer.eos_id]
        return ids, ids.copy()
    ids, labels = [tokenizer.bos_id], [-100]
    messages = record['messages']
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('A training conversation must end with assistant')
    for i, m in enumerate(messages):
        if m['role'] not in ('system', 'user', 'assistant'):
            raise ValueError('Unsupported role')
        prefix = tokenizer.encode(m['role'] + ': ')
        body = tokenizer.encode(m['content'])
        end = [tokenizer.eos_id if i == len(messages) - 1 else tokenizer.sep_id]
        ids += prefix + body + end
        labels += [-100] * len(prefix)
        labels += body + end if m['role'] == 'assistant' else [-100] * (len(body) + 1)
    return ids, labels if assistant_only else ids.copy()


class TextDataset(Dataset):
    """One document/conversation per item; right pad, never mix split boundaries."""
    def __init__(self, records, max_seq_len, assistant_only=True):
        if max_seq_len < 2:
            raise ValueError('max_seq_len must be at least 2')
        self.items = []
        tok = ByteTokenizer()
        for record in records:
            ids, labels = encode_record(record, tok, assistant_only)
            ids, labels = ids[:max_seq_len], labels[:max_seq_len]
            if not any(x != -100 for x in labels[1:]):
                continue  # Truncation removed all targets: never train on NaN loss.
            pad = max_seq_len - len(ids)
            self.items.append((torch.tensor(ids + [tok.pad_id] * pad),
                               torch.tensor(labels + [-100] * pad)))
        if not self.items:
            raise ValueError('No supervised tokens; increase context or supply shorter records')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare_demo(directory):
    """Write original, fixed educational fixtures without a network dependency."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pairs = [('Hi', 'Hello!'), ('Hello', 'Hi!'), ('Bye', 'Goodbye!'),
             ('Thanks', 'You are welcome.'), ('What is your name?', 'I am Torch.'),
             ('What is two plus two?', 'Four.'), ('Name a color.', 'Blue.'),
             ('Name an animal.', 'A cat.'), ('What do plants need?', 'Water and light.'),
             ('What is ice?', 'Frozen water.'), ('Say yes.', 'Yes.'), ('Say no.', 'No.')]
    train = [{'messages': [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]} for q, a in pairs]
    train.append({'messages': [{'role': 'user', 'content': 'My name is Ada.'},
                              {'role': 'assistant', 'content': 'Hello, Ada.'},
                              {'role': 'user', 'content': 'What is my name?'},
                              {'role': 'assistant', 'content': 'Ada.'}]})
    valid = [{'messages': [{'role': 'user', 'content': 'Name a fruit.'},
                           {'role': 'assistant', 'content': 'An apple.'}]},
             {'messages': [{'role': 'user', 'content': 'Good morning'},
                           {'role': 'assistant', 'content': 'Hello!'}]}]
    for name, rows in [('train', train), ('validation', valid)]:
        path = directory / (name + '.jsonl')
        if not path.exists():
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return directory
