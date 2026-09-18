# Data for the final LLM assignments

`train.jsonl` (13 conversations) and `validation.jsonl` (2 conversations) are original,
fixed educational examples authored for TorchCode. They are released under CC0-1.0.
They include greetings, simple questions, and a multi-turn name recall example.
They are deliberately tiny: use them to debug, overfit, and demonstrate narrow replies.
Held-out loss on two examples is not a meaningful general-language benchmark.
The supplied `prepare_demo` helper can recreate these fixtures offline in local,
Docker, or Colab notebook working directories. It preserves existing files.

Each line is one of:

```json
{"text": "A document for next-token pretraining."}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

Use one complete document/conversation per record; a conversation must end with an
assistant turn. Supported roles are system, user and assistant. Split before
encoding; never use validation records for updates or tokenizer fitting.
The byte tokenizer has a fixed vocabulary and needs no fitting.

## Larger language and chat corpora (optional downloads)

- [TinyStories dataset and download files](https://huggingface.co/datasets/roneneldan/TinyStories):
  English short stories for language-model pretraining; dataset card lists CDLA-Sharing-1.0.
  Stories alone do not teach assistant dialogue formatting.
- [smol-smoltalk dataset and download files](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk):
  instruction conversations for subsequent supervised fine-tuning; consult its dataset
  card and license for source attribution and use terms.

Nothing large is downloaded by Run All or by the judge. Download only when ready
for a longer experiment. The following optional conversion writes bounded subsets
and records the immutable dataset revision and output checksums. Install the extra
`datasets` and `huggingface_hub` packages in your training environment first.
Run from the repository root, then point notebook 42 at the resulting files.

```python
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

out = Path('datasets/llm/external')
out.mkdir(parents=True, exist_ok=True)
manifest = {}
for name, repo, kind in [
    ('stories', 'roneneldan/TinyStories', 'text'),
    ('chat', 'HuggingFaceTB/smol-smoltalk', 'messages'),
]:
    revision = HfApi().dataset_info(repo).sha
    # First 10,200 records only; streaming avoids materializing the full corpus.
    rows = load_dataset(repo, split='train', revision=revision, streaming=True)
    seen, train, validation = set(), [], []
    for row in rows:
        record = {kind: row[kind]}
        if kind == 'messages' and record[kind][-1]['role'] != 'assistant':
            continue
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        (validation if len(validation) < 200 else train).append(text)
        if len(train) == 10_000:
            break
    assert train and validation
    files = {}
    for split, data in [('train', train), ('validation', validation)]:
        path = out / f'{name}_{split}.jsonl'
        path.write_text('\n'.join(data) + '\n', encoding='utf-8')
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest[name] = {'repo': repo, 'revision': revision, 'sha256': files}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
```

Keep the manifest and reuse its revision for repeated downloads; resolving the
latest revision again is a new experiment. This sequential split is an accessible
starting point, not a representative benchmark; design a held-out evaluation for
serious experiments and inspect duplicates/source grouping across larger corpora.

Pretrain using `stories_train.jsonl`, then continue the same model on
`chat_train.jsonl` at a smaller learning rate. Use their respective validation files.
Increase the context window beyond 128 bytes, inspect truncation and dropped
examples, and budget for larger models and many more training tokens. The example
subset sizes are for pipeline exploration, not enough to promise a capable assistant.
The graded baseline keeps the fixed byte tokenizer, architecture contract, CPU
fixtures and greedy decoder independent of these external data versions.
