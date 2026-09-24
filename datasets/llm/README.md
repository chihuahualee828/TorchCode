# Small datasets for the three LLM training stages

The notebooks run offline with tiny, original `demo_*.jsonl` fixtures. They check the
pipeline, not language quality. For a more meaningful experiment, **manually** download
one source file per stage. No notebook or judge downloads data. The same byte tokenizer,
model configuration and model weights must continue across all three stages.

| Stage | Manual download | Save as | Approx. size | Source license |
|---|---|---|---:|---|
| 43 pretrain | [TinyStories V2 GPT-4 story file](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt?download=true) | `pretrain/raw/TinyStoriesV2-GPT4-valid.txt` | 22.5 MB | CDLA-Sharing-1.0 |
| 44 SFT | [No Robots train Parquet](https://huggingface.co/datasets/HuggingFaceH4/no_robots/resolve/main/data/train-00000-of-00001.parquet?download=true) | `sft/raw/train-00000-of-00001.parquet` | 10.5 MB | CC BY-NC 4.0 |
| 45 DPO | [UltraFeedback binarized preference Parquet](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/resolve/main/data/test_prefs-00000-of-00001.parquet?download=true) | `preference/raw/test_prefs-00000-of-00001.parquet` | 7.29 MB | MIT |

The TinyStories file is the publisher's *validation* file and the UltraFeedback file is
its *test_prefs* split. Here they are repurposed as small **source pools**, then split
locally into training and validation. Do not report scores against the publishers'
original validation/test splits as held-out benchmarks. The converter groups by
story/prompt ID before splitting and records the source SHA-256 in `manifest.json`.
The original SFT fixture has 13 training and two validation conversations.

From the repository root, after placing the downloads at the paths above:

```bash
python scripts/prepare_llm_data.py pretrain
python -m pip install pyarrow  # only needed for the two Parquet conversions
python scripts/prepare_llm_data.py sft
python scripts/prepare_llm_data.py preference
```

The converter writes `train.jsonl` and `validation.jsonl` beside each stage's `raw/`
folder. These external files and `artifacts/llm/` checkpoints are gitignored. Each
notebook automatically uses the converted split when present, otherwise its offline
demo split. Use `--limit N` to convert fewer examples or `--force` to replace a
previous conversion. Inspect the records, truncation rate and source licenses before
long training runs.

For a longer TinyStories experiment, change the configuration in notebook 43
*before* training—for example, `d_model=256`, `num_layers=4`, `num_heads=8`,
`num_kv_heads=2`, `hidden_dim=768`, `max_seq_len=512`—and run enough optimizer
steps to see held-out loss improve. Notebook 44 and 45 load that saved config
without changing tokenizer or context. This is a few-million-parameter model;
GPU time is recommended, and coherent simple-story snippets are a reasonable
experiment goal, not guaranteed general understanding or chat. The notebook's
small default keeps the offline exercise fast and its samples may be nonsense.

The JSONL schemas are:

```json
{"text": "A document of natural language for next-token pretraining."}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"prompt": [{"role": "user", "content": "Hi"}], "chosen": "Hello!", "rejected": "Goodbye!"}
```

Pretraining predicts every next token in raw text. SFT predicts only assistant
responses conditioned on prompts. DPO compares whole chosen/rejected completion
log-probabilities against a frozen SFT reference. It does not need reward-model
training or GRPO sampling. A 22.5 MB story file and a small decoder will not yield
general human-language understanding; increase model capacity, context, data quality,
training tokens and compute if your goal is fluent open-ended output. Use held-out
loss/perplexity and generated samples to assess progress rather than a fixed answer.
