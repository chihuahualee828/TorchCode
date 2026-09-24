"""Run from repo root: .venv/bin/python scripts/tests/test_llm_capstone.py.

Executes four reference notebooks offline in a temporary workspace and runs every
capstone judge case, checkpoint handoff checks, data checks, and negative controls.
"""
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch_judge.capstone import ByteTokenizer, PretrainDataset, PreferenceDataset, TextDataset, read_jsonl
from torch_judge.capstone.grading import model_case, training_case, cache_case, sft_case, preference_case
from torch_judge.tasks import get_task


def execute(path, namespace):
    for cell in json.loads(path.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            header, body = source.split('\n', 1)
            Path(header.split()[1]).write_text(body)
        else:
            source = '\n'.join(line for line in source.splitlines() if not line.strip().startswith('check('))
            exec(compile(source, str(path), 'exec'), namespace)


def judge(task_id, cls):
    task = get_task(task_id)
    for case in task['tests']:
        exec(case['code'].replace('{fn}', task['function_name']), {task['function_name']: cls})
    return len(task['tests'])


def must_fail(action):
    try:
        action()
    except (AssertionError, RuntimeError):
        return
    raise AssertionError('Negative control unexpectedly passed')


def renamed_mini_llm():
    """The model checker must not depend on reference attribute names."""
    source = Path('mini_llm_reference.py').read_text()
    replacements = [
        ('self.embed_tokens', 'self.word_table'), ('self.layers', 'self.tower'),
        ('self.attn_norm', 'self.normal_a'), ('self.ffn_norm', 'self.normal_b'),
        ('self.q_proj', 'self.question'), ('self.k_proj', 'self.key'),
        ('self.v_proj', 'self.value'), ('self.o_proj', 'self.merge'),
        ('self.gate_proj', 'self.gate'), ('self.up_proj', 'self.expand'),
        ('self.down_proj', 'self.compress'), ('self.lm_head', 'self.vocab_head'),
        ('self.norm', 'self.output_norm'),
    ]
    for old, new in replacements:
        source = source.replace(old, new)
    namespace = {}
    exec(compile(source, '<renamed_mini_llm>', 'exec'), namespace)
    return namespace['MiniLLM']


def same_state(actual, expected):
    assert set(actual) == set(expected)
    assert all(torch.equal(v, expected[k]) for k, v in actual.items())


def main():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        sys.path.insert(0, tmp)
        shutil.copytree(ROOT / 'datasets/llm', Path('datasets/llm'), ignore=shutil.ignore_patterns('raw', 'train.jsonl', 'validation.jsonl', 'manifest.json'))
        try:
            ns = {}
            execute(ROOT / 'solutions/42_mini_llm_solution.ipynb', ns)
            execute(ROOT / 'solutions/43_llm_training_solution.ipynb', ns)
            model_cls, pretrain_cls = ns['MiniLLM'], ns['LLMTrainer']
            a = ns['run_pretraining'](epochs=2)
            b = ns['run_pretraining'](epochs=2)
            assert a[2] == b[2]
            same_state(a[0].state_dict(), b[0].state_dict())
            base = torch.load('artifacts/llm/base_reference.pt', weights_only=True)
            assert base['stage'] == 'pretrain'
            execute(ROOT / 'solutions/44_llm_sft_solution.ipynb', ns)
            sft_cls = ns['SFTTrainer']
            same_state(ns['run_sft'](epochs=0)[0].state_dict(), base['model_state'])
            instruction = torch.load('artifacts/llm/instruction_reference.pt', weights_only=True)
            assert instruction['stage'] == 'sft'
            assert sft_cls.reply(ns['model'], ByteTokenizer(), [{'role':'user','content':'Hi'}]) == sft_cls.reply(ns['model'], ByteTokenizer(), [{'role':'user','content':'Hi'}])
            execute(ROOT / 'solutions/45_llm_dpo_solution.ipynb', ns)
            dpo_cls = ns['DPOTrainer']
            same_state(ns['run_dpo'](epochs=0)[0].state_dict(), instruction['model_state'])
            same_state(ns['reference'].state_dict(), instruction['model_state'])
            aligned = torch.load('artifacts/llm/aligned_reference.pt', weights_only=True)
            assert aligned['stage'] == 'dpo'
            total = 0
            for _ in range(2):
                total = sum((judge('mini_llm', model_cls), judge('llm_training', pretrain_cls),
                             judge('llm_sft', sft_cls), judge('llm_preference', dpo_cls)))
            assert judge('mini_llm', renamed_mini_llm()) == len(get_task('mini_llm')['tests'])
            tok = ByteTokenizer()
            assert tok.decode(tok.encode('Hello 世界 🐍')) == 'Hello 世界 🐍'
            pretrain = PretrainDataset([{'text':'abcdef'}], 4)
            targets = [int(i) for _, labels in pretrain for i in labels[1:] if i != -100]
            assert targets == tok.encode('abcdef') + [2]
            records = read_jsonl(ROOT / 'datasets/llm/sft/demo_train.jsonl')
            data = TextDataset(records, 128)
            ids, labels = data[0]
            assert ids[0] == 1 and labels[0] == -100
            assert labels[labels != -100].tolist() == tok.encode('Hello!') + [2]
            assert (labels[ids == 0] == -100).all()
            assert set(map(str, records)).isdisjoint(map(str, read_jsonl(ROOT / 'datasets/llm/sft/demo_validation.jsonl')))
            pair = PreferenceDataset([{'prompt':'Hi','chosen':'Hello!','rejected':'Bye!'}], 32)[0]
            assert pair[1][pair[1] != -100].tolist() == tok.encode('Hello!') + [2]
            assert pair[3][pair[3] != -100].tolist() == tok.encode('Bye!') + [2]
            import mini_llm_reference as module
            original_rope = module.apply_rope
            module.apply_rope = lambda x, base, offset=0: x
            must_fail(lambda: model_case(model_cls, 'forward'))
            module.apply_rope = lambda x, base, offset=0: original_rope(x, base, 0)
            must_fail(lambda: cache_case(model_cls, 'gqa'))
            module.apply_rope = original_rope
            source = Path('mini_llm_reference.py').read_text()
            bad_mask = 'mask = key_positions[None, :] > query_positions[:, None]'
            assert bad_mask in source
            broken = {}
            exec(source.replace(bad_mask, 'mask = torch.zeros(s, offset + s, dtype=torch.bool)'), broken)
            must_fail(lambda: cache_case(broken['MiniLLM'], 'gqa'))
            class Recomputing(model_cls):
                def forward(self, input_ids, past_key_values=None, use_cache=False):
                    if past_key_values is not None:
                        super().forward(self.prompt)
                    else:
                        self.prompt = input_ids
                    return super().forward(input_ids, past_key_values, use_cache)
            must_fail(lambda: cache_case(Recomputing, 'work'))
            class Unshifted(pretrain_cls):
                @staticmethod
                def loss(logits, labels):
                    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            must_fail(lambda: training_case(Unshifted, 'loss'))
            class Unmasked(sft_cls):
                @staticmethod
                def encode_messages(messages, tokenizer):
                    ids, _ = sft_cls.encode_messages(messages, tokenizer)
                    return ids, ids.copy()
            must_fail(lambda: sft_case(Unmasked, 'mask'))
            class ReversedDPO(dpo_cls):
                @staticmethod
                def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=.1):
                    return torch.nn.functional.softplus(beta*((pi_c-pi_r)-(ref_c-ref_r))).mean()
            must_fail(lambda: preference_case(ReversedDPO, 'loss'))
            print(f'PASS: {total} judge cases twice; four stages, checkpoints, datasets and negative controls')
        finally:
            os.chdir(previous)
            sys.path.remove(tmp)


if __name__ == '__main__':
    main()
