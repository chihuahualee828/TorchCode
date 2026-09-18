"""Run from repo root: .venv/bin/python scripts/tests/test_llm_capstone.py.
Executes both solution notebooks in a temporary workspace, all judge cases twice,
and negative controls. No IPython, network or persistent judge progress required.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch_judge.capstone import deterministic, ByteTokenizer, TextDataset, read_jsonl
from torch_judge.capstone.grading import model_case, training_case
from torch_judge.tasks import get_task


def execute(path, namespace):
    for cell in json.loads(path.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            header, body = source.split('\n', 1)
            Path(header.split()[1]).write_text(body)
        elif "check('" not in source:
            exec(compile(source, str(path), 'exec'), namespace)


def judge(task_id, cls):
    task = get_task(task_id)
    for case in task['tests']:
        code = case['code'].replace('{fn}', task['function_name'])
        exec(code, {task['function_name']: cls})
    return len(task['tests'])


def must_fail(action):
    try:
        action()
    except (AssertionError, RuntimeError):
        return
    raise AssertionError('Negative control unexpectedly passed')


def main():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        sys.path.insert(0, tmp)
        try:
            ns = {}
            execute(ROOT / 'solutions/42_mini_llm_solution.ipynb', ns)
            execute(ROOT / 'solutions/43_llm_training_solution.ipynb', ns)
            for _ in range(2):
                total = judge('mini_llm', ns['MiniLLM']) + judge('llm_training', ns['LLMTrainer'])
            model, optimizer, history = ns['run_training']()
            assert history == ns['history'], 'Training history must repeat exactly'
            for k, v in model.state_dict().items():
                assert torch.equal(v, ns['model'].state_dict()[k]), k
            assert ns['chat_reply'](model, 'Hi')[0] == ns['chat_reply'](ns['model'], 'Hi')[0]
            assert history[-1][0] < history[0][0] * .5, 'Toy training should learn'
            tok = ByteTokenizer()
            assert tok.decode(tok.encode('Hello 世界 🐍')) == 'Hello 世界 🐍'
            records = read_jsonl('datasets/llm/train.jsonl')
            data = TextDataset(records, 128)
            ids, labels = data[0]
            assert ids[0] == 1 and labels[0] == -100
            supervised = labels[labels != -100].tolist()
            assert supervised == tok.encode('Hello!') + [2]
            assert (labels[ids == 0] == -100).all()
            plain = TextDataset([{'text': 'abc'}], 8)
            assert plain[0][0].tolist() == [1] + tok.encode('abc') + [2, 0, 0, 0]
            assert plain[0][1].tolist() == [1] + tok.encode('abc') + [2, -100, -100, -100]
            multi = TextDataset([records[-1]], 128)[0][1]
            assert multi[multi != -100].tolist() == tok.encode('Hello, Ada.') + [3] + tok.encode('Ada.') + [2]
            try:
                TextDataset(records, 2)
            except ValueError:
                pass
            else:
                raise AssertionError('Truncation with no supervised tokens must fail clearly')
            assert set(map(str, records)).isdisjoint(map(str, read_jsonl('datasets/llm/validation.jsonl')))
            import mini_llm_reference as module
            original_rope = module.apply_rope
            module.apply_rope = lambda x, base: x
            must_fail(lambda: model_case(ns['MiniLLM'], 'forward'))
            module.apply_rope = original_rope
            class Unshifted(ns['LLMTrainer']):
                @staticmethod
                def loss(logits, labels):
                    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            must_fail(lambda: training_case(Unshifted, 'loss'))
            print(f'PASS: {total} judge cases twice; full training twice; checkpoint/chat; data; negative controls')
        finally:
            os.chdir(previous)
            sys.path.remove(tmp)


if __name__ == '__main__':
    main()
