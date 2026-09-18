"""Capstone task; fixed CPU numerical checks."""

TASK = {'title': 'Final Assignment: Decoder-only LLM',
 'difficulty': 'Hard',
 'function_name': 'MiniLLM',
 'hint': 'Follow the exact notebook contract; inspect shifted targets, RoPE pair ordering, and '
         'causal head dimensions.',
 'tests': [{'name': 'Structure',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'structure')"},
           {'name': 'Forward',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'forward')"},
           {'name': 'Mha',
            'code': "from torch_judge.capstone.grading import model_case\nmodel_case({fn}, 'mha')"},
           {'name': 'Mqa',
            'code': "from torch_judge.capstone.grading import model_case\nmodel_case({fn}, 'mqa')"},
           {'name': 'Deep',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'deep')"},
           {'name': 'Causal',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'causal')"},
           {'name': 'Lengths',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'lengths')"},
           {'name': 'Gradients',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'gradients')"},
           {'name': 'Roundtrip',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'roundtrip')"},
           {'name': 'Dimensions',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'dimensions')"},
           {'name': 'Batch',
            'code': 'from torch_judge.capstone.grading import model_case\n'
                    "model_case({fn}, 'batch')"}]}
