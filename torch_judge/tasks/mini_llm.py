"""Capstone task; fixed CPU numerical checks."""

TASK = {'title': 'Final Assignment: Decoder-only LLM',
 'difficulty': 'Hard',
 'function_name': 'MiniLLM',
 'hint': 'Follow the architecture contract: pre-norm residuals, RoPE Q/K, causal GQA, SwiGLU, '
         'and tied vocabulary weights. Component names are your choice.',
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

TASK['tests'] += [
    {'name': 'KV cache: ' + name,
     'code': 'from torch_judge.capstone.grading import cache_case\n'
             + 'cache_case({fn}, ' + repr(case) + ')'}
    for name, case in [
        ('GQA prefill and chunk equivalence', 'gqa'),
        ('MHA equivalence', 'mha'),
        ('MQA equivalence', 'mqa'),
        ('rotated K and unrotated V', 'contents'),
        ('reuse without mutation or conversation leakage', 'reuse'),
        ('only project new tokens', 'work'),
        ('total context limit', 'limits'),
        ('invalid cache validation', 'invalid'),
        ('identical greedy tokens', 'greedy'),
    ]
]
