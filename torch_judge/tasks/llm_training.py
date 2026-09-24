"""Capstone task; fixed CPU numerical checks."""

TASK = {'title': 'Final Assignment: Raw-text pretraining',
 'difficulty': 'Hard',
 'function_name': 'LLMTrainer',
 'hint': 'Shift targets exactly once; weight accumulated gradients by valid token count.',
 'tests': [{'name': 'Loss',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'loss')"},
           {'name': 'Masked',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'masked')"},
           {'name': 'Empty Loss',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'empty_loss')"},
           {'name': 'Update',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'update')"},
           {'name': 'Accumulation',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'accumulation')"},
           {'name': 'Repeat',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'repeat')"},
           {'name': 'Empty Step',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'empty_step')"},
           {'name': 'Greedy',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'greedy')"},
           {'name': 'Eos',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'eos')"},
           {'name': 'Ties',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'ties')"},
           {'name': 'Zero Tokens',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'zero_tokens')"},
           {'name': 'Context',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'context')"},
           {'name': 'Invalid Generation',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'invalid_generation')"},
           {'name': 'Exception Mode',
            'code': 'from torch_judge.capstone.grading import training_case\n'
                    "training_case({fn}, 'exception_mode')"}]}
