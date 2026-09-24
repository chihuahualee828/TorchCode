"""Supervised instruction fine-tuning checks."""

TASK = {
    'title': 'Final Assignment: Instruction SFT',
    'difficulty': 'Hard',
    'function_name': 'SFTTrainer',
    'hint': 'Mask prompt and role text; preserve the assistant content and end markers.',
    'tests': [
        {'name': name, 'code': "from torch_judge.capstone.grading import sft_case\nsft_case({fn}, " + repr(case) + ')'}
        for name, case in [('Encoding','encode'),('Assistant mask','mask'),('Multi-turn','multiturn'),
                           ('Invalid records','invalid'),('Reply','reply'),('Zero budget','budget'),
                           ('Invalid reply','invalid_reply')]
    ],
}
