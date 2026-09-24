"""DPO preference-stage checks."""

TASK = {
    'title': 'Final Assignment: Preference DPO',
    'difficulty': 'Hard',
    'function_name': 'DPOTrainer',
    'hint': 'Sum only answer-token log-probabilities and keep the reference model frozen.',
    'tests': [
        {'name': name, 'code': "from torch_judge.capstone.grading import preference_case\npreference_case({fn}, " + repr(case) + ')'}
        for name, case in [('Preference data','dataset'),('Sequence log-probs','logps'),
                           ('Masked log-probs','masked_logps'),('Log-prob gradients','logps_grad'),
                           ('DPO loss','loss'),('DPO gradients','loss_grad'),('Beta','beta'),
                           ('Invalid inputs','invalid'),('Policy update','update'),
                           ('Frozen reference','reference'),('Repeatability','repeat')]
    ],
}
