"""Deterministic, independent functional oracle and capstone checks."""
import copy
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import LLMConfig, deterministic, ByteTokenizer, TextDataset


def config(**kwargs):
    args = dict(vocab_size=19, d_model=16, num_heads=4, num_kv_heads=2,
                num_layers=2, hidden_dim=23, max_seq_len=16, rope_base=37.0)
    args.update(kwargs)
    return LLMConfig(**args)


def fixed_model(cls, c):
    model = cls(c).cpu().double()
    # Independent of constructor RNG and parameter registration order.
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            values = torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape)
            p.copy_(0.2 * torch.sin(values * 0.37 + sum(map(ord, name))))
            if 'norm' in name:
                p.add_(1.0)
    return model


def oracle(ids, w, c):
    """Per-query-head attention, complex RoPE: deliberately different from solution."""
    def norm(x, key):
        return x / (x.square().mean(-1, keepdim=True) + c.eps).sqrt() * w[key]
    def linear(x, key):
        return F.linear(x, w[key])
    def rotate(x):
        freq = c.rope_base ** (-torch.arange(0, x.size(-1), 2, dtype=x.dtype) / x.size(-1))
        angles = torch.outer(torch.arange(x.size(1), dtype=x.dtype), freq)
        phase = torch.polar(torch.ones_like(angles), angles)
        z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        return torch.view_as_real(z * phase).flatten(-2)
    x = F.embedding(ids, w['embed_tokens.weight'])
    b, s, d = x.shape
    h = d // c.num_heads
    for i in range(c.num_layers):
        prefix = 'layers.' + str(i) + '.'
        z = norm(x, prefix + 'attn_norm.weight')
        q = linear(z, prefix + 'q_proj.weight').reshape(b, s, c.num_heads, h)
        k = linear(z, prefix + 'k_proj.weight').reshape(b, s, c.num_kv_heads, h)
        v = linear(z, prefix + 'v_proj.weight').reshape(b, s, c.num_kv_heads, h)
        heads = []
        for j in range(c.num_heads):
            kv = j // (c.num_heads // c.num_kv_heads)
            scores = rotate(q[:, :, j]) @ rotate(k[:, :, kv]).transpose(1, 2) / h ** 0.5
            scores = scores.masked_fill(torch.arange(s)[None, :] > torch.arange(s)[:, None], -torch.inf)
            heads.append(scores.softmax(-1) @ v[:, :, kv])
        x = x + linear(torch.cat(heads, -1), prefix + 'o_proj.weight')
        z = norm(x, prefix + 'ffn_norm.weight')
        gate = linear(z, prefix + 'gate_proj.weight')
        x = x + linear(gate * gate.sigmoid() * linear(z, prefix + 'up_proj.weight'), prefix + 'down_proj.weight')
    return linear(norm(x, 'norm.weight'), 'embed_tokens.weight')


def model_case(cls, case):
    with deterministic(1729):
        c = config()
        if case in ('mha', 'mqa', 'deep'):
            c = config(num_kv_heads={'mha': 4, 'mqa': 1, 'deep': 2}[case], num_layers=3 if case == 'deep' else 2)
        if case == 'dimensions':
            c = config(d_model=24, num_heads=3, num_kv_heads=1, hidden_dim=35, eps=.03)
        m = fixed_model(cls, c)
        ids = torch.tensor([[1, 8, 3, 11, 7], [4, 6, 9, 2, 5]])
        if case == 'structure':
            assert m.lm_head.weight is m.embed_tokens.weight, 'Tie embedding and output weights'
            assert len(m.layers) == c.num_layers
            expected = {'embed_tokens.weight', 'lm_head.weight', 'norm.weight'}
            for i in range(c.num_layers):
                expected.update('layers.' + str(i) + '.' + k + '.weight' for k in
                                ('attn_norm', 'ffn_norm', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'))
            assert set(m.state_dict()) == expected, 'Follow the documented parameter schema; no biases or learned positions'
        elif case == 'causal':
            other = ids.clone()
            other[:, 3:] = 0
            torch.testing.assert_close(m(ids)[:, :3], m(other)[:, :3], rtol=0, atol=1e-12)
        elif case == 'batch':
            torch.testing.assert_close(m(ids), torch.cat([m(row[None]) for row in ids]), rtol=1e-7, atol=1e-9)
        elif case == 'lengths':
            for n in (1, c.max_seq_len):
                x = torch.ones(1, n, dtype=torch.long)
                torch.testing.assert_close(m(x), oracle(x, dict(m.named_parameters()), c))
            for n in (0, c.max_seq_len + 1):
                try:
                    m(torch.ones(1, n, dtype=torch.long))
                except ValueError:
                    pass
                else:
                    raise AssertionError('Reject empty/overlong sequences with ValueError')
        elif case == 'gradients':
            w = {k: p.detach().clone().requires_grad_() for k, p in m.named_parameters()}
            upstream = torch.randn(2, 5, c.vocab_size, dtype=torch.double)
            (m(ids) * upstream).sum().backward()
            (oracle(ids, w, c) * upstream).sum().backward()
            for name, p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                torch.testing.assert_close(p.grad, w[name].grad, rtol=1e-6, atol=1e-8, msg=name)
        elif case == 'roundtrip':
            clone = cls(c).double()
            clone.load_state_dict(m.state_dict())
            assert torch.equal(m(ids), clone(ids)), 'state_dict must preserve logits'
            assert torch.equal(m(ids), m(ids)), 'No stochastic forward operations'
        else:
            torch.testing.assert_close(m(ids), oracle(ids, dict(m.named_parameters()), c), rtol=1e-7, atol=1e-9)


class ToyLM(torch.nn.Module):
    """Training checks isolate the trainer from the student's model assignment."""
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(260, 8)
        self.head = torch.nn.Linear(8, 260)
        self.config = SimpleNamespace(max_seq_len=12)

    def forward(self, ids):
        return self.head(self.embed(ids))


def training_case(cls, case):
    with deterministic(314):
        if case in ('loss', 'masked', 'empty_loss'):
            logits = torch.randn(2, 5, 7, dtype=torch.double, requires_grad=True)
            labels = torch.tensor([[1, -100, 3, 2, -100], [0, 4, 2, -100, 5]])
            if case == 'empty_loss':
                labels.fill_(-100)
            expected = logits.sum() * 0 if case == 'empty_loss' else F.cross_entropy(logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100)
            actual = cls.loss(logits, labels)
            torch.testing.assert_close(actual, expected)
            ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
            ge = torch.autograd.grad(expected, logits)[0]
            torch.testing.assert_close(ga, ge)
            if case == 'masked':
                assert torch.count_nonzero(ga[:, -1]) == 0
                assert torch.count_nonzero(ga[:, :-1][labels[:, 1:] == -100]) == 0
            return
        if case in ('update', 'accumulation', 'repeat', 'empty_step'):
            m = ToyLM().double()
            ref = copy.deepcopy(m)
            ids = torch.tensor([[1, 4, 6, 3, 2], [1, 8, 3, 7, 2], [1, 9, 4, 6, 2]])
            labels = ids.clone()
            labels[0, 2:] = -100
            labels[1, 3:] = -100
            if case == 'empty_step':
                labels.fill_(-100)
                try:
                    cls.train_step(m, torch.optim.SGD(m.parameters(), lr=.1), [(ids, labels)])
                except ValueError:
                    for a, b in zip(m.parameters(), ref.parameters()):
                        assert torch.equal(a, b)
                    return
                raise AssertionError('Reject a batch with no supervised targets')
            opt = torch.optim.AdamW(m.parameters(), lr=.01)
            ropt = torch.optim.AdamW(ref.parameters(), lr=.01)
            batches = [(ids[:1], labels[:1]), (ids[1:], labels[1:])] if case == 'accumulation' else [(ids, labels)]
            # Two updates expose failure to zero gradients and optimizer state errors.
            for _ in range(2):
                ropt.zero_grad(set_to_none=True)
                expected = F.cross_entropy(ref(ids)[:, :-1].reshape(-1, 260), labels[:, 1:].reshape(-1), ignore_index=-100)
                expected.backward()
                torch.nn.utils.clip_grad_norm_(ref.parameters(), .07)
                ropt.step()
                value = cls.train_step(m, opt, batches, max_norm=.07)
                assert isinstance(value, float)
                assert abs(value - expected.item()) < 1e-9
                for a, b in zip(m.parameters(), ref.parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)
            if case == 'repeat':
                def run():
                    torch.manual_seed(45)
                    model = ToyLM().double()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
                    losses = [cls.train_step(model, optimizer, batches) for _ in range(3)]
                    return losses, cls.generate(model, ids[:1], 4), model.state_dict()
                a, b = run(), run()
                assert a[0] == b[0] and torch.equal(a[1], b[1])
                assert all(torch.equal(a[2][k], b[2][k]) for k in a[2])
            return
        class ScriptedLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(max_seq_len=7)
                self.calls = 0
            def forward(self, ids):
                assert not self.training and not torch.is_grad_enabled(), 'Use eval and no_grad'
                self.calls += 1
                logits = torch.zeros(1, ids.size(1), 9)
                token = 2 if case == 'eos' else 4
                logits[:, -1, token] = 5
                if case == 'ties':
                    logits[:, -1, 5] = 5
                return logits
        m = ScriptedLM()
        prompt = torch.tensor([[1, 6]])
        if case == 'invalid_generation':
            for bad, budget in [(torch.ones(2, 2, dtype=torch.long), 1),
                                (torch.ones(1, 0, dtype=torch.long), 1),
                                (torch.ones(1, 8, dtype=torch.long), 1),
                                (torch.ones(2, dtype=torch.long), 1), (prompt, -1)]:
                try:
                    cls.generate(m, bad, budget)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid generation input must raise ValueError')
            return
        if case == 'exception_mode':
            def fail(ids):
                raise RuntimeError('intentional forward failure')
            m.forward = fail
            try:
                cls.generate(m, prompt, 1)
            except RuntimeError:
                assert m.training, 'Restore training mode on errors too'
                return
            raise AssertionError('Propagate model errors')
        original = prompt.clone()
        n = 0 if case == 'zero_tokens' else 20 if case == 'context' else 3
        out = cls.generate(m, prompt, n)
        tokens = [] if n == 0 else [2] if case == 'eos' else [4] * min(n, 5)
        assert torch.equal(out, torch.tensor([[1, 6] + tokens])), 'Wrong greedy continuation/stop condition'
        assert torch.equal(prompt, original), 'Do not mutate the prompt'
        assert m.training, 'Restore training mode'
        m.eval()
        assert torch.equal(out, cls.generate(m, prompt, n))
        assert not m.training, 'Restore eval mode'
