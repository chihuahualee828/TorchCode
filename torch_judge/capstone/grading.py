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


def _modules(module, kind):
    return [child for child in module.modules() if isinstance(child, kind)]


def _layer_parts(layer, c):
    """Discover a decoder layer without relying on student-chosen attribute names.

    The exercise intentionally keeps separate Q/K/V/O and SwiGLU projections.
    Shapes identify each projection family; members of the same family receive
    identical fixed weights, so their attribute and registration order do not
    affect numerical comparison.
    """
    d = c.d_model
    kv = c.num_kv_heads * (d // c.num_heads)
    h = c.hidden_dim
    norms = _modules(layer, torch.nn.RMSNorm)
    linears = _modules(layer, torch.nn.Linear)
    assert len(norms) == 2 and all(tuple(n.weight.shape) == (d,) for n in norms), \
        'Each decoder layer needs two nn.RMSNorm(d_model) modules'
    assert all(linear.bias is None for linear in linears), 'Decoder projections must use bias=False'
    square = [linear for linear in linears if tuple(linear.weight.shape) == (d, d)]
    kv_proj = [linear for linear in linears if tuple(linear.weight.shape) == (kv, d)]
    up = [linear for linear in linears if tuple(linear.weight.shape) == (h, d)]
    down = [linear for linear in linears if tuple(linear.weight.shape) == (d, h)]
    if kv == d:  # MHA: Q/K/V/O share the same matrix shape.
        assert len(square) == 4 and len(up) == 2 and len(down) == 1 and len(linears) == 7, \
            'Each layer needs separate Q/K/V/O and SwiGLU gate/up/down projections with config dimensions'
        kv_proj = []
    else:
        assert len(square) == len(kv_proj) == len(up) == 2 and len(down) == 1 and len(linears) == 7, \
            'Each layer needs separate Q/K/V/O and SwiGLU gate/up/down projections with config dimensions'
    return {'norms': norms, 'square': square, 'kv': kv_proj, 'up': up, 'down': down}


def _structure(model, c):
    """Find the required components structurally, never by attribute name."""
    embeddings = [m for m in _modules(model, torch.nn.Embedding)
                  if tuple(m.weight.shape) == (c.vocab_size, c.d_model)]
    assert len(embeddings) == 1, 'Need one nn.Embedding(vocab_size, d_model)'
    embedding = embeddings[0]

    heads = [m for m in _modules(model, torch.nn.Linear)
             if tuple(m.weight.shape) == (c.vocab_size, c.d_model) and m.bias is None]
    heads = [m for m in heads if m.weight is embedding.weight]
    assert len(heads) == 1, 'Need one bias-free vocabulary head whose weight is tied to the embedding weight'

    candidates = []
    layer_errors = []
    for group in _modules(model, torch.nn.ModuleList):
        if len(group) != c.num_layers:
            continue
        try:
            parts = [_layer_parts(layer, c) for layer in group]
        except AssertionError as error:
            layer_errors.append(str(error))
            continue
        candidates.append((group, parts))
    if not candidates and layer_errors:
        raise AssertionError(layer_errors[0])
    assert len(candidates) == 1, 'Need one ModuleList containing config.num_layers decoder layers'
    layers, layer_parts = candidates[0]
    layer_ids = {id(module) for layer in layers for module in layer.modules()}
    final_norms = [m for m in _modules(model, torch.nn.RMSNorm)
                   if id(m) not in layer_ids and tuple(m.weight.shape) == (c.d_model,)]
    assert len(final_norms) == 1, 'Need one final nn.RMSNorm(d_model) outside decoder layers'
    assert not _modules(model, torch.nn.Dropout), 'Dropout is not part of this assignment architecture'

    expected_parameters = c.vocab_size * c.d_model + c.d_model
    per_layer = 2 * c.d_model + 2 * c.d_model * c.d_model + 2 * c.num_kv_heads * (c.d_model // c.num_heads) * c.d_model + 3 * c.hidden_dim * c.d_model
    expected_parameters += c.num_layers * per_layer
    actual_parameters = sum(p.numel() for p in model.parameters())
    assert actual_parameters == expected_parameters, \
        f'Expected {expected_parameters} trainable parameters, got {actual_parameters}; do not add biases or learned positions'
    return {'embedding': embedding, 'head': heads[0], 'layers': layers,
            'layer_parts': layer_parts, 'final_norm': final_norms[0]}


def _values(shape, dtype, device, label, norm=False):
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=dtype, device=device).reshape(shape)
    result = 0.2 * torch.sin(values * 0.37 + sum(map(ord, label)))
    return result + 1.0 if norm else result


def _canonical_weights(c, dtype=torch.double, device='cpu'):
    d = c.d_model
    kv = c.num_kv_heads * (d // c.num_heads)
    return {
        'embedding': _values((c.vocab_size, d), dtype, device, 'embedding'),
        'norm': _values((d,), dtype, device, 'norm', norm=True),
        'square': _values((d, d), dtype, device, 'square'),
        'kv': _values((d, d), dtype, device, 'square') if kv == d else _values((kv, d), dtype, device, 'kv'),
        'up': _values((c.hidden_dim, d), dtype, device, 'up'),
        'down': _values((d, c.hidden_dim), dtype, device, 'down'),
    }


def fixed_model(cls, c):
    model = cls(c).cpu().double()
    parts = _structure(model, c)
    weights = _canonical_weights(c)
    with torch.no_grad():
        parts['embedding'].weight.copy_(weights['embedding'])
        parts['final_norm'].weight.copy_(weights['norm'])
        for layer in parts['layer_parts']:
            for norm in layer['norms']:
                norm.weight.copy_(weights['norm'])
            for linear in layer['square']:
                linear.weight.copy_(weights['square'])
            for linear in layer['kv']:
                linear.weight.copy_(weights['kv'])
            for linear in layer['up']:
                linear.weight.copy_(weights['up'])
            layer['down'][0].weight.copy_(weights['down'])
    return model, parts


def oracle(ids, c):
    """Per-query-head attention, complex RoPE: deliberately different from solution."""
    def norm(x):
        return x / (x.square().mean(-1, keepdim=True) + c.eps).sqrt() * weights['norm']
    def linear(x, key):
        return F.linear(x, weights[key])
    def rotate(x):
        freq = c.rope_base ** (-torch.arange(0, x.size(-1), 2, dtype=x.dtype) / x.size(-1))
        angles = torch.outer(torch.arange(x.size(1), dtype=x.dtype), freq)
        phase = torch.polar(torch.ones_like(angles), angles)
        z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        return torch.view_as_real(z * phase).flatten(-2)
    weights = _canonical_weights(c, ids.dtype if ids.is_floating_point() else torch.double, ids.device)
    x = F.embedding(ids, weights['embedding'])
    b, s, d = x.shape
    h = d // c.num_heads
    for _ in range(c.num_layers):
        z = norm(x)
        q = linear(z, 'square').reshape(b, s, c.num_heads, h)
        k = linear(z, 'kv').reshape(b, s, c.num_kv_heads, h)
        v = linear(z, 'kv').reshape(b, s, c.num_kv_heads, h)
        heads = []
        for j in range(c.num_heads):
            kv = j // (c.num_heads // c.num_kv_heads)
            scores = rotate(q[:, :, j]) @ rotate(k[:, :, kv]).transpose(1, 2) / h ** 0.5
            scores = scores.masked_fill(torch.arange(s)[None, :] > torch.arange(s)[:, None], -torch.inf)
            heads.append(scores.softmax(-1) @ v[:, :, kv])
        x = x + linear(torch.cat(heads, -1), 'square')
        z = norm(x)
        gate = linear(z, 'up')
        x = x + linear(gate * gate.sigmoid() * linear(z, 'up'), 'down')
    return linear(norm(x), 'embedding')


def model_case(cls, case):
    with deterministic(1729):
        c = config()
        if case in ('mha', 'mqa', 'deep'):
            c = config(num_kv_heads={'mha': 4, 'mqa': 1, 'deep': 2}[case], num_layers=3 if case == 'deep' else 2)
        if case == 'dimensions':
            c = config(d_model=24, num_heads=3, num_kv_heads=1, hidden_dim=35, eps=.03)
        m, parts = fixed_model(cls, c)
        ids = torch.tensor([[1, 8, 3, 11, 7], [4, 6, 9, 2, 5]])
        if case == 'structure':
            assert parts['head'].weight is parts['embedding'].weight, 'Tie embedding and output weights'
            assert len(parts['layers']) == c.num_layers
        elif case == 'causal':
            other = ids.clone()
            other[:, 3:] = 0
            torch.testing.assert_close(m(ids)[:, :3], m(other)[:, :3], rtol=0, atol=1e-12)
        elif case == 'batch':
            torch.testing.assert_close(m(ids), torch.cat([m(row[None]) for row in ids]), rtol=1e-7, atol=1e-9)
        elif case == 'lengths':
            for n in (1, c.max_seq_len):
                x = torch.ones(1, n, dtype=torch.long)
                torch.testing.assert_close(m(x), oracle(x, c))
            for n in (0, c.max_seq_len + 1):
                try:
                    m(torch.ones(1, n, dtype=torch.long))
                except ValueError:
                    pass
                else:
                    raise AssertionError('Reject empty/overlong sequences with ValueError')
        elif case == 'gradients':
            upstream = torch.randn(2, 5, c.vocab_size, dtype=torch.double)
            (m(ids) * upstream).sum().backward()
            for name, p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
        elif case == 'roundtrip':
            clone = cls(c).double()
            clone.load_state_dict(m.state_dict())
            assert torch.equal(m(ids), clone(ids)), 'state_dict must preserve logits'
            assert torch.equal(m(ids), m(ids)), 'No stochastic forward operations'
        else:
            torch.testing.assert_close(m(ids), oracle(ids, c), rtol=1e-7, atol=1e-9)


def _cached_call(model, c, ids, past=None):
    result = model(ids, past_key_values=past, use_cache=True)
    assert isinstance(result, (tuple, list)) and len(result) == 2, \
        'use_cache=True must return (logits, present_key_values)'
    logits, cache = result
    assert logits.shape == (*ids.shape, c.vocab_size), 'Return logits only for new tokens'
    assert torch.isfinite(logits).all(), 'Cached logits must be finite'
    assert isinstance(cache, (tuple, list)) and len(cache) == c.num_layers, \
        'Return one (K, V) pair per decoder layer'
    length = ids.size(1) + (0 if past is None else past[0][0].size(2))
    expected = (ids.size(0), c.num_kv_heads, length, c.d_model // c.num_heads)
    for pair in cache:
        assert isinstance(pair, (tuple, list)) and len(pair) == 2
        for tensor in pair:
            assert isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == expected, \
                f'Cache must retain compact KV heads, shape {expected}'
            assert tensor.dtype == logits.dtype and tensor.device == logits.device
            assert torch.isfinite(tensor).all(), 'Cache must be finite'
    return logits, cache


def _assert_cache_equal(actual, expected):
    for a_pair, e_pair in zip(actual, expected):
        for a, e in zip(a_pair, e_pair):
            torch.testing.assert_close(a, e, rtol=1e-7, atol=1e-9)


def cache_case(cls, case):
    """Use distinct seeded random parameters to exercise actual cache behavior."""
    with deterministic(8128), torch.no_grad():
        c = config(num_layers=3, max_seq_len=12)
        if case == 'mha':
            c = config(num_kv_heads=4)
        elif case == 'mqa':
            c = config(num_kv_heads=1)
        m = cls(c).cpu().double().eval()
        ids = torch.tensor([[1, 8, 3, 11, 7, 6, 9], [4, 6, 9, 2, 5, 12, 10]])
        close = lambda a, b: torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)

        if case in ('gqa', 'mha', 'mqa'):
            full = m(ids)
            prefill, full_cache = _cached_call(m, c, ids)
            close(prefill, full)
            # Token-wise and multi-token chunks catch both offset and rectangular-mask errors.
            for chunks in ((1, 1, 1, 1, 1, 1, 1), (2, 3, 2)):
                cache, offset = None, 0
                for size in chunks:
                    logits, cache = _cached_call(m, c, ids[:, offset:offset + size], cache)
                    close(logits, full[:, offset:offset + size])
                    offset += size
                _assert_cache_equal(cache, full_cache)
            # Empty cache has the same meaning as no cached prefix.
            empty = tuple((k[:, :, :0], v[:, :, :0]) for k, v in full_cache)
            actual, _ = _cached_call(m, c, ids, empty)
            close(actual, full)
        elif case == 'contents':
            parts = _structure(m, c)
            observed, handles = {}, []
            def record(module, args, output):
                observed[id(module)] = output.clone()
            try:
                for layer in parts['layer_parts']:
                    for module in layer['kv']:
                        handles.append(module.register_forward_hook(record))
                _, cache = _cached_call(m, c, ids)
            finally:
                for handle in handles:
                    handle.remove()
            h = c.d_model // c.num_heads
            angles = torch.outer(torch.arange(ids.size(1), dtype=torch.double),
                                 c.rope_base ** (-torch.arange(0, h, 2, dtype=torch.double) / h))
            phase = torch.polar(torch.ones_like(angles), angles)
            def rotate(x):
                z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
                return torch.view_as_real(z * phase).flatten(-2)
            for pair, layer in zip(cache, parts['layer_parts']):
                raw = [observed[id(module)].reshape(ids.size(0), ids.size(1), c.num_kv_heads, h)
                       .transpose(1, 2) for module in layer['kv']]
                # K and V have the same dimensions; accept either registration order.
                matched = any(torch.allclose(pair[0], rotate(raw[k]), rtol=1e-7, atol=1e-9)
                              and torch.allclose(pair[1], raw[1-k], rtol=1e-7, atol=1e-9)
                              for k in (0, 1))
                assert matched, 'Cache must contain RoPE-rotated K and unrotated V per layer'
        elif case == 'reuse':
            _, prefix = _cached_call(m, c, ids[:, :3])
            saved = tuple((k.clone(), v.clone()) for k, v in prefix)
            params = {k: v.clone() for k, v in m.state_dict().items()}
            for suffix in (ids[:, 3:5], ids[:, 5:7]):
                # An unrelated conversation must not contaminate the supplied cache.
                _cached_call(m, c, ids.flip(0))
                logits, cache = _cached_call(m, c, suffix, prefix)
                full_ids = torch.cat((ids[:, :3], suffix), dim=1)
                expected, expected_cache = _cached_call(m, c, full_ids)
                close(logits, expected[:, 3:])
                _assert_cache_equal(cache, expected_cache)
                for pair, snapshot in zip(prefix, saved):
                    for tensor, original in zip(pair, snapshot):
                        assert torch.equal(tensor, original), 'Do not mutate a supplied cache'
            assert set(m.state_dict()) == set(params), 'Cache must not become model state'
            assert all(torch.equal(v, params[k]) for k, v in m.state_dict().items())
            close(m(ids), m(ids))
        elif case == 'work':
            _, prefix = _cached_call(m, c, ids[:, :3])
            seen, handles = [], []
            def record(module, args):
                seen.append((module, args[0].shape))
            modules = _modules(m, torch.nn.Embedding) + _modules(m, torch.nn.Linear)
            try:
                for module in modules:
                    handles.append(module.register_forward_pre_hook(record))
                logits, _ = _cached_call(m, c, ids[:, 3:5], prefix)
            finally:
                for handle in handles:
                    handle.remove()
            assert {id(module) for module, _ in seen} == {id(module) for module in modules}, \
                'Incremental decoding must execute embedding and all model projections'
            for module in modules:
                shapes = [shape for called, shape in seen if called is module]
                tokens = sum(shape.numel() if isinstance(module, torch.nn.Embedding)
                             else shape.numel() // module.in_features for shape in shapes)
                assert tokens == ids.size(0) * 2, 'Project only the new tokens, once per module'
            close(logits, m(ids[:, :5])[:, 3:])
        elif case == 'limits':
            full_ids = torch.ones(2, c.max_seq_len, dtype=torch.long)
            _, prefix = _cached_call(m, c, full_ids[:, :-1])
            logits, cache = _cached_call(m, c, full_ids[:, -1:], prefix)
            close(logits, m(full_ids)[:, -1:])
            for bad_ids, bad_cache in ((full_ids[:, :1], cache), (full_ids[:, :0], prefix)):
                try:
                    m(bad_ids, past_key_values=bad_cache, use_cache=True)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Reject empty new input or total length above max_seq_len')
        elif case == 'invalid':
            _, cache = _cached_call(m, c, ids[:, :3])
            bad = [cache[:-1], (cache[0][0],) + tuple(cache[1:])]
            for transform in (lambda t: t[0], lambda t: t[:1],
                              lambda t: t[:, :1], lambda t: t[..., :-1],
                              lambda t: t[:, :, :-1], lambda t: t.float()):
                bad.append(((transform(cache[0][0]), cache[0][1]),) + tuple(cache[1:]))
            for candidate in bad:
                try:
                    m(ids[:, 3:4], past_key_values=candidate, use_cache=True)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Reject malformed/inconsistent cache with ValueError')
            try:
                m(ids[:, 3:4], past_key_values=cache, use_cache=False)
            except ValueError:
                pass
            else:
                raise AssertionError('Supplying a cache requires use_cache=True')
        elif case == 'greedy':
            sequence = ids[:, :2]
            logits, cache = _cached_call(m, c, sequence)
            for _ in range(5):
                expected = m(sequence)[:, -1]
                close(logits[:, -1], expected)
                token = logits[:, -1].argmax(-1, keepdim=True)
                assert torch.equal(token, expected.argmax(-1, keepdim=True))
                sequence = torch.cat((sequence, token), dim=1)
                logits, cache = _cached_call(m, c, token, cache)
        else:
            raise ValueError(f'Unknown cache test: {case}')


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
