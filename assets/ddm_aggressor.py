import math
import time
import fire
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
VOCAB = '0123456789+='
K, Kx = (len(VOCAB), len(VOCAB) + 1)
M = K
L = 9

def get_batch(B):
    a, b = np.random.randint(0, 100, (2, B))
    s = [f'{x:02d}+{y:02d}={x + y:03d}' for x, y in zip(a, b)]
    return mx.array(np.array([[VOCAB.index(c) for c in t] for t in s]))

def decode(x):
    return [''.join(('_' if i in (M, -1) else VOCAB[i] for i in row)) for row in np.array(x).tolist()]

def alpha_fn(schedule):
    return (lambda t: 1.0 - t) if schedule == 'linear' else lambda t: np.cos(t * np.pi / 2)

def digit_metric():
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i < 10 and j < 10:
                D[i, j] = abs(i - j)
            elif i == j:
                D[i, j] = 0.0
            else:
                D[i, j] = 10.0
    return D

def make_Q(kind, b, eta=0.1, P=None):
    Q = np.eye(Kx)
    if kind == 'absorb':
        for j in range(K):
            Q[j, j] = 1 - b
            Q[j, M] = b
    elif kind == 'uniform':
        Q[:K, :K] = (1 - b) * np.eye(K) + b / K * np.ones((K, K))
    elif kind == 'hybrid':
        for j in range(K):
            Q[j, j] = 1 - b
            Q[j, M] = b * (1 - eta)
            Q[j, :K] += b * eta / K
    elif kind == 'structured':
        Q[:K, :K] = (1 - b) * np.eye(K) + b * P
    return Q

def build_kernels(kind, T, schedule='linear', eta=0.1, sigma=2.0):
    t = np.linspace(0.0, 1.0, T + 1)
    al = alpha_fn(schedule)(t)
    betas = 1.0 - al[1:] / np.maximum(al[:-1], 1e-12)
    P = None
    if kind == 'structured':
        W = np.exp(-digit_metric() ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)
        P = W / W.sum(1, keepdims=True)
    Qs = np.stack([make_Q(kind, b, eta, P) for b in betas])
    Qbar = np.zeros((T + 1, Kx, Kx))
    Qbar[0] = np.eye(Kx)
    for i in range(T):
        Qbar[i + 1] = Qbar[i] @ Qs[i]
    return (Qs, Qbar, al)

def np_posterior(Qs, Qbar, ti, xt, p0):
    fwd = Qs[ti - 1][:, xt].T if xt.ndim else Qs[ti - 1][:, xt]
    prev = p0 @ Qbar[ti - 1]
    post = fwd * prev
    return post / np.maximum(post.sum(-1, keepdims=True), 1e-30)

def onehot(x, n):
    return (x[..., None] == mx.arange(n)).astype(mx.float32)

def t_of(ti, T):
    return (ti.astype(mx.float32) / T).reshape(-1, 1)

class Kernels:

    def __init__(self, kind, T, schedule='linear', eta=0.1, sigma=2.0):
        self.kind, self.T = (kind, T)
        Qs, Qbar, al = build_kernels(kind, T, schedule, eta, sigma)
        self.Qs, self.QsT, self.Qbar = (mx.array(Qs), mx.array(np.swapaxes(Qs, 1, 2)), mx.array(Qbar))
        self.al, self._afun = (al, alpha_fn(schedule))

    def alpha(self, t):
        return float(self._afun(t))

    def corrupt(self, x0, ti):
        rows = onehot(x0, Kx) @ mx.take(self.Qbar, ti, axis=0)
        xt = mx.random.categorical(mx.log(rows + 1e-12))
        return (xt, rows)

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        gate, x = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)

class Attention(nn.Module):

    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head, self.scale = (n_head, (dim // n_head) ** (-0.5))
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        B, S, _ = x.shape
        q, k, v = [z.reshape(B, S, self.n_head, -1).transpose(0, 2, 1, 3) for z in mx.split(self.qkv_proj(x), 3, axis=-1)]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, S, -1))

class Layer(nn.Module):

    def __init__(self, dim, n_head):
        super().__init__()
        self.self_attn, self.mlp = (Attention(dim, n_head), MLP(dim))
        self.input_layernorm, self.post_attention_layernorm = (nn.RMSNorm(dim), nn.RMSNorm(dim))

    def __call__(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))

class Denoiser(nn.Module):

    def __init__(self, dim=128, n_head=4, n_layer=4, param='x0', use_time=False):
        super().__init__()
        self.param = param
        self.embed = nn.Embedding(Kx, dim)
        self.pos = nn.Embedding(L, dim)
        self.layers = [Layer(dim, n_head) for _ in range(n_layer)]
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, K if param == 'x0' else Kx, bias=False)
        self.te = nn.Sequential(nn.SinusoidalPositionalEncoding(dim), nn.Linear(dim, dim), nn.SiLU()) if use_time else None

    def __call__(self, xt, t=None):
        h = self.embed(xt) + self.pos(mx.arange(L))[None]
        if self.te is not None:
            h = h + self.te(t.reshape(-1))[:, None, :]
        for l in self.layers:
            h = l(h)
        return self.head(self.norm(h))

def predict_mu(model, x, t_cont, B):
    t = mx.full((B, 1), float(t_cont))
    out = model(x, t)
    if model.param == 'x0':
        return (mx.softmax(out, -1), out)
    s = mx.exp(out)[..., :K]
    mu = s / mx.maximum(s.sum(-1, keepdims=True), 1e-09)
    return (mu, mx.log(mu + 1e-09))

def _draw_ti(ker, B, ti):
    return mx.random.randint(1, ker.T + 1, (B,)) if ti is None else mx.full((B,), int(ti), dtype=mx.int32)

def loss_simple(model, ker, x0, ti=None):
    ti = _draw_ti(ker, x0.shape[0], ti)
    xt, _ = ker.corrupt(x0, ti)
    ce = nn.losses.cross_entropy(model(xt, t_of(ti, ker.T)), x0, reduction='none')
    h = (xt != x0).astype(ce.dtype)
    return ((ce * h).sum(-1) / mx.maximum(h.sum(-1), 1)).mean()

def loss_elbo(model, ker, x0, ti=None):
    B, Ls = x0.shape
    ti = _draw_ti(ker, B, ti)
    xt, _ = ker.corrupt(x0, ti)
    logits = model(xt, t_of(ti, ker.T))
    p0 = mx.softmax(logits, -1)
    fwd = onehot(xt, Kx) @ mx.take(ker.QsT, ti - 1, axis=0)
    Qprev = mx.take(ker.Qbar, ti - 1, axis=0)
    q_prev = onehot(x0, Kx) @ Qprev
    p_prev = mx.concatenate([p0, mx.zeros((B, Ls, 1))], -1) @ Qprev
    qp = fwd * q_prev
    qp = qp / mx.maximum(qp.sum(-1, keepdims=True), 1e-12)
    pp = fwd * p_prev
    pp = pp / mx.maximum(pp.sum(-1, keepdims=True), 1e-12)
    kl = (qp * (mx.log(qp + 1e-12) - mx.log(pp + 1e-12))).sum(-1)
    rec = nn.losses.cross_entropy(logits, x0, reduction='none')
    return ker.T * mx.where((ti == 1)[:, None], rec, kl).mean()

def loss_score(model, ker, x0, ti=None):
    ti = _draw_ti(ker, x0.shape[0], ti)
    xt, rows = ker.corrupt(x0, ti)
    qx = (rows * onehot(xt, Kx)).sum(-1, keepdims=True)
    r = rows / mx.maximum(qx, 1e-12)
    s = mx.exp(model(xt, t_of(ti, ker.T)))
    off = 1.0 - onehot(xt, Kx)
    rlogr = mx.where(r > 0, r * mx.log(mx.maximum(r, 1e-12)), 0.0)
    rlogs = mx.where(r > 0, r * mx.log(s + 1e-12), 0.0)
    return (off * (s - rlogs + rlogr - r)).sum(-1).mean()

LOSSES = {'simple': loss_simple, 'elbo': loss_elbo, 'score_entropy': loss_score}

def sample_committing(model, ker, B=8, steps=8, mode='confidence', temp=0.0, prompt=None, block_size=2):
    a = ker.alpha
    x = mx.full((B, L), M)
    free = mx.ones((B, L), dtype=mx.bool_) if prompt is None else prompt < 0
    if prompt is not None:
        x = mx.where(free, x, prompt)
    n_free = int(free.sum(-1)[0].item())
    ts = np.linspace(1.0, 0.0, steps + 1)
    for t, s in zip(ts[:-1], ts[1:]):
        mu, logits = predict_mu(model, x, t, B)
        x0hat = mx.argmax(logits, -1) if temp == 0 else mx.random.categorical(logits / temp)
        masked = x == M
        if mode == 'remask':
            proposal = mx.where(masked, x0hat, x)
            pconf = mx.take_along_axis(mu, proposal[..., None], -1).squeeze(-1)
            k_keep = int(round(n_free * a(s)))
            sc = mx.where(free, pconf, -mx.inf)
            rank = mx.argsort(mx.argsort(-sc, -1), -1)
            x = mx.where(free, mx.where(free & (rank < k_keep), proposal, M), x)
        else:
            k = block_size if mode == 'block' else int(round(n_free * a(s))) - int(round(n_free * a(t)))
            conf = mx.take_along_axis(mu, x0hat[..., None], -1).squeeze(-1)
            score = {'confidence': conf, 'left': -mx.arange(L, dtype=mx.float32)[None] * mx.ones((B, 1)), 'block': -mx.arange(L, dtype=mx.float32)[None] * mx.ones((B, 1)), 'random': mx.random.uniform(shape=x.shape)}[mode]
            score = mx.where(masked, score, -mx.inf)
            rank = mx.argsort(mx.argsort(-score, -1), -1)
            x = mx.where(masked & (rank < max(k, 0)), x0hat, x)
        mx.eval(x)
    mu, logits = predict_mu(model, x, 0.0, B)
    return mx.where(x == M, mx.argmax(logits, -1), x)

def sample_ancestral(model, ker, B=8, steps=None, temp=1.0, prompt=None):
    strided = steps is not None and ker.kind in ('absorb', 'uniform')
    grid = [int(round(g)) for g in np.linspace(ker.T, 0, steps + 1)] if strided else list(range(ker.T, -1, -1))
    free = mx.ones((B, L), dtype=mx.bool_) if prompt is None else prompt < 0
    u = mx.random.randint(0, K, (B, L))
    x = mx.random.categorical(mx.log(onehot(u, Kx) @ ker.Qbar[ker.T] + 1e-12))
    if prompt is not None:
        x = mx.where(free, x, prompt)
    for gi, gj in zip(grid[:-1], grid[1:]):
        if gi == gj:
            continue
        mu, _ = predict_mu(model, x, gi / ker.T, B)
        if strided and gj != gi - 1:
            Qint = mx.array(make_Q(ker.kind, 1.0 - ker.al[gi] / max(ker.al[gj], 1e-12)).T)
            fwd = onehot(x, Kx) @ Qint
        else:
            fwd = onehot(x, Kx) @ ker.QsT[gi - 1]
        prev = mx.concatenate([mu, mx.zeros((B, L, 1))], -1) @ ker.Qbar[gj]
        post = fwd * prev
        lg = mx.log(post / mx.maximum(post.sum(-1, keepdims=True), 1e-12) + 1e-12)
        xn = mx.argmax(lg, -1) if temp == 0 else mx.random.categorical(lg / temp)
        x = mx.where(free, xn, x)
        mx.eval(x)
    return x

def sample_tau(model, ker, B=8, steps=8, prompt=None):
    a = ker.alpha
    free = mx.ones((B, L), dtype=mx.bool_) if prompt is None else prompt < 0
    x = mx.full((B, L), M) if ker.kind == 'absorb' else mx.random.randint(0, K, (B, L))
    if prompt is not None:
        x = mx.where(free, x, prompt)
    ts = np.linspace(1.0, 0.0, steps + 1)
    for t, s in zip(ts[:-1], ts[1:]):
        out = model(x, mx.full((B, 1), float(t)))
        if model.param == 'ratio':
            sc = mx.exp(out)[..., :K]
        else:
            sc = a(t) / max(1 - a(t), 1e-09) * mx.softmax(out, -1)
        if ker.kind == 'absorb':
            p = sc * ((a(s) - a(t)) / max(a(t), 1e-09))
            active = x == M
        else:
            tau_i = min(math.log(max(a(s), 1e-06) / max(a(t), 1e-06)), 4.0)
            p = sc * (1.0 / K) * tau_i
            p = p * (1.0 - onehot(mx.minimum(x, K - 1), K))
            active = free
        jump = active & free & (mx.random.uniform(shape=x.shape) < mx.minimum(p.sum(-1), 1.0))
        dest = mx.random.categorical(mx.log(mx.maximum(p, 1e-30)))
        x = mx.where(jump, dest, x)
        mx.eval(x)
    if ker.kind == 'absorb':
        _, logits = predict_mu(model, x, 0.0, B)
        x = mx.where((x == M) & free, mx.argmax(logits, -1), x)
    return x

def run_sampler(model, ker, sampler, B, steps, temp, prompt, block_size=2):
    if sampler == 'ancestral':
        return sample_ancestral(model, ker, B, steps=steps, temp=temp, prompt=prompt)
    if sampler == 'tau':
        return sample_tau(model, ker, B, steps=steps, prompt=prompt)
    return sample_committing(model, ker, B, steps, mode=sampler, temp=temp, prompt=prompt, block_size=block_size)

def valid_samplers(kind, param):
    v = []
    if param == 'x0' or kind == 'absorb':
        v.append('ancestral')
    if kind in ('absorb', 'hybrid'):
        v += ['confidence', 'random', 'left', 'block'] + (['remask'] if param == 'x0' else [])
    if kind == 'absorb' or (kind == 'uniform' and param == 'ratio'):
        v.append('tau')
    return v

def report_prior(ker, n=512):
    rows = onehot(get_batch(n), Kx) @ ker.Qbar[ker.T]
    prior = rows.mean((0, 1), keepdims=True)
    kl = (rows * (mx.log(rows + 1e-12) - mx.log(prior + 1e-12))).sum(-1).mean().item()
    print(f'  L_prior (Eq 15) = {kl:.4f} nats/token (theta-free; the ELBO term training never touches)')

def loss_curve(model, ker, objective, n=512, bins=8):
    out = []
    for tv in np.unique(np.linspace(max(1, ker.T // bins), ker.T, bins).astype(int)):
        out.append((tv / ker.T, LOSSES[objective](model, ker, get_batch(n), ti=int(tv)).item()))
    return out

def evaluate(model, ker, sampler, steps, temp=0.0, n=256, block_size=2):
    x0 = get_batch(n)
    prompt = mx.where(mx.arange(L)[None] < 6, x0, -1)
    out = decode(run_sampler(model, ker, sampler, n, steps, temp, prompt, block_size))
    ok = sum((t[6:].isdigit() and int(t[6:]) == int(t[:2]) + int(t[3:5]) for t in out))
    return (ok / n, out[:3])

def evaluate_rev(model, ker, sampler, steps, temp=1.0, n=256, block_size=2):
    x0 = get_batch(n)
    prompt = mx.where(mx.arange(L)[None] >= 5, x0, -1)
    out = decode(run_sampler(model, ker, sampler, n, steps, temp, prompt, block_size))
    ok = sum((t[:2].isdigit() and t[3:5].isdigit() and t[6:].isdigit() and (int(t[:2]) + int(t[3:5]) == int(t[6:])) for t in out))
    return (ok / n, out[:3])

def train(model, ker, objective, steps=3000, B=256, lr=0.0003):
    loss_and_grad = nn.value_and_grad(model, lambda m, x: LOSSES[objective](m, ker, x))
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps, 1e-05))
    tic, run = (time.perf_counter(), 0.0)
    for i in range(steps):
        loss, grads = loss_and_grad(model, get_batch(B))
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        run += loss.item()
        if (i + 1) % 500 == 0:
            print(f'  step {i + 1:5d}  loss {run / 500:.4f}  ({time.perf_counter() - tic:.1f}s)')
            run = 0.0
    return model

def main(kind='absorb', param='x0', objective='simple', sampler='confidence', steps=8, T=32, train_steps=3000, use_time=None, schedule='linear', eta=0.1, sigma=2.0, temp=1.0, block_size=2, selftest_only=False):
    if selftest_only:
        return selftest()
    assert (param == 'ratio') == (objective == 'score_entropy'), 'Table 6 pairing: ratio <-> score_entropy; x0 <-> simple/elbo'
    use_time = param == 'ratio' if use_time is None else use_time
    ker = Kernels(kind, T, schedule, eta, sigma)
    model = Denoiser(param=param, use_time=use_time)
    mx.eval(model.parameters())
    avail = valid_samplers(kind, param)
    assert sampler in avail, f'--sampler {sampler} incompatible with kind={kind}, param={param}; valid: {avail}'
    print(f'kind={kind}  param={param}  objective={objective}  sampler={sampler}  T={T}  steps={steps}  use_time={use_time}')
    print(f'  compatible samplers for this cell: {avail}')
    report_prior(ker)
    train(model, ker, objective, steps=train_steps)
    if objective == 'elbo':
        print('  (elbo loss window above ~ NELBO nats/token, Eq 12-14 estimate; add L_prior for the full bound)')
    print('\n  L(t)  [Table 4 "Denoising loss curve", from the training objective]')
    print('    ' + '  '.join((f't={t:.2f}:{v:.2f}' for t, v in loss_curve(model, ker, objective))))
    print('\n  FORWARD  pin "AA+BB=", generate "CCC"  (1 valid completion)')
    for smp in avail:
        acc, ex = evaluate(model, ker, smp, steps, temp=0.0, block_size=block_size)
        lbl = f'{('T=' + str(T) if smp == 'ancestral' and kind in ('hybrid', 'structured') else steps)} steps'
        print(f'    {smp:<11}{lbl:<11} acc {acc:.3f}   {ex}')
    print('\n  REVERSE  pin "=CCC", generate "AA+BB"  (19-67 valid completions; factorization bites)')
    for smp in avail:
        if smp == 'ancestral' and kind in ('hybrid', 'structured'):
            acc, ex = evaluate_rev(model, ker, smp, None, temp=temp)
            print(f'    {smp:<11}T={T} steps: {acc:.3f}   {ex}')
            continue
        row = []
        for st in [1, 2, 3, 5]:
            acc, _ = evaluate_rev(model, ker, smp, st, temp=temp, block_size=block_size)
            row.append(f'{st}st:{acc:.3f}')
        _, ex = evaluate_rev(model, ker, smp, 5, temp=temp, block_size=block_size)
        print(f'    {smp:<11}' + '  '.join(row) + f'   {ex}')
    x0 = get_batch(6)
    hole = mx.where(mx.arange(L)[None] == 1, -1, x0)
    print('\n  INFILL  pin all but one operand digit  [Eq 6 conditioning; Sec 10.4]')
    print('    ', list(zip(decode(hole), decode(run_sampler(model, ker, sampler, 6, 4, temp, hole, block_size)))))

def selftest():
    rng = np.random.default_rng(0)
    oh = lambda x, n: np.eye(n)[x]
    gs = lambda p: np.argmax(np.log(np.maximum(p, 1e-30)) + rng.gumbel(size=p.shape), -1)
    ok = [True]

    def chk(name, cond, detail=''):
        ok[0] &= bool(cond)
        print(f'  [{('PASS' if cond else 'FAIL')}] {name} {detail}')
    T = 32
    x0seq = np.array([VOCAB.index(c) for c in '47+85=132'])
    for kind in ['absorb', 'uniform', 'hybrid', 'structured']:
        Qs, Qbar, al = build_kernels(kind, T)
        chk(f'{kind}: Qt and Qbar rows sum to 1 (Eq 3/4)', np.allclose(Qs.sum(-1), 1, atol=1e-09) and np.allclose(Qbar.sum(-1), 1, atol=1e-09))
    Qs, Qbar, al = build_kernels('uniform', T)
    ti = 20
    chk('Eq 8 composed: Qbar_t = a I + (1-a)/K 11^T', np.allclose(Qbar[ti][:K, :K], al[ti] * np.eye(K) + (1 - al[ti]) / K, atol=1e-09))
    Qs, Qbar, al = build_kernels('absorb', T)
    chk('Eq 9 composed: Qbar[j,j] = a, Qbar[j,m] = 1-a', np.allclose(np.diag(Qbar[ti])[:K], al[ti]) and np.allclose(Qbar[ti][:K, M], 1 - al[ti]))
    for kind in ['uniform', 'absorb']:
        chk(f'{kind}: interval closure Q(a1)Q(a2)=Q(a1a2) (Sec 5.4 strided sampling)', np.allclose(make_Q(kind, 0.3) @ make_Q(kind, 0.4), make_Q(kind, 1 - 0.7 * 0.6), atol=1e-12))
    Ts = 6
    Qs6, Qbar6, _ = build_kernels('structured', Ts)
    N = 60000
    st = np.full((N, Ts + 1), 3)
    for i in range(Ts):
        st[:, i + 1] = gs(Qs6[i][st[:, i]])
    worst = 0.0
    for v in range(Kx):
        sel = st[:, 4] == v
        if sel.sum() < 2500:
            continue
        emp = np.bincount(st[:, 3][sel], minlength=Kx) / sel.sum()
        worst = max(worst, np.abs(emp - np_posterior(Qs6, Qbar6, 4, np.array(v), oh(np.array(3), Kx))).max())
    chk('Eq 7 posterior == simulated chain conditional (structured)', worst < 0.03, f'max err {worst:.4f}')
    for kind in ['absorb', 'uniform', 'hybrid', 'structured']:
        Qs, Qbar, al = build_kernels(kind, T)
        Ntr = 150
        x = gs(np.repeat(Qbar[T][x0seq][None], Ntr, 0))
        for ti_ in range(T, 0, -1):
            fwd = np.swapaxes(Qs[ti_ - 1], 0, 1)[x]
            prev = (oh(x0seq, Kx) @ Qbar[ti_ - 1])[None]
            post = fwd * prev
            post /= np.maximum(post.sum(-1, keepdims=True), 1e-30)
            x = gs(post)
        chk(f'{kind}: oracle predict-x0 ancestral recovers x0 (Eq 7 + Sec 7.1)', (x == x0seq).all(-1).mean() > 0.97)
    r = rng.uniform(0.1, 3.0, 7)
    f = lambda s_: np.sum(s_ - r * np.log(s_) + r * np.log(r) - r)
    chk('score entropy: f(r)=0, f(1.3r)>0 (Sec 6.3)', abs(f(r)) < 1e-09 and f(1.3 * r) > 0)
    Qs, Qbar, al = build_kernels('absorb', T)
    rows = Qbar[20][x0seq[0]]
    exp_ = np.zeros(Kx)
    exp_[x0seq[0]] = al[20] / (1 - al[20])
    exp_[M] = 1
    chk('Eq 17: ratios at m == a/(1-a) one-hot(x0)', np.allclose(rows / rows[M], exp_, atol=1e-09))
    af = alpha_fn('linear')
    Ntr, stp = (800, 8)
    x = np.full((Ntr, L), M)
    mid = 0.0
    for t, s in zip(np.linspace(1, 0, stp + 1)[:-1], np.linspace(1, 0, stp + 1)[1:]):
        at, as_ = (float(af(t)), float(af(s)))
        sy = np.where(np.arange(Kx)[None, None, :] == x0seq[None, :, None], at / max(1 - at, 1e-09), 0.0)[..., :K]
        p = sy * ((as_ - at) / max(at, 1e-09))
        jump = (rng.random(x.shape) < np.minimum(p.sum(-1), 1)) & (x == M)
        x = np.where(jump, gs(np.maximum(p, 1e-30)), x)
        if abs(s - 0.5) < 1e-09:
            mid = (x != M).mean()
    chk('tau absorb: oracle recovery, marginal tracks alpha (Eq 11 / Table 5)', (x == x0seq).all(-1).mean() > 0.99, f'unmasked@0.5={mid:.3f} vs alpha=0.500')
    Qs, Qbar, al = build_kernels('uniform', T)
    for stp in [8, 64]:
        Ntr = 500
        x = rng.integers(0, K, (Ntr, L))
        for t, s in zip(np.linspace(1, 0, stp + 1)[:-1], np.linspace(1, 0, stp + 1)[1:]):
            rows = Qbar[max(int(round(t * T)), 1)][x0seq]
            qx = np.take_along_axis(np.repeat(rows[None], Ntr, 0), x[..., None], -1)
            p = (rows[None] / np.maximum(qx, 1e-30))[..., :K] / K * min(np.log(max(af(s), 1e-06) / max(af(t), 1e-06)), 4.0)
            np.put_along_axis(p, x[..., None], 0.0, -1)
            jump = rng.random(x.shape) < np.minimum(p.sum(-1), 1)
            x = np.where(jump, gs(np.maximum(p, 1e-30)), x)
        print(f'  [INFO] tau uniform, {stp:2d} steps: oracle recovery {(x == x0seq).all(-1).mean():.3f} (first-order leap)')
    print('  == ALL PASS ==' if ok[0] else '  == FAILURES ==')
    return ok[0]

if __name__ == '__main__':
    fire.Fire(main)

# kind=absorb  param=x0  objective=simple  sampler=confidence  T=32  steps=8  use_time=False
#   compatible samplers for this cell: ['ancestral', 'confidence', 'random', 'left', 'block', 'remask', 'tau']
#   L_prior (Eq 15) = 0.0000 nats/token (theta-free; the ELBO term training never touches)
#   step   500  loss 1.3699  (7.0s)
#   step  1000  loss 1.1882  (14.9s)
#   step  1500  loss 0.9864  (22.7s)
#   step  2000  loss 0.9684  (30.4s)
#   step  2500  loss 0.9687  (38.0s)
#   step  3000  loss 0.9659  (45.9s)

#   L(t)  [Table 4 "Denoising loss curve", from the training objective]
#     t=0.12:0.10  t=0.25:0.44  t=0.38:0.76  t=0.50:1.02  t=0.62:1.28  t=0.75:1.46  t=0.88:1.57  t=1.00:1.61

#   FORWARD  pin "AA+BB=", generate "CCC"  (1 valid completion)
#     ancestral  8 steps     acc 1.000   ['53+62=115', '10+10=020', '36+61=097']
#     confidence 8 steps     acc 1.000   ['36+14=050', '55+57=112', '85+25=110']
#     random     8 steps     acc 1.000   ['20+65=085', '60+16=076', '81+88=169']
#     left       8 steps     acc 1.000   ['69+84=153', '75+41=116', '29+05=034']
#     block      8 steps     acc 1.000   ['35+68=103', '98+24=122', '00+39=039']
#     remask     8 steps     acc 1.000   ['39+33=072', '17+33=050', '06+36=042']
#     tau        8 steps     acc 0.973   ['10+72=082', '45+16=061', '91+88=179']

#   REVERSE  pin "=CCC", generate "AA+BB"  (19-67 valid completions; factorization bites)
#     ancestral  1st:0.023  2st:0.262  3st:0.414  5st:0.605   ['18+14=032', '92+70=162', '18+48=116']
#     confidence 1st:0.008  2st:0.078  3st:0.117  5st:0.746   ['92+65=157', '79+46=125', '48+74=122']
#     random     1st:0.012  2st:0.379  3st:0.656  5st:0.902   ['83+17=100', '92+96=188', '05+83=088']
#     left       1st:0.016  2st:0.906  3st:0.918  5st:0.957   ['08+26=034', '25+80=105', '29+08=017']
#     block      1st:0.957  2st:0.922  3st:0.926  5st:0.918   ['45+25=070', '76+49=125', '40+90=130']
#     remask     1st:0.012  2st:0.074  3st:0.734  5st:0.836   ['30+71=101', '88+21=109', '02+54=056']
#     tau        1st:0.012  2st:0.016  3st:0.273  5st:0.586   ['11+42=063', '32+06=038', '35+97=132']

#   INFILL  pin all but one operand digit  [Eq 6 conditioning; Sec 10.4]
#      [('5_+45=103', '58+45=103'), ('8_+18=104', '86+18=104'), ('1_+96=115', '19+96=115'), ('1_+64=079', '15+64=079'), ('8_+99=186', '87+99=186'), ('9_+35=125', '90+35=125')]
