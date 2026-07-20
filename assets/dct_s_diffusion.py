import time
from datetime import datetime
import fire
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset
from mlx.utils import tree_flatten
from PIL import Image
from scipy.fftpack import dctn, idctn

EPS = 1e-05
_YCBCR = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
_YCBCR_INV = np.linalg.inv(_YCBCR)
_YCBCR_BIAS = np.array([0.0, 0.5, 0.5])

def rgb_to_ycbcr(x):
    return x @ _YCBCR.T + _YCBCR_BIAS if x.shape[-1] == 3 else x

def ycbcr_to_rgb(x):
    return (x - _YCBCR_BIAS) @ _YCBCR_INV.T if x.shape[-1] == 3 else x

def fit_scale(images_uint8, eps=0.001):
    x = images_uint8.astype(np.float64) / 255.0
    x = rgb_to_ycbcr(x)
    coeffs = dctn(x, axes=(1, 2), norm='ortho')
    return np.maximum(coeffs.std(axis=0), eps).astype(np.float32)

def to_whitened(images_uint8, scale):
    x = images_uint8.astype(np.float64) / 255.0
    x = rgb_to_ycbcr(x)
    return (dctn(x, axes=(1, 2), norm='ortho') / scale).astype(np.float32)

def from_whitened(coeffs, scale):
    x = idctn(coeffs * scale, axes=(1, 2), norm='ortho')
    x = ycbcr_to_rgb(x)
    return np.clip(x, 0, 1)

def fourier_features(idx, n_freqs=6, max_period=64.0):
    idx = np.asarray(idx, dtype=np.float64)
    freqs = 2 * np.pi / max_period ** (np.arange(n_freqs) / n_freqs)
    ang = idx[..., None] * freqs
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)

def build_coord_table(H, W, n_freqs=6):
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    fi = fourier_features(ii.flatten(), n_freqs, max_period=H)
    fj = fourier_features(jj.flatten(), n_freqs, max_period=W)
    return np.concatenate([fi, fj], axis=-1).astype(np.float32)

def prepare_sequences(images_uint8, scale, k_keep):
    B, H, W, C = images_uint8.shape
    whitened = to_whitened(images_uint8, scale)
    raw = whitened * scale
    energy = (raw ** 2).sum(axis=-1).reshape(B, H * W)
    order = np.argsort(-energy, axis=1)[:, :k_keep].astype(np.int32)
    values = np.take_along_axis(whitened.reshape(B, H * W, C), order[..., None], axis=1)
    return (order, values.astype(np.float32))

def reconstruct(order, values, scale, H, W, C):
    B = order.shape[0]
    flat = np.zeros((B, H * W, C), dtype=np.float32)
    np.put_along_axis(flat, order[..., None], values, axis=1)
    return from_whitened(flat.reshape(B, H, W, C), scale)

class MLP(nn.Module):

    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.gate_up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(dim, out_dim, bias=False)

    def __call__(self, x):
        gate, x = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)

class Attention(nn.Module):

    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.scale = (dim // n_head) ** (-0.5)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        B, L, _ = x.shape
        q, k, v = mx.split(self.qkv_proj(x), 3, axis=-1)
        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((L, L), -mx.inf), k=1)
        w = mx.softmax(q * self.scale @ k.transpose(0, 1, 3, 2) + mask, axis=-1)
        o = (w @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)

class Layer(nn.Module):

    def __init__(self, dim, n_head):
        super().__init__()
        self.self_attn = Attention(dim, n_head)
        self.mlp = MLP(dim)
        self.input_layernorm = nn.RMSNorm(dim, eps=EPS)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=EPS)

    def __call__(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Transformer(nn.Module):

    def __init__(self, dim, n_head, n_layer):
        super().__init__()
        self.layers = [Layer(dim, n_head) for _ in range(n_layer)]
        self.norm = nn.RMSNorm(dim, eps=EPS)

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return self.norm(x)

def log_softmax(x, axis=-1):
    m = mx.stop_gradient(mx.max(x, axis=axis, keepdims=True))
    z = x - m
    return z - mx.log(mx.sum(mx.exp(z), axis=axis, keepdims=True))

class SparseAR(nn.Module):

    def __init__(self, H, W, C, n_classes, dim=256, n_head=4, n_layer=6, n_freqs=6):
        super().__init__()
        self.H, self.W, self.C = (H, W, C)
        self.n_pos = H * W
        coord_dim = 4 * n_freqs
        self.coord_dim = coord_dim
        self.coord_table = mx.array(build_coord_table(H, W, n_freqs))
        self.class_embed = nn.Embedding(n_classes, dim)
        self.token_in = nn.Linear(coord_dim + C, dim, bias=False)
        self.transformer = Transformer(dim, n_head, n_layer)
        self.coord_head = nn.Linear(dim, self.n_pos, bias=False)
        self.value_head = nn.Sequential(nn.Linear(dim + coord_dim, dim, bias=False), nn.SiLU(), nn.Linear(dim, C, bias=False))

    def _embed(self, positions, values):
        coord_feats = self.coord_table[positions]
        return self.token_in(mx.concatenate([coord_feats, values], axis=-1))

    def loss(self, positions, values, labels):
        start = self.class_embed(labels)[:, None, :]
        toks = self._embed(positions[:, :-1], values[:, :-1])
        h = self.transformer(mx.concatenate([start, toks], axis=1))
        coord_logits = self.coord_head(h)
        logp = log_softmax(coord_logits, axis=-1)
        onehot = (mx.arange(self.n_pos) == positions[..., None]).astype(mx.float32)
        coord_loss = -mx.mean(mx.sum(onehot * logp, axis=-1))
        true_coord_feats = self.coord_table[positions]
        value_pred = self.value_head(mx.concatenate([h, true_coord_feats], axis=-1))
        value_loss = mx.mean((value_pred - values) ** 2)
        return (coord_loss, value_loss)

    def sample(self, labels, k_keep, temperature=0.9):
        B = labels.shape[0]
        x = self.class_embed(labels)[:, None, :]
        used = np.zeros((B, self.n_pos), dtype=bool)
        out_pos = np.zeros((B, k_keep), dtype=np.int32)
        out_val = np.zeros((B, k_keep, self.C), dtype=np.float32)
        for t in range(k_keep):
            h = self.transformer(x)[:, -1]
            logits = np.array(self.coord_head(h))
            logits[used] = -np.inf
            if temperature <= 0:
                pos = logits.argmax(axis=-1)
            else:
                z = logits - logits.max(axis=-1, keepdims=True)
                p = np.exp(z / temperature)
                p /= p.sum(axis=-1, keepdims=True)
                pos = np.array([np.random.choice(self.n_pos, p=p[b]) for b in range(B)])
            used[np.arange(B), pos] = True
            pos_mx = mx.array(pos.astype(np.int32))
            coord_feats = self.coord_table[pos_mx]
            val = self.value_head(mx.concatenate([h, coord_feats], axis=-1))
            out_pos[:, t] = pos
            out_val[:, t] = np.array(val)
            next_tok = self.token_in(mx.concatenate([coord_feats, val], axis=-1))[:, None, :]
            x = mx.concatenate([x, next_tok], axis=1)
        return (out_pos, out_val)

def load_data(dataset_name, split='train'):
    ds = load_dataset(dataset_name, split=split)
    key = 'image' if 'image' in ds.features else 'img'
    images = np.stack([np.array(x) for x in ds[key]])
    if images.ndim == 3:
        images = images[..., None]
    labels = np.array(ds['label']).astype(np.int32)
    return (images.astype(np.uint8), labels)

def save_grid(images01, f_name, n_side):
    x = (images01 * 255).astype(np.uint8)
    C = x.shape[-1]
    x = x.reshape(n_side, n_side, *x.shape[1:])
    H, W = (x.shape[2], x.shape[3])
    grid = x.transpose(0, 2, 1, 3, 4).reshape(n_side * H, n_side * W, C)
    if C == 1:
        grid = grid.squeeze(-1)
    Image.fromarray(grid).save(f_name)

def sample_and_save(model, scale, n_classes, k_keep, f_name, n_side=8, temperature=0.9):
    model.eval()
    reps = n_side * n_side // n_classes + 1
    labels = mx.array(np.tile(np.arange(n_classes), reps)[:n_side * n_side].astype(np.int32))
    order, values = model.sample(labels, k_keep=k_keep, temperature=temperature)
    imgs = reconstruct(order, values, scale, model.H, model.W, model.C)
    save_grid(imgs, f_name, n_side)
    model.train()
    print(f'saved {f_name}')

def train(dataset_name='mnist', k_keep=None, n_epoch=20, batch_size=128, lr=0.0003, dim=256, n_head=4, n_layer=6, n_freqs=6, n_scale_samples=4000, postfix=''):
    images, labels = load_data(dataset_name)
    H, W, C = images.shape[1:]
    n_classes = int(labels.max()) + 1
    if k_keep is None:
        k_keep = H * W // 4
    rng = np.random.default_rng(0)
    idx = rng.choice(len(images), size=min(n_scale_samples, len(images)), replace=False)
    scale = fit_scale(images[idx])
    model = SparseAR(H, W, C, n_classes, dim=dim, n_head=n_head, n_layer=n_layer, n_freqs=n_freqs)
    mx.eval(model)
    optimizer = optim.AdamW(learning_rate=lr)

    def loss_fn(model, positions, values, labels):
        coord_loss, value_loss = model.loss(positions, values, labels)
        return coord_loss + value_loss
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    f_name = f'{dataset_name}_sparsear_{datetime.now().strftime('%Y%m%d_%H%M%S')}{postfix}'
    print(f'{f_name}  H={H} W={W} C={C}  n_classes={n_classes}  k_keep={k_keep}/{H * W}')
    for e in range(n_epoch):
        perm = rng.permutation(len(images))
        tic = time.perf_counter()
        total, n_batch = (0.0, 0)
        for i in range(0, len(images) - batch_size + 1, batch_size):
            b = perm[i:i + batch_size]
            order, values = prepare_sequences(images[b], scale, k_keep)
            pos_mx, val_mx, lbl_mx = (mx.array(order), mx.array(values), mx.array(labels[b]))
            loss, grads = loss_and_grad_fn(model, pos_mx, val_mx, lbl_mx)
            optimizer.update(model, grads)
            mx.eval(model, optimizer, loss)
            total += loss.item()
            n_batch += 1
        print(f'{total / n_batch:.4f} @ epoch {e} in {time.perf_counter() - tic:.2f}s')
        if (e + 1) % max(1, n_epoch // 5) == 0:
            sample_and_save(model, scale, n_classes, k_keep, f'{f_name}_e{e}.png')
    mx.save_safetensors(f'{f_name}.safetensors', dict(tree_flatten(model.trainable_parameters())))
    np.save(f'{f_name}_scale.npy', scale)
    print(f'done: {f_name}.safetensors')

if __name__ == '__main__':
    fire.Fire(train)
