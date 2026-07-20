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
    x = rgb_to_ycbcr(images_uint8.astype(np.float64) / 255.0)
    coeffs = dctn(x, axes=(1, 2), norm='ortho')
    return np.maximum(coeffs.std(axis=0), eps).astype(np.float32)

def to_whitened(images_uint8, scale):
    x = rgb_to_ycbcr(images_uint8.astype(np.float64) / 255.0)
    return (dctn(x, axes=(1, 2), norm='ortho') / scale).astype(np.float32)

def from_whitened(coeffs, scale):
    x = idctn(coeffs * scale, axes=(1, 2), norm='ortho')
    x = ycbcr_to_rgb(x)
    return np.clip(x, 0, 1)

def reconstruct(order, values, scale, H, W, C):
    B = order.shape[0]
    flat = np.zeros((B, H * W, C), dtype=np.float32)
    np.put_along_axis(flat, order[..., None], values, axis=1)
    return from_whitened(flat.reshape(B, H, W, C), scale)

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
DEFAULT_CHANNEL_WEIGHT = {1: np.array([1.0], dtype=np.float32), 3: np.array([1.0, 0.25, 0.25], dtype=np.float32)}

def compute_energy(images_uint8, scale, channel_weight):
    whitened = to_whitened(images_uint8, scale)
    raw = whitened * scale
    B, H, W, C = raw.shape
    return (raw ** 2 * channel_weight).sum(axis=-1).reshape(B, H * W)

def fit_energy_norm(images_uint8, scale, channel_weight, eps=1e-08):
    log_e = np.log(compute_energy(images_uint8, scale, channel_weight) + eps)
    return (log_e.mean(axis=0).astype(np.float32), np.maximum(log_e.std(axis=0), 0.001).astype(np.float32))

def energy_target(images_uint8, scale, channel_weight, energy_mean, energy_std, eps=1e-08):
    log_e = np.log(compute_energy(images_uint8, scale, channel_weight) + eps)
    return (((log_e - energy_mean) / energy_std).astype(np.float32), log_e.astype(np.float32))

def gumbel_topk(log_energy, k_keep, noise_scale, rng):
    if noise_scale <= 0:
        return np.argsort(-log_energy, axis=1)[:, :k_keep].astype(np.int32)
    u = rng.uniform(1e-08, 1 - 1e-08, size=log_energy.shape)
    gumbel = -np.log(-np.log(u)).astype(np.float32)
    perturbed = log_energy + gumbel * noise_scale
    return np.argsort(-perturbed, axis=1)[:, :k_keep].astype(np.int32)

def sinusoidal_time_embed_table(dim):
    half = dim // 2
    return np.exp(-np.log(10000.0) * np.arange(half) / max(half - 1, 1)).astype(np.float32)

class GatedMLP(nn.Module):

    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.gate_up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(dim, out_dim, bias=False)

    def __call__(self, x):
        gate, x = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)

class PlainMLP(nn.Module):

    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, out_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.gelu(self.up_proj(x)))

class SparseFlow(nn.Module):

    def __init__(self, H, W, C, n_classes, ctx_dim=128, dist_hidden=512, dist_layer=3, denoise_hidden=256, denoise_layer=2, n_freqs=6, channel_weight=None, mlp_variant='gated'):
        super().__init__()
        self.H, self.W, self.C = (H, W, C)
        self.n_pos = H * W
        coord_dim = 4 * n_freqs
        self.ctx_dim = ctx_dim
        self._coord_table = mx.array(build_coord_table(H, W, n_freqs))
        self._time_freqs = mx.array(sinusoidal_time_embed_table(ctx_dim))
        if channel_weight is None:
            channel_weight = DEFAULT_CHANNEL_WEIGHT.get(C, np.ones(C, dtype=np.float32))
        self._channel_weight = mx.array(channel_weight.astype(np.float32))
        Block = GatedMLP if mlp_variant == 'gated' else PlainMLP
        self.dist_class_embed = nn.Embedding(n_classes, ctx_dim)
        self.dist_in = nn.Linear(self.n_pos + 2 * ctx_dim, dist_hidden, bias=False)
        self.dist_layers = [Block(dist_hidden) for _ in range(dist_layer)]
        self.dist_norms = [nn.RMSNorm(dist_hidden, eps=EPS) for _ in range(dist_layer)]
        self.dist_final_norm = nn.RMSNorm(dist_hidden, eps=EPS)
        self.dist_out = nn.Linear(dist_hidden, self.n_pos, bias=False)
        self.den_class_embed = nn.Embedding(n_classes, denoise_hidden)
        self.den_coord_proj = nn.Linear(coord_dim, denoise_hidden, bias=False)
        self.den_layers = [Block(denoise_hidden) for _ in range(denoise_layer)]
        self.den_norms = [nn.RMSNorm(denoise_hidden, eps=EPS) for _ in range(denoise_layer)]
        self.den_final_norm = nn.RMSNorm(denoise_hidden, eps=EPS)
        self.den_out = nn.Linear(denoise_hidden, C, bias=False)

    def _time_embed(self, t):
        ang = t[:, None] * self._time_freqs[None, :]
        return mx.concatenate([mx.sin(ang), mx.cos(ang)], axis=-1)

    def velocity(self, x_t, t, labels):
        c = mx.concatenate([self.dist_class_embed(labels), self._time_embed(t)], axis=-1)
        h = self.dist_in(mx.concatenate([x_t, c], axis=-1))
        for l, n in zip(self.dist_layers, self.dist_norms):
            h = h + l(n(h))
        return self.dist_out(self.dist_final_norm(h))

    def denoise(self, labels, positions):
        coord_feats = self._coord_table[positions]
        ctx = self.den_class_embed(labels)[:, None, :]
        h = self.den_coord_proj(coord_feats) + ctx
        for l, n in zip(self.den_layers, self.den_norms):
            h = h + l(n(h))
        return self.den_out(self.den_final_norm(h))

    def loss(self, x0, x1, t, labels, den_positions, den_values):
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        target_v = x1 - x0
        pred_v = self.velocity(x_t, t, labels)
        flow_loss = mx.mean((pred_v - target_v) ** 2)
        value_pred = self.denoise(labels, den_positions)
        value_loss = mx.mean((value_pred - den_values) ** 2 * self._channel_weight)
        return (flow_loss, value_loss)

    def sample(self, labels, k_keep, energy_mean, energy_std, n_steps=1):
        B = labels.shape[0]
        x = mx.random.normal((B, self.n_pos))
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = mx.full((B,), i * dt)
            v = self.velocity(x, t, labels)
            x = x + v * dt
        pred_norm = np.array(x)
        pred_energy = pred_norm * np.array(energy_std)[None, :] + np.array(energy_mean)[None, :]
        order = np.argsort(-pred_energy, axis=1)[:, :k_keep].astype(np.int32)
        values = np.array(self.denoise(labels, mx.array(order)))
        return (order, values)

def load_data(dataset_name, split='train'):
    ds = load_dataset(dataset_name, split=split)
    key = 'image' if 'image' in ds.features else 'img'
    images = np.stack([np.array(x) for x in ds[key]])
    if images.ndim == 3:
        images = images[..., None]
    labels = np.array(ds['label']).astype(np.int32)
    return (images.astype(np.uint8), labels)

def module_norm(module):
    total = 0.0
    for _, p in tree_flatten(module.parameters()):
        total += float(mx.sum(p.astype(mx.float32) ** 2).item())
    return total ** 0.5

def mx_max_abs(a):
    return float(mx.max(mx.abs(a)).item())

def mx_has_bad(a):
    return bool(mx.any(mx.isnan(a) | mx.isinf(a)).item())

def debug_stats(model, x0, x1, t, labels, den_pos):
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    v = model.velocity(x_t, t, labels)
    coord_feats = model._coord_table[den_pos]
    ctx = model.den_class_embed(labels)[:, None, :]
    h = model.den_coord_proj(coord_feats) + ctx
    block_maxes = [mx_max_abs(h)]
    for l, n in zip(model.den_layers, model.den_norms):
        h = h + l(n(h))
        block_maxes.append(mx_max_abs(h))
    h_normed = model.den_final_norm(h)
    block_maxes.append(mx_max_abs(h_normed))
    val = model.den_out(h_normed)
    return {'velocity_max': mx_max_abs(v), 'velocity_bad': mx_has_bad(v), 'den_block_maxes': block_maxes, 'den_value_pred_max': mx_max_abs(val), 'den_value_bad': mx_has_bad(val), 'den_out_norm': module_norm(model.den_out), 'den_coord_proj_norm': module_norm(model.den_coord_proj), 'den_final_norm_norm': module_norm(model.den_final_norm), 'dist_out_norm': module_norm(model.dist_out), 'dist_final_norm_norm': module_norm(model.dist_final_norm), 'dist_in_norm': module_norm(model.dist_in), 'channel_weight': np.array(model._channel_weight).tolist()}

def print_debug_stats(prefix, ds, lr):
    block_str = '->'.join((f'{m:.3g}' for m in ds['den_block_maxes']))
    print(f'{prefix} lr={lr:.2e}  channel_weight={ds['channel_weight']}  den_h[in->b1->b2->final]=[{block_str}]  value_max={ds['den_value_pred_max']:.3g} bad={ds['den_value_bad']}  weight_norms[den_out={ds['den_out_norm']:.3g} den_coord_proj={ds['den_coord_proj_norm']:.3g} den_final_norm={ds['den_final_norm_norm']:.3g} dist_out={ds['dist_out_norm']:.3g} dist_in={ds['dist_in_norm']:.3g} dist_final_norm={ds['dist_final_norm_norm']:.3g}]  velocity_max={ds['velocity_max']:.3g} velocity_bad={ds['velocity_bad']}')

def save_grid(images01, f_name, n_side):
    x = (images01 * 255).astype(np.uint8)
    C = x.shape[-1]
    x = x.reshape(n_side, n_side, *x.shape[1:])
    H, W = (x.shape[2], x.shape[3])
    grid = x.transpose(0, 2, 1, 3, 4).reshape(n_side * H, n_side * W, C)
    if C == 1:
        grid = grid.squeeze(-1)
    Image.fromarray(grid).save(f_name)

def sample_and_save(model, scale, n_classes, k_keep, energy_mean, energy_std, f_name, n_side=8, n_steps=1):
    reps = n_side * n_side // n_classes + 1
    labels = mx.array(np.tile(np.arange(n_classes), reps)[:n_side * n_side].astype(np.int32))
    order, values = model.sample(labels, k_keep, energy_mean, energy_std, n_steps=n_steps)
    imgs = reconstruct(order, values, scale, model.H, model.W, model.C)
    save_grid(imgs, f_name, n_side)
    print(f'saved {f_name}')

def train(dataset_name='mnist', k_keep=None, n_epoch=20, batch_size=128, lr=0.0003, ctx_dim=128, dist_hidden=512, dist_layer=3, denoise_hidden=256, denoise_layer=2, n_freqs=6, n_scale_samples=4000, select_noise=0.5, n_sample_steps=1, warmup_frac=0.05, grad_clip=1.0, weight_decay=0.0, debug_every=50, mlp_variant='gated', postfix=''):
    images, labels = load_data(dataset_name)
    H, W, C = images.shape[1:]
    n_classes = int(labels.max()) + 1
    if k_keep is None:
        k_keep = H * W // 4
    rng = np.random.default_rng(0)
    idx = rng.choice(len(images), size=min(n_scale_samples, len(images)), replace=False)
    scale = fit_scale(images[idx])
    channel_weight = DEFAULT_CHANNEL_WEIGHT.get(C, np.ones(C, dtype=np.float32))
    energy_mean, energy_std = fit_energy_norm(images[idx], scale, channel_weight)
    model = SparseFlow(H, W, C, n_classes, ctx_dim=ctx_dim, dist_hidden=dist_hidden, dist_layer=dist_layer, denoise_hidden=denoise_hidden, denoise_layer=denoise_layer, n_freqs=n_freqs, channel_weight=channel_weight, mlp_variant=mlp_variant)
    mx.eval(model)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(images) // batch_size)
    total_steps = steps_per_epoch * n_epoch
    warmup_steps = max(50, int(total_steps * warmup_frac))

    def loss_fn(model, x0, x1, t, labels, den_pos, den_val):
        flow_loss, value_loss = model.loss(x0, x1, t, labels, den_pos, den_val)
        return flow_loss + value_loss
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    f_name = f'{dataset_name}_sparseflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}{postfix}'
    print(f'{f_name}  H={H} W={W} C={C}  n_classes={n_classes}  k_keep={k_keep}/{H * W}  n_sample_steps={n_sample_steps}  warmup_steps={warmup_steps}  grad_clip={grad_clip}  mlp_variant={mlp_variant}')
    step = 0
    for e in range(n_epoch):
        perm = rng.permutation(len(images))
        tic = time.perf_counter()
        total_f, total_v, n_batch = (0.0, 0.0, 0)
        for i in range(0, len(images) - batch_size + 1, batch_size):
            step += 1
            optimizer.learning_rate = lr * min(1.0, step / warmup_steps)
            b = perm[i:i + batch_size]
            batch_imgs = images[b]
            x1, log_e = energy_target(batch_imgs, scale, channel_weight, energy_mean, energy_std)
            x0 = rng.standard_normal(x1.shape).astype(np.float32)
            t = rng.uniform(0, 1, size=len(b)).astype(np.float32)
            den_pos = gumbel_topk(log_e, k_keep, select_noise, rng)
            whitened = to_whitened(batch_imgs, scale)
            den_val = np.take_along_axis(whitened.reshape(len(b), H * W, C), den_pos[..., None], axis=1).astype(np.float32)
            args = (mx.array(x0), mx.array(x1), mx.array(t), mx.array(labels[b]), mx.array(den_pos), mx.array(den_val))
            loss, grads = loss_and_grad_fn(model, *args)
            grads, _ = optim.clip_grad_norm(grads, max_norm=grad_clip)
            optimizer.update(model, grads)
            f_loss, v_loss = model.loss(*args)
            mx.eval(model, optimizer, f_loss, v_loss)
            cur_lr = optimizer.learning_rate
            cur_lr = cur_lr.item() if hasattr(cur_lr, 'item') else float(cur_lr)
            fv, vv = (f_loss.item(), v_loss.item())
            if not (np.isfinite(fv) and np.isfinite(vv)) or fv > 10000.0 or vv > 10000.0:
                ds = debug_stats(model, *args[:4], args[4])
                print(f'\ntraining diverged at epoch {e}, step {step} (flow={fv}, value={vv})')
                print_debug_stats('  [diverged]', ds, cur_lr)
                print('-- stopping early rather than burning more epochs on a wrecked model.')
                return
            if step % debug_every == 0:
                ds = debug_stats(model, *args[:4], args[4])
                print_debug_stats(f'  [step {step:6d}]', ds, cur_lr)
                print(f'    flow={fv:.4f}  value={vv:.4f}')
            total_f += fv
            total_v += vv
            n_batch += 1
        print(f'flow={total_f / n_batch:.4f}  value={total_v / n_batch:.4f}  @ epoch {e} in {time.perf_counter() - tic:.2f}s')
        if (e + 1) % max(1, n_epoch // 5) == 0:
            sample_and_save(model, scale, n_classes, k_keep, energy_mean, energy_std, f'{f_name}_e{e}.png', n_steps=n_sample_steps)
    mx.save_safetensors(f'{f_name}.safetensors', dict(tree_flatten(model.trainable_parameters())))
    np.save(f'{f_name}_scale.npy', scale)
    np.save(f'{f_name}_energy_mean.npy', energy_mean)
    np.save(f'{f_name}_energy_std.npy', energy_std)
    print(f'done: {f_name}.safetensors')

if __name__ == '__main__':
    fire.Fire(train)
