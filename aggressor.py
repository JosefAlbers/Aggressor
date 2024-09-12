import math
import os
import time

import numpy as np
from datasets import load_dataset
from einops.array_api import rearrange
from PIL import Image

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

n_epoch = 50
n_steps = 1000
mnist_shape = (28, 28)
patch_size = 14
patch_dim = patch_size**2

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['dim']
        self.gate_up_proj = nn.Linear(dim, 2*dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, affine=False)

    def __call__(self, x):
        gate, x = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.norm(self.down_proj(nn.silu(gate) * x))
        
class Denoiser(nn.Module):
    def __init__(self, config=dict(dim=patch_dim)):
        super().__init__()
        dim = config['dim']
        self.layers = [(MLP(config), nn.Linear(dim, dim, bias=False)) for _ in range(30)]
        self.te = nn.Sequential(
            nn.SinusoidalPositionalEncoding(dim),
            nn.Linear(dim, dim, bias=False)
        )
    def __call__(self, x, t, c):
        t = self.te(t)[:,None,:]
        for layer_x, layer_c in self.layers:
            r = x
            x = layer_x(x + layer_c(c)) * t
            x = x + r
        return x

class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['dim'] // config['n_heads']
        self._inv_freq = 1.0 / (10000.0**(mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    def __call__(self, q, k, position_ids):
        cos, sin = self._get_cos_sin(position_ids)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k
    def _get_cos_sin(self, position_ids):
        position_ids_expanded = position_ids[:, None, :]
        inv_freq_expanded = mx.repeat(self._inv_freq[None, :, None], position_ids.shape[0], axis=0)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb), axis=1)
        sin = mx.expand_dims(mx.sin(emb), axis=1)
        return cos, sin
    def _rotate_half(self, x):
        midpoint = x.shape[-1] // 2
        x1, x2 = x[..., :midpoint], x[..., midpoint:]
        return mx.concatenate([-x2, x1], axis=-1)

class Attention(nn.Module):
    def __init__(self, config=dict(dim=patch_dim, n_heads=2)):
        super().__init__()
        dim = config['dim']
        self.n_heads = n_heads = config['n_heads']
        self.scale = (dim // n_heads)**-0.5
        self.qkv_proj = nn.Linear(dim, 3*dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(config)
    def __call__(self, x, position_ids=None, attention_mask=None, cache=None):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        if cache is None:
            position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] if position_ids is None else position_ids
            q, k = self.rope(q,k,position_ids)
            mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
            if attention_mask is not None:
                mask += mx.where(attention_mask[:, :, None]*attention_mask[:, None, :]==1, 0, -mx.inf)
                mask = mx.expand_dims(mask, 1)
            else:
                mask = mask[None, None]
        else:
            past_k, past_v, past_p, past_m = cache
            position_ids = past_p[:,-1:]+1
            mask = mx.pad(past_m[:,:,-1:,:], ((0,0),(0,0),(0,0),(0,1)))
            q, k = self.rope(q, k, position_ids)
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        cache = (k, v, position_ids, mask)
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(x.dtype), cache

class Scheduler():
    def __init__(self, min_beta=0.0001, max_beta=0.02):
        self.n_steps = n_steps
        self.betas = mx.linspace(min_beta, max_beta, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_cumprods = mx.cumprod(self.alphas, axis=0)
    def forward(self, x_0, t, eps):
        alpha_bar = self.alpha_cumprods[t][:,None,None]
        res = mx.sqrt(alpha_bar) * x_0 + mx.sqrt(1 - alpha_bar) * eps
        return res
    def backward(self, model, x_t, t):
        eps_t = model(x_t, mx.array([t] * x_t.shape[0]))
        mu_t = (x_t - (1 - self.alphas[t]) / mx.sqrt(1 - self.alpha_cumprods[t]) * eps_t) / mx.sqrt(self.alphas[t])
        if t == 0:
            return mu_t
        beta_t =  (1 - self.alpha_cumprods[t - 1]) / (1 - self.alpha_cumprods[t]) * self.betas[t]
        noise_t = mx.sqrt(beta_t) * mx.random.normal(x_t.shape)
        return mu_t + noise_t

class Aggressor(nn.Module):
    def __init__(self, config=dict(dim=patch_dim)):
        super().__init__()
        self.dim = config['dim']
        self.transformer = Attention()
        self.diffusion = Denoiser()
        self.scheduler = Scheduler()
        self.start_token = mx.zeros(self.dim)
    def __call__(self, seq):
        seq = rearrange(seq, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_size, p2=patch_size)
        B, seq_len, dim = seq.shape
        cond_seq = mx.concatenate([mx.repeat(self.start_token[None], B, 0)[:,None,:], seq[:, :-1]], axis=1)
        cond, _ = self.transformer(cond_seq)
        t = mx.random.randint(0, n_steps, (B,))
        eps = mx.random.normal(seq.shape)
        x_t = self.scheduler.forward(seq, t, eps)
        eps_theta = self.diffusion(x_t, t, cond)
        return nn.losses.mse_loss(eps_theta, eps)
    def sample(self, batch_size, image_size=(28, 28)):
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        generated = mx.zeros((batch_size, 0, self.dim))
        for _ in range(num_patches):
            cond_seq = mx.concatenate([mx.repeat(self.start_token[None], batch_size, 0)[:,None,:], generated], axis=1)
            cond, _ = self.transformer(cond_seq)
            x = mx.random.normal((batch_size, 1, self.dim))
            for t in range(n_steps - 1, -1, -1):
                t_batch = mx.array([t] * batch_size)
                eps_theta = self.diffusion(x, t_batch, cond[:, -1:])
                x = self.scheduler.backward(lambda x, t: self.diffusion(x, t, cond[:, -1:]), x, t)
            generated = mx.concatenate([generated, x], axis=1)
        generated = rearrange(generated, 'b (h w) (p1 p2) -> b (h p1) (w p2)', 
                              h=image_size[0]//patch_size, w=image_size[1]//patch_size, 
                              p1=patch_size, p2=patch_size)
        return generated

def infer(model, n_sample_per_side=10):
    model = model.eval()
    samples = model.sample(batch_size=n_sample_per_side**2, image_size=mnist_shape)
    samples = ((samples + 1) / 2 * 255).astype(mx.uint8)
    samples = samples.reshape(n_sample_per_side, n_sample_per_side, *mnist_shape)
    image_grid = np.concatenate([np.concatenate(row, axis=1) for row in samples], axis=0)
    Image.fromarray(image_grid).save('aggressor.png')
    
def train(model, batch_size=512):
    def get_batch(dataset):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_img = (((np.array(batch['image']) / 255) - 0.5) * 2)
            yield mx.array(batch_img, dtype=mx.float32)
    def loss_fn(model, x):
        return model(x)
    dataset = load_dataset("mnist", split='train')
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    _n_steps = math.ceil(n_epoch * len(dataset) / batch_size)
    _warmup = optim.linear_schedule(1e-6, 1e-3, steps=10)
    _cosine = optim.cosine_decay(1e-3, _n_steps-10, 1e-5)
    optimizer = optim.Lion(learning_rate=optim.join_schedules([_warmup, _cosine], [10]))
    mx.eval(model, optimizer)
    for e in range(n_epoch):
        dataset = dataset.shuffle()
        total_loss = 0
        total_step = 0
        tic = time.perf_counter()
        for x in get_batch(dataset):
            model.train()
            loss, grads = loss_and_grad_fn(model, x)
            optimizer.update(model, grads)
            mx.eval(model, optimizer)
            total_loss += loss.item() * x.shape[0]
            total_step += x.shape[0]
        print(f'{total_loss / total_step:.4f} @ {e} in {(time.perf_counter() - tic):.2f}')
    mx.save_safetensors(f'aggressor.safetensors', dict(tree_flatten(model.trainable_parameters())))

model = Aggressor()
train(model)
# model.load_weights(f'aggressor.safetensors')
infer(model)