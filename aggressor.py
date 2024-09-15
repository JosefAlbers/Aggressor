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

n_epoch = 30
n_steps = 1000

def get_dataset_info(dataset_name):
    dataset = load_dataset(dataset_name, split='train')
    sample = dataset[0]['image'] if 'image' in dataset[0] else dataset[0]['img']
    sample = np.array(sample)
    if sample.ndim < 3:
        sample = sample[:, :, None]
    image_shape = sample.shape
    patch_size = (image_shape[0] // 2, image_shape[1] // 2)
    patch_dim = patch_size[0] * patch_size[1] * image_shape[2]
    return dataset, image_shape, patch_size, patch_dim
        
class MLP(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.gate_up_proj = nn.Linear(dim, 2*dim, bias=False)
        self.down_proj = nn.Linear(dim, out_dim, bias=False)
    def __call__(self, x):
        gate, x = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['dim']
        self.n_heads = n_heads = config['n_heads']
        self.scale = (dim // n_heads)**-0.5
        self.qkv_proj = nn.Linear(dim, 3*dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
    def __call__(self, x, position_ids=None, attention_mask=None, cache=None):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        if cache is None:
            mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
            if attention_mask is not None:
                mask += mx.where(attention_mask[:, :, None]*attention_mask[:, None, :]==1, 0, -mx.inf)
                mask = mx.expand_dims(mask, 1)
            else:
                mask = mask[None, None]
        else:
            past_k, past_v, past_m = cache
            mask = mx.pad(past_m[:,:,-1:,:], ((0,0),(0,0),(0,0),(0,1)))
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        cache = (k, v, mask)
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        o = self.o_proj(o)
        return o, cache

class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['dim']
        self.self_attn = Attention(config)
        self.mlp = MLP(dim)
        self.input_layernorm = nn.RMSNorm(dim)
        self.post_attention_layernorm = nn.RMSNorm(dim)
    def __call__(self, x, position_ids=None, attention_mask=None, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), position_ids, attention_mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['dim']
        self.layers = [Layer(config) for _ in range(4)]
        self.norm = nn.RMSNorm(dim)
        self.o_proj = nn.Linear(dim, dim, bias=False)
    def __call__(self, x, position_ids=None, attention_mask=None, cache=None):
        cache = [None]*len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, position_ids=position_ids, attention_mask=attention_mask, cache=cache[i])
        x = self.o_proj(self.norm(x))
        return x, cache

class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['dim']
        self.layers = [MLP(2*dim, dim) for _ in range(10)]
        self.te = nn.Sequential(
            nn.SinusoidalPositionalEncoding(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU()
        )
        self.norm = nn.RMSNorm(dim)
    def __call__(self, x, t, c):
        t = self.te(t)[:,None,:]
        for layer in self.layers:
            r = x
            x = self.norm(layer(mx.concatenate([x, c], axis=-1))) * t
            x = x + r
        return x

class Scheduler(nn.Module):
    def __init__(self, min_beta=0.0001, max_beta=0.02):
        super().__init__()
        self._betas = mx.linspace(min_beta**0.5, max_beta**0.5, n_steps) ** 2
        self._alphas = 1 - self._betas
        self._alpha_cumprods = mx.cumprod(self._alphas, axis=0)
    def forward(self, x_0, t, eps):
        alpha_bar = self._alpha_cumprods[t][:,None,None]
        res = mx.sqrt(alpha_bar) * x_0 + mx.sqrt(1 - alpha_bar) * eps
        return res
    def backward(self, model, x_t, t):
        eps_t = model(x_t, mx.array([t] * x_t.shape[0]))
        mu_t = (x_t - (1 - self._alphas[t]) / mx.sqrt(1 - self._alpha_cumprods[t]) * eps_t) / mx.sqrt(self._alphas[t])
        if t == 0:
            return mu_t
        beta_t =  (1 - self._alpha_cumprods[t - 1]) / (1 - self._alpha_cumprods[t]) * self._betas[t]
        noise_t = mx.sqrt(beta_t) * mx.random.normal(x_t.shape)
        return mu_t + noise_t

class Aggressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['dim']
        self.patch_size = config['patch_size']
        self.image_shape = config['image_shape']
        self.transformer = Transformer(config)
        self.diffusion = Denoiser(config)
        self.scheduler = Scheduler()
        self.start_token = mx.zeros(self.dim)
        self.se = nn.Linear(self.dim, self.dim)
        self.pe = nn.Embedding(self.dim, self.dim)
    def __call__(self, seq):
        seq = rearrange(seq, 'b (h p1) (w p2) c -> b (h w) (c p1 p2)', p1=self.patch_size[0], p2=self.patch_size[1])
        B, S, _ = seq.shape
        cond_seq = seq[:, :-1]
        cond_seq = self.se(cond_seq)
        cond_seq = mx.concatenate([mx.repeat(self.start_token[None], B, 0)[:,None,:], cond_seq], axis=1)
        cond_seq = cond_seq + self.pe(mx.arange(S))
        cond, _ = self.transformer(cond_seq)
        t = mx.random.randint(0, n_steps, (B,))
        eps = mx.random.normal(seq.shape)
        x_t = self.scheduler.forward(seq, t, eps)
        eps_theta = self.diffusion(x_t, t, cond)
        return nn.losses.mse_loss(eps_theta, eps)
    def sample(self, batch_size):
        num_patches = (self.image_shape[0] // self.patch_size[0]) * (self.image_shape[1] // self.patch_size[1])
        start_token = mx.repeat(self.start_token[None], batch_size, 0)[:,None,:]
        generated = mx.zeros((batch_size, 0, self.dim))
        for _ in range(num_patches):
            cond_seq = generated
            cond_seq = self.se(cond_seq)
            cond_seq = mx.concatenate([start_token, cond_seq], axis=1)
            cond_seq = cond_seq + self.pe(mx.arange(cond_seq.shape[1]))
            cond, _ = self.transformer(cond_seq)
            x = mx.random.normal((batch_size, 1, self.dim))
            for t in range(n_steps - 1, -1, -1):
                t_batch = mx.array([t] * batch_size)
                eps_theta = self.diffusion(x, t_batch, cond[:, -1:])
                x = self.scheduler.backward(lambda x, t: self.diffusion(x, t, cond[:, -1:]), x, t)
            generated = mx.concatenate([generated, x], axis=1)
        generated = rearrange(generated, 'b (h w) (c p1 p2) -> b (h p1) (w p2) c', 
                              h=self.image_shape[0]//self.patch_size[0], w=self.image_shape[1]//self.patch_size[1], 
                              p1=self.patch_size[0], p2=self.patch_size[1], c=self.image_shape[2])
        return generated

def sample(model, n_sample_per_side=4):
    model.eval()
    tic = time.perf_counter()
    x = model.sample(batch_size=n_sample_per_side**2)
    x = ((mx.clip(x, -1, 1) + 1) / 2 * 255).reshape(n_sample_per_side, n_sample_per_side, *model.image_shape).astype(mx.uint8)
    x = rearrange(x, 'h w ph pw c -> (h ph) (w pw) c')
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    Image.fromarray(np.array(x)).save(f'aggressor_{dataset_name}.png')
    print(f'Generated {n_sample_per_side**2} images in {time.perf_counter() - tic:.2f} seconds')
    
def train(model, dataset, batch_size=512, lr=3e-4):
    def get_batch(dataset):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_img = np.array(batch['image' if 'image' in batch else 'img'])
            if batch_img.ndim < 4:
                batch_img = batch_img[:, :, :, None]
            batch_img = (((batch_img / 255) - 0.5) * 2)
            yield mx.array(batch_img, dtype=mx.float32)
    def loss_fn(model, x):
        return model(x)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    _n_steps = math.ceil(n_epoch * len(dataset) / batch_size)
    _n_warmup = _n_steps // 6
    _warmup = optim.linear_schedule(1e-6, lr, steps=_n_warmup)
    _cosine = optim.cosine_decay(lr, _n_steps-_n_warmup, 1e-5)
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
            mx.eval(loss, model, optimizer)
            total_loss += loss.item() * x.shape[0]
            total_step += x.shape[0]
        print(f'{total_loss / total_step:.4f} @ {e} in {(time.perf_counter() - tic):.2f}')
        if (e+1) % 10 == 0:
            mx.save_safetensors(f'aggressor_{dataset_name}.safetensors', dict(tree_flatten(model.trainable_parameters())))
            sample(model)

if __name__ == '__main__':
    dataset_name = 'mnist' # or 'cifar10'
    dataset, image_shape, patch_size, patch_dim = get_dataset_info(dataset_name)
    config = dict(dim=patch_dim, n_heads=4, patch_size=patch_size, image_shape=image_shape)
    model = Aggressor(config)
    train(model, dataset)
    # model.load_weights(f'aggressor_{dataset_name}.safetensors')
    # sample(model)