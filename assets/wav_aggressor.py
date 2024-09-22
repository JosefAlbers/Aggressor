import math
import os
import time
from datetime import datetime

import fire
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset, load_from_disk
from einops.array_api import rearrange
from mlx.utils import tree_flatten
from PIL import Image
from scipy.fftpack import idct
from scipy.io import wavfile
import soundfile as sf

EPS = 1e-5

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
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head=n_head
        self.scale = (dim // n_head)**-0.5
        self.qkv_proj = nn.Linear(dim, 3*dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
    def __call__(self, x, position_ids=None, attention_mask=None, cache=None):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
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
    def __init__(self, dim, n_head):
        super().__init__()
        self.self_attn = Attention(dim, n_head)
        self.mlp = MLP(dim)
        self.input_layernorm = nn.RMSNorm(dim, eps=EPS)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=EPS)
    def __call__(self, x, position_ids=None, attention_mask=None, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), position_ids, attention_mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class Transformer(nn.Module):
    def __init__(self, dim, n_head, n_layer):
        super().__init__()
        self.layers = [Layer(dim, n_head) for _ in range(n_layer)]
        self.norm = nn.RMSNorm(dim, eps=EPS)
        self.o_proj = nn.Linear(dim, dim-2, bias=False)
    def __call__(self, x, position_ids=None, attention_mask=None, cache=None):
        cache = [None]*len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, position_ids=position_ids, attention_mask=attention_mask, cache=cache[i])
        x = self.o_proj(self.norm(x))
        return x, cache

class Denoiser(nn.Module):
    def __init__(self, dim, n_layer):
        super().__init__()
        self.layers = [MLP(2*dim, dim) for _ in range(n_layer)]
        self.te = nn.Sequential(
            nn.SinusoidalPositionalEncoding(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU()
        )
        self.norm = nn.RMSNorm(dim, eps=EPS)
    def __call__(self, x, t, c):
        t = self.te(t)[:,None,:]
        for layer in self.layers:
            r = x
            x = self.norm(layer(mx.concatenate([x, c], axis=-1))) * t
            x = x + r
        return x

class Scheduler(nn.Module):
    def __init__(self, min_beta=0.0001, max_beta=0.02, n_diff=1000):
        super().__init__()
        self._betas = mx.linspace(min_beta, max_beta, n_diff)
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

def decompress_audio(sample, f_name):
    compressed, segment_size, sample_rate = sample['compressed_audio'], sample['segment_size'], sample['sample_rate']
    decompressed_segments = []
    for segment in compressed:
        full_segment = np.zeros(segment_size)
        full_segment[:len(segment)] = segment
        decompressed_segments.append(idct(full_segment, norm='ortho'))
    decompressed_audio = np.concatenate(decompressed_segments)
    if np.max(np.abs(decompressed_audio)) > 0:
        decompressed_audio = decompressed_audio / np.max(np.abs(decompressed_audio)) * 32767
    decompressed_audio = np.int16(decompressed_audio)
    # wavfile.write(f'{f_name}.wav', sample['sample_rate'], decompressed_audio)
    sf.write(f'{f_name}.mp3', decompressed_audio, sample['sample_rate'], format='mp3')

class Aggressor(nn.Module):
    def __init__(self, image_shape, n_head, n_diff, n_loop, n_layer, segment_size, sample_rate):
        super().__init__()
        self.image_shape = image_shape
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.dim = dim = image_shape[-1]
        self.n_diff = n_diff
        self.transformer = Transformer(dim=dim+2, n_head=n_head, n_layer=n_layer)
        self.diffusion = Denoiser(dim=dim, n_layer=n_layer)
        self.scheduler = Scheduler(n_diff=n_diff)
        self.start_token = mx.zeros(dim)[None, None]
        self.n_loop = n_loop
        self._pe = mx.array(np.indices((1, image_shape[0]))).reshape(2, -1).T
        self._scale_factors = self.create_scale_factors(dim)
    def create_scale_factors(self, num_coeffs, min_scale=1e4, max_scale=1e6):
        x = np.linspace(0, 1, num_coeffs)
        factors = 1 / (1 + np.exp((x - 0.5) * 10))
        factors = min_scale + (max_scale - min_scale) * factors
        return mx.array(factors)
    def __call__(self, seq):
        seq = mx.arcsinh(seq * self._scale_factors[None, None, :])
        B, S, _ = seq.shape
        cond_seq = seq[:, :-1]
        cond_seq = mx.concatenate([mx.repeat(self.start_token, B, 0), cond_seq], axis=1)
        cond_seq = mx.concatenate([cond_seq, mx.repeat(self._pe[None,:,:], B, 0)], axis = -1)
        cond, _ = self.transformer(cond_seq)
        sum_loss = 0
        step = 0
        for _ in range(self.n_loop):
            t = mx.random.randint(0, self.n_diff, (B,))
            eps = mx.random.normal(seq.shape)
            x_t = self.scheduler.forward(seq, t, eps)
            eps_theta = self.diffusion(x_t, t, cond)
            loss = mx.sum((eps - eps_theta) ** 2)
            if mx.isnan(loss):
                print(loss.item())
                continue
            sum_loss += loss
            step += eps.size
        avg_loss = sum_loss / step
        return avg_loss
    def sample(self, batch_size):
        generated = mx.zeros((batch_size, 0, self.dim))
        cond_seq = mx.repeat(self.start_token, batch_size, 0)
        cache = None
        for p in range(self.image_shape[0]):
            cond_seq = mx.concatenate([cond_seq, mx.repeat(self._pe[p][None,None,:], batch_size, 0)], axis = -1)
            cond, cache = self.transformer(cond_seq, cache=cache)
            x = mx.random.normal((batch_size, 1, self.dim))
            for t in range(self.n_diff - 1, -1, -1):
                t_batch = mx.array([t] * batch_size)
                eps_theta = self.diffusion(x, t_batch, cond[:, -1:])
                x = self.scheduler.backward(lambda x, t: self.diffusion(x, t, cond[:, -1:]), x, t)
            generated = mx.concatenate([generated, x], axis=1)
            cond_seq = x
            mx.eval(cache, cond_seq, generated)
        generated = mx.sinh(generated) / self._scale_factors[None, None, :]
        return np.array(generated)

def sample(model, f_name='aggressor', n_samples=1):
    model.eval()
    mx.eval(model)
    tic = time.perf_counter()
    X = model.sample(batch_size=n_samples)
    for i, x in enumerate(X):
        sample = dict(compressed_audio=x, segment_size=model.segment_size, sample_rate=model.sample_rate)
        decompress_audio(sample, f'{f_name}_{i}')
    print(f'Saved {n_samples} samples to {f_name} ({time.perf_counter() - tic:.2f} sec)')

def train(model, dataset, n_epoch, batch_size, lr, postfix):
    def get_batch(dataset):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_img = np.array(batch['compressed_audio'], dtype=np.float32)
            yield mx.array(batch_img, dtype=mx.float32)
    def evaluate(model, dataset):
        model.eval()
        loss = 0
        step = 0
        for x in get_batch(dataset):
            loss += model(x).item()
            step += 1
        return loss / step
    def loss_fn(model, x):
        return model(x)
    f_name = f'{dataset.info.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{n_epoch}{postfix}'
    print(f'{f_name} {model.image_shape} {model.dim}')
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    _n_steps = math.ceil(n_epoch * len(dataset) / batch_size)
    _n_warmup = _n_steps//5
    _warmup = optim.linear_schedule(1e-6, lr, steps=_n_warmup)
    _cosine = optim.cosine_decay(lr, _n_steps-_n_warmup, 1e-5)
    optimizer = optim.Lion(learning_rate=optim.join_schedules([_warmup, _cosine], [10]))
    model.train()
    mx.eval(model, optimizer)
    best_avg_loss = mx.inf
    best_eval_loss = mx.inf
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
        _avg_loss = total_loss/total_step
        print(f'{_avg_loss:.4f} @ {e} in {(time.perf_counter() - tic):.2f}')
        if e > n_epoch//5 and _avg_loss < best_avg_loss:
            _eval_loss = evaluate(model, dataset)
            print(f'- {_eval_loss:.4f}')
            if _eval_loss < best_eval_loss:
                print('- Saved weights')
                mx.save_safetensors(f'{f_name}.safetensors', dict(tree_flatten(model.trainable_parameters())))
                best_eval_loss = _eval_loss
                best_avg_loss = _avg_loss
        if (e+1) % (n_epoch//5) == 0:
            sample(model=model, f_name=f_name)
    model.load_weights(f'{f_name}.safetensors')
    sample(model=model, f_name=f_name, n_samples=4)

def get_audio_dataset(path_ds, batch_size):
    dataset = load_dataset(path_ds, split='train')
    _take = (len(dataset) // batch_size) * batch_size
    dataset = dataset.take(_take)
    print(f'{len(dataset)=}')
    return dataset, np.array(dataset[0]['compressed_audio']).shape, dataset[0]['segment_size'], dataset[0]['sample_rate']

def main(path_ds='JosefAlbers/fluent_speech_commands_female', n_head=1, n_diff=1000, n_epoch=200, batch_size=1, lr=3e-4, n_loop=4, n_layer=16, postfix=''):
    dataset, image_shape, segment_size, sample_rate = get_audio_dataset(path_ds=path_ds, batch_size=batch_size)
    model = Aggressor(image_shape=image_shape, n_head=n_head, n_diff=n_diff, n_loop=n_loop, n_layer=n_layer, segment_size=segment_size, sample_rate=sample_rate)
    train(model=model, dataset=dataset, n_epoch=n_epoch, batch_size=batch_size, lr=lr, postfix=postfix)

if __name__ == '__main__':
    fire.Fire(main)