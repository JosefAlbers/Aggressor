import math
import os
import time
from datetime import datetime

import fire
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset
from einops.array_api import rearrange
from mlx.utils import tree_flatten
from PIL import Image
from scipy.fftpack import dctn, idctn

EPS = 1e-5

def get_dataset_info(dataset_name, batch_size, label=5):
    dataset = load_dataset(dataset_name, split='train')
    if label is not None and label in dataset['label']:
        dataset = dataset.filter(lambda x: x['label'] == label)
    _take = (len(dataset) // batch_size) * batch_size
    dataset = dataset.take(_take)
    sample = np.array(dataset[0]['image'] if 'image' in dataset[0] else dataset[0]['img'])
    if sample.ndim < 3:
        sample = sample[:, :, None]
    return dataset, sample.shape

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

def get_decay(image_shape, gamma=0.999):
    i, j = np.ogrid[:image_shape[0], :image_shape[1]]
    gamma = gamma**(i + j)
    gamma = np.repeat(gamma[...,None], image_shape[-1], -1)
    return mx.array(gamma, dtype=mx.float32)

class Aggressor(nn.Module):
    def __init__(self, image_shape, n_chop, n_head, n_diff, n_loop, n_layer, use_dct):
        super().__init__()
        self.image_shape = image_shape
        self.n_chop = n_chop
        self.patch_size = patch_size = (image_shape[0] // n_chop, image_shape[1] // n_chop)
        self.dim = dim = patch_size[0] * patch_size[1] * image_shape[-1]
        self.n_diff = n_diff
        self.transformer = Transformer(dim=dim+2, n_head=n_head, n_layer=n_layer)
        self.diffusion = Denoiser(dim=dim, n_layer=n_layer)
        self.scheduler = Scheduler(n_diff=n_diff)
        self.start_token = mx.zeros(dim)[None, None]
        self.n_loop = n_loop
        self._pe = mx.array(np.indices((n_chop, n_chop))).reshape(2, -1).T
        self.use_dct = use_dct
        if use_dct:
            _decay = get_decay(image_shape)
        else:
            _decay = mx.ones(image_shape, dtype=mx.float32)
        self._decay = rearrange(_decay, f'(h ph) (w pw) c -> (h w) (ph pw c)', ph=self.patch_size[0], pw=self.patch_size[1])
    def __call__(self, seq):
        seq = rearrange(seq, f'b (h ph) (w pw) c -> b (h w) (ph pw c)', ph=self.patch_size[0], pw=self.patch_size[1])
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
            loss = mx.sum(((eps - eps_theta) ** 2) * self._decay)
            if mx.isnan(loss):
                print(loss.item())
                continue
            sum_loss += loss
            step += eps.size
        avg_loss = sum_loss / step
        return avg_loss
    def sample(self, batch_size):
        num_patches = self.n_chop**2
        generated = mx.zeros((batch_size, 0, self.dim))
        cond_seq = mx.repeat(self.start_token, batch_size, 0)
        cache = None
        for p in range(num_patches):
            cond_seq = mx.concatenate([cond_seq, mx.repeat(self._pe[p][None,None,:], batch_size, 0)], axis = -1)
            cond, cache = self.transformer(cond_seq, cache=cache)
            x = mx.random.normal((batch_size, 1, self.dim))
            for t in range(self.n_diff - 1, -1, -1):
                t_batch = mx.array([t] * batch_size)
                eps_theta = self.diffusion(x, t_batch, cond[:, -1:])
                x = self.scheduler.backward(lambda x, t: self.diffusion(x, t, cond[:, -1:]), x, t)
            generated = mx.concatenate([generated, x], axis=1)
            cond_seq = x
            mx.eval(cond_seq, generated)
        generated = rearrange(generated, f'b (h w) (ph pw c) -> b (h ph) (w pw) c',
                                h=self.n_chop, w=self.n_chop,
                                ph=self.patch_size[0], pw=self.patch_size[1], c=self.image_shape[-1])
        generated = np.array(generated)
        if self.use_dct:
            generated = idctn(generated, axes=(1, 2), norm='ortho')
            generated = np.clip(generated, 0, 1) * 255
        else:
            generated = (np.clip(generated, -1, 1) + 1) / 2 * 255
        return generated.astype(np.uint8)

def sample(model, f_name='aggressor', n_sample_per_side=4):
    model.eval()
    mx.eval(model)
    tic = time.perf_counter()
    x = model.sample(batch_size=n_sample_per_side**2)
    x = x.reshape(n_sample_per_side, n_sample_per_side, *model.image_shape)
    x = rearrange(x, 'bh bw h w c -> (bh h) (bw w) c')
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    Image.fromarray(np.array(x)).save(f'{f_name}.png')
    print(f'Saved {n_sample_per_side**2} images to {f_name}.png ({time.perf_counter() - tic:.2f} sec)')

def train(model, dataset, n_epoch, batch_size, lr, postfix):
    def get_batch(dataset):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_img = np.array(batch['image' if 'image' in batch else 'img'], dtype=np.float32)
            if batch_img.ndim < 4:
                batch_img = batch_img[:, :, :, None]
            if model.use_dct:
                batch_img = batch_img / 255.0
                batch_img = dctn(batch_img, axes=(1, 2), norm='ortho')
            else:
                batch_img = (((batch_img / 255.0) - 0.5) * 2.0)
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
    print(f'{f_name} {model.image_shape} {model.n_chop} {model.patch_size} {model.dim}')
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
            # grads, _ = optim.clip_grad_norm(grads, max_norm=0.1)
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
    sample(model=model, f_name=f_name, n_sample_per_side=10)

def main(dataset_name='mnist', label=None, n_chop=2, n_head=1, n_diff=1000, n_epoch=20, batch_size=32, lr=3e-4, n_loop=4, n_layer=4, use_dct=False, postfix=''):
    dataset, image_shape = get_dataset_info(dataset_name=dataset_name, batch_size=batch_size, label=label)
    model = Aggressor(image_shape=image_shape, n_chop=n_chop, n_head=n_head, n_diff=n_diff, n_loop=n_loop, n_layer=n_layer, use_dct=use_dct)
    train(model=model, dataset=dataset, n_epoch=n_epoch, batch_size=batch_size, lr=lr, postfix=postfix)
    # model.load_weights('cifar.safetensors')
    # sample(model=model, f_name='aggressor_cifar', n_sample_per_side=10)

if __name__ == '__main__':
    # main(dataset_name='cifar10', label=5, n_chop=8, n_epoch=200, n_layer=16)
    fire.Fire(main)
