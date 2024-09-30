import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset
from PIL import Image

class Encoder(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
        )
        self.avg = nn.Linear(512, latent)
        self.var = nn.Linear(512, latent)
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.encode(x)
        return self.avg(x), self.var(x)

class Decoder(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(latent, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),
        )
    def __call__(self, x):
        x = self.decode(x)
        return x.reshape((x.shape[0], 28, 28))

class VAE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)
    def __call__(self, x):
        avg, var = self.encoder(x)
        z = avg + mx.random.normal(avg.shape) * mx.exp(0.5 * var)
        return self.decoder(z), avg, var

def loss_fn(model, x, beta=0.1):
    xhat, avg, var = model(x)
    mse = mx.sum((x - xhat)**2)
    kl = mx.sum(mx.exp(var) + avg**2 - var - 1)
    return (mse + beta * kl) / x.shape[0]

def get_batch(dataset, batch_size=128):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch = np.array(batch['image']).astype(np.float32) / 255.0
        yield mx.array(batch)

def train(model, dataset, n_epoch=20):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Lion(learning_rate=1e-4)
    model.train()
    mx.eval(model, optimizer)
    for e in range(n_epoch):
        total_loss = 0
        for x in get_batch(dataset):
            model.train()
            loss, grads = loss_and_grad_fn(model, x)
            optimizer.update(model, grads)
            mx.eval(loss, model, optimizer)
            total_loss += loss.item()
        print(f'{total_loss:.1f}')

def sample(model, n=20, range_val=3.0):
    x = np.linspace(-range_val, range_val, n)
    y = np.linspace(-range_val, range_val, n)
    xx, yy = np.meshgrid(x, y)
    z = np.column_stack([xx.ravel(), yy.ravel()])
    model.eval()
    samples = model.decoder(mx.array(z))
    samples = np.clip(np.array(samples) * 255, 0, 255).astype(np.uint8)
    samples = samples.reshape(n, n, 28, 28)
    samples = np.block([[samples[i,j] for j in range(n)] for i in range(n)])
    Image.fromarray(samples).save(f'vae_mnist.png')

model = VAE()
dataset = load_dataset('mnist', split='train')
train(model, dataset)
sample(model)
