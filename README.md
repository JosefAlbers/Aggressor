# Aggressor: Ultra-minimal autoregressive diffusion model for image generation

A simplest possible implementation of [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838).

## Key Features

- **Simple Architecture**: A tiny transformer for autoregression and an MLP for diffusion.
- **Single-File Implementation**: Entire model in one Python file.
- **Minimal Dependencies**: Built from scratch using only basic MLX operations.

## Components

- `Aggressor`: Main model class combining transformer and diffusion.
- `Transformer`: Multi-layer transformer with attention and MLP blocks.
- `Denoiser`: MLP-based diffusion process with time embedding.
- `Scheduler`: Handles forward and backward processes for diffusion.

## Usage

```zsh
python aggressor.py
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Aggressor/main/assets/aggressor_cifar.png)

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Aggressor/main/assets/aggressor_mnist.png)

*(Training on 60000 images x 20 epochs takes approximately 7~8 minutes on 8GB M2 MacBook.)*

## Acknowledgements

Thanks to [lucidrains](https://github.com/lucidrains/autoregressive-diffusion-pytorch)' fantastic code that inspired this project. The official implementation is available [here](https://github.com/LTH14/mar).
