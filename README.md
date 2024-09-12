# Aggressor: Ultra-minimal autoregressive diffusion model for image generation

A simplest possible implementation of [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838).

## Key Features

- **Basic Architecture**: Single attention block for transformer and a tiny MLP for diffusion.
- **Single-File Implementation**: Entire model in one Python file.
- **Minimal Dependencies**: Core components built from scratch using basic MLX operations.

## Components

- `Aggressor`: Main model class combining transformer and diffusion.
- `Attention`: Single block with basic Rotary Position Embedding (RoPE).
- `Denoiser`: Small MLP-based diffusion process.
- `Scheduler`: Handles sampling processes.

## Usage

```zsh
python aggressor.py
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Aggressor/main/assets/aggressor.png)

*(Training on 60000 images x 50 epochs takes approximately 15~16 minutes on 8GB M1 MacBook.)*

## Acknowledgements

Thanks to [lucidrains](https://github.com/lucidrains/autoregressive-diffusion-pytorch)' fantastic code that inspired this project. The official implementation is available [here](https://github.com/LTH14/mar).
