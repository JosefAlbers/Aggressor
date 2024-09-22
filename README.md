# Aggressor: Ultra-minimal autoregressive diffusion model for image and speech generation

<table>
<tr>
<td align="center">

**CIFAR**

</td>
<td align="center">

**MNIST**

</td>
<td align="center">

**AUDIO**

</td>
</tr>
<tr>
<td align="center">

![cifar](https://raw.githubusercontent.com/JosefAlbers/Aggressor/main/assets/aggressor_cifar.png)

</td>
<td align="center">

![mnist](https://raw.githubusercontent.com/JosefAlbers/Aggressor/main/assets/aggressor_mnist.png)

</td>
<td align="center">

[audio](https://github.com/user-attachments/assets/89574625-2ec0-4aeb-884f-bf03e5a4aab0)

</td>
</tr>
</table>

A simplest possible implementation of [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838).

## Key Features

- **Simple Architecture**: A tiny transformer for autoregression and an MLP for diffusion.
- **Minimal Dependencies**: Built from scratch using only basic MLX operations.
- **Single-File Implementation**: Entire model in one Python file `aggressor.py`.

## Components

- `Aggressor`: Main model class combining transformer and diffusion.
- `Transformer`: Multi-layer transformer with attention and MLP blocks.
- `Denoiser`: MLP-based diffusion process with time embedding.
- `Scheduler`: Handles forward and backward processes for diffusion.

## Usage

```zsh
python aggressor.py
```

*(Training on 60000 images x 20 epochs takes approximately 7~8 minutes on 8GB M2 MacBook.)*

## Acknowledgements

Thanks to [lucidrains](https://github.com/lucidrains/autoregressive-diffusion-pytorch)' fantastic code that inspired this project. The official implementation is available [here](https://github.com/LTH14/mar).

