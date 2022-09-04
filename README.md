---
language:
- en
tags:
- stable-diffusion
- text-to-image
license: bigscience-bloom-rail-1.0
inference: false

---

# waifu-diffusion - Diffusion for Weebs

waifu-diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning on high quality anime images.

## Model Description

The model used for fine-tuning is [Stable Diffusion V1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), which is a latent text-to-image diffusion model trained on [LAION2B-en](https://huggingface.co/datasets/laion/laion2B-en).

The current model is fine-tuned from 56 thousand images from Danbooru selected with an aesthetic score greater than `6.0`.

With [Textual Inversion](https://github.com/rinongal/textual_inversion), the embeddings for the text encoder has been trained to align more with anime-styled images, reducing excessive prompting.

## Training Data & Annotative Prompting

The data used for fine-tuning has come from a random sample of 56k Danbooru images, which were filtered based on [CLIP Aesthetic Scoring](https://github.com/christophschuhmann/improved-aesthetic-predictor) where only images with an aesthetic score greater than `6.0` were used.

## Downstream Uses

This model can be used for entertainment purposes and as a generative art assistant. The EMA model can be used for additional fine-tuning.

## Example Code

```
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "hakurei/waifu-diffusion"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = "a photo of reimu hakurei. anime style"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save("reimu_hakurei.png")
```

## Team Members and Acknowledgements

This project would not have been possible without the incredible work by the [CompVis Researchers](https://ommer-lab.com/).

- [Anthony Mercurio](https://github.com/harubaru)
- [Salt](https://github.com/sALTaccount/)