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

waifu-diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through [Textual Inversion](https://github.com/rinongal/textual_inversion).

<img src=https://cdn.discordapp.com/attachments/872361510133981234/1016022078635388979/unknown.png?3867929 width=40% height=40%>
<sub>Prompt: touhou 1girl komeiji_koishi portrait</sub>

## Model Description

The model originally used for fine-tuning is [Stable Diffusion V1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), which is a latent image diffusion model trained on [LAION2B-en](https://huggingface.co/datasets/laion/laion2B-en).

The current model has been fine-tuned with a learning rate of 5.0e-5 for 1 epoch on 56k Danbooru text-image pairs which all have an aesthetic rating greater than `6.0`.

## Training Data & Annotative Prompting

The data used for Textual Inversion has come from a random sample of 56k Danbooru images, which were filtered based on [CLIP Aesthetic Scoring](https://github.com/christophschuhmann/improved-aesthetic-predictor) where only images with an aesthetic score greater than `6.0` were used.

Captions are Danbooru-style captions.

## Downstream Uses

This model can be used for entertainment purposes and as a generative art assistant.

## Example Code

```python
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "hakurei/waifu-diffusion"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16')
pipe = pipe.to(device)

prompt = "touhou hakurei_reimu 1girl solo portrait"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save("reimu_hakurei.png")
```

## Team Members and Acknowledgements

This project would not have been possible without the incredible work by the [CompVis Researchers](https://ommer-lab.com/).

- [Anthony Mercurio](https://github.com/harubaru)
- [Salt](https://github.com/sALTaccount/)