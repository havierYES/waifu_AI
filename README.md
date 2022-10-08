---
language:
- en
tags:
- stable-diffusion
- text-to-image
license: creativeml-openrail-m
inference: false

---

# waifu-diffusion v1.3 - Diffusion for Weebs

waifu-diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning.

<<<<<<< HEAD
<img src=https://i.imgur.com/Y5Tmw1S.png width=75% height=75%>

[Original Weights](https://huggingface.co/hakurei/waifu-diffusion-v1-3)

# Gradio & Colab

We also support a [Gradio](https://github.com/gradio-app/gradio) Web UI and Colab with Diffusers to run Waifu Diffusion:
[![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/hakurei/waifu-diffusion-demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_8wPN7dJO746QXsFnB09Uq2VGgSRFuYE#scrollTo=1HaCauSq546O)
=======
<img src=https://cdn.discordapp.com/attachments/930499731451428926/1017258164439220254/unknown.png width=20% height=20%>
>>>>>>> b45bafccd9d0e0757b70a54c7ebc32ff56ca9ee1

## Model Description

[See here for a full model overview.](https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1)

<<<<<<< HEAD
## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
=======
The current model has been fine-tuned with a learning rate of 5.0e-6 for 4 epochs on 56k Danbooru text-image pairs which all have an aesthetic rating greater than `6.0`.

## Training Data & Annotative Prompting

The data used for fine-tuning has come from a random sample of 56k Danbooru images, which were filtered based on [CLIP Aesthetic Scoring](https://github.com/christophschuhmann/improved-aesthetic-predictor) where only images with an aesthetic score greater than `6.0` were used.

Captions are Danbooru-style captions.
>>>>>>> b45bafccd9d0e0757b70a54c7ebc32ff56ca9ee1

## Downstream Uses

This model can be used for entertainment purposes and as a generative art assistant.

## Example Code

```python
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    'waifu-diffusion',
    torch_dtype=torch.float32
).to('cuda')

<<<<<<< HEAD
prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
=======

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16')
pipe = pipe.to(device)

prompt = "touhou hakurei_reimu 1girl solo portrait"
>>>>>>> b45bafccd9d0e0757b70a54c7ebc32ff56ca9ee1
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=6)["sample"][0]  
    
image.save("test.png")
```

## Team Members and Acknowledgements

This project would not have been possible without the incredible work by the [CompVis Researchers](https://ommer-lab.com/).

- [Anthony Mercurio](https://github.com/harubaru)
- [Salt](https://github.com/sALTaccount/)
- [Sta @ Bit192](https://twitter.com/naclbbr)

In order to reach us, you can join our [Discord server](https://discord.gg/touhouai).

[![Discord Server](https://discordapp.com/api/guilds/930499730843250783/widget.png?style=banner2)](https://discord.gg/touhouai)