#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 18:22:46
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline

def edit(
    model: str,
    input: str,
    instruction: str,
    n: int = 1,
    temperature: float = 1,
    top_p: float = 1,
):
    pass


def image_edit(
    image: str,
    prompt: str,
    mask: str | None = None,
    n: int = 1,
    size: str = "1024x1024",
    response_format: str | None = None,
):
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
    init_image = Image.open(image).convert("RGB")
    init_image = init_image.resize((768, 512))
    images = pipe(
        prompt=prompt, 
        image=init_image, 
        strength=0.75, 
        guidance_scale=7.5, 
        mask_image=mask, 
        num_images_per_prompt=n
    ).images

    return images