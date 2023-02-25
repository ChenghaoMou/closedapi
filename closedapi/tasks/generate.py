#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 10:12:57
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Dict, List

from transformers.tokenization_utils_base import TruncationStrategy
from transformers import pipeline, AutoTokenizer
from diffusers import StableDiffusionPipeline

def generate(
    model: str,
    prompt: str,
    max_tokens: int = 40,
    temperature: float = 0.6,
    num_generations: int = 1,
    k: int = 0,
    p: float = 0.9,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    end_sequences: List[str] = None,
    stop_sequences: List[str] = None,
    return_likelihoods: bool = False,
    logit_bias: Dict[str, float] = None,
    truncate: TruncationStrategy = TruncationStrategy.ONLY_SECOND,
):

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline("text-generation", model=model, tokenizer=model)
    return pipe(
        prompt, 
        max_new_tokens=max_tokens, 
        temperature=temperature, 
        return_full_text=False, 
        do_sample=True,
        num_return_sequences=num_generations,
        num_beams=1,
        top_p=p,
        top_k=k,
        # repetition_penalty=frequency_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id= tokenizer.encode(stop_sequences) if stop_sequences else None,
        output_scores=return_likelihoods,
    )

def image_generate(
    prompt: str,
    n: int = 1,
    size: str = "1024x1024",
    response_format: str | None = None,
):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    return pipe(prompt, num_images_per_prompt=n).images

if __name__ == "__main__":
    
    for image in image_generate("a dog", n=3):
        image.show()

