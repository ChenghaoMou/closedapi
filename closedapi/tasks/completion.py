#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 18:22:46
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from typing import Dict, List

from transformers import pipeline, AutoTokenizer

def completion(
    model: str,
    prompt: str | None = None,
    suffix: str | None = None,
    max_tokens: int = 16,
    temperature: float = 1,
    top_p: int = 1,
    n: int = 1,
    stream: bool = False,
    logprobs: int | None = None,
    echo: bool = False,
    stop: str | List[str] | None = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    best_of: int = 1,
    logit_bias: Dict[int, float] | None = None,
):
    
    if prompt is None:
        prompt = ""
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline("text-generation", model=model, tokenizer=model)
    return pipe(
        prompt, 
        max_new_tokens=max_tokens, 
        temperature=temperature, 
        return_full_text=echo, 
        do_sample=True,
        num_return_sequences=n,
        num_beams=best_of,
        top_p=top_p,
        # repetition_penalty=frequency_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id= tokenizer.encode(stop) if stop else None,
    )


if __name__ == "__main__":
    print(completion("gpt2", "Hello world"))