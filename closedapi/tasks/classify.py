#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 10:12:57
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List, Tuple

from transformers import pipeline


def classify(
    model: str,
    inputs: List[str],
    examples: List[Tuple[str, str]],
):
    prompt = ""
    for inp, label in examples:
        prompt += f"Input: {inp}\nClass: {label}\n###\n"
    
    inputs = [f'{prompt}Input: {inp}\nClass:' for inp in inputs]
    pipe = pipeline("text-generation", model=model, tokenizer=model)
    generations = pipe(inputs, max_new_tokens=5, temperature=0.5, return_full_text=False, do_sample=True)
    return [
        generation[0]['generated_text'].split('###', 1)[0].strip(' \n') for generation in generations
    ]

if __name__ == "__main__":

    print(classify("EleutherAI/gpt-neo-1.3B", [
        "how are you",
        "see you tomorrow",
        ], [
        ("hello", "A"),
        ("goodbye", "B"),
        ("what's up", "A"),
        ("see you later", "B"),
    ]))
