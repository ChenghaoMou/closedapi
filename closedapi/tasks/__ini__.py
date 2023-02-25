#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 10:12:57
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from closedapi.tasks.generate import generate, image_generate
from closedapi.tasks.classify import classify
from closedapi.tasks.completion import completion
from closedapi.tasks.detect_language import detect_language
from closedapi.tasks.embed import embed, embeddings
from closedapi.tasks.tokenizer import tokenize
from closedapi.tasks.detokenize import detokenize

__all__ = [
    "generate",
    "image_generate",
    "classify",
    "completion",
    "detect_language",
    "embed",
    "embeddings",
    "tokenize",
    "detokenize",
]