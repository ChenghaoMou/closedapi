#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 10:12:57
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List

from transformers import AutoTokenizer

def detokenize(
    model: str,
    inputs: List[int],
) -> List[str]:
    """
    De-tokenize each input ids with the given model into a string.

    Parameters
    ----------
    model : str
        The model to use for de-tokenization.
    inputs : List[int]
        The inputs to de-tokenize.
    
    Returns
    -------
    List[str]
        The de-tokenized inputs.

    Examples
    --------
    >>> detokenize("gpt2", [[15496, 995, 0]])
    ['Hello world!']
    >>> detokenize("gpt2", [[15496, 995, 0, 1212, 318, 257, 1332]])
    ['Hello world!This is a test']
    >>> detokenize("gpt2", [[15496, 995, 0], [1212, 318, 257, 1332]])
    ['Hello world!', 'This is a test']
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    return [tokenizer.decode(inp) for inp in inputs]