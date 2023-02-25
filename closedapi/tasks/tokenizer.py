#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 10:12:57
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List, Tuple, Union

from transformers.tokenization_utils_base import TruncationStrategy
from transformers import AutoTokenizer

def tokenize(
    model: str,
    inputs: List[Union[str, Tuple[str, str]]],
    truncate: TruncationStrategy = TruncationStrategy.ONLY_SECOND,
    add_special_tokens: bool = True,
) -> List[List[int]]:
    """
    Tokenize each input string with the given model into a list of tokens.

    Parameters
    ----------
    model : str
        The model to use for tokenization.
    inputs : List[Union[str, Tuple[str, str]]]
        The inputs to tokenize.
    truncate : TruncationStrategy, optional
        The truncation strategy to use, by default TruncationStrategy.ONLY_SECOND
    add_special_tokens : bool, optional
        Whether to add special tokens, by default True
    
    Returns
    -------
    List[List[str]]
        The tokenized inputs.

    Examples
    --------
    >>> tokenize("gpt2", ["Hello world!"])
    [[15496, 995, 0]]
    >>> tokenize("gpt2", [("Hello world!", "This is a test")])
    [[15496, 995, 0, 1212, 318, 257, 1332]]
    >>> tokenize("gpt2", ["Hello world!", "This is a test"])
    [[15496, 995, 0], [1212, 318, 257, 1332]]
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer(inputs, truncation=truncate, add_special_tokens=add_special_tokens)['input_ids']