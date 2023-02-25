#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 10:12:57
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy


def embed(
    model: str,
    inputs: List[Union[str, Tuple[str, str]]],
    truncate: TruncationStrategy = TruncationStrategy.ONLY_SECOND,
) -> np.ndarray:
    """
    Embed each input string with the given model into a vector.

    Parameters
    ----------
    model : str
        The model to use for embedding.
    inputs : List[Union[str, Tuple[str, str]]]
        The inputs to embed.
    truncate : TruncationStrategy, optional
        The truncation strategy to use, by default TruncationStrategy.ONLY_SECOND

    Returns
    -------
    np.ndarray
        The embedded inputs.

    Examples
    --------
    >>> embed("bert-base-uncased", ["Hello world!"]).shape
    (1, 768)
    >>> embed("bert-base-uncased", [("Hello world!", "This is a test")]).shape
    (1, 768)
    >>> embed("bert-base-uncased", ["Hello world!", "This is a test"]).shape
    (2, 768)
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    batch_size = 8
    embeddings = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        batch = tokenizer(
            batch,
            truncation=truncate,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            # Use mean pooling instead of CLS token for embedding since CLS token is not always available.
            batch_embeddings = model(**batch).last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.detach().cpu().numpy()

embeddings = embed

__all__ = ["embeddings", "embed"]