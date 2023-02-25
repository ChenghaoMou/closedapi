#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-02-11 10:12:57
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Any, Dict, List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.tokenization_utils_base import TruncationStrategy


def detect_language(
    model: str,
    inputs: List[str],
    truncate: TruncationStrategy = TruncationStrategy.ONLY_SECOND,
) -> List[Dict[str, Any]]:
    """
    Detects the language of a given text.
    
    Parameters
    ----------
    model : str
        The model to use for the prediction.
    inputs : List[str]
        The texts to detect the language of.
    truncate : TruncationStrategy, optional
        The truncation strategy to use, by default TruncationStrategy.ONLY_SECOND

    Returns
    -------
    List[str]
        The detected languages with their confidence scores. Languages are represented by their ISO 639-1 code.

    Examples
    --------
    >>> results = detect_language("papluca/xlm-roberta-base-language-detection", ["Hello world!", "Bonjour le monde!"])
    >>> results[0]['label']
    'en'
    >>> results[1]['label']
    'fr'
    """

    tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
    model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe(inputs, truncation=truncate)
