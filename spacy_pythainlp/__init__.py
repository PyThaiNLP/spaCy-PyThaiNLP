"""
spaCy-PyThaiNLP: Thai language support for spaCy using PyThaiNLP.

This package provides spaCy pipeline components and a custom tokenizer
that integrate PyThaiNLP's Thai NLP capabilities, including tokenization,
POS tagging, NER, sentence segmentation, dependency parsing, and word vectors.
"""

from spacy_pythainlp.components import (
    PyThaiNLPLemmatizer,
    PyThaiNLPNER,
    PyThaiNLPParser,
    PyThaiNLPSentencizer,
    PyThaiNLPTagger,
    PyThaiNLPVectors,
)
from spacy_pythainlp.core import (
    PyThaiNLP,
    blank,
    load,
)
from spacy_pythainlp.tokenizer import (
    PyThaiNLPTokenizer,
    create_pythainlp_tokenizer,
)

__version__ = "1.1.0"
__all__ = [
    "PyThaiNLP",
    "PyThaiNLPTokenizer",
    "PyThaiNLPSentencizer",
    "PyThaiNLPTagger",
    "PyThaiNLPNER",
    "PyThaiNLPParser",
    "PyThaiNLPVectors",
    "PyThaiNLPLemmatizer",
    "load",
    "blank",
    "create_pythainlp_tokenizer",
]
