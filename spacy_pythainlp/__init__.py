"""
spaCy-PyThaiNLP: Thai language support for spaCy using PyThaiNLP.

This package provides a spaCy pipeline component that integrates PyThaiNLP's
Thai NLP capabilities, including tokenization, POS tagging, NER, sentence
segmentation, dependency parsing, and word vectors.
"""

from spacy_pythainlp.core import PyThaiNLP

__version__ = "1.0.0"
__all__ = ["PyThaiNLP"]
