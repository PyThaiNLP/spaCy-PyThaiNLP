"""
spaCy-PyThaiNLP Tokenizer.

Provides a custom spaCy Tokenizer class and factory using PyThaiNLP
word_tokenize.
"""

from typing import Any, Optional

from pythainlp.tokenize import (
    DEFAULT_WORD_TOKENIZE_ENGINE,
    word_tokenize,
)
from spacy import registry, util
from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import DummyTokenizer
from spacy.vocab import Vocab


class PyThaiNLPTokenizer(DummyTokenizer):
    """
    Custom spaCy tokenizer powered by PyThaiNLP.

    Preserves original whitespace and character alignment using spaCy's
    get_words_and_spaces utility.
    """

    def __init__(
        self,
        vocab: Vocab,
        engine: str = DEFAULT_WORD_TOKENIZE_ENGINE,
        custom_dict: Optional[Any] = None,
        keep_whitespace: bool = True,
        join_broken_num: bool = True,
    ) -> None:
        """
        Initialize the tokenizer.

        Args:
            vocab: The spaCy Vocab object.
            engine: PyThaiNLP word tokenization engine
                (e.g. 'newmm', 'longest').
            custom_dict: Custom dictionary or Trie for word tokenization.
            keep_whitespace: Whether to preserve original whitespace in Doc.
            join_broken_num: Whether to rejoin broken formatted numbers.
        """
        self.vocab = vocab
        self.engine = engine
        self.custom_dict = custom_dict
        self.keep_whitespace = keep_whitespace
        self.join_broken_num = join_broken_num

    def __call__(self, text: str) -> Doc:
        """
        Tokenize a text string into a spaCy Doc.

        Args:
            text: The text to tokenize.

        Returns:
            A spaCy Doc with tokens and preserved whitespace.
        """
        if not text:
            return Doc(self.vocab, words=[], spaces=[])

        tokens = list(
            word_tokenize(
                text,
                engine=self.engine,
                custom_dict=self.custom_dict,
                keep_whitespace=False,
                join_broken_num=self.join_broken_num,
            )
        )

        try:
            words, spaces = util.get_words_and_spaces(tokens, text)
        except ValueError:
            # Fallback if get_words_and_spaces encounters an unexpected anomaly
            raw_tokens = list(
                word_tokenize(
                    text,
                    engine=self.engine,
                    custom_dict=self.custom_dict,
                    keep_whitespace=True,
                    join_broken_num=self.join_broken_num,
                )
            )
            words = raw_tokens
            spaces = [False] * len(words)

        return Doc(self.vocab, words=words, spaces=spaces)

    def to_bytes(self, **kwargs) -> bytes:
        """Serialize tokenizer configuration to bytes."""
        return self.engine.encode("utf-8")

    def from_bytes(self, bytes_data: bytes, **kwargs) -> "PyThaiNLPTokenizer":
        """Deserialize tokenizer configuration from bytes."""
        self.engine = bytes_data.decode("utf-8")
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        """Serialize tokenizer to disk."""
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLPTokenizer":
        """Deserialize tokenizer from disk."""
        return self


@registry.tokenizers("pythainlp_tokenizer")
def create_pythainlp_tokenizer(
    engine: str = DEFAULT_WORD_TOKENIZE_ENGINE,
    keep_whitespace: bool = True,
    join_broken_num: bool = True,
):
    """
    spaCy registry factory for PyThaiNLPTokenizer.

    Args:
        engine: PyThaiNLP word tokenization engine.
        keep_whitespace: Whether to preserve original whitespace in the Doc.
        join_broken_num: Whether to rejoin broken formatted numbers.

    Returns:
        A factory function creating PyThaiNLPTokenizer for an nlp instance.
    """
    def pythainlp_tokenizer_factory(nlp: Language) -> PyThaiNLPTokenizer:
        return PyThaiNLPTokenizer(
            nlp.vocab,
            engine=engine,
            keep_whitespace=keep_whitespace,
            join_broken_num=join_broken_num,
        )

    return pythainlp_tokenizer_factory
