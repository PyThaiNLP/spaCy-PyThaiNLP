"""
spaCy-PyThaiNLP: Thai language support for spaCy using PyThaiNLP.

This module provides the main PyThaiNLP pipeline component and convenience
functions for loading Thai language models in spaCy.
"""

from typing import Any, Optional

from pythainlp.tag import pos_tag
from pythainlp.tokenize import (
    DEFAULT_WORD_TOKENIZE_ENGINE,
    word_tokenize,
)
from spacy import Language, util
from spacy.tokens import Doc

from spacy_pythainlp.components import (
    DEFAULT_NER_ENGINE,
    DEFAULT_POS_ENGINE,
    DEFAULT_SENT_ENGINE,
    NER_TAG_BEGIN,
    NER_TAG_OUTSIDE,
    UD_CORPORA,
    UPOS_TAGS,
    PyThaiNLPLemmatizer,
    PyThaiNLPNER,
    PyThaiNLPParser,
    PyThaiNLPSentencizer,
    PyThaiNLPTagger,
    PyThaiNLPVectors,
    _is_sentenced,
    parse_iob,
)
from spacy_pythainlp.tokenizer import (
    PyThaiNLPTokenizer,
    create_pythainlp_tokenizer,
)

# Constants for sentence splitting (retained for backward compatibility)
SENTENCE_SPLIT_MARKER = "SPLIT"


@Language.factory(
    "pythainlp",
    assigns=["token.pos", "token.tag", "token.is_sent_start", "doc.ents"],
    default_config={
        "pos_engine": DEFAULT_POS_ENGINE,
        "pos": True,
        "pos_corpus": "orchid_ud",
        "sent_engine": DEFAULT_SENT_ENGINE,
        "sent": True,
        "ner_engine": DEFAULT_NER_ENGINE,
        "ner": True,
        "tokenize_engine": DEFAULT_WORD_TOKENIZE_ENGINE,
        "tokenize": False,
        "dependency_parsing": False,
        "dependency_parsing_engine": "esupar",
        "dependency_parsing_model": None,
        "word_vector": True,
        "word_vector_model": "thai2fit_wv",
    },
)
class PyThaiNLP:
    """
    SpaCy pipeline component for Thai language processing using PyThaiNLP.

    This component provides Thai-specific NLP capabilities including:
    - Word tokenization
    - Part-of-speech tagging
    - Named entity recognition
    - Sentence segmentation
    - Dependency parsing
    - Word vectors
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        tokenize_engine: str = DEFAULT_WORD_TOKENIZE_ENGINE,
        pos_engine: str = DEFAULT_POS_ENGINE,
        sent_engine: str = DEFAULT_SENT_ENGINE,
        ner_engine: str = DEFAULT_NER_ENGINE,
        dependency_parsing_engine: str = "esupar",
        tokenize: bool = False,
        pos: bool = True,
        sent: bool = True,
        ner: bool = True,
        dependency_parsing: bool = False,
        word_vector: bool = True,
        dependency_parsing_model: Optional[str] = None,
        word_vector_model: str = "thai2fit_wv",
        pos_corpus: str = "orchid_ud",
    ) -> None:
        """
        Initialize the PyThaiNLP pipeline component.

        Args:
            nlp: The spaCy Language object
            name: Name of the pipeline component
            tokenize_engine: Engine for word tokenization
            pos_engine: Engine for part-of-speech tagging
            sent_engine: Engine for sentence segmentation
            ner_engine: Engine for named entity recognition
            dependency_parsing_engine: Engine for dependency parsing
            tokenize: Enable word tokenization
            pos: Enable part-of-speech tagging
            sent: Enable sentence segmentation
            ner: Enable named entity recognition
            dependency_parsing: Enable dependency parsing
            word_vector: Enable word vectors
            dependency_parsing_model: Model for dependency parsing
            word_vector_model: Model for word vectors
            pos_corpus: Corpus for POS tagging
        """
        self.nlp = nlp
        self.name = name
        self.word_vector = word_vector
        self.word_vector_model = word_vector_model
        if self.word_vector:
            self._vec()
        self.pos_engine = pos_engine
        self.sent_engine = sent_engine
        self.ner_engine = ner_engine
        self.tokenize_engine = tokenize_engine
        self.on_ner = ner
        self.on_pos = pos
        self.on_sent = sent
        self.on_tokenize = tokenize
        self.pos_corpus = pos_corpus
        self.dependency_parsing = dependency_parsing
        self.dependency_parsing_engine = dependency_parsing_engine
        self.dependency_parsing_model = dependency_parsing_model
        if self.on_ner:
            from pythainlp.tag import NER
            self.ner = NER(engine=self.ner_engine)
        else:
            self.ner = None

    def __call__(self, doc: Doc) -> Doc:
        """
        Process a Doc object through the PyThaiNLP pipeline.

        Args:
            doc: The spaCy Doc to process

        Returns:
            The processed Doc with Thai NLP annotations
        """
        if self.dependency_parsing:
            doc = self._dep(doc)
        elif self.on_tokenize:
            doc = self._tokenize(doc)

        if self.on_sent and not self.dependency_parsing:
            doc = self._sent(doc)
        if self.on_pos:
            doc = self._pos(doc)
        if self.on_ner:
            doc = self._ner(doc)
        return doc

    def _tokenize(self, doc: Doc) -> Doc:
        """
        Tokenize text using PyThaiNLP tokenizer while preserving whitespace.

        Args:
            doc: The spaCy Doc to tokenize

        Returns:
            New Doc with tokenized words and preserved whitespace
        """
        tokens = list(
            word_tokenize(
                doc.text,
                engine=self.tokenize_engine,
                keep_whitespace=False,
            )
        )
        try:
            words, spaces = util.get_words_and_spaces(tokens, doc.text)
        except ValueError:
            raw_tokens = list(
                word_tokenize(
                    doc.text,
                    engine=self.tokenize_engine,
                    keep_whitespace=True,
                )
            )
            words = raw_tokens
            spaces = [False] * len(words)
        return Doc(self.nlp.vocab, words=words, spaces=spaces)

    def _pos(self, doc: Doc) -> Doc:
        """
        Add part-of-speech tags to tokens.

        Args:
            doc: The spaCy Doc to tag

        Returns:
            Doc with POS tags added
        """
        if len(doc) == 0:
            return doc

        if _is_sentenced(doc):
            token_groups = [[token for token in sent] for sent in doc.sents]
        else:
            token_groups = [[token for token in doc]]

        for group in token_groups:
            if not group:
                continue
            words = [t.text for t in group]
            tagged = pos_tag(
                words, engine=self.pos_engine, corpus=self.pos_corpus
            )
            for token, (_, raw_tag) in zip(group, tagged):
                tag_str = str(raw_tag) if raw_tag is not None else ""
                token.tag_ = tag_str
                if tag_str.upper() in UPOS_TAGS:
                    token.pos_ = tag_str.upper()
                elif self.pos_corpus in UD_CORPORA:
                    token.pos_ = "X"
        return doc

    def _sent(self, doc: Doc) -> Doc:
        """
        Add sentence boundaries without leaving unassigned tokens.

        Args:
            doc: The spaCy Doc to segment

        Returns:
            Doc with sentence boundaries marked
        """
        if len(doc) == 0:
            return doc
        if len(doc) == 1:
            doc[0].is_sent_start = True
            return doc

        from pythainlp.tokenize import sent_tokenize

        sentences = sent_tokenize(str(doc.text), engine=self.sent_engine)
        if not sentences:
            doc[0].is_sent_start = True
            for token in doc[1:]:
                token.is_sent_start = False
            return doc

        sent_starts = set()
        cur_pos = 0
        for s in sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            idx = doc.text.find(s_stripped, cur_pos)
            if idx != -1:
                for token in doc:
                    if token.idx >= idx:
                        sent_starts.add(token.i)
                        break
                cur_pos = idx + len(s_stripped)

        for token in doc:
            token.is_sent_start = (token.i in sent_starts) or (token.i == 0)

        return doc

    def _dep(self, doc: Doc) -> Doc:
        """
        Perform dependency parsing on the document.

        Args:
            doc: The spaCy Doc to parse

        Returns:
            New Doc with dependency annotations

        Raises:
            ValueError: If dependency parsing output has fewer than 10 fields
        """
        from pythainlp.parse import dependency_parsing

        text = str(doc.text)
        words = []
        spaces = []
        pos_tags = []
        deps = []
        heads = []

        dep_output = dependency_parsing(
            text,
            model=self.dependency_parsing_model,
            engine=self.dependency_parsing_engine,
            tag="list",
        )

        n_tokens = len(dep_output)
        for i, fields in enumerate(dep_output):
            if len(fields) < 10:
                raise ValueError(
                    f"Expected at least 10 fields in dependency parsing "
                    f"output, got {len(fields)}"
                )
            # Extract CoNLL-U format fields (only first 10)
            idx, word, _, postag, _, _, head_val, dep, _, space = fields[:10]
            words.append(word)
            pos_tags.append(postag)
            deps.append(dep)

            # Check SpaceAfter=No / SpaceAfter=Yes / '_'
            has_no_space = any("SpaceAfter=No" in str(f) for f in fields[9:])
            if has_no_space:
                spaces.append(False)
            elif any("SpaceAfter=Yes" in str(f) for f in fields[9:]):
                spaces.append(True)
            else:
                spaces.append(space == "_")

            head_int = int(head_val)
            if head_int == 0 or dep.lower() == "root":
                heads.append(i)
            elif head_int == n_tokens:
                heads.append(head_int - 1)
            elif 0 <= head_int < n_tokens:
                heads.append(head_int)
            else:
                heads.append(max(0, head_int - 1))

        return Doc(
            self.nlp.vocab,
            words=words,
            spaces=spaces,
            pos=pos_tags,
            deps=deps,
            heads=heads,
        )

    def _ner(self, doc: Doc) -> Doc:
        """
        Add named entity recognition tags to the document.

        Args:
            doc: The spaCy Doc to tag

        Returns:
            Doc with named entities added
        """
        if len(doc) == 0:
            return doc

        if self.ner is None:
            return doc

        # Extract text segments with their base character offsets
        if _is_sentenced(doc):
            segments = [(sent.text, sent.start_char) for sent in doc.sents]
        else:
            segments = [(doc.text, 0)]

        raw_spans = []
        for segment_text, base_offset in segments:
            if not segment_text.strip():
                continue
            ner_tags = self.ner.tag(segment_text, pos=False)
            raw_spans.extend(parse_iob(ner_tags, base_offset=base_offset))

        entities = []
        for start, end, label in raw_spans:
            span = doc.char_span(
                start, end, label=label, alignment_mode="contract"
            )
            if span is None:
                span = doc.char_span(
                    start, end, label=label, alignment_mode="expand"
                )
            if span is not None:
                entities.append(span)

        doc.ents = util.filter_spans(entities)
        return doc

    def _vec(self) -> None:
        """
        Load word vectors into the spaCy vocabulary.
        """
        from pythainlp.word_vector import WordVector

        wv = WordVector(model_name=self.word_vector_model)
        try:
            width = wv.model.vector_size
        except AttributeError:
            width = wv.model["แมว"].shape[0]

        if not self.nlp.vocab.vectors_length:
            self.nlp.vocab.reset_vectors(width=width)
        words = list(dict(wv.model.key_to_index).keys())
        for word in words:
            self.nlp.vocab[word].vector = wv.model[word]

    def to_bytes(self, **kwargs) -> bytes:
        """Serialize the component to bytes."""
        return b""

    def from_bytes(self, _bytes_data: bytes, **kwargs) -> "PyThaiNLP":
        """Deserialize the component from bytes."""
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        """Serialize the component to disk."""
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLP":
        """Deserialize the component from disk."""
        return self


def blank(
    name: str = "th",
    *,
    tokenize_engine: str = DEFAULT_WORD_TOKENIZE_ENGINE,
    custom_dict: Optional[Any] = None,
    **kwargs,
) -> Language:
    """
    Create a blank spaCy Language model with PyThaiNLPTokenizer.

    Args:
        name: Language code (default: 'th').
        tokenize_engine: Word tokenization engine (default: 'newmm').
        custom_dict: Optional custom dictionary or Trie.
        **kwargs: Additional arguments passed to spacy.blank.

    Returns:
        A spaCy Language object configured with PyThaiNLPTokenizer.
    """
    import spacy

    nlp = spacy.blank(name, **kwargs)
    nlp.tokenizer = PyThaiNLPTokenizer(
        nlp.vocab,
        engine=tokenize_engine,
        custom_dict=custom_dict,
    )
    return nlp


def load(
    name: str = "th",
    *,
    tokenize_engine: str = DEFAULT_WORD_TOKENIZE_ENGINE,
    custom_dict: Optional[Any] = None,
    pos: bool = True,
    pos_engine: str = DEFAULT_POS_ENGINE,
    pos_corpus: str = "orchid_ud",
    sent: bool = True,
    sent_engine: str = DEFAULT_SENT_ENGINE,
    ner: bool = True,
    ner_engine: str = DEFAULT_NER_ENGINE,
    dependency_parsing: bool = False,
    dependency_parsing_engine: str = "esupar",
    dependency_parsing_model: Optional[str] = None,
    word_vector: bool = False,
    word_vector_model: str = "thai2fit_wv",
    **kwargs,
) -> Language:
    """
    Load a Thai spaCy pipeline configured with PyThaiNLP.

    Args:
        name: Language code (default: 'th').
        tokenize_engine: Word tokenization engine (default: 'newmm').
        custom_dict: Optional custom dictionary for word tokenization.
        pos: Enable part-of-speech tagging (default: True).
        pos_engine: POS tagger engine (default: 'perceptron').
        pos_corpus: POS tagger corpus (default: 'orchid_ud').
        sent: Enable sentence segmentation (default: True).
        sent_engine: Sentence segmentation engine (default: 'crfcut').
        ner: Enable named entity recognition (default: True).
        ner_engine: NER engine (default: 'thainer').
        dependency_parsing: Enable dependency parsing (default: False).
        dependency_parsing_engine: Dependency parser (default: 'esupar').
        dependency_parsing_model: Optional dependency parsing model.
        word_vector: Enable word vectors (default: False).
        word_vector_model: Word vector model (default: 'thai2fit_wv').
        **kwargs: Additional arguments passed to spacy.blank.

    Returns:
        A ready-to-use spaCy Language pipeline with Thai NLP capabilities.
    """
    nlp = blank(
        name,
        tokenize_engine=tokenize_engine,
        custom_dict=custom_dict,
        **kwargs,
    )
    nlp.add_pipe(
        "pythainlp",
        config={
            "pos_engine": pos_engine,
            "pos": pos,
            "pos_corpus": pos_corpus,
            "sent_engine": sent_engine,
            "sent": sent,
            "ner_engine": ner_engine,
            "ner": ner,
            "tokenize_engine": tokenize_engine,
            "tokenize": False,
            "dependency_parsing": dependency_parsing,
            "dependency_parsing_engine": dependency_parsing_engine,
            "dependency_parsing_model": dependency_parsing_model,
            "word_vector": word_vector,
            "word_vector_model": word_vector_model,
        },
    )
    return nlp


__all__ = [
    "DEFAULT_NER_ENGINE",
    "DEFAULT_POS_ENGINE",
    "DEFAULT_SENT_ENGINE",
    "NER_TAG_BEGIN",
    "NER_TAG_OUTSIDE",
    "SENTENCE_SPLIT_MARKER",
    "UD_CORPORA",
    "UPOS_TAGS",
    "PyThaiNLP",
    "PyThaiNLPLemmatizer",
    "PyThaiNLPNER",
    "PyThaiNLPParser",
    "PyThaiNLPSentencizer",
    "PyThaiNLPTagger",
    "PyThaiNLPTokenizer",
    "PyThaiNLPVectors",
    "blank",
    "create_pythainlp_tokenizer",
    "load",
    "parse_iob",
]
