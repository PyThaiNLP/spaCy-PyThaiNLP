"""
spaCy-PyThaiNLP Modular Components.

Provides standalone spaCy pipeline components for:
- Sentence segmentation (PyThaiNLPSentencizer)
- Part-of-speech tagging (PyThaiNLPTagger)
- Named entity recognition (PyThaiNLPNER)
- Dependency parsing (PyThaiNLPParser)
- Word vectors (PyThaiNLPVectors)
- Lemmatization and text normalization (PyThaiNLPLemmatizer)
"""

from typing import List, Optional, Tuple

from pythainlp.tag import pos_tag
from pythainlp.tokenize import (
    DEFAULT_SENT_TOKENIZE_ENGINE,
    sent_tokenize,
)
from spacy import Language, util
from spacy.tokens import Doc


DEFAULT_SENT_ENGINE = DEFAULT_SENT_TOKENIZE_ENGINE
DEFAULT_POS_ENGINE = "perceptron"
DEFAULT_NER_ENGINE = "thainer"

# Constants for NER tags
NER_TAG_BEGIN = "B-"
NER_TAG_OUTSIDE = "O"

UPOS_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"
}
UD_CORPORA = {"orchid_ud", "blackboard_ud", "pud", "tdtb", "tud"}


def _is_sentenced(doc: Doc) -> bool:
    """Check if document has sentence boundaries set without warnings."""
    if hasattr(doc, "has_annotation"):
        return doc.has_annotation("SENT_START")
    return getattr(doc, "is_sentenced", False)


def parse_iob(
    ner_tags: List[Tuple[str, str]], base_offset: int = 0
) -> List[Tuple[int, int, str]]:
    """
    Parse IOB entity tags into character spans (start, end, label).

    Args:
        ner_tags: List of (word, tag) tuples from PyThaiNLP NER.
        base_offset: Starting character offset in the document.

    Returns:
        List of (start_char, end_char, label) tuples.
    """
    spans = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_label: Optional[str] = None
    char_offset = base_offset

    for word, tag in ner_tags:
        w_len = len(word)
        start = char_offset
        end = char_offset + w_len
        char_offset += w_len

        if tag.startswith(NER_TAG_BEGIN):
            if cur_label is not None and cur_start is not None:
                spans.append((cur_start, cur_end, cur_label))
            cur_start = start
            cur_end = end
            cur_label = tag[len(NER_TAG_BEGIN):]
        elif tag.startswith("I-"):
            tag_label = tag[2:]
            if cur_label is not None and tag_label == cur_label:
                cur_end = end
            else:
                if cur_label is not None and cur_start is not None:
                    spans.append((cur_start, cur_end, cur_label))
                cur_start = start
                cur_end = end
                cur_label = tag_label
        else:
            if cur_label is not None and cur_start is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_start = None
                cur_end = None
                cur_label = None

    if cur_label is not None and cur_start is not None:
        spans.append((cur_start, cur_end, cur_label))

    return spans


@Language.factory(
    "pythainlp_sentencizer",
    assigns=["token.is_sent_start"],
    default_config={
        "engine": DEFAULT_SENT_ENGINE,
        "keep_whitespace": True,
    },
)
class PyThaiNLPSentencizer:
    """Sentence segmentation component using PyThaiNLP sent_tokenize."""

    def __init__(
        self,
        nlp: Language,
        name: str,
        engine: str = DEFAULT_SENT_ENGINE,
        keep_whitespace: bool = True,
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.engine = engine
        self.keep_whitespace = keep_whitespace

    def __call__(self, doc: Doc) -> Doc:
        if len(doc) == 0:
            return doc
        if len(doc) == 1:
            doc[0].is_sent_start = True
            return doc

        sentences = sent_tokenize(
            str(doc.text),
            engine=self.engine,
            keep_whitespace=self.keep_whitespace,
        )
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

    def to_bytes(self, **kwargs) -> bytes:
        return b""

    def from_bytes(
        self, _bytes_data: bytes, **kwargs
    ) -> "PyThaiNLPSentencizer":
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLPSentencizer":
        return self


@Language.factory(
    "pythainlp_tagger",
    assigns=["token.tag", "token.pos"],
    default_config={
        "engine": DEFAULT_POS_ENGINE,
        "corpus": "orchid_ud",
        "set_tag": True,
        "set_pos": True,
    },
)
class PyThaiNLPTagger:
    """Part-of-speech tagging component using PyThaiNLP pos_tag."""

    def __init__(
        self,
        nlp: Language,
        name: str,
        engine: str = DEFAULT_POS_ENGINE,
        corpus: str = "orchid_ud",
        set_tag: bool = True,
        set_pos: bool = True,
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.engine = engine
        self.corpus = corpus
        self.set_tag = set_tag
        self.set_pos = set_pos

    def __call__(self, doc: Doc) -> Doc:
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
            tagged = pos_tag(words, engine=self.engine, corpus=self.corpus)
            for token, (_, raw_tag) in zip(group, tagged):
                tag_str = str(raw_tag) if raw_tag is not None else ""
                if self.set_tag:
                    token.tag_ = tag_str
                if self.set_pos:
                    if tag_str.upper() in UPOS_TAGS:
                        token.pos_ = tag_str.upper()
                    elif self.corpus in UD_CORPORA:
                        token.pos_ = "X"
        return doc

    def to_bytes(self, **kwargs) -> bytes:
        return b""

    def from_bytes(self, _bytes_data: bytes, **kwargs) -> "PyThaiNLPTagger":
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLPTagger":
        return self


@Language.factory(
    "pythainlp_ner",
    assigns=["doc.ents"],
    default_config={
        "engine": DEFAULT_NER_ENGINE,
        "corpus": "thainer",
        "nested": False,
        "span_key": "pythainlp_nner",
    },
)
class PyThaiNLPNER:
    """Named entity recognition component using PyThaiNLP NER and NNER."""

    def __init__(
        self,
        nlp: Language,
        name: str,
        engine: str = DEFAULT_NER_ENGINE,
        corpus: str = "thainer",
        nested: bool = False,
        span_key: str = "pythainlp_nner",
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.engine = engine
        self.corpus = corpus
        self.nested = nested
        self.span_key = span_key
        if self.nested:
            from pythainlp.tag import NNER
            self.nner = NNER()
            self.ner = None
        else:
            from pythainlp.tag import NER
            self.ner = NER(engine=self.engine, corpus=self.corpus)
            self.nner = None

    def __call__(self, doc: Doc) -> Doc:
        if len(doc) == 0:
            return doc

        if self.nested and self.nner is not None:
            _, spans = self.nner.tag(str(doc.text))
            span_list = []
            for ent_info in spans:
                ent_type = ent_info.get("entity_type", "ENTITY")
                ent_text = "".join(ent_info.get("text", []))
                if ent_text:
                    idx = doc.text.find(ent_text)
                    if idx != -1:
                        sp = doc.char_span(
                            idx,
                            idx + len(ent_text),
                            label=ent_type,
                            alignment_mode="expand",
                        )
                        if sp is not None:
                            span_list.append(sp)
            doc.spans[self.span_key] = span_list
            return doc

        if self.ner is None:
            return doc

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

    def to_bytes(self, **kwargs) -> bytes:
        return b""

    def from_bytes(self, _bytes_data: bytes, **kwargs) -> "PyThaiNLPNER":
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLPNER":
        return self


@Language.factory(
    "pythainlp_parser",
    assigns=["token.dep", "token.head", "token.pos"],
    default_config={
        "engine": "esupar",
        "model": None,
    },
)
class PyThaiNLPParser:
    """Dependency parsing component using PyThaiNLP dependency_parsing."""

    def __init__(
        self,
        nlp: Language,
        name: str,
        engine: str = "esupar",
        model: Optional[str] = None,
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.engine = engine
        self.model = model

    def __call__(self, doc: Doc) -> Doc:
        from pythainlp.parse import dependency_parsing
        text = str(doc.text)
        words = []
        spaces = []
        pos_tags = []
        deps = []
        heads = []

        dep_output = dependency_parsing(
            text,
            model=self.model,
            engine=self.engine,
            tag="list",
        )

        n_tokens = len(dep_output)
        for i, fields in enumerate(dep_output):
            if len(fields) < 10:
                raise ValueError(
                    f"Expected at least 10 fields in dependency parsing "
                    f"output, got {len(fields)}"
                )
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

    def to_bytes(self, **kwargs) -> bytes:
        return b""

    def from_bytes(self, _bytes_data: bytes, **kwargs) -> "PyThaiNLPParser":
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLPParser":
        return self


@Language.factory(
    "pythainlp_vectors",
    default_config={
        "model": "thai2fit_wv",
    },
)
class PyThaiNLPVectors:
    """Word vectors component using PyThaiNLP WordVector."""

    def __init__(
        self,
        nlp: Language,
        name: str,
        model: str = "thai2fit_wv",
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.model = model
        self._load_vectors()

    def _load_vectors(self) -> None:
        from pythainlp.word_vector import WordVector
        wv = WordVector(model_name=self.model)
        try:
            width = wv.model.vector_size
        except AttributeError:
            width = wv.model["แมว"].shape[0]
        if not self.nlp.vocab.vectors_length:
            self.nlp.vocab.reset_vectors(width=width)
        words = list(dict(wv.model.key_to_index).keys())
        for word in words:
            self.nlp.vocab[word].vector = wv.model[word]

    def __call__(self, doc: Doc) -> Doc:
        return doc

    def to_bytes(self, **kwargs) -> bytes:
        return b""

    def from_bytes(self, _bytes_data: bytes, **kwargs) -> "PyThaiNLPVectors":
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLPVectors":
        return self


@Language.factory(
    "pythainlp_lemmatizer",
    assigns=["token.lemma", "token.norm"],
    default_config={
        "normalize_text": True,
    },
)
class PyThaiNLPLemmatizer:
    """
    Lemmatization and text normalization component using PyThaiNLP normalize.
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        normalize_text: bool = True,
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.normalize_text = normalize_text

    def __call__(self, doc: Doc) -> Doc:
        from pythainlp.util import normalize
        for token in doc:
            if self.normalize_text:
                norm = normalize(token.text)
                token.norm_ = norm
                token.lemma_ = norm
            else:
                token.lemma_ = token.text
        return doc

    def to_bytes(self, **kwargs) -> bytes:
        return b""

    def from_bytes(
        self, _bytes_data: bytes, **kwargs
    ) -> "PyThaiNLPLemmatizer":
        return self

    def to_disk(self, _path: str, **kwargs) -> None:
        return None

    def from_disk(self, _path: str, **kwargs) -> "PyThaiNLPLemmatizer":
        return self
