from typing import List, Optional

from pythainlp.tag import pos_tag
from pythainlp.tokenize import (
    word_tokenize,
    DEFAULT_SENT_TOKENIZE_ENGINE,
    DEFAULT_WORD_TOKENIZE_ENGINE
)
from spacy import Language, util
from spacy.tokens import Doc, Span


DEFAULT_SENT_ENGINE = DEFAULT_SENT_TOKENIZE_ENGINE
DEFAULT_POS_ENGINE = "perceptron"
DEFAULT_NER_ENGINE = "thainer"

# Constants for sentence splitting
SENTENCE_SPLIT_MARKER = "SPLIT"

# Constants for NER tags
NER_TAG_BEGIN = "B-"
NER_TAG_OUTSIDE = "O"


@Language.factory(
    "pythainlp",
    assigns=["token.pos", "token.is_sent_start", "doc.ents"],
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
        "word_vector_model": "thai2fit_wv"
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
        tokenize_engine: str,
        pos_engine: str,
        sent_engine: str,
        ner_engine: str,
        dependency_parsing_engine: str,
        tokenize: bool,
        pos: bool,
        sent: bool,
        ner: bool,
        dependency_parsing: bool,
        word_vector: bool,
        dependency_parsing_model: Optional[str],
        word_vector_model: str,
        pos_corpus: str
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
            self.on_tokenize = False
            self.on_sent = False
        if self.on_tokenize:
            doc = self._tokenize(doc)
        if self.on_sent:
            doc = self._sent(doc)
        if self.on_pos:
            doc = self._pos(doc)
        if self.on_ner:
            doc = self._ner(doc)
        return doc
    
    def _tokenize(self, doc: Doc) -> Doc:
        """
        Tokenize text using PyThaiNLP tokenizer.

        Args:
            doc: The spaCy Doc to tokenize

        Returns:
            New Doc with tokenized words
        """
        words = list(word_tokenize(doc.text, engine=self.tokenize_engine))
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
        pos_tags = []
        if doc.is_sentenced:
            list_txt = [[token.text for token in sent] for sent in doc.sents]
        else:
            list_txt = [[token.text for token in doc]]
        for words in list_txt:
            tagged = pos_tag(words, engine=self.pos_engine, corpus=self.pos_corpus)
            pos_tags.extend([tag for _, tag in tagged])
        for i in range(len(pos_tags)):
            doc[i].pos_ = pos_tags[i]
        return doc

    def _sent(self, doc: Doc) -> Doc:
        """
        Add sentence boundaries to the document.

        Args:
            doc: The spaCy Doc to segment

        Returns:
            Doc with sentence boundaries marked
        """
        from pythainlp.tokenize import sent_tokenize
        sentences = sent_tokenize(str(doc.text), engine=self.sent_engine)
        tokenized = word_tokenize(SENTENCE_SPLIT_MARKER.join(sentences), engine=self.tokenize_engine)
        number_skip = 0
        seen_break = False
        new_tokens = []
        for word in tokenized:
            if SENTENCE_SPLIT_MARKER in word:
                if word.startswith(SENTENCE_SPLIT_MARKER):
                    new_tokens.append(SENTENCE_SPLIT_MARKER)
                    new_tokens.append(word.replace(SENTENCE_SPLIT_MARKER, ''))
                elif word.endswith(SENTENCE_SPLIT_MARKER):
                    new_tokens.append(word.replace(SENTENCE_SPLIT_MARKER, ''))
                    new_tokens.append(SENTENCE_SPLIT_MARKER)
                else:
                    new_tokens.append(word)
            else:
                new_tokens.append(word)
        for i, word in enumerate(new_tokens):
            if i - number_skip == len(doc) - 1:
                break
            elif i == 0:
                doc[i - number_skip].is_sent_start = True
            elif seen_break:
                doc[i - number_skip].is_sent_start = True
                seen_break = False
            elif SENTENCE_SPLIT_MARKER in word:
                seen_break = True
                number_skip += 1
            else:
                doc[i - number_skip].is_sent_start = False
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
            tag="list"
        )
        
        for fields in dep_output:
            if len(fields) < 10:
                raise ValueError(f"Expected at least 10 fields in dependency parsing output, got {len(fields)}")
            # Extract CoNLL-U format fields (only first 10)
            idx, word, _, postag, _, _, head, dep, _, space = fields[:10]
            words.append(word)
            pos_tags.append(postag)
            heads.append(int(head))
            deps.append(dep)
            spaces.append(space == '_')
        
        return Doc(self.nlp.vocab, words=words, spaces=spaces, pos=pos_tags, deps=deps, heads=heads)


    def _ner(self, doc: Doc) -> Doc:
        """
        Add named entity recognition tags to the document.

        Args:
            doc: The spaCy Doc to tag

        Returns:
            Doc with named entities added
        """
        # Extract text segments
        if doc.is_sentenced:
            text_segments = [sent.text for sent in doc.sents]
        else:
            text_segments = [token.text for token in doc]
        
        # Get NER tags for all segments
        ner_tags = []
        for segment in text_segments:
            ner_tags.extend(self.ner.tag(segment, pos=False))
        
        # Merge consecutive entity tokens into spans
        entity_spans = []
        char_offset = 0
        current_entity_label = ""
        
        for i, (word, tag) in enumerate(ner_tags):
            word_length = len(word)
            is_last = (i + 1 == len(ner_tags))
            
            if is_last and current_entity_label:
                entity_spans[-1][1] = char_offset + word_length
            elif is_last and tag.startswith(NER_TAG_BEGIN):
                entity_label = tag.replace(NER_TAG_BEGIN, "")
                entity_spans.append([char_offset, char_offset + word_length, entity_label])
            elif tag.startswith(NER_TAG_BEGIN) and not current_entity_label:
                current_entity_label = tag.replace(NER_TAG_BEGIN, "")
                entity_spans.append([char_offset, None, current_entity_label])
            elif tag.startswith(NER_TAG_BEGIN) and current_entity_label:
                entity_spans[-1][1] = char_offset
                current_entity_label = tag.replace(NER_TAG_BEGIN, "")
                entity_spans.append([char_offset, None, current_entity_label])
            elif tag == NER_TAG_OUTSIDE and current_entity_label:
                entity_spans[-1][1] = char_offset
                current_entity_label = ""
            
            char_offset += word_length
        
        # Create entity spans
        entities = []
        for start, end, label in entity_spans:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                entities.append(span)
        
        doc.ents = entities
        return doc

    def _vec(self) -> None:
        """
        Load word vectors into the spaCy vocabulary.
        """
        from pythainlp.word_vector import WordVector
        wv = WordVector(model_name=self.word_vector_model)
        self.nlp.vocab.reset_vectors(width=wv.model["แมว"].shape[0])
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