from pythainlp.tag import pos_tag
from pythainlp.tag import NER
from pythainlp.tokenize import (
    sent_tokenize,
    word_tokenize,
    DEFAULT_SENT_TOKENIZE_ENGINE,
    DEFAULT_WORD_TOKENIZE_ENGINE
)
from spacy import Language, util
from spacy.tokens import Doc, Span


DEFAULT_SENT_ENGINE = DEFAULT_SENT_TOKENIZE_ENGINE
DEFAULT_POS_ENGINE = "perceptron"
DEFAULT_NER = "thainer"


@Language.factory(
    "pythainlp",
    assigns=["token.pos","token.is_sent_start","doc.ents"],
    default_config={
        "pos_engine": DEFAULT_POS_ENGINE,
        "pos": True,
        "pos_corpus": "orchid_ud",
        "sent_engine": DEFAULT_SENT_ENGINE,
        "sent": True,
        "ner_engine": DEFAULT_NER,
        "ner": True,
        "tokenize_engine": DEFAULT_WORD_TOKENIZE_ENGINE,
        "tokenize": False,
    },
)
class PyThaiNLP:
    """
    SpaCy - PyThaiNLP
    """

    def __init__(
        self,
        nlp,
        name,
        tokenize_engine,
        pos_engine,
        sent_engine,
        ner_engine,
        tokenize,
        pos,
        sent,
        ner,
        pos_corpus
    ):
        """
        Initialize
        """
        self.nlp = nlp
        self.pos_engine = pos_engine
        self.sent_engine = sent_engine
        self.ner_engine = ner_engine
        self.tokenize_engine = tokenize_engine
        self.on_ner = ner
        self.on_pos = pos
        self.on_sent = sent
        self.on_tokenize = tokenize
        self.pos_corpus = pos_corpus
        if self.on_ner:
            self.ner = NER(engine=DEFAULT_NER)

    def __call__(self, doc:Doc):
        if self.on_tokenize:
            doc = self._tokenize(doc)
        if self.on_sent:
            doc = self._sent(doc)
        if self.on_pos:
            doc = self._pos(doc)
        if self.on_ner:
            doc = self._ner(doc)
        return doc
    
    def _tokenize(self,doc:Doc):
        words = list(word_tokenize(doc.text, engine=self.tokenize_engine))
        spaces = [i.isspace() for i in words]
        return Doc(self.nlp.vocab, words=words, spaces=spaces)

    def _pos(self,doc:Doc):
        _pos_tag = []
        if doc.is_sentenced:
            _list_txt = [i.text for i in list(doc.sents)]
        else:
            _list_txt = [doc.text]
        for i in _list_txt:
            _w = word_tokenize(i, engine=self.tokenize_engine)
            _tag_ = pos_tag(_w, engine=self.pos_engine,corpus=self.pos_corpus)
            _pos_tag.extend([tag for _,tag in _tag_])
        for i,_ in enumerate(_pos_tag):
            doc[i].pos_ = _pos_tag[i]
        return doc

    def _sent(self, doc:Doc):
        _text = sent_tokenize(str(doc.text), engine=self.sent_engine)
        _doc = word_tokenize('SplitThword'.join(_text), engine=self.tokenize_engine)
        number_skip = 0
        seen_break = False
        for i,word in enumerate(_doc):
            if i == 0:
                doc[i-number_skip].is_sent_start = True
            elif seen_break:
                doc[i-number_skip].is_sent_start = True
                seen_break = False
            elif word == 'SplitThword':
                seen_break = True
                number_skip += 1
            else:
                doc[i-number_skip].is_sent_start = False
        return doc

    def _dep(self, doc:Doc):
        # TODO
        pass

    def _ner(self, doc:Doc):
        _ner_ = self.ner.tag(doc.text, pos=False)
        _new_ner = []
        c=0
        _t=""
        for i,(w, tag) in enumerate(_ner_):
            len_w = len(w)
            if i == len(_ner_) -1 and _t != "":
                _new_ner[-1][1] = c+len_w
            elif tag.startswith("B-") and _t=="":
                _t = tag.replace("B-","")
                _new_ner.append([c,None,_t])
            elif tag.startswith("B-") and _t!="":
                _new_ner[-1][1] = c
                _t =  tag.replace("B-","")
                _new_ner.append([c,None,_t])
            elif tag == "O" and _t!="":
                _new_ner[-1][1] = c
                _t=""
            c+=len_w
        _ents = []
        for start, end, label in _new_ner:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                pass
            else:
                _ents.append(span)

        doc.ents = _ents
        return doc
