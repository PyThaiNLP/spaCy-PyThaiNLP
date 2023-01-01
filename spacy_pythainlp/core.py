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


@Language.factory(
    "pythainlp",
    assigns=["token.pos","token.is_sent_start","doc.ents"],
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
        dependency_parsing_engine,
        tokenize,
        pos,
        sent,
        ner,
        dependency_parsing,
        word_vector,
        dependency_parsing_model,
        word_vector_model,
        pos_corpus
    ):
        """
        Initialize
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

    def __call__(self, doc:Doc):
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
    
    def _tokenize(self, doc:Doc):
        words = list(word_tokenize(doc.text, engine=self.tokenize_engine))
        spaces = [i.isspace() for i in words]
        return Doc(self.nlp.vocab, words=words, spaces=spaces)

    def _pos(self, doc:Doc):
        _pos_tag = []
        if doc.is_sentenced:
            _list_txt = [[j.text for j in i] for i in list(doc.sents)]
        else:
            _list_txt = [[j.text for j in doc]]
        for i in _list_txt:
            _w = i
            _tag_ = pos_tag(_w, engine=self.pos_engine,corpus=self.pos_corpus)
            _pos_tag.extend([tag for _,tag in _tag_])
        for i,_ in enumerate(_pos_tag):
            #print(doc[i])
            doc[i].pos_ = _pos_tag[i]
        return doc

    def _sent(self, doc:Doc):
        from pythainlp.tokenize import sent_tokenize
        _text = sent_tokenize(str(doc.text), engine=self.sent_engine)
        _doc = word_tokenize('SPLIT'.join(_text), engine=self.tokenize_engine)
        #print(_doc)
        number_skip = 0
        seen_break = False
        _new_cut = []
        for i,word in enumerate(_doc):
            if 'SPLIT' in word:
                if word.startswith("SPLIT"):
                    _new_cut.append("SPLIT")
                    _new_cut.append(word.replace('SPLIT',''))
                elif word.endswith("SPLIT"):
                    _new_cut.append(word.replace('SPLIT',''))
                    _new_cut.append("SPLIT")
                else:
                    _new_cut.append(word)
            else:
                _new_cut.append(word)
        #print(_new_cut)
        for i,word in enumerate(_new_cut):
            #print(str(i),str(word))
            if i-number_skip == len(doc) -1:
                break
            elif i == 0:
                doc[i-number_skip].is_sent_start = True
            elif seen_break:
                doc[i-number_skip].is_sent_start = True
                seen_break = False
            elif 'SPLIT' in word:
                seen_break = True
                number_skip += 1
            else:
                doc[i-number_skip].is_sent_start = False
        return doc

    def _dep(self, doc:Doc):
        from pythainlp.parse import dependency_parsing
        text = str(doc.text)
        words = []
        spaces = []
        pos = []
        tags = []
        morphs = []
        deps = []
        heads = []
        lemmas = []
        offset = 0
        _dep_temp = dependency_parsing(text,model=self.dependency_parsing_model, engine=self.dependency_parsing_engine, tag="list")
        for i in _dep_temp:
            idx,word,_,postag,_,_,head,dep,_,space =  i
            words.append(word)
            pos.append(postag)
            heads.append(int(head))
            deps.append(dep)
            if space=='_':
                spaces.append(True)
            else:
                spaces.append(False)
        return Doc(self.nlp.vocab, words=words, spaces=spaces,pos=pos,deps=deps,heads=heads)


    def _ner(self, doc:Doc):
        _list_txt = []
        if doc.is_sentenced:
            _list_txt = [i.text for i in list(doc.sents)]
        else:
            _list_txt = [j.text for j in doc]
        _ner_ =[]
        for i in _list_txt:
            _ner_.extend(self.ner.tag(i, pos=False))
        #print(_ner_)
        _new_ner = []
        c=0
        _t=""
        for i,(w, tag) in enumerate(_ner_):
            len_w = len(w)
            #print(str(i),str(w),str(tag))
            if i+1 == len(_ner_) and _t != "":
                _new_ner[-1][1] = c+len_w
            elif i+1 == len(_ner_) and tag.startswith("B-"):
                _t =  tag.replace("B-","")
                _new_ner.append([c,c+len_w,_t])
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
        #print(_new_ner)
        for start, end, label in _new_ner:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                pass
            else:
                _ents.append(span)

        doc.ents = _ents
        return doc
    
    def _vec(self):
        from pythainlp.word_vector import WordVector
        _wv = WordVector(model_name=self.word_vector_model)
        self.nlp.vocab.reset_vectors(width=_wv.model["แมว"].shape[0])
        _temp = list(dict(_wv.model.key_to_index).keys())
        for i in _temp:
            self.nlp.vocab[i].vector = _wv.model[i]

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes_data, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self