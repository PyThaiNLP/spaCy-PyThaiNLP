from pythainlp.tokenize import sent_tokenize, word_tokenize
from spacy import Language, util
from spacy.tokens import Doc, Span

DEFAULT_SENT_ENGINE = "crfcut"


@Language.factory(
    "th_sents",
    assigns=["token.is_sent_start"],
    default_config={"engine": DEFAULT_SENT_ENGINE},
)
class ThaiSents:
    """Segment the Doc into sentences using PyThaiNLP.
    """

    def __init__(
        self,
        nlp,
        name,
        engine,
    ):
        """Initialize
        """
        self.nlp = nlp
        self.engine = engine

    def __call__(self, doc):
        tags = self.predict(doc)
        return tags

    def predict(self, doc:Doc):
        _text = sent_tokenize(str(doc.text), engine=self.engine)
        _doc = word_tokenize('SplitThword'.join(_text))
        number_skip=0
        seen_break = False
        for i,word in enumerate(_doc):
            if i ==0:
                doc[i-number_skip].is_sent_start = True
            elif seen_break:
                doc[i-number_skip].is_sent_start = True
                seen_break = False
            elif word == 'SplitThword':
                seen_break = True
                number_skip+=1
            else:
                doc[i-number_skip].is_sent_start = False
        return doc
