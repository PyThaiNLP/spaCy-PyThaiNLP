# spaCy-PyThaiNLP
PyThaiNLP For spaCy

Work in processing...

## Install

> pip install spacy-pythainlp

## How to use


**Example**
```python
import spacy
from spacy_pythainlp.core import *

nlp = spacy.blank("th")
# Segment the Doc into sentences
nlp.add_pipe(
   "pythainlp", 
)

data=nlp("ผมเป็นคนไทย   แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน  ผมอยากไปเที่ยว")
print(list(list(data.sents)))
# output: [ผมเป็นคนไทย   แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน  , ผมอยากไปเที่ยว]
```

You can config the setting in the nlp.add_pipe.
```python
nlp.add_pipe(
    "pythainlp", 
    config={
        "pos_engine": "perceptron",
        "pos": True,
        "pos_corpus": "orchid_ud",
        "sent_engine": "crfcut",
        "sent": True,
        "ner_engine": "thainer",
        "ner": True,
        "tokenize_engine": "newmm",
        "tokenize": False,
    }
)
```

- tokenize: Bool (True or False) to change the word tokenize. (the default spaCy is newmm of PyThaiNLP)
- tokenize_engine: The tokenize engine. You can read more: [Options for engine](https://pythainlp.github.io/dev-docs/api/tokenize.html#pythainlp.tokenize.word_tokenize)
- sent: Bool (True or False) to turn on the sentence tokenizer.
- sent_engine: The sentence tokenizer engine. You can read more: [Options for engine](https://pythainlp.github.io/dev-docs/api/tokenize.html#pythainlp.tokenize.sent_tokenize)
- pos:  Bool (True or False) to turn on the part-of-speech.
- pos_engine: The part-of-speech engine. You can read more: [Options for engine](https://pythainlp.github.io/dev-docs/api/tag.html#pythainlp.tag.pos_tag)
- ner: Bool (True or False) to turn on the NER.
- ner_engine: The NER engine. You can read more: [Options for engine](https://pythainlp.github.io/dev-docs/api/tag.html#pythainlp.tag.NER)


## License

```
   Copyright 2016-2023 PyThaiNLP Project

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```