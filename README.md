# spaCy-PyThaiNLP
This package wraps the PyThaiNLP library to add support Thai for spaCy.

**Support List**
- Word segmentation
- Part-of-speech
- Named entity recognition
- Sentence segmentation
- Dependency parsing


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
        "dependency_parsing": False,
        "dependency_parsing_engine": "esupar",
        "dependency_parsing_model": None,
    }
)
```

- tokenize: Bool (True or False) to change the word tokenize. (the default spaCy is newmm of PyThaiNLP)
- tokenize_engine: The tokenize engine. You can read more: [Options for engine](https://pythainlp.github.io/docs/3.1/api/tokenize.html#pythainlp.tokenize.word_tokenize)
- sent: Bool (True or False) to turn on the sentence tokenizer.
- sent_engine: The sentence tokenizer engine. You can read more: [Options for engine](https://pythainlp.github.io/docs/3.1/api/tokenize.html#pythainlp.tokenize.sent_tokenize)
- pos:  Bool (True or False) to turn on the part-of-speech.
- pos_engine: The part-of-speech engine. You can read more: [Options for engine](https://pythainlp.github.io/docs/3.1/api/tag.html#pythainlp.tag.pos_tag)
- ner: Bool (True or False) to turn on the NER.
- ner_engine: The NER engine. You can read more: [Options for engine](https://pythainlp.github.io/docs/3.1/api/tag.html#pythainlp.tag.NER)
- dependency_parsing: Bool (True or False) to turn on the Dependency parsing.
- dependency_parsing_engine: The Dependency parsing engine. You can read more: [Options for engine](https://pythainlp.github.io/docs/3.1/api/parse.html#pythainlp.parse.dependency_parsing)
- dependency_parsing_model: The Dependency parsing model. You can read more: [Options for model](https://pythainlp.github.io/docs/3.1/api/parse.html#pythainlp.parse.dependency_parsing)

**Note: If you turn on Dependency parsing, word segmentation and sentence segmentation are turn off to use word segmentation and sentence segmentation from Dependency parsing.**

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
