# spaCy-PyThaiNLP
PyThaiNLP For spaCy

Work in processing...

## Install

> pip install https://github.com/PyThaiNLP/spaCy-PyThaiNLP/archive/refs/heads/main.zip

## How to use

```python
import spacy
from spacy_pythainlp.sents import *

nlp = spacy.blank("th")
# Segment the Doc into sentences
nlp.add_pipe(
   "th_sents", 
)

data=nlp("ผมเป็นคนไทย   แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน  ผมอยากไปเที่ยว")
print(list(list(data.sents)))
# output: [ผมเป็นคนไทย   แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน  , ผมอยากไปเที่ยว]
```

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