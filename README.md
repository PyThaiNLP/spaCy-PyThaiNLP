# spaCy-PyThaiNLP

[![PyPI version](https://img.shields.io/pypi/v/spacy-pythainlp.svg)](https://pypi.org/project/spacy-pythainlp/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This package wraps the [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) library to add Thai language support for [spaCy](https://spacy.io/).

## Features

- **Word Tokenization**: Custom `PyThaiNLPTokenizer` preserving exact whitespace and token offsets, supporting custom dictionaries and engines (`newmm`, `longest`, `attacut`, etc.).
- **Sentence Segmentation**: Boundary detection preserving token integrity via `PyThaiNLPSentencizer` and engines like `crfcut`, `whitespace`, `thaisum`.
- **Part-of-Speech Tagging**: Supports Universal Dependencies tags (`token.pos_`) and fine-grained tags (`token.tag_`) across corpora (`orchid_ud`, `pud`, `blackboard_ud`, `tdtb`, `tud`, `orchid`).
- **Named Entity Recognition**: Flat NER (`token.ents`) and Nested NER (`doc.spans`) via `PyThaiNLPNER`.
- **Dependency Parsing**: Integration with PyThaiNLP's dependency parsers (`esupar`, etc.).
- **Word Vectors**: Word vector support via `thai2fit_wv` and other models.
- **Text Normalization / Lemmatization**: Thai text normalization via `PyThaiNLPLemmatizer`.
- **Flexible Pipeline Architecture**: Use the one-line `spacy_pythainlp.load()`, the blank model `spacy_pythainlp.blank()`, the all-in-one `pythainlp` pipe, or individual modular components.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tokenizer & Custom Dictionaries](#tokenizer--custom-dictionaries)
- [Modular Components](#modular-components)
- [Usage Examples](#usage-examples)
  - [Sentence Segmentation](#sentence-segmentation)
  - [Part-of-Speech Tagging](#part-of-speech-tagging)
  - [Named Entity Recognition](#named-entity-recognition)
  - [Dependency Parsing](#dependency-parsing)
  - [Word Vectors](#word-vectors)
  - [Lemmatization and Text Normalization](#lemmatization-and-text-normalization)
- [All-in-One Configuration](#all-in-one-configuration)
- [License](#license)

## Installation

### Prerequisites

- Python 3.9 or higher
- spaCy 3.0 or higher
- PyThaiNLP 3.1.0 or higher

### Install via pip

```bash
pip install spacy-pythainlp
```

## Quick Start

### One-line Pipeline Loader

The easiest way to get started is with `spacy_pythainlp.load()`:

```python
import spacy_pythainlp

# Load Thai model with tokenizer, sentence segmentation, POS tagging, and NER
nlp = spacy_pythainlp.load()

doc = nlp("ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน ผมอยากไปเที่ยว")

# Access sentences
for sent in doc.sents:
    print(sent.text)

# Access tokens and POS tags
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")
```

### Standard spaCy Pipeline Setup

You can also add the `pythainlp` component to a blank model:

```python
import spacy
import spacy_pythainlp

nlp = spacy.blank("th")
nlp.add_pipe("pythainlp")

doc = nlp("ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียน")
```

## Tokenizer & Custom Dictionaries

`PyThaiNLPTokenizer` preserves exact whitespace and character offsets, making `doc.text` identical to the input text.

```python
import spacy
from spacy_pythainlp import PyThaiNLPTokenizer
from pythainlp.util import dict_trie

nlp = spacy.blank("th")

# Use a custom dictionary
custom_words = {"แอนตี้กราวิตี้", "ภาษาไทย"}
trie = dict_trie(dict_source=custom_words)

nlp.tokenizer = PyThaiNLPTokenizer(nlp.vocab, engine="newmm", custom_dict=trie)
doc = nlp("แอนตี้กราวิตี้และการประมวลผลภาษาไทย")

print([token.text for token in doc])
```

You can also create a blank model directly:

```python
import spacy_pythainlp

nlp = spacy_pythainlp.blank("th", tokenize_engine="newmm")
doc = nlp("สวัสดีครับ วันนี้อากาศดี")
```

And in spaCy config files:

```ini
[nlp]
lang = "th"

[nlp.tokenizer]
@tokenizers = "pythainlp_tokenizer"
engine = "newmm"
```

## Modular Components

Instead of enabling or disabling features in a single component, you can add individual modular components to any spaCy pipeline:

```python
import spacy
import spacy_pythainlp

nlp = spacy_pythainlp.blank("th")

# Add only sentence segmentation
nlp.add_pipe("pythainlp_sentencizer", config={"engine": "crfcut"})

# Add only POS tagging
nlp.add_pipe("pythainlp_tagger", config={"corpus": "orchid_ud"})

# Add only NER
nlp.add_pipe("pythainlp_ner", config={"engine": "thainer"})

# Add only lemmatization / text normalization
nlp.add_pipe("pythainlp_lemmatizer", config={"normalize_text": True})

doc = nlp("วันที่ 15 กันยายน 2564 ทดสอบระบบที่กรุงเทพ")
```

Available component factories:
- `"pythainlp_sentencizer"`: Sentence segmentation
- `"pythainlp_tagger"`: Part-of-speech tagging
- `"pythainlp_ner"`: Named entity recognition
- `"pythainlp_parser"`: Dependency parsing
- `"pythainlp_vectors"`: Word vectors
- `"pythainlp_lemmatizer"`: Lemmatization and text normalization
- `"pythainlp"`: All-in-one component (backward compatible)

## Usage Examples

### Sentence Segmentation

```python
import spacy_pythainlp

nlp = spacy_pythainlp.load(sent=True, pos=False, ner=False)

doc = nlp("ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน ผมอยากไปเที่ยว")

for i, sent in enumerate(doc.sents, 1):
    print(f"Sentence {i}: {sent.text}")
```

### Part-of-Speech Tagging

```python
import spacy_pythainlp

nlp = spacy_pythainlp.load(pos=True, pos_corpus="orchid_ud")

doc = nlp("ผมเป็นคนไทย")

for token in doc:
    print(f"{token.text}: UPOS={token.pos_}, TAG={token.tag_}")
```

### Named Entity Recognition

```python
import spacy_pythainlp

nlp = spacy_pythainlp.load(ner=True, ner_engine="thainer")

doc = nlp("วันที่ 15 กันยายน 2564 ทดสอบระบบที่กรุงเทพ")

for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

### Dependency Parsing

```python
import spacy_pythainlp

nlp = spacy_pythainlp.load(dependency_parsing=True, dependency_parsing_engine="esupar")

doc = nlp("ผมเป็นคนไทย")

for token in doc:
    print(f"{token.text}: {token.dep_} <- {token.head.text}")
```

### Word Vectors

```python
import spacy_pythainlp

nlp = spacy_pythainlp.load(word_vector=True, word_vector_model="thai2fit_wv")

doc = nlp("แมว สุนัข")

token1 = doc[0]  # แมว
token2 = doc[1]  # สุนัข
print(f"Similarity: {token1.similarity(token2)}")
```

### Lemmatization and Text Normalization

```python
import spacy
import spacy_pythainlp

nlp = spacy_pythainlp.blank("th")
nlp.add_pipe("pythainlp_lemmatizer")

doc = nlp("สระ  เ เ ม ว")
for token in doc:
    print(f"{token.text} -> lemma: {token.lemma_}, norm: {token.norm_}")
```

## All-in-One Configuration

You can customize the `pythainlp` pipeline component with `config`:

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
        "word_vector": True,
        "word_vector_model": "thai2fit_wv"
    }
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenize` | `bool` | `False` | Enable/disable word tokenization in component |
| `tokenize_engine` | `str` | `"newmm"` | Tokenization engine (`newmm`, `longest`, `attacut`, `deepcut`, etc.) |
| `sent` | `bool` | `True` | Enable/disable sentence segmentation |
| `sent_engine` | `str` | `"crfcut"` | Sentence tokenizer engine (`crfcut`, `whitespace`, `whitespace+newline`, `thaisum`, `tltk`, `wtp`) |
| `pos` | `bool` | `True` | Enable/disable part-of-speech tagging |
| `pos_engine` | `str` | `"perceptron"` | POS tagging engine (`perceptron`, `unigram`, `tltk`, `wangchanberta`) |
| `pos_corpus` | `str` | `"orchid_ud"` | Corpus for POS tagging (`orchid_ud`, `pud`, `blackboard_ud`, `tdtb`, `tud`, `orchid`, `blackboard`) |
| `ner` | `bool` | `True` | Enable/disable named entity recognition |
| `ner_engine` | `str` | `"thainer"` | NER engine (`thainer`, `thainer-v2`, `phayathaibert`, `wangchanberta`, `thai-nner`) |
| `dependency_parsing` | `bool` | `False` | Enable/disable dependency parsing |
| `dependency_parsing_engine` | `str` | `"esupar"` | Dependency parsing engine (`esupar`, `spacy_thai`, `transformers_ud`, `ud_goeswith`, `attaparse`) |
| `dependency_parsing_model` | `str` | `None` | Dependency parsing model |
| `word_vector` | `bool` | `True` | Enable/disable word vectors |
| `word_vector_model` | `str` | `"thai2fit_wv"` | Word vector model (`thai2fit_wv`, etc.) |

## Resources

- [PyThaiNLP Documentation](https://pythainlp.github.io/)
- [spaCy Documentation](https://spacy.io/)
- [GitHub Repository](https://github.com/PyThaiNLP/spaCy-PyThaiNLP)
- [Issue Tracker](https://github.com/PyThaiNLP/spaCy-PyThaiNLP/issues)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

```
   Copyright 2016-2026 PyThaiNLP Project

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
