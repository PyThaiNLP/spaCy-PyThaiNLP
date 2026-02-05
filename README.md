# spaCy-PyThaiNLP

[![PyPI version](https://img.shields.io/pypi/v/spacy-pythainlp.svg)](https://pypi.org/project/spacy-pythainlp/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This package wraps the [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) library to add Thai language support for [spaCy](https://spacy.io/).

## Features

**Support List**
- Word segmentation (tokenization)
- Part-of-speech tagging
- Named entity recognition (NER)
- Sentence segmentation
- Dependency parsing
- Word vectors

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Basic Sentence Segmentation](#basic-sentence-segmentation)
  - [Part-of-Speech Tagging](#part-of-speech-tagging)
  - [Named Entity Recognition](#named-entity-recognition)
  - [Dependency Parsing](#dependency-parsing)
  - [Word Vectors](#word-vectors)
- [Configuration](#configuration)
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

```python
import spacy
import spacy_pythainlp.core

# Create a blank Thai language model
nlp = spacy.blank("th")

# Add the PyThaiNLP pipeline component
nlp.add_pipe("pythainlp")

# Process text
doc = nlp("ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน ผมอยากไปเที่ยว")

# Access sentences
for sent in doc.sents:
    print(sent)
# Output:
# ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน
# ผมอยากไปเที่ยว
```

## Usage Examples

### Basic Sentence Segmentation

```python
import spacy
import spacy_pythainlp.core

nlp = spacy.blank("th")
nlp.add_pipe("pythainlp")

doc = nlp("ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน ผมอยากไปเที่ยว")

# Get sentences
sentences = list(doc.sents)
print(f"Number of sentences: {len(sentences)}")
for i, sent in enumerate(sentences, 1):
    print(f"Sentence {i}: {sent.text}")
```

### Part-of-Speech Tagging

```python
import spacy
import spacy_pythainlp.core

nlp = spacy.blank("th")
nlp.add_pipe("pythainlp", config={"pos": True})

doc = nlp("ผมเป็นคนไทย")

# Print tokens with POS tags
for token in doc:
    print(f"{token.text}: {token.pos_}")
```

### Named Entity Recognition

```python
import spacy
import spacy_pythainlp.core

nlp = spacy.blank("th")
nlp.add_pipe("pythainlp", config={"ner": True})

doc = nlp("วันที่ 15 กันยายน 2564 ทดสอบระบบที่กรุงเทพ")

# Print named entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

### Dependency Parsing

```python
import spacy
import spacy_pythainlp.core

nlp = spacy.blank("th")
nlp.add_pipe("pythainlp", config={"dependency_parsing": True})

doc = nlp("ผมเป็นคนไทย")

# Print dependency relations
for token in doc:
    print(f"{token.text}: {token.dep_} <- {token.head.text}")
```

### Word Vectors

```python
import spacy
import spacy_pythainlp.core

nlp = spacy.blank("th")
nlp.add_pipe("pythainlp", config={"word_vector": True, "word_vector_model": "thai2fit_wv"})

doc = nlp("แมว สุนัข")

# Access word vectors
for token in doc:
    print(f"{token.text}: vector shape = {token.vector.shape}")
    
# Calculate similarity
token1 = doc[0]  # แมว
token2 = doc[1]  # สุนัข
print(f"Similarity: {token1.similarity(token2)}")
```

## Configuration

You can customize the PyThaiNLP pipeline component by passing a configuration dictionary to `nlp.add_pipe()`:

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
| `tokenize` | `bool` | `False` | Enable/disable word tokenization (spaCy uses PyThaiNLP's newmm by default) |
| `tokenize_engine` | `str` | `"newmm"` | Tokenization engine. [See options](https://pythainlp.github.io/docs/3.1/api/tokenize.html#pythainlp.tokenize.word_tokenize) |
| `sent` | `bool` | `True` | Enable/disable sentence segmentation |
| `sent_engine` | `str` | `"crfcut"` | Sentence tokenizer engine. [See options](https://pythainlp.github.io/docs/3.1/api/tokenize.html#pythainlp.tokenize.sent_tokenize) |
| `pos` | `bool` | `True` | Enable/disable part-of-speech tagging |
| `pos_engine` | `str` | `"perceptron"` | POS tagging engine. [See options](https://pythainlp.github.io/docs/3.1/api/tag.html#pythainlp.tag.pos_tag) |
| `pos_corpus` | `str` | `"orchid_ud"` | Corpus for POS tagging |
| `ner` | `bool` | `True` | Enable/disable named entity recognition |
| `ner_engine` | `str` | `"thainer"` | NER engine. [See options](https://pythainlp.github.io/docs/3.1/api/tag.html#pythainlp.tag.NER) |
| `dependency_parsing` | `bool` | `False` | Enable/disable dependency parsing |
| `dependency_parsing_engine` | `str` | `"esupar"` | Dependency parsing engine. [See options](https://pythainlp.github.io/docs/3.1/api/parse.html#pythainlp.parse.dependency_parsing) |
| `dependency_parsing_model` | `str` | `None` | Dependency parsing model. [See options](https://pythainlp.github.io/docs/3.1/api/parse.html#pythainlp.parse.dependency_parsing) |
| `word_vector` | `bool` | `True` | Enable/disable word vectors |
| `word_vector_model` | `str` | `"thai2fit_wv"` | Word vector model. [See options](https://pythainlp.github.io/docs/3.1/api/word_vector.html#pythainlp.word_vector.WordVector) |

**Important Notes:**
- When `dependency_parsing` is enabled, word segmentation and sentence segmentation are automatically disabled to use the tokenization from the dependency parser.
- All configuration options are optional and have sensible defaults.

## Resources

- [PyThaiNLP Documentation](https://pythainlp.github.io/)
- [spaCy Documentation](https://spacy.io/)
- [GitHub Repository](https://github.com/PyThaiNLP/spaCy-PyThaiNLP)
- [Issue Tracker](https://github.com/PyThaiNLP/spaCy-PyThaiNLP/issues)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

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
