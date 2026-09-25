# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-25

### Added
- **`PyThaiNLPTokenizer`**: Custom spaCy tokenizer class utilizing PyThaiNLP's `word_tokenize` with proper whitespace and character offset preservation using `spacy.util.get_words_and_spaces`.
- **spaCy Tokenizer Registry**: Registered `@registry.tokenizers("pythainlp_tokenizer")` for declarative tokenizer configuration in spaCy config files (`config.cfg`).
- **Custom Dictionary Support**: Added `custom_dict` support in `PyThaiNLPTokenizer` for custom words and Trie dictionaries.
- **Convenience Top-Level APIs**:
  - `spacy_pythainlp.load()`: Quick one-liner to initialize and configure a Thai spaCy pipeline.
  - `spacy_pythainlp.blank()`: Creates a blank Thai model preconfigured with `PyThaiNLPTokenizer`.
- **Modular Pipeline Components**:
  - `pythainlp_sentencizer`: Standalone sentence boundary segmentation component.
  - `pythainlp_tagger`: Standalone part-of-speech tagger supporting Universal POS tags (`token.pos_`) and fine-grained tags (`token.tag_`).
  - `pythainlp_ner`: Standalone named entity recognition component supporting flat entities (`doc.ents`) and nested entities (`doc.spans`).
  - `pythainlp_parser`: Standalone dependency parsing component.
  - `pythainlp_vectors`: Standalone word vectors loading component.
  - `pythainlp_lemmatizer`: Standalone text normalization and lemmatization component.
- **Entry Points**: Registered `spacy_factories` and `spacy_tokenizers` in `setup.py` for automatic discovery by spaCy.
- **Test Suite**: Added comprehensive test coverage for tokenizer, sentence segmentation, POS tagging, NER, dependency parsing, lemmatization, and disk serialization (`to_disk` / `from_disk`).

### Changed
- Updated API compatibility for modern spaCy (3.x / 3.8+) and PyThaiNLP (5.x).
- Handled Universal POS tags (`token.pos_`) and fine-grained tags (`token.tag_`) separately to prevent `ValueError: [E1021]` in spaCy 3.8+ when using non-UD corpora like `orchid` or `blackboard`.
- Replaced deprecated `Doc.is_sentenced` with `Doc.has_annotation("SENT_START")`.
- Updated word vector loading to use `wv.model.vector_size` for compatibility across gensim 4.x models.
- Updated documentation in `README.md` with guides for the new APIs, modular components, and configuration tables.

### Fixed
- Fixed sentence boundary segmentation bug where premature loop exit left the final token with `token.is_sent_start = None`.
- Fixed NER bug on un-sentenced documents that processed tokens individually, losing sequence context.
- Fixed IOB entity span parsing bug where unclosed entity spans caused `TypeError` in `doc.char_span`.
- Fixed component state mutation bug where `on_tokenize` and `on_sent` flags were permanently mutated to `False` on the component instance during `__call__`.
- Fixed dependency parsing head index mapping for ROOT (`head == 0 -> token.i`) and space preservation for `SpaceAfter=No`.

## [1.0.0] - 2026-02-05

### Added
- Added unit test suite and GitHub Actions workflow for testing across Python 3.10, 3.11, and 3.12.
- Added type hints and docstrings across `core.py`.
- Updated documentation, badges, and examples in `README.md`.

### Changed
- Updated minimum Python requirement to 3.9+.

### Fixed
- Fixed `ValueError` in dependency parsing when handling variable-length CoNLL-U field tuples from PyThaiNLP.

## [0.1.0] - 2023-01-03

### Added
- Initial release of `spacy-pythainlp`.
- Integrated PyThaiNLP pipeline component for spaCy (`pythainlp`).
- Support for word tokenization, POS tagging, NER, sentence segmentation, dependency parsing, and word vectors.
