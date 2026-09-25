"""
Tests for PyThaiNLPTagger.
"""

import shutil
import tempfile
import unittest
import spacy

from spacy_pythainlp.components import PyThaiNLPTagger, UPOS_TAGS
from spacy_pythainlp.tokenizer import PyThaiNLPTokenizer


class TestTagger(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("th")
        self.nlp.tokenizer = PyThaiNLPTokenizer(self.nlp.vocab)
        self.tagger = PyThaiNLPTagger(
            self.nlp, name="pythainlp_tagger", corpus="orchid_ud"
        )

    def test_pos_tagging_ud(self):
        doc = self.nlp("ผมเป็นคนไทย")
        doc = self.tagger(doc)
        for token in doc:
            self.assertNotEqual(token.pos_, "")
            self.assertIn(token.pos_, UPOS_TAGS)
            self.assertNotEqual(token.tag_, "")

    def test_pos_tagging_orchid_non_ud(self):
        orchid_tagger = PyThaiNLPTagger(
            self.nlp, name="orchid_tagger", corpus="orchid"
        )
        doc = self.nlp("เก้าอี้มีจำนวนขา3")
        # Should not raise ValueError [E1021] for non-UD tags
        doc = orchid_tagger(doc)
        for token in doc:
            self.assertNotEqual(token.tag_, "")

    def test_empty_doc(self):
        doc = self.nlp("")
        doc = self.tagger(doc)
        self.assertEqual(len(doc), 0)

    def test_add_pipe(self):
        nlp = spacy.blank("th")
        nlp.tokenizer = PyThaiNLPTokenizer(nlp.vocab)
        nlp.add_pipe("pythainlp_tagger", config={"corpus": "pud"})
        doc = nlp("สวัสดีครับ")
        self.assertGreater(len(doc), 0)

    def test_serialization(self):
        self.assertEqual(self.tagger.to_bytes(), b"")
        self.assertIs(self.tagger.from_bytes(b""), self.tagger)
        tmpdir = tempfile.mkdtemp()
        try:
            self.assertIsNone(self.tagger.to_disk(tmpdir))
            self.assertIs(self.tagger.from_disk(tmpdir), self.tagger)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
