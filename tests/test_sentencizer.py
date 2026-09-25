"""
Tests for PyThaiNLPSentencizer.
"""

import shutil
import tempfile
import unittest
import spacy

from spacy_pythainlp.components import PyThaiNLPSentencizer
from spacy_pythainlp.tokenizer import PyThaiNLPTokenizer


class TestSentencizer(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("th")
        self.nlp.tokenizer = PyThaiNLPTokenizer(self.nlp.vocab)
        self.sentencizer = PyThaiNLPSentencizer(
            self.nlp, name="pythainlp_sentencizer"
        )

    def test_sentence_segmentation(self):
        text = "ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน ผมอยากไปเที่ยว"
        doc = self.nlp(text)
        doc = self.sentencizer(doc)
        sents = list(doc.sents)
        self.assertGreaterEqual(len(sents), 2)
        # Verify no token has is_sent_start as None
        for token in doc:
            self.assertIsNotNone(token.is_sent_start)
            self.assertIsInstance(token.is_sent_start, bool)

    def test_single_token(self):
        doc = self.nlp("สวัสดี")
        doc = self.sentencizer(doc)
        sents = list(doc.sents)
        self.assertEqual(len(sents), 1)
        self.assertTrue(doc[0].is_sent_start)

    def test_empty_doc(self):
        doc = self.nlp("")
        doc = self.sentencizer(doc)
        sents = list(doc.sents)
        self.assertEqual(len(sents), 0)

    def test_add_pipe(self):
        nlp = spacy.blank("th")
        nlp.tokenizer = PyThaiNLPTokenizer(nlp.vocab)
        nlp.add_pipe("pythainlp_sentencizer")
        doc = nlp("ประโยคที่หนึ่ง ประโยคที่สอง")
        self.assertGreater(len(list(doc.sents)), 0)

    def test_serialization(self):
        self.assertEqual(self.sentencizer.to_bytes(), b"")
        self.assertIs(self.sentencizer.from_bytes(b""), self.sentencizer)
        tmpdir = tempfile.mkdtemp()
        try:
            self.assertIsNone(self.sentencizer.to_disk(tmpdir))
            self.assertIs(self.sentencizer.from_disk(tmpdir), self.sentencizer)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
