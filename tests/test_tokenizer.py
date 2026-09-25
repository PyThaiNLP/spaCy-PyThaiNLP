"""
Tests for PyThaiNLPTokenizer.
"""

import shutil
import tempfile
import unittest
from pythainlp.util import dict_trie
import spacy
from spacy.tokens import Doc

from spacy_pythainlp.tokenizer import PyThaiNLPTokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("th")
        self.tokenizer = PyThaiNLPTokenizer(self.nlp.vocab, engine="newmm")

    def test_basic_tokenization(self):
        doc = self.tokenizer("สวัสดีครับ")
        self.assertIsInstance(doc, Doc)
        self.assertGreater(len(doc), 0)
        self.assertEqual(doc.text, "สวัสดีครับ")

    def test_whitespace_preservation(self):
        test_cases = [
            "สวัสดี ครับ 123",
            "  สวัสดี   ครับ\n",
            "ผมเป็นคนไทย แต่มะลิอยากไปโรงเรียนส่วนผมจะไปไหน ผมอยากไปเที่ยว",
            "วันที่ 15 กันยายน 2564 ทดสอบระบบที่กรุงเทพ",
        ]
        for text in test_cases:
            doc = self.tokenizer(text)
            self.assertEqual(doc.text, text)

    def test_empty_and_whitespace_only(self):
        doc_empty = self.tokenizer("")
        self.assertEqual(len(doc_empty), 0)
        self.assertEqual(doc_empty.text, "")

    def test_custom_dict(self):
        custom_words = {"สวัสดีชาวโลก", "แอนตี้กราวิตี้"}
        trie = dict_trie(dict_source=custom_words)
        tok = PyThaiNLPTokenizer(
            self.nlp.vocab, engine="newmm", custom_dict=trie
        )
        doc = tok("สวัสดีชาวโลกและแอนตี้กราวิตี้")
        token_texts = [t.text for t in doc]
        self.assertIn("สวัสดีชาวโลก", token_texts)
        self.assertIn("แอนตี้กราวิตี้", token_texts)

    def test_different_engines(self):
        for engine in ["newmm", "longest"]:
            tok = PyThaiNLPTokenizer(self.nlp.vocab, engine=engine)
            doc = tok("ทดสอบการตัดคำ")
            self.assertGreater(len(doc), 0)

    def test_serialization(self):
        bytes_data = self.tokenizer.to_bytes()
        new_tok = PyThaiNLPTokenizer(self.nlp.vocab)
        new_tok.from_bytes(bytes_data)
        self.assertEqual(new_tok.engine, self.tokenizer.engine)

        tmpdir = tempfile.mkdtemp()
        try:
            self.tokenizer.to_disk(tmpdir)
            self.tokenizer.from_disk(tmpdir)
        finally:
            shutil.rmtree(tmpdir)

    def test_registry_factory(self):
        nlp = spacy.blank(
            "th",
            config={
                "nlp": {
                    "tokenizer": {
                        "@tokenizers": "pythainlp_tokenizer",
                        "engine": "newmm",
                    }
                }
            },
        )
        doc = nlp("สวัสดีครับผม")
        self.assertIsInstance(doc, Doc)
        self.assertGreater(len(doc), 0)


if __name__ == "__main__":
    unittest.main()
