"""
Tests for PyThaiNLPParser modular component.
"""

import shutil
import tempfile
import unittest
from unittest.mock import patch
import spacy
from spacy.tokens import Doc

from spacy_pythainlp.components import PyThaiNLPParser


class TestParser(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("th")

    @patch("pythainlp.parse.dependency_parsing")
    def test_parser_component(self, mock_dep_parsing):
        mock_dep_parsing.return_value = [
            ["1", "ฉัน", "ฉัน", "PRON", "PRON", "_", "1",
             "nsubj", "_", "SpaceAfter=No"],
            ["2", "ชอบ", "ชอบ", "VERB", "VERB", "_", "0",
             "root", "_", "SpaceAfter=No"],
            ["3", "แมว", "แมว", "NOUN", "NOUN", "_", "1",
             "obj", "_", "SpaceAfter=No"],
        ]

        parser = PyThaiNLPParser(self.nlp, name="pythainlp_parser")
        doc = self.nlp("ฉันชอบแมว")
        doc = parser(doc)

        self.assertIsInstance(doc, Doc)
        self.assertEqual(len(doc), 3)
        self.assertEqual(doc[0].dep_, "nsubj")
        self.assertEqual(doc[1].dep_, "root")
        self.assertEqual(doc[2].dep_, "obj")

    def test_serialization(self):
        parser = PyThaiNLPParser(self.nlp, name="pythainlp_parser")
        self.assertEqual(parser.to_bytes(), b"")
        self.assertIs(parser.from_bytes(b""), parser)
        tmpdir = tempfile.mkdtemp()
        try:
            self.assertIsNone(parser.to_disk(tmpdir))
            self.assertIs(parser.from_disk(tmpdir), parser)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
