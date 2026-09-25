"""
Tests for PyThaiNLPNER.
"""

import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
import spacy

from spacy_pythainlp.components import PyThaiNLPNER
from spacy_pythainlp.tokenizer import PyThaiNLPTokenizer


class TestNER(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("th")
        self.nlp.tokenizer = PyThaiNLPTokenizer(self.nlp.vocab)

    @patch("pythainlp.tag.NER")
    def test_ner_unsentenced(self, mock_ner_class):
        mock_ner_instance = Mock()
        mock_ner_instance.tag.return_value = [
            ("วันที่", "O"),
            (" ", "O"),
            ("15", "B-DATE"),
            (" ", "I-DATE"),
            ("กันยายน", "I-DATE"),
            (" ", "I-DATE"),
            ("2564", "I-DATE"),
            (" ", "O"),
            ("ทดสอบระบบ", "O"),
            ("ที่", "O"),
            ("กรุงเทพ", "B-LOCATION"),
        ]
        mock_ner_class.return_value = mock_ner_instance

        ner_comp = PyThaiNLPNER(self.nlp, name="pythainlp_ner")
        doc = self.nlp("วันที่ 15 กันยายน 2564 ทดสอบระบบที่กรุงเทพ")
        doc = ner_comp(doc)

        ent_labels = [e.label_ for e in doc.ents]
        self.assertIn("DATE", ent_labels)
        self.assertIn("LOCATION", ent_labels)

    @patch("pythainlp.tag.NER")
    def test_ner_sentenced(self, mock_ner_class):
        mock_ner_instance = Mock()
        mock_ner_instance.tag.return_value = [
            ("กรุงเทพ", "B-LOCATION"),
        ]
        mock_ner_class.return_value = mock_ner_instance

        ner_comp = PyThaiNLPNER(self.nlp, name="pythainlp_ner")
        doc = self.nlp("กรุงเทพ")
        doc[0].is_sent_start = True
        doc = ner_comp(doc)

        self.assertEqual(len(doc.ents), 1)
        self.assertEqual(doc.ents[0].label_, "LOCATION")

    def test_empty_doc(self):
        with patch("pythainlp.tag.NER"):
            ner_comp = PyThaiNLPNER(self.nlp, name="pythainlp_ner")
            doc = self.nlp("")
            doc = ner_comp(doc)
            self.assertEqual(len(doc.ents), 0)

    @patch("pythainlp.tag.NNER")
    def test_nested_ner(self, mock_nner_class):
        mock_nner_instance = Mock()
        mock_nner_instance.tag.return_value = (
            ["แมว", "ตอน", "ห้า", "โมง"],
            [
                {"text": ["ห้า"], "span": [2, 3], "entity_type": "cardinal"},
                {"text": ["ห้า", "โมง"], "span": [2, 4],
                 "entity_type": "time"},
            ],
        )
        mock_nner_class.return_value = mock_nner_instance

        ner_comp = PyThaiNLPNER(self.nlp, name="pythainlp_nner", nested=True)
        doc = self.nlp("แมวตอนห้าโมง")
        doc = ner_comp(doc)

        self.assertIn("pythainlp_nner", doc.spans)
        self.assertGreater(len(doc.spans["pythainlp_nner"]), 0)

    def test_serialization(self):
        with patch("pythainlp.tag.NER"):
            ner_comp = PyThaiNLPNER(self.nlp, name="pythainlp_ner")
            self.assertEqual(ner_comp.to_bytes(), b"")
            self.assertIs(ner_comp.from_bytes(b""), ner_comp)
            tmpdir = tempfile.mkdtemp()
            try:
                self.assertIsNone(ner_comp.to_disk(tmpdir))
                self.assertIs(ner_comp.from_disk(tmpdir), ner_comp)
            finally:
                shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
