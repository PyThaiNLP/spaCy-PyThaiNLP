"""
Tests for top-level pipeline APIs: load, blank, and all-in-one PyThaiNLP.
"""

import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
import spacy
from spacy.tokens import Doc

from spacy_pythainlp.core import blank, load


class TestPipeline(unittest.TestCase):
    def test_blank_factory(self):
        nlp = blank("th")
        self.assertEqual(nlp.lang, "th")
        doc = nlp("สวัสดีครับ วันนี้อากาศดี")
        self.assertIsInstance(doc, Doc)
        self.assertEqual(doc.text, "สวัสดีครับ วันนี้อากาศดี")

    @patch("pythainlp.tag.NER")
    def test_load_convenience(self, mock_ner_class):
        mock_ner_instance = Mock()
        mock_ner_instance.tag.return_value = [("กรุงเทพ", "B-LOCATION")]
        mock_ner_class.return_value = mock_ner_instance

        nlp = load(
            "th",
            pos=True,
            sent=True,
            ner=True,
            word_vector=False,
        )
        self.assertEqual(nlp.lang, "th")
        doc = nlp("ผมไปกรุงเทพ")
        self.assertGreater(len(doc), 0)
        self.assertGreater(len(list(doc.sents)), 0)
        self.assertNotEqual(doc[0].pos_, "")

    @patch("pythainlp.tag.NER")
    def test_multiple_docs_no_flag_mutation(self, mock_ner_class):
        mock_ner_instance = Mock()
        mock_ner_instance.tag.return_value = [("แมว", "O")]
        mock_ner_class.return_value = mock_ner_instance

        nlp = spacy.blank("th")
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": True,
                "sent": True,
                "ner": False,
                "tokenize": True,
                "dependency_parsing": False,
                "word_vector": False,
            },
        )
        component = nlp.get_pipe("pythainlp")

        # Process multiple docs
        doc1 = nlp("แมวกินปลา")
        self.assertGreater(len(doc1), 0)
        self.assertTrue(component.on_tokenize)
        self.assertTrue(component.on_sent)

        doc2 = nlp("หมาวิ่งเล่น")
        self.assertTrue(component.on_tokenize)
        self.assertTrue(component.on_sent)
        self.assertGreater(len(doc2), 0)

    def test_lemmatizer(self):
        nlp = spacy.blank("th")
        nlp.add_pipe("pythainlp_lemmatizer", config={"normalize_text": True})
        doc = nlp("แมว")
        self.assertEqual(doc[0].lemma_, "แมว")
        self.assertEqual(doc[0].norm_, "แมว")

    @patch("pythainlp.word_vector.WordVector")
    def test_word_vectors_loading(self, mock_wv_class):
        import numpy as np

        mock_wv_instance = Mock()
        mock_wv_instance.model = Mock()
        mock_wv_instance.model.vector_size = 300
        mock_wv_instance.model.key_to_index = {"แมว": 0}
        mock_wv_instance.model.__getitem__ = Mock(
            return_value=np.zeros(300, dtype=np.float32)
        )
        mock_wv_class.return_value = mock_wv_instance

        nlp = spacy.blank("th")
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": False,
                "dependency_parsing": False,
                "word_vector": True,
            },
        )
        self.assertGreater(nlp.vocab.vectors_length, 0)

    def test_save_load_disk(self):
        nlp = blank("th")
        nlp.add_pipe("pythainlp_sentencizer")
        nlp.add_pipe("pythainlp_tagger")

        tmpdir = tempfile.mkdtemp()
        try:
            nlp.to_disk(tmpdir)
            nlp2 = spacy.load(tmpdir)
            doc = nlp2("สวัสดีครับ")
            self.assertEqual(doc.text, "สวัสดีครับ")
            self.assertGreater(len(list(doc.sents)), 0)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
