"""
Tests for spacy-pythainlp dependency parsing functionality.

This test suite verifies the fix for handling variable-length CoNLL-U format
output from PyThaiNLP's dependency_parsing function.
"""

import unittest
from unittest.mock import Mock, patch
import spacy
from spacy.tokens import Doc


class TestDependencyParsing(unittest.TestCase):
    """Test cases for dependency parsing with variable-length field tuples."""

    def setUp(self):
        """Set up test fixtures."""
        self.nlp = spacy.blank('th')

    def test_import_spacy_pythainlp(self):
        """Test that spacy_pythainlp can be imported."""
        import spacy_pythainlp.core
        self.assertIsNotNone(spacy_pythainlp.core)

    def test_add_pythainlp_pipe(self):
        """Test that pythainlp pipeline can be added."""
        import spacy_pythainlp.core
        
        # Add pipeline with minimal configuration
        self.nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": False,
                "dependency_parsing": False,
                "word_vector": False,
            }
        )
        self.assertIn("pythainlp", self.nlp.pipe_names)

    @patch('pythainlp.parse.dependency_parsing')
    def test_dependency_parsing_with_10_fields(self, mock_dep_parsing):
        """Test dependency parsing with exactly 10 fields (standard CoNLL-U)."""
        import spacy_pythainlp.core
        
        # Mock the dependency_parsing function to return 10-field tuples
        # Using head indices: token 0 points to token 1 (head=1), token 1 is root (head=0)
        mock_dep_parsing.return_value = [
            ['1', 'ฉัน', 'ฉัน', 'PRON', 'PRON', '_', '1', 'nsubj', '_', '_'],
            ['2', 'ชอบ', 'ชอบ', 'VERB', 'VERB', '_', '0', 'root', '_', '_'],
        ]
        
        nlp = spacy.blank('th')
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": False,
                "dependency_parsing": True,
                "dependency_parsing_engine": "esupar",
                "word_vector": False,
            }
        )
        
        doc = nlp("ฉันชอบ")
        
        # Verify the document was processed
        self.assertIsInstance(doc, Doc)
        self.assertEqual(len(doc), 2)
        self.assertEqual(doc[0].text, 'ฉัน')
        self.assertEqual(doc[1].text, 'ชอบ')

    @patch('pythainlp.parse.dependency_parsing')
    def test_dependency_parsing_with_11_fields(self, mock_dep_parsing):
        """Test dependency parsing with 11 fields (extra field beyond standard)."""
        import spacy_pythainlp.core
        
        # Mock the dependency_parsing function to return 11-field tuples
        # This simulates the issue reported in the bug
        mock_dep_parsing.return_value = [
            ['1', 'ฉัน', 'ฉัน', 'PRON', 'PRON', '_', '1', 'nsubj', '_', '_', 'SpaceAfter=No'],
            ['2', 'ชอบ', 'ชอบ', 'VERB', 'VERB', '_', '0', 'root', '_', '_', 'SpaceAfter=Yes'],
        ]
        
        nlp = spacy.blank('th')
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": False,
                "dependency_parsing": True,
                "dependency_parsing_engine": "esupar",
                "word_vector": False,
            }
        )
        
        # This should NOT raise ValueError anymore
        doc = nlp("ฉันชอบ")
        
        # Verify the document was processed correctly
        self.assertIsInstance(doc, Doc)
        self.assertEqual(len(doc), 2)
        self.assertEqual(doc[0].text, 'ฉัน')
        self.assertEqual(doc[1].text, 'ชอบ')

    @patch('pythainlp.parse.dependency_parsing')
    def test_dependency_parsing_with_12_fields(self, mock_dep_parsing):
        """Test dependency parsing with 12 fields (multiple extra fields)."""
        import spacy_pythainlp.core
        
        # Mock with 12 fields
        mock_dep_parsing.return_value = [
            ['1', 'ฉัน', 'ฉัน', 'PRON', 'PRON', '_', '0', 'root', '_', '_', 'extra1', 'extra2'],
        ]
        
        nlp = spacy.blank('th')
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": False,
                "dependency_parsing": True,
                "dependency_parsing_engine": "esupar",
                "word_vector": False,
            }
        )
        
        # Should handle extra fields gracefully
        doc = nlp("ฉัน")
        self.assertIsInstance(doc, Doc)
        self.assertEqual(len(doc), 1)

    @patch('pythainlp.parse.dependency_parsing')
    def test_dependency_parsing_with_insufficient_fields(self, mock_dep_parsing):
        """Test that dependency parsing raises error with fewer than 10 fields."""
        import spacy_pythainlp.core
        
        # Mock with only 9 fields (insufficient)
        mock_dep_parsing.return_value = [
            ['1', 'ฉัน', 'ฉัน', 'PRON', 'PRON', '_', '2', 'nsubj', '_'],
        ]
        
        nlp = spacy.blank('th')
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": False,
                "dependency_parsing": True,
                "dependency_parsing_engine": "esupar",
                "word_vector": False,
            }
        )
        
        # Should raise ValueError with clear message
        with self.assertRaises(ValueError) as context:
            doc = nlp("ฉัน")
        
        self.assertIn("Expected at least 10 fields", str(context.exception))

    @patch('pythainlp.parse.dependency_parsing')
    def test_dependency_parsing_pos_and_dep_tags(self, mock_dep_parsing):
        """Test that POS tags and dependency relations are correctly extracted."""
        import spacy_pythainlp.core
        
        # Mock with complete CoNLL-U data
        # Head indices: token 0 and 2 point to token 1, token 1 is root
        mock_dep_parsing.return_value = [
            ['1', 'ฉัน', 'ฉัน', 'PRON', 'PRON', '_', '1', 'nsubj', '_', '_'],
            ['2', 'ชอบ', 'ชอบ', 'VERB', 'VERB', '_', '0', 'root', '_', '_'],
            ['3', 'แมว', 'แมว', 'NOUN', 'NOUN', '_', '1', 'obj', '_', '_'],
        ]
        
        nlp = spacy.blank('th')
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": False,
                "dependency_parsing": True,
                "dependency_parsing_engine": "esupar",
                "word_vector": False,
            }
        )
        
        doc = nlp("ฉันชอบแมว")
        
        # Check POS tags
        self.assertEqual(doc[0].pos_, 'PRON')
        self.assertEqual(doc[1].pos_, 'VERB')
        self.assertEqual(doc[2].pos_, 'NOUN')
        
        # Check dependency relations
        self.assertEqual(doc[0].dep_, 'nsubj')
        self.assertEqual(doc[1].dep_, 'root')
        self.assertEqual(doc[2].dep_, 'obj')


class TestBasicFunctionality(unittest.TestCase):
    """Test basic spacy-pythainlp functionality."""

    def test_blank_model_creation(self):
        """Test that a blank Thai model can be created."""
        nlp = spacy.blank('th')
        self.assertIsNotNone(nlp)
        self.assertEqual(nlp.lang, 'th')

    def test_pipeline_with_tokenization(self):
        """Test pythainlp pipeline with tokenization enabled."""
        import spacy_pythainlp.core
        
        nlp = spacy.blank('th')
        nlp.add_pipe(
            "pythainlp",
            config={
                "pos": False,
                "sent": False,
                "ner": False,
                "tokenize": True,
                "tokenize_engine": "newmm",
                "dependency_parsing": False,
                "word_vector": False,
            }
        )
        
        doc = nlp("ผมเป็นนักศึกษา")
        self.assertIsInstance(doc, Doc)
        self.assertGreater(len(doc), 0)


if __name__ == '__main__':
    unittest.main()
