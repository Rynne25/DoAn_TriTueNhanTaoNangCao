"""
tests/test_bilingual_app.py
Unit and AppTest tests for Bilingual Presets and Streamlit UI Integration.
Verifies get_presets(), get_english_presets(), sidebar dual-library layout, and bilingual prediction flow.
"""

import os
import unittest
from typing import Dict, List

import app.app as demo_app


class TestBilingualApp(unittest.TestCase):
    """Tests for app/app.py dual preset libraries and bilingual integration."""

    def test_vietnamese_presets_intact(self):
        vi_presets = demo_app.get_presets()
        self.assertEqual(len(vi_presets), 10)
        for item in vi_presets:
            for k in ("id", "title", "text", "expected_sentiment", "expected_topic", "note"):
                self.assertIn(k, item)
            self.assertTrue(item["text"].strip())

    def test_english_presets_structure_and_coverage(self):
        en_presets = demo_app.get_english_presets()
        self.assertGreaterEqual(len(en_presets), 10)

        sentiments = {item["expected_sentiment"] for item in en_presets}
        topics = {item["expected_topic"] for item in en_presets}
        self.assertEqual(sentiments, {"negative", "neutral", "positive"})
        self.assertEqual(topics, {"lecturer", "training_program", "facility", "others"})

        ids = [item["id"] for item in en_presets]
        texts = [item["text"].strip() for item in en_presets]
        self.assertEqual(len(ids), len(set(ids)), "English preset IDs must be unique")
        self.assertEqual(len(texts), len(set(texts)), "English preset texts must be unique")

    def test_presets_no_overlap(self):
        vi_texts = {item["text"].strip().lower() for item in demo_app.get_presets()}
        en_texts = {item["text"].strip().lower() for item in demo_app.get_english_presets()}
        self.assertEqual(len(vi_texts & en_texts), 0, "VI and EN presets must not overlap")

    def test_apptest_sidebar_dual_libraries_and_loading(self):
        from streamlit.testing.v1 import AppTest

        app_path = os.path.join(demo_app.PROJECT_ROOT, "app", "app.py")
        at = AppTest.from_file(app_path, default_timeout=25).run()
        self.assertFalse(at.exception)

        # At least 2 selectboxes and 2 buttons in sidebar
        self.assertGreaterEqual(len(at.sidebar.selectbox), 2)
        self.assertGreaterEqual(len(at.sidebar.button), 2)

        vi_presets = demo_app.get_presets()
        en_presets = demo_app.get_english_presets()

        # Test loading Vietnamese preset (selectbox[0] + button[0])
        at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[1]).run()
        at.sidebar.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.text_area[0].value, vi_presets[1]["text"])

        # Test loading English preset (selectbox[1] + button[1])
        at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[0]).run()
        at.sidebar.button[1].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.text_area[0].value, en_presets[0]["text"])


if __name__ == "__main__":
    unittest.main()
