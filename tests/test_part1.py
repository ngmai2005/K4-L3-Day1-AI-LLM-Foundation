"""
Checkpoint 1 (phút 100) — Part 1: API cơ bản
Chạy: pytest tests/test_part1.py -v
Tất cả API đều được mock — không cần API key thật.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


def _make_gemini_response(text: str = "Hello from Gemini"):
    """Tạo mock tối thiểu giống Gemini response."""
    resp = MagicMock()
    resp.text = text
    return resp


class TestCallOpenAI(unittest.TestCase):

    @patch("google.genai.Client")
    def test_returns_non_empty_string(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_client.models.generate_content.return_value = (
            _make_gemini_response("Test response")
        )

        result, latency = MOD.call_openai("Hello")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("google.genai.Client")
    def test_latency_is_positive_float(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_client.models.generate_content.return_value = (
            _make_gemini_response()
        )

        _, latency = MOD.call_openai("Hello")

        self.assertIsInstance(latency, float)
        self.assertGreater(latency, 0.0)

    @patch("google.genai.Client")
    def test_returns_tuple_of_two(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_client.models.generate_content.return_value = (
            _make_gemini_response()
        )

        result = MOD.call_openai("Hello")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestCallOpenAIMini(unittest.TestCase):

    @patch("google.genai.Client")
    def test_returns_non_empty_string(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_client.models.generate_content.return_value = (
            _make_gemini_response("Test response")
        )

        result, latency = MOD.call_openai_mini("Hello")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("google.genai.Client")
    def test_uses_mini_model(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_client.models.generate_content.return_value = (
            _make_gemini_response()
        )

        MOD.call_openai_mini("Hello")

        mock_client.models.generate_content.assert_called_once()

        _, kwargs = mock_client.models.generate_content.call_args

        self.assertEqual(
            kwargs.get("model"),
            MOD.GEMINI_MINI_MODEL
        )

    @patch("google.genai.Client")
    def test_returns_tuple_of_two(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_client.models.generate_content.return_value = (
            _make_gemini_response()
        )

        result = MOD.call_openai_mini("Hello")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestCompareModels(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        with patch.object(
            MOD,
            "call_openai",
            return_value=("Gemini answer", 0.5)
        ), patch.object(
            MOD,
            "call_openai_mini",
            return_value=("Mini answer", 0.3)
        ):
            result = MOD.compare_models("Test prompt")

        required_keys = {
            "gpt4o_response",
            "mini_response",
            "gpt4o_latency",
            "mini_latency",
            "gpt4o_cost_estimate",
        }

        self.assertIsInstance(result, dict)

        for key in required_keys:
            self.assertIn(key, result, f"Thiếu key: {key}")

    def test_latency_values_are_positive(self):
        with patch.object(
            MOD,
            "call_openai",
            return_value=("Gemini answer", 0.5)
        ), patch.object(
            MOD,
            "call_openai_mini",
            return_value=("Mini answer", 0.3)
        ):
            result = MOD.compare_models("Test prompt")

        self.assertGreater(result["gpt4o_latency"], 0)
        self.assertGreater(result["mini_latency"], 0)

    def test_responses_are_non_empty_strings(self):
        with patch.object(
            MOD,
            "call_openai",
            return_value=("Gemini answer", 0.5)
        ), patch.object(
            MOD,
            "call_openai_mini",
            return_value=("Mini answer", 0.3)
        ):
            result = MOD.compare_models("Test prompt")

        self.assertIsInstance(result["gpt4o_response"], str)
        self.assertGreater(len(result["gpt4o_response"]), 0)

        self.assertIsInstance(result["mini_response"], str)
        self.assertGreater(len(result["mini_response"]), 0)

    def test_cost_estimate_is_non_negative(self):
        with patch.object(
            MOD,
            "call_openai",
            return_value=("word " * 100, 0.5)
        ), patch.object(
            MOD,
            "call_openai_mini",
            return_value=("word " * 100, 0.3)
        ):
            result = MOD.compare_models("Test prompt")

        self.assertGreaterEqual(
            result["gpt4o_cost_estimate"],
            0
        )


if __name__ == "__main__":
    unittest.main()