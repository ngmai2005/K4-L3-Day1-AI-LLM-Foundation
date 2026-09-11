"""
Checkpoint 3 (phút 190) — Part 3: Streaming & độ bền
Chạy:  pytest tests/test_part3.py -v
Tất cả API đều được mock — không cần API key thật.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


def _make_stream(text: str):
    """Tạo mock stream: cắt text thành các chunk giống OpenAI streaming."""
    chunks = []
    for piece in (text[: len(text) // 2], text[len(text) // 2 :]):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = piece
        chunks.append(chunk)
    # Chunk cuối có delta.content = None — giống stream thật, code phải xử lý
    final = MagicMock()
    final.choices = [MagicMock()]
    final.choices[0].delta.content = None
    chunks.append(final)
    return chunks


class TestStreamingChatbot(unittest.TestCase):

    def test_function_exists_and_is_callable(self):
        self.assertTrue(callable(MOD.streaming_chatbot))

    @patch("builtins.input", side_effect=["quit"])
    @patch.object(MOD, "_get_gemini_client")
    def test_exits_on_quit(self, mock_get_client, mock_input):
        """Chatbot phải thoát sạch khi người dùng gõ quit."""
        MOD.streaming_chatbot()

    @patch("builtins.input", side_effect=["Xin chào", "quit"])
    @patch.object(MOD, "_get_gemini_client")
    def test_streams_one_turn_with_stream_true(
        self,
        mock_get_client,
        mock_input,
    ):
        """Một lượt chat phải gọi Gemini streaming API."""

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.text = "Chào "

        chunk2 = MagicMock()
        chunk2.text = "bạn!"

        chunk3 = MagicMock()
        chunk3.text = None

        mock_client.models.generate_content_stream.return_value = [
            chunk1,
            chunk2,
            chunk3,
        ]

        MOD.streaming_chatbot()

        self.assertTrue(
            mock_client.models.generate_content_stream.called,
            "Phải gọi Gemini API khi người dùng nhập tin nhắn",
        )

        _, kwargs = (
            mock_client.models.generate_content_stream.call_args
        )

        self.assertIn("model", kwargs)
        self.assertIn("contents", kwargs)
        self.assertIn("config", kwargs)

class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_on_first_try(self):
        result = MOD.retry_with_backoff(lambda: 42)
        self.assertEqual(result, 42)

    def test_retries_on_transient_exception(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("transient")
            return "ok"

        result = MOD.retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 2)

    def test_raises_after_max_retries(self):
        def always_fail():
            raise ValueError("permanent failure")

        with self.assertRaises(ValueError):
            MOD.retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)


if __name__ == "__main__":
    unittest.main()
