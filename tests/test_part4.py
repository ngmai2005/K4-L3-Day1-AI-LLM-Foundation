"""
Checkpoint 4 (phút 230) — Part 4: Mini-project run_assistant
Chạy: pytest tests/test_part4.py -v

Tất cả API đều được mock — không cần API key thật.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


REQUIRED_KEYS = {
    "num_turns",
    "total_tokens",
    "total_cost",
    "history",
}


def _make_gemini_stream(text: str):
    """Tạo mock stream giống Gemini generate_content_stream()."""

    chunks = []

    for piece in (
        text[: len(text) // 2],
        text[len(text) // 2 :],
    ):
        chunk = MagicMock()
        chunk.text = piece
        chunks.append(chunk)

    final = MagicMock()
    final.text = None
    chunks.append(final)

    return chunks


class TestRunAssistantBasic(unittest.TestCase):

    def test_function_exists_and_is_callable(self):
        self.assertTrue(callable(MOD.run_assistant))

    @patch.object(MOD, "_get_gemini_client")
    def test_quit_immediately_returns_stats_dict(self, mock_get_client):
        mock_get_client.return_value = MagicMock()

        get_input = MagicMock(side_effect=["quit"])

        result = MOD.run_assistant(
            "Bạn là trợ lý.",
            get_input=get_input,
        )

        self.assertIsInstance(result, dict)

        for key in REQUIRED_KEYS:
            self.assertIn(key, result, f"Thiếu key: {key}")

        self.assertEqual(result["num_turns"], 0)

    @patch.object(MOD, "_get_gemini_client")
    def test_exit_is_case_insensitive(self, mock_get_client):
        mock_get_client.return_value = MagicMock()

        get_input = MagicMock(side_effect=["EXIT"])

        result = MOD.run_assistant(
            "Bạn là trợ lý.",
            get_input=get_input,
        )

        self.assertEqual(result["num_turns"], 0)

    @patch.object(MOD, "_get_gemini_client")
    def test_max_turns_zero_returns_without_reading_input(
        self,
        mock_get_client,
    ):
        mock_get_client.return_value = MagicMock()

        get_input = MagicMock(side_effect=[])

        result = MOD.run_assistant(
            "Bạn là trợ lý.",
            get_input=get_input,
            max_turns=0,
        )

        self.assertEqual(result["num_turns"], 0)
        get_input.assert_not_called()


class TestRunAssistantScenario(unittest.TestCase):
    """Demo tự động: kịch bản hội thoại nhiều lượt."""

    PERSONA = "Bạn là trợ giảng thân thiện của khóa AI."

    def _run_conversation(
        self,
        mock_get_client,
        user_messages,
        replies,
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.models.generate_content_stream.side_effect = [
            _make_gemini_stream(reply)
            for reply in replies
        ]

        get_input = MagicMock(
            side_effect=list(user_messages) + ["quit"]
        )

        result = MOD.run_assistant(
            self.PERSONA,
            get_input=get_input,
        )

        return result, mock_client

    @patch.object(MOD, "_get_gemini_client")
    def test_two_turns_counted_and_stats_positive(
        self,
        mock_get_client,
    ):
        result, _ = self._run_conversation(
            mock_get_client,
            [
                "Xin chào",
                "Kể một sự thật thú vị",
            ],
            [
                "Chào bạn, mình giúp gì được?",
                "Việt Nam có hơn 3000 km bờ biển.",
            ],
        )

        self.assertEqual(result["num_turns"], 2)
        self.assertGreater(result["total_tokens"], 0)
        self.assertGreater(result["total_cost"], 0.0)

    @patch.object(MOD, "_get_gemini_client")
    def test_api_called_with_stream_and_persona(
        self,
        mock_get_client,
    ):
        _, mock_client = self._run_conversation(
            mock_get_client,
            ["Xin chào"],
            ["Chào bạn!"],
        )

        self.assertTrue(
            mock_client.models.generate_content_stream.called
        )

        _, kwargs = (
            mock_client.models.generate_content_stream.call_args
        )

        self.assertIn("model", kwargs)
        self.assertIn("contents", kwargs)
        self.assertIn("config", kwargs)

        config = kwargs["config"]

        # Gemini dùng system_instruction trong config
        self.assertEqual(
            config.system_instruction,
            self.PERSONA,
        )

    @patch.object(MOD, "_get_gemini_client")
    def test_history_contains_last_turn(
        self,
        mock_get_client,
    ):
        result, _ = self._run_conversation(
            mock_get_client,
            [
                "Câu hỏi thứ nhất",
                "Câu hỏi thứ hai",
            ],
            [
                "Trả lời thứ nhất.",
                "Trả lời thứ hai.",
            ],
        )

        history_text = str(result["history"])

        self.assertIn("Câu hỏi thứ hai", history_text)
        self.assertIn("Trả lời thứ hai", history_text)

    @patch.object(MOD, "_get_gemini_client")
    def test_history_trimmed_to_three_turns(
        self,
        mock_get_client,
    ):
        user_messages = [
            f"Câu hỏi số {i}"
            for i in range(1, 6)
        ]

        replies = [
            f"Trả lời số {i}."
            for i in range(1, 6)
        ]

        result, _ = self._run_conversation(
            mock_get_client,
            user_messages,
            replies,
        )

        self.assertEqual(result["num_turns"], 5)

        self.assertLessEqual(
            len(result["history"]),
            6,
            "History phải được cắt còn tối đa 3 lượt.",
        )

    @patch.object(MOD, "_get_gemini_client")
    def test_max_turns_limits_conversation(
        self,
        mock_get_client,
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.models.generate_content_stream.side_effect = [
            _make_gemini_stream(f"Trả lời {i}")
            for i in range(10)
        ]

        get_input = MagicMock(
            side_effect=[
                f"Câu {i}"
                for i in range(10)
            ]
        )

        result = MOD.run_assistant(
            self.PERSONA,
            get_input=get_input,
            max_turns=2,
        )

        self.assertEqual(result["num_turns"], 2)
        self.assertEqual(
            mock_client.models.generate_content_stream.call_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()