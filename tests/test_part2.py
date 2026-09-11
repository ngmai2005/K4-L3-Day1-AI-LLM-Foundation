import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


class TestChatWithSystemPrompt(unittest.TestCase):

    @patch("google.genai.Client")
    def test_returns_tuple_str_float(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Đây là câu trả lời giả lập."

        mock_client.models.generate_content.return_value = mock_response

        result = MOD.chat_with_system_prompt(
            system_prompt="Bạn là trợ lý thân thiện.",
            user_prompt="Xin chào",
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        response_text, latency = result

        self.assertIsInstance(response_text, str)
        self.assertIsInstance(latency, float)
        self.assertEqual(response_text, "Đây là câu trả lời giả lập.")


    @patch("google.genai.Client")
    def test_messages_contain_system_and_user_roles(
        self,
        mock_client_class,
    ):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Câu trả lời kiểm thử."

        mock_client.models.generate_content.return_value = mock_response

        system_prompt = "Bạn là một giáo viên Python."
        user_prompt = "Giải thích biến trong Python."

        MOD.chat_with_system_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        mock_client.models.generate_content.assert_called_once()

        call_kwargs = (
            mock_client.models.generate_content.call_args.kwargs
        )

        self.assertEqual(
            call_kwargs["contents"],
            user_prompt,
        )

        config = call_kwargs["config"]

        self.assertEqual(
            config.system_instruction,
            system_prompt,
        )


    @patch("google.genai.Client")
    def test_system_prompt_content_is_sent(
        self,
        mock_client_class,
    ):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Kết quả."

        mock_client.models.generate_content.return_value = mock_response

        system_prompt = (
            "Bạn là chuyên gia an toàn thông tin. "
            "Trả lời bằng tiếng Việt."
        )

        user_prompt = "An toàn thông tin là gì?"

        MOD.chat_with_system_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        call_kwargs = (
            mock_client.models.generate_content.call_args.kwargs
        )

        config = call_kwargs["config"]

        self.assertIn(
            "an toàn thông tin",
            config.system_instruction.lower(),
        )


class TestCountTokens(unittest.TestCase):

    def test_returns_positive_int(self):
        result = MOD.count_tokens("Xin chào")

        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)


    def test_empty_text_returns_positive_int(self):
        result = MOD.count_tokens("")

        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)


class TestEstimateCost(unittest.TestCase):

    def setUp(self):
        self.input_text = (
            "Đây là nội dung đầu vào để kiểm tra số token."
        )

        self.output_text = (
            "Đây là nội dung đầu ra được tạo bởi mô hình."
        )


    def test_returns_dict_with_required_keys(self):
        result = MOD.estimate_cost(
            self.input_text,
            self.output_text,
            model="gpt-4o",
        )

        required_keys = {
            "input_tokens",
            "output_tokens",
            "input_cost",
            "output_cost",
            "total_cost",
        }

        self.assertIsInstance(result, dict)
        self.assertTrue(
            required_keys.issubset(result.keys())
        )


    def test_token_counts_are_positive_ints(self):
        result = MOD.estimate_cost(
            self.input_text,
            self.output_text,
            model="gpt-4o",
        )

        self.assertIsInstance(
            result["input_tokens"],
            int,
        )

        self.assertIsInstance(
            result["output_tokens"],
            int,
        )

        self.assertGreater(
            result["input_tokens"],
            0,
        )

        self.assertGreater(
            result["output_tokens"],
            0,
        )


    def test_total_equals_input_plus_output(self):
        result = MOD.estimate_cost(
            self.input_text,
            self.output_text,
            model="gpt-4o",
        )

        expected_total = (
            result["input_cost"]
            + result["output_cost"]
        )

        self.assertAlmostEqual(
            result["total_cost"],
            expected_total,
            places=10,
        )


    def test_mini_is_cheaper_than_gpt4o(self):
        gpt4o_result = MOD.estimate_cost(
            self.input_text,
            self.output_text,
            model="gpt-4o",
        )

        mini_result = MOD.estimate_cost(
            self.input_text,
            self.output_text,
            model="gpt-4o-mini",
        )

        self.assertLess(
            mini_result["total_cost"],
            gpt4o_result["total_cost"],
        )


if __name__ == "__main__":
    unittest.main()