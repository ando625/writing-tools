"""Gemini API との通信処理のテスト。実際の API は呼ばない。"""

import unittest
from unittest.mock import MagicMock, patch

import httpx
from google.genai import errors

import gemini_client
from gemini_client import GeminiError, stream_text

API_KEY = "test-api-key-value"
PROMPT = "利用者が入力した秘密の文章"
GOOGLE_MESSAGE = "detail from google"


def fake_client(stream=None, error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    client.__enter__.return_value = client
    if error is not None:
        client.models.generate_content_stream.side_effect = error
    else:
        client.models.generate_content_stream.return_value = stream
    return client


def run_stream(client: MagicMock, api_key: str = API_KEY) -> list[str]:
    with patch.object(gemini_client.genai, "Client", return_value=client):
        return list(stream_text(api_key, "model", "instruction", PROMPT))


def api_error(cls: type[errors.APIError], code: int, message: str = GOOGLE_MESSAGE) -> errors.APIError:
    return cls(
        code,
        {
            "error": {
                "code": code,
                "message": message,
                "status": "SOME_STATUS",
                "details": [{"reason": "SOME_REASON"}],
            }
        },
    )


class ErrorMessageTest(unittest.TestCase):
    def message_for(self, error: Exception) -> str:
        with self.assertLogs("gemini_client", "ERROR"), self.assertRaises(GeminiError) as caught:
            run_stream(fake_client(error=error))
        return str(caught.exception)

    def test_existing_guidance_is_kept(self) -> None:
        cases = (
            (api_error(errors.ClientError, 400, "API key not valid."), "APIキー"),
            (api_error(errors.ClientError, 403), "APIキー"),
            (api_error(errors.ClientError, 404), "別のモデル"),
            (api_error(errors.ClientError, 429), "利用上限"),
            (api_error(errors.ServerError, 500), "Gemini API側"),
            (httpx.ConnectError("boom"), "接続できません"),
        )
        for error, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, self.message_for(error))

    def test_google_message_is_not_shown(self) -> None:
        message = self.message_for(api_error(errors.ClientError, 418))
        self.assertIn("418", message)
        self.assertNotIn(GOOGLE_MESSAGE, message)

    def test_unexpected_error_becomes_fixed_guidance(self) -> None:
        message = self.message_for(RuntimeError(f"internal {API_KEY} {PROMPT}"))
        self.assertEqual(message, gemini_client.UNEXPECTED_ERROR_MESSAGE)

    def test_non_ascii_api_key_stops_before_request(self) -> None:
        client = fake_client(stream=iter([]))
        with self.assertRaises(GeminiError) as caught:
            run_stream(client, api_key="日本語のキー")
        self.assertIn("使えない文字", str(caught.exception))
        client.models.generate_content_stream.assert_not_called()

    def test_failure_after_partial_output_keeps_received_text(self) -> None:
        def broken_stream():
            yield MagicMock(text="途中まで")
            raise RuntimeError("broken")

        received: list[str] = []
        with patch.object(gemini_client.genai, "Client", return_value=fake_client(stream=broken_stream())):
            with self.assertLogs("gemini_client", "ERROR"), self.assertRaises(GeminiError):
                for chunk in stream_text(API_KEY, "model", "instruction", PROMPT):
                    received.append(chunk)
        self.assertEqual(received, ["途中まで"])


class LoggingTest(unittest.TestCase):
    def logged_text(self, error: Exception) -> str:
        with self.assertLogs("gemini_client", "ERROR") as logs, self.assertRaises(GeminiError):
            run_stream(fake_client(error=error))
        return "\n".join(logs.output)

    def test_api_error_logs_only_code_status_and_reason(self) -> None:
        secret_message = f"{GOOGLE_MESSAGE} {API_KEY} {PROMPT}"
        logged = self.logged_text(api_error(errors.ClientError, 429, secret_message))
        for expected in ("429", "SOME_STATUS", "SOME_REASON"):
            self.assertIn(expected, logged)
        for secret in (GOOGLE_MESSAGE, API_KEY, PROMPT):
            self.assertNotIn(secret, logged)

    def test_unexpected_error_log_has_type_without_message(self) -> None:
        logged = self.logged_text(RuntimeError(f"internal {API_KEY} {PROMPT}"))
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(API_KEY, logged)
        self.assertNotIn(PROMPT, logged)


if __name__ == "__main__":
    unittest.main()
