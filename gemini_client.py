"""Gemini API との通信と、エラーを日本語の案内へ変換する処理。"""

import logging
import traceback
from collections.abc import Iterator

import httpx
from google import genai
from google.genai import errors, types

REQUEST_TIMEOUT_MS = 120_000
UNEXPECTED_ERROR_MESSAGE = "予期しないエラーが起きました。時間をおいて再実行してください。"

logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """画面にそのまま表示できる日本語メッセージを持つエラー。"""


def stream_text(
    api_key: str,
    model: str,
    system_instruction: str,
    prompt: str,
) -> Iterator[str]:
    """生成された文章を、届いた分から順に返す。失敗時は GeminiError を送出する。"""
    if not api_key.isascii():
        raise GeminiError("APIキーに使えない文字が含まれています。キーをコピーし直してください。")

    has_output = False
    try:
        with genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ) as client:
            stream = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
            )
            for chunk in stream:
                if chunk.text:
                    has_output = True
                    yield chunk.text
    except errors.APIError as error:
        logger.error(
            "Gemini API error: code=%s status=%s reason=%s",
            error.code,
            error.status,
            _error_reasons(error),
        )
        raise GeminiError(_describe_api_error(error)) from error
    except httpx.TransportError as error:
        _log_without_message("Gemini API connection error", error)
        raise GeminiError("Gemini APIに接続できませんでした。ネットワークを確認してください。") from error
    except Exception as error:
        _log_without_message("Unexpected error", error)
        raise GeminiError(UNEXPECTED_ERROR_MESSAGE) from error

    if not has_output:
        raise GeminiError("AIの応答が空でした。入力内容を変えて、もう一度試してください。")


def _log_without_message(summary: str, error: Exception) -> None:
    """例外の文面にはキーや入力文が含まれうるため、種類と発生箇所だけを記録する。"""
    location = "".join(traceback.format_tb(error.__traceback__))
    logger.error("%s: %s\n%s", summary, type(error).__name__, location)


def _error_reasons(error: errors.APIError) -> list[str]:
    """API の応答から、自由文ではなく機械的な理由コードだけを取り出す。"""
    body = error.details.get("error", {}) if isinstance(error.details, dict) else {}
    details = body.get("details", []) if isinstance(body, dict) else []
    return [item["reason"] for item in details if isinstance(item, dict) and "reason" in item]


def _describe_api_error(error: errors.APIError) -> str:
    code = error.code
    if code in (401, 403) or (code == 400 and "API key" in str(error)):
        return "APIキーが正しくないか、権限がありません。サイドバーのAPIキーを確認してください。"
    if code == 404:
        return "このモデルは現在使えません。サイドバーで別のモデルを選んでください。"
    if code == 429:
        return "無料枠の利用上限に達しました。時間をおいて再実行するか、別のモデルを選んでください。"
    if code is not None and code >= 500:
        return "Gemini API側で問題が起きています。時間をおいて再実行してください。"
    return f"Gemini APIのエラーです（コード: {code}）。入力内容やモデルを変えて、もう一度試してください。"
