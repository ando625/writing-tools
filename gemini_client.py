"""Gemini API との通信と、エラーを日本語の案内へ変換する処理。"""

from collections.abc import Iterator

import httpx
from google import genai
from google.genai import errors, types

REQUEST_TIMEOUT_MS = 120_000


class GeminiError(Exception):
    """画面にそのまま表示できる日本語メッセージを持つエラー。"""


def stream_text(
    api_key: str,
    model: str,
    system_instruction: str,
    prompt: str,
) -> Iterator[str]:
    """生成された文章を、届いた分から順に返す。失敗時は GeminiError を送出する。"""
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
        raise GeminiError(_describe_api_error(error)) from error
    except httpx.TransportError as error:
        raise GeminiError("Gemini APIに接続できませんでした。ネットワークを確認してください。") from error

    if not has_output:
        raise GeminiError("AIの応答が空でした。入力内容を変えて、もう一度試してください。")


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
    return f"Gemini APIのエラーです（コード: {code}）。{error.message or ''}".strip()
