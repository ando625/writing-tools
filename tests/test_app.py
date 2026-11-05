"""画面の結果表示のテスト。Gemini API は呼ばない。"""

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
IMAGE_SYNTAX = "!["


def generate(output: str) -> AppTest:
    """AIの出力を output に差し替えて、ブログ記事の生成まで操作する。"""

    def fake_stream_text(**_: str):
        yield output

    with patch("gemini_client.stream_text", fake_stream_text):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        app.text_input(key="api_key").input("dummy").run()
        app.text_input(key="input.blog.topic").input("テーマ")
        return app.button[0].click().run()


def displayed_text(app: AppTest) -> str:
    return "\n".join(item.value for item in app.markdown)


class ResultDisplayTest(unittest.TestCase):
    def test_image_syntax_is_not_rendered(self) -> None:
        patterns = (
            "![a](https://example.test/x.png)",
            "\\![a](https://example.test/x.png)",
            "\\\\![a](https://example.test/x.png)",
            "![a][ref]\n\n[ref]: https://example.test/x.png",
            "[![a](https://example.test/x.png)](https://example.test/)",
        )
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                app = generate(pattern)
                self.assertFalse(app.exception)
                self.assertNotIn(IMAGE_SYNTAX, displayed_text(app))

    def test_normal_markdown_is_kept(self) -> None:
        output = "# 見出し\n\n**太字**と[リンク](https://example.test/)、感嘆符!だけ。"
        self.assertIn(output, displayed_text(generate(output)))

    def test_copy_text_keeps_original(self) -> None:
        output = "![a](https://example.test/x.png)"
        self.assertEqual(generate(output).code[0].value, output)


if __name__ == "__main__":
    unittest.main()
