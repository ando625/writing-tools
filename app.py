"""AIライティングツールの画面。"""

import streamlit as st

from gemini_client import GeminiError, stream_text
from tools import TOOLS, InputField, WritingTool

# モデルの提供状況は変わるため、変更はここだけで済むようにしている。先頭が既定値。
MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite")
INPUT_KEY_PREFIX = "input."
API_KEY_STATE = "api_key"


def without_images(text: str) -> str:
    """AIの出力内の画像記法を文字扱いにする。画像の自動読み込みで入力内容が外部へ送られるのを防ぐ。"""
    return text.replace("![", "!\\[")


def input_key(tool: WritingTool, field_key: str) -> str:
    return f"{INPUT_KEY_PREFIX}{tool.key}.{field_key}"


def keep_inputs_across_tools() -> None:
    """表示していない機能の入力欄は Streamlit が消すため、値を保持し直す。"""
    for key in list(st.session_state):
        if isinstance(key, str) and key.startswith(INPUT_KEY_PREFIX):
            st.session_state[key] = st.session_state[key]


def queue_request(tool: WritingTool, model: str) -> None:
    """「生成する」が押されたときの受付。実際の通信は画面の描画後に行う。"""
    if not st.session_state.get(API_KEY_STATE, "").strip():
        st.session_state.errors[tool.key] = "サイドバーにGemini APIキーを入力してください。"
        return

    values = {
        field.key: st.session_state.get(input_key(tool, field.key), "")
        for field in tool.fields
    }
    missing = tool.missing_labels(values)
    if missing:
        st.session_state.errors[tool.key] = f"次の項目を入力してください: {'、'.join(missing)}"
        return

    st.session_state.errors.pop(tool.key, None)
    st.session_state.pending = {
        "tool": tool,
        "model": model,
        "prompt": tool.build_prompt(values),
    }


def render_sidebar() -> tuple[WritingTool, str, str]:
    with st.sidebar:
        st.title("✍️ AIライティング")
        api_key = st.text_input("Gemini APIキー", type="password", key=API_KEY_STATE)
        tool = st.radio("機能", TOOLS, format_func=lambda item: item.name, key="selected_tool")
        model = st.selectbox("モデル", MODELS, key="selected_model")
        st.caption("APIキー、入力、生成結果はブラウザを再読み込みすると消えます。")
    return tool, model, api_key.strip()


def render_field(tool: WritingTool, field: InputField, disabled: bool) -> None:
    key = input_key(tool, field.key)
    label = f"{field.label}（必須）" if field.required else field.label
    if field.kind == "select":
        st.selectbox(label, field.options, key=key, disabled=disabled)
    elif field.kind == "textarea":
        st.text_area(label, key=key, height=200, placeholder=field.placeholder, disabled=disabled)
    else:
        st.text_input(label, key=key, placeholder=field.placeholder, disabled=disabled)


def render_inputs(tool: WritingTool, model: str, generating: bool) -> None:
    # st.form は送信まで入力値を保持しないため、機能を切り替えても入力が残るよう使わない。
    for field in tool.fields:
        render_field(tool, field, disabled=generating)
    st.button(
        "生成する",
        type="primary",
        disabled=generating,
        on_click=queue_request,
        args=(tool, model),
    )


def run_pending_request(api_key: str) -> None:
    request = st.session_state.pending
    tool: WritingTool = request["tool"]
    text = ""
    placeholder = st.empty()
    try:
        with st.spinner("生成中です…"):
            for chunk in stream_text(
                api_key=api_key,
                model=request["model"],
                system_instruction=tool.system_instruction,
                prompt=request["prompt"],
            ):
                text += chunk
                st.session_state.results[tool.key] = text
                placeholder.markdown(without_images(text))
    except GeminiError as error:
        interrupted = "途中までの結果を表示しています。" if text else ""
        st.session_state.errors[tool.key] = f"{error} {interrupted}".strip()
    finally:
        # 生成中に別の操作で中断された場合も、ボタンが押せないままにならないようにする。
        st.session_state.pending = None
    st.rerun()


def render_result(tool: WritingTool) -> None:
    error = st.session_state.errors.get(tool.key)
    if error:
        st.error(error)

    result = st.session_state.results.get(tool.key)
    if not result:
        return

    st.subheader("生成結果")
    preview_tab, copy_tab = st.tabs(["表示", "コピー用"])
    with preview_tab:
        st.markdown(without_images(result))
    with copy_tab:
        st.code(result, language=None, wrap_lines=True)
    st.download_button(
        "ダウンロード",
        data=result,
        file_name=f"{tool.key}.md",
        mime="text/markdown",
        on_click="ignore",
    )


def main() -> None:
    st.set_page_config(page_title="AIライティング", page_icon="✍️")
    st.session_state.setdefault("results", {})
    st.session_state.setdefault("errors", {})
    st.session_state.setdefault("pending", None)
    keep_inputs_across_tools()

    tool, model, api_key = render_sidebar()
    st.header(tool.name)
    st.write(tool.description)
    if tool.notice:
        st.warning(tool.notice)

    pending = st.session_state.pending
    render_inputs(tool, model, generating=pending is not None)

    if pending is not None:
        run_pending_request(api_key)
    render_result(tool)


main()
