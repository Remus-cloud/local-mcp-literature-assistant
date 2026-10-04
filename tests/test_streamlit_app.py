"""Smoke tests for the Streamlit user interface."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from literature_bot.streamlit_app import format_error


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "literature_bot"
    / "streamlit_app.py"
)


def test_streamlit_initial_page_and_quick_actions_render():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "📚 文献搜索与问答机器人"
    button_labels = [button.label for button in app.button]
    assert "🔎 搜索文献" in button_labels
    assert "📄 分析论文" in button_labels
    assert "📝 总结论文" in button_labels


def test_search_button_opens_search_form_without_calling_model():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    search_button = next(
        button for button in app.button if button.label == "🔎 搜索文献"
    )

    search_button.click().run(timeout=10)

    assert not app.exception
    assert any(field.label == "研究主题" for field in app.text_input)
    assert any(slider.label == "返回论文数量" for slider in app.slider)


def test_nested_task_group_error_is_unwrapped_for_the_user():
    error = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [RuntimeError("Connection error.")],
    )

    assert format_error(error) == "Connection error."


def test_saved_answer_exposes_full_page_and_pdf_link():
    app = AppTest.from_file(str(APP_PATH))
    app.session_state['messages'] = [{
        'role': 'assistant', 'content': '论文结论[第 2 页]。',
        'citation_pages': [{
            'arxiv_id': '2508.19294v2', 'title': 'Test paper',
            'page_number': 2, 'text': 'Full original page text',
            'pdf_url': 'https://arxiv.org/pdf/2508.19294v2#page=2',
        }],
    }]
    app.run(timeout=10)
    assert not app.exception
    assert any('查看原文' in item.label for item in app.expander)
    assert any(item.value == 'Full original page text' for item in app.text)
    assert any('#page=2' in item.value for item in app.markdown)
