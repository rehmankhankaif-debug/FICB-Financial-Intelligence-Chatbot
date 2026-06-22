from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_ui_exceptions() -> None:
    app = AppTest.from_file("app.py").run(timeout=10)

    assert len(app.exception) == 0
    assert app.title[0].value == "Financial Intelligence Chatbot"
    assert "Production-grade financial document" in app.caption[0].value
