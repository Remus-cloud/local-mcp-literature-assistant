"""Tests for loading the OpenAI-compatible endpoint configuration."""

import pytest

from literature_bot.config import load_settings


def test_load_settings_reads_required_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DASHSCOPE_API_KEY=test-key\n"
        "DASHSCOPE_BASE_URL=https://example.invalid/v1\n"
        "DASHSCOPE_MODEL=test-model\n",
        encoding="utf-8",
    )

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings(env_path)

    assert settings.api_key == "test-key"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.model == "test-model"


def test_load_settings_rejects_missing_variable(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DASHSCOPE_BASE_URL=https://example.invalid/v1\n"
        "DASHSCOPE_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        load_settings(env_path)


def test_load_settings_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.env")
