"""Load application configuration from the project .env file."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration required by the OpenAI-compatible DashScope endpoint."""

    api_key: str
    base_url: str
    model: str
    request_extra_body: dict = field(default_factory=dict)


def _reject_nonfinite(value):
    raise ValueError("nonfinite JSON number")


def parse_extra_body(raw: str) -> dict:
    """Parse provider extensions without echoing potentially sensitive values."""
    try:
        if len(raw) > 8192:
            raise ValueError("too large")
        value = json.loads(raw, parse_constant=_reject_nonfinite)
        if not isinstance(value, dict):
            raise ValueError("object required")
        reserved = {
            "model", "messages", "tools", "tool_choice", "response_format", "stream",
            "temperature", "max_tokens", "max_completion_tokens", "api_key",
            "base_url", "authorization",
        }
        if reserved & {key.lower() for key in value}:
            raise ValueError("reserved fields")
        return value
    except (ValueError, TypeError):
        raise ValueError(
            "LITERATURE_BOT_EXTRA_BODY must be a JSON object of provider extensions "
            "(<=8192 characters); core request fields and credentials are not allowed."
        ) from None


def load_settings(env_path: str | Path, *, override: bool = True) -> Settings:
    """Load and validate settings without exposing the API key."""

    resolved_path = Path(env_path).expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved_path}")

    load_dotenv(dotenv_path=resolved_path, override=override)

    variable_names = {
        "api_key": "DASHSCOPE_API_KEY",
        "base_url": "DASHSCOPE_BASE_URL",
        "model": "DASHSCOPE_MODEL",
    }
    values = {
        field_name: (os.getenv(variable_name) or "").strip()
        for field_name, variable_name in variable_names.items()
    }
    missing = [
        variable_name
        for field_name, variable_name in variable_names.items()
        if not values[field_name]
    ]

    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return Settings(**values, request_extra_body=parse_extra_body(
        os.getenv("LITERATURE_BOT_EXTRA_BODY", "{}")
    ))
