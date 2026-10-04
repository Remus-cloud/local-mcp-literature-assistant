"""Load application configuration from the project .env file."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration required by the OpenAI-compatible DashScope endpoint."""

    api_key: str
    base_url: str
    model: str


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

    return Settings(**values)
