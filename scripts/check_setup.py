"""Read-only environment check; no API requests or implicit weight downloads."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dotenv import dotenv_values
    values = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    result = {
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "configuration_present": {
            name: bool(values.get(name)) for name in
            ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL")
        },
        "packages": {},
        "weights": {},
        "network_or_api_called": False,
    }
    for name in ("streamlit", "mcp", "openai", "arxiv", "PyMuPDF", "torch",
                 "transformers", "adapters", "sentencepiece"):
        try:
            result["packages"][name] = version(name)
        except PackageNotFoundError:
            result["packages"][name] = "missing"
    try:
        from huggingface_hub import try_to_load_from_cache
        from literature_bot.verification.minicheck import MODEL_ID, REVISION
        for repo, filename, revision in [
            ("allenai/specter2_base", "config.json", "main"),
            ("allenai/specter2", "adapter_config.json", "main"),
            ("allenai/specter2_adhoc_query", "adapter_config.json", "main"),
            (MODEL_ID, "pytorch_model.bin", REVISION),
        ]:
            result["weights"][repo] = isinstance(
                try_to_load_from_cache(repo, filename, revision=revision), str)
        result["cache_note"] = "Cache presence does not verify inference. Try a paper search and a question with scientific checks enabled in the app."
    except ImportError:
        result["cache_note"] = "huggingface_hub missing"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
