"""Explicitly download the pinned MiniCheck model; never called by chat requests."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_bot.verification.minicheck import MODEL_ID, REVISION


def main():
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} at {REVISION} (about 3.2 GB).", flush=True)
    location = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "spiece.model", "pytorch_model.bin"],
        max_workers=1,
    )
    print(f"Ready in Hugging Face cache: {location}")


if __name__ == "__main__":
    main()
