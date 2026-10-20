"""CPU-capable wrapper around the official, pinned MiniCheck checkpoint.

Uses the published Flan-T5 input format and first-token logits [3, 209].
Reference: https://github.com/Liyan06/MiniCheck/blob/main/minicheck/inference.py
No remote code is executed. No silent truncation or max-over-excerpts pooling:
the exact, bounded evidence supplied by the caller is scored as one document.
"""

from functools import lru_cache
import math
from threading import Lock

MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
MAX_INPUT_TOKENS = 2048


class MiniCheckScorer:
    def __init__(self, *, local_files_only: bool = True, device: str = "cpu",
                 cpu_threads: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if not 1 <= cpu_threads <= 32:
            raise ValueError("cpu_threads must be between 1 and 32")
        if device == "cpu":
            # PyTorch's intra-op setting is process-wide. The MCP retrieval
            # server runs in a separate process and is unaffected.
            torch.set_num_threads(cpu_threads)
        self.torch = torch
        self.lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=local_files_only,
            trust_remote_code=False,
        )
        # The tokenizer metadata retains T5's generic 512-token warning;
        # MiniCheck's published inference limit is 2048, enforced below.
        self.tokenizer.model_max_length = MAX_INPUT_TOKENS
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=local_files_only,
            trust_remote_code=False,
        ).to(device).eval()
        self.device = device

    def score(self, document: str, claim: str) -> float:
        if not document.strip() or not claim.strip():
            raise ValueError("Empty MiniCheck document or claim")
        text = "predict: " + document + self.tokenizer.eos_token + claim
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
            raise ValueError("context_too_long")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        torch = self.torch
        with self.lock, torch.inference_mode():
            start = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            output = self.model(**inputs, decoder_input_ids=start)
            score = output.logits[0, 0, [3, 209]].softmax(-1)[1].item()
        if not math.isfinite(score):
            raise ValueError("nonfinite_model_score")
        return score


@lru_cache(maxsize=1)
def get_scorer() -> MiniCheckScorer:
    """Reuse weights only; no user answers or evidence are cached globally."""
    return MiniCheckScorer()
