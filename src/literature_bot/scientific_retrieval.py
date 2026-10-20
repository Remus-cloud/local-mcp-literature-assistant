"""Optional SPECTER2 paper-level ranking.

SPECTER2 is used only for ranking paper metadata (title + abstract).  It does
not replace the page-aware BM25 retrieval used after a PDF has been prepared.
The scientific dependencies are imported lazily so the original application
continues to work when the optional ``scientific`` extra is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import PaperMetadata


SPECTER2_BASE_MODEL = "allenai/specter2_base"
SPECTER2_PROXIMITY_ADAPTER = "allenai/specter2"
SPECTER2_ADHOC_QUERY_ADAPTER = "allenai/specter2_adhoc_query"


@dataclass(frozen=True)
class PaperSearchHit:
    """A paper ranked by a paper-level semantic retriever."""

    paper: PaperMetadata
    score: float
    rank: int


class Specter2PaperRetriever:
    """Rank arXiv metadata with the official SPECTER2 adapters.

    The query uses the official ``adhoc_query`` adapter. Candidate documents
    use the official ``proximity`` adapter and are represented as
    ``title [SEP] abstract``. Embeddings are L2-normalized, so their dot
    product is cosine similarity.
    """

    def __init__(
        self,
        *,
        base_model: str = SPECTER2_BASE_MODEL,
        proximity_adapter: str = SPECTER2_PROXIMITY_ADAPTER,
        query_adapter: str = SPECTER2_ADHOC_QUERY_ADAPTER,
        batch_size: int = 8,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "SPECTER2 requires the optional dependencies. Install with "
                "python -m pip install -e \".[scientific]\"."
            ) from error

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._model = AutoAdapterModel.from_pretrained(base_model)
        self._model.load_adapter(
            proximity_adapter,
            source="hf",
            load_as="proximity",
        )
        self._model.load_adapter(
            query_adapter,
            source="hf",
            load_as="adhoc_query",
        )
        self._model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        if self._device.type == "cpu":
            # This model runs in the dedicated MCP process. Avoid CPU thread
            # oversubscription on desktop machines with many logical cores.
            torch.set_num_threads(4)
        self._model.to(self._device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._candidate_cache: dict[str, object] = {}

    @property
    def device(self) -> str:
        """Return the device used for inference."""

        return str(self._device)

    def _embed(self, texts: Sequence[str], adapter_name: str):
        if not texts:
            raise ValueError("texts cannot be empty")

        torch = self._torch
        self._model.set_active_adapters(adapter_name)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=self._max_length,
                )
                inputs = {name: value.to(self._device) for name, value in inputs.items()}
                output = self._model(**inputs)
                embeddings = output.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                batches.append(embeddings.detach().cpu())
        return torch.cat(batches, dim=0)

    def _candidate_text(self, paper: PaperMetadata) -> str:
        return f"{paper.title}{self._tokenizer.sep_token}{paper.summary or ''}"

    def rank(
        self,
        query: str,
        papers: Sequence[PaperMetadata],
        *,
        top_k: int | None = None,
    ) -> list[PaperSearchHit]:
        """Return papers sorted by semantic relevance to ``query``."""

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if not papers:
            return []
        if top_k is None:
            top_k = len(papers)
        if not 1 <= top_k <= len(papers):
            raise ValueError("top_k must be between 1 and the number of papers")

        query_embedding = self._embed([query], "adhoc_query")[0]
        missing: list[PaperMetadata] = []
        for paper in papers:
            if paper.arxiv_id not in self._candidate_cache:
                missing.append(paper)
        if missing:
            embeddings = self._embed(
                [self._candidate_text(paper) for paper in missing],
                "proximity",
            )
            for paper, embedding in zip(missing, embeddings):
                self._candidate_cache[paper.arxiv_id] = embedding

        scored = []
        for paper in papers:
            embedding = self._candidate_cache[paper.arxiv_id]
            score = float(torch_dot(query_embedding, embedding))
            scored.append((score, paper))
        scored.sort(key=lambda item: (-item[0], item[1].arxiv_id))
        return [
            PaperSearchHit(paper=paper, score=score, rank=rank)
            for rank, (score, paper) in enumerate(scored[:top_k], start=1)
        ]


def torch_dot(left, right) -> float:
    """Compute a scalar dot product without exposing torch at module import."""

    return left @ right
