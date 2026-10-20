"""Shadow verification with explicit unknown states and traceable evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

from .claims import Claim, MAX_CLAIMS
from .minicheck import MODEL_ID, REVISION, get_scorer

CITATION = re.compile(
    r"\[(?:(?:arXiv\s*:?\s*)?([A-Za-z0-9./-]+)\s*[,，]\s*)?第\s*(\d+)\s*页\]",
    re.IGNORECASE,
)
CJK = re.compile(r"[\u3400-\u9fff]")
NUMBERS = re.compile(r"(?<![A-Za-z0-9_.])[+-]?\d+(?:\.\d+)?%?")
LIMITATIONS = [
    "Shadow only: answer is never changed, blocked, or certified.",
    "Checks selected claims, not every sentence; extraction coverage is unknown.",
    "Supported means model support from supplied excerpts, not scientific truth.",
    "A low score is not proof of contradiction: evidence may be incomplete.",
    "Translation by the same Flash model is not independent validation.",
    "Scores are uncalibrated; 0.5 is a diagnostic threshold, not a truth probability.",
    "No statistical significance, causal validity, table parsing or unit algebra verification.",
]


def numeric_literals(text: str) -> set[str]:
    """Normalize unambiguous thousands separators, not units or arithmetic."""
    text = re.sub(
        r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])",
        lambda match: match.group().replace(",", ""),
        text,
    )
    return set(NUMBERS.findall(text))


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_fingerprint(sources: list[dict]) -> str:
    """Tie post-generation diagnostics to the full tool-source payload."""
    return fingerprint(json.dumps(sources, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def evidence_index(sources: list[dict]) -> dict[tuple[str, int], list[dict]]:
    index: dict[tuple[str, int], list[dict]] = {}
    seen = set()
    for source in sources:
        paper = source.get("arxiv_id")
        if not isinstance(paper, str) or not paper:
            continue
        for item in source.get("evidence", []):
            page, text = item.get("page_number"), item.get("text")
            if type(page) is not int or page < 1 or not isinstance(text, str) or not text.strip():
                continue
            key = (paper, page, fingerprint(text))
            if key in seen:
                continue
            seen.add(key)
            index.setdefault((paper, page), []).append({
                "arxiv_id": paper, "page_number": page,
                "chunk_id": item.get("chunk_id"), "text": text, "sha256": key[2],
            })
    return index


def audit_citations(text: str, index: dict) -> list[dict]:
    papers = {paper for paper, _ in index}
    audits = []
    for match in CITATION.finditer(text):
        paper, page = match.group(1), int(match.group(2))
        if paper is None and len(papers) == 1:
            paper = next(iter(papers))
        if paper is None:
            status = "ambiguous_paper" if papers else "no_evidence"
        elif paper not in papers:
            status = "unknown_paper_or_version"
        elif (paper, page) not in index:
            status = "page_not_in_retrieved_evidence"
        else:
            status = "linked"
        audits.append({
            "citation": match.group(), "arxiv_id": paper,
            "page_number": page, "status": status,
        })
    return audits


def lexical_warnings(original: str, english: str, document: str) -> list[str]:
    """Conservative warnings, never proof of a wrong value or translation."""
    original = CITATION.sub("", original)
    warnings = []
    numbers = numeric_literals(english)
    if numbers - numeric_literals(document):
        warnings.append("numbers_not_literal_in_evidence")
    if CJK.search(original):
        warnings.append("translation_not_independently_verified")
        if numeric_literals(original) != numbers:
            warnings.append("translation_numeric_mismatch")
        checks = [
            (r"不|没有|未|无", r"\b(no|not|never|without|cannot|neither|lack\w*|unchanged|unaffected)\b", "negation"),
            (r"所有|全部|任何|总是", r"\b(all|every|any|always|entire)\b", "scope"),
            (r"仅|只", r"\b(only|solely|just|limited)\b", "restriction"),
            (r"显著", r"\bsignificant\w*\b", "significance"),
            (r"导致|因果", r"\b(caus\w*|leads? to|results? in)\b", "causality"),
            (r"可能|或许", r"\b(may|might|could|possible|potential\w*)\b", "uncertainty"),
        ]
        for source_pattern, target_pattern, name in checks:
            if re.search(source_pattern, original) and not re.search(target_pattern, english, re.I):
                warnings.append("translation_" + name + "_mismatch")
    if re.search(r"\b(significant\w*|caus\w*|all|every)\b", english, re.I):
        warnings.append("scientific_scope_requires_review")
    return warnings


class AnswerVerifier:
    def __init__(self, extractor, *, scorer=None, max_seconds: float = 90):
        self.extractor = extractor
        self.scorer = scorer
        self.max_seconds = max_seconds

    def verify_bound(self, answer: str, sources: list[dict], binding: dict) -> dict:
        from .bound import verify_bound
        return verify_bound(self, answer, sources, binding)

    def verify(self, answer: str, sources: list[dict[str, Any]]) -> dict:
        started = time.perf_counter()
        index = evidence_index(sources)
        report = {
            "schema_version": 1, "mode": "shadow", "status": "pending",
            "answer_sha256": fingerprint(answer), "answer_modified": False,
            "scope": "selected_claims_only", "coverage": "unknown",
            "model": MODEL_ID, "model_revision": REVISION,
            "threshold": 0.5, "claims": [], "evidence": [],
            "citation_audit": audit_citations(answer, index),
            "limitations": LIMITATIONS.copy(),
        }
        try:
            if not index:
                report["status"] = "skipped_no_body_evidence"
                return report
            claims = self.extractor.extract(answer)
            report["extraction_usage"] = getattr(self.extractor, "last_usage", None)
            if len(claims) > MAX_CLAIMS:
                raise ValueError("too_many_claims")
            if not claims:
                report["status"] = "no_claims_extracted"
                return report
            for number, claim in enumerate(claims, 1):
                if not isinstance(claim, Claim):
                    claim = Claim.model_validate(claim)
                entry = {
                    "id": f"C{number}", **claim.model_dump(),
                    "status": "not_checked", "support_score": None, "warnings": [],
                    "evidence_refs": [],
                }
                report["claims"].append(entry)
                if time.perf_counter() - started > self.max_seconds:
                    entry["status"] = "skipped_budget"
                    continue
                if claim.original_text not in answer:
                    entry["status"] = "invalid_original_span"
                    continue
                entry["answer_start"] = answer.index(claim.original_text)
                entry["answer_end"] = entry["answer_start"] + len(claim.original_text)
                citations = audit_citations(claim.original_text, index)
                entry["citations"] = citations
                if not citations:
                    entry["status"] = "missing_citation"
                    continue
                if any(c["status"] != "linked" for c in citations):
                    entry["status"] = "invalid_or_ambiguous_citation"
                    continue
                if len({c["arxiv_id"] for c in citations}) != 1:
                    entry["status"] = "multi_paper_claim_requires_review"
                    continue
                selected = []
                keys = set()
                for c in citations:
                    for evidence in index[(c["arxiv_id"], c["page_number"])]:
                        key = (evidence["arxiv_id"], evidence["page_number"], evidence["sha256"])
                        if key not in keys:
                            selected.append(evidence)
                            keys.add(key)
                entry["evidence_refs"] = [
                    {k: e[k] for k in ("arxiv_id", "page_number", "chunk_id", "sha256")}
                    for e in selected
                ]
                for e in selected:
                    if e not in report["evidence"]:
                        report["evidence"].append(e)
                document = "\n\n".join(e["text"] for e in selected)
                entry["warnings"] = lexical_warnings(
                    claim.original_text, claim.english_claim, document,
                )
                if CJK.search(claim.english_claim):
                    entry["status"] = "english_translation_required"
                    continue
                if any(w.startswith("translation_") and w.endswith("_mismatch") for w in entry["warnings"]):
                    entry["status"] = "translation_requires_review"
                    continue
                try:
                    scorer = self.scorer or get_scorer()
                    score = float(scorer.score(document, claim.english_claim))
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("invalid_model_score")
                    entry["support_score"] = score
                    entry["model_label"] = "supported" if score > 0.5 else "unsupported"
                    entry["status"] = (
                        "support_signal" if score > 0.5 else "insufficient_support_signal"
                    )
                except ValueError as error:
                    entry["status"] = (
                        "context_too_long" if str(error) == "context_too_long" else "scorer_error"
                    )
                    entry["error_type"] = type(error).__name__
                except Exception as error:
                    entry["status"] = "scorer_error"
                    entry["error_type"] = type(error).__name__
            report["status"] = "diagnostics_recorded"
            report["scored_claims"] = sum(c["support_score"] is not None for c in report["claims"])
        except Exception as error:
            report["status"] = "verification_error"
            report["error_type"] = type(error).__name__
        finally:
            report["duration_seconds"] = round(time.perf_counter() - started, 3)
        return report
