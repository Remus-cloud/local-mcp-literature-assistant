"""Check explicitly bound excerpts without page-level evidence expansion."""
from collections import defaultdict
from types import SimpleNamespace
import time

from .claims import Claim, MAX_CLAIMS
from .service import LIMITATIONS, fingerprint


def verify_bound(verifier, answer, sources, binding):
    started = time.perf_counter()
    report = {
        "schema_version": 2, "mode": "shadow", "status": "pending",
        "binding_mode": "structured", "answer_modified": False,
        "answer_sha256": fingerprint(answer), "coverage": "declared_claims_only",
        "claims": [], "evidence": [], "limitations": LIMITATIONS.copy(),
        "binding": binding,
    }
    report["limitations"][1] = (
        "Bindings cover declared claims only; implicit facts may be omitted. "
        "Semantic scoring checks at most 8 bound claims."
    )
    try:
        if binding.get("answer_sha256") != fingerprint(answer):
            raise ValueError("binding_answer_mismatch")
        by_id = {}
        for source in sources:
            for e in source.get("evidence", []):
                key = e.get("evidence_id")
                if not key:
                    continue
                value = {"arxiv_id": source["arxiv_id"], **e}
                if key in by_id and (
                    by_id[key]["arxiv_id"], by_id[key]["page_number"], by_id[key]["text"]
                ) != (value["arxiv_id"], value["page_number"], value["text"]):
                    raise ValueError("conflicting_evidence_id")
                if key not in by_id:
                    by_id[key] = value
        pending = []
        for claim in binding["claims"]:
            entry = {
                "id": claim["claim_id"], "original_text": claim["rendered_text"],
                "status": "not_checked" if claim["status"] == "bound" else claim["status"],
                "support_score": None,
                "evidence_refs": claim["evidence_refs"], "warnings": [],
            }
            report["claims"].append(entry)
            if claim["status"] != "bound":
                continue
            if answer[claim["answer_start"]:claim["answer_end"]] != claim["rendered_text"]:
                entry["status"] = "invalid_original_span"
                continue
            selected = []
            refs = claim["evidence_refs"]
            if {r["evidence_id"] for r in refs} != set(claim["evidence_ids"]) or not refs:
                entry["status"] = "invalid_binding_reference"
                continue
            for ref in refs:
                actual = by_id.get(ref["evidence_id"])
                if actual is None or any(actual.get(k) != ref.get(k) for k in ("arxiv_id", "page_number", "chunk_id")):
                    break
                if fingerprint(actual["text"]) != ref["sha256"]:
                    break
                selected.append(actual)
            if len(selected) != len(refs):
                entry["status"] = "invalid_binding_reference"
                continue
            if len(pending) >= MAX_CLAIMS:
                entry["status"] = "skipped_claim_limit"
                continue
            pending.append((claim, entry, selected))
        translations = verifier.extractor.translate_bound([
            {"claim_id": c["claim_id"], "text": c["text"]} for c, _, _ in pending
        ]) if pending else {}
        report["translation_usage"] = getattr(verifier.extractor, "last_usage", None) if pending else None
        from .service import AnswerVerifier
        for claim, entry, selected in pending:
            remaining = verifier.max_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                entry["status"] = "skipped_budget"
                continue
            english = translations.get(claim["claim_id"])
            if not english:
                entry["status"] = "translation_missing"
                continue
            fixed = Claim(original_text=claim["rendered_text"], english_claim=english)
            extractor = SimpleNamespace(extract=lambda answer, fixed=fixed: [fixed])
            grouped = defaultdict(list)
            for e in selected:
                grouped[e["arxiv_id"]].append(e)
            # Crucially, only referenced chunks enter the legacy semantic checker.
            sub = AnswerVerifier(extractor, scorer=verifier.scorer, max_seconds=remaining).verify(
                claim["rendered_text"],
                [{"arxiv_id": paper, "evidence": evidence} for paper, evidence in grouped.items()],
            )
            if sub["claims"]:
                entry.update(sub["claims"][0])
                entry.update(id=claim["claim_id"], evidence_ids=claim["evidence_ids"],
                             answer_start=claim["answer_start"], answer_end=claim["answer_end"])
            else:
                entry["status"] = sub["status"]
            for e in sub["evidence"]:
                if e not in report["evidence"]:
                    report["evidence"].append(e)
        report["status"] = "diagnostics_recorded"
        report["scored_claims"] = sum(c["support_score"] is not None for c in report["claims"])
    except Exception as error:
        report["status"] = "verification_error"
        report["error_type"] = type(error).__name__
    finally:
        report["duration_seconds"] = round(time.perf_counter() - started, 3)
    return report
