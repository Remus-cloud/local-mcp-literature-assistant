"""P2 record/diagnostic layer, post-generation and strictly shadow-only."""
from collections import Counter
from copy import deepcopy
import math
import re
import time

from .records import ClaimRecord, candidate, classify_claim, claim_context, collect_evidence
from .numeric_rules import identity
from .conflicts import compare_sources
from .numeric import audit_numeric, select_bound_sources
from .service import fingerprint, source_fingerprint, evidence_index, audit_citations


def record_key(ref):
    return ref["arxiv_id"], ref["page_number"], ref["sha256"]


def make_claims(answer, sources, binding, by_key):
    claims = []
    if binding and binding.get("status") == "invalid_schema":
        return claims, "invalid_binding_schema"
    if binding and binding.get("status") == "structured":
        if binding.get("answer_sha256") != fingerprint(answer):
            raise ValueError("binding_answer_mismatch")
        for old in binding["claims"]:
            selected, status = select_bound_sources(answer, old, sources)
            claims.append(ClaimRecord(
                claim_id=old["claim_id"], text=old["text"], rendered_text=old["rendered_text"],
                answer_start=old["answer_start"], answer_end=old["answer_end"],
                evidence_record_ids=list(dict.fromkeys(by_key[record_key(e["ref"])].record_id for e in selected)),
                binding_status=status, declared_context=claim_context(old["text"]),
                **classify_claim(old["text"]),
            ))
        if len({c.claim_id for c in claims}) != len(claims):
            raise ValueError("duplicate_claim_id")
        return claims, "structured_exact_binding"
    index = evidence_index(sources)
    offset = 0
    for number, text in enumerate(re.split(r"\n\s*\n", answer), 1):
        start = answer.find(text, offset)
        offset = start + len(text)
        citations = audit_citations(text, index)
        ids = []
        if citations and all(c["status"] == "linked" for c in citations):
            for citation in citations:
                ids.extend(by_key[record_key(e)].record_id
                           for e in index[(citation["arxiv_id"], citation["page_number"])])
        claims.append(ClaimRecord(
            claim_id=f"P{number}", text=text, rendered_text=text, answer_start=start, answer_end=offset,
            evidence_record_ids=list(dict.fromkeys(ids)), binding_status="legacy_page_candidates" if ids else "unbound",
            declared_context=claim_context(text), **classify_claim(text),
        ))
    return claims, "legacy_paragraphs_not_atomic_claims"


def issue(category, reason, **extra):
    return {"category": category, "reason": reason, **extra}


def summarize_claim(claim, numeric, semantic, events):
    checks = deepcopy(numeric.get("checks", [])) if numeric else []
    dimensions = {
        "provenance": {"status": claim.binding_status, "evidence_record_ids": claim.evidence_record_ids},
        "source_conditions": [c for c in checks if c["rule"] in {"metric_condition", "experiment_condition"}],
        "arithmetic_and_ratio": [c for c in checks if c["rule"] in {"arithmetic", "denominator", "direction"}],
        "units": [c for c in checks if c["rule"] == "unit"],
        "numeric_coverage": [c for c in checks if c["rule"] == "coverage"],
        "semantic": deepcopy(semantic) if semantic else {"status": "not_run_or_not_aligned"},
    }
    issues = []
    if claim.binding_status not in {"bound_exact_chunks", "legacy_page_candidates"}:
        if claim.kind != "recommendation":
            issues.append(issue("insufficient_evidence", claim.binding_status))
        else:
            issues.append(issue("non_evidential_candidate", "recommendation_type_is_unverified"))
    elif claim.binding_status == "legacy_page_candidates":
        issues.append(issue("uncertain_attribution", "legacy_page_citation_is_not_exact_binding"))
    if not numeric:
        issues.append(issue("coverage_gap", "numeric_not_available_or_not_aligned"))
    for check in checks:
        if check["status"] == "conflict":
            category = "conditions_not_comparable" if check["reason"] == "explicit_metric_or_split_mismatch" else "deterministic_inconsistency"
            issues.append(issue(category, check["reason"], rule=check["rule"],
                                caveat="Only the named rule was checked; not whole-claim falsity."))
        elif check["status"] == "review":
            category = "coverage_gap" if check["rule"] == "coverage" else "check_requires_review"
            issues.append(issue(category, check["reason"], rule=check["rule"]))
    if not semantic:
        issues.append(issue("coverage_gap", "semantic_not_run_or_not_aligned"))
    elif semantic.get("status") != "support_signal":
        # Low support is never converted to a source contradiction.
        issues.append(issue("verifier_uncertain", semantic.get("status", "unknown")))
    for event in events:
        if set(event["evidence_record_ids"]) & set(claim.evidence_record_ids):
            issues.append(issue(event["category"], "see_source_comparison", event_id=event["event_id"]))
    if re.search(r"从未测试|全文.{0,15}未出现|没有测试.{0,30}Qwen|不可能.*实验证明", claim.text, re.I):
        issues.append(issue("scope_requires_review", "retrieved_excerpts_cannot_establish_global_absence"))
    return {
        "claim_id": claim.claim_id, "dimensions": dimensions, "issues": issues,
        "review_status": "needs_review" if issues else "no_issue_in_executed_checks",
        "certifies_truth": False,
    }


def build_diagnostics(answer, sources, binding=None, verification=None, question="", *, numeric_report=None):
    """Build records and a diagnostic view without mutating any input.

    An upstream report must match the current answer AND claim text before reuse.
    Original semantic/numeric reports remain separate and unchanged.
    """
    started = time.perf_counter()
    report = {
        "schema_version": 1, "mode": "shadow", "status": "pending",
        "answer_sha256": fingerprint(answer), "answer_modified": False, "api_calls": 0,
        "evidence_records": [], "claim_records": [], "claim_diagnostics": [], "source_events": [],
        "limitations": [
            "No global truth score, no automatic rewriting, no retrieval or model calls.",
            "Claim types and parsed experiment fields are candidates, not independently verified facts.",
            "Candidate source discrepancies require review; differing conditions are not contradictions.",
            "Unknown conditions remain unknown. Missing/low verifier signals are not proof of falsity.",
            "Only existing retrieved evidence is inspected; this is not an exhaustive paper audit.",
        ],
    }
    try:
        if len(answer) > 100000:
            raise ValueError("answer_budget_exceeded")
        records, by_key = collect_evidence(sources)
        claims, scope = make_claims(answer, sources, binding, by_key)
        report["scope"] = scope
        report["evidence_records"] = [r.model_dump() for r in records]
        if scope == "invalid_binding_schema":
            report["status"] = "skipped_binding_failure"
            return report
        numeric = numeric_report if numeric_report is not None else audit_numeric(answer, sources, binding, question)
        numeric_valid = (
            numeric.get("answer_sha256") == fingerprint(answer)
            and numeric.get("sources_sha256") == source_fingerprint(sources)
            and numeric.get("status") == "diagnostics_recorded"
        )
        nmap = {c["claim_id"]: c for c in numeric.get("claims", [])} if numeric_valid else {}
        verification = verification or {}
        semantic_valid = verification.get("answer_sha256") == fingerprint(answer)
        smap = {c["id"]: c for c in verification.get("claims", [])} if semantic_valid else {}
        report["upstream"] = {
            "numeric_status": numeric.get("status"), "numeric_answer_aligned": numeric_valid,
            "semantic_status": verification.get("status", "not_run"), "semantic_answer_aligned": semantic_valid,
            "semantic_claims_available": len(smap), "semantic_claims_aligned": 0,
        }
        events = compare_sources(records)
        report["source_events"] = events
        for claim in claims:
            nc = nmap.get(claim.claim_id)
            if nc and nc.get("text") != claim.text:
                nc = None
            sc = smap.get(claim.claim_id)
            if sc and sc.get("original_text") != claim.rendered_text:
                sc = None
            if sc and sc.get("status") in {"support_signal", "insufficient_support_signal"}:
                score = sc.get("support_score")
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    sc = None
            if sc and sc.get("support_score") is not None:
                allowed = {record_key(r.model_dump()) for r in records if r.record_id in claim.evidence_record_ids}
                refs = sc.get("evidence_refs", [])
                if not refs or any(record_key(ref) not in allowed for ref in refs):
                    sc = None
            if sc:
                report["upstream"]["semantic_claims_aligned"] += 1
            if nc:
                claim.numeric_records = deepcopy(nc.get("numeric_records", []))
                facts = [f for check in nc.get("checks", [])
                         if check["rule"] == "metric_condition" for f in check.get("facts", [])]
                for field in ("model", "dataset"):
                    values = {f[field] for f in facts if identity(f[field]) in identity(claim.text)}
                    if len(values) == 1:
                        setattr(claim.declared_context, field, candidate(next(iter(values)), claim.text))
            report["claim_records"].append(claim.model_dump())
            report["claim_diagnostics"].append(summarize_claim(claim, nc, sc, events))
        report["counts"] = {
            "claims": len(claims), "evidence_records": len(records),
            "claim_kinds": dict(Counter(c.kind for c in claims)),
            "source_event_categories": dict(Counter(e["category"] for e in events)),
            "issue_categories": dict(Counter(i["category"] for d in report["claim_diagnostics"] for i in d["issues"])),
        }
        report["status"] = "diagnostics_recorded"
    except Exception as error:
        report.update(status="verification_error", error_type=type(error).__name__)
    finally:
        report["duration_seconds"] = round(time.perf_counter() - started, 6)
    return report
