"""Read-only numeric shadow audit of declared claims or legacy answer segments.

No generative model or evidence widening is used. Checks are independent, never
an overall truth score. Regex/table parsing has intentionally bounded coverage.
"""
from collections import Counter
from copy import deepcopy
import re
import time

from .numeric_rules import NUMBER, TOKEN, LITERAL, normalize, decimal_value, check_equation, identity, table_facts
from .service import CITATION, fingerprint, source_fingerprint, evidence_index, audit_citations

EQUATION = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<lhs>[+-]?\(?\d[\d.\s%()+*/\-]*?)\s*(?P<eq>=|≈)\s*"
    rf"(?P<rhs>{TOKEN})(?![\d.])"
)
ARROW = re.compile(rf"(?P<before>{TOKEN})\s*(?:→|->)\s*(?P<after>{TOKEN})")


def diagnostic(rule, status, reason, **extra):
    return {"rule": rule, "status": status, "reason": reason, **extra}


def clean(text):
    # Offsets below refer to this explicitly stored normalized text, not raw answer.
    text = CITATION.sub("", text)
    return normalize(text.replace("**", "").replace("`", ""))


def condition_checks(text, facts):
    """Only a single explicit row/dataset/value assertion can be matched."""
    checks = []
    values = re.findall(rf"(?:为|是|equals|is|:)\s*({TOKEN})(?![\d.])", text, re.I)
    datasets = {f["dataset"] for f in facts if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(f['dataset'])}(?![A-Za-z0-9])", text, re.I)}
    names = {f["model"] for f in facts if identity(f["model"]) in identity(text)}
    # Avoid matching 'No NSP' inside 'LTR & No NSP'.
    names = {n for n in names if not any(n != m and identity(n) in identity(m) for m in names)}
    if not values or len(datasets) != 1 or len(names) != 1 or len(values) != 1:
        return checks
    if re.search(r"不(?:为|是)|并非|假设|若|如果|\b(?:not|if|suppose)\b", text, re.I):
        return [diagnostic("metric_condition", "review", "negated_or_hypothetical_assertion")]
    selected = [f for f in facts if f["dataset"] in datasets and f["model"] in names]
    table = re.search(r"(?:Table|表)\s*(\d+)", text, re.I)
    if table:
        selected = [f for f in selected if f["table"] == table.group(1)]
    if not selected:
        return [diagnostic("metric_condition", "review", "requested_table_not_parsed")]
    # Negated contrasts must not turn 'not test' into a claimed test split.
    positive = re.sub(r"(?:而非|不是|非|not\s+)(?:测试集|开发集|test|dev|accuracy|F1)", "", text, flags=re.I)
    splits = set()
    for label, pattern in (("dev", r"\bdev\b|开发集|验证集"),
                           ("test", r"\btest\b|测试集"), ("train", r"\btrain\b|训练集")):
        if re.search(pattern, positive, re.I):
            splits.add(label)
    metrics = set()
    for label, pattern in (("acc", r"(?<![A-Za-z])(?:acc|accuracy)(?![A-Za-z])|准确率"), ("f1", r"(?<![A-Za-z0-9])F1(?![A-Za-z0-9])"),
                           ("bleu", r"(?<![A-Za-z])BLEU(?![A-Za-z])"), ("em", r"(?<![A-Za-z])EM(?![A-Za-z])")):
        if re.search(pattern, positive, re.I):
            metrics.add(label)
    conditions = {(f["split"], f["metric"], f["value"]) for f in selected}
    if len(conditions) != 1:
        return [diagnostic("metric_condition", "review", "conflicting_source_rows", facts=selected)]
    split, metric, value = next(iter(conditions))
    if len(splits) > 1 or len(metrics) > 1:
        status, reason = "review", "compound_or_ambiguous_conditions"
    elif (splits and split not in splits) or (metrics and metric not in metrics):
        status, reason = "conflict", "explicit_metric_or_split_mismatch"
    elif not splits or not metrics:
        status, reason = "review", "missing_explicit_metric_or_split"
    elif values[0].endswith("%") != value.endswith("%"):
        status, reason = "review", "source_scale_not_explicit"
    elif decimal_value(values[0]) != decimal_value(value) and (
        decimal_value(values[0]) * 100 == decimal_value(value)
        or decimal_value(value) * 100 == decimal_value(values[0])
    ):
        status, reason = "review", "possible_fraction_vs_score_scale"
    elif decimal_value(values[0]) != decimal_value(value):
        status, reason = "conflict", "reported_value_mismatch"
    else:
        status, reason = "pass", "explicit_row_column_and_conditions_match"
    checks.append(diagnostic("metric_condition", status, reason, value=values[0], facts=selected))
    return checks


def denominator_checks(text, evidence):
    """Explicit conditional MASK/random/unchanged example, not generic inference."""
    checks = []
    for e in evidence:
        conditional = re.search(r"If the .{0,35}?token is chosen,(.{0,650}?)(?:Then,|$)",
                                e["text"], re.S | re.I)
        if not conditional:
            continue
        for kind, source_pattern, claim_pattern in (
            ("mask", rf"\[MASK\]\s*token\s*({NUMBER})%", r"\[?MASK\]?"),
            ("random", rf"random\s*token\s*({NUMBER})%", r"随机|random"),
            ("unchanged", rf"unchanged.{0,20}?token\s*({NUMBER})%", r"不变|unchanged"),
        ):
            rate = re.search(source_pattern, conditional.group(1), re.I)
            if not rate or not re.search(claim_pattern, text, re.I):
                continue
            if not re.search(rf"(?<![\d.]){re.escape(rate.group(1))}%", text):
                continue
            if re.search(r"并非|不替换|不是\s*\[?MASK|假设|若|如果", text, re.I):
                checks.append(diagnostic("denominator", "review", "negated_or_hypothetical_assertion"))
                continue
            positive = re.sub(r"(?:而不是|而非|不是|非)\s*(?:全部|所有|整个输入序列的全部)\s*(?:token|位置)", "", text, flags=re.I)
            selected = bool(re.search(r"被选中|选中的|目标位置|selected|chosen", positive, re.I))
            all_tokens = bool(re.search(r"(?:全部|所有)\s*(?:输入\s*)?(?:WordPiece\s*)?token|all tokens", positive, re.I))
            if selected and all_tokens:
                status, reason = "review", "multiple_denominators_or_derived_ratio"
            elif selected:
                status, reason = "pass", "conditional_denominator_explicit"
            elif all_tokens:
                # Only direct proportion statements, not a multiplication deriving an overall rate.
                status, reason = ("review", "derived_ratio_needs_explicit_inputs") if "*" in text else (
                    "conflict", "conditional_rate_misattributed_to_all_tokens")
            else:
                status, reason = "review", "denominator_not_explicit_in_claim"
            checks.append(diagnostic("denominator", status, reason, category=kind,
                                     rate=rate.group(1) + "%", source=e["ref"],
                                     source_quote=conditional.group(0)))
    return checks


def check_text(text, evidence, question=""):
    normalized = clean(text)
    checks, records = [], []
    facts = []
    for e in evidence:
        facts.extend({**fact, "source": e["ref"]} for fact in table_facts(e["text"]))
    checks.extend(condition_checks(normalized, facts))
    checks.extend(denominator_checks(normalized, evidence))
    if len({e["ref"].get("arxiv_id") for e in evidence}) > 1:
        for check in checks:
            if check["status"] == "pass":
                check.update(status="review", reason="multiple_paper_scope_requires_review")
    for check in checks:
        if check["rule"] == "metric_condition" and "value" in check:
            records.append({
                "role": "reported_measurement", "value": check["value"],
                "source_facts": deepcopy(check.get("facts", [])),
                "provenance_status": check["status"], "reason": check["reason"],
            })
    for match in EQUATION.finditer(normalized):
        lhs, rhs = match["lhs"].strip(), match["rhs"]
        if not re.search(r"[+*/-]", lhs.lstrip("+-")):
            continue
        preceding = normalized[:match.start()].rstrip()
        if preceding and re.search(r"[A-Za-z0-9_^*/+\-]$", preceding):
            checks.append(diagnostic("arithmetic", "review", "unsupported_expression_context",
                                     quote=match.group()))
            continue
        tail = normalized[match.end():match.end() + 12]
        approx = match["eq"] == "≈" or bool(re.search(r"约|approximately", normalized[max(0, match.start()-8):match.start()], re.I))
        result = check_equation(lhs, rhs, approx, "百分点" in tail or "percentage point" in tail)
        checks.append(diagnostic("arithmetic", result.pop("status"), result.pop("reason"),
                                 expression=lhs, result_literal=rhs, quote=match.group(),
                                 normalized_span=[match.start(), match.end()], **result))
        for literal in LITERAL.finditer(lhs):
            value = literal.group()
            candidates = []
            for e in evidence:
                for found in LITERAL.finditer(normalize(e["text"])):
                    if found.group() == value:
                        candidates.append({"source": e["ref"], "normalized_text_offset": found.start()})
            in_question = value in {m.group() for m in LITERAL.finditer(normalize(question))}
            records.append({
                "value": value, "role": "operand", "expression": lhs,
                "source_candidates": candidates, "occurs_in_user_question": in_question,
                "provenance_status": "review",
                "reason": "Literal occurrence is not semantic attribution; model/condition and user-assumption roles need explicit confirmation.",
            })
    for match in ARROW.finditer(normalized):
        before, after = decimal_value(match["before"]), decimal_value(match["after"])
        prefix = normalized[max(0, match.start()-100):match.start()]
        directions = list(re.finditer(r"下降|降低|变差|升到|上升|提高", prefix))
        if not directions or re.search(r"不|未|没有", prefix[max(0, directions[-1].start()-6):directions[-1].start()]):
            continue
        if match["before"].endswith("%") != match["after"].endswith("%"):
            checks.append(diagnostic("direction", "review", "mixed_arrow_scales"))
            continue
        direction = directions[-1].group()
        negative = direction in {"下降", "降低", "变差"}
        agrees = after < before if negative else after > before
        checks.append(diagnostic(
            "direction", "pass" if agrees else "conflict", "reported_values_direction_only",
            before=match["before"], after=match["after"], asserted=direction,
            quote=normalized[max(0, match.start()-100):match.end()],
            caveat="At reported precision only; local wording scope may require human review.",
        ))
    if re.search(r"(?<![A-Za-z0-9])F1(?![A-Za-z0-9])", normalized, re.I) and "百分点" in normalized:
        checks.append(diagnostic("unit", "review", "prefer_F1_points_not_percentage_points",
                                 caveat="Terminology warning, not a false numeric result."))
    if re.search(r"相对.{0,12}(?:下降|增加|变化).{0,15}百分点", normalized):
        checks.append(diagnostic("unit", "review", "relative_rate_vs_percentage_points"))
    if re.search(r"中段准确率\s*-\s*两端准确率", normalized) and re.search(r"大于零|>\s*0", normalized) and "middle-loss" in normalized:
        checks.append(diagnostic("direction", "review", "middle_loss_sign_and_endpoint_aggregation_require_review",
                                 caveat="No significance test or endpoint definition is inferred."))
    if re.search(r"(?:Table|表)\s*\d+", normalized, re.I) and re.search(r"重启|restarts?", normalized, re.I):
        checks.append(diagnostic("experiment_condition", "review", "restart_count_table_attribution_requires_review"))
    if LITERAL.search(normalized) and not checks:
        checks.append(diagnostic("coverage", "review", "numeric_text_outside_supported_patterns"))
    return {"normalized_text": normalized, "checks": checks, "numeric_records": records,
            "parsed_table_facts": facts}


def select_bound_sources(answer, claim, sources):
    """Validate all declared references before source-based checking; no partial pass."""
    if answer[claim["answer_start"]:claim["answer_end"]] != claim["rendered_text"]:
        return [], "answer_span_mismatch"
    if claim["status"] != "bound":
        return [], claim["status"]
    if not claim["evidence_refs"] or {r["evidence_id"] for r in claim["evidence_refs"]} != set(claim["evidence_ids"]):
        return [], "reference_set_mismatch"
    selected = []
    for ref in claim["evidence_refs"]:
        candidates = [(s, e) for s in sources for e in s.get("evidence", [])
                      if e.get("evidence_id") == ref["evidence_id"]]
        # Keep canonical first chunk, matching EvidenceRegistry's deduplication policy.
        if not candidates:
            return [], "source_missing"
        s, e = candidates[0]
        if any((s["arxiv_id"], e["page_number"], e["text"]) !=
               (ss["arxiv_id"], xx["page_number"], xx["text"])
               for ss, xx in candidates):
            return [], "conflicting_evidence_id"
        if s["arxiv_id"] != ref["arxiv_id"] or any(e.get(k) != ref[k] for k in ("page_number", "chunk_id")) or fingerprint(e["text"]) != ref["sha256"]:
            return [], "source_fingerprint_mismatch"
        selected.append({"text": e["text"], "ref": deepcopy(ref)})
    return selected, "bound_exact_chunks"


def audit_numeric(answer, sources, binding=None, question=""):
    """Never mutates answer, bindings, evidence or previous semantic diagnostics."""
    started = time.perf_counter()
    report = {"schema_version": 1, "mode": "shadow", "status": "pending",
              "answer_modified": False, "answer_sha256": fingerprint(answer),
              "claims": [], "api_calls": 0,
              "limitations": [
                  "Not a truth certificate; extraction coverage is incomplete.",
                  "Only narrow explicit table layouts are parsed; missing context abstains.",
                  "Arithmetic passes do not certify source operands, units, experimental validity or significance.",
                  "No automatic rewriting, general unit algebra, OCR or symbolic proof.",
              ]}
    try:
        report["sources_sha256"] = source_fingerprint(sources)
        if len(answer) > 100000:
            raise ValueError("answer_too_long")
        segments = []
        if binding and binding.get("status") == "structured":
            if binding.get("answer_sha256") != fingerprint(answer):
                raise ValueError("binding_answer_mismatch")
            report["scope"] = "declared_claims_exact_bound_chunks"
            for claim in binding["claims"]:
                selected, status = select_bound_sources(answer, claim, sources)
                segments.append((claim["claim_id"], claim["text"], selected, status))
        elif binding and binding.get("status") == "invalid_schema":
            report["status"] = "skipped_binding_failure"
            return report
        else:
            report["scope"] = "legacy_paragraphs_no_inherited_citations"
            index = evidence_index(sources)
            for i, text in enumerate(re.split(r"\n\s*\n", answer), 1):
                citations = audit_citations(text, index)
                selected = []
                if citations and all(c["status"] == "linked" for c in citations):
                    seen = set()
                    for c in citations:
                        for e in index[(c["arxiv_id"], c["page_number"])]:
                            key = (e["arxiv_id"], e["page_number"], e["sha256"])
                            if key not in seen:
                                selected.append({"text": e["text"], "ref": {k: v for k, v in e.items() if k != "text"}})
                                seen.add(key)
                segments.append((f"P{i}", text, selected, "legacy_page_candidates_not_exact_binding"))
        for claim_id, text, selected, source_status in segments:
            entry = {"claim_id": claim_id, "text": text, "source_status": source_status}
            entry.update(check_text(text, selected, question))
            if selected and source_status.startswith("legacy"):
                # Legacy page candidates are useful context, not exact claim bindings.
                for check in entry["checks"]:
                    if check["rule"] in {"metric_condition", "denominator"} and check["status"] == "pass":
                        check.update(status="review", reason="legacy_page_association_not_exact_binding")
            report["claims"].append(entry)
        report["counts"] = dict(Counter(c["status"] for e in report["claims"] for c in e["checks"]))
        report["status"] = "diagnostics_recorded"
    except Exception as error:
        report.update(status="verification_error", error_type=type(error).__name__)
    finally:
        report["duration_seconds"] = round(time.perf_counter() - started, 6)
    return report
