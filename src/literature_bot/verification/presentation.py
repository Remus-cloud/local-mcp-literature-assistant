"""Read-only, experimental presentation. Never interprets scores as truth."""

from copy import deepcopy
import json

from .service import fingerprint


ISSUE_LABELS = {
    "insufficient_evidence": "正文证据绑定不足；不代表结论一定错误",
    "non_evidential_candidate": "可能是研究建议；尚未核验其可行性",
    "uncertain_attribution": "仅有页码候选，尚未精确绑定片段",
    "coverage_gap": "部分检查未执行、未对齐或超出支持范围",
    "conditions_not_comparable": "指标或实验条件可能不一致，请核对",
    "deterministic_inconsistency": "指定数值规则发现不一致，请核对计算与条件",
    "check_requires_review": "数值、单位或实验归属需要人工复核",
    "verifier_uncertain": "语义核验未提供明确支持信号；不是反驳",
    "possible_source_discrepancy": "来源数值可能存在差异；不能据此选择正确值",
    "not_directly_comparable": "来源实验条件不同，不能直接比较",
    "scope_requires_review": "局部检索不足以支持全文或普遍性断言",
}

SEMANTIC_LABELS = {
    "support_signal": "MiniCheck 给出支持信号（不是科学真伪认证）",
    "insufficient_support_signal": "MiniCheck 支持信号较低（可能误报，不代表论文反驳）",
    "scorer_error": "MiniCheck 未能完成推理；检查本地权重和运行环境",
    "context_too_long": "证据超过核验长度上限；未静默截断",
    "translation_requires_review": "翻译可能改变数字、否定或范围，未评分",
    "translation_missing": "缺少核验所需翻译，未评分",
    "skipped_budget": "达到核验时间预算，未评分",
    "skipped_claim_limit": "超过本次最多 8 条语义检查范围",
    "not_run_or_not_aligned": "语义检查未执行或未与本结论对齐",
}


def diagnostic_view(answer: str, verification: dict) -> dict:
    """Validate answer identity and evidence hashes before rendering saved data."""
    report = verification.get("unified", {})
    view = {"status": report.get("status", "not_run"), "claims": [], "events": [],
            "scored_claims": 0, "declared_claims": 0}
    if not report:
        return view
    if report.get("answer_sha256") != fingerprint(answer):
        return {**view, "status": "answer_mismatch"}
    if report.get("status") != "diagnostics_recorded":
        return view
    records = {r["record_id"]: r for r in report.get("evidence_records", [])}
    if any(fingerprint(r["text"]) != r["sha256"] for r in records.values()):
        return {**view, "status": "evidence_mismatch"}
    diagnostics = {d["claim_id"]: d for d in report.get("claim_diagnostics", [])}
    for claim in report.get("claim_records", []):
        if answer[claim["answer_start"]:claim["answer_end"]] != claim["rendered_text"]:
            return {**view, "status": "claim_mismatch", "claims": [], "scored_claims": 0}
        diagnostic = diagnostics.get(claim["claim_id"], {})
        semantic = diagnostic.get("dimensions", {}).get("semantic", {})
        status = semantic.get("status", "not_run_or_not_aligned")
        if status in {"support_signal", "insufficient_support_signal"}:
            view["scored_claims"] += 1
        ids = claim.get("evidence_record_ids", [])
        if any(key not in records for key in ids):
            return {**view, "status": "evidence_mismatch", "claims": [], "scored_claims": 0}
        view["claims"].append({
            "id": claim["claim_id"], "text": claim["text"],
            "issues": list(dict.fromkeys(ISSUE_LABELS.get(i["category"], "未识别诊断，需要人工复核")
                                         for i in diagnostic.get("issues", []))),
            "semantic": SEMANTIC_LABELS.get(status, "语义检查未完成：" + status),
            "evidence": [deepcopy(records[key]) for key in ids],
            "details": deepcopy(diagnostic),
        })
    view["declared_claims"] = len(view["claims"])
    view["events"] = deepcopy(report.get("source_events", []))
    return view


def export_trial(message: dict, *, blind: bool = False) -> str:
    """Explicit user download only. No settings, credentials or environment."""
    payload = {
        "schema_version": 1,
        "purpose": "human_blind_review" if blind else "experimental_diagnostic_trial",
        "independent_validation_completed": False,
        "question": message.get("question", ""),
        "answer": message["content"],
        "answer_sha256": fingerprint(message["content"]),
        "evidence_sources": message.get("evidence_sources", []),
        "citation_pages": message.get("citation_pages", []),
    }
    if blind:
        payload["human_labels"] = []
        payload["instructions"] = "请先根据原文独立标注；本文件不包含模块诊断或分数。"
    else:
        payload.update(verification=message.get("verification", {}),
                       search_runs=message.get("search_runs", []),
                       trial_settings=message.get("trial_settings", {}))
    return json.dumps(payload, ensure_ascii=False, indent=2)
