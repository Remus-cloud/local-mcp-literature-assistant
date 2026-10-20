"""Versioned post-generation records; no prompt changes or model declarations.

Observed text is not verified experimental context. Parsed fields remain candidates.
"""
from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field

from .numeric import EQUATION, clean
from .numeric_rules import table_facts
from .service import fingerprint


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ContextField(Record):
    value: str | None = None
    status: Literal["unknown", "text_observed", "parser_candidate"] = "unknown"
    basis: str | None = None


class ScientificContext(Record):
    model: ContextField = Field(default_factory=ContextField)
    dataset: ContextField = Field(default_factory=ContextField)
    task: ContextField = Field(default_factory=ContextField)
    split: ContextField = Field(default_factory=ContextField)
    metric: ContextField = Field(default_factory=ContextField)
    unit: ContextField = Field(default_factory=ContextField)
    experiment: ContextField = Field(default_factory=ContextField)


class Observation(Record):
    observation_id: str
    value: str
    quote: str
    start: int
    end: int
    context: ScientificContext
    extraction: str
    status: Literal["parser_candidate"] = "parser_candidate"


class EvidenceRecord(Record):
    schema_version: int = 1
    record_id: str
    tool_evidence_ids: list[str] = Field(default_factory=list)
    arxiv_id: str
    page_number: int
    chunk_id: int | str | None
    sha256: str
    text: str
    provenance_status: str = "retrieved_text_fingerprinted_not_scientifically_verified"
    location_mentions: list[dict] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)


class ClaimRecord(Record):
    schema_version: int = 1
    claim_id: str
    text: str
    rendered_text: str
    answer_start: int
    answer_end: int
    kind: Literal["reported_result", "calculation", "inference", "recommendation", "mixed", "unknown"]
    kind_status: Literal["heuristic_candidate", "unknown"]
    kind_candidates: list[str]
    kind_basis: list[str]
    declared_context: ScientificContext
    evidence_record_ids: list[str]
    binding_status: str
    numeric_records: list[dict] = Field(default_factory=list)


def candidate(value, basis):
    return ContextField(value=value, status="parser_candidate", basis=basis)


def classify_claim(text):
    kinds, basis = [], []
    for kind, pattern in (
        ("recommendation", r"建议|应当设计|recommend|propos(?:e|ed) experiment"),
        ("calculation", r"计算|期望数量|差值|(?<![A-Za-z])calculated|derived"),
        ("inference", r"推断|假设|可能|hypothes|infer"),
        ("reported_result", r"论文(?:报告|指出|发现|实际评测)|原文(?:写|报告)|Table\s*\d+|摘要报告"),
    ):
        match = re.search(pattern, text, re.I)
        if match:
            kinds.append(kind)
            basis.append(match.group())
    formula = EQUATION.search(clean(text))
    if formula and "calculation" not in kinds:
        kinds.append("calculation")
        basis.append(formula.group())
    # Table reference in a calculation is provenance, not a claim of reported result.
    if "calculation" in kinds and "reported_result" in kinds and not re.search(r"论文报告|原文报告", text):
        kinds.remove("reported_result")
    return {
        "kind": kinds[0] if len(kinds) == 1 else "mixed" if kinds else "unknown",
        "kind_status": "heuristic_candidate" if kinds else "unknown",
        "kind_candidates": kinds, "kind_basis": basis,
    }


def claim_context(text):
    context = ScientificContext()
    for key, patterns in {
        "split": [("dev", r"\bdev\b|开发集"), ("test", r"\btest\b|测试集")],
        "metric": [("f1", r"(?<![A-Za-z0-9])F1(?![A-Za-z0-9])"),
                   ("accuracy", r"accuracy|准确率"), ("bleu", r"BLEU")],
    }.items():
        found = [(v, re.search(p, text, re.I)) for v, p in patterns]
        found = [(v, m) for v, m in found if m]
        if len(found) == 1 and not re.search(r"而非|不是|not\b", text, re.I):
            setattr(context, key, candidate(found[0][0], found[0][1].group()))
    table = re.search(r"(?:Table|表)\s*(\d+)", text, re.I)
    if table:
        context.experiment = candidate("table:" + table.group(1), table.group())
    return context


def make_observation(record, value, quote, context, extraction):
    start = record.text.find(quote)
    if start < 0:
        raise ValueError("observation_quote_not_in_source")
    return Observation(
        observation_id=f"{record.record_id}:O{len(record.observations)+1}", value=value,
        quote=quote, start=start, end=start + len(quote), context=context, extraction=extraction,
    )


def enrich_evidence(record):
    for match in re.finditer(r"(?m)^\s*(?:Abstract|(?:\d+\.)*\d+\s+[A-Z][^\n]{2,65}|Table\s+\d+\s*:[^\n]*)", record.text):
        record.location_mentions.append({
            "text": match.group().strip(), "start": match.start(), "end": match.end(),
            "status": "text_observed", "scope": "mention_only_not_chunk_ownership",
        })
    for fact in table_facts(record.text):
        context = ScientificContext(**{
            key: candidate(value, fact["header_quote"] + fact["caption_quote"])
            for key, value in {
                "model": fact["model"], "dataset": fact["dataset"], "task": fact["dataset"],
                "split": fact["split"], "metric": fact["metric"],
                "unit": fact["reported_scale"], "experiment": "table:" + fact["table"],
            }.items()
        })
        record.observations.append(make_observation(
            record, fact["value"], fact["row_quote"], context, "explicit_tasks_table",
        ))
    translation_observations(record)
    return record


def translation_observations(record):
    """Narrow BLEU layout and explicit WMT result sentences, never general OCR.

    WMT2014/newstest2014 and 'our big model' mappings are parser candidates.
    Missing split/configuration stays unknown, so discrepancies require review.
    """
    text = record.text
    header = re.search(r"BLEU[^\n]*Training Cost[\s\S]{0,80}?EN-DE\s+EN-FR\s+EN-DE\s+EN-FR", text)
    if header and "newstest2014" in text[:header.start()] and re.search(r"Table\s+\d+:", text[:header.start()]):
        for row in re.finditer(r"(?m)^[ \t]*(Transformer \((?:big|base model)\))[ \t]{2,}(\d+\.\d+)[ \t]+(\d+\.\d+)[^\n]*", text[header.end():]):
            quote = row.group()
            for task, value in (("en-de", row.group(2)), ("en-fr", row.group(3))):
                context = ScientificContext(
                    model=candidate(row.group(1).replace(" ", "").lower(), quote),
                    dataset=candidate("wmt2014", "newstest2014 (dataset alias; requires review)"),
                    task=candidate(task, header.group()), metric=candidate("bleu", header.group()),
                    split=candidate("test", "newstest2014 tests"),
                    unit=candidate("bleu_points", header.group()),
                    # No complete training/decoding configuration is established by the row.
                )
                record.observations.append(make_observation(record, value, quote, context, "bleu_translation_table"))
    pattern = (
        r"On the WMT\s+2014\s+English-to-(French|German)\s+translation task,"
        r"\s+(our (?:big )?model)[^.]{0,120}?BLEU score of\s+(\d+\.\d+)"
    )
    for match in re.finditer(pattern, text, re.I):
        context = ScientificContext(
            dataset=candidate("wmt2014", match.group()), task=candidate("en-fr" if match[1].lower() == "french" else "en-de", match.group()),
            metric=candidate("bleu", match.group()), unit=candidate("bleu_points", match.group()),
        )
        # Require an explicit local name; never map 'our model' in the abstract to big.
        local = text[match.start():match.end()+400]
        name = re.search(r"The Transformer \(big\) model trained for English-to-" + match[1], local, re.I)
        if "big" in match[2].lower() and name:
            context.model = candidate("transformer(big)", name.group())
        record.observations.append(make_observation(record, match[3], match.group(), context, "explicit_wmt_sentence"))


def collect_evidence(sources):
    records, by_key = [], {}
    for source in sources:
        paper = source.get("arxiv_id")
        if not isinstance(paper, str) or not paper:
            continue
        for item in source.get("evidence", []):
            text, page = item.get("text"), item.get("page_number")
            if not isinstance(text, str) or type(page) is not int or page < 1:
                continue
            if len(text) > 50000 or len(records) > 100:
                raise ValueError("source_budget_exceeded")
            key = (paper, page, fingerprint(text))
            if key not in by_key:
                record = EvidenceRecord(
                    record_id=f"R{len(records)+1}", arxiv_id=paper, page_number=page,
                    chunk_id=item.get("chunk_id"), sha256=key[2], text=text,
                )
                by_key[key] = record
                records.append(enrich_evidence(record))
            evidence_id = item.get("evidence_id")
            if isinstance(evidence_id, str) and evidence_id not in by_key[key].tool_evidence_ids:
                by_key[key].tool_evidence_ids.append(evidence_id)
    return records, by_key
