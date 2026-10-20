"""Compare source observations without declaring which source is correct."""
from itertools import combinations
from decimal import Decimal

from .numeric_rules import identity


def compare_sources(records):
    observations = [(r, o) for r in records for o in r.observations]
    if len(observations) > 500:
        raise ValueError("observation_budget_exceeded")
    events, seen = [], set()
    for (left, a), (right, b) in combinations(observations, 2):
        if left.arxiv_id != right.arxiv_id:
            continue  # Never conflate paper versions or cross-paper experiments.
        fields = ("model", "dataset", "task", "metric")
        ac, bc = a.context, b.context
        if any(getattr(ac, f).value is None or getattr(bc, f).value is None for f in fields):
            continue  # Missing entity identity is not a comparable observation.
        if any(identity(getattr(ac, f).value) != identity(getattr(bc, f).value) for f in fields):
            continue
        if a.value == b.value:
            continue
        if a.value.endswith("%") == b.value.endswith("%") and Decimal(a.value.rstrip("%")) == Decimal(b.value.rstrip("%")):
            continue
        varying, missing = [], []
        for key in ("split", "unit", "experiment"):
            av, bv = getattr(ac, key).value, getattr(bc, key).value
            if av is None or bv is None:
                missing.append(key)
            elif av != bv:
                varying.append(key)
        kind = "not_directly_comparable" if varying else "possible_source_discrepancy"
        signature = (kind, left.arxiv_id, tuple(getattr(ac, f).value for f in fields),
                     tuple(sorted((a.value, b.value))), tuple(varying), tuple(missing),
                     tuple(sorted((a.quote, b.quote))))
        if signature in seen:
            continue
        seen.add(signature)
        events.append({
            "event_id": f"S{len(events)+1}", "category": kind,
            "status": "requires_review", "evidence_record_ids": [left.record_id, right.record_id],
            "observation_ids": [a.observation_id, b.observation_id],
            "values": [a.value, b.value], "different_conditions": varying,
            "missing_conditions": missing, "quotes": [a.quote, b.quote],
            "interpretation": (
                "Explicit conditions differ; do not treat values as a contradiction."
                if varying else
                "Different reported values for a candidate matching context; do not select a winner or infer a cause."
            ),
            "context_certified": False,
        })
    return events
