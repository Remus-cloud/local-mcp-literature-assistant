"""Small, deterministic numeric primitives. No model, eval(), or external I/O.

Passing arithmetic is not a finding about the source, metric or scientific claim.
Unsupported syntax and ambiguous PDF layout must abstain.
"""
import ast
from decimal import Decimal, InvalidOperation, localcontext
import re

NUMBER = r"[+-]?\d+(?:\.\d+)?"
TOKEN = rf"{NUMBER}%?"
LITERAL = re.compile(rf"(?<![A-Za-z0-9_.]){TOKEN}(?![A-Za-z0-9_.])")


def normalize(text):
    text = text.translate(str.maketrans({"−": "-", "×": "*", "÷": "/", "％": "%", "＝": "="}))
    return re.sub(r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])",
                  lambda m: m.group().replace(",", ""), text)


def decimal_value(token):
    token = normalize(token.strip())
    return Decimal(token.rstrip("%")) / (100 if token.endswith("%") else 1)


def calculate(expression):
    """Evaluate a bounded grammar: decimal literals, + - * / and parentheses."""
    expression = normalize(expression).strip()
    if len(expression) > 256 or not re.fullmatch(r"[\d\s.+*/()%\-]+", expression):
        raise ValueError("unsupported_expression")
    expression = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expression)
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 100:
        raise ValueError("expression_too_complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Decimal(ast.get_source_segment(expression, node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if not right:
                raise ValueError("zero_denominator")
            return left / right
        raise ValueError("unsupported_expression")

    with localcontext() as context:
        context.prec = 40
        result = visit(tree.body)
        if not result.is_finite() or abs(result) > Decimal("1e30"):
            raise ValueError("result_out_of_bounds")
        return result


def check_equation(expression, stated, approximate=False, percentage_points=False):
    try:
        expected = calculate(expression)
        observed = decimal_value(stated)
        # 88.4% - 84.9% = 3.5 percentage points; keep fractional scale.
        if percentage_points and "%" in expression and not stated.endswith("%"):
            observed /= 100
        # Unmarked source numbers may be fractions or 0..100 scores. Do not guess.
        if stated.endswith("%") and "%" not in expression:
            return {"status": "review", "reason": "percentage_scale_requires_definition",
                    "computed": str(expected), "stated": str(observed)}
        precision = Decimal(stated.rstrip("%")).as_tuple().exponent
        tolerance = Decimal(5).scaleb(precision - 1) if approximate else Decimal(0)
        if stated.endswith("%") or (percentage_points and "%" in expression):
            tolerance /= 100
        return {
            "status": "pass" if abs(expected - observed) <= tolerance else "conflict",
            "reason": "arithmetic_only", "computed": str(expected), "stated": str(observed),
            "tolerance": str(tolerance), "approximate": approximate,
        }
    except (ValueError, SyntaxError, InvalidOperation, OverflowError, RecursionError) as error:
        return {"status": "review", "reason": str(error) if isinstance(error, ValueError)
                else "unsupported_expression"}


def identity(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def table_facts(text):
    """Read only explicit Tasks / metric / fixed-width rows / Table caption.

    Supports a narrow PDF text layout (including BERT Table 5), not a universal
    table extractor. No known paper IDs, row names or values are hard-coded.
    Full row, header and caption spans are retained for audit.
    """
    lines = text.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    facts = []
    for i, line in enumerate(lines):
        header = re.match(r"^\s*Tasks\s+(.+)", line)
        if not header or i + 1 >= len(lines):
            continue
        metric_tokens = re.findall(r"\((Acc|F1|EM|BLEU)\)", lines[i + 1], re.I)
        columns = header.group(1).split()[:len(metric_tokens)]
        if len(columns) < 2 or len(set(columns)) != len(columns):
            continue
        prefix = "".join(lines[max(0, i - 2):i + 1])
        splits = re.findall(r"\b(Dev|Test|Train)\s+Set\b", prefix, re.I)
        if len(set(s.lower() for s in splits)) != 1:
            continue
        caption_index = next((j for j in range(i + 2, min(i + 20, len(lines)))
                              if re.match(r"^\s*Table\s+\d+\s*:", lines[j], re.I)), None)
        if caption_index is None:
            continue
        table = re.search(r"Table\s+(\d+)", lines[caption_index], re.I).group(1)
        rows = []
        pattern = rf"^\s*(.+?)\s{{2,}}((?:{TOKEN}\s+){{{len(columns)-1}}}{TOKEN})(?:\s{{2,}}(.*))?\s*$"
        ambiguous = False
        for j in range(i + 2, caption_index):
            match = re.match(pattern, lines[j].rstrip())
            if match:
                if match.group(3) and re.match(rf"\s*{TOKEN}(?:\s|$)", match.group(3)):
                    ambiguous = True  # extra numeric column: do not shift silently
                    break
                values = match.group(2).split()
                rows.append((j, match.group(1).strip(), values))
        if ambiguous or len(rows) < 2 or len({identity(r[1]) for r in rows}) != len(rows):
            continue
        for j, model, values in rows:
            for dataset, metric, value in zip(columns, metric_tokens, values):
                facts.append({
                    "model": model, "dataset": dataset, "metric": metric.lower(),
                    "split": splits[0].lower(), "table": table, "value": value,
                    "row_quote": lines[j].rstrip(), "row_start": offsets[j],
                    "header_quote": "".join(lines[max(0, i - 2):i + 2]),
                    "caption_quote": lines[caption_index].rstrip(),
                    "reported_scale": "percent" if value.endswith("%") else "table_display_units",
                })
    return facts
