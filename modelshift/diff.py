"""Behavioral diffing: does the new model's output behave like the old one?

Each comparison yields pass/fail *checks* plus machine-readable *symptoms*
(code_fence, preamble, value_drift, refusal, ...) that the repair engine targets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .core import LLMResponse

if TYPE_CHECKING:
    from .suite import Case

REFUSAL_RE = re.compile(
    r"\b(I can(?:'|no)t (?:help|assist)|I(?:'m| am) (?:not able|unable) to|I won'?t be able to|"
    r"I(?:'m| am) sorry, but)\b",
    re.IGNORECASE,
)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    system: str | None
    user_input: str
    expected: LLMResponse
    actual: LLMResponse
    checks: list[Check] = field(default_factory=list)
    symptoms: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "user_input": self.user_input,
            "expected": self.expected.to_dict(),
            "actual": self.actual.to_dict(),
            "checks": [c.__dict__ for c in self.checks],
            "symptoms": self.symptoms,
        }


# ---------- JSON helpers ----------


def strict_json(text: str) -> Any:
    """Parse only if the *entire* output is a JSON object/array."""
    s = (text or "").strip()
    if not s or s[0] not in "{[":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, (dict, list)) else None


def lenient_json(text: str) -> Any:
    """Dig a JSON object out of fences / surrounding chatter."""
    obj = strict_json(text)
    if obj is not None:
        return obj
    m = FENCE_RE.search(text or "")
    if m:
        obj = strict_json(m.group(1))
        if obj is not None:
            return obj
    s = text or ""
    start, end = s.find("{"), s.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def _is_categorical(v: Any) -> bool:
    if isinstance(v, (bool, int)) or v is None:
        return True
    return isinstance(v, str) and len(v) <= 32 and len(v.split()) <= 3


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def validate_schema(obj: Any, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-schema validator (type/enum/required/properties/items/additionalProperties)."""
    errors: list[str] = []
    t = schema.get("type")
    if isinstance(t, str) and t in _TYPES:
        ok = isinstance(obj, _TYPES[t]) and not (t in ("integer", "number") and isinstance(obj, bool))
        if not ok:
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in obj:
                errors.append(f"{path}: missing required '{req}'")
        for k, v in obj.items():
            if k in props:
                errors += validate_schema(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected key '{k}'")
    if isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            errors += validate_schema(item, schema["items"], f"{path}[{i}]")
    return errors


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


# ---------- main comparison ----------


def compare(
    case: Case,
    actual: LLMResponse,
    *,
    similarity_threshold: float = 0.35,
    length_bounds: tuple[float, float] = (0.4, 2.5),
) -> CaseResult:
    from .core import last_user_text

    exp = case.expected
    res = CaseResult(case.id, case.request.system, last_user_text(case.request), exp, actual)
    checks, symptoms = res.checks, res.symptoms

    # 1. Refusal drift
    if is_refusal(actual.text) and not is_refusal(exp.text):
        checks.append(Check("refusal", False, actual.text[:160]))
        symptoms.append({"type": "refusal"})
        return res  # nothing else is meaningful

    # 2. Tool-call behavior
    if exp.tool_calls or actual.tool_calls:
        en = [c["name"] for c in exp.tool_calls]
        an = [c["name"] for c in actual.tool_calls]
        ok = en == an
        checks.append(Check("tool_names", ok, f"expected {en}, got {an}"))
        if not ok:
            symptoms.append({"type": "tool_mismatch", "expected": en, "got": an})
        else:
            for e, a in zip(exp.tool_calls, actual.tool_calls, strict=True):
                ek, ak = set(e["arguments"]), set(a["arguments"])
                good = ek == ak
                checks.append(
                    Check(f"tool_args:{e['name']}", good, f"expected keys {sorted(ek)}, got {sorted(ak)}")
                )
                if not good:
                    symptoms.append(
                        {
                            "type": "tool_args",
                            "tool": e["name"],
                            "missing": sorted(ek - ak),
                            "extra": sorted(ak - ek),
                        }
                    )

    # 3. Structured (JSON) output
    exp_json = strict_json(exp.text)
    if exp_json is not None:
        act_json = strict_json(actual.text)
        if act_json is None:
            act_json = lenient_json(actual.text)
            stripped = actual.text.strip()
            checks.append(Check("json_valid", False, stripped[:120]))
            if "```" in actual.text:
                symptoms.append({"type": "code_fence"})
            if act_json is not None and not stripped.startswith(("{", "[", "```")):
                symptoms.append({"type": "preamble", "prefix": stripped[: stripped.find("{")][:60]})
            if act_json is None:
                symptoms.append({"type": "invalid_json"})
        else:
            checks.append(Check("json_valid", True))

        if isinstance(exp_json, dict) and isinstance(act_json, dict):
            missing = sorted(set(exp_json) - set(act_json))
            extra = sorted(set(act_json) - set(exp_json))
            checks.append(Check("json_keys", not missing and not extra, f"missing={missing} extra={extra}"))
            if missing:
                symptoms.append({"type": "missing_keys", "keys": missing})
            if extra:
                symptoms.append({"type": "extra_keys", "keys": extra})
            drift = []
            for k in set(exp_json) & set(act_json):
                ev, av = exp_json[k], act_json[k]
                if _is_categorical(ev) and ev != av:
                    drift.append(f"{k}: {ev!r} -> {av!r}")
                    symptoms.append({"type": "value_drift", "key": k, "expected": ev, "got": av})
            checks.append(Check("json_values", not drift, "; ".join(drift)))

        rf = case.request.response_format or {}
        if rf.get("type") == "json_schema" and act_json is not None:
            errs = validate_schema(act_json, rf.get("json_schema", {}).get("schema", {}))
            checks.append(Check("schema", not errs, "; ".join(errs[:5])))
            if errs:
                symptoms.append({"type": "schema", "errors": errs[:5]})
        return res

    # 4. Free text
    if exp.text and not exp.tool_calls:
        ratio = len(actual.text) / max(1, len(exp.text))
        lo, hi = length_bounds
        checks.append(Check("length", lo <= ratio <= hi, f"ratio={ratio:.2f}"))
        if ratio > hi:
            symptoms.append({"type": "too_long", "ratio": round(ratio, 2)})
        elif ratio < lo:
            symptoms.append({"type": "too_short", "ratio": round(ratio, 2)})
        sim = _jaccard(exp.text, actual.text)
        checks.append(Check("similarity", sim >= similarity_threshold, f"jaccard={sim:.2f}"))
    return res
