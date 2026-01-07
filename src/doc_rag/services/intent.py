from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentPlan:
    name: str
    preferred_sections: tuple[str, ...]
    prompt_style: str  # "about" | "objectives_conclusions"


_OBJ_RE = re.compile(r"\b(objetiv|aim|goal|purpose|contribution)\b", re.I)
_CONC_RE = re.compile(r"\b(conclus|findings?|takeaways?)\b", re.I)


def infer_intent(question: str) -> IntentPlan:
    q = question.strip()
    if _OBJ_RE.search(q) or _CONC_RE.search(q):
        return IntentPlan(
            name="objectives_conclusions",
            preferred_sections=("abstract", "introduction", "conclusion", "discussion"),
            prompt_style="objectives_conclusions",
        )
    return IntentPlan(
        name="about",
        preferred_sections=("abstract", "introduction", "results", "discussion", "conclusion"),
        prompt_style="about",
    )
