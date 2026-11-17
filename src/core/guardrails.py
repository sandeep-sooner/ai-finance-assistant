"""Guardrails on both sides of the model.

Input:  empty / oversized messages, prompt-injection attempts, and requests for a personal
        buy/sell decision (allowed, but flagged so the agent explains instead of advising).
Output: every answer carries the educational disclaimer; if the text nevertheless reads like
        a directive ("you should buy ...") or a promise ("guaranteed returns"), the stronger
        disclaimer is used and the turn is flagged for the eval suite and the logs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DISCLAIMER = "Educational information only, not financial, investment or tax advice."
STRONG_DISCLAIMER = (
    "Educational information only. This is not a recommendation to buy or sell anything; "
    "consider speaking to a licensed adviser before acting."
)

_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|prompts|rules)"
    r"|disregard (all |any )?(previous|prior|above|your) (instructions|prompts|rules)"
    r"|reveal (your )?(system )?prompt|show (me )?your (system )?(prompt|instructions)"
    r"|you are now|act as (?:an? )?(?:unrestricted|jailbroken|dan)\b|developer mode|jailbreak",
    re.I,
)
_ADVICE = re.compile(
    r"\bshould i (buy|sell|invest in|put (all )?my money|short|hold)\b|\bwhich (stock|etf|fund|crypto)s? (should|to) (i )?buy\b"
    r"|\bis (it|now) a good time to (buy|sell)\b|\bwhat should i (buy|invest in)\b|\b(tell|give) me (a |the )?(best )?stocks? to buy\b",
    re.I,
)
_DIRECTIVE = re.compile(
    r"\byou should (definitely |really )?(buy|sell|short|dump)\b|\bi (strongly )?recommend (you )?(buy|sell|buying|selling)\b"
    r"|\b(buy|sell) (it|this stock|these shares) (now|today|immediately)\b", re.I)
_PROMISE = re.compile(r"\bguaranteed? (returns?|profits?|gains?)\b|\brisk[- ]free (returns?|profits?)\b|\bcan(?:'|no)t lose\b", re.I)


@dataclass(frozen=True)
class InputCheck:
    ok: bool
    reason: str = ""
    advice_request: bool = False


@dataclass(frozen=True)
class OutputCheck:
    disclaimer: str
    flags: list[str] = field(default_factory=list)


def check_input(text: str, max_chars: int = 2000) -> InputCheck:
    text = (text or "").strip()
    if not text:
        return InputCheck(False, "Please type a question.")
    if len(text) > max_chars:
        return InputCheck(False, f"That message is too long. Please keep it under {max_chars} characters.")
    if _INJECTION.search(text):
        return InputCheck(False, "I can't follow instructions that try to change how I work. Ask me a finance question instead.")
    return InputCheck(True, advice_request=bool(_ADVICE.search(text)))


def check_output(text: str) -> OutputCheck:
    flags = []
    if _DIRECTIVE.search(text or ""):
        flags.append("directive_language")
    if _PROMISE.search(text or ""):
        flags.append("promised_returns")
    return OutputCheck(STRONG_DISCLAIMER if flags else DISCLAIMER, flags)
