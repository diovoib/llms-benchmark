from __future__ import annotations

import re
from typing import Any


class Matcher:
    def matches(self, actual: Any, *, present: bool) -> bool:
        raise NotImplementedError


class _MatchAny(Matcher):
    def matches(self, actual: Any, *, present: bool) -> bool:
        return present


class _MatchAnyOrNull(Matcher):
    def matches(self, actual: Any, *, present: bool) -> bool:
        return True


class _MatchNull(Matcher):
    """Key missing or JSON null. Does not match \"\"."""

    def matches(self, actual: Any, *, present: bool) -> bool:
        if not present:
            return True
        return actual is None


class _MatchEmpty(Matcher):
    """Matches \"\" only, not null and not a missing key."""

    def matches(self, actual: Any, *, present: bool) -> bool:
        return present and actual == ""


class MatchRegex(Matcher):
    def __init__(self, pattern: str):
        self.pattern = pattern
        self._re = re.compile(pattern)

    def matches(self, actual: Any, *, present: bool) -> bool:
        if not present or not isinstance(actual, str):
            return False
        return self._re.search(actual) is not None


class MatchEqual(Matcher):
    def __init__(self, expected: Any, *, ignore_case: bool = False):
        self.expected = expected
        self.ignore_case = ignore_case

    def matches(self, actual: Any, *, present: bool) -> bool:
        if not present:
            return False
        if self.ignore_case:
            return str(actual).lower() == str(self.expected).lower()
        return actual == self.expected


def parse_hour(value: Any) -> int | None:
    """Clock hour 0–23. '9:00' → 9; '00:09' → 0 (minutes are not the hour)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        hour = int(value)
        return hour if 0 <= hour <= 23 else None
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace(".", "")
    meridiem = None
    if text.endswith("am") or " am" in f" {text} ":
        meridiem = "am"
        text = text.replace("am", "")
    elif text.endswith("pm") or " pm" in f" {text} ":
        meridiem = "pm"
        text = text.replace("pm", "")
    text = text.strip()
    if not text:
        return None
    head = text.split(":")[0].strip()
    try:
        hour = int(head)
    except ValueError:
        return None
    if meridiem == "am":
        if hour == 12:
            hour = 0
    elif meridiem == "pm":
        if hour != 12:
            hour += 12
    return hour if 0 <= hour <= 23 else None


class MatchWhen(Matcher):
    def __init__(self, date: str, hour: int):
        self.date = date
        self.hour = hour

    def matches(self, actual: Any, *, present: bool) -> bool:
        if not present or not isinstance(actual, dict):
            return False
        if str(actual.get("date") or "") != str(self.date):
            return False
        return parse_hour(actual.get("hour")) == int(self.hour)


MatchAny = _MatchAny()
MatchAnyOrNull = _MatchAnyOrNull()
MatchNull = _MatchNull()
MatchEmpty = _MatchEmpty()
