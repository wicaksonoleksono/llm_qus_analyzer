from dataclasses import dataclass


@dataclass
class Violation:
    parts: set[str]
    issue: str
    suggestion: str
