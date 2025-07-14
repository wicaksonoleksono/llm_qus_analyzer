from typing import Callable, Optional
from .client import LLMClient, LLMResult
from .type import Violation
from .chunker.models import UserStoryComponent


def analyze_individual_with_basic(
    checkers: list[Callable[[UserStoryComponent], Optional[Violation]]],
    component: UserStoryComponent
):
    violations: list[Violation] = []
    for checker in checkers:
        violation = checker(component)
        if violation:
            violations.append(violation)
    return violations


def analyze_individual_with_llm(
    checkers: list[Callable[[LLMClient, int, UserStoryComponent], tuple[Optional[Violation], Optional[LLMResult]]]],
    client: LLMClient, model_idx: int, component: UserStoryComponent
):
    violations: list[Violation] = []
    results: list[Optional[LLMResult]] = []
    for checker in checkers:
        violation, result = checker(client, model_idx, component)
        if violation:
            violations.append(violation)
        results.append(result)
    return violations, results
