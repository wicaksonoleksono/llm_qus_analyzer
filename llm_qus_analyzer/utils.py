from typing import Callable, Optional
from .client import LLMClient, LLMUsage
from .type import Violation
from .chunker.models import QUSComponent


def analyze_individual_with_basic(
    checkers: list[Callable[[QUSComponent], Optional[Violation]]],
    component: QUSComponent
) -> list[Violation]:
    """Executes a series of basic validation checks on a user story component.

    Args:
        checkers: List of validation functions that each take a QUSComponent
                 and return an optional Violation
        component (QUSComponent): The user story component to analyze

    Returns:
        list[Violation]: Aggregated list of all violations found by the checkers

    Example:
        >>> checkers = [lambda c: Violation(...) if ... else None]
        >>> component = QUSComponent(...)
        >>> violations = analyze_individual_with_basic(checkers, component)
    """
    violations: list[Violation] = []
    for checker in checkers:
        violation = checker(component)
        if violation:
            violations.append(violation)
    return violations


def analyze_individual_with_llm(
    checkers: list[Callable[[LLMClient, int, QUSComponent], tuple[Optional[Violation], Optional[LLMUsage]]]],
    client: LLMClient,
    model_idx: int,
    component: QUSComponent
) -> tuple[list[Violation], list[Optional[LLMUsage]]]:
    """Executes a series of LLM-powered validation checks on a user story component.

    Args:
        checkers: List of validation functions that each take:
                  - LLMClient: The client to use for analysis
                  - int: Model index to use
                  - QUSComponent: Component to analyze
                  and returns a tuple of (Optional[Violation], Optional[LLMUsage])
        client (LLMClient): Configured LLM client instance
        model_idx (int): Index of the LLM model to use
        component (QUSComponent): The user story component to analyze

    Returns:
        tuple[list[Violation], list[Optional[LLMUsage]]]:
            - List of all violations found
            - List of corresponding LLM usage metrics for each checker

    Note:
        The usage metrics list maintains 1:1 correspondence with the input checkers list,
        containing None for checkers that didn't perform LLM operations.
    """
    violations: list[Violation] = []
    usages: list[Optional[LLMUsage]] = []
    for checker in checkers:
        violation, usage = checker(client, model_idx, component)
        if violation:
            violations.append(violation)
        usages.append(usage)
    return violations, usages
