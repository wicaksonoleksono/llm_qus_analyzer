from typing import Callable, Optional, Any, Type
from .client import LLMClient, LLMUsage
from .type import Violation, PairwiseViolation, FullSetViolation
from .chunker.models import QUSComponent


def analyze_individual_with_basic(
    checkers: list[Callable[[QUSComponent], Optional[Violation]]],
    component: QUSComponent,
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
    checkers: list[
        Callable[
            [LLMClient, int, QUSComponent],
            tuple[Optional[Violation] | list[Violation], Optional[LLMUsage]],
        ]
    ],
    client: LLMClient,
    model_idx: int,
    component: QUSComponent,
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
        tuple[list[Violation],list[Optional[LLMUsage]]]:
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
            if isinstance(violation, list):
                violations.extend(violation)
            else:
                violations.append(violation)
        usages.append(usage)
    return violations, usages


def analyze_set_pairwise(
    analyzer_class: Type[Any],
    client: LLMClient,
    model_idx: int,
    components: list[QUSComponent]
) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
    """Generic pairwise analysis for any set analyzer.

    Args:
        analyzer_class: The analyzer class (e.g., ConflictFreeAnalyzer)
        client: LLM client for analysis
        model_idx: Index of the LLM model to use
        components: List of components to analyze

    Returns:
        Tuple containing list of all pairwise violations and LLM usage data
    """
    all_violations: list[PairwiseViolation] = []
    all_usages: dict[str, LLMUsage] = {}

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            violations, usages = analyzer_class.analyze_pairwise(
                client, model_idx, components[i], components[j]
            )
            all_violations.extend(violations)

            # Merge usage data with unique keys
            for key, usage in usages.items():
                # Use component IDs if available, otherwise fall back to indices
                first_id = components[i].id or str(i)
                second_id = components[j].id or str(j)
                all_usages[f"{key}_pair_{first_id}_{second_id}"] = usage

    return all_violations, all_usages


def analyze_set_fullset(
    analyzer_class: Type[Any],
    client: LLMClient,
    model_idx: int,
    components: list[QUSComponent]
) -> tuple[list[FullSetViolation], dict[str, LLMUsage]]:
    """Generic fullset analysis for any set analyzer.

    Args:
        analyzer_class: The analyzer class (e.g., ConflictFreeAnalyzer)
        client: LLM client for analysis
        model_idx: Index of the LLM model to use
        components: List of components to analyze

    Returns:
        Tuple containing list of fullset violations and LLM usage data
    """
    return analyzer_class.analyze_full_set(client, model_idx, components)


def format_set_results_pairwise(
    violations: list[PairwiseViolation],
    usages: dict[str, LLMUsage],
    components: list[QUSComponent]
) -> list[tuple[list[PairwiseViolation], dict[str, LLMUsage]]]:
    """Format pairwise analysis results in the expected format for set analyzers.

    Args:
        violations: List of pairwise violations found
        usages: Dictionary of LLM usage data
        components: List of components that were analyzed

    Returns:
        List of (violations, usage) tuples where first component gets all violations,
        others get empty results
    """
    if violations:
        return [(violations, usages)] + [([], {}) for _ in components[1:]]
    else:
        return [([], {}) for _ in components]


def format_set_results_fullset(
    violations: list[FullSetViolation],
    usages: dict[str, LLMUsage],
    components: list[QUSComponent]
) -> list[tuple[list[FullSetViolation], dict[str, LLMUsage]]]:
    """Format fullset analysis results in the expected format for set analyzers.

    Args:
        violations: List of fullset violations found
        usages: Dictionary of LLM usage data
        components: List of components that were analyzed

    Returns:
        List of (violations, usage) tuples where first component gets all violations,
        others get empty results
    """
    if violations:
        return [(violations, usages)] + [([], {}) for _ in components[1:]]
    else:
        return [([], {}) for _ in components]
