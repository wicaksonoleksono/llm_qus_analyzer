from typing import Optional
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..utils import analyze_individual_with_basic
from ..type import Violation


class MinimalAnalyzer:
    """Analyzer for checking basic syntax and structure of user stories.

    Performs minimal validation checks to ensure user stories:
    - Contain only allowed special characters
    - Don't have unnecessary trailing information
    """

    @classmethod
    def __is_not_contain_special(cls, component: QUSComponent) -> Optional[Violation]:
        """Checks if the user story contains invalid special characters.

        Args:
            component (QUSComponent): The parsed user story component to validate

        Returns:
            Optional[Violation]:
                - Violation if invalid characters are found
                - None if only allowed characters are present

        Note:
            Allowed characters include:
            - Alphanumeric characters (a-z, A-Z, 0-9)
            - Basic punctuation (apostrophes, commas, periods, hyphens)
            - Spaces
        """
        us = component.text

        def not_valid(ch: str) -> bool:
            """Helper function to identify invalid characters.

            Args:
                ch (str): Single character to check

            Returns:
                bool: True if character is invalid, False otherwise
            """
            if ch.isalnum():
                return False
            if ch in ["'", ",", ".", "-", " "]:
                return False
            return True

        chars = [ch for ch in us if not_valid(ch)]
        chars = list(set(chars))  # Get unique invalid characters

        if len(chars) > 0:
            tmp = ", ".join(chars)
            return Violation(
                parts=set(["full"]),
                issue=f'The user story contains invalid characters: "{tmp}"',
                suggestion="Remove invalid characters from the user story",
            )

        return None

    @classmethod
    def __is_us_no_tail(cls, component: QUSComponent) -> Optional[Violation]:
        """Checks if the user story has unnecessary trailing information.

        Args:
            component (QUSComponent): The parsed user story component to validate

        Returns:
            Optional[Violation]:
                - Violation if trailing information is found
                - None if no trailing information exists

        Note:
            Trailing information refers to text after the main user story
            components that doesn't contribute to the core functionality.
        """
        tail = component.template.tail

        if tail:
            return Violation(
                parts=set(["full"]),
                issue=f'The user story contains unnecessary info: "{tail}"',
                suggestion="Remove unnecessary info from the user story.",
            )

        return None

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], dict[str, LLMUsage]]:
        """Executes all minimal validation checks on a user story.

        Args:
            client (LLMClient): LLM client (maintained for interface consistency)
            model_idx (int): Model index (maintained for interface consistency)
            component (QUSComponent): Parsed user story components to validate

        Returns:
            tuple[list[Violation],dict[str,LLMUsage]]:
                - List of found violations (empty if all checks pass)
                - Empty dictionary (maintained for interface consistency)

        Note:
            Current checks include:
            1. Valid character check
            2. Trailing information check
        """
        basic_checker = [
            cls.__is_not_contain_special,
            cls.__is_us_no_tail,
        ]

        violations = analyze_individual_with_basic(basic_checker, component)

        return violations, {}
