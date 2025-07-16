from copy import deepcopy
from typing import Any, Optional
from ..type import Violation
from ..client import LLMClient, LLMUsage
from ..chunker.models import QUSComponent
from ..chunker.parser import Template


class UniformAnalyzer:
    """Analyzer for enforcing template uniformity across user stories.

    Identifies the most common template pattern among user stories and flags
    stories that deviate significantly from this pattern.
    """

    __PUNCT = '[PUNCT]'
    """Special token for punctuation marks"""

    __PUNCT_WEIGHT = 0.2
    """Reduced weight for punctuation in distance calculations"""

    __THRESHOLD = 3
    """Maximum allowed template deviation score"""

    @classmethod
    def __pos_distance(cls, list1: list[str], list2: list[str]) -> float:
        """Calculates weighted edit distance between two POS tag sequences.

        Args:
            list1 (list[str]): First sequence of POS tags
            list2 (list[str]): Second sequence of POS tags

        Returns:
            float: Weighted edit distance where punctuation has reduced impact

        Note:
            Uses dynamic programming with custom weights:
            - Regular tags have weight 1.0
            - Punctuation tags have weight __PUNCT_WEIGHT (0.2)
        """
        m, n = len(list1), len(list2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        # Initialize first row and column
        for i in range(1, m + 1):
            weight = cls.__PUNCT_WEIGHT if list1[i - 1] == cls.__PUNCT else 1
            dp[i][0] = dp[i - 1][0] + weight

        for j in range(1, n + 1):
            weight = cls.__PUNCT_WEIGHT if list2[j - 1] == cls.__PUNCT else 1
            dp[0][j] = dp[0][j - 1] + weight

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                w_i = cls.__PUNCT_WEIGHT if list1[i - 1] == cls.__PUNCT else 1
                w_j = cls.__PUNCT_WEIGHT if list2[j - 1] == cls.__PUNCT else 1
                candidates = [
                    dp[i - 1][j] + w_i,  # Deletion
                    dp[i][j - 1] + w_j,  # Insertion
                    dp[i - 1][j - 1] + w_i + w_j,  # Substitution
                ]
                if list1[i - 1] == list2[j - 1]:  # Match
                    candidates.append(dp[i - 1][j - 1])
                dp[i][j] = min(candidates)

        return dp[m][n]

    @classmethod
    def __existence_handling(cls, temp1: Template, temp2: Template, token: str, optional: bool = False) -> float:
        """Calculates distance penalty for missing components.

        Args:
            temp1 (Template): First template to compare
            temp2 (Template): Second template to compare
            token (str): Component token to check ('[ROLE]', '[MEANS]', '[ENDS]')
            optional (bool): Whether the component is optional (default: False)

        Returns:
            float: Distance penalty (1.0 if required component missing, 
                   otherwise POS distance between components)
        """
        dist = 0.0
        if token not in temp1.chunk or token not in temp2.chunk:
            if not optional:
                sep1 = temp1.chunk.get(token, [])
                sep2 = temp2.chunk.get(token, [])
                if len(sep1) == 0 and len(sep2) == 0:
                    dist = 1  # Both missing gets full penalty
                else:
                    dist = cls.__pos_distance(sep1, sep2)  # Partial penalty
            # Clean up for subsequent comparisons
            temp1.chunk.pop(token, None)
            temp2.chunk.pop(token, None)
            temp1.order = [t for t in temp1.order if t != token]
            temp2.order = [t for t in temp2.order if t != token]
        return dist

    @classmethod
    def __get_index(cls, seq: list, value: Any) -> Optional[int]:
        """Finds first index of value in sequence.

        Args:
            seq (list): Sequence to search
            value (Any): Value to find

        Returns:
            int: Index if found, None otherwise
        """
        for i, v in enumerate(seq):
            if v == value:
                return i
        return None

    @classmethod
    def __order_handling(cls, temp1: Template, temp2: Template, token: str, optional: bool = False) -> float:
        """Calculates distance penalty for component order mismatches.

        Args:
            temp1 (Template): First template to compare
            temp2 (Template): Second template to compare
            token (str): Component token to check
            optional (bool): Whether the component is optional

        Returns:
            float: Combined distance from POS difference and order mismatch
        """
        idx1 = cls.__get_index(temp1.order, token)
        idx2 = cls.__get_index(temp2.order, token)
        chunk1 = temp1.chunk.get(token, [])
        chunk2 = temp2.chunk.get(token, [])
        dist = cls.__pos_distance(chunk1, chunk2)

        # Add order mismatch penalty
        if idx1 != idx2:
            w1 = sum([(cls.__PUNCT_WEIGHT if c == cls.__PUNCT else 1)
                     for c in chunk1])
            w2 = sum([(cls.__PUNCT_WEIGHT if c == cls.__PUNCT else 1)
                     for c in chunk2])
            dist += max(w1, w2)

        # Add existence penalty if required
        if not optional and (token not in temp1.chunk or token not in temp2.chunk):
            dist += 1

        return dist

    @classmethod
    def __template_distance(cls, temp1: Template, temp2: Template) -> float:
        """Calculates comprehensive distance between two templates.

        Args:
            temp1 (Template): First template to compare
            temp2 (Template): Second template to compare

        Returns:
            float: Total distance score considering:
                   - Tail content
                   - Component existence
                   - Component order
                   - POS patterns
        """
        total_distance = 0.0
        _temp1 = deepcopy(temp1)
        _temp2 = deepcopy(temp2)

        # Penalize tail mismatch
        if temp1.tail or temp2.tail:
            total_distance += 1

        # Check component existence
        total_distance += cls.__existence_handling(_temp1, _temp2, '[ROLE]')
        total_distance += cls.__existence_handling(_temp1, _temp2, '[MEANS]')
        total_distance += cls.__existence_handling(
            _temp1, _temp2, '[ENDS]', optional=True)

        # Check component order and patterns
        total_distance += cls.__order_handling(_temp1, _temp2, '[ROLE]')
        total_distance += cls.__order_handling(_temp1, _temp2, '[MEANS]')
        total_distance += cls.__order_handling(
            _temp1, _temp2, '[ENDS]', optional=True)

        return total_distance

    @classmethod
    def __find_top_template(cls, templates: list[Template]) -> int:
        """Identifies the most representative template.

        Args:
            templates (list[Template]): List of templates to analyze

        Returns:
            int: Index of template with minimal total distance to others
        """
        data: list[tuple[int, float]] = []
        for i, tmp1 in enumerate(templates):
            total = sum([cls.__template_distance(tmp1, tmp2)
                        for tmp2 in templates])
            data.append((i, total))
        return min(data, key=lambda d: d[1])[0]

    @classmethod
    def __generate_violation_data(cls, text_template: str, component: QUSComponent) -> Violation:
        """Generates violation details for template deviations.

        Args:
            text_template (str): The expected template pattern
            component (QUSComponent): The deviating user story component

        Returns:
            Violation: Structured suggestion for template alignment
        """
        role = ' or '.join(component.role) or '[ROLE]'
        means = component.means or '[MEANS]'
        ends = component.ends or '[ENDS]'

        return Violation(
            parts={'template'},
            issue=f"User story doesn't match the frequent template: \"{text_template}\"",
            suggestion=f'Consider reformatting to: "{text_template.format(ROLE=role, MEANS=means, ENDS=ends)}"'
        )

    @classmethod
    def run(cls, client: LLMClient, model_idx: int, components: list[QUSComponent]) -> list[tuple[list[Violation], dict[str, LLMUsage]]]:
        """Analyzes user stories for template uniformity.

        Args:
            client (LLMClient): LLM client (unused, maintained for interface consistency)
            model_idx (int): Model index (unused, maintained for interface consistency)
            components (list[QUSComponent]): List of user story components to analyze

        Returns:
            List of (violations, usage) tuples for each input component

        Note:
            - Identifies the most common template pattern
            - Flags stories deviating beyond __THRESHOLD
            - Returns empty violations for compliant stories
        """
        templates = [comp.template for comp in components]
        if not templates:
            return [([], {}) for _ in components]

        top_idx = cls.__find_top_template(templates)
        top_template = templates[top_idx]

        return [
            ([cls.__generate_violation_data(top_template.text, components[i])], {})
            if cls.__template_distance(top_template, comp.template) > cls.__THRESHOLD
            else ([], {})
            for i, comp in enumerate(components)
        ]
