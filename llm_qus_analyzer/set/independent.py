from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation, PairwiseViolation, FullSetViolation
from ..utils import analyze_set_pairwise, analyze_set_fullset, format_set_results_pairwise, format_set_results_fullset
from dataclasses import dataclass
from typing import Any, Optional

_definition = """
**Evaluate whether two user stories are 'Independent' based on their [Means], [Ends], and [Role]:**
Independent: a user story is self-contained, schedulable and implementable in any order, without hidden prerequisites on other stories.

1. **[Means] Causality Check**  
    – Do the [Means] across stories avoid requires-before relations between verbs on the same object state (one action is impossible without the other)?
    NOTE: 
        if  the [Means] describe **soft overlaps—actions on the same object that are equally schedulable and testable in isolation and therefore remain independent it's a valid userstory
        to check if its a causal statement, Can it be both implemented independently
2. **[Means] Superclass/Object Check**  
    – Do the [Means] across stories avoid superclass–subclass coupling on the direct object that would make one story's object subsume the other?
    NOTE: Sharing the same object is allowed. Do not flag just because both stories act on the same object or a field of it. Only flag when the direct objects are different and there is a real is-a relation (one truly generalizes the other).

3. **[Ends] Purpose = Means Check**  
    – Do the [Ends] across stories avoid stating a purpose that is semantically identical to another story's [Means] (same action verb + direct object, paraphrase-equivalent)?

4. **[Role] Same Story, Different Roles (impact note)**  
    – Different roles with same means and/or ends indicate relation but do not by themselves break independence if all checks above are YES.

**Suggestions to fix:**  
– Add an explicit dependency when hard causality exists.
– Replace superclasses with explicit subtypes or split per subtype.
– When an End is semantically identical to another story's Means, promote that End to its own Means or link the stories with a dependency.
– Keep role variants separate only if each is independently shippable/testable; otherwise consolidate or add a dependency.
– Clarify object scope to avoid subsumption; use precise objects instead of umbrella terms.
– For soft overlaps, ensure isolation via defined interfaces/contracts so each story can be scheduled and tested independently.

"""

_in_format = """
**User Stories to Evaluate:**  
id: {id1} user story
- [Role]: {r1}
- [Means]: {m1}
- [Ends]: {e1}

id: {id2} user story 
- [Role]: {r2}
- [Means]: {m2}
- [Ends]: {e2}
"""

_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If independent: `{{"valid": true}}`
- If dependent:
  ```json
  {{
      "valid": false,
      "violations": [
        {{
            "id_pair": {{"first": 0, "second": 1}},
            "issue": "Description of the dependency type and how they are dependent",
            "first_suggestion": "How to make the first user story independent", 
            "second_suggestion": "How to make the second user story independent"
        }}
      ]
  }}
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_all_set_definition = """
**Evaluate whether multiple user stories are 'Independent' across the entire set based on their [Means], [Ends], and [Role]:**
Independent: a user story is self-contained, schedulable and implementable in any order, without hidden prerequisites on other stories.

1. **[Means] Causality Check**  
    – Do the [Means] across stories avoid requires-before relations between verbs on the same object state (one action is impossible without the other)?
    NOTE: if  the [Means] describe **soft overlaps—actions on the same object that are equally schedulable and testable in isolation and therefore remain independent it's a valid userstory

2. **[Means] Superclass/Object Check**  
    – Do the [Means] across stories avoid superclass–subclass coupling on the direct object that would make one story's object subsume the other?
    NOTE: Sharing the same object is allowed. Do not flag just because both stories act on the same object or a field of it. Only flag when the direct objects are different and there is a real is-a relation (one truly generalizes the other).

3. **[Ends] Purpose = Means Check**  
    – Do the [Ends] across stories avoid stating a purpose that is semantically identical to another story's [Means] (same action verb + direct object, paraphrase-equivalent)?

4. **[Role] Same Story, Different Roles (impact note)**  
    – Different roles with same means and/or ends indicate relation but do not by themselves break independence if all checks above are YES.

**Suggestions to fix:**  
– Add an explicit dependency when hard causality exists.
– Replace superclasses with explicit subtypes or split per subtype.
– When an End is semantically identical to another story's Means, promote that End to its own Means or link the stories with a dependency.
– Keep role variants separate only if each is independently shippable/testable; otherwise consolidate or add a dependency.
– Clarify object scope to avoid subsumption; use precise objects instead of umbrella terms.
– For soft overlaps, ensure isolation via defined interfaces/contracts so each story can be scheduled and tested independently.

"""

_all_set_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_all_set_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If independent: `{{"valid": true}}`
- If dependent:
```json
{{
    "valid": false,
    "violations": [
      {{
          "story_ids": [1, 2, 3],
          "parts_per_story": [["means"], ["ends"], ["means", "ends"]],
          "issue": "Description of the dependency across these stories",
          "suggestion": "How to make these stories independent"
      }}
    ]
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class IndependentVerdictData:
    """Data class representing the verdict of an independence analysis."""

    valid: bool
    """Boolean indicating whether the components are independent."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


@dataclass
class IndependentFullSetVerdictData:
    """Data class representing the verdict of a full-set independence analysis."""

    valid: bool
    """Boolean indicating whether the components are independent."""

    violations: list[FullSetViolation]
    """List of FullSetViolation objects found in the analysis."""


def format_stories_list(components: list[QUSComponent]) -> str:
    """Formats a list of QUSComponent objects into a structured story list for LLM input.

    Args:
        components: List of QUSComponent objects to format

    Returns:
        Formatted string with numbered stories using structured format
    """
    stories = []
    for i, comp in enumerate(components):
        story = f"Story {i+1}:\n- [Role]: {comp.role}\n- [Means]: {comp.means}\n- [Ends]: {comp.ends}"
        stories.append(story)
    return "\n\n".join(stories)


_PART_MAP = {
    "[Role]": "role",
    "[Means]": "means",
    "[Ends]": "ends",
}


class IndependentParserModel:
    """Unified parser for independence analysis supporting both pairwise and fullset modes."""

    def __init__(self, mode: str):
        """Initialize parser with specified mode.

        Args:
            mode: Either "pairwise" or "fullset"
        """
        if mode not in ["pairwise", "fullset"]:
            raise ValueError("Mode must be 'pairwise' or 'fullset'")

        self.mode = mode
        self.key = f"independent-{mode}"

        if mode == "pairwise":
            self.__analyzer = LLMAnalyzer[IndependentVerdictData](key=self.key)
            self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        else:  # fullset
            self.__analyzer = LLMAnalyzer[IndependentFullSetVerdictData](key=self.key)
            self.__analyzer.build_prompt(_all_set_definition, _all_set_in_format, _all_set_out_format)

        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> IndependentVerdictData | IndependentFullSetVerdictData:
        """Parses raw JSON output from LLM into structured data.
        Args:
            raw_json: Raw JSON output from the LLM analysis.
        Returns:
            IndependentVerdictData or IndependentFullSetVerdictData depending on mode.
        """
        if not isinstance(raw_json, dict):
            if self.mode == "pairwise":
                return IndependentVerdictData(True, [])
            else:
                return IndependentFullSetVerdictData(True, [])

        valid = raw_json.get("valid", True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = True

        if self.mode == "pairwise":
            return self.__parse_pairwise(raw_json, valid)
        else:
            return self.__parse_fullset(raw_json, valid)

    def __parse_pairwise(self, raw_json: dict, valid: bool) -> IndependentVerdictData:
        """Parse pairwise analysis result."""
        violations: list[Violation] = []
        default_vio = Violation({}, "Unknown dependency", "Review stories for independence")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    # Extract id_pair for independence pair identification
                    id_pair = t.get("id_pair", {})
                    first_idx = id_pair.get("first", 0) if isinstance(id_pair, dict) else 0
                    second_idx = id_pair.get("second", 1) if isinstance(id_pair, dict) else 1

                    # Default parts for independence analysis
                    first_parts = {"role", "means", "ends"}
                    second_parts = {"role", "means", "ends"}

                    # Store both parts in violation for later PairwiseViolation creation
                    violation = Violation(
                        parts=first_parts.union(second_parts),
                        issue=t.get("issue", ""),
                        suggestion=t.get("first_suggestion", t.get("suggestion", "")),
                    )
                    # Store additional data for PairwiseViolation
                    violation._first_parts = first_parts
                    violation._second_parts = second_parts
                    violation._second_suggestion = t.get("second_suggestion", "")
                    violation._id_pair = {"first": first_idx, "second": second_idx}

                    violations.append(violation)
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return IndependentVerdictData(valid=valid, violations=violations)

    def __parse_fullset(self, raw_json: dict, valid: bool) -> IndependentFullSetVerdictData:
        """Parse fullset analysis result."""
        violations: list[FullSetViolation] = []
        default_vio = FullSetViolation([], [], "Unknown dependency", "Review stories for independence")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    story_ids = t.get("story_ids", [])
                    if isinstance(story_ids, list):
                        story_ids = [int(sid) - 1 for sid in story_ids if isinstance(sid,
                                                                                     (int, str)) and str(sid).isdigit()]
                    else:
                        story_ids = []

                    parts_per_story = t.get("parts_per_story", [])
                    if not isinstance(parts_per_story, list):
                        parts_per_story = []

                    # Convert string parts to sets
                    processed_parts = []
                    for parts in parts_per_story:
                        if isinstance(parts, list):
                            part_set = set()
                            for part in parts:
                                mapped_part = _PART_MAP.get(f"[{part.capitalize()}]", part.lower())
                                part_set.add(mapped_part)
                            processed_parts.append(part_set)
                        else:
                            processed_parts.append(set())

                    violations.append(
                        FullSetViolation(
                            story_ids=story_ids,
                            parts_per_story=processed_parts,
                            issue=t.get("issue", ""),
                            suggestion=t.get("suggestion", "")
                        )
                    )
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return IndependentFullSetVerdictData(valid=valid, violations=violations)

    def analyze_pairwise(
        self, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], LLMResult | None]:
        """Analyzes two QUS components for independence violations.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component1 (QUSComponent): First QUSComponent to compare.
            component2 (QUSComponent): Second QUSComponent to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM result.
        """
        if self.mode != "pairwise":
            raise ValueError("This parser is not in pairwise mode")

        values = {
            "id1": component1.id,
            "id2": component2.id,
            "r1": component1.role,
            "m1": component1.means,
            "e1": component1.ends,
            "r2": component2.role,
            "m2": component2.means,
            "e2": component2.ends,
        }
        data, usage = self.__analyzer.run(client, model_idx, values)

        pairwise_violations: list[PairwiseViolation] = []
        for violation in data.violations:
            # Use stored parts from parser if available, otherwise fallback to same parts
            first_parts = getattr(violation, '_first_parts', violation.parts)
            second_parts = getattr(violation, '_second_parts', violation.parts)
            second_suggestion = getattr(violation, '_second_suggestion', violation.suggestion)

            # Ensure we have proper suggestion format
            if second_suggestion and second_suggestion != violation.suggestion:
                combined_suggestion = f"First story: {violation.suggestion}. Second story: {second_suggestion}"
            else:
                combined_suggestion = violation.suggestion

            # Use component IDs if available, otherwise use placeholder values
            first_id = component1.id or "component_1"
            second_id = component2.id or "component_2"

            pairwise_violations.append(
                PairwiseViolation(
                    first_parts=first_parts,
                    second_parts=second_parts,
                    first_id=first_id,
                    second_id=second_id,
                    issue=violation.issue,
                    suggestion=combined_suggestion,
                )
            )
        return pairwise_violations, usage

    def analyze_full_set(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], LLMResult | None]:
        """Analyzes multiple QUS components for dependencies in a single LLM call.
        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            components (list[QUSComponent]): List of QUSComponents to analyze.
        Returns:
            Tuple containing list of full-set violations and LLM result.
        """
        if self.mode != "fullset":
            raise ValueError("This parser is not in fullset mode")

        if len(components) < 2:
            return [], None

        stories_list = format_stories_list(components)
        values = {"stories_list": stories_list}
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.violations, usage


class IndependentAnalyzer:
    """Main analyzer class for independence evaluation.

    Provides class methods for running independence checks on sets of QUS components.
    """

    __independent_parser_pairwise = IndependentParserModel("pairwise")
    __independent_parser_fullset = IndependentParserModel("fullset")

    @classmethod
    def analyze_pairwise(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes two components for independence violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM usage data.
        """
        violations, usage = cls.__independent_parser_pairwise.analyze_pairwise(
            client, model_idx, component1, component2
        )
        usage_dict = {cls.__independent_parser_pairwise.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def analyze_all_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes all pairwise combinations in a set for independence violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of all pairwise violations and LLM usage data.
        """
        return analyze_set_pairwise(cls, client, model_idx, components)

    @classmethod
    def analyze_full_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], dict[str, LLMUsage]]:
        """Analyzes all components for dependencies using single LLM call (batch processing).

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of full-set violations and LLM usage data.
        """
        violations, usage = cls.__independent_parser_fullset.analyze_full_set(
            client, model_idx, components
        )
        usage_dict = {cls.__independent_parser_fullset.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, *args, mode: str = "pairwise"
    ) -> tuple[list[PairwiseViolation | FullSetViolation], dict[str, LLMUsage]]:
        """Runs independence analysis on user story components.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            *args: Variable arguments based on mode:
                - For pairwise mode: component1, component2 (two QUSComponent objects)
                - For fullset mode: components (list[QUSComponent])
            mode (str): Analysis mode - "pairwise" or "fullset". Defaults to "pairwise".

        Returns:
            Tuple containing violations and LLM usage data.

        Note:
            - Pairwise mode: Compares two individual components
            - Fullset mode: Analyzes entire set using batch processing
        """
        if mode == "pairwise":
            if len(args) != 2:
                raise ValueError("Pairwise mode requires exactly 2 components")
            component1, component2 = args
            return cls.analyze_pairwise(client, model_idx, component1, component2)

        elif mode == "fullset":
            if len(args) != 1 or not isinstance(args[0], list):
                raise ValueError("Fullset mode requires a list of components")
            components = args[0]
            if len(components) < 2:
                return [], {}
            return cls.analyze_full_set(client, model_idx, components)

        else:
            raise ValueError("Mode must be 'pairwise' or 'fullset'")
