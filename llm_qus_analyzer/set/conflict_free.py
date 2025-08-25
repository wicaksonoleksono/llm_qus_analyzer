from copy import deepcopy
from typing import Any, Optional
from ..type import Violation, PairwiseViolation, FullSetViolation
from ..client import LLMClient, LLMUsage, LLMResult
from ..chunker.models import QUSComponent
from ..chunker.parser import Template
from dataclasses import dataclass
from ..analyzer import LLMAnalyzer
from ..utils import analyze_set_pairwise, analyze_set_fullset, format_set_results_pairwise, format_set_results_fullset
# https://link.springer.com/article/10.1007/s00766-016-0250-x Based off the 4 rules of conflict free but we saw it through the semantic scope (Refer to this)
_definition = """
**Evaluate whether two user stories are 'Conflict-Free' based on their [Means], [Ends], and [Role]:**
Conflict free : A user story should not be inconsistent with any other user story
1. **[Means] and [Ends] Check:**  
   - Do both stories have the **same [Means]** but **contradictory [Ends]** (mutually exclusive outcomes)?  
   - Do both stories have **different [Means]** toward the **same [Ends]**, where the means are **incompatible** (cannot coexist without inconsistency)?

2. **[Means] Check:**  
   - Do both stories specify the **same feature on the same object** but with **incompatible constraints** (e.g., self-only vs global)?  
   - Are there **contradictory state effects** on the same object (e.g., permanent retention vs immediate deletion)?

3. **[Role] and [Means] Check:**  
   - Do different [Role]s impose **outcomes or permissions that cannot both be satisfied** for the same [Means]/object (true mutual exclusion)?

4. **Out-of-scope for Conflict (do not flag here):**  
   - **Dependency** (one story requires another) → track under **Independent**, not Conflict.  
   - **Missing [Ends]** alone does not create a conflict; check via [Means]/constraints instead.

**Suggestions to fix:**  
- **Same [Means], contradictory [Ends]:** choose one policy, or split by **Role/Context** (e.g., Admin vs User); codify the rule and retire the losing story.  
- **Different [Means], same [Ends] but incompatible:** define **precedence** or **mode** (feature flag, setting), or refactor into a single orchestrated [Means] with clear acceptance criteria.  
- **Scope clash (self vs global):** parameterize scope; split into two stories with explicit scope constraints.  
- **State-effect clash (retain vs delete):** define a lifecycle (e.g., soft-delete + retention); split actions and specify ordering/constraints.  
- **Role-permission clash:** introduce a permission matrix; rewrite stories so each Role’s allowed [Means] is explicit.  
- **Actually a dependency:** reclassify under **Independent**; add an explicit dependency note and delivery order.

"""

_in_format = """
**User Story to Evaluate:**  
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
**Strictly follow this output format (JSON) wihtout any other explanation:**
- If valid: `{{"valid":true}}`
- If invalid:
```json 
  {{
      "valid": false,
      "violations": [
        {{  
            "id_pair": {{"first": 0, "second": 1}},
            "issue": "Description of the flaw specifically on which user story",
            "first_suggestion": "How to fix the first user story", 
            "second_suggestion": "How to fix the second user story"
        }}
      ]
  }}
```
"""
_all_set_definition = """
**Evaluate whether multiple user stories are 'Conflict-Free' by checking for conflicts across the entire set:**
1. **[Means] and [Ends] Check:**  
   - Do any stories have the **same [Means]** but **contradictory [Ends]** (mutually exclusive outcomes)?  
   - Do any stories have **different [Means]** toward the **same [Ends]**, where the means are **incompatible** (cannot coexist without inconsistency)?

2. **[Means] Check:**  
   - Do any stories specify the **same feature on the same object** but with **incompatible constraints** (e.g., self-only vs global)?  
   - Are there **contradictory state effects** on the same object (e.g., permanent retention vs immediate deletion)?

3. **[Role] and [Means] Check:**  
   - Do different [Role]s impose **outcomes or permissions that cannot both be satisfied** for the same [Means]/object (true mutual exclusion)?

4. **Out-of-scope for Conflict (do not flag here):**  
   - **Dependency** (one story requires another) → track under **Independent**, not Conflict.  
   - **Missing [Ends]** alone does not create a conflict; check via [Means]/constraints instead.
"""

_all_set_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_all_set_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If valid: `{{"valid":true}}`
- If invalid:
```json 
  {{
      "valid": false,
      "violations": [
        {{
            "story_ids": [1, 2, 3],
            "parts_per_story": [["means"], ["ends"], ["means", "ends"]],
            "issue": "Description of the conflict across these stories",
            "suggestion": "How to resolve the conflicts in this set of stories"
        }}
      ]
  }}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_PART_MAP = {
    "[Role]": "role",
    "[Means]": "means",
    "[Ends]": "ends",
}


@dataclass
class CFVerdictData:
    """Data class representing the verdict of a conflict-free analysis."""

    valid: bool
    """Boolean indicating whether the components are conflict-free."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


@dataclass
class CFFullSetVerdictData:
    """Data class representing the verdict of a full-set conflict-free analysis."""

    valid: bool
    """Boolean indicating whether the components are conflict-free."""

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


class ConflictFreeParserModel:
    """Unified parser for conflict-free analysis supporting both pairwise and fullset modes."""

    def __init__(self, mode: str):
        """Initialize parser with specified mode.

        Args:
            mode: Either "pairwise" or "fullset"
        """
        if mode not in ["pairwise", "fullset"]:
            raise ValueError("Mode must be 'pairwise' or 'fullset'")

        self.mode = mode
        self.key = f"conflict-free-{mode}"

        if mode == "pairwise":
            self.__analyzer = LLMAnalyzer[CFVerdictData](key=self.key)
            self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        else:  # fullset
            self.__analyzer = LLMAnalyzer[CFFullSetVerdictData](key=self.key)
            self.__analyzer.build_prompt(_all_set_definition, _all_set_in_format, _all_set_out_format)

        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> CFVerdictData | CFFullSetVerdictData:
        """Parses raw JSON output from LLM into structured data.
        Args:
            raw_json: Raw JSON output from the LLM analysis.
        Returns:
            CFVerdictData or CFFullSetVerdictData depending on mode.
        """
        if not isinstance(raw_json, dict):
            if self.mode == "pairwise":
                return CFVerdictData(False, [])
            else:
                return CFFullSetVerdictData(True, [])

        valid = raw_json.get("valid", False if self.mode == "pairwise" else True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = False if self.mode == "pairwise" else True

        if self.mode == "pairwise":
            return self.__parse_pairwise(raw_json, valid)
        else:
            return self.__parse_fullset(raw_json, valid)

    def __parse_pairwise(self, raw_json: dict, valid: bool) -> CFVerdictData:
        """Parse pairwise analysis result."""
        violations: list[Violation] = []
        default_vio = Violation({}, "Unknown conflict", "Review stories for conflicts")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    # Extract id_pair for conflict pair identification
                    id_pair = t.get("id_pair", {})
                    first_idx = id_pair.get("first", 0) if isinstance(id_pair, dict) else 0
                    second_idx = id_pair.get("second", 1) if isinstance(id_pair, dict) else 1

                    # Default parts for conflict analysis
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
        return CFVerdictData(valid=valid, violations=violations)

    def __parse_fullset(self, raw_json: dict, valid: bool) -> CFFullSetVerdictData:
        """Parse fullset analysis result."""
        violations: list[FullSetViolation] = []
        default_vio = FullSetViolation([], [], "Unknown conflict", "Review stories for conflicts")
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
        return CFFullSetVerdictData(valid=valid, violations=violations)

    def analyze_pairwise(
        self, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], LLMResult | None]:
        """Analyzes two QUS components for conflicts.
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
        """Analyzes multiple QUS components for conflicts in a single LLM call.
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


class ConflictFreeAnalyzer:
    """Main analyzer class for conflict-free evaluation.

    Provides class methods for running conflict-free checks on sets of QUS components.
    """

    __cf_parser_pairwise = ConflictFreeParserModel("pairwise")
    __cf_parser_fullset = ConflictFreeParserModel("fullset")

    @classmethod
    def analyze_pairwise(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes two specific components for conflicts.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM usage data.
        """
        violations, usage = cls.__cf_parser_pairwise.analyze_pairwise(
            client, model_idx, component1, component2
        )
        usage_dict = {cls.__cf_parser_pairwise.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def analyze_all_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes all pairwise combinations in a set for conflict violations.

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
        """Analyzes all components for conflicts using single LLM call (batch processing).

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of full-set violations and LLM usage data.
        """
        violations, usage = cls.__cf_parser_fullset.analyze_full_set(
            client, model_idx, components
        )
        usage_dict = {cls.__cf_parser_fullset.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, *args, mode: str = "pairwise"
    ) -> tuple[list[PairwiseViolation | FullSetViolation], dict[str, LLMUsage]]:
        """Runs conflict-free analysis on user story components.

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
