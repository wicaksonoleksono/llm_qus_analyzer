from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation, PairwiseViolation, FullSetViolation
from ..utils import analyze_set_pairwise, analyze_set_fullset, format_set_results_pairwise, format_set_results_fullset
from dataclasses import dataclass
from typing import Any, Optional

_pairwise_definition = """
**Evaluate whether two user stories are 'Complete' by checking if they have prerequisite dependencies:**

1. **[Means] Prerequisite Check:**
   - Does the first story's [Means] require prerequisite actions that the second story provides?
   - Does the second story's [Means] require prerequisite actions that the first story provides?
   - Do either story's [Means] reference states that the other story establishes?

2. **[Means] Object Dependency Check:**
   - Does one story's [Means] operate on objects that the other story creates or defines?
   - Are there complementary creation and operation [Means] between the stories?

3. **[Means] Workflow Dependency Check:**
   - Do the two stories represent sequential steps in a workflow?
   - Is one story a prerequisite for the other to be meaningful?

4. **[Means] Foundation Coverage Check:**
   - For modification operations: Does one story provide the creation that the other requires?
   - For process flows: Does one story provide initiation that the other continues?
"""

_pairwise_in_format = """
**User Stories to Evaluate:**
First user story:
- [Role]: {r1}
- [Means]: {m1}
- [Ends]: {e1}

Second user story:
- [Role]: {r2}
- [Means]: {m2}
- [Ends]: {e2}
"""

_pairwise_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If complete: `{"valid": true}`
- If incomplete:
  ```json
  {
      "valid": false,
      "violations": [
        {
            "id_pair": {"first": 0, "second": 1},
            "issue": "Description of missing prerequisites or dependencies",
            "first_suggestion": "How to make the first story more complete",
            "second_suggestion": "How to make the second story more complete"
        }
      ]
  }
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_all_set_definition = """
**Evaluate whether the user story set is 'Complete' based on [Means] action dependencies:**

1. **[Means] Prerequisite Check:**
   - Does any [Means] action verb require prerequisite actions on the same direct object (e.g., edit→create, delete→create, view→create)?
   - Are all prerequisite [Means] actions covered by other stories in the set?
   - Does any [Means] reference a state that no other [Means] establishes (e.g., pause without play, submit without draft)?

2. **[Means] Object Existence Check:**
   - Does any [Means] operate on direct objects that no story creates or defines?
   - Are there [Means] that assume objects exist without corresponding creation [Means]?
   - Do any [Means] reference collections/databases that no story populates?

3. **[Means] Workflow Completeness Check:**
   - Do the [Means] actions enable complete functional workflows without critical gaps?
   - Are there logical sequence breaks where intermediate [Means] steps are missing (e.g., checkout without add-to-cart)?
   - Can users accomplish end-to-end tasks with the available [Means] set?

4. **[Means] Foundation Coverage Check:**
   - For modification operations: Are corresponding creation [Means] present (e.g., update profile needs create profile)?
   - For process flows: Are initiation [Means] present for all continuation [Means] (e.g., start process before complete process)?
   - For state changes: Are setup [Means] present for all operational [Means] (e.g., configure before modify)?
"""

_all_set_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_all_set_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If complete: `{"valid": true}`
- If incomplete:
```json
{
    "valid": false,
    "violations": [
      {
          "story_ids": [1, 3, 5],
          "parts_per_story": [["means"], ["means"], ["means"]],
          "issue": "Description of missing prerequisites or gaps",
          "suggestion": "How to complete the story set with missing dependencies"
      }
    ]
}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class CompleteVerdictData:
    """Data class representing the verdict of a completeness analysis."""

    valid: bool
    """Boolean indicating whether the components are complete."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


@dataclass
class CompleteFullSetVerdictData:
    """Data class representing the verdict of a full-set completeness analysis."""

    valid: bool
    """Boolean indicating whether the story set is complete."""

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
    "[Means]": "means",
}


class CompleteParserModel:
    """Unified parser for completeness analysis supporting both pairwise and fullset modes."""

    def __init__(self, mode: str):
        """Initialize parser with specified mode.
        
        Args:
            mode: Either "pairwise" or "fullset"
        """
        if mode not in ["pairwise", "fullset"]:
            raise ValueError("Mode must be 'pairwise' or 'fullset'")
        
        self.mode = mode
        self.key = f"complete-{mode}"
        
        if mode == "pairwise":
            self.__analyzer = LLMAnalyzer[CompleteVerdictData](key=self.key)
            self.__analyzer.build_prompt(_pairwise_definition, _pairwise_in_format, _pairwise_out_format)
        else:  # fullset
            self.__analyzer = LLMAnalyzer[CompleteFullSetVerdictData](key=self.key)
            self.__analyzer.build_prompt(_all_set_definition, _all_set_in_format, _all_set_out_format)
        
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> CompleteVerdictData | CompleteFullSetVerdictData:
        """Parses raw JSON output from LLM into structured data.
        Args:
            raw_json: Raw JSON output from the LLM analysis.
        Returns:
            CompleteVerdictData or CompleteFullSetVerdictData depending on mode.
        """
        if not isinstance(raw_json, dict):
            if self.mode == "pairwise":
                return CompleteVerdictData(True, [])
            else:
                return CompleteFullSetVerdictData(True, [])

        valid = raw_json.get("valid", True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = True

        if self.mode == "pairwise":
            return self.__parse_pairwise(raw_json, valid)
        else:
            return self.__parse_fullset(raw_json, valid)

    def __parse_pairwise(self, raw_json: dict, valid: bool) -> CompleteVerdictData:
        """Parse pairwise analysis result."""
        violations: list[Violation] = []
        default_vio = Violation({"means"}, "Unknown completeness issue", "Review stories for missing dependencies")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    # Parse id_pair for component identification
                    id_pair = t.get("id_pair", {"first": 0, "second": 1})
                    
                    violation = Violation(
                        parts={"means"},  # Default to means for completeness analysis
                        issue=t.get("issue", ""),
                        suggestion=t.get("first_suggestion", t.get("suggestion", "")),
                    )
                    # Store additional data for PairwiseViolation
                    violation._id_pair = id_pair
                    violation._first_suggestion = t.get("first_suggestion", "")
                    violation._second_suggestion = t.get("second_suggestion", "")
                    
                    violations.append(violation)
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return CompleteVerdictData(valid=valid, violations=violations)

    def __parse_fullset(self, raw_json: dict, valid: bool) -> CompleteFullSetVerdictData:
        """Parse fullset analysis result."""
        violations: list[FullSetViolation] = []
        default_vio = FullSetViolation([], [], "Unknown completeness issue", "Review stories for missing dependencies")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    story_ids = t.get("story_ids", [])
                    if isinstance(story_ids, list):
                        story_ids = [int(sid) - 1 for sid in story_ids if isinstance(sid, (int, str)) and str(sid).isdigit()]
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
                                # For completeness, usually "means"
                                part_set.add(part.lower())
                            processed_parts.append(part_set)
                        else:
                            processed_parts.append({"means"})
                    
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
        return CompleteFullSetVerdictData(valid=valid, violations=violations)

    def analyze_pairwise(
        self, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], LLMResult | None]:
        """Analyzes two QUS components for completeness.
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
            # Get suggestions from stored data
            first_suggestion = getattr(violation, '_first_suggestion', violation.suggestion)
            second_suggestion = getattr(violation, '_second_suggestion', violation.suggestion)
            
            # Combine suggestions properly
            if first_suggestion and second_suggestion and first_suggestion != second_suggestion:
                combined_suggestion = f"First story: {first_suggestion}. Second story: {second_suggestion}"
            else:
                combined_suggestion = first_suggestion or violation.suggestion
            
            pairwise_violations.append(
                PairwiseViolation(
                    first_parts=violation.parts,
                    second_parts=violation.parts,
                    issue=violation.issue,
                    suggestion=combined_suggestion,
                )
            )
        return pairwise_violations, usage

    def analyze_full_set(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], LLMResult | None]:
        """Analyzes multiple QUS components for completeness in a single LLM call.
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


class CompleteAnalyzer:
    """Main analyzer class for completeness evaluation.

    Provides class methods for running completeness checks on sets of QUS components.
    """

    __complete_parser_pairwise = CompleteParserModel("pairwise")
    __complete_parser_fullset = CompleteParserModel("fullset")

    @classmethod
    def analyze_pairwise(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes two components for completeness violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM usage data.
        """
        violations, usage = cls.__complete_parser_pairwise.analyze_pairwise(
            client, model_idx, component1, component2
        )
        usage_dict = {cls.__complete_parser_pairwise.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def analyze_all_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes all pairwise combinations in a set for completeness violations.

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
        """Analyzes all components for completeness using single LLM call (batch processing).

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of full-set violations and LLM usage data.
        """
        violations, usage = cls.__complete_parser_fullset.analyze_full_set(
            client, model_idx, components
        )
        usage_dict = {cls.__complete_parser_fullset.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, *args, mode: str = "fullset"
    ) -> tuple[list[PairwiseViolation | FullSetViolation], dict[str, LLMUsage]]:
        """Runs completeness analysis on user story components.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            *args: Variable arguments based on mode:
                - For pairwise mode: component1, component2 (two QUSComponent objects)
                - For fullset mode: components (list[QUSComponent])
            mode (str): Analysis mode - "pairwise" or "fullset". Defaults to "fullset".

        Returns:
            Tuple containing violations and LLM usage data.

        Note:
            - Pairwise mode: Compares two individual components for dependencies
            - Fullset mode: Analyzes entire set to identify missing dependencies
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