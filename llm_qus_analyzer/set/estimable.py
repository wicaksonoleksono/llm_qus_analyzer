from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation, PairwiseViolation, FullSetViolation
from dataclasses import dataclass
from typing import Any, Optional

_pairwise_definition = """
**Evaluate whether two user stories are 'Estimable' by checking for consistency in scope and complexity:**

1. **[Means] Scope Consistency Check:**
   - Do both [Means] actions have similar complexity levels?
   - Is there a significant difference in complexity between the two [Means] (e.g., simple "view" vs complex "generate report with multiple filters")?
   - Do both [Means] operate at consistent abstraction levels (e.g., both high-level features vs both detailed operations)?

2. **[Ends] Value Consistency Check:**
   - Do both [Ends] provide comparable business value?
   - Is there a vast difference in impact levels between the two stories (e.g., critical system function vs nice-to-have feature)?
   - Do both [Ends] have measurable and comparable outcomes?

3. **Story Detail Consistency:**
   - Do both stories have similar levels of detail and specification?
   - Is one story significantly more vague than the other?
   - Do both stories provide sufficient information for estimation?

4. **Implementation Complexity Alignment:**
   - Would these stories require significantly different effort levels?
   - Do both stories fit within similar technical complexity brackets?
"""

_pairwise_in_format = """
**User Stories to Evaluate:**
First story: "{story1}"
Second story: "{story2}"
"""

_pairwise_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If estimable: `{"valid": true}`
- If not estimable:
  ```json
  {
      "valid": false,
      "violations": [
        {
            "first_parts": "means,ends",
            "second_parts": "means,ends",
            "issue": "Description of estimation inconsistency between the two stories",
            "first_suggestion": "How to make the first story more consistently estimable",
            "second_suggestion": "How to make the second story more consistently estimable"
        }
      ]
  }
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_all_set_definition = """
**Evaluate whether the user story set is 'Estimable' by checking for consistent scope and complexity:**

1. **[Means] Scope Consistency Check:**
   - Do all [Means] actions have similar complexity levels across the set?
   - Are there stories with significantly more complex [Means] than others (e.g., simple "view" vs complex "generate report with multiple filters")?
   - Do all [Means] operate at consistent abstraction levels (e.g., all high-level features vs all detailed operations)?

2. **[Ends] Value Consistency Check:**
   - Do all [Ends] provide comparable business value within the set?
   - Are there stories with vastly different impact levels (e.g., critical system function vs nice-to-have feature)?
   - Do all [Ends] have measurable and comparable outcomes?

3. **Story Length and Detail Consistency:**
   - Do all stories have similar levels of detail and specification?
   - Are there stories that are too vague compared to others in the set?
   - Do all stories provide sufficient information for estimation?

4. **Implementation Complexity Alignment:**
   - Are there stories that would require significantly different effort levels?
   - Do all stories fit within similar technical complexity brackets?
   - Are there dependencies that make some stories much harder to estimate than others?
"""

_all_set_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_all_set_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If estimable: `{"valid": true}`
- If not estimable:
```json
{
    "valid": false,
    "violations": [
      {
          "story_ids": [1, 4, 7],
          "parts_per_story": [["means"], ["ends"], ["means", "ends"]],
          "issue": "Description of estimation inconsistency across these stories",
          "suggestion": "How to make the story set more consistently estimable"
      }
    ]
}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class EstimableVerdictData:
    """Data class representing the verdict of an estimability analysis."""

    valid: bool
    """Boolean indicating whether the components are estimable."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


@dataclass
class EstimableFullSetVerdictData:
    """Data class representing the verdict of a full-set estimability analysis."""

    valid: bool
    """Boolean indicating whether the story set is estimable."""

    violations: list[FullSetViolation]
    """List of FullSetViolation objects found in the analysis."""


def format_stories_list(components: list[QUSComponent]) -> str:
    """Formats a list of QUSComponent objects into a numbered story list for LLM input.
    
    Args:
        components: List of QUSComponent objects to format
        
    Returns:
        Formatted string with numbered stories
    """
    return "\n".join([
        f"Story {i+1}: \"{comp.text}\""
        for i, comp in enumerate(components)
    ])


_PART_MAP = {
    "means": "means",
    "ends": "ends",
}


class EstimableParserModel:
    """Unified parser for estimability analysis supporting both pairwise and fullset modes."""

    def __init__(self, mode: str):
        """Initialize parser with specified mode.
        
        Args:
            mode: Either "pairwise" or "fullset"
        """
        if mode not in ["pairwise", "fullset"]:
            raise ValueError("Mode must be 'pairwise' or 'fullset'")
        
        self.mode = mode
        self.key = f"estimable-{mode}"
        
        if mode == "pairwise":
            self.__analyzer = LLMAnalyzer[EstimableVerdictData](key=self.key)
            self.__analyzer.build_prompt(_pairwise_definition, _pairwise_in_format, _pairwise_out_format)
        else:  # fullset
            self.__analyzer = LLMAnalyzer[EstimableFullSetVerdictData](key=self.key)
            self.__analyzer.build_prompt(_all_set_definition, _all_set_in_format, _all_set_out_format)
        
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> EstimableVerdictData | EstimableFullSetVerdictData:
        """Parses raw JSON output from LLM into structured data.
        Args:
            raw_json: Raw JSON output from the LLM analysis.
        Returns:
            EstimableVerdictData or EstimableFullSetVerdictData depending on mode.
        """
        if not isinstance(raw_json, dict):
            if self.mode == "pairwise":
                return EstimableVerdictData(True, [])
            else:
                return EstimableFullSetVerdictData(True, [])

        valid = raw_json.get("valid", True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = True

        if self.mode == "pairwise":
            return self.__parse_pairwise(raw_json, valid)
        else:
            return self.__parse_fullset(raw_json, valid)

    def __parse_pairwise(self, raw_json: dict, valid: bool) -> EstimableVerdictData:
        """Parse pairwise analysis result."""
        violations: list[Violation] = []
        default_vio = Violation({}, "Unknown estimability issue", "Review stories for consistent estimability")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    # For backwards compatibility, try both formats
                    first_parts_str = t.get("first_parts", t.get("part", ""))
                    second_parts_str = t.get("second_parts", t.get("part", ""))
                    
                    first_parts = set()
                    second_parts = set()
                    
                    # Parse comma-separated parts
                    if first_parts_str:
                        for part_str in first_parts_str.split(","):
                            part_str = part_str.strip()
                            mapped_part = _PART_MAP.get(part_str, part_str.lower())
                            if mapped_part:
                                first_parts.add(mapped_part)
                    
                    if second_parts_str:
                        for part_str in second_parts_str.split(","):
                            part_str = part_str.strip()
                            mapped_part = _PART_MAP.get(part_str, part_str.lower())
                            if mapped_part:
                                second_parts.add(mapped_part)
                    
                    # If no specific parts found, default to means
                    if not first_parts and not second_parts:
                        first_parts = {"means"}
                        second_parts = {"means"}
                    
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
                    
                    violations.append(violation)
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return EstimableVerdictData(valid=valid, violations=violations)

    def __parse_fullset(self, raw_json: dict, valid: bool) -> EstimableFullSetVerdictData:
        """Parse fullset analysis result."""
        violations: list[FullSetViolation] = []
        default_vio = FullSetViolation([], [], "Unknown estimability issue", "Review stories for consistent estimability")
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
                                # For estimability, could be means, ends, or both
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
        return EstimableFullSetVerdictData(valid=valid, violations=violations)

    def analyze_pairwise(
        self, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], LLMResult | None]:
        """Analyzes two QUS components for estimability.
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
            "story1": component1.text,
            "story2": component2.text,
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
            
            pairwise_violations.append(
                PairwiseViolation(
                    first_parts=first_parts,
                    second_parts=second_parts,
                    issue=violation.issue,
                    suggestion=combined_suggestion,
                )
            )
        return pairwise_violations, usage

    def analyze_full_set(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], LLMResult | None]:
        """Analyzes multiple QUS components for estimability in a single LLM call.
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


class EstimableAnalyzer:
    """Main analyzer class for estimability evaluation.

    Provides class methods for running estimability checks on sets of QUS components.
    """

    __estimable_parser_pairwise = EstimableParserModel("pairwise")
    __estimable_parser_fullset = EstimableParserModel("fullset")

    @classmethod
    def analyze_pairwise(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes two components for estimability violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM usage data.
        """
        violations, usage = cls.__estimable_parser_pairwise.analyze_pairwise(
            client, model_idx, component1, component2
        )
        usage_dict = {cls.__estimable_parser_pairwise.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def analyze_all_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes all pairwise combinations in a set for estimability violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of all pairwise violations and LLM usage data.
        """
        all_violations: list[PairwiseViolation] = []
        all_usages: dict[str, LLMUsage] = {}

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                violations, usages = cls.analyze_pairwise(
                    client, model_idx, components[i], components[j]
                )
                all_violations.extend(violations)
                
                # Merge usage data with unique keys
                for key, usage in usages.items():
                    all_usages[f"{key}_pair_{i}_{j}"] = usage

        return all_violations, all_usages

    @classmethod
    def analyze_full_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], dict[str, LLMUsage]]:
        """Analyzes all components for estimability using single LLM call (batch processing).

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of full-set violations and LLM usage data.
        """
        violations, usage = cls.__estimable_parser_fullset.analyze_full_set(
            client, model_idx, components
        )
        usage_dict = {cls.__estimable_parser_fullset.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> list[tuple[list[FullSetViolation], dict[str, LLMUsage]]]:
        """Runs estimability analysis on a set of user story components.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of user story components to analyze.

        Returns:
            List of (violations, usage) tuples for estimability analysis results.

        Note:
            Performs full-set estimability analysis to identify inconsistent estimation complexity.
        """
        if len(components) < 2:
            return [([], {}) for _ in components]

        violations, usage_dict = cls.analyze_full_set(client, model_idx, components)

        # Return results in the expected format for set analyzers
        # First component gets all violations, others get empty results
        if violations:
            return [(violations, usage_dict)] + [([], {}) for _ in components[1:]]
        else:
            return [([], {}) for _ in components]