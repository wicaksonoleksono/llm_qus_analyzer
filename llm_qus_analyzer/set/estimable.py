from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation, PairwiseViolation, FullSetViolation
from dataclasses import dataclass
from typing import Any, Optional

_definition = """
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


class EstimableFullSetParserModel:
    """Parser model for analyzing estimability across multiple stories using LLM."""

    def __init__(self):
        """Initializes the full-set parser model with analyzer configuration."""
        self.key = "estimable-fullset"
        self.__analyzer = LLMAnalyzer[EstimableFullSetVerdictData](key=self.key)
        self.__analyzer.build_prompt(_definition, _all_set_in_format, _all_set_out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> EstimableFullSetVerdictData:
        """Parses raw JSON output from LLM into structured data.
        Args:
            raw_json: Raw JSON output from the LLM analysis.
        Returns:
            EstimableFullSetVerdictData: Containing the parsed validation results and violations.
        """
        if not isinstance(raw_json, dict):
            return EstimableFullSetVerdictData(True, [])
        
        valid = raw_json.get("valid", True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = True

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

    __estimable_fullset_parser = EstimableFullSetParserModel()

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
        violations, usage = cls.__estimable_fullset_parser.analyze_full_set(
            client, model_idx, components
        )
        usage_dict = {cls.__estimable_fullset_parser.key: usage} if usage else {}
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