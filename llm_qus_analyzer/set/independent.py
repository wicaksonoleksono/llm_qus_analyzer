from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation, PairwiseViolation, FullSetViolation
from dataclasses import dataclass
from typing import Any, Optional

_definition = """
**Evaluate whether two user stories are 'Independent' by checking for dependencies:**
1. **[Means] Check:**  
   - **Causality Dependencies**: Does the means have causality dependencies? (Before doing A you have to do B)
     e.g. 
       story 1: as an admin, i want to view a person profile 
       story 2: as an admin, i want to add a new person to database
     You can't view a person profile; you have to create a profile first
   - **Superclass Dependencies**: Does the means have superclass dependencies?  
     e.g. 
       story 1: ..., i want to edit a content.. 
       story 2: ..., i want to upload a video content.. 
       story 3: ..., i want to upload audio content.. 
     Edit is a superclass action of upload and content is a superclass object of audio & video content
2. **[Ends] Check:**
   - **Ends Containing Means**: Are the ends a purpose that can be implemented separately? (does the ends contain another means?)
     e.g. 
       story1: As a user, I want to register, so that i can contribute to articles
       story2: As a user I want to contribute to articles 
"""

_in_format = """
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

_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If independent: `{"valid": true}`
- If dependent:
  ```json
  {
      "valid": false,
      "violations": [
        {
            "first_parts": "[Role],[Means],[Ends]",
            "second_parts": "[Role],[Means],[Ends]",
            "issue": "Description of the dependency type and how they are dependent",
            "first_suggestion": "How to make the first user story independent", 
            "second_suggestion": "How to make the second user story independent"
        }
      ]
  }
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_all_set_definition = """
**Evaluate whether multiple user stories are 'Independent' by checking for dependencies across the entire set:**
1. **[Means] Check:**  
   - **Causality Dependencies**: Do any stories have causality dependencies? (Before doing A you have to do B)
     e.g. 
       story 1: as an admin, i want to view a person profile 
       story 2: as an admin, i want to add a new person to database
     You can't view a person profile; you have to create a profile first
   - **Superclass Dependencies**: Do any stories have superclass dependencies?  
     e.g. 
       story 1: ..., i want to edit a content.. 
       story 2: ..., i want to upload a video content.. 
       story 3: ..., i want to upload audio content.. 
     Edit is a superclass action of upload and content is a superclass object of audio & video content
2. **[Ends] Check:**
   - **Ends Containing Means**: Are the ends a purpose that can be implemented separately? (does the ends contain another means?)
     e.g. 
       story1: As a user, I want to register, so that i can contribute to articles
       story2: As a user I want to contribute to articles 
"""

_all_set_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_all_set_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If independent: `{"valid": true}`
- If dependent:
```json
{
    "valid": false,
    "violations": [
      {
          "story_ids": [1, 2, 3],
          "parts_per_story": [["means"], ["ends"], ["means", "ends"]],
          "issue": "Description of the dependency across these stories",
          "suggestion": "How to make these stories independent"
      }
    ]
}
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
                    # For backwards compatibility, try both formats
                    first_parts_str = t.get("first_parts", t.get("part", ""))
                    second_parts_str = t.get("second_parts", t.get("part", ""))
                    
                    first_parts = set()
                    second_parts = set()
                    
                    # Parse comma-separated parts
                    if first_parts_str:
                        for part_str in first_parts_str.split(","):
                            part_str = part_str.strip()
                            mapped_part = _PART_MAP.get(part_str, part_str.lower().replace("[", "").replace("]", ""))
                            if mapped_part:
                                first_parts.add(mapped_part)
                    
                    if second_parts_str:
                        for part_str in second_parts_str.split(","):
                            part_str = part_str.strip()
                            mapped_part = _PART_MAP.get(part_str, part_str.lower().replace("[", "").replace("]", ""))
                            if mapped_part:
                                second_parts.add(mapped_part)
                    
                    # Fallback to checking individual parts in the string
                    if not first_parts and not second_parts:
                        for part_key, part_val in _PART_MAP.items():
                            if part_key in first_parts_str:
                                first_parts.add(part_val)
                            if part_key in second_parts_str:
                                second_parts.add(part_val)
                    
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
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> list[tuple[list[PairwiseViolation], dict[str, LLMUsage]]]:
        """Runs independence analysis on a set of user story components.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of user story components to analyze.

        Returns:
            List of (violations, usage) tuples for independence analysis results.

        Note:
            Performs pairwise independence analysis between all component combinations.
        """
        if len(components) < 2:
            return [([], {}) for _ in components]

        all_violations, all_usages = cls.analyze_all_set(client, model_idx, components)

        # Return results in the expected format for set analyzers
        # First component gets all violations, others get empty results
        if all_violations:
            return [(all_violations, all_usages)] + [([], {}) for _ in components[1:]]
        else:
            return [([], {}) for _ in components]