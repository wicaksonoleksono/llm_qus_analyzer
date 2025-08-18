import re
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation, PairwiseViolation, FullSetViolation
from dataclasses import dataclass
from typing import Any, Optional

_semantic_definition = """
**Evaluate whether two user stories are 'Semantically Similar' despite different wording:**
1. **Similar Action Check:**
   - Do both stories describe essentially the same action or functionality?
   - Are the core behaviors or system operations equivalent?
   
2. **Similar Output Check:**
   - Do both stories aim for the same outcome or benefit?
   - Are the end goals or results essentially identical?

Note: Stories are semantically similar if they request the same thing using different words.
"""

_semantic_in_format = """
**User Stories to Compare:**
First story: "{story1}"
Second story: "{story2}"
"""

_semantic_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If not similar: `{{"valid": true}}`
- If semantically similar:
  ```json
  {{
      "valid": false,
      "violations": [
        {{
            "first_parts": "semantic_duplicate",
            "second_parts": "semantic_duplicate",
            "issue": "Description of how the stories are semantically similar",
            "first_suggestion": "How to consolidate or differentiate the first story",
            "second_suggestion": "How to consolidate or differentiate the second story"
        }}
      ]
  }}
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_all_set_definition = """
**Evaluate whether multiple user stories are 'Unique' by checking for duplicates across the entire set:**
1. **Full Duplicate Check:**
   - Are there any stories that are identical in text (case-insensitive)?
   
2. **Semantic Similarity Check:**
   - **Similar Action Check:**
     - Do any stories describe essentially the same action or functionality?
     - Are the core behaviors or system operations equivalent?
   - **Similar Output Check:**
     - Do any stories aim for the same outcome or benefit?
     - Are the end goals or results essentially identical?

Note: Stories are duplicates if they request the same thing using different words or are textually identical.
"""

_all_set_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_all_set_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If unique: `{{"valid": true}}`
- If duplicates found:
```json
{{
    "valid": false,
    "violations": [
      {{
          "story_ids": [1, 3, 5],
          "parts_per_story": [["text"], ["text"], ["text"]],
          "issue": "Description of the duplicate stories found",
          "suggestion": "How to resolve the duplicate stories in this set"
      }}
    ]
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class UniqueVerdictData:
    """Data class representing the verdict of a uniqueness analysis."""

    valid: bool
    """Boolean indicating whether the components are unique (not duplicates)."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


@dataclass
class UniqueFullSetVerdictData:
    """Data class representing the verdict of a full-set uniqueness analysis."""

    valid: bool
    """Boolean indicating whether the components are unique (not duplicates)."""

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
    "semantic_duplicate": "semantic",
}


class SemanticSimilarityParserModel:
    """Parser model for analyzing semantic similarity between user stories using LLM."""

    def __init__(self):
        """Initializes the parser model with analyzer configuration."""
        self.key = "semantic-similarity"
        self.__analyzer = LLMAnalyzer[UniqueVerdictData](key=self.key)
        self.__analyzer.build_prompt(_semantic_definition, _semantic_in_format, _semantic_out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> UniqueVerdictData:
        """Parses raw JSON output from LLM into structured data.

        Args:
            raw_json: Raw JSON output from the LLM analysis.

        Returns:
            UniqueVerdictData: Containing the parsed validation results and violations.
        """
        if not isinstance(raw_json, dict):
            return UniqueVerdictData(True, [])

        valid = raw_json.get("valid", True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = True

        violations: list[Violation] = []
        default_vio = Violation({}, "Unknown semantic similarity", "Review stories for duplicates")
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
                    
                    # If no specific parts found, default to semantic
                    if not first_parts and not second_parts:
                        first_parts = {"semantic"}
                        second_parts = {"semantic"}
                    
                    # Store both parts in violation for later PairwiseViolation creation
                    violation = Violation(
                        parts=first_parts.union(second_parts),
                        issue=t.get("issue", ""),
                        suggestion=t.get("first_suggestion", t.get("suggestion", "")),
                    )
                    # Store additional data for PairwiseViolation
                    violation._first_parts = first_parts
                    violation._second_parts = second_parts
                    violation._second_suggestion = t.get("second_suggestion", t.get("suggestion", ""))
                    
                    violations.append(violation)
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return UniqueVerdictData(valid=valid, violations=violations)

    def analyze_semantic_similarity(
        self, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[Violation], LLMResult | None]:
        """Analyzes two QUS components for semantic similarity.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component1 (QUSComponent): First QUSComponent to compare.
            component2 (QUSComponent): Second QUSComponent to compare.

        Returns:
            Tuple containing list of violations and LLM result.
        """
        values = {
            "story1": component1.text,
            "story2": component2.text,
        }
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.violations, usage




class UniqueFullSetParserModel:
    """Parser model for analyzing uniqueness across multiple stories using LLM."""

    def __init__(self):
        """Initializes the full-set parser model with analyzer configuration."""
        self.key = "unique-fullset"
        self.__analyzer = LLMAnalyzer[UniqueFullSetVerdictData](key=self.key)
        self.__analyzer.build_prompt(_all_set_definition, _all_set_in_format, _all_set_out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> UniqueFullSetVerdictData:
        """Parses raw JSON output from LLM into structured data.
        Args:
            raw_json: Raw JSON output from the LLM analysis.
        Returns:
            UniqueFullSetVerdictData: Containing the parsed validation results and violations.
        """
        if not isinstance(raw_json, dict):
            return UniqueFullSetVerdictData(True, [])
        
        valid = raw_json.get("valid", True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = True

        violations: list[FullSetViolation] = []
        default_vio = FullSetViolation([], [], "Unknown duplicates", "Review stories for uniqueness")
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
                                # For uniqueness, usually just "text"
                                part_set.add(part.lower())
                            processed_parts.append(part_set)
                        else:
                            processed_parts.append({"text"})
                    
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
        return UniqueFullSetVerdictData(valid=valid, violations=violations)

    def analyze_full_set(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], LLMResult | None]:
        """Analyzes multiple QUS components for duplicates in a single LLM call.
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


class UniqueAnalyzer:
    """Main analyzer class for uniqueness evaluation.

    Provides class methods for running uniqueness checks on sets of QUS components
    using both full duplicate detection and semantic similarity analysis.
    """

    __semantic_parser = SemanticSimilarityParserModel()
    __unique_fullset_parser = UniqueFullSetParserModel()

    @classmethod
    def analyze_pairwise(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes two components for uniqueness violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM usage data.
        """
        violations: list[PairwiseViolation] = []
        usage_dict: dict[str, LLMUsage] = {}
        
        # Step 1: Check for full duplicates (fast regex check)
        if cls._is_full_duplicate(component1, component2):
            violations.append(
                PairwiseViolation(
                    first_parts={"text"},
                    second_parts={"text"},
                    issue="Stories are identical duplicates",
                    suggestion="Remove one of the duplicate stories",
                )
            )
            return violations, usage_dict

        # Step 2: Check for semantic similarity (LLM analysis)
        semantic_violation, usage = cls._is_semantically_similar(
            client, model_idx, component1, component2
        )
        
        if semantic_violation:
            violations.append(semantic_violation)
        
        if usage:
            usage_dict[cls.__semantic_parser.key] = usage

        return violations, usage_dict

    @classmethod
    def _is_full_duplicate(cls, component1: QUSComponent, component2: QUSComponent) -> bool:
        """Checks if two components are full duplicates using case-insensitive text comparison.

        Args:
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            bool: True if components are full duplicates, False otherwise.
        """
        if not component1.text or not component2.text:
            return False
        
        # Case-insensitive comparison of expanded user story text
        text1 = re.sub(r'\s+', ' ', component1.text.strip().lower())
        text2 = re.sub(r'\s+', ' ', component2.text.strip().lower())
        
        return text1 == text2

    @classmethod
    def _is_semantically_similar(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[Optional[PairwiseViolation], Optional[LLMUsage]]:
        """Checks if two components are semantically similar using LLM analysis.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing optional PairwiseViolation and LLM usage data.
        """
        violations, usage = cls.__semantic_parser.analyze_semantic_similarity(
            client, model_idx, component1, component2
        )
        
        if violations:
            # Convert Violation to PairwiseViolation
            violation = violations[0]  # Take first violation
            
            # Use stored parts from parser if available, otherwise default to text
            first_parts = getattr(violation, '_first_parts', {"text"})
            second_parts = getattr(violation, '_second_parts', {"text"})
            second_suggestion = getattr(violation, '_second_suggestion', violation.suggestion)
            
            # Ensure we have proper suggestion format
            if second_suggestion and second_suggestion != violation.suggestion:
                combined_suggestion = f"First story: {violation.suggestion}. Second story: {second_suggestion}"
            else:
                combined_suggestion = violation.suggestion
            
            pairwise_violation = PairwiseViolation(
                first_parts=first_parts,
                second_parts=second_parts,
                issue=violation.issue,
                suggestion=combined_suggestion,
            )
            return pairwise_violation, usage
        
        return None, usage

    @classmethod
    def analyze_pairwise(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes two components for uniqueness violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM usage data.
        """
        violations: list[PairwiseViolation] = []
        usage_dict: dict[str, LLMUsage] = {}
        
        # Step 1: Check for full duplicates (fast regex check)
        if cls._is_full_duplicate(component1, component2):
            violations.append(
                PairwiseViolation(
                    first_parts={"text"},
                    second_parts={"text"},
                    issue="Stories are identical duplicates",
                    suggestion="Remove one of the duplicate stories",
                )
            )
            return violations, usage_dict

        # Step 2: Check for semantic similarity (LLM analysis)
        semantic_violation, usage = cls._is_semantically_similar(
            client, model_idx, component1, component2
        )
        
        if semantic_violation:
            violations.append(semantic_violation)
        
        if usage:
            usage_dict[cls.__semantic_parser.key] = usage

        return violations, usage_dict

    @classmethod
    def analyze_all_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes all pairwise combinations in a set for uniqueness violations.

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
        """Analyzes all components for duplicates using single LLM call (batch processing).

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of full-set violations and LLM usage data.
        """
        violations, usage = cls.__unique_fullset_parser.analyze_full_set(
            client, model_idx, components
        )
        usage_dict = {cls.__unique_fullset_parser.key: usage} if usage else {}
        return violations, usage_dict

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> list[tuple[list[PairwiseViolation], dict[str, LLMUsage]]]:
        """Runs uniqueness analysis on a set of user story components.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of user story components to analyze.

        Returns:
            List of (violations, usage) tuples for uniqueness analysis results.

        Note:
            Performs both full duplicate detection and semantic similarity analysis.
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
