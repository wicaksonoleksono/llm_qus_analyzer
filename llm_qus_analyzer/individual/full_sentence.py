from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation
from dataclasses import dataclass
from typing import Any, Optional
from ..utils import analyze_individual_with_llm


# TODO:
# Full sentence :
#     is the scope clearly defined and bounded? (Clear object, clear action, clear Role)
#     Cann development effort be reasonably estimated
#     Does it avoid multiple hidden functionalities
# if wf true
# - Does it read as a complete, well-formed sentence rather than fragments or phrases?
# Few shot for general knowledge on ``
_definition = """
**Evaluate whether this user story is 'Full Sentence' based on linguistic correctness (template-agnostic):**

1. **[user_story] Grammatical Check:**  
    - Is it a complete sentence with a subject and a finite verb (not a fragment or a run-on)?  
    - Does it start with a capital letter and end with terminal punctuation (. ? !)?  
    - Is subject–verb agreement correct (including irregular verbs)?  
    - Are articles/demonstratives number-correct (a/an = singular only; no a/an with plurals; this/that = singular; these/those = plural)?

2. **[user_story] Punctuation & Spelling:**  
    - Are commas used correctly between clauses (no comma splices)?  
    - Are quotation marks and brackets **paired and balanced**: “ ”, ‘ ’, ( ), [ ], {{}}?  
      *(Apostrophes in contractions/possessives—it’s, user’s, users’—are single characters, not pairs.)*  
    - Are there any spelling errors or typos?
3. flag as correct
    Using prural subject is ok, aslong as its consistent e.g. as developers, we want .. (use a proper pronoun based off the first word)
**Suggestion to fix:**  
- If a fragment, add the missing subject or finite verb; if a run-on, split or add a conjunction.  
- Fix subject–verb agreement and tense consistency; repair comma splices; balance quotes/brackets; correct spelling.
"""
_in_format = """
**User Story to Evaluate:** given [user_story] is {user_story}
"""

_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If valid: `{{ "valid": true }}`
- If invalid:
  ```json
  {{
      "valid": false,
      "violations": [
        {{
            "part": "[user_story]",
            "issue": "Description of the flaw",
            "suggestion": "How to fix it"
        }}
      ]
  }}
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class FullSentenceVerdictData:
    """Data class representing the verdict of a full sentence analysis."""

    valid: bool
    """Boolean indicating whether the component is a full sentence."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


_PART_MAP = {
    "[user_story]": "user_story",
}


class FullSentenceParserModel:
    """Parser model for analyzing full sentence quality of user stories using LLM.

    This class handles the parsing and analysis of user stories to determine
    if they are grammatically correct and complete sentences.
    """

    def __init__(self):
        """Initializes the parser model with analyzer configuration."""
        self.key = "full-sentence"
        self.__analyzer = LLMAnalyzer[FullSentenceVerdictData](key=self.key)
        self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> FullSentenceVerdictData:
        """Parses raw JSON output from LLM into structured data.

        Args:
            raw_json: Raw JSON output from the LLM analysis.

        Returns:
            FullSentenceVerdictData: Containing the parsed validation results and violations.
        """
        if not isinstance(raw_json, dict):
            return FullSentenceVerdictData(False, [])

        valid = raw_json.get("valid", False)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = False

        violations: list[Violation] = []
        default_vio = Violation({}, "Unknown", "Unknown")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    part = _PART_MAP.get(t.get("part", ""))
                    violations.append(
                        Violation(
                            parts={part} if part else {},
                            issue=t.get("issue", ""),
                            suggestion=t.get("suggestion"),
                        )
                    )
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return FullSentenceVerdictData(valid=valid, violations=violations)

    def analyze_single(
        self, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], LLMResult | None]:
        """Analyzes a single user story for full sentence quality.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.

        Returns:
            Tuple containing list of violations and LLM result/usage data.
        """
        values = {"user_story": component.original_text or component.text}
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.violations, usage

    def analyze_list(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> list[tuple[list[str], LLMResult | None]]:
        """Analyzes a list of user stories for full sentence quality.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            components (QUSComponent): List of QUSComponents to analyze.

        Returns:
            List of tuples containing violations and LLM results for each component.
        """
        return [
            self.analyze_single(client, model_idx, component)
            for component in components
        ]


class FullSentenceAnalyzer:
    """Main analyzer class for full sentence evaluation.

    Provides class methods for running full sentence checks on user stories.
    """

    __fs_parser = FullSentenceParserModel()

    @classmethod
    def __not_violated(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], Optional[LLMUsage]]:
        """Checks if a user story violates full sentence rules.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.

        Returns:
            Tuple containing list of violations and LLM usage data.
        """
        violations, result = cls.__fs_parser.analyze_single(
            client, model_idx, component
        )
        return violations, result

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], dict[str, LLMUsage]]:
        """Runs the complete full sentence analysis pipeline.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.

        Returns:
            Tuple containing:
            - List of all violations found
            - Dictionary of LLM usage statistics by task key
        """
        llm_checker = [cls.__not_violated]
        task_keys = [cls.__fs_parser.key]
        violations, usages = analyze_individual_with_llm(
            llm_checker, client, model_idx, component
        )
        llm_usage = {k: r for k, r in zip(task_keys, usages) if r is not None}

        return violations, llm_usage
