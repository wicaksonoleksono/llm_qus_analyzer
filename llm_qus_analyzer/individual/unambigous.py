from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation
from dataclasses import dataclass
from typing import Any, Optional
from ..utils import analyze_individual_with_llm
# https://link.springer.com/chapter/10.1007/978-1-4615-0465-8_2 refer to this paper for this prompt.
_definition = """
**Evaluate whether this user story is 'Unambiguous' based on its [Means] and [Ends]:**  
unambiguous: the story has a single clear interpretation on its own; no term should plausibly mean different things in this domain.

1. **[Means] Check:**  
    - Does [Means] avoid **superclass (hypernym) terms** that could mean multiple domain items (e.g., “content,” “media,” generic “data”)?  
    - Does [Means] use verbs that are **understood one way in this context**? If a verb could imply different actions (e.g., “manage” = create/update/delete/moderate), rewrite to **list the exact actions**.  
    - Are object nouns **clear in context**, not allowing competing readings?  
      *(e.g., “view telemetry data logs” is clear; “work with data” is ambiguous.)*

2. **[Ends] Check (if present):**  
    - Does [Ends] avoid **ambiguous references** (“this,” “it,” “there”) that could point to different things?  
    - Does [Ends] avoid **multi-meaning phrases** (e.g., “better,” “improved,” “efficient”) when it’s unclear **which aspect** is meant?

3. **[Means] and [Ends] Link (if present):**  
    - Is it **clear which aspect of the [Means]** the [Ends] refers to, so only one interpretation of the rationale is possible?  
      *(e.g., “help patrons quickly” → ambiguous; “provide book-search results to patrons quickly at the desk” → single reading.)*

**Suggestion to fix:**  
- Replace superclass terms with **explicit domain items** (e.g., “edit content” → “edit video, photo, and audio”).  
- Replace ambiguous verbs with the **intended actions** (e.g., “manage records” → “create, update, and delete patient records”).  
- Clarify generic nouns **in context** (e.g., “access data” → “access exported customer order data”).  
- Clarify pronouns and multi-meaning phrases so each has **one obvious referent/aspect** (e.g., “help patrons quickly” → “return book-search results to patrons quickly at the desk”).


"""
_in_format = """
**User Story to Evaluate:**  
- [Means]: {means}
- [Ends]: {ends}
"""
_out_format = """
**Stricly follow this output format (JSON) without any other explanation:**  
- If valid: `{{ "valid": true }}`  
- If invalid:  
  ```json
  {{
      "valid": false,
      "violations": [
        {{
            "part": "[Means]", 
            "issue": "Description of the flaw",
            "suggestion": "How to fix it"
        }}
      ]
  }}
  **Please only display the final answer without any explanation, description, or any redundant text.**
  """


@dataclass
class UnverdictData:
    valid: bool
    """Boolean indicating whether the component is conceptually sound."""
    violations: list[Violation]
    """List of Violation objects found in the analysis."""


_PART_MAP = {
    "[Means]": "means",
    "[Ends]": "ends",
}


class UnParserModel:
    def __init__(self):
        self.key = "unambiguous"
        self.__analyzer = LLMAnalyzer[UnverdictData](key=self.key)
        self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> UnverdictData:
        """Parses raw JSON output from LLM into structured CSVerdictData.

        Args:
            raw_json: Raw JSON output from the LLM analysis.

        Returns:
            CSVerdictData: Containing the parsed validation results and violations.
        """
        if not isinstance(raw_json, dict):
            return UnverdictData(False, [])

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
        return UnverdictData(valid=valid, violations=violations)

    def analyze_single(
        self, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], LLMResult | None]:
        """Analyzes a single QUS component for problem oriented.
        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.
        Returns:
            Tuple containing list of violations and LLM result/usage data.
        """
        if component.means is None:
            return [], None
        values = {"role": component.role, "means": component.means, "ends": component.ends}
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.violations, usage

    def analyze_list(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> list[tuple[list[str], LLMResult | None]]:
        """Analyzes a list of QUS components for problem oriented.

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


class UnambiguousAnalyzer:
    __un_parser = UnParserModel()

    @classmethod
    def __not_violated(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], Optional[LLMUsage]]:
        """Checks if a component violates ambiguaity.
        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.
        Returns:
            Tuple containing list of violations and LLM usage data.
        """
        means = component.means
        if not means:
            return [], None
        violations, result = cls.__un_parser.analyze_single(
            client, model_idx, component
        )
        return violations, result

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], dict[str, LLMUsage]]:
        """Runs the complete problem oriented analysis pipeline.
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
        task_keys = [cls.__un_parser.key]
        violations, usages = analyze_individual_with_llm(
            llm_checker, client, model_idx, component
        )
        llm_usage = {k: r for k, r in zip(task_keys, usages) if r is not None}

        return violations, llm_usage
