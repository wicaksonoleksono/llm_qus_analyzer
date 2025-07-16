from dataclasses import dataclass
from typing import Any, Optional
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMUsage
from .parser import Template, TemplateParser

_definition = """
By definition,
[Role]: A stakeholder or persona that expresses the need. Typically, [Role] are taken from the softwares application domain.
[Means]: The phrase or clause that describes the primary capability, action, or system behavior the [Role] wants to perform or see happen. It represents the core functional need of the user story, including any conditions (triggers or preconditions) that are directly tied to that need, indicating by 'when', 'where', etc.
[Ends]: A direct value of the [Means] for the [Role] or explaining why the [Means] are requested or a dependency on other functionality of the [Means]

Additionally,
the [Role] can be the name of the persona acts as the role: Joe, Alice, and not a subject like I, you, they, etc.
the [Means] should not start with a phrasal modal verb (or semi-modal verb), like "be able to", "want to" etc. Even though if [Role] is not exists, [Means] still can be exists.
the [Ends] should start with a pronoun (if it exist) or verb, not a causal phrase such as "so that", "in order to", "to" etc, and not including any unnecessary text behind it.
Use that definition to get a better understanding about Quality User Story.
"""

_in_format = """
Extract the [Role], [Means] and [Ends] from the following user story:
"{user_story}"
Also please expand all the short version of english verb like "i'm" into "i am", etc.
**Please only display the single final answer without any explanation, fixing steps, or any redundant text.**
"""

_out_format = """
**Strictly follow this output format (JSON):**  
```json
{{
    "expanded": "Expanded user story",
    "component": {{
          "[Role]": "List of string",
          "[Means]": "String or None if not exists",
          "[Ends]": "String or None if not exists"
    }}
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class QUSChunkData:
    """Container for the analyzed components of a user story."""

    expanded: str
    """The user story with all contractions expanded."""

    role: list[str]
    """List of identified role(s) in the user story."""

    means: Optional[str]
    """The identified means component, if present."""

    ends: Optional[str]
    """The identified ends component, if present."""


@dataclass
class QUSComponent:
    """Comprehensive representation of a parsed user story with template."""

    text: str
    """The expanded text of the user story."""

    role: list[str]
    """List of identified role(s)."""

    means: Optional[str]
    """The means component, if present."""

    ends: Optional[str]
    """The ends component, if present."""

    template: Template
    """Templatized version of the user story."""


class QUSChunkerModel:
    """Model for analyzing and chunking user stories into components.

    Uses an LLM analyzer to identify roles, means, and ends components,
    then creates templates from the analyzed stories.
    """

    def __init__(self) -> None:
        """Initializes the chunker model with predefined prompts and parser."""
        self.key = 'chunker'
        self.__analyzer = LLMAnalyzer[QUSChunkData](key=self.key)
        self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw: Any) -> QUSChunkData:
        """Parses raw LLM output into structured type.

        Args:
            raw (Any): The raw JSON output from the LLM analyzer.

        Returns:
            QUSChunkData: Structured representation of the parsed components.

        Note:
            Handles various edge cases in the raw output including:
            - Normalizing role to always be a list
            - Converting string 'none' or empty strings to None
            - Handling both list and string representations of roles
        """
        expanded = raw['expanded']
        role = raw['component']['[Role]']
        if role is None:
            role = []
        if isinstance(role, str):
            if role.lower() == 'none' or role == '':
                role = []
            else:
                role = [role]
        means = raw['component']['[Means]']
        if isinstance(means, str):
            if means.lower() == 'none' or means == '':
                means = None
        ends = raw['component']['[Ends]']
        if isinstance(ends, str):
            if ends.lower() == 'none' or ends == '':
                ends = None
        return QUSChunkData(expanded, role, means, ends)

    def analyze_single(self, client: LLMClient, model_idx: int, user_story: str) -> tuple[QUSComponent, LLMUsage]:
        """Analyzes a single user story into its components.

        Args:
            client (LLMClient): The LLM client to use for analysis.
            model_idx (int): Index of the specific LLM model to use.
            user_story (str): The user story text to analyze.

        Returns:
            tuple[QUSComponent,LLMUsage]: 
                - The fully parsed user story components
                - The LLM usage object

        Note:
            The analysis pipeline:
            1. LLM extracts roles, means, and ends
            2. TemplateParser creates a template pattern
            3. Results are packaged into QUSComponent
        """
        values = {'user_story': user_story}
        data, usage = self.__analyzer.run(client, model_idx, values)
        TemplateParser.prepare()
        template = TemplateParser.parse(
            data.expanded, data.role, data.means, data.ends
        )
        component = QUSComponent(
            text=data.expanded,
            role=data.role,
            means=data.means,
            ends=data.ends,
            template=template,
        )
        return component, usage

    def analyze_list(self, client: LLMClient, model_idx: int, user_stories: list[str]) -> list[tuple[QUSComponent, LLMUsage]]:
        """Analyzes multiple user stories in batch.

        Args:
            client: The LLM client to use for analysis.
            model_idx: Index of the specific LLM model to use.
            user_stories: List of user story texts to analyze.

        Returns:
            list[tuple[QUSComponent,LLMUsage]]: 
                List of analysis results (component, usage) for each input story.
        """
        return [
            self.analyze_single(client, model_idx, user_story)
            for user_story in user_stories
        ]
