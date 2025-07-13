from dataclasses import dataclass
from typing import Any, Optional
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult
from .parser import TemplateParser

__definition = """
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

__in_format = """
Extract the [Role], [Means] and [Ends] from the following user story:
"{user_story}"
Also please expand all the short version of english verb like "i'm" into "i am", etc.
"""

__out_format = """
**Strictly follow this output format (JSON) without any other explanation:**  
```json
{{
    "expanded": "Expanded user story",
    "component": {{
          "[Role]": "List of string",
          "[Means]": "String or None if not exists",
          "[Ends]": "String or None if not exists"
    }}
}}
"""


@dataclass
class ChunkerData:
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
class UserStoryComponent:
    """Comprehensive representation of a parsed user story with template."""

    text: str
    """The expanded text of the user story."""

    role: list[str]
    """List of identified role(s)."""

    means: Optional[str]
    """The means component, if present."""

    ends: Optional[str]
    """The ends component, if present."""

    template: str
    """Templatized version of the user story."""

    tail: Optional[str]
    """Any remaining non-templatized text."""


class ChunkerModel:
    """Model for analyzing and chunking user stories into components.

    Uses an LLM analyzer to identify roles, means, and ends components,
    then creates templates from the analyzed stories.
    """

    def __init__(self):
        """Initializes the chunker model with predefined prompts and parser."""
        self.__analyzer = LLMAnalyzer[ChunkerData](key='chunker')
        self.__analyzer.build_prompt(__definition, __in_format, __out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw: Any) -> ChunkerData:
        """Parses raw LLM output into structured ChunkerData.

        Args:
            raw: The raw JSON output from the LLM analyzer.

        Returns:
            ChunkerData: Structured representation of the parsed components.

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
        return ChunkerData(expanded, role, means, ends)

    def analyze_single(self, client: LLMClient, user_story: str, which_model: int) -> tuple[UserStoryComponent, LLMResult]:
        """Analyzes a single user story into its components.

        Args:
            client: The LLM client to use for analysis.
            user_story: The user story text to analyze.
            which_model: Index of the specific LLM model to use.

        Returns:
            tuple[UserStoryComponent, LLMResult]: 
                - The fully parsed user story components
                - The raw LLM result object

        Note:
            The analysis pipeline:
            1. LLM extracts roles, means, and ends
            2. TemplateParser creates a template pattern
            3. Results are packaged into UserStoryComponent
        """
        value = {'user_story': user_story}
        data, raw = self.__analyzer.run(client, value, which_model)
        template, tail = TemplateParser.extract_template(
            data.expanded, data.role, data.means, data.ends
        )
        component = UserStoryComponent(
            text=data.expanded,
            role=data.role,
            means=data.means,
            ends=data.ends,
            template=template,
            tail=tail,
        )
        return component, raw

    def analyze_list(self, client: LLMClient, user_stories: list[str], which_model: int) -> list[tuple[UserStoryComponent, LLMResult]]:
        """Analyzes multiple user stories in batch.

        Args:
            client: The LLM client to use for analysis.
            user_stories: List of user story texts to analyze.
            which_model: Index of the specific LLM model to use.

        Returns:
            list[tuple[UserStoryComponent, LLMResult]]: 
                List of analysis results (component, raw) for each input story.
        """
        return [
            self.analyze_single(client, user_story, which_model)
            for user_story in user_stories
        ]
