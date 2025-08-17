from copy import deepcopy
from typing import Any, Optional
from ..type import Violation
from ..client import LLMClient, LLMUsage
from ..chunker.models import QUSComponent
from ..chunker.parser import Template
from dataclasses import dataclass
from ..analyzer import LLMAnalyzer
_definition = """
**Evaluate whether two user stories are 'Conflict-Free' based on their [Means], [Ends], and [Role]:**
1. **[Means] and [Ends] Check:**  
   - Do both stories have the same [Means] but contradictory [Ends]?  
   - Do both stories have the same [Ends] but prescribe incompatible [Means]?  
   - Does one story’s [Ends] equal the other’s [Means], creating an impossible or unsatisfied dependency?  
2. **[Means] Check:**  
   - Do both stories describe the same feature on the same object but with incompatible scope (e.g., self-only vs global)?  
   - Do both stories describe the same feature on the same object but with incompatible state effects (e.g., temporary vs permanent)?  
3. **[Role] and [Means] Check:**  
   - Do different [Role]s demand outcomes, permissions, or constraints that cannot both be satisfied for the same [Means] and object?  
4. **Empty [Ends] Check:**  
   - If [Ends] are missing, can the implied purposes from the [Means] not both hold together? If so, they are in conflict.  
"""


_in_fmt = """
**User Story to Evaluate:**  
first user story
- [Role]: {r1}
- [Means]: {m1}
- [Ends]: {e1}

second user story 
- [Role]: {r2}
- [Means]: {m2}
- [Ends]: {e2}
"""
_out_fmt = """
**Strictly follow this output format (JSON) wihtout any other explanation:**
- If valid: `{{"valid":true}}`
- If invalid:
```json 
  {{
      "valid": false,
      "violations": [
        {{  
            "part": "[Role],[Means],[Ends]",
            "issue": "Description of the flaw specifically on which user story",
            "first_suggestion": "How to fix the first user story", 
            "second_suggestion": "How to fix the second user story"
        }},
      ]
  }}

```
"""
_PART_MAP = {
    "[Role]": "Role",
    "[Means]": "means",
    "[Ends]": "ends",
}


@dataclass
class CFVerdictData:

    valid: bool
    violations: list[Violation]


class cfParserModel:
    """Parser for conflict-free"""

    def __init__(self):
        self.key = "conflict-free"
        self.__analyzer = LLMAnalyzer[CFVerdictData](key=self.key)
        self.__analyzer.build_prompt(_definition, _in_fmt, _out_fmt)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> CFVerdictData:
        if not isinstance(raw_json, dict):
            return CFVerdictData(False, [])
        # valid =
        pass


class ConflictFreeAnalyzer:
    # def _create_a_feature(cls):

    @classmethod
    def __does_contradict(cls, c: QUSComponent) -> Optional[Violation]:
        """Checks semantically does the use story contradicts
        Args:
            component(QUSComponent): The parsed user story component to validate

        Returns:
            Optional[Violation]:
                - Violation if invalid characters are found
                - None if only allowed characters are present

        """
        m, r, e = c.means, c.role, c.ends
