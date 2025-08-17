from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation
from dataclasses import dataclass
from typing import Any, Optional
from ..utils import analyze_individual_with_llm
# TODO: create a feature, maker .
# isSemDuplicate, a that dupliocateds the requyest of us2 whole using a different text.
# isSemDuplicate(us1,us2) <-> us1=us2 \and us1 !=(syn) us2
#  Strat to do : to make it reallyexplicit we can create a semantic meaning extractor
#  same means different ends.
#
#
__semantic = """



"""


@dataclass
class UniqueData:
    valid: bool
    """Boolean indicating whether the component is conceptually sound."""
    violations: list[Violation]
    """List of Violation objects found in the analysis."""


class unqiqueAnalyzer:
    @classmethod
    def _is_full_duplicate():
        pass

    @classmethod
    def _semantic_duplicate():
        def __create_semantic_output(us):

            pass
        pass

    @classmethod
    def _different_m():
        pass

    @classmethod
    def _is_full_duplicate():
        pass

    @classmethod
    def _is_full_duplicate():
        pass

    @classmethod
    def _is_full_duplicate():
        pass

# isSemDuplicate, a that dupliocateds the requyest of us2 whole using a different text.
# isSemDuplicate(us1,us2) <-> us1=us2 \and us1 !=(syn) us2
#  Strat to do : to make it reallyexplicit we can create a semantic meaning extractor
#
