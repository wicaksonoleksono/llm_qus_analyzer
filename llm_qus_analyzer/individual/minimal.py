from ..client import LLMClient, LLMResult
from ..chunker.models import UserStoryComponent
from ..utils import analyze_individual_with_basic
from ..type import Violation


class MinimalAnalyzer:
    @classmethod
    def __is_not_contain_special(cls, component: UserStoryComponent):
        us = component.text

        def not_valid(ch: str):
            if ch.isalnum():
                return False
            if ch in ["'", ',', '.', '-', ' ']:
                return False
            else:
                return True

        chars = [ch for ch in us if not_valid(ch)]
        chars = list(set(chars))

        if len(chars) > 0:
            tmp = ', '.join(chars)
            return Violation(
                parts=set(['full']),
                issue=f'The user story contains invalid characters: "{tmp}"',
                suggestion='Remove invalid characters from the user story'
            )

        return None

    @classmethod
    def __is_us_no_tail(cls, component: UserStoryComponent):
        tail = component.tail

        if tail:
            return Violation(
                parts=set(['full']),
                issue=f'The user story contains unnecessary info: "{tail}"',
                suggestion='Remove unnecessary info from the user story.'
            )

        return None

    @classmethod
    def run(cls, client: LLMClient, model_idx: int, component: UserStoryComponent):
        llm_usage: dict[str, LLMResult] = {}

        basic_checker = [
            cls.__is_not_contain_special,
            cls.__is_us_no_tail,
        ]

        violations = analyze_individual_with_basic(basic_checker, component)

        return violations, llm_usage
