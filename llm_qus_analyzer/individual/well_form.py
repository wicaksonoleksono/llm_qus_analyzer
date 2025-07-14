from ..client import LLMClient, LLMResult
from ..chunker.models import UserStoryComponent
from ..utils import analyze_individual_with_basic
from ..type import Violation


class WellFormAnalyzer:
    @classmethod
    def __is_have_role(cls, component: UserStoryComponent):
        role = component.role
        if not role or len(role) == 0:
            return Violation(
                parts=set(['role']),
                issue='The [Role] is missing',
                suggestion='Add the [Role] related to the user story, such as "user", etc.'
            )
        return None

    @classmethod
    def __is_have_means(cls, component: UserStoryComponent):
        means = component.means
        if not means:
            return Violation(
                parts=set(['means']),
                issue='The [Means] is missing',
                suggestion='Add the [Means] related to the user story.'
            )
        return None

    @classmethod
    def run(cls, client: LLMClient, model_idx: int, component: UserStoryComponent):
        llm_usage: dict[str, LLMResult] = {}

        basic_checker = [
            cls.__is_have_role,
            cls.__is_have_means,
        ]

        violations = analyze_individual_with_basic(basic_checker, component)

        return violations, llm_usage
