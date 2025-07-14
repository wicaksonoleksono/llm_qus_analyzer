from dataclasses import dataclass
from typing import Any
from ..utils import analyze_individual_with_basic, analyze_individual_with_llm
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult
from ..chunker.models import UserStoryComponent
from ..type import Violation

_definition = """
A user story should concern only one feature. 
"""

_in_format = """
Given user story "{user_story}" and it [Means] "{means}".
Without adding any context outside this user story, 
please extract the individual and unique tasks explicitly mentioned from this [Means].
"""

_out_format = """
**Strictly follow this output format (JSON):**  
```json
{{
    "tasks": "List of string"
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class TasksData:
    tasks: list[str]


class TasksParserModel:
    def __init__(self):
        self.key = 'means-tasks'
        self.__analyzer = LLMAnalyzer[TasksData](key=self.key)
        self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw: Any) -> TasksData:
        tasks = raw['tasks']
        if isinstance(tasks, str):
            if tasks.lower() == 'none' or tasks == '':
                tasks = []
            else:
                tasks = [tasks]
        return TasksData(tasks=tasks)

    def analyze_single(self, client: LLMClient, component: UserStoryComponent, which_model: int) -> tuple[list[str], LLMResult | None]:
        if component.means is None:
            return [], None
        value = {'user_story': component.text, 'means': component.means}
        data, raw = self.__analyzer.run(client, value, which_model)
        return data.tasks, raw

    def analyze_list(self, client: LLMClient, components: list[UserStoryComponent], which_model: int) -> list[tuple[list[str], LLMResult | None]]:
        return [
            self.analyze_single(client, component, which_model)
            for component in components
        ]


llm_parser = TasksParserModel()


class AtomicAnalyzer:
    @classmethod
    def __is_role_single(cls, component: UserStoryComponent):
        role = component.role
        if not role:
            return None

        if len(role) > 1:
            tmp = ', '.join(role)
            return Violation(
                parts=set(['role']),
                issue=f'The [Role] is more than 1: {tmp}',
                suggestion='Select one [Role] that suitable to the user story, or separate it to the different user story.'
            )

        return None

    @classmethod
    def __is_means_single_task(cls, llm_client: LLMClient, model_idx: int, component: UserStoryComponent) -> tuple[Violation | None, LLMResult | None]:
        means = component.means
        if not means:
            return None, None

        tasks, result = llm_parser.analyze_single(
            llm_client, component, model_idx)

        if len(tasks) > 1:
            tmp = '\n'.join(
                [f'({i+1}) {task}' for i, task in enumerate(tasks)])
            return Violation(
                parts=set(['means']),
                issue=f'The [Means] contain more than 1 tasks:\n{tmp}',
                suggestion='Select one task in [Means] that suitable to the user story, or separate it to the different user story.'
            ), result

        return None, result

    @classmethod
    def run(cls, client: LLMClient, model_idx: int, component: UserStoryComponent):
        basic_checker = [
            cls.__is_role_single,
        ]

        violations = analyze_individual_with_basic(basic_checker, component)

        llm_checker = [
            cls.__is_means_single_task
        ]
        llm_keys = [llm_parser.key]
        more_violations, results = analyze_individual_with_llm(
            llm_checker, client, model_idx, component)
        llm_usage = {
            k: r
            for k, r in zip(llm_keys, results) if r is not None
        }

        violations.extend(more_violations)

        return violations, llm_usage
