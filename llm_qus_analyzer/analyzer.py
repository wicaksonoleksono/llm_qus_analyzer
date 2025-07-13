import json
from typing import Any, Callable, Generic, TypeVar
from langchain_core.prompts import ChatPromptTemplate
from .client import LLMClient

T = TypeVar("T")


class LLMAnalyzer(Generic[T]):
    """A generic analyzer for processing and evaluating content using LLMs.

    This class provides a framework for defining analysis tasks, building prompts,
    parsing results, and running evaluations through an LLM client.

    Type Parameters:
        T: The type of the parsed output that will be returned by the analyzer.
    """

    def __init__(self, key: str):
        """Initializes the LLMAnalyzer with a unique key.

        Args:
            key (str): A unique identifier for this analyzer instance, used to 
                manage prompt injection in the LLM client.
        """
        self.__key = key

    def build_prompt(self, definition: str, in_format: str, out_format: str):
        """Constructs the prompt template for the LLM analysis task.

        Args:
            definition (str): Description of the analysis task or framework being used.
            in_format (str): Explanation of the input format the LLM should expect.
            out_format (str): Description of the desired output format.
        """
        self.__prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are an expert in evaluating user stories according to the Quality User Story (QUS) framework.'),
            ('user', f'{definition}\n{in_format}\n{out_format}'),
        ])

    def build_parser(self, parser: Callable[[Any], T]):
        """Sets up the parser function for processing the LLM's raw output.

        Args:
            parser (Callable[[Any], T]): A function that takes the raw JSON output
                from the LLM and returns a parsed result of type T.
        """
        self.__parser = parser

    def run(self, client: LLMClient, value: dict, which_model: int) -> tuple[T, Any]:
        """Executes the analysis using the specified LLM model.

        Args:
            client (LLMClient): The LLM client to use for the analysis.
            value (dict): The input values to be analyzed.
            which_model (int): Index of the specific LLM model to use.

        Returns:
            tuple[T, LLMResult]: A tuple containing:
                - The parsed result (of type T)
                - The raw LLMResult object with complete response details

        Raises:
            NotImplementedError: If either the prompt or parser haven't been built.
            json.JSONDecodeError: If the LLM's output isn't valid JSON.

        Note:
            The method automatically cleans common JSON formatting artifacts
            (like code block markers) before parsing.
        """
        if not hasattr(self, '_LLMAnalyzer__prompt'):
            raise NotImplementedError(
                'There is no prompt exist. Please build the prompt first')
        if not hasattr(self, '_LLMAnalyzer__parser'):
            raise NotImplementedError(
                'There is no parser exist. Please build the parser first')
        client.inject_prompt(self.__key, self.__prompt)
        raw_result = client.run(value, [which_model])[which_model]
        raw_json = raw_result.content.replace('```', '').replace('json\n', '')
        parsed_result = self.__parser(json.loads(raw_json))
        return parsed_result, raw_result
