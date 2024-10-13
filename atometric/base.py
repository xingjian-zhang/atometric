import os
from hashlib import md5
from typing import Dict, List

from langchain_community.cache import SQLiteCache
from langchain.chat_models.base import BaseChatModel
from langchain.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class FewShotChatModel:
    def __init__(
        self,
        llm_model: BaseChatModel = None,
        system_prompt: str = None,
        user_prompt: str = None,
        few_shot_examples: Dict[str, List[str]] = None,
        structured_output_cls: type = None,
        name: str = None,
        cache: bool = True,
    ):
        """
        Initialize the AtomicStatementDecomposer with a given language model and optional few-shot examples.

        Args:
            llm_model (BaseChatModel, optional): A custom language model to use. Defaults to GPT-4.
            few_shot_examples (Dict[str, List[str]], optional): A dictionary of example sentences and their
            atomic decompositions for few-shot learning. Defaults to None.
            name (str, optional): The name of the decomposer used for caching.
            Defaults to None.
            cache (bool, optional): Whether to use a cache for the language model. Defaults to True.
        """
        # Initialize the system and user prompts.
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.few_shot_examples = few_shot_examples
        if few_shot_examples:
            self.system_prompt += self._format_few_shot_prompt(few_shot_examples)

        # Initialize the language model and the prompt template.
        self.llm_model = (
            llm_model if llm_model else ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
        )
        if name is None:
            name = self.llm_model.model_name
            name += "_" + md5(self.system_prompt.encode()).hexdigest()

        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("user", self.user_prompt)]
        )
        if structured_output_cls is not None:
            self.llm_model = self.llm_model.with_structured_output(
                structured_output_cls, method="json_mode"
            )
        self.chain = self.prompt_template | self.llm_model

        # Initialize the cache.
        if cache:
            os.makedirs(".cache", exist_ok=True)
            cache_path = f".cache/{name}.db"
            set_llm_cache(SQLiteCache(database_path=cache_path))

    def _format_few_shot_prompt(self, few_shot_examples: Dict[str, List[str]]) -> str:
        """
        Format the few-shot examples into a string to be appended to the system prompt.

        Args:
            few_shot_examples (Dict[str, List[str]]): A dictionary where keys are example sentences and
            values are lists of atomic statements.

        Returns:
            str: A formatted string representing the examples.
        """
        examples_str = "\nHere are a few examples for you to follow:\n\n"
        for user_input, decomposed_statements in few_shot_examples.items():
            formatted_decomposed = "\n".join(
                [str(statement) for statement in decomposed_statements]
            )
            examples_str += f"Input: {user_input}\nOutput:\n{formatted_decomposed}\n\n"
        return examples_str
