from typing import Dict, List, Optional

from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel, Field

from atometric.base import FewShotChatModel

DEFAULT_SYSTEM_PROMPT = """
You will be given a sentence or sentences. Decompose them into no more than 10
atomic statements. Each atomic statement is a proposition that is simple,
clear, non-trivial, and self-contained. The subject of each statement must be a
common noun. **Strictly avoid using proper nouns (e.g. abbreviation) or
pronouns in the atomic statements.**

Respond in json with the key 'atomic_statements' containing a list of strings
(atomic statements). The first atomic statement should be a concise summary of
the entire input sentence(s).
"""

USER_PROMPT = """
Please breakdown the following sentence into independent atomicstatements.
**Strictly avoid using proper nouns (e.g. abbreviation) or pronouns in the
atomic statements.**

Input: {sentence}
Output:
"""


class AtomicStatementDecomposeOutput(BaseModel):
    """
    A Pydantic model representing the output of atomic statement decomposition.

    Attributes:
        atomic_statements (List[str]): A list of atomic statements that are decomposed
        from the input sentence(s).
    """

    atomic_statements: List[str] = Field(..., description="List of atomic statements.")


class AtomicStatementDecomposer(FewShotChatModel):
    """
    A class to decompose complex sentences into atomic statements using a language model.

    This class wraps around a language model (LLM) to generate atomic statements that express
    clear and simple propositions from more complex sentences.
    """

    def __init__(
        self,
        llm_model: Optional[BaseChatModel] = None,
        few_shot_examples: Optional[Dict[str, List[str]]] = None,
        name: Optional[str] = None,
        cache: bool = True,
        persona: Optional[str] = None,
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
            persona (str, optional): A specific persona for the system prompt. Defaults to None.
        """
        system_prompt = DEFAULT_SYSTEM_PROMPT
        if persona:
            system_prompt = f"You are {persona}. " + system_prompt

        super().__init__(
            llm_model=llm_model,
            system_prompt=system_prompt.strip(),
            user_prompt=USER_PROMPT.strip(),
            few_shot_examples=few_shot_examples,
            structured_output_cls=AtomicStatementDecomposeOutput,
            name=name,
            cache=cache,
        )

    def decompose(self, sentence: str) -> AtomicStatementDecomposeOutput:
        """
        Decompose a sentence or sentences into atomic statements using the language model.

        Args:
            sentence (str): The sentence or group of sentences to decompose.

        Returns:
            AtomicStatementOutput: A model containing the decomposed atomic statements.
        """
        result = self.chain.invoke({"sentence": sentence})
        return result

    def decompose_batch(
        self, sentences: List[str], max_concurrency: int = 50
    ) -> List[AtomicStatementDecomposeOutput]:
        """
        Decompose a batch of sentences into atomic statements using the language model.

        Args:
            sentences (List[str]): The list of sentences to decompose.

        Returns:
            List[AtomicStatementDecomposeOutput]: A list of models containing the decomposed atomic statements.
        """
        results = self.chain.batch(
            sentences, config={"max_concurrency": max_concurrency}
        )
        return results
