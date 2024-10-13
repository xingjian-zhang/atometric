from typing import Dict, List, Optional

from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel, Field

from atometric.base import FewShotChatModel


USER_PROMPT = """
Please verify the following atomic statement given relevant reference information.

- Reference information
{reference}

- Atomic statement
{statement}
"""


class AtomicStatementVerifierOutput(BaseModel):
    """
    A Pydantic model representing the output of atomic statement verification.

    Attributes:
        level (int): The level of support for the atomic statement.
    """

    level: int = Field(
        ..., description="The level of support for the atomic statement."
    )


def verifier_output_to_dict(
    output: AtomicStatementVerifierOutput,
) -> Dict[str, bool]:
    return dict(
        strict=output.level == 1,
        moderate=output.level <= 2,
        loose=output.level <= 3,
    )


class AtomicStatementVerifier(FewShotChatModel):
    """
    A class to verify the truth/relevancy of atomic statements using a language model.

    This class wraps around a language model (LLM) to verify the truth of atomic statements
    by generating a response that confirms or denies the truth of the statement.
    """

    def __init__(
        self,
        system_prompt: str,
        llm_model: Optional[BaseChatModel] = None,
        name: Optional[str] = None,
        cache: bool = True,
    ):
        """
        Initialize the AtomicStatementVerifier with a given language model and optional few-shot examples.

        Args:
            system_prompt (str): The system prompt to use for the language model.
            llm_model (BaseChatModel, optional): A custom language model to use. Defaults to GPT-4.
            system_prompt (str, optional): The system prompt to use for the language model. Defaults to None.
            user_prompt (str, optional): The user prompt to use for the language model. Defaults to None.
            name (str, optional): The name of the verifier used for caching. Defaults to None.
            cache (bool, optional): Whether to use a cache for the language model. Defaults to True.
        """
        super().__init__(
            llm_model=llm_model,
            system_prompt=system_prompt.strip(),
            user_prompt=USER_PROMPT.strip(),
            structured_output_cls=AtomicStatementVerifierOutput,
            name=name,
            cache=cache,
        )

    def verify_batch(
        self,
        statements: List[List[str]],
        references: List[str],
        max_concurrency: int = 50,
    ) -> List[List[Dict[str, bool]]]:
        # Flatten the list of statements and references
        tasks = []
        for statement_list, reference in zip(statements, references):
            for statement in statement_list:
                tasks.append({"statement": statement, "reference": reference})
        flattened_results = self.chain.batch(
            tasks, config={"max_concurrency": max_concurrency}
        )
        flattened_results = [
            verifier_output_to_dict(result) for result in flattened_results
        ]
        # Rearrange the results to match the original structure
        results = []
        index = 0
        for statement_list in statements:
            result_sublist = flattened_results[index : index + len(statement_list)]
            results.append(result_sublist)
            index += len(statement_list)
        return results
