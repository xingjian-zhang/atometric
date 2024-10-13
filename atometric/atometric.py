# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atometric is a metric for evaluating the quality of atomic statements in a text given reference information."""

import json
import os
from typing import Dict, List

import datasets
import evaluate
from langchain.chat_models.base import BaseChatModel

from atometric.decomposer import AtomicStatementDecomposer
from atometric.verifier import AtomicStatementVerifier

# TODO: Add BibTeX citation
_CITATION = None

# TODO: Add description of the module here
_DESCRIPTION = """
Atometric is a metric for evaluating the quality of atomic statements in a text given reference information.
It first decomposes the text into atomic statements and then verifies each statement given the reference information.
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    precision: whether each atomic statement in the prediction meets certain criteria when compared to the reference
    recall: whether each atomic statement in the reference meets certain criteria when compared to the prediction
    f1: harmonic mean of precision and recall
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Atometric(evaluate.Metric):
    """
    Atometric is a metric for evaluating the quality of atomic statements in a text given reference information.
    It first decomposes the text into atomic statements and then verifies each statement given the reference information.
    """

    def __init__(
        self,
        system_prompt: str,
        llm_model: BaseChatModel = None,
        few_shot_examples_decomposer: Dict[str, List[str]] = None,
        name: str = None,
        cache: bool = False,
        debug: bool = False,
    ):
        """
        Args:
            system_prompt: The system prompt to use for the language model.
            llm_model: The LLM model to use for decomposing and verifying atomic statements.
            few_shot_examples_decomposer: A dictionary of few-shot examples for decomposing atomic statements.
            name: The name of the metric.
            cache: Whether to cache the results of the metric.
            debug: Whether to log intermediate steps for debugging.
        """
        super().__init__()
        self.debug = debug
        self.decomposer = AtomicStatementDecomposer(
            llm_model=llm_model,
            few_shot_examples=few_shot_examples_decomposer,
            name=name,
            cache=cache,
        )
        self.verifier = AtomicStatementVerifier(
            system_prompt=system_prompt,
            llm_model=llm_model,
            name=name,
            cache=cache,
        )

    @classmethod
    def from_task_type(cls, task_type: str):
        if task_type not in (
            "cs_research_context",
            "cs_research_idea",
            "persona",
            "red_team_attempts",
        ):
            raise ValueError(f"Invalid task type: {task_type}")
        pwd = os.path.dirname(os.path.abspath(__file__))
        few_shot_examples_path = os.path.join(
            pwd, "assets", "examples", f"{task_type}.json"
        )
        system_prompt_path = os.path.join(
            pwd, "assets", "system_prompts", f"{task_type}.txt"
        )
        with open(few_shot_examples_path, "r") as f:
            few_shot_examples = json.load(f)
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        return cls(
            system_prompt=system_prompt, few_shot_examples_decomposer=few_shot_examples
        )

    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
        )

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _compute(self, predictions, references):
        """Returns the scores"""
        atomic_statements_pred = self.decomposer.decompose_batch(predictions)
        atomic_statements_ref = self.decomposer.decompose_batch(references)

        atomic_statements_pred = [
            result.atomic_statements for result in atomic_statements_pred
        ]
        atomic_statements_ref = [
            result.atomic_statements for result in atomic_statements_ref
        ]

        # Verify precision
        precision_results = self.verifier.verify_batch(
            atomic_statements_pred, references
        )

        # Verify recall
        recall_results = self.verifier.verify_batch(atomic_statements_ref, predictions)

        # Compute precision, recall, and f1 by averaging the results
        precision = self._average_results(precision_results)
        recall = self._average_results(recall_results)
        f1 = {k: self._f1_score(precision[k], recall[k]) for k in precision}

        # Print the intermediate results
        for pred_statement, ref, precision_result in zip(
            atomic_statements_pred, references, precision_results
        ):
            self._debug_print(f"Reference: {ref}")
            for statement, result in zip(pred_statement, precision_result):
                self._debug_print(f"  Statement: {statement}")
                self._debug_print(f"    Result: {result}")

        for ref_statement, pred, recall_result in zip(
            atomic_statements_ref, predictions, recall_results
        ):
            self._debug_print(f"Prediction: {pred}")
            for statement, result in zip(ref_statement, recall_result):
                self._debug_print(f"  Statement: {statement}")
                self._debug_print(f"    Result: {result}")

        return {"precision": precision, "recall": recall, "f1": f1}

    def _f1_score(self, precision: float, recall: float) -> float:
        """Compute the f1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _average_results(self, results: List[Dict[str, bool]]) -> Dict[str, float]:
        """
        Average the results of the verification.

        Args:
            results: A list of dictionaries containing verification results for each prediction.

        Returns:
            A dictionary with averaged scores for 'loose', 'moderate', and 'strict' criteria.
        """
        score_keys = ["loose", "moderate", "strict"]
        scores = {key: 0.0 for key in score_keys}

        for per_pred_result in results:
            per_pred_scores = {key: 0.0 for key in score_keys}

            # Sum up scores for each statement in the prediction
            for per_statement_result in per_pred_result:
                for key in score_keys:
                    per_pred_scores[key] += per_statement_result[key]

            # Calculate average for this prediction and add to total scores
            num_statements = len(per_pred_result)
            for key in score_keys:
                avg_score = per_pred_scores[key] / num_statements
                scores[key] += avg_score

        # Calculate final average across all predictions
        num_predictions = len(results)
        for key in score_keys:
            scores[key] /= num_predictions

        return scores
