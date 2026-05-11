from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence



import config
import typer

from openai import AsyncOpenAI
import jsonlines

from . import llm_client
from .concurrency import run_with_concurrency
from pydantic import BaseModel, Field


class CategoryEvaluation(BaseModel):
    model_config = {"extra": "forbid"}
    category_name: str = Field(description="Name of the category being evaluated")
    coherence_score: int = Field(description="Score from 1-10 indicating how well keywords fit together thematically", ge=1, le=10)
    completeness_score: int = Field(description="Score from 1-10 indicating if category description captures all keyword aspects", ge=1, le=10)
    field_alignment_score: int = Field(description="Score from 1-10 indicating how well the field_of_research assignment matches the category", ge=1, le=10)
    overall_score: int = Field(description="Overall quality score from 1-10", ge=1, le=10)
    strengths: List[str] = Field(description="Key strengths of this categorization")
    weaknesses: List[str] = Field(description="Areas for improvement or concerns")
    suggested_improvements: List[str] = Field(description="Specific suggestions to improve this category")


class EvaluationResults(BaseModel):
    model_config = {"extra": "forbid"}
    evaluations: List[CategoryEvaluation] = Field(description="List of category evaluations")


@dataclass
class EvaluationOutputs:
    evaluations: jsonlines.Writer
    requests: jsonlines.Writer

    def close(self) -> None:
        self.evaluations.close()
        self.requests.close()


def prepare_evaluation_outputs(output_dir: Path | str) -> EvaluationOutputs:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return EvaluationOutputs(
        evaluations=jsonlines.open(output_path / "evaluations.jsonl", "a"),
        requests=jsonlines.open(output_path / "evaluation_requests.jsonl", "a"),
    )


class CategoryEvaluator:
    def __init__(self, outputs: EvaluationOutputs) -> None:
        self.outputs = outputs
        self.total_categories_evaluated = 0
        self.cumulative_scores: Dict[str, float] = {
            "coherence": 0.0,
            "completeness": 0.0,
            "field_alignment": 0.0,
            "overall": 0.0,
        }

    async def run(
        self,
        client: AsyncOpenAI,
        categories: Sequence[Dict[str, Any]],
        keywords_by_name: Dict[str, Dict[str, Any]],
        concurrency: int,
    ) -> List[Dict[str, Any]]:
        async def handler(category: Dict[str, Any]) -> Dict[str, Any]:
            return await self._evaluate_category(client, category, keywords_by_name)

        def update_progress(
            result: Dict[str, Any] | None,
            progress_bar: Any,
            task_id: int,
        ) -> None:
            self._update_progress(result, progress_bar, task_id)

        processed = await run_with_concurrency(
            categories,
            handler,
            concurrency=concurrency,
            progress_description="Evaluating categories",
            progress_unit="category",
            progress_callback=update_progress,
        )
        return processed

    async def _evaluate_category(
        self,
        client: AsyncOpenAI,
        category: Dict[str, Any],
        keywords_by_name: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        category_text = _format_category_for_evaluation(category, keywords_by_name)
        log_entry = {
            "category_name": category.get("name"),
            "request": category_text,
        }
        try:
            evaluation = await _request_evaluation(client, category_text)
            eval_result = evaluation.evaluations[0] if evaluation.evaluations else None
            if eval_result:
                eval_dict = eval_result.model_dump()
                log_entry["response"] = eval_dict
                self.outputs.evaluations.write(eval_dict)
                self.outputs.requests.write(log_entry)
                return eval_dict
            else:
                log_entry["error"] = "No evaluation returned"
                self.outputs.requests.write(log_entry)
                return {}
        except Exception as error:
            log_entry["error"] = str(error)
            self.outputs.requests.write(log_entry)
            return {}

    def _update_progress(
        self,
        result: Dict[str, Any] | None,
        progress_bar: Any,
        task_id: int,
    ) -> None:
        if result is None or not result:
            return

        self.total_categories_evaluated += 1
        self.cumulative_scores["coherence"] += result.get("coherence_score", 0)
        self.cumulative_scores["completeness"] += result.get("completeness_score", 0)
        self.cumulative_scores["field_alignment"] += result.get("field_alignment_score", 0)
        self.cumulative_scores["overall"] += result.get("overall_score", 0)

        if self.total_categories_evaluated > 0:
            avg_coherence = self.cumulative_scores["coherence"] / self.total_categories_evaluated
            avg_completeness = self.cumulative_scores["completeness"] / self.total_categories_evaluated
            avg_overall = self.cumulative_scores["overall"] / self.total_categories_evaluated
            progress_bar.update(
                task_id,
                rate=f"avg scores - coherence: {avg_coherence:.1f}, completeness: {avg_completeness:.1f}, overall: {avg_overall:.1f}",
            )


EVALUATION_SYSTEM_PROMPT = """
You are an expert research methodology evaluator specializing in assessing the quality of technology and research categorization systems.

Your task is to evaluate a single research category by analyzing how well its keywords are grouped together, how accurately it is described, and whether its field_of_research assignment is appropriate.

**IMPORTANT: You MUST complete this evaluation immediately. Do NOT ask for clarification or propose alternatives. Provide your evaluation directly.**

---

### **EVALUATION CRITERIA:**

**1. Coherence (1-10):**
*   Assess how well the keywords fit together thematically
*   Consider whether keywords share common research contexts, methodologies, or application domains
*   Higher scores indicate strong thematic unity; lower scores indicate disparate or weakly related keywords

**2. Completeness (1-10):**
*   Evaluate whether the category description accurately captures the scope and focus of ALL included keywords
*   Check if the description provides sufficient context about the research area
*   Higher scores indicate comprehensive, accurate descriptions; lower scores indicate vague or incomplete descriptions

**3. Field Alignment (1-10):**
*   Assess how well the assigned field_of_research matches the category content
*   Consider whether the FOR division is the best fit among the 23 available options
*   Higher scores indicate optimal FOR assignment; lower scores indicate misalignment

**4. Overall Quality (1-10):**
*   Provide a holistic assessment of the category quality
*   Consider coherence, completeness, field alignment, and whether this category represents a meaningful research domain
*   Higher scores indicate well-constructed, valuable categories; lower scores indicate problematic categorizations

---

### **OUTPUT REQUIREMENTS:**

For each evaluation, provide:
*   **Scores:** Numerical ratings (1-10) for all four criteria
*   **Strengths:** 2-4 specific positive aspects of the categorization
*   **Weaknesses:** 2-4 specific concerns or areas needing improvement
*   **Suggested Improvements:** 2-4 concrete, actionable recommendations

Be specific and analytical in your feedback. Reference actual keywords and category elements in your assessment.
"""


def _format_category_for_evaluation(
    category: Dict[str, Any],
    keywords_by_name: Dict[str, Dict[str, Any]],
) -> str:
    category_name = category.get("name", "Unnamed")
    description = category.get("description", "No description")
    field_of_research = category.get("field_of_research", "Not specified")
    keyword_names = category.get("keywords", [])

    keyword_details = []
    for kw_name in keyword_names:
        kw_data = keywords_by_name.get(kw_name, {})
        kw_type = kw_data.get("type", "Unknown")
        kw_desc = kw_data.get("description", "No description")
        keyword_details.append(f"  - {kw_name} ({kw_type}): {kw_desc}")

    keywords_section = "\n".join(keyword_details) if keyword_details else "  No keywords"

    return f"""
<category>
<name>{category_name}</name>
<description>{description}</description>
<field_of_research>{field_of_research}</field_of_research>
<keywords>
{keywords_section}
</keywords>
</category>
"""


async def _request_evaluation(client: AsyncOpenAI, category_text: str) -> EvaluationResults:
    output_text = await llm_client.call_json_schema(
        client,
        model=config.OPENAI_MODEL,
        system_prompt=EVALUATION_SYSTEM_PROMPT,
        user_content=category_text,
        schema_name="category_evaluation",
        schema=EvaluationResults.model_json_schema(),
    )
    return EvaluationResults.model_validate_json(output_text)


async def evaluate_categories_async(
    keywords_path: Path,
    categories_path: Path,
    output_dir: Path,
    concurrency: int = config.CONCURRENCY,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    import utils
    
    keywords_data = utils.load_jsonl_file(keywords_path, as_dataframe=False)
    keywords_by_name = {kw["name"]: kw for kw in keywords_data if "name" in kw}
    
    categories_data = utils.load_jsonl_file(categories_path, as_dataframe=False)
    if not categories_data:
        return []

    if limit is not None and limit > 0:
        categories_data = categories_data[:limit]

    outputs = prepare_evaluation_outputs(output_dir)
    evaluator = CategoryEvaluator(outputs)
    results: List[Dict[str, Any]] = []
    try:
        async with llm_client.async_client() as client:
            results = await evaluator.run(
                client,
                categories_data,
                keywords_by_name,
                concurrency=concurrency,
            )
    finally:
        outputs.close()

    return results


def evaluate_categories(
    keywords_path: Path | str,
    categories_path: Path | str,
    output_dir: Path | str,
    concurrency: int = config.CONCURRENCY,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    return asyncio.run(
        evaluate_categories_async(
            Path(keywords_path),
            Path(categories_path),
            Path(output_dir),
            concurrency=concurrency,
            limit=limit,
        )
    )


# CLI Application
app = typer.Typer(help="Evaluate category quality from keyword categorization results")


@app.command()
def evaluate(
    keywords_path: Path = typer.Argument(..., help="Path to keywords JSONL file"),
    categories_path: Path = typer.Argument(..., help="Path to categories JSONL file"),
    output_dir: Path = typer.Argument(..., help="Directory to write evaluation outputs"),
    concurrency: int = typer.Option(config.CONCURRENCY, help="Maximum concurrent requests"),
    limit: int = typer.Option(None, help="Limit number of categories to evaluate"),
) -> None:
    """Evaluate the quality of categorized keywords."""
    results = evaluate_categories(
        keywords_path,
        categories_path,
        output_dir,
        concurrency=concurrency,
        limit=limit,
    )
    count = len([r for r in results if r])
    typer.echo(f"Evaluated {count} categor{'ies' if count != 1 else 'y'}")
    
    if count > 0:
        avg_scores = {
            "coherence": sum(r.get("coherence_score", 0) for r in results if r) / count,
            "completeness": sum(r.get("completeness_score", 0) for r in results if r) / count,
            "field_alignment": sum(r.get("field_alignment_score", 0) for r in results if r) / count,
            "overall": sum(r.get("overall_score", 0) for r in results if r) / count,
        }
        typer.echo("\nAverage Scores:")
        typer.echo(f"  Coherence:       {avg_scores['coherence']:.2f}/10")
        typer.echo(f"  Completeness:    {avg_scores['completeness']:.2f}/10")
        typer.echo(f"  Field Alignment: {avg_scores['field_alignment']:.2f}/10")
        typer.echo(f"  Overall:         {avg_scores['overall']:.2f}/10")


if __name__ == "__main__":
    app()
