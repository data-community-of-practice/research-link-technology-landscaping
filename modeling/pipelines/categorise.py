from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import config
from models import FieldOfResearch

from openai import AsyncOpenAI

import jsonlines

from . import llm_client
from .cluster_io import ClusterBatch, load_cluster_file
from .concurrency import run_with_concurrency
from pydantic import BaseModel, Field


@dataclass
class CategoryOutputs:
    categories: jsonlines.Writer
    unknown: jsonlines.Writer
    missing: jsonlines.Writer
    requests: jsonlines.Writer

    def close(self) -> None:
        self.categories.close()
        self.unknown.close()
        self.missing.close()
        self.requests.close()


def prepare_category_outputs(output_dir: Path | str) -> CategoryOutputs:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return CategoryOutputs(
        categories=jsonlines.open(output_path / "output.jsonl", "a"),
        unknown=jsonlines.open(output_path / "unknown.jsonl", "a"),
        missing=jsonlines.open(output_path / "missing.jsonl", "a"),
        requests=jsonlines.open(output_path / "requests.jsonl", "a"),
    )


def write_categorisation_result(
    records: Iterable[dict[str, Any]],
    result: Any,
    outputs: CategoryOutputs,
) -> tuple[int, int, int]:
    keyword_records = [record for record in records if record.get("name")]
    records_by_keyword: dict[str, list[dict[str, Any]]] = {}
    for record in keyword_records:
        name = record["name"]
        bucket = records_by_keyword.get(name)
        if bucket is None:
            records_by_keyword[name] = [record]
        else:
            bucket.append(record)
    assigned_keywords: set[str] = set()
    unknown_count = 0

    for category in result.categories:
        payload = category.model_dump()
        keywords = payload.get("keywords", [])
        assigned_keywords.update(keywords)
        if payload.get("name") == "Unknown":
            for keyword in keywords:
                matches = records_by_keyword.get(keyword, [])
                for match in matches:
                    outputs.unknown.write(match)
                unknown_count += len(matches)
        else:
            outputs.categories.write(payload)

    missing_count = 0
    for record in keyword_records:
        if record["name"] not in assigned_keywords:
            outputs.missing.write(record)
            missing_count += 1

    total_keywords = len(keyword_records)
    return total_keywords, missing_count, unknown_count


class Category(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: List[str] = Field(description="List of keywords associated with this category")
    field_of_research: FieldOfResearch = Field(description="The field of research division this category falls under")


class CategoryList(BaseModel):
    model_config = {"extra": "forbid"}
    categories: List[Category] = Field(description="List of research categories")



class KeywordCategoriser:
    def __init__(self, outputs: CategoryOutputs) -> None:
        self.outputs = outputs
        self.total_keywords_processed = 0
        self.total_missing_keywords = 0
        self.total_unknown_keywords = 0

    async def run(
        self,
        client: AsyncOpenAI,
        batches: Sequence[ClusterBatch],
        concurrency: int,
    ) -> list[tuple[int, str, int, int, int]]:
        async def handler(batch: ClusterBatch) -> tuple[int, str, int, int, int]:
            return await self._handle_batch(client, batch)

        def update_progress(
            result: tuple[int, str, int, int, int] | None,
            progress_bar: Any,
            task_id: int,
        ) -> None:
            self._update_progress(result, progress_bar, task_id)

        processed = await run_with_concurrency(
            batches,
            handler,
            concurrency=concurrency,
            progress_description="Categorising keywords",
            progress_unit="batch",
            progress_callback=update_progress,
        )
        if processed:
            processed.sort(key=lambda item: item[0])
        return processed

    async def _handle_batch(
        self,
        client: AsyncOpenAI,
        batch: ClusterBatch,
    ) -> tuple[int, str, int, int, int]:
        records = batch.records
        batch_text = _format_keywords(records)
        log_entry = {
            "batch_id": batch.batch_id,
            "request": batch_text,
        }
        try:
            categories = await _request_categories(client, batch_text)
            log_entry["response"] = categories.model_dump()
            self.outputs.requests.write(log_entry)
            total_keywords, missing_keywords, unknown_keywords = write_categorisation_result(
                records,
                categories,
                self.outputs,
            )
            return batch.index, batch.batch_id, total_keywords, missing_keywords, unknown_keywords
        except Exception as error:
            log_entry["error"] = str(error)
            self.outputs.requests.write(log_entry)
            keyword_records = [record for record in records if record.get("name")]
            for record in keyword_records:
                self.outputs.unknown.write(record)
            total_keywords = len(keyword_records)
            return batch.index, batch.batch_id, total_keywords, 0, total_keywords

    def _update_progress(
        self,
        result: tuple[int, str, int, int, int] | None,
        progress_bar: Any,
        task_id: int,
    ) -> None:
        if result is None:
            return

        _, _, batch_total, batch_missing, batch_unknown = result
        self.total_keywords_processed += batch_total
        self.total_missing_keywords += batch_missing
        self.total_unknown_keywords += batch_unknown
        if self.total_keywords_processed:
            missing_rate = self.total_missing_keywords / self.total_keywords_processed
            unknown_rate = self.total_unknown_keywords / self.total_keywords_processed
            progress_bar.update(
                task_id,
                rate=f"missing rate {missing_rate:.2%} | unknown rate {unknown_rate:.2%}",
            )


CATEGORISE_SYSTEM_PROMPT = """
You are an expert Technology Analyst and Innovation Forecaster, specializing in synthesizing information to identify and map emerging technological domains for strategic planning.

Your task is to analyze a list of user-provided keywords and generate a comprehensive set of research and technology categories in a specific JSON format. Each category must be linked to an appropriate FOR (Fields of Research) division.

**IMPORTANT: You MUST complete this task immediately and provide the full categorization. Do NOT ask for clarification, approval, or propose alternative approaches. Proceed directly with categorizing all provided keywords.**
å
**Core Objective:**
Your primary goal is to organize the provided keywords into meaningful, emergent categories that bridge the specificity of the keywords with the broadness of the 23 top-level FOR divisions. Your analysis should favor the identification of potential breakthroughs and new interdisciplinary fields.

---

### **CRITICAL RULES FOR CATEGORIZATION:**

**1. Category Creation & Granularity:**
*   **Natural Groupings:** Create categories based on the natural, thematic relationships between the keywords. The number of categories should be determined by these organic groupings, not a predefined target.
*   **Optimal Abstraction:** Categories should be more specific than the broad FOR divisions but more general than individual keywords. They should represent recognizable research areas.
*   **Focus on Emergence:** Actively look for intersections between keywords that suggest new or interdisciplinary domains. When in doubt, err on the side of creating a new, more specific category to avoid missing potential innovations.

**2. Keyword Handling & Coverage:**
*   **Complete Coverage:** EVERY keyword term provided by the user MUST be included in the `keywords` list of exactly one category. No keywords should be left uncategorized.
*   **No Placeholder Categories:** Do NOT create categories like "Clarification Required" or similar placeholder responses. Only create real, meaningful research categories.
*   **Contextual Analysis:** Use the keyword `(type)` and `description` to understand the context and make more informed grouping decisions.
*   **Exact Term Usage:** The `keywords` array within each category must contain the exact `keyword_term` strings from the input.

**3. Output JSON Structure & Content:**
*   **Field of Research Assignment:** Every category MUST be assigned to exactly ONE of the 23 field of research divisions using the descriptive enum name (e.g., "INFORMATION_COMPUTING_SCIENCES", "PHYSICAL_SCIENCES", "ENGINEERING"). The field_of_research field should contain the full enum name as a string.
*   **Available field of research Divisions:** AGRICULTURAL_VETERINARY_FOOD_SCIENCES, BIOLOGICAL_SCIENCES, BIOMEDICAL_CLINICAL_SCIENCES, BUILT_ENVIRONMENT_DESIGN, CHEMICAL_SCIENCES, COMMERCE_MANAGEMENT_TOURISM_SERVICES, CREATIVE_ARTS_WRITING, EARTH_SCIENCES, ECONOMICS, EDUCATION, ENGINEERING, ENVIRONMENTAL_SCIENCES, HEALTH_SCIENCES, HISTORY_HERITAGE_ARCHAEOLOGY, HUMAN_SOCIETY, INDIGENOUS_STUDIES, INFORMATION_COMPUTING_SCIENCES, LANGUAGE_COMMUNICATION_CULTURE, LAW_LEGAL_STUDIES, MATHEMATICAL_SCIENCES, PHILOSOPHY_RELIGIOUS_STUDIES, PHYSICAL_SCIENCES, PSYCHOLOGY
*   **Category Naming:** Each category requires a short, descriptive `name` (ideally under 4 words) that captures its core focus.
*   **High-Quality Descriptions:** The `description` for each category must be detailed and insightful. It should explain the category's scope and key focus areas, directly reflecting the keywords it contains.

**4. MANDATORY "Unknown" Category for Outliers:**
*   **Creation:** You MUST create a category with the exact name `Unknown`. This category serves as a container for any keywords that are true outliers.
*   **Condition for Use:** Place a keyword in the `Unknown` category ONLY if, after careful analysis, it cannot be logically grouped with any other keywords to form a coherent, thematic category. This is the designated place for single, disparate concepts that do not fit elsewhere.
*   **Description Requirement:** The `description` for the `Unknown` category must explicitly state: "This category contains disparate keywords and technologies that do not fit into the other defined domains and represent potential standalone areas of research."
*   **Field of Research Assignment:** For the `Unknown` category, analyze the keywords placed within it and assign the field of research division that represents the best possible fit for the majority of these keywords, or select "HUMAN_SOCIETY" if no clear fit emerges as it serves as the most general multidisciplinary division.
"""


def _format_keywords(records: Sequence[dict]) -> str:
    entries: Iterable[str] = (config.Keywords.template(record) for record in records)
    return "\n".join(entries)


async def _request_categories(client: AsyncOpenAI, batch_text: str) -> CategoryList:
    output_text = await llm_client.call_json_schema(
        client,
        model=config.OPENAI_MODEL,
        system_prompt=CATEGORISE_SYSTEM_PROMPT,
        user_content=batch_text,
        schema_name="keyword_categories",
        schema=CategoryList.model_json_schema(),
    )
    return CategoryList.model_validate_json(output_text)


async def categorise_async(
    clusters_path: Path,
    output_dir: Path,
    concurrency: int = config.CONCURRENCY,
    limit: int | None = None,
) -> List[str]:
    cluster_file = load_cluster_file(clusters_path)
    batches = list(cluster_file.batches)
    if not batches:
        return []

    if limit is not None and limit > 0:
        batches = batches[:limit]

    outputs = prepare_category_outputs(output_dir)
    categoriser = KeywordCategoriser(outputs)
    processed: list[tuple[int, str, int, int, int]] = []
    try:
        async with llm_client.async_client() as client:
            processed = await categoriser.run(
                client,
                batches,
                concurrency=concurrency,
            )
    finally:
        outputs.close()

    return [batch_id for _, batch_id, _, _, _ in processed]


def categorise(
    clusters_path: Path | str,
    output_dir: Path | str,
    concurrency: int = config.CONCURRENCY,
    limit: int | None = None,
) -> List[str]:
    return asyncio.run(
        categorise_async(
            Path(clusters_path),
            Path(output_dir),
            concurrency=concurrency,
            limit=limit,
        )
    )
