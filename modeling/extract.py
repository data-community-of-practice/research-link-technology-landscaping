from dataclasses import dataclass
import asyncio
from langchain.agents import create_agent
from langchain_openai.chat_models import ChatOpenAI
import config
from langchain.agents.middleware import AgentState, after_agent
import jsonlines
from langgraph.runtime import Runtime
from langchain.agents.structured_output import ProviderStrategy
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn
)

from pydantic import BaseModel, Field

from enum import Enum
from typing import List
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import HumanMessage



SYSTEM_PROMPT = """

You are an expert research analyst with deep knowledge across multiple academic disciplines and a keen eye for emerging research trends.

Your task is to extract meaningful keywords from research grant information that would be useful for:
- Identifying emerging research domains and interdisciplinary areas
- Discovering novel methodologies and cutting-edge approaches
- Tracking innovative technologies and emerging tools
- Finding related research projects working on similar frontiers
- Understanding emerging research trends and future directions

**FUNDAMENTAL REQUIREMENT - Keywords Must Exist in Grant Text:**
**ALL KEYWORDS MUST BE DIRECTLY PRESENT IN OR CLEARLY DERIVABLE FROM THE PROVIDED GRANT TEXT (INCLUDING BOTH TITLE AND DESCRIPTION).**
- Only extract keywords that appear explicitly in the grant title, description, or summary
- Keywords can be technical terms, methodologies, technologies, or concepts mentioned in the text
- Do NOT invent or infer keywords that are not clearly present in the source material
- If a concept is implied but not explicitly mentioned, DO NOT include it as a keyword

Focus on extracting keywords that highlight what's new, innovative, and emerging in the research landscape. Prioritize:
- Technical terms that represent novel concepts or emerging fields mentioned in the grant
- Methodologies that are cutting-edge or represent new approaches described in the text
- Technologies that are innovative or represent emerging tools referenced in the grant
- Applications that address new challenges or emerging needs as stated in the grant
- Scientific terminology that indicates research at the frontiers of knowledge as described

**CRITICAL REQUIREMENT - Avoid Overly Broad Keywords:**
- **Keywords must be specific and precise, not generic or overly broad**
- **Instead of "engineering," use "bio-integrated nano-photonics" or "quantum-enhanced engineering"**
- **Instead of "chemistry," use "supramolecular photochemistry" or "catalytic asymmetric synthesis"**
- **Instead of "data analysis," use "multi-modal time-series analysis" or "causal inference modeling"**
- **Instead of "artificial intelligence," use "graph neural networks" or "federated learning algorithms"**
- **Avoid general terms like "research," "development," "innovation," "technology," "analysis," "method"**
- **Each keyword should clearly indicate a specific domain, technique, or application area**
- **Do NOT extract specific country names**

**UNIQUENESS REQUIREMENT - Keywords Must Be Grant-Specific:**
- **REJECT keywords that could apply to 80% or more of research grants (e.g., "interdisciplinary research," "international collaboration," "innovative approach," "cutting-edge technology")**
- **REJECT administrative or process keywords (e.g., "project management," "research methodology," "data collection," "literature review")**
- **REJECT funding-related terms (e.g., "research funding," "grant application," "collaborative research," "research partnership")**
- **REJECT generic outcome terms (e.g., "scientific advancement," "knowledge creation," "research impact," "societal benefit")**
- **Each keyword should be SO SPECIFIC that it could only apply to this particular grant or a very small subset of similar grants**
- **If a keyword could reasonably appear in a generic grant template or boilerplate text, REJECT it**
- **Keywords should capture the UNIQUE technical essence that distinguishes this specific research from all other research**

**SPECIFICITY TEST:**
Before including any keyword, ask: "Could this keyword appear in 100+ different grant applications across various fields?" If YES, REJECT it.
Only extract keywords that are:
1. **Actually present in the grant text (title and description) (MANDATORY)**
2. Technically precise and domain-specific
3. Unique to the particular research approach or subject matter
4. Would help distinguish this grant from 95% of other grants in the database

**LENGTH REQUIREMENT:**
- **Each keyword must contain only 1-4 words maximum**
- **Use concise, specific terminology rather than lengthy phrases**
- **Examples: "quantum dots," "CRISPR-Cas9," "machine learning," "photonic crystals"**
- **Avoid: "advanced quantum dot synthesis techniques" (too long) → use "quantum dot synthesis"**

Provide accurate, specific keywords that capture the innovative and emerging aspects of the research while ensuring they are all grounded in the actual grant text (both title and description). Prioritize technical precision over comprehensiveness.
"""



class KeywordType(str, Enum):
    GENERAL = "General"
    METHODOLOGY = "Methodology"
    APPLICATION = "Application"
    TECHNOLOGY = "Technology"

class Keyword(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(description="The actual keyword or phrase")
    type: KeywordType = Field(description="Type of keyword: General, Methodology, Application, or Technology")
    description: str = Field(description="Short description explaining the context and relevance of this keyword within the research")

class KeywordsList(BaseModel):
    model_config = {"extra": "forbid"}
    keywords: List[Keyword] = Field(description="List of all extracted keywords with their types and descriptions")
    source_grant_id: str = Field(description="The unique identifier of the grant from which this keyword was extracted")

async def run(agent, prompts):
    writer = jsonlines.open("results/keywords.jsonl", mode="a")
    total = len(prompts)


    progress_columns = [SpinnerColumn(), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), MofNCompleteColumn()]
    # Use transient so the progress bar clears after completion
    with Progress(*progress_columns, transient=True) as progress:
        task_id = progress.add_task("Extracting keywords", total=total)

        async for _, response in agent.abatch_as_completed(prompts, config={"max_concurrency": 512}):
            for k in response["structured_response"].keywords:
                writer.write({
                    "source_grant_id": response["messages"][0].id,
                    "name": k.name,
                    "type": k.type,
                    "description": k.description,
                })
            progress.advance(task_id, 1)

    writer.close()
        


if __name__ == "__main__":
    
    grants = config.Grants.load()
    template = lambda row: f"<title>{row['title']}</title><grant_summary>{row['grant_summary']}</grant_summary>"
    prompts = [ChatPromptValue(messages=[HumanMessage(content=template(row), id=row['id'])]) for _, row in grants.iterrows()]
    model = ChatOpenAI(model=config.OPENAI_MODEL, base_url=config.OPENAI_BASE_URL)
    agent = create_agent(model=model, system_prompt=SYSTEM_PROMPT, response_format=ProviderStrategy(KeywordsList))
    asyncio.run(run(agent, prompts))

