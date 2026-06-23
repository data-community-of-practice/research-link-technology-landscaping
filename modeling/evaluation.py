import asyncio
import config
from langchain.agents import create_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import HumanMessage
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field
from enum import Enum
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
import jsonlines

SYSTEM_PROMPT = """
You are an expert research analyst evaluating the quality of extracted keywords from research grants.

Your task is to assess whether each keyword is:
1. **Actually present** in the grant text (title or description)
2. **Specific and precise** rather than generic or overly broad
3. **Unique** to the particular research (not applicable to 80%+ of grants)
4. **Appropriate length** (1-4 words)
5. **Correctly typed** (General, Methodology, Application, or Technology)

Evaluate each keyword and provide a numerical score from 0-10 and detailed reasoning.
"""

class KeywordQuality(str, Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"

class KeywordEvaluation(BaseModel):
    model_config = {"extra": "forbid"}
    score: int = Field(description="Quality score from 0-10", ge=0, le=10)
    quality: KeywordQuality = Field(description="Overall quality rating")
    is_present_in_text: bool = Field(description="Whether the keyword is actually present in the grant text")
    is_specific: bool = Field(description="Whether the keyword is specific rather than generic")
    is_unique: bool = Field(description="Whether the keyword is unique to this particular research")
    is_appropriate_length: bool = Field(description="Whether the keyword is 1-4 words")
    reasoning: str = Field(description="Detailed explanation of the evaluation")

async def run_evaluation(agent, prompts):
    writer = jsonlines.open("results/keyword_evaluations.jsonl", mode="w")
    total = len(prompts)

    progress_columns = [SpinnerColumn(), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), MofNCompleteColumn()]
    with Progress(*progress_columns, transient=True) as progress:
        task_id = progress.add_task("Evaluating keywords", total=total)

        async for _, response in agent.abatch_as_completed(prompts, config={"max_concurrency": 256}):
            eval_result = response["structured_response"]
            message = response["messages"][0]
            writer.write({
                "keyword_name": message.name,
                "source_grant_id": message.id,
                "score": eval_result.score,
                "quality": eval_result.quality,
                "is_present_in_text": eval_result.is_present_in_text,
                "is_specific": eval_result.is_specific,
                "is_unique": eval_result.is_unique,
                "is_appropriate_length": eval_result.is_appropriate_length,
                "reasoning": eval_result.reasoning,
            })
            progress.advance(task_id, 1)

    writer.close()

def create_evaluation_prompt(record):
    return f"""
Evaluate the following extracted keyword from a research grant for its relevance and accuracy based on the grant's title and description.

Grant Title: {record['title']}
Grant Description: {record['grant_summary']}

Extracted Keyword: {record['name_keyword']}
Extracted Keyword Type: {record['type_keyword']}
Extracted Keyword Description: {record['description_keyword']}

Provide a comprehensive evaluation considering:
- Is the keyword actually present in the grant text?
- Is it specific enough (not generic like "engineering" or "data analysis")?
- Is it unique to this research (not applicable to 80%+ of grants)?
- Is it the right length (1-4 words)?
- Is the type classification correct?
"""

if __name__ == "__main__":
    keywords = config.Keywords.load()
    keywords = keywords.add_suffix("_keyword")
    grants = config.Grants.load()
    grants = grants.set_index("id")
    keywords = keywords.merge(grants, left_on="source_grant_id_keyword", right_index=True, how="left")

    prompts = [
        ChatPromptValue(messages=[HumanMessage(
            content=create_evaluation_prompt(record), 
            id=record['source_grant_id_keyword'],
            name=record['name_keyword']
        )])
        for idx, record in keywords.iterrows()
    ]
    
    model = ChatOpenAI(model=config.OPENAI_MODEL, base_url=config.OPENAI_BASE_URL)
    agent = create_agent(
        model=model, 
        system_prompt=SYSTEM_PROMPT, 
        response_format=ProviderStrategy(KeywordEvaluation)
    )
    
    asyncio.run(run_evaluation(agent, prompts))

