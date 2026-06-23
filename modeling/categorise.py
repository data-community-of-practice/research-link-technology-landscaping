from pydantic import BaseModel, Field
from typing import List


from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from textwrap import dedent
import config
import numpy as np
import umap
from pathlib import Path

import utils
from langchain.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate



def run_reorderining():
    df = config.Grants.load()
    embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, base_url=config.OPENAI_BASE_URL)
    documents = df.apply(lambda row: Document(page_content=f"<grant><title>{row['title']}</title><grant_summary>{row['grant_summary']}</grant_summary></grant>", metadata=row.to_dict()), axis=1).tolist()
    if Path("results/grant_embeddings.npy").exists():
        vec = np.load("results/grant_embeddings.npy")
    else:
        contents = [doc.page_content for doc in documents]
        vec = embeddings.embed_documents(contents)
        np.save("results/grant_embeddings.npy", vec)
    reducer = umap.UMAP(n_components=1, random_state=42)
    vec1d = reducer.fit_transform(vec).flatten()
    ordering = vec1d.argsort()
    df = df.reindex(ordering)
    utils.save_jsonl_file(df.to_dict(orient="records"), "results/grants.jsonl")

class Category(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")


class CategoryList(BaseModel):
    model_config = {"extra": "forbid"}
    categories: List[Category] = Field(description="List of research categories")

prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=dedent(
                """
                You are an expert technology categorisation designer. Your task is to help create clear and distinct research categories for given research documentations. Each category should have a specific focus area and be described in a way that highlights its unique aspects compared to other categories.
                """
            ).strip(),
        ),
        HumanMessagePromptTemplate.from_template(
            dedent(
                """
                Title: {title}
                Grant Summary: {grant_summary}
                """
            ).strip(),
        ),
    ]
)


grants = config.Grants.load()

from itertools import batched

import pandas as pd


grant_batches = [grants.iloc[i:i+5] for i in range(0, len(grants), 5)]

grant_batches[0]

print("\n".join(grant_batches[0].apply(lambda row: f"<title>{row['title']}</title><grant_summary>{row['grant_summary']}</grant_summary>", axis=1).to_list()))

inputs = grants.loc[:, ["title", "grant_summary"]].to_dict(orient="records")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=config.OPENAI_MODEL, base_url=config.OPENAI_BASE_URL)



chain = prompt_template | llm.with_structured_output(CategoryList)

response = chain.invoke(inputs[0])

response.categories[0]


# PROMPT_DESIGNER_PROMPT.batch(inputs)

# async def run(agent, prompts):
#     writer = jsonlines.open("results/keywords.jsonl", mode="a")
#     total = len(prompts)
#     progress_columns = [SpinnerColumn(), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), MofNCompleteColumn()]
#     # Use transient so the progress bar clears after completion
#     with Progress(*progress_columns, transient=True) as progress:
#         task_id = progress.add_task("Extracting keywords", total=total)
#         async for _, response in agent.abatch_as_completed(prompts, config={"max_concurrency": 512}):
#             for k in response["structured_response"].keywords:
#                 writer.write({
#                     "source_grant_id": response["messages"][0].id,
#                     "name": k.name,
#                     "type": k.type,
#                     "description": k.description,
#                 })
#             progress.advance(task_id, 1)

#     writer.close()
        


# if __name__ == "__main__":
    
#     grants = config.Grants.load()
#     template = lambda row: f"<title>{row['title']}</title><grant_summary>{row['grant_summary']}</grant_summary>"
#     prompts = [ChatPromptValue(messages=[HumanMessage(content=template(row), id=row['id'])]) for _, row in grants.iterrows()]
#     model = ChatOpenAI(model=config.OPENAI_MODEL, base_url=config.OPENAI_BASE_URL)
#     agent = create_agent(model=model, system_prompt=SYSTEM_PROMPT, response_format=ProviderStrategy(KeywordsList))
#     asyncio.run(run(agent, prompts))
