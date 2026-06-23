from pathlib import Path
from dataclasses import dataclass

from dotenv import dotenv_values

import utils
import pandas as pd
CONFIG = dotenv_values()
ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

OPENAI_BASE_URL = CONFIG.get("OPENAI_BASE_URL", None)
OPENAI_MODEL = CONFIG.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_EMBEDDING_MODEL = CONFIG.get("OPENAI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_TIMEOUT_SECONDS = 600
CONCURRENCY = 512


def _item_template(record):
    return f"""<item><name>{record['name']}</name><description>{record['description']}</description></item>"""


@dataclass
class Keywords:
    # keywords_dir = RESULTS_DIR / "keywords"
    # keywords_dir.mkdir(parents=True, exist_ok=True)
    # extracted_keywords_path: Path = keywords_dir / "extracted_keywords.jsonl"
    keywords_path: Path = RESULTS_DIR / "keywords.jsonl"
    template = staticmethod(_item_template)
    
    def load(as_dataframe=True):
        return utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=as_dataframe)
    
    # def load_extracted(as_dataframe=True):
    #     return utils.load_jsonl_file(Keywords.extracted_keywords_path, as_dataframe=as_dataframe)
    


@dataclass
class Categories:

    CATEGORIY_PATH = RESULTS_DIR / "categories.jsonl"
    template = staticmethod(_item_template)
    
    def last_merged():
        list_of_digits_dirs = [i for i in RESULTS_DIR.iterdir() if i.stem.isdigit() and i.is_dir()]
        if len(list_of_digits_dirs) == 0:
            return None
        else:
            return max([int(i.stem) for i in list_of_digits_dirs])
    def last_merged_path():
        return RESULTS_DIR / str(Categories.last_merged()) / "output.jsonl"
    def load_last_merged():
        return utils.load_jsonl_file(Categories.last_merged_path(), as_dataframe=True)

    def load(as_dataframe=True):
        return utils.load_jsonl_file(Categories.CATEGORIY_PATH, as_dataframe=as_dataframe)

@dataclass
class Grants:
    source_path = RESULTS_DIR / "grants.json"
    grants_path = RESULTS_DIR / "grants.jsonl"

    @staticmethod
    def template(record):
        return (
            f"<grant><title>{record.get('title')}</title>"
            f"<description>{record.get('grant_summary')}</description></grant>"
        )
    @staticmethod
    def preprocess():
        df = utils.load_json_file(Grants.source_path, as_dataframe=True)
        df = df[~df.title.isna()] 
        df = df[~df.grant_summary.isnull()]
        df = df[df.grant_summary != "Not Applicable"]
        df = df[~df.title.str.contains("equipment grant", case=False) & ~df.title.str.contains("travel grant", case=False)]

        
        df["funding_amount"] = pd.to_numeric(df["funding_amount"], errors="coerce")
        df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year.astype("Int64")
        df["end_year"] = pd.to_datetime(df["end_date"], errors="coerce").dt.year.astype("Int64")
        utils.save_jsonl_file(df.to_dict(orient="records"), Grants.grants_path)
        return df

    @staticmethod
    def load(as_dataframe: bool = True):
        if not Grants.grants_path.exists():
            Grants.preprocess()
        return utils.load_jsonl_file(Grants.grants_path, as_dataframe=as_dataframe)
    
    