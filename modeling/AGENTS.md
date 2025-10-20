# Repository Guidelines

## Project Overview
This repository implements an automated research landscape analysis pipeline that extracts keywords from research grants, organizes them into hierarchical categories, and visualizes the technology landscape through an interactive Streamlit dashboard. The system uses LLM-powered extraction, semantic embeddings, and UMAP-based clustering to identify research trends and emerging technologies across grant portfolios.

## Project Structure & Module Organization

### Core Entry Points
- **`cli.py`**: Main Typer CLI application exposing commands for keyword extraction (`extract`), categorization (`categorise`), merging (`merge`), embedding (`embed`), clustering (`cluster`), and combined workflows (`embed-cluster`)
- **`web/entrypoint.py`**: Streamlit application entry point displaying dataset summary statistics and navigation to analysis pages

### Pipeline Modules (`pipelines/`)
- **`extract.py`**: LLM-powered keyword extraction from grant titles/descriptions with structured output (name, type, description). Implements concurrency control and checkpoint recovery for previously processed grants
- **`categorise.py`**: Groups keywords into thematic categories with descriptions and field-of-research classifications using hierarchical LLM prompts over keyword clusters
- **`merge.py`**: Consolidates duplicate/similar categories across multiple categorization runs by identifying semantic overlap and merging source categories
- **`clustering.py`**: UMAP-based dimensionality reduction and ordering to create semantically coherent clusters from embeddings, with configurable batch sizes
- **`embedding.py`**: Creates semantic embeddings using SentenceTransformers (default model configurable via `DEFAULT_EMBEDDING_MODEL`)
- **`llm_client.py`**: Async OpenAI client wrapper for JSON schema-based completions with timeout handling
- **`concurrency.py`**: Rich progress bar integration for async batch processing with semaphore-based concurrency limiting
- **`cluster_io.py`**: Helper utilities for loading and writing cluster JSON files

### Configuration & Data Models
- **`config.py`**: Centralized configuration using dotenv for LLM endpoints, models, concurrency limits. Defines `Keywords`, `Categories`, and `Grants` dataclasses with paths and template methods for LLM input formatting
- **`models.py`**: Pydantic models for structured outputs (`Keyword`, `KeywordsList`, `Category`, `CategoryList`, `MergedCategory`) with strict validation and field-of-research enum
- **`utils.py`**: JSONL/JSON file I/O, keyword normalization using NLTK stemming for deduplication
- **`process.py`**: Keyword deduplication logic selecting best variant by description length
- **`metric.py`**: Custom inspect-ai metric for totaling sample scores
- **`scorer.py`**: Scoring utilities (inspect_ai integration)

### Web Application (`web/`)
- **`entrypoint.py`**: Landing page with dataset statistics (keywords, grants, funders, date range)
- **`pages/`**: Multi-page Streamlit app with views for keywords, categories, grants, organizations, research landscape visualization, raw data exploration, and organizational comparisons
- **`shared_utils.py`**: Common data loading, filtering, and rendering utilities for Streamlit pages
- **`static/css/styles.css`**: Custom styling for the web interface

### Data Artifacts (`results/`)
- **`grants.json`**: Source grant data (raw)
- **`grants.jsonl`**: Preprocessed grants with filtered equipment/travel grants and date normalization
- **`keywords/`**: Extracted keywords, embeddings, and keyword-level clusters
- **`categories/`**: Numbered iteration directories (1, 2, 3...) containing category outputs, embeddings, and cluster assignments
- **`categories.jsonl`**: Final consolidated categories

## Build, Test, and Development Commands

### Installation
```bash
uv sync  # Installs dependencies and links sibling pyrla package
```

### Running Pipelines
```bash
# Extract keywords from grants (with checkpoint recovery)
uv run python cli.py extract [--skip-finished] [--concurrency N]

# Generate embeddings for keywords
uv run python cli.py embed keywords.jsonl embeddings.npy [--model MODEL] [--force]

# Create clusters from embeddings
uv run python cli.py cluster keywords.jsonl embeddings.npy clusters.json [--batch-size N]

# Combined embedding + clustering
uv run python cli.py embed-cluster keywords.jsonl embeddings.npy clusters.json

# Categorize keyword clusters into research themes
uv run python cli.py categorise clusters.json output_dir/ [--concurrency N] [--limit N]

# Merge similar categories across iterations
uv run python cli.py merge category_clusters.json output_dir/ [--concurrency N]
```

### Web Dashboard
```bash
uv run streamlit run web/entrypoint.py  # Launch on http://localhost:8501
```

### Docker Deployment
```bash
docker compose up --build  # Requires .env file with API credentials
# Mounts results/ read-only, exposes port 8501
```

### Development Workflow
- Use `demo.ipynb` for exploratory data analysis and prototyping transformations
- Clear notebook outputs before committing (`Cell → All Output → Clear`)
- Validate changes by running relevant CLI pipeline stages against sample data
- Test UI changes by exercising new Streamlit page paths

## Coding Style & Naming Conventions
- **Python Version**: 3.12 with modern type hints and dataclasses
- **Indentation**: Four spaces (no tabs)
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Type Hints**: Required for function signatures (see `pipelines/clustering.py` as reference)
- **Documentation**: Short docstrings for non-obvious logic; avoid over-commenting
- **Function Design**: Prefer small, composable functions with single responsibilities
- **Configuration**: Keep module-level settings in `config.py` to centralize environment-dependent values
- **Streamlit Patterns**: Follow existing widget usage (st.metric, st.columns, st.sidebar) and data loading patterns from `shared_utils.py`
- **Error Handling**: Minimal; avoid excessive conditionals and loops per instructions
- **Print Statements**: Do NOT use in final code (testing only, then remove)

## Key Dependencies & Technologies
- **LLM Integration**: OpenAI-compatible API (configurable endpoint) with JSON schema mode for structured extraction
- **Embedding Models**: SentenceTransformers (Hugging Face) for semantic encoding
- **Clustering**: UMAP for dimensionality reduction, custom ordering-based clustering
- **Web Framework**: Streamlit with multi-page app architecture
- **Data Processing**: pandas, numpy, jsonlines for data manipulation
- **Async Processing**: asyncio with Rich progress bars and semaphore-based concurrency
- **Validation**: Pydantic v2 with strict mode for structured outputs
- **Normalization**: NLTK PorterStemmer for keyword deduplication
- **Package Management**: uv for fast dependency resolution and virtual environment management

## Testing Guidelines
- **No Automated Tests**: Currently no pytest suite; validation via manual pipeline execution
- **Validation Process**: 
  1. Run CLI commands against sample data in `results/`
  2. Verify JSONL outputs contain expected structure
  3. Check Streamlit UI for correct data display
  4. Use `demo.ipynb` for exploratory testing of data transformations
- **Future Test Structure**: Use `test_<module>.py` naming, execute with `uv run pytest`
- **Checkpoint Verification**: Test resume behavior by running extract with `--skip-finished` after partial completion

## Commit & Pull Request Guidelines
- **Commit Messages**: Short, imperative style (e.g., "add merge workflow", "fix keyword deduplication")
- **Atomic Commits**: Group related changes (code + data migrations) in single commit
- **PR Description Template**:
  - Workflow touched (extract/categorise/merge/UI)
  - Commands executed for testing (e.g., `uv run python cli.py extract --concurrency 256`)
  - Link to related notebooks or issues
  - Screenshots/GIFs for Streamlit UI changes
- **Data Artifacts**: Include regenerated `results/` files when schema changes require data refresh
- **Code Review Focus**: Verify pipeline outputs are correct, UI accurately reflects data, no secrets committed

## Data & Environment Notes

### Environment Configuration
- **Required Secrets** (`.env`):
  - `OPENAI_BASE_URL`: LLM API endpoint (default: `http://localhost:8000/v1`)
  - `OPENAI_MODEL`: Model identifier (default: `Qwen/Qwen3-4B-Instruct-2507`)
  - `EMBEDDING_MODEL`: Embedding model (default: `Qwen/Qwen3-Embedding-8B`)
- **Never commit** `.env` file or credentials to repository

### Data Flow
1. **Input**: `grants.json` → preprocessed to `grants.jsonl` (filters equipment/travel grants)
2. **Extraction**: Grants → LLM keywords → `keywords/extracted_keywords.jsonl`
3. **Deduplication**: Extracted keywords → `keywords/keywords.jsonl` (normalized, merged variants)
4. **Embedding**: Keywords → `keywords/embeddings.npy` (semantic vectors)
5. **Clustering**: Embeddings → `keywords/keywords_clusters.json` (grouped by similarity)
6. **Categorization**: Keyword clusters → `categories/N/output.jsonl` (thematic groupings)
7. **Merging**: Category iterations → consolidated `categories.jsonl`

### Data Artifacts (`results/`)
- **Immutable Sources**: `grants.json` (raw source data)
- **Generated Artifacts**: All other files are pipeline outputs and can be regenerated
- **Manual Editing**: Avoid editing generated files directly; modify source data or pipeline logic instead
- **Version Control**: Treat `results/` as fixture data for development; regenerate for production deployments

### Grants Data Schema
- **Required Fields**: `title`, `grant_summary` (or `description`)
- **Optional Fields**: `funding_amount`, `start_date`, `end_date`, `funder`, `source`
- **Preprocessing**: Removes grants with equipment/travel in title, normalizes dates to `start_year`/`end_year`

### Keywords Schema
- **Fields**: `name` (str), `type` (General/Methodology/Application/Technology), `description` (str), `grant_id` (str)
- **Normalization**: NLTK stemming for deduplication, longest description wins for variant selection

### Categories Schema  
- **Fields**: `name`, `description`, `keywords` (list), `field_of_research` (enum), optional `source_categories` (for merged)
- **Hierarchy**: Keywords → Categories → Merged Categories
- **Field of Research**: 23 high-level divisions (e.g., BIOLOGICAL_SCIENCES, ENGINEERING, MATHEMATICAL_SCIENCES)
