# Research Link Technology Landscaping

This repository contains tools and pipelines for analyzing research grant data from Research Link Australia (RLA) to understand technology and research landscapes.

## Project Structure

The project consists of two main components:

### 1. PyRLA - Python Interface for RLA API

PyRLA is a high-performance Python package that provides an async interface to interact with the Research Link Australia API. It enables efficient retrieval of researcher, grant, organisation, and publication data.

**Key Features:**
- Async-first design for concurrent API requests
- Clean, object-oriented interface
- Command-line interface for quick queries
- Optimized for large-scale data retrieval

📖 **[See full PyRLA documentation →](pyrla/README.md)**

### 2. Modeling - Research Landscape Analysis Pipeline

The modeling component provides a comprehensive pipeline for extracting, categorizing, and analyzing research grants data. It includes keyword extraction, clustering, categorization workflows, and an interactive web dashboard for visualization.

**Key Features:**
- LLM-powered keyword extraction from grant abstracts
- Embedding-based clustering and categorization
- Interactive Streamlit dashboard for data exploration
- Comparative analysis tools for organizations and research areas

📖 **[See full Modeling documentation →](modeling/README.md)**

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/data-community-of-practice/research-link-technology-landscaping.git
cd research-link-technology-landscaping
```

2. Install PyRLA:
```bash
cd pyrla
uv sync
```

3. Install Modeling pipeline:
```bash
cd ../modeling
uv sync
```

### Usage Examples

**Fetch data with PyRLA:**
```bash
cd pyrla
uv run pyrla search-researchers "machine learning"
```

**Run the analysis dashboard:**
```bash
cd modeling
uv run streamlit run web/entrypoint.py
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
