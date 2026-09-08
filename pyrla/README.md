# PyRLA - Python Interface for Research Link Australia API

PyRLA is a high-performance Python package that provides an async-only interface to interact with the Research Link Australia (RLA) API at `https://researchlink.ardc.edu.au/v3/api-docs/public`.

## Usage

Retrieve all grants and save as JSON Lines:

```
uv run pyrla grants search "" --all --jsonl-file grants.jsonl
```
