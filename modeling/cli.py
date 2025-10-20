from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import typer

import config
import utils
from pipelines import categorise as categorise_pipeline
from pipelines import extract as extract_pipeline
from pipelines import merge as merge_pipeline
import numpy as np
from pipelines.clustering import ClusteringResult, ClusterManager
from pipelines.embedding import DEFAULT_EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer


app = typer.Typer(help="Unified interface for keyword extraction and categorisation workflows")


def _load_dataset(data_path: Path):
    data = utils.load_jsonl_file(data_path, as_dataframe=True)
    return data_path, data


def _ensure_embeddings(
    texts,
    output_path: Path,
    model: str,
    batch_size: int,
    force: bool,
    embedding_type: str = "transformer",
    data = None,
) -> Tuple[Path, bool]:
    embeddings_path = output_path
    if embeddings_path.exists() and not force:
        typer.echo(f"Embeddings already exist at {embeddings_path}; use --force to overwrite")
        return embeddings_path, False

    if embedding_type == "tfidf":
        # Use TF-IDF for embeddings
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+(?:\s+\w+)*\b',  # Support multi-word keywords
            lowercase=False,
            min_df=1
        )
        array = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        typer.echo(f"Generated TF-IDF embeddings: {array.shape[0]} items × {array.shape[1]} features")
    else:
        # Use transformer-based embeddings (default)
        encoder = SentenceTransformer(model)
        array = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        typer.echo(f"Generated transformer embeddings: {array.shape[0]} items × {array.shape[1]} dimensions")
    
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, array)
    typer.echo(f"Saved embeddings to {embeddings_path}")
    return embeddings_path, True


@app.command()
def extract(
    skip_finished: bool = typer.Option(True, help="Skip grants that already have extracted keywords"),
    concurrency: int = typer.Option(config.CONCURRENCY, help="Maximum concurrent requests"),
) -> None:
    processed = extract_pipeline(filter_finished=skip_finished, concurrency=concurrency)
    count = len(processed)
    typer.echo(f"Processed {count} grant{'s' if count != 1 else ''}")


@app.command()
def categorise(
    clusters_path: Path = typer.Argument(..., help="Keyword clusters json file"),
    output_dir: Path = typer.Argument(..., help="Directory to write categorisation outputs"),
    concurrency: int = typer.Option(config.CONCURRENCY, help="Maximum concurrent requests"),
    limit: Optional[int] = typer.Option(None, help="Limit number of records to process"),
) -> None:
    processed = categorise_pipeline(
        clusters_path,
        output_dir,
        concurrency=concurrency,
        limit=limit,
    )
    count = len(processed)
    typer.echo(f"Processed {count} cluster{'s' if count != 1 else ''}")


@app.command()
def merge(
    clusters_path: Path = typer.Argument(..., help="Category clusters json file"),
    output_dir: Path = typer.Argument(..., help="Directory to write merged outputs"),
    concurrency: int = typer.Option(config.CONCURRENCY, help="Maximum concurrent requests"),
    limit: Optional[int] = typer.Option(None, help="Limit number of records to process"),
) -> None:
    processed = merge_pipeline(
        clusters_path,
        output_dir,
        concurrency=concurrency,
        limit=limit,
    )
    count = len(processed)
    typer.echo(f"Processed {count} cluster{'s' if count != 1 else ''}")


@app.command()
def embed(
    input_path: Path = typer.Argument(..., help="Path to source jsonl"),
    embeddings_path: Path = typer.Argument(..., help="Path to write embeddings npy file"),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="Local Hugging Face model id"),
    batch_size: int = typer.Option(64, help="Number of texts per embedding batch"),
    embedding_type: str = typer.Option("transformer", help="Embedding type: 'transformer' or 'tfidf'"),
    force: bool = typer.Option(False, help="Regenerate embeddings even if cache exists"),
) -> None:
    data_path, data = _load_dataset(input_path)
    texts = data.apply(config.Keywords.template, axis=1).tolist()

    if not texts:
        typer.echo("No records to embed")
        return

    _ensure_embeddings(texts, embeddings_path, model, batch_size, force, embedding_type, data)


@app.command("embed-cluster")
def embed_and_cluster(
    input_path: Path = typer.Argument(..., help="Path to source jsonl"),
    embeddings_path: Path = typer.Argument(..., help="Path to write embeddings npy file"),
    clusters_path: Path = typer.Argument(..., help="Path to write clusters json file"),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="Local Hugging Face model id"),
    batch_size: int = typer.Option(64, help="Number of texts per embedding batch"),
    cluster_batch_size: int = typer.Option(50, help="Target cluster size"),
    embedding_type: str = typer.Option("transformer", help="Embedding type: 'transformer' or 'tfidf'"),
    force: bool = typer.Option(False, help="Regenerate embeddings even if cache exists"),
) -> None:
    data_path, data = _load_dataset(input_path)
    texts = data.apply(config.Keywords.template, axis=1).tolist()

    if not texts:
        typer.echo("No records to embed")
        return

    embeddings_path, _ = _ensure_embeddings(texts, embeddings_path, model, batch_size, force, embedding_type, data)

    result = _cluster_and_save(data_path, embeddings_path, clusters_path, cluster_batch_size)
    typer.echo(
        f"Clustered {result.total_items} items into {result.n_clusters} groups and saved to {clusters_path}"
    )


def _cluster_and_save(
    data_path: Path,
    embeddings_path: Path,
    clusters_path: Path,
    batch_size: int,
) -> ClusteringResult:
    data = utils.load_jsonl_file(data_path, as_dataframe=True)
    embeddings = np.load(embeddings_path)

    if len(data) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch between data items ({len(data)}) and embeddings ({embeddings.shape[0]})"
        )

    manager = ClusterManager()
    result = manager.cluster(embeddings, data, batch_size)

    clusters_path.parent.mkdir(parents=True, exist_ok=True)
    with clusters_path.open("w") as handle:
        json.dump(
            {
                "total_items": result.total_items,
                "n_clusters": result.n_clusters,
                "target_batch_size": result.target_batch_size,
                "clusters": result.clusters,
            },
            handle,
            indent=2,
            default=str,
        )

    return result

@app.command("cluster")
def clustering(
    data_path: Path = typer.Argument(..., help="Input dataset jsonl"),
    embeddings_path: Path = typer.Argument(..., help="Embeddings npy file"),
    clusters_path: Path = typer.Argument(..., help="Clusters json path"),
    batch_size: int = typer.Option(50, help="Target cluster size"),
) -> None:
    result = _cluster_and_save(data_path, embeddings_path, clusters_path, batch_size)
    typer.echo(
        f"Clustered {result.total_items} items into {result.n_clusters} groups and saved to {clusters_path}"
    )
if __name__ == "__main__":
    app()
