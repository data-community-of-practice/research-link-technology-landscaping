from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import umap

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    total_items: int
    n_clusters: int
    target_batch_size: int
    clusters: Dict[str, List[Dict]]
    metadata: Optional[Dict[str, Any]] = None


class ClusterManager:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def cluster(
        self, 
        embeddings: np.ndarray, 
        data: Any, 
        batch_size: int
    ) -> ClusteringResult:
        n_clusters = self._calculate_clusters(len(data), batch_size)
        logger.info("Clustering %s items into %s clusters (target batch_size=%s)", 
                   len(data), n_clusters, batch_size)

        if len(data) <= batch_size:
            cluster_labels = np.zeros(len(data), dtype=int)
            metadata = {"method": "umap_ordering", "target_batch_size": batch_size}
        else:
            neighbors = self._umap_neighbors(len(data))
            cluster_labels = self._chunk_with_umap(embeddings, batch_size, neighbors)
            metadata = {
                "method": "umap_ordering",
                "target_batch_size": batch_size,
                "umap_neighbors": neighbors
            }

        clusters = self._organize_clusters(data, cluster_labels)

        return ClusteringResult(
            total_items=len(data),
            n_clusters=len(clusters),
            target_batch_size=batch_size,
            clusters=clusters,
            metadata=metadata,
        )

    def _calculate_clusters(self, data_size: int, batch_size: int) -> int:
        n_clusters = max(1, data_size // batch_size)
        if data_size % batch_size > 0:
            n_clusters += 1
        return n_clusters

    def _organize_clusters(self, data: Any, labels: np.ndarray) -> Dict[str, List[Dict]]:
        clusters: Dict[str, List[Dict]] = {}
        if hasattr(data, "to_dict"):
            data_records = data.to_dict(orient="records")
        else:
            data_records = data

        for index, (item, cluster_id) in enumerate(zip(data_records, labels)):
            cluster_key = f"cluster_{cluster_id}"
            clusters.setdefault(cluster_key, [])
            item_with_meta = item.copy()
            item_with_meta["embedding_index"] = index
            clusters[cluster_key].append(item_with_meta)

        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        if cluster_sizes:
            logger.info("Created %s clusters with sizes: %s", len(clusters), cluster_sizes)
            logger.info("Average cluster size: %.1f", np.mean(cluster_sizes))
            logger.info("Min/Max cluster sizes: %s/%s", min(cluster_sizes), max(cluster_sizes))
        return clusters

    def _chunk_with_umap(self, embeddings: np.ndarray, batch_size: int, neighbors: int) -> np.ndarray:
        total_items = embeddings.shape[0]
        reducer = umap.UMAP(
            n_components=1,
            random_state=self.random_state,
            n_neighbors=neighbors,
            metric="cosine",
        )
        reduced = reducer.fit_transform(embeddings)
        ordering = np.argsort(reduced[:, 0])

        labels = np.empty(total_items, dtype=int)
        cluster_index = 0
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            labels[ordering[start:end]] = cluster_index
            cluster_index += 1
        return labels

    def _umap_neighbors(self, total_items: int) -> int:
        if total_items <= 5:
            return max(2, total_items - 1)
        return min(15, total_items - 1)
