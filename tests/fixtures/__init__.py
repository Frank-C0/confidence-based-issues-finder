"""
Simplified test fixtures for Datalab testing.

Provides simple dataset generators with real data from HuggingFace:
- SimpleDatasetGenerator: Creates small and medium datasets for all task types

All datasets include labels, pred_probs, features, and knn_graph.
Results are cached with joblib to avoid recomputation.
"""

from .simple_datasets import SimpleDatasetGenerator

__all__ = [
    "SimpleDatasetGenerator",
]
