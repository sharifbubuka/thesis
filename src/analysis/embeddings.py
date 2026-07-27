from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.contribution.contracts import ContributionResult


def pca_embedding_frame(
    results: list[ContributionResult],
    *,
    components: int = 2,
) -> pd.DataFrame:
    usable = [result for result in results if result.cls_embedding is not None]
    if len(usable) < components:
        raise ValueError(f"At least {components} samples with CLS embeddings are required.")
    matrix = np.stack([result.cls_embedding for result in usable])
    reduced = PCA(n_components=components).fit_transform(matrix)
    records = []
    for result, coordinates in zip(usable, reduced, strict=True):
        record = {
            "sample_id": result.sample_id,
            "benchmark": result.benchmark,
            "image_share": result.image_share,
            "text_share": result.text_share,
        }
        record.update({f"pc{index + 1}": float(value) for index, value in enumerate(coordinates)})
        records.append(record)
    return pd.DataFrame(records)
