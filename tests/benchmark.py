from rtg_score import compute_RTG_score
import numpy as np
import pandas as pd
from time import time


def generate_dataset(n_samples=10_000, n_dimensions=32, n_groups_metadata=10):
    embeddings = np.random.randn(n_samples, n_dimensions)
    metadata = pd.DataFrame({
        f'column{i}': np.random.randint(0, n_groups_metadata, n_samples)
        for i in range(4)
    })

    return embeddings, metadata


embs, meta = generate_dataset()

for fast in [False, True, 'mann']:
    start = time()
    score = compute_RTG_score(
        metadata=meta,
        embeddings=embs,
        include_confounders=['column0'],
        exclude_confounders=['column1'],
    )
    print(f'fast: {fast}')
    print(time() - start)
