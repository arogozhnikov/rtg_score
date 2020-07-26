import pandas as pd
from typing import List, Dict
from sklearn.metrics import roc_auc_score, pairwise_distances as sklearn_pairwise_distances
import numpy as np


def compute_pairwise_distances(
        n_samples,
        embeddings=None,
        pairwise_distances=None,
):
    if (pairwise_distances is None) + (embeddings is None) != 1:
        raise RuntimeError('Embeddings or pairwise_distances should be provided (only one, not both)')
    if pairwise_distances is None:
        embeddings = np.asarray_chkfinite(embeddings)
        assert len(embeddings) == n_samples
        assert np.ndim(embeddings) == 2
        return sklearn_pairwise_distances(embeddings)
    else:
        assert pairwise_distances.shape == (n_samples, n_samples)
        return pairwise_distances


def compute_r2g_score(
        metadata: pd.DataFrame,
        include_confounders: List[str],
        exclude_confounders: List[str],
        *,
        embeddings=None,
        pairwise_distances=None,
        minimal_n_samples=30,
):
    """

    :param metadata: DataFrame with confounds (may contain additional variables) of shape [n_samples, n_variables].
        Examples of variables: clone, donor, batch, plate, position on a plate
    :param include_confounders:
    :param exclude_confounders:
    :param embeddings: numerical description of
    :param pairwise_distances: distances between all the pairs
    :return: score or NaN 
        NaN if too few samples can provide ranking. E.g. if both include and exclude are the same variable
    """
    n_samples = len(metadata)
    pairwise_distances = compute_pairwise_distances(n_samples, embeddings, pairwise_distances)

    for column in [*include_confounders, *exclude_confounders]:
        if metadata[column].isna().sum() > 0:
            raise RuntimeError(f'Metadata has Nones in "{column}"')

    inc_cat = ''
    for category in include_confounders:
        inc_cat = inc_cat + metadata[category].map(str)

    exc_cat = ''
    for category in exclude_confounders:
        exc_cat = exc_cat + metadata[category].map(str)

    # recoding for simpler comparison
    _, inc_indices = np.unique(inc_cat, return_inverse=True)
    _, exc_indices = np.unique(exc_cat, return_inverse=True)

    aucs = []
    for sample in range(n_samples):
        mask = exc_indices != exc_indices[sample]
        target = inc_indices == inc_indices[sample]
        distances = pairwise_distances[sample]
        if len(set(target[mask])) == 2:
            aucs.append(roc_auc_score(target[mask], -distances[mask]))

    if len(aucs) < minimal_n_samples:
        return np.nan
    else:
        return np.mean(aucs)


def compute_contibution_matrix(
    metadata: pd.DataFrame,
    include_same_dict: Dict[str, List[str]],
    exclude_same_dict: Dict[str, List[str]],
    *,
    embeddings=None,
    pairwise_distances=None,
):
    n_samples = len(metadata)
    pairwise_distances = compute_pairwise_distances(n_samples, embeddings, pairwise_distances)

    results = {}
    for col_name, include_same in include_same_dict.items():
        for row_name, exclude_same in exclude_same_dict.items():
            results.setdefault(col_name, {})[row_name] = compute_r2g_score(
                metadata, include_confounders=include_same, exclude_confounders=exclude_same, pairwise_distances=pairwise_distances,
            )
    return pd.DataFrame(results)