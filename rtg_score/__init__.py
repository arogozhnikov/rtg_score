from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, pairwise_distances as sklearn_pairwise_distances

__version__ = '0.1.0'


class InputErrorRTG(RuntimeError):
    """Generic error caused by incorrect input"""
    pass


def compute_pairwise_distances(
        n_samples,
        embeddings=None,
        pairwise_distances=None,
        metric='euclidean',
):
    if (pairwise_distances is None) + (embeddings is None) != 1:
        raise RuntimeError('Embeddings or pairwise_distances should be provided (only one, not both)')
    if pairwise_distances is None:
        embeddings = np.asarray(embeddings)
        assert np.ndim(embeddings) == 2, 'embeddings should be 2-dimensional matric [n_samples, n_features]'
        assert len(embeddings) == n_samples, 'number of embeddings should be the same as number of rows in metadata'
        if metric == 'hellinger':
            if embeddings.min() < 0:
                raise InputErrorRTG('hellinger distance requires non-negative elements in embedding')
            return sklearn_pairwise_distances(np.sqrt(embeddings), metric='euclidean')
        return sklearn_pairwise_distances(embeddings, metric=metric)
    else:
        if metric != 'euclidean':
            raise RuntimeWarning(f'Passed metric ({metric}) not used as distances are passed')
        assert pairwise_distances.shape == (n_samples, n_samples), 'wrong shape of distances passed'
        return pairwise_distances


def compute_RTG_score(
        metadata: pd.DataFrame,
        include_confounders: List[str],
        exclude_confounders: List[str],
        *,
        embeddings=None,
        metric='euclidean',
        pairwise_distances=None,
        minimal_n_samples=30,
):
    """
    Compute a single number to

    :param metadata: DataFrame with confounds (may contain additional variables) of shape [n_samples, n_variables].
        Examples of variables: clone, donor, batch, plate, position on a plate
    :param include_confounders: list of confounders to estimate their joint contribution
        Example: pass ['batch', 'donor'] to evaluate a fraction of variability
    :param exclude_confounders: list of
    :param embeddings: numerical description of each sample
    :param metric: distance used to evaluate similarity. Possible choices are:
        - 'euclidean', relevant e.g. for delta Ct gene expression or for different embeddings
        - 'hellinger', relevant e.g. for cell type fractions in scRNA-seq
        - 'cosine', more appropriate for some embeddings
        - other distances from scipy and sklearn are supported
    :param pairwise_distances: alternatively distances between all the pairs can be readily provided
        (in this case, don't pass embeddings)
    :param minimal_n_samples: number of samples that can provide ranking (otherwise function returns NaN).
        E.g. if both include and exclude are the same confounders, or if latter includes former, there are no elements
        that can provide ranking.
    :return: score or NaN
        NaN (Not-a-Number) if too few samples can provide ranking.
    """
    n_samples = len(metadata)
    pairwise_distances = compute_pairwise_distances(n_samples, embeddings, pairwise_distances, metric=metric)

    for column in [*include_confounders, *exclude_confounders]:
        if metadata[column].isna().sum() > 0:
            raise RuntimeError(f'Metadata has Nones in "{column}"')

    if len(include_confounders) == 0 or len(exclude_confounders) == 0:
        raise InputErrorRTG(f'include_confounders and exclude_confounders should be non-empty')

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


def compute_RTG_contribution_matrix(
        metadata: pd.DataFrame,
        include_same_dict: Dict[str, List[str]],
        exclude_same_dict: Dict[str, List[str]],
        *,
        embeddings=None,
        metric='euclidean',
        pairwise_distances=None,
):
    n_samples = len(metadata)
    pairwise_distances = compute_pairwise_distances(n_samples, embeddings, pairwise_distances, metric=metric)

    results = {}
    for col_name, include_same in include_same_dict.items():
        for row_name, exclude_same in exclude_same_dict.items():
            results.setdefault(col_name, {})[row_name] = compute_RTG_score(
                metadata=metadata,
                include_confounders=include_same,
                exclude_confounders=exclude_same,
                pairwise_distances=pairwise_distances,
            )
    return pd.DataFrame(results)
