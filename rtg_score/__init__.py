"""
RTG score:
statistical tool to check contribution of confounding variables to biological models.
Works with any modality
"""
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, pairwise_distances as sklearn_pairwise_distances

__version__ = '0.1.0'


class InputErrorRTG(RuntimeError):
    """Generic error caused by incorrect input"""
    pass


def to_codes(array):
    """replace categories with unique integer codes"""
    return np.unique(array, return_inverse=True)[1]


def to_codes_str_series(array):
    """replace categories with unique string codes"""
    return pd.Series(to_codes(array)).map(str)


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
        assert np.ndim(embeddings) == 2, 'embeddings should be 2-dimensional metric [n_samples, n_features]'
        assert len(embeddings) == n_samples, 'number of embeddings should be the same as number of rows in metadata'
        if metric == 'hellinger':
            if np.min(embeddings) < 0:
                raise InputErrorRTG('Hellinger distance requires non-negative elements in embedding')
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
) -> float:
    """
    Compute (single) RTG score.

    :param metadata: DataFrame with confounds (may contain additional variables) of shape [n_samples, n_variables].
        Examples of variables: clone, donor, batch, plate, position on a plate
    :param include_confounders: list of confounders to estimate their joint contribution
        Example: pass ['batch', 'donor']
    :param exclude_confounders: list of confounders to exclude,
        Example: ['clone']
        Explanation: if ['batch', 'donor'] are included while ['clone'] is excluded, we measure how much samples
        with the same batch AND donor, but different clones are similar to each other.
    :param embeddings: numerical description of each sample. DataFrame or np.array of shape [n_sample, n_features].
        Order of embeddings should match order of rows in metadata
    :param metric: distance used to evaluate similarity. Possible choices are:
        - 'euclidean', relevant e.g. for delta Ct gene expression or for different embeddings
        - 'hellinger', relevant e.g. for cell type fractions in scRNA-seq
        - 'cosine', frequently more appropriate for DL embeddings
        - other distances from scipy and sklearn are supported
    :param pairwise_distances: alternatively distances between all the pairs can be readily provided.
        np.array of shape [n_samples, n_samples] (in this case, don't pass embeddings and metric)
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
        inc_cat = inc_cat + '_' + to_codes_str_series(metadata[category])

    exc_indices_collection = [
        to_codes(metadata[category]) for category in exclude_confounders
    ]
    # recoding for simpler comparison
    inc_indices = to_codes(inc_cat)

    aucs = []
    for sample in range(n_samples):
        mask = True
        for exc_indices in exc_indices_collection:
            mask = mask & (exc_indices != exc_indices[sample])
        target = inc_indices == inc_indices[sample]
        distances = pairwise_distances[sample]
        if len(set(target[mask])) == 2:
            aucs.append(roc_auc_score(target[mask], -distances[mask]))

    if len(aucs) < minimal_n_samples:
        return np.nan
    else:
        return float(np.mean(aucs))


def compute_RTG_contribution_matrix(
        metadata: pd.DataFrame,
        include_confounders_dict: Dict[str, List[str]],
        exclude_confounders_dict: Dict[str, List[str]],
        *,
        embeddings=None,
        metric='euclidean',
        pairwise_distances=None,
        minimal_n_samples=30,
):
    """
    Compute RTG scores for multiple combinations of included and excluded confounding variables.

    :param metadata: DataFrame with confounds (may contain additional variables) of shape [n_samples, n_variables].
        Examples of variables: clone, donor, batch, plate, position on a plate
    :param include_confounders_dict: dictionary with confounders and their combinations,
        Example: {
            'only donor': ['donor'],
            'donor&batch': ['donor', 'batch']
        }
    :param exclude_confounders_dict: dictionary with confounders and their combinations
        Example: {
            'donor': ['donor'],
            'clone': ['clone']
        }
        Score is computed for all pairs of included and excluded confounding variables.
    :param embeddings: numerical description of each sample. DataFrame or np.array of shape [n_sample, n_features].
        Order of embeddings should match order of rows in metadata
    :param metric: distance used to evaluate similarity. Possible choices are:
        - 'euclidean', relevant e.g. for delta Ct gene expression or for different embeddings
        - 'hellinger', relevant e.g. for cell type fractions in scRNA-seq
        - 'cosine', frequently more appropriate for DL embeddings
        - other distances from scipy and sklearn are supported
    :param pairwise_distances: alternatively distances between all the pairs can be readily provided.
        np.array of shape [n_samples, n_samples] (in this case, don't pass embeddings and metric)
    :param minimal_n_samples: number of samples that can provide ranking (otherwise function returns NaN).
        E.g. if both include and exclude are the same confounders, or if latter includes former, there are no elements
        that can provide ranking.
    :return: resulting scores are organized in pd.DataFrame (NaN elements mean not enough statistics)
    """
    n_samples = len(metadata)
    pairwise_distances = compute_pairwise_distances(n_samples, embeddings, pairwise_distances, metric=metric)

    results = {}
    for col_name, included in include_confounders_dict.items():
        for row_name, excluded in exclude_confounders_dict.items():
            results.setdefault(col_name, {})[row_name] = compute_RTG_score(
                metadata=metadata,
                include_confounders=included,
                exclude_confounders=excluded,
                pairwise_distances=pairwise_distances,
                minimal_n_samples=minimal_n_samples,
            )
    return pd.DataFrame(results)
