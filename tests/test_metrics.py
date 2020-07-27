from itertools import combinations

import numpy as np
import pandas as pd

from rtg_score import compute_RTG_contribution_matrix


def test_random_contributions(n_samples=1000, n_metadata_columns=3, n_categories=10, n_features=10):
    rng = np.random.RandomState(42)
    metadata = pd.DataFrame(
        data=rng.randint(0, n_categories, size=[n_samples, n_metadata_columns]),
        columns=[f'Conf{i}' for i in range(1, 1 + n_metadata_columns)]
    )

    # all elements are positive, to allow hellinger's distance computation
    embeddings = rng.uniform(0, 1, size=[n_samples, n_features])

    include_confounders_dict = {
        'Conf1': ['Conf1'],
        'Conf2': ['Conf2'],
        'Conf12': ['Conf1', 'Conf2'],
    }
    exclude_confounders_dict = {
        'Conf2': ['Conf2'],
        'Conf3': ['Conf3'],
        'Conf13': ['Conf3', 'Conf1'],
    }

    # check that all
    dataframes = {
        metric: compute_RTG_contribution_matrix(
            metadata,
            include_confounders_dict=include_confounders_dict,
            exclude_confounders_dict=exclude_confounders_dict,
            metric=metric,
            embeddings=embeddings,
        )
        for metric in ['euclidean', 'hellinger', 'cosine']
    }

    # verify all scores are close to 0.5
    for metric, metric_df in dataframes.items():
        assert np.max(abs(metric_df - 0.5).stack().dropna()) < 0.05

    for metric1, metric2 in combinations(dataframes.keys(), r=2):
        assert not np.allclose(dataframes[metric1], dataframes[metric2], equal_nan=True)
