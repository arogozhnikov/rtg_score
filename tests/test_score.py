from rtg_score import compute_RTG_contribution_matrix, compute_RTG_score
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).parent.parent


def prepare_qpcr():
    filename = repo_root / 'example/expression_data.csv'
    expression_with_metadata = pd.read_csv(filename, sep='\t')
    expression_with_metadata.head()
    genes_expression = expression_with_metadata[
        ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Gene6', 'Gene7', 'Gene8']].copy()
    # compute delta Ct
    genes_delta_ct = genes_expression.sub(genes_expression['Gene1'], axis=0).drop(columns='Gene1')
    # normalize contribution of each gene
    genes_delta_ct_normalized = genes_delta_ct.div(genes_delta_ct.std())
    return expression_with_metadata, genes_delta_ct_normalized


def test_against_reference():
    expression_with_metadata, genes_delta_ct_normalized = prepare_qpcr()

    for fast in [True, False, 'auto']:
        contribution_matrix = compute_RTG_contribution_matrix(
            expression_with_metadata,
            include_confounders_dict={
                'batch': ['batch'],
                'donor': ['donor'],
                'clone': ['clone'],
                'batch+\ndonor': ['batch', 'donor'],
                'batch+\nclone': ['batch', 'clone'],
            },
            exclude_confounders_dict={
                'exclude same\norganoid': ['organoid_id'],
                'exclude same\nclone': ['clone'],
                'exclude same\ndonor': ['donor'],
                'exclude same\nbatch': ['batch'],
            },
            embeddings=genes_delta_ct_normalized,
            fast=fast,
        )

        reference_result = {
            'batch': {'exclude same\norganoid': 0.5995943128819183,
                      'exclude same\nclone': 0.581381293631887,
                      'exclude same\ndonor': 0.5717474846071052,
                      'exclude same\nbatch': np.nan},
            'donor': {'exclude same\norganoid': 0.6627428876496726,
                      'exclude same\nclone': 0.602667260670832,
                      'exclude same\ndonor': np.nan,
                      'exclude same\nbatch': 0.623785995680053},
            'clone': {'exclude same\norganoid': 0.7234854098248196,
                      'exclude same\nclone': np.nan,
                      'exclude same\ndonor': np.nan,
                      'exclude same\nbatch': 0.6606615343718841},
            'batch+\ndonor': {'exclude same\norganoid': 0.8121230673103979,
                              'exclude same\nclone': 0.6930298371813395,
                              'exclude same\ndonor': np.nan,
                              'exclude same\nbatch': np.nan},
            'batch+\nclone': {'exclude same\norganoid': 0.9372202606462032,
                              'exclude same\nclone': np.nan,
                              'exclude same\ndonor': np.nan,
                              'exclude same\nbatch': np.nan}
        }

        assert np.allclose(pd.DataFrame(reference_result), contribution_matrix, equal_nan=True)


def test_internal_agreement():
    expression_with_metadata, genes_delta_ct_normalized = prepare_qpcr()

    scores = compute_RTG_contribution_matrix(
        expression_with_metadata,
        include_confounders_dict={
            'batch': ['batch'],
            'donor': ['donor'],
            'clone': ['clone'],
            'clone+donor': ['clone', 'donor']
        },
        exclude_confounders_dict={
            'exclude donor': ['donor'],
            'exclude clone+donor': ['clone', 'donor'],
            'exclude batch': ['batch'],
            'exclude batch+organoid_id': ['batch', 'organoid_id'],
        },
        embeddings=genes_delta_ct_normalized,
    )

    # exclusion works as union
    assert scores.loc['exclude donor'].equals(scores.loc['exclude clone+donor'])
    assert scores.loc['exclude batch'].equals(scores.loc['exclude batch+organoid_id'])
    # inclusion works as intersection
    assert scores['clone'].equals(scores['clone+donor'])
