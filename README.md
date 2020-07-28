# Rank-To-Group (RTG) score evaluates contribution of confounders

<img src="https://github.com/System1Bio/rtg_score/blob/master/example/confounder_contribution.png?raw=true" width="500" />

Batch, cell line, donor, plate, reprogramming, protocol — these and other confounding factors influence cell cultures *in vitro*.

RTG score tracks contribution of different factors to variability by estimating how **R**ank maps **T**o **G**roup. 
Scoring relies on ranking by similarity, so there are no explicit or implicit assumptions of linearity.

RTG perfectly works with both well-interpretable data (gene expressions, cell types) 
and embeddings provided by deep learning.

## Usage 

`rtg_score` is python package. Installation:
```bash
pip install rtg_score
```

RTG score requires two DataFrames: one with confounds and one with embeddings (or other features, e.g. gene expressions)
```python
from rtg_score import compute_RTG_score
# following code corresponds to computing an element of the figure above
# (exclude same organoid_id and batch+donor)
score = compute_RTG_score(
    metadata=confounders_metadata,
    include_confounders=['batch', 'donor'],
    exclude_confounders=['organoid_id'],
    embeddings=qpcr_delta_ct, 
)
```

Use `compute_RTG_contribution_matrix` to compute multiple RTG scores in a bulk . <br />
Example + code for plotting are available in [`example`](https://github.com/System1Bio/rtg_score/blob/master/example/Example_qPCR.ipynb) subfolder.

## Parameters

- `metadata` - DataFrame with confounding variables
    - sample_id, batch, donor, clone, plate, etc. 
- `embeddings` - numerical description of samples (DataFrame or 2d array). 
    - Examples: qPCR delta Cts, deep learning embeddings, cell types fractions
- `metric` - how to define similarity?
    - use `euclidean` for qPCR and various embeddings
      and `hellinger` for cell type distributions
- included and excluded confounders in example:
    - including ['donor', 'batch'] and excluding ['clone', 'plate'] will estimate 
      how similar are samples with same donor and same batch, 
      while omitting pairs which have same clone or grown on the same plate
    - most use-cases are simple, like include batch effect while exclude plate, 
      but framework is very flexible 

      