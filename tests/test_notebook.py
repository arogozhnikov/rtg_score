import nbformat

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


def run_notebook(notebook_filename):
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    notebook_folder = Path(notebook_filename).parent
    ep.preprocess(nb, {'metadata': {'path': notebook_folder}})

    # for reference: saving output notebook after running
    # with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
    #     nbformat.write(nb, f)


def test_notebook():
    notebook_path = Path(__file__).parent.parent / 'example' / 'Example_qPCR.ipynb'
    run_notebook(notebook_path)
