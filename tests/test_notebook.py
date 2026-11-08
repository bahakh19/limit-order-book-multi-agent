import json
import os
import sys

import nbformat
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'limit_order_book.ipynb')
sys.path.insert(0, os.path.join(ROOT, 'tools'))


def test_notebook_cells_are_in_sync_with_src():
    """The notebook embeds the modules of src/: rebuild the cells and compare them with the committed notebook."""
    import build_notebook
    nb = json.load(open(NOTEBOOK, encoding='utf-8'))
    code = {c['id']: ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'}
    import hashlib
    for tag, module in (('engine', 'orderbook.py'), ('demo', 'demo.py'), ('agents', 'agents.py'),
                        ('analysis', 'analysis.py'), ('simulation', 'simulation.py'), ('ecology', 'ecology.py'),
                        ('experiments', 'experiments.py')):
        cell_id = hashlib.sha1(tag.encode()).hexdigest()[:8]
        assert code[cell_id].rstrip('\n') == build_notebook.module_source(module).rstrip('\n'), module


def test_committed_notebook_is_valid_and_has_outputs_without_errors():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == 'code':
            assert not any(o.output_type == 'error' for o in c.outputs)


@pytest.mark.slow
def test_notebook_executes_end_to_end(tmp_path):
    from nbclient import NotebookClient
    nb = nbformat.read(NOTEBOOK, as_version=4)
    work = tmp_path / 'notebooks'
    work.mkdir()
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(work)}}).execute()
    text = ''.join(o.get('text', '') for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'stream')
    assert 'simulations: 100' in text
    assert 'Lag-1 impact R(1)' in text
    assert 'exponent (Q >= 200)' in text
    assert (tmp_path / 'results').is_dir()
