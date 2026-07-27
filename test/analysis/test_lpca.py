from pathlib import Path

import pandas as pd

import spras.config.config as config
from spras.analysis.lpca import run_lpca
from spras.analysis.ml import summarize_networks

config.init_from_file("config/config.yaml")

TEST_DIR = Path('test/analysis/')
OUT_DIR = TEST_DIR / 'output'

INPUT_FILES = [
    'test/evaluate/input/data-test-params-123/pathway.txt',
    'test/evaluate/input/data-test-params-456/pathway.txt',
    'test/evaluate/input/data-test-params-789/pathway.txt',
]

class TestLpca:
    """
    Run Logistic PCA (LPCA) analysis tests
    """
    @classmethod
    def setup_class(cls):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_lpca_output_exists(self):
        """Test that LPCA produces an output scores file"""
        out_path = OUT_DIR / 'lpca-scores.csv'
        matrix_path = OUT_DIR / 'lpca-binary-matrix.csv'
        out_path.unlink(missing_ok=True)

        summary_df = summarize_networks(INPUT_FILES)
        run_lpca(
            dataframe=summary_df,
            output_scores=str(out_path),
            output_matrix=str(matrix_path),
            k=2,
            m=4,
            cv=False,
        )

        assert out_path.exists(), "LPCA scores file was not created"

    def test_lpca_output_shape(self):
        """Test that LPCA scores have the correct shape (runs x k)"""
        out_path = OUT_DIR / 'lpca-scores-shape.csv'
        matrix_path = OUT_DIR / 'lpca-binary-matrix-shape.csv'
        out_path.unlink(missing_ok=True)

        summary_df = summarize_networks(INPUT_FILES)
        run_lpca(
            dataframe=summary_df,
            output_scores=str(out_path),
            output_matrix=str(matrix_path),
            k=2,
            m=4,
            cv=False,
        )

        scores = pd.read_csv(out_path, index_col=0)
        assert scores.shape[1] == 2, f"Expected 2 PC columns, got {scores.shape[1]}"
        assert scores.shape[0] > 0, "Scores file is empty"
