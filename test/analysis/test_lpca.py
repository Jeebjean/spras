from pathlib import Path

import pandas as pd

import spras.config.config as config
from spras.analysis.lpca import run_lpca

config.init_from_file("config/config.yaml")

TEST_DIR = Path('test/analysis/')
OUT_DIR = TEST_DIR / 'output'

# Reuse pathway files from the evaluate test directory
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
        out_path.unlink(missing_ok=True)

        run_lpca(
            file_paths=INPUT_FILES,
            output_scores=str(out_path),
            k=2,
            m=4,
            cv=False,
            transpose=False
        )

        assert out_path.exists(), "LPCA scores file was not created"

    def test_lpca_output_shape(self):
        """Test that LPCA scores have the correct shape (edges x k)"""
        out_path = OUT_DIR / 'lpca-scores-shape.csv'
        out_path.unlink(missing_ok=True)

        run_lpca(
            file_paths=INPUT_FILES,
            output_scores=str(out_path),
            k=2,
            m=4,
            cv=False,
            transpose=False
        )

        scores = pd.read_csv(out_path, index_col=0)
        # k=2 so should have 2 columns
        assert scores.shape[1] == 2, f"Expected 2 PC columns, got {scores.shape[1]}"
        # Should have at least 1 row (edge)
        assert scores.shape[0] > 0, "Scores file is empty"

    def test_lpca_transposed_shape(self):
        """Test that transposed LPCA scores have the correct shape (runs x k)"""
        out_path = OUT_DIR / 'lpca-scores-transposed.csv'
        out_path.unlink(missing_ok=True)

        run_lpca(
            file_paths=INPUT_FILES,
            output_scores=str(out_path),
            k=2,
            m=4,
            cv=False,
            transpose=True
        )

        scores = pd.read_csv(out_path, index_col=0)
        # k=2 so should have 2 columns
        assert scores.shape[1] == 2, f"Expected 2 PC columns, got {scores.shape[1]}"
        # Should have 3 rows (one per pathway run)
        assert scores.shape[0] == len(INPUT_FILES), \
            f"Expected {len(INPUT_FILES)} rows, got {scores.shape[0]}"
