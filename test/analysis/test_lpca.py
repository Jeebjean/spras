from pathlib import Path

import docker
import pandas as pd
import pytest

import spras.config.config as config
from spras.analysis.lpca import run_lpca

config.init_from_file("config/config.yaml")

TEST_DIR = Path('test/analysis/')
OUT_DIR = TEST_DIR / 'output'

INPUT_FILES = [
    'test/evaluate/input/data-test-params-123/pathway.txt',
    'test/evaluate/input/data-test-params-456/pathway.txt',
    'test/evaluate/input/data-test-params-789/pathway.txt',
]

def lpca_image_available():
    """Check if the LPCA Docker image is available locally or on Docker Hub"""
    try:
        client = docker.from_env()
        client.images.get('reedcompbio/lpca:v1')
        return True
    except docker.errors.ImageNotFound:
        return False
    except Exception:
        return False

skip_if_no_lpca_image = pytest.mark.skipif(
    not lpca_image_available(),
    reason='reedcompbio/lpca:v1 Docker image not available'
)

class TestLpca:
    """
    Run Logistic PCA (LPCA) analysis tests
    """
    @classmethod
    def setup_class(cls):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    @skip_if_no_lpca_image
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

    @skip_if_no_lpca_image
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
        assert scores.shape[1] == 2, f"Expected 2 PC columns, got {scores.shape[1]}"
        assert scores.shape[0] > 0, "Scores file is empty"

    @skip_if_no_lpca_image
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
        assert scores.shape[1] == 2, f"Expected 2 PC columns, got {scores.shape[1]}"
        assert scores.shape[0] == len(INPUT_FILES), \
            f"Expected {len(INPUT_FILES)} rows, got {scores.shape[0]}"
