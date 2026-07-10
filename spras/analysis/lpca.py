# Logistic PCA analysis for SPRAS
# Runs LPCA on the binary edge x algorithm matrix produced by summarize_networks.
# Configured through the analysis.lpca block (k, m, cv, transpose).

from os import PathLike
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from spras.analysis.ml import summarize_networks
from spras.config.container_schema import ProcessedContainerSettings
from spras.containers import prepare_volume, run_container_and_log

# Published as docker.io/reedcompbio/lpca:v1. Only the suffix is given here;
# the registry prefix is resolved from the container settings.
LPCA_CONTAINER_SUFFIX = 'lpca:v1'
LPCA_WORK_DIR = '/app'


def run_lpca(
    file_paths: Iterable[Union[str, PathLike]],
    output_scores: str,
    k: int = 2,
    m: float = 6,
    cv: bool = False,
    transpose: bool = False,
    container_settings=None
) -> None:
    """
    Runs Logistic PCA on the binary edge x algorithm matrix built from SPRAS
    algorithm output files.

    @param file_paths: pathway.txt file paths from SPRAS algorithm outputs
    @param output_scores: path to write the LPCA PC scores CSV
    @param k: number of principal components (default 2)
    @param m: fixed logisticPCA tuning parameter, used when cv is False
    @param cv: if True, determine m by cross-validation; if False, use the
        fixed m directly (default False)
    @param transpose: if True, run LPCA on the transposed (runs x edges) matrix
        instead of the default (edges x runs)
    @param container_settings: configure the container runtime (Docker or Singularity)
    """
    if not container_settings:
        container_settings = ProcessedContainerSettings()

    # Step 1: build the binary edge x algorithm matrix
    print('LPCA: Building binary edge x algorithm matrix...')
    matrix = summarize_networks(file_paths)

    # Optionally transpose so the runs become the observations rather than the edges
    if transpose:
        matrix = matrix.T
    print(f'LPCA: Matrix shape: {matrix.shape}')

    # Step 2: write the matrix next to the outputs, namespaced by algorithm
    output_dir = Path(output_scores).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_name = Path(output_scores).name.replace('-lpca-scores.csv', '')
    matrix_path = str(output_dir / f'{algo_name}-lpca_binary_matrix.csv')
    matrix.to_csv(matrix_path)
    print(f'LPCA: Binary matrix saved to {matrix_path}')

    # Step 3: mount the matrix and the scores output
    volumes = []
    bind_path, mapped_matrix = prepare_volume(matrix_path, LPCA_WORK_DIR, container_settings)
    volumes.append(bind_path)
    bind_path, mapped_scores = prepare_volume(output_scores, LPCA_WORK_DIR, container_settings)
    volumes.append(bind_path)

    # Step 4: choose m, optionally via cross-validation
    if cv:
        cv_output_path = str(output_dir / f'{algo_name}-lpca_cv_result.csv')
        bind_path, mapped_cv_output = prepare_volume(cv_output_path, LPCA_WORK_DIR, container_settings)
        volumes.append(bind_path)

        print(f'LPCA: Running cross-validation with k={k}...')
        command_cv = ['Rscript', '/app/run_cv.R', mapped_matrix, mapped_cv_output, str(k)]
        run_container_and_log('LPCA-CV', LPCA_CONTAINER_SUFFIX, command_cv, volumes,
                              LPCA_WORK_DIR, None, container_settings)

        m_used = pd.read_csv(cv_output_path)['best_m'][0]
        print(f'LPCA: Best m found by CV: {m_used}')
    else:
        m_used = m
        print(f'LPCA: Using fixed m={m_used}')

    # Step 5: run LPCA with the chosen k and m
    print(f'LPCA: Running LPCA with k={k}, m={m_used}...')
    command_lpca = ['Rscript', '/app/run_lpca.R', mapped_matrix, mapped_scores, str(k), str(m_used)]
    run_container_and_log('LPCA', LPCA_CONTAINER_SUFFIX, command_lpca, volumes,
                          LPCA_WORK_DIR, None, container_settings)

    print(f'LPCA: Done! Scores saved to {output_scores}')
