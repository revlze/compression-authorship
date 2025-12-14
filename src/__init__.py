# src package
# Makes src a proper Python package for relative imports

from .compressors import (
    gzip_compressor,
    bzip2_compressor,
    lzma_compressor,
    zstd_compressor,
    brotli_compressor,
)
from .data_pipeline import process, clean_text, preprocess_text, build
from .theory import phi, phi_authors, build_table, chi_square, significance_level, get_accuracy
from .preprocessing import run_data_pipeline, split_into_slices
from .neural_method import (
    llm_compress,
    paq_compressor,
    ts_zip_compressor,
    clear_model_cache,
)

__all__ = [
    # Compressors
    'gzip_compressor',
    'bzip2_compressor',
    'lzma_compressor',
    'zstd_compressor',
    'brotli_compressor',
    # Neural compressors
    'llm_compress',
    'paq_compressor',
    'ts_zip_compressor',
    'clear_model_cache',
    # Data pipeline
    'process',
    'clean_text',
    'preprocess_text',
    'build',
    # Theory
    'phi',
    'phi_authors',
    'build_table',
    'chi_square',
    'significance_level',
    'get_accuracy',
    # Preprocessing
    'run_data_pipeline',
    'split_into_slices',
]
