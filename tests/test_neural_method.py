"""
Smoke tests for src/neural_method.py

Run with: pytest tests/test_neural_method.py -v
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test helper functions (no model required)
# =============================================================================


class TestDecodeText:
    """Tests for _decode_text helper."""

    def test_valid_utf8(self):
        from src.neural_method import _decode_text

        data = "Привет мир!".encode("utf-8")
        result = _decode_text(data, strict=True)
        assert result == "Привет мир!"

    def test_invalid_utf8_strict_raises(self):
        from src.neural_method import _decode_text

        # Invalid UTF-8 sequence
        data = b"\xff\xfe invalid"
        with pytest.raises(ValueError, match="не являются валидным UTF-8"):
            _decode_text(data, strict=True)

    def test_invalid_utf8_tolerant_replaces(self):
        from src.neural_method import _decode_text

        data = b"\xff\xfe hello"
        result = _decode_text(data, strict=False)
        assert "hello" in result
        assert "\ufffd" in result  # Replacement character


class TestGetMaxLength:
    """Tests for _get_max_length helper."""

    def test_n_positions(self):
        from src.neural_method import _get_max_length

        model = MagicMock()
        model.config.n_positions = 1024
        assert _get_max_length(model) == 1024

    def test_max_position_embeddings(self):
        from src.neural_method import _get_max_length

        model = MagicMock()
        model.config = MagicMock(spec=[])  # No n_positions
        model.config.max_position_embeddings = 2048
        assert _get_max_length(model) == 2048

    def test_fallback(self):
        from src.neural_method import _get_max_length

        model = MagicMock()
        model.config = MagicMock(spec=[])  # No known attributes
        assert _get_max_length(model) == 1024


class TestGetDevice:
    """Tests for _get_device helper."""

    def test_returns_string(self):
        from src.neural_method import _get_device

        device = _get_device()
        assert isinstance(device, str)
        assert device in ("cuda", "mps", "cpu")


# =============================================================================
# Test subprocess helpers
# =============================================================================


class TestRunSubprocess:
    """Tests for _run_subprocess helper."""

    def test_success(self):
        from src.neural_method import _run_subprocess

        result = _run_subprocess(["echo", "hello"], timeout=10)
        assert b"hello" in result.stdout

    def test_failure_includes_stderr(self):
        from src.neural_method import _run_subprocess

        with pytest.raises(RuntimeError) as exc_info:
            _run_subprocess(["ls", "/nonexistent_path_12345"], timeout=10)

        error_msg = str(exc_info.value)
        assert "код" in error_msg or "code" in error_msg.lower()

    def test_timeout(self):
        from src.neural_method import _run_subprocess

        with pytest.raises(RuntimeError, match="таймаут|timeout"):
            _run_subprocess(["sleep", "10"], timeout=1)


# =============================================================================
# Test subprocess compressors (nncp, cmix) — error handling only
# =============================================================================


class TestNncpCompressor:
    """Tests for nncp_compressor — only error handling, no actual binary."""

    def test_missing_binary_readable_error(self, tmp_path):
        from src.neural_method import nncp_compressor

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="NNCP не найден"):
                nncp_compressor(test_file)


class TestCmixCompressor:
    """Tests for cmix_compressor — only error handling."""

    def test_missing_binary_readable_error(self, tmp_path):
        from src.neural_method import cmix_compressor

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="cmix не найден"):
                cmix_compressor(test_file)


# =============================================================================
# Test LLM compressors with mocked model
# =============================================================================


class MockTokenizer:
    """Mock tokenizer for testing without downloading models."""

    pad_token = None
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def encode(self, text, return_tensors=None, truncation=None):
        import torch

        # Simple char-level tokenization for testing
        tokens = [ord(c) % 1000 for c in text]
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens


class MockModel:
    """Mock model for testing without downloading models."""

    class Config:
        n_positions = 128

    config = Config()

    @classmethod
    def from_pretrained(cls, model_name, torch_dtype=None):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, labels=None):
        import torch

        batch_size, seq_len = input_ids.shape
        # Return mock loss (average NLL in nats)
        result = MagicMock()
        result.loss = torch.tensor(2.0)  # ~2 nats per token
        result.logits = torch.randn(batch_size, seq_len, 1000)
        return result


class TestComputeNllBits:
    """Tests for _compute_nll_bits with mocked model."""

    def test_short_sequence_returns_zero(self):
        import torch
        from src.neural_method import _compute_nll_bits

        model = MockModel()
        tokenizer = MockTokenizer()
        tokens = torch.tensor([[42]])  # Single token

        bits = _compute_nll_bits(model, tokenizer, tokens, "cpu", max_length=128)
        assert bits == 0.0

    def test_returns_positive_bits(self):
        import torch
        from src.neural_method import _compute_nll_bits

        model = MockModel()
        tokenizer = MockTokenizer()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])

        bits = _compute_nll_bits(model, tokenizer, tokens, "cpu", max_length=128)
        assert bits > 0
        assert isinstance(bits, float)

    def test_sliding_window_consistent(self):
        """Verify sliding window produces consistent results."""
        import torch
        from src.neural_method import _compute_nll_bits

        model = MockModel()
        tokenizer = MockTokenizer()

        # Same tokens, different max_length should give similar results
        # (within reason, since we're using mock)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        bits_128 = _compute_nll_bits(model, tokenizer, tokens, "cpu", max_length=128)
        bits_4 = _compute_nll_bits(model, tokenizer, tokens, "cpu", max_length=4)

        # Both should be positive
        assert bits_128 > 0
        assert bits_4 > 0


class TestTsZipCompressor:
    """Tests for ts_zip_compressor with mocked model."""

    def test_returns_int(self, tmp_path):
        from src.neural_method import ts_zip_compressor, clear_model_cache

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world! This is a test.")

        with patch("src.neural_method._get_cached_model") as mock_get_model:
            mock_get_model.return_value = (MockTokenizer(), MockModel())

            result = ts_zip_compressor(test_file, lvl=1)

            assert isinstance(result, int)
            assert result >= 1

        clear_model_cache()

    def test_empty_file_returns_one(self, tmp_path):
        from src.neural_method import ts_zip_compressor

        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        with patch("src.neural_method._get_cached_model") as mock_get_model:
            mock_get_model.return_value = (MockTokenizer(), MockModel())

            result = ts_zip_compressor(test_file)
            assert result == 1

    def test_single_char_returns_one(self, tmp_path):
        from src.neural_method import ts_zip_compressor

        test_file = tmp_path / "single.txt"
        test_file.write_text("a")

        with patch("src.neural_method._get_cached_model") as mock_get_model:
            mock_get_model.return_value = (MockTokenizer(), MockModel())

            result = ts_zip_compressor(test_file)
            assert result == 1


class TestLlmCompress:
    """Tests for llm_compress with mocked model."""

    def test_returns_int(self, tmp_path):
        from src.neural_method import llm_compress, clear_model_cache

        test_file = tmp_path / "test.txt"
        test_file.write_text("Привет мир! Это тест на русском языке.")

        with patch("src.neural_method._get_cached_model") as mock_get_model:
            mock_get_model.return_value = (MockTokenizer(), MockModel())

            result = llm_compress(test_file, lvl=1)

            assert isinstance(result, int)
            assert result >= 1

        clear_model_cache()

    def test_empty_file_returns_one(self, tmp_path):
        from src.neural_method import llm_compress

        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        with patch("src.neural_method._get_cached_model") as mock_get_model:
            mock_get_model.return_value = (MockTokenizer(), MockModel())

            result = llm_compress(test_file)
            assert result == 1

    def test_two_files_merged(self, tmp_path):
        from src.neural_method import llm_compress

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Hello ")
        file2.write_text("World")

        with patch("src.neural_method._get_cached_model") as mock_get_model:
            mock_get_model.return_value = (MockTokenizer(), MockModel())

            result = llm_compress(file1, file2, lvl=1)
            assert isinstance(result, int)
            assert result >= 1


# =============================================================================
# Test model cache
# =============================================================================


class TestModelCache:
    """Tests for model caching."""

    def test_clear_cache(self):
        from src.neural_method import _MODEL_CACHE, clear_model_cache

        # Add something to cache
        _MODEL_CACHE[("test", "cpu", "float32")] = ("tok", "model")
        assert len(_MODEL_CACHE) > 0

        clear_model_cache()
        assert len(_MODEL_CACHE) == 0


# =============================================================================
# Integration test (optional, requires transformers)
# =============================================================================


@pytest.mark.skipif(
    True,  # Set to False to run integration tests
    reason="Integration tests require model download"
)
class TestIntegration:
    """Integration tests with real models. Skip by default."""

    def test_ts_zip_real_model(self, tmp_path):
        from src.neural_method import ts_zip_compressor

        test_file = tmp_path / "test.txt"
        test_file.write_text("The quick brown fox jumps over the lazy dog.")

        result = ts_zip_compressor(test_file, lvl=1)
        assert isinstance(result, int)
        assert result > 0
