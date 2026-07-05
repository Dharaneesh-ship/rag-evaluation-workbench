"""Unit tests for LocalFileLoader."""

import pytest

from rag.infra.loading.local_file_loader import (
    FileTooLargeError,
    LocalFileLoader,
    RawArtifact,
)


@pytest.fixture
def loader():
    return LocalFileLoader()


@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello, RAG framework!\nLine two.", encoding="utf-8")
    return f


@pytest.mark.unit
class TestLocalFileLoader:
    def test_load_txt_file_returns_raw_artifact(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert isinstance(artifact, RawArtifact)

    def test_load_returns_correct_bytes(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.raw_bytes == b"Hello, RAG framework!\nLine two."

    def test_load_text_property(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert "Hello, RAG framework!" in artifact.text
        assert "Line two." in artifact.text

    def test_load_metadata_file_name(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.metadata["file_name"] == "sample.txt"

    def test_load_metadata_extension(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.metadata["extension"] == ".txt"

    def test_load_metadata_file_size(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.metadata["file_size_bytes"] == len(b"Hello, RAG framework!\nLine two.")
        assert artifact.size_bytes == artifact.metadata["file_size_bytes"]

    def test_load_metadata_modified_at(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert "modified_at" in artifact.metadata
        assert "T" in artifact.metadata["modified_at"]  # ISO 8601

    def test_load_source_path_is_absolute(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.source_path == str(sample_txt.resolve())

    def test_load_missing_file_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")

    def test_load_directory_raises(self, loader, tmp_path):
        with pytest.raises(IsADirectoryError):
            loader.load(tmp_path)

    def test_load_binary_file(self, loader, tmp_path):
        binary_file = tmp_path / "data.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")
        artifact = loader.load(binary_file)
        assert artifact.raw_bytes == b"\x00\x01\x02\x03"
        assert artifact.metadata["extension"] == ".bin"


@pytest.mark.unit
class TestLocalFileLoaderSizeGuard:
    def test_oversized_file_raises_before_reading(self, tmp_path):
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * 1000)
        loader = LocalFileLoader(max_file_size_bytes=500)

        with pytest.raises(FileTooLargeError, match="500"):
            loader.load(big_file)

    def test_file_too_large_error_is_an_os_error(self, tmp_path):
        # Callers already catching OSError from load() (FileNotFoundError,
        # PermissionError) must catch this too without code changes.
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * 1000)
        loader = LocalFileLoader(max_file_size_bytes=500)

        with pytest.raises(OSError):
            loader.load(big_file)

    def test_file_at_exactly_the_limit_is_allowed(self, tmp_path):
        f = tmp_path / "exact.bin"
        f.write_bytes(b"x" * 500)
        loader = LocalFileLoader(max_file_size_bytes=500)

        artifact = loader.load(f)  # must not raise
        assert artifact.size_bytes == 500

    def test_file_under_the_limit_is_allowed(self, sample_txt):
        loader = LocalFileLoader(max_file_size_bytes=10_000)
        loader.load(sample_txt)  # must not raise

    def test_none_disables_the_check(self, tmp_path):
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * 10_000)
        loader = LocalFileLoader(max_file_size_bytes=None)

        artifact = loader.load(big_file)  # must not raise
        assert artifact.size_bytes == 10_000

    def test_default_loader_uses_500mb_limit(self):
        from rag.infra.loading.local_file_loader import DEFAULT_MAX_FILE_SIZE_BYTES

        assert DEFAULT_MAX_FILE_SIZE_BYTES == 500 * 1024 * 1024
        assert LocalFileLoader()._max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_BYTES
