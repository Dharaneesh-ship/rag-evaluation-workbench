"""Tests for the R2 index manifest — persistence and IndexManager integration."""

from __future__ import annotations

from pathlib import Path

from rag.infra.indexes.index_manager import IndexManager
from rag.infra.indexes.manifest import IndexManifest, SCHEMA_VERSION


class TestIndexManifestRoundTrip:
    def test_empty_manifest_has_no_provider(self):
        m = IndexManifest.empty()
        assert m.embedding_provider is None
        assert m.schema_version == SCHEMA_VERSION

    def test_save_and_load_round_trip(self, tmp_path: Path):
        m = IndexManifest(embedding_provider="openai", embedding_model="text-embedding-3-small", vector_dimension=1536)
        m.save(tmp_path)
        loaded = IndexManifest.load(tmp_path)
        assert loaded == m

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        loaded = IndexManifest.load(tmp_path / "does_not_exist")
        assert loaded == IndexManifest.empty()

    def test_load_corrupt_file_returns_empty_instead_of_raising(self, tmp_path: Path):
        (tmp_path / "index_manifest.json").write_text("{not json", encoding="utf-8")
        loaded = IndexManifest.load(tmp_path)
        assert loaded == IndexManifest.empty()


class TestIndexManagerManifestIntegration:
    def test_fresh_index_manager_has_empty_manifest(self, tmp_path: Path):
        mgr = IndexManager(index_dir=tmp_path / "indexes")
        assert mgr.manifest == IndexManifest.empty()

    def test_record_embedding_config_then_save_persists_manifest(self, tmp_path: Path):
        index_dir = tmp_path / "indexes"
        mgr = IndexManager(index_dir=index_dir)
        mgr.record_embedding_config("openai", "text-embedding-3-small", 1536)
        mgr.save()

        mgr2 = IndexManager(index_dir=index_dir)
        assert mgr2.manifest.embedding_provider == "openai"
        assert mgr2.manifest.embedding_model == "text-embedding-3-small"
        assert mgr2.manifest.vector_dimension == 1536

    def test_record_embedding_config_with_none_provider_is_a_noop(self, tmp_path: Path):
        index_dir = tmp_path / "indexes"
        mgr = IndexManager(index_dir=index_dir)
        mgr.record_embedding_config(None, "text-embedding-3-small", 1536)
        assert mgr.manifest == IndexManifest.empty()

    def test_reload_picks_up_manifest_written_by_another_process(self, tmp_path: Path):
        index_dir = tmp_path / "indexes"
        writer = IndexManager(index_dir=index_dir)
        writer.record_embedding_config("openai", "m", 42)
        writer.save()

        reader = IndexManager(index_dir=index_dir)
        reader.reload()
        assert reader.manifest.embedding_provider == "openai"
        assert reader.manifest.vector_dimension == 42
