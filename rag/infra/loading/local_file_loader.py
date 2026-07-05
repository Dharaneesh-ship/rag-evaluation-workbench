"""Local file loader — reads a file from disk into a RawArtifact."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default cap on file size read into memory. read_bytes() loads the whole
# file into a single Python bytes object with no streaming, so an unbounded
# read on a very large file (a multi-GB video mistakenly pointed at ingest,
# say) risks exhausting memory before any format/content check even runs.
DEFAULT_MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


class FileTooLargeError(OSError):
    """Raised when a file exceeds the loader's configured size limit.

    Subclasses OSError so callers that already catch loader I/O errors
    (FileNotFoundError, PermissionError, etc.) catch this too unchanged.
    """


@dataclass
class RawArtifact:
    """Raw file content and metadata produced by the loader stage.

    Passed from the loader to the sniffer, which detects the MIME type
    and routes it to the appropriate parser.

    Attributes:
        source_path: Absolute path to the source file.
        raw_bytes: Raw file content as bytes.
        metadata: Basic file metadata (size, name, extension, timestamps).
    """

    source_path: str
    raw_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Decode raw_bytes as UTF-8, replacing undecodable bytes.

        Returns:
            Decoded text string.
        """
        return self.raw_bytes.decode("utf-8", errors="replace")

    @property
    def size_bytes(self) -> int:
        """Size of the raw content in bytes."""
        return len(self.raw_bytes)


class LocalFileLoader:
    """Loads a local file from disk into a RawArtifact.

    Args:
        max_file_size_bytes: Files larger than this raise
            ``FileTooLargeError`` before any read is attempted. Guards
            against an oversized file (or one pointed at the wrong path)
            being read entirely into memory with no size check —
            ``read_bytes()`` has no streaming fallback. Defaults to
            ``DEFAULT_MAX_FILE_SIZE_BYTES`` (500 MB); pass ``None`` to
            disable the check entirely.

    Usage::

        loader = LocalFileLoader()
        artifact = loader.load("/path/to/document.pdf")
    """

    def __init__(self, max_file_size_bytes: int | None = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
        self._max_file_size_bytes = max_file_size_bytes

    def load(self, path: str | Path) -> RawArtifact:
        """Read a file and return a RawArtifact with content and metadata.

        Args:
            path: Absolute or relative path to the file to load.

        Returns:
            A RawArtifact containing the raw bytes and file metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the path points to a directory.
            FileTooLargeError: If the file exceeds ``max_file_size_bytes``.
            PermissionError: If the file cannot be read.
            OSError: For other I/O errors.
        """
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")

        stat = file_path.stat()

        if (
            self._max_file_size_bytes is not None
            and stat.st_size > self._max_file_size_bytes
        ):
            raise FileTooLargeError(
                f"File '{file_path}' is {stat.st_size:,} bytes "
                f"({stat.st_size / (1024 * 1024):.1f} MB), which exceeds the "
                f"{self._max_file_size_bytes:,}-byte "
                f"({self._max_file_size_bytes / (1024 * 1024):.0f} MB) limit. "
                f"Increase max_file_size_bytes if this file is expected."
            )

        raw_bytes = file_path.read_bytes()

        metadata: dict[str, Any] = {
            "file_name": file_path.name,
            "extension": file_path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "absolute_path": str(file_path),
        }

        return RawArtifact(
            source_path=str(file_path),
            raw_bytes=raw_bytes,
            metadata=metadata,
        )
