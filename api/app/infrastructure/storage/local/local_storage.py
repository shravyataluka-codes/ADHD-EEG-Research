"""Safe local filesystem implementation of FileStorageProvider."""
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO

from ..interfaces import InvalidStorageKey, StorageConflict, StorageError, StorageUnavailable

_KEY = re.compile(r"^[a-f0-9]{32}\.(csv|mat)$")


class LocalFileStorageProvider:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self.root.is_dir():
                raise OSError("storage root is not a directory")
        except OSError as exc:
            raise StorageUnavailable("Local file storage is unavailable.") from exc

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise InvalidStorageKey("Invalid storage key.")
        path = (self.root / key).resolve()
        if path.parent != self.root:
            raise InvalidStorageKey("Invalid storage key.")
        return path

    def save(self, key: str, source: BinaryIO, *, max_bytes: int) -> int:
        target = self._path(key)
        temp_name = None
        size = 0
        try:
            fd, temp_name = tempfile.mkstemp(prefix=".upload-", dir=self.root)
            with os.fdopen(fd, "wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise StorageError("File exceeds the configured size limit.")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            # Atomic no-overwrite publication prevents duplicate key replacement.
            # Windows rename fails if the destination exists; POSIX hard-link
            # provides the same no-replace behavior without a race.
            if os.name == "nt":
                os.rename(temp_name, target)
                temp_name = None
            else:
                os.link(temp_name, target)
            return size
        except FileExistsError as exc:
            raise StorageConflict("Storage key already exists.") from exc
        except StorageError:
            raise
        except OSError as exc:
            raise StorageUnavailable("Could not save the stored file.") from exc
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass

    def retrieve(self, key: str) -> BinaryIO:
        path = self._path(key)
        try:
            return path.open("rb")
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise StorageUnavailable("Could not read the stored file.") from exc

    def delete(self, key: str) -> bool:
        try:
            self._path(key).unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise StorageUnavailable("Could not delete the stored file.") from exc

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).is_file()
        except OSError as exc:
            raise StorageUnavailable("Could not inspect the stored file.") from exc

    def reference(self, key: str) -> str:
        self._path(key)
        return key
