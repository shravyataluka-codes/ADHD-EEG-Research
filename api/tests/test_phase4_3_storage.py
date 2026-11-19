import io

import pytest

from api.app.infrastructure.storage.interfaces import InvalidStorageKey, StorageConflict
from api.app.infrastructure.storage.local.local_storage import LocalFileStorageProvider


def test_local_storage_save_retrieve_delete_and_duplicate(tmp_path):
    storage = LocalFileStorageProvider(tmp_path)
    key = "a" * 32 + ".csv"
    assert storage.save(key, io.BytesIO(b"eeg"), max_bytes=3) == 3
    assert storage.exists(key)
    with storage.retrieve(key) as result:
        assert result.read() == b"eeg"
    with pytest.raises(StorageConflict):
        storage.save(key, io.BytesIO(b"other"), max_bytes=10)
    assert storage.delete(key)
    assert not storage.exists(key)


@pytest.mark.parametrize("key", ["../secret.csv", "..\\secret.csv", "/tmp/x.mat", "bad.csv"])
def test_local_storage_rejects_invalid_and_traversal_keys(tmp_path, key):
    storage = LocalFileStorageProvider(tmp_path)
    with pytest.raises(InvalidStorageKey):
        storage.exists(key)


def test_local_storage_size_limit_and_missing_file(tmp_path):
    storage = LocalFileStorageProvider(tmp_path)
    key = "b" * 32 + ".mat"
    with pytest.raises(Exception, match="size limit"):
        storage.save(key, io.BytesIO(b"1234"), max_bytes=3)
    with pytest.raises(FileNotFoundError):
        storage.retrieve(key)
