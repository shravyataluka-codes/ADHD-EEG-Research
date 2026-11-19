"""Thin application facing wrapper around the storage provider."""
class StorageService:
    def __init__(self, provider):
        self.provider = provider

    def save(self, *args, **kwargs): return self.provider.save(*args, **kwargs)
    def retrieve(self, *args, **kwargs): return self.provider.retrieve(*args, **kwargs)
    def delete(self, *args, **kwargs): return self.provider.delete(*args, **kwargs)
    def exists(self, *args, **kwargs): return self.provider.exists(*args, **kwargs)
    def reference(self, *args, **kwargs): return self.provider.reference(*args, **kwargs)
