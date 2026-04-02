"""Simple in-memory cache with TTL."""

import time
from typing import Any, Optional


class Cache:
    """Thread-safe in-memory cache with TTL expiration."""

    def __init__(self, default_ttl: int = 900):
        self._store: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            value, expiry = self._store[key]
            if time.time() < expiry:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expiry = time.time() + (ttl or self._default_ttl)
        self._store[key] = (value, expiry)

    def clear(self):
        self._store.clear()

    def has(self, key: str) -> bool:
        return self.get(key) is not None


# Global cache instance
cache = Cache()
