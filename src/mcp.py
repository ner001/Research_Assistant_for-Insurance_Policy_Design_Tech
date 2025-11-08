import threading
from typing import Any, Dict, Optional

class Context:
    """Simple thread-safe key/value context for agents."""
    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()