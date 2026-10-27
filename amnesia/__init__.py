"""Amnesia - local-first long-term memory for Ollama models.

Your local models forgot. Amnesia remembers.

Public entry points are intentionally thin; import from the submodules:

    from amnesia.config import Settings, load_settings
    from amnesia.store import MemoryStore
    from amnesia.retrieval import ContextBuilder
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
