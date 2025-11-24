import os
from typing import List
from langchain_community.document_loaders import WebBaseLoader


def _parse_data_urls(env_value: str) -> List[str]:
    if not env_value:
        return []
    return [u.strip() for u in env_value.split(",") if u.strip()]


def load_data() -> List:
    """Load documents from web using WebBaseLoader.

    - Reads `DATA_URL` (comma-separated) from environment or uses the default Mintlify page.
    - Uses `USER_AGENT` env var if present, otherwise falls back to a generic UA.
    - Uses Playwright when `USE_PLAYWRIGHT` is set to a truthy value (default: True).
    - If `ONLY_TXT` is set (non-empty), returns only documents whose source ends with `.txt`.
    """
    raw = os.getenv("DATA_URL", "https://celoref.mintlify.app/overview/use-cases")
    urls = _parse_data_urls(raw)
    if not urls:
        raise ValueError("No DATA_URL provided. Set the DATA_URL env var to one or more URLs.")

    # Older/newer WebBaseLoader implementations have different parameters.
    # Use the minimal supported call (web_paths) to maximize compatibility.
    try:
        loader = WebBaseLoader(web_paths=urls)
        docs = loader.load()
    except TypeError:
        # Fall back to calling without any kwargs if the signature differs
        loader = WebBaseLoader(urls)
        docs = loader.load()

    only_txt = bool(os.getenv("ONLY_TXT", ""))
    if only_txt:
        filtered = []
        for d in docs:
            src = None
            if hasattr(d, "metadata") and isinstance(d.metadata, dict):
                src = d.metadata.get("source")
            if not src:
                src = getattr(d, "source", "")
            if isinstance(src, str) and src.lower().endswith(".txt"):
                filtered.append(d)
        return filtered

    return docs


__all__ = ["load_data"]
