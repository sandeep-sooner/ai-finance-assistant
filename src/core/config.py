"""Settings: defaults from config.yaml, overridden by environment variables (and .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"

_DEFAULT_MODELS = {
    "fast": {"openai": "gpt-4o-mini", "gemini": "gemini-2.0-flash"},
    "smart": {"openai": "gpt-4o", "gemini": "gemini-2.5-pro"},
}


def load_yaml(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml. A missing file or missing PyYAML just means 'use the built-in defaults'."""
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except (ImportError, OSError):
        return {}


def _dig(cfg: dict, path: str, default: Any) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


@dataclass(frozen=True)
class Settings:
    llm_provider: str          # "openai", "gemini" or "fake" (demo mode, no API key needed)
    llm_model: str             # the fast-tier model (kept for backwards compatibility)
    llm_smart_model: str
    openai_api_key: str | None
    openai_base_url: str | None  # optional proxy endpoint for OpenAI-compatible keys
    google_api_key: str | None
    temperature: float
    llm_calls_per_min: int     # central cap on LLM calls (keep under your key's rate limit)
    chat_requests_per_min: int # per-client cap on endpoints that call the LLM
    api_requests_per_min: int  # per-client cap on data endpoints (market, analysis)
    market_data_provider: str  # "yfinance", "alphavantage" or "sample" (offline, deterministic)
    market_quote_ttl: float    # seconds a quote is reused before refetching
    market_timeout: float      # seconds before a provider call is abandoned
    max_history_messages: int
    max_message_chars: int
    cors_origins: tuple[str, ...]
    log_level: str
    # ---- model router / resilience / cache ----
    model_router_threshold: float = 0.5
    router_llm_fallback: bool = True
    llm_retries: int = 2
    llm_backoff_base: float = 0.5
    llm_token_timeout: float = 30.0
    breaker_failures: int = 3
    breaker_cooldown: float = 60.0
    llm_cache_ttl: float = 3600.0
    llm_cache_max_items: int = 1000
    retrieval_cache_max_items: int = 500
    pricing: dict = field(default_factory=dict)
    tier_models: dict = field(default_factory=lambda: _DEFAULT_MODELS)
    # ---- RAG ----
    rag_embedder: str = "auto"
    rag_embedding_model: str = "text-embedding-3-small"
    rag_chunk_chars: int = 600
    rag_chunk_overlap: int = 100
    rag_top_k: int = 4
    rag_dense_weight: float = 0.6
    rag_min_score: float = 0.22
    rag_relative_cut: float = 0.6
    rag_index_dir: str = ".cache/rag"
    # ---- market ----
    alphavantage_api_key: str | None = None
    market_history_ttl: float = 900.0
    market_news_ttl: float = 600.0
    market_retries: int = 1

    @classmethod
    def load(cls, config: dict | None = None) -> "Settings":
        cfg = load_yaml() if config is None else config

        def env(name: str, path: str, default: Any, cast=str) -> Any:
            raw = os.getenv(name)
            value = raw if raw not in (None, "") else _dig(cfg, path, default)
            if cast is bool:
                return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
            return cast(value)

        openai_key = os.getenv("OPENAI_API_KEY") or None
        google_key = os.getenv("GOOGLE_API_KEY") or None
        provider = env("LLM_PROVIDER", "llm.provider", "auto").strip().lower()
        if provider in ("", "auto"):
            provider = "openai" if openai_key else "gemini" if google_key else "fake"

        tiers = {tier: {**_DEFAULT_MODELS[tier], **(_dig(cfg, f"llm.tiers.{tier}", {}) or {})} for tier in _DEFAULT_MODELS}
        fast = os.getenv("LLM_FAST_MODEL") or os.getenv("LLM_MODEL") or tiers["fast"].get(provider, "")
        smart = os.getenv("LLM_SMART_MODEL") or tiers["smart"].get(provider, "") or fast
        origins = tuple(o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip())

        return cls(
            llm_provider=provider,
            llm_model=fast,
            llm_smart_model=smart,
            openai_api_key=openai_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            google_api_key=google_key,
            temperature=env("LLM_TEMPERATURE", "llm.temperature", 0.3, float),
            llm_calls_per_min=env("LLM_CALLS_PER_MIN", "llm.calls_per_min", 50, int),
            chat_requests_per_min=env("CHAT_REQUESTS_PER_MIN", "api.chat_requests_per_min", 20, int),
            api_requests_per_min=env("API_REQUESTS_PER_MIN", "api.api_requests_per_min", 120, int),
            market_data_provider=env("MARKET_DATA_PROVIDER", "market.provider", "yfinance").strip().lower(),
            market_quote_ttl=env("MARKET_QUOTE_TTL", "market.quote_ttl", 60, float),
            market_timeout=env("MARKET_TIMEOUT", "market.timeout", 8, float),
            max_history_messages=env("MAX_HISTORY_MESSAGES", "api.max_history_messages", 10, int),
            max_message_chars=env("MAX_MESSAGE_CHARS", "api.max_message_chars", 2000, int),
            cors_origins=origins,
            log_level=env("LOG_LEVEL", "api.log_level", "INFO"),
            model_router_threshold=env("MODEL_ROUTER_THRESHOLD", "model_router.smart_threshold", 0.5, float),
            router_llm_fallback=env("ROUTER_LLM_FALLBACK", "model_router.llm_intent_fallback", True, bool),
            llm_retries=env("LLM_RETRIES", "resilience.retries", 2, int),
            llm_backoff_base=env("LLM_BACKOFF_BASE", "resilience.backoff_base", 0.5, float),
            llm_token_timeout=env("LLM_TOKEN_TIMEOUT", "resilience.token_timeout", 30, float),
            breaker_failures=int(_dig(cfg, "resilience.breaker_failures", 3)),
            breaker_cooldown=float(_dig(cfg, "resilience.breaker_cooldown", 60)),
            llm_cache_ttl=env("LLM_CACHE_TTL", "cache.llm_ttl", 3600, float),
            llm_cache_max_items=int(_dig(cfg, "cache.llm_max_items", 1000)),
            retrieval_cache_max_items=int(_dig(cfg, "cache.retrieval_max_items", 500)),
            pricing=dict(_dig(cfg, "llm.pricing", {}) or {}),
            tier_models=tiers,
            rag_embedder=env("RAG_EMBEDDER", "rag.embedder", "auto").strip().lower(),
            rag_embedding_model=env("RAG_EMBEDDING_MODEL", "rag.embedding_model", "text-embedding-3-small"),
            rag_chunk_chars=int(_dig(cfg, "rag.chunk_chars", 600)),
            rag_chunk_overlap=int(_dig(cfg, "rag.chunk_overlap", 100)),
            rag_relative_cut=float(_dig(cfg, "rag.relative_cut", 0.6)),
            rag_top_k=env("RAG_TOP_K", "rag.top_k", 4, int),
            rag_dense_weight=float(_dig(cfg, "rag.dense_weight", 0.6)),
            rag_min_score=env("RAG_MIN_SCORE", "rag.min_score", 0.22, float),
            rag_index_dir=env("RAG_INDEX_DIR", "rag.index_dir", ".cache/rag"),
            alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY") or None,
            market_history_ttl=float(_dig(cfg, "market.history_ttl", 900)),
            market_news_ttl=float(_dig(cfg, "market.news_ttl", 600)),
            market_retries=int(_dig(cfg, "market.retries", 1)),
        )
