"""Central OpenAI / Azure OpenAI client factory.

Every module that talks to an LLM or an embedding model resolves its client
through this module instead of constructing one itself.  That keeps the
public-OpenAI ↔ Azure-OpenAI choice in exactly one place: flipping
``LLM_BACKEND`` re-points the entire pipeline — query analysis, expansion,
critic, generation, embeddings, and the evaluation harness — with no call-site
changes.  It mirrors what ``retrieval.factory`` does for the retrieval
substrate.

Backends
--------
* ``openai`` (default) — the public API at ``api.openai.com``, authenticated
  with ``OPENAI_API_KEY``.  This is the local-development path.
* ``azure_openai`` — an Azure OpenAI resource, authenticated with either an API
  key (``AZURE_OPENAI_API_KEY``) or, preferably, Managed Identity when no key
  is set.

Deployment names vs model names
-------------------------------
This is the one genuinely awkward difference between the two backends.  The
public API takes a *model* name (``"gpt-4o"``).  Azure OpenAI takes a
*deployment* name — an arbitrary label you chose when deploying the model,
which may or may not match the underlying model.  The same ``model=`` argument
therefore means different things depending on the backend.

:func:`get_generation_model` and :func:`get_embedding_model` resolve the right
value for the active backend, so callers pass the result of those rather than
reading ``GENERATION_MODEL`` / ``EMBEDDING_MODEL`` directly.  Naming your Azure
deployments identically to the models they serve (``gpt-4o``,
``text-embedding-3-small``) makes the two backends agree and is the recommended
setup.

Client caching
--------------
Clients are cached per-process with :func:`functools.lru_cache`.  The OpenAI
SDK client is thread-safe and holds a connection pool, so building one per call
(as the pre-factory code did in eight separate modules) wasted connections for
no benefit.

Configuration (environment variables)
-------------------------------------
* ``LLM_BACKEND``                     — ``openai`` (default) or ``azure_openai``.
* ``OPENAI_API_KEY``                  — required for the ``openai`` backend.
* ``AZURE_OPENAI_ENDPOINT``           — e.g. ``https://<name>.openai.azure.com/``.
* ``AZURE_OPENAI_API_KEY``            — optional; omit to use Managed Identity.
* ``AZURE_OPENAI_API_VERSION``        — REST API version (default below).
* ``AZURE_OPENAI_GPT_DEPLOYMENT``     — generation deployment name.
* ``AZURE_OPENAI_ANALYSIS_DEPLOYMENT``— analysis/critic deployment name.
* ``AZURE_OPENAI_EMBED_DEPLOYMENT``   — embedding deployment name.
* ``GENERATION_MODEL``                — model name for the ``openai`` backend.
* ``ANALYSIS_MODEL``                  — model name for the ``openai`` backend.
* ``EMBEDDING_MODEL``                 — model name for the ``openai`` backend.

The pipeline uses three distinct models: a large one for final generation, a
small fast one for query analysis / expansion / critic / eval scoring, and an
embedding model.  An Azure deployment is required for each.
"""

import logging
import os
from functools import lru_cache

from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Either concrete client the factory can return.  Both expose the same
#: ``chat.completions`` and ``embeddings`` surface, so annotate parameters with
#: this alias rather than committing a call site to one backend.
LLMClient = OpenAI | AzureOpenAI

_AZURE_ALIASES: frozenset[str] = frozenset(
    {"azure", "azure_openai", "azure-openai"}
)
_OPENAI_ALIASES: frozenset[str] = frozenset({"openai", "public"})

# Pinned so a rolling API version can never silently change request or response
# shapes underneath the pipeline's structured outputs.
_DEFAULT_AZURE_API_VERSION: str = "2024-10-21"

_DEFAULT_GENERATION_MODEL: str = "gpt-4o"
_DEFAULT_ANALYSIS_MODEL: str = "gpt-4o-mini"
_DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# Azure AD scope for data-plane calls to Azure OpenAI (Managed Identity path).
_COGNITIVE_SERVICES_SCOPE: str = "https://cognitiveservices.azure.com/.default"

#: Transient failures worth retrying.  Deliberately excludes auth, bad-request,
#: and not-found errors: a wrong key or a missing deployment fails identically
#: on every attempt, and retrying only delays a clear error message.
_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

_EMBED_MAX_ATTEMPTS: int = 5

#: Valid values for the reasoning-effort controls below.
_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"minimal", "low", "medium", "high"}
)
#: Values that explicitly disable sending the parameter at all.
_REASONING_OFF: frozenset[str] = frozenset({"", "off", "none"})

#: Maps a pipeline role to the environment variable controlling its effort.
#: Roles are split by how much the step's accuracy depends on deliberation, not
#: by which model serves it — see :func:`reasoning_extra_body`.
_REASONING_ENV_VARS: dict[str, str] = {
    # Query classification, retrieval critic, eval judge: each makes a
    # judgement that routes or gates the pipeline.
    "analysis": "REASONING_EFFORT_ANALYSIS",
    # Query expansion: mechanical rephrasing, no judgement to get wrong.
    "expansion": "REASONING_EFFORT_EXPANSION",
    # Final answer generation.
    "generation": "REASONING_EFFORT_GENERATION",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_azure_backend() -> bool:
    """Report whether the Azure OpenAI backend is selected.

    Returns:
        ``True`` if ``LLM_BACKEND`` names the Azure backend, else ``False``.

    Raises:
        ValueError: If ``LLM_BACKEND`` is set to an unrecognised value.
    """
    backend = os.environ.get("LLM_BACKEND", "openai").strip().lower()
    if backend in _AZURE_ALIASES:
        return True
    if backend in _OPENAI_ALIASES:
        return False
    raise ValueError(
        f"Unknown LLM_BACKEND '{backend}'. "
        f"Expected one of: {sorted(_OPENAI_ALIASES | _AZURE_ALIASES)}."
    )


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return the process-wide OpenAI-compatible client for the active backend.

    Both returned types expose the same ``chat.completions`` and ``embeddings``
    surface, so callers need not branch on which one they received — only the
    value passed as ``model=`` differs (see :func:`get_generation_model`).

    Returns:
        A cached :class:`openai.OpenAI` or :class:`openai.AzureOpenAI` client.

    Raises:
        ValueError: If ``LLM_BACKEND`` is unrecognised, or if the selected
            backend's required configuration is missing.
    """
    if is_azure_backend():
        return _build_azure_client()
    return _build_public_client()


def get_generation_model() -> str:
    """Return the model/deployment identifier for chat completions.

    Returns:
        The Azure deployment name when the Azure backend is active, otherwise
        the public model name from ``GENERATION_MODEL``.
    """
    if is_azure_backend():
        return os.environ.get(
            "AZURE_OPENAI_GPT_DEPLOYMENT",
            os.environ.get("GENERATION_MODEL", _DEFAULT_GENERATION_MODEL),
        )
    return os.environ.get("GENERATION_MODEL", _DEFAULT_GENERATION_MODEL)


def reasoning_extra_body(role: str) -> dict:
    """Return ``extra_body`` kwargs controlling reasoning effort for a role.

    GPT-5-family models emit hidden *reasoning tokens* before their visible
    answer.  They are billed as output and dominate latency: a one-word query
    classification measured 192 reasoning tokens against 12 visible ones, and
    took ~4x longer than the same call with reasoning disabled.

    Effort is **not** a free speed dial, and the split below was measured, not
    guessed.  On ``gpt-5-mini``, ``minimal`` classified easy queries correctly
    at ~3x the speed, but misclassified the cross-document query "Compare our
    scaffold SOP against OSHA requirements" as ``out_of_scope`` — which makes
    the pipeline refuse to answer before retrieval runs.  It was also *slower*
    on that query than leaving reasoning on, because the model floundered
    without it.  Query classification routes every downstream decision, so it
    is the wrong place to trade accuracy for latency.  Expansion, which only
    rephrases, tolerates a lower effort.

    ``reasoning_effort`` is passed through ``extra_body`` rather than as a named
    argument because the pinned ``openai`` SDK predates the GPT-5 family and
    rejects it as an unexpected keyword.  ``extra_body`` merges into the JSON
    request payload, so the parameter reaches the service unchanged.

    This is **opt-in**: with no environment variable set, nothing is sent.  That
    keeps the parameter away from models that would reject it — the public
    OpenAI backend running ``gpt-4o``, or a non-reasoning Azure deployment —
    without this module having to guess a model's capabilities from an
    arbitrary deployment name.

    Args:
        role: ``"analysis"`` (classifier, critic, eval judge — steps whose
            judgement routes or gates the pipeline), ``"expansion"`` (query
            rephrasing), or ``"generation"`` (final answer).

    Returns:
        ``{"reasoning_effort": <effort>}`` when the role's environment variable
        names a valid effort, otherwise an empty dict.

    Raises:
        KeyError: If ``role`` is not a recognised pipeline role.
        ValueError: If the environment variable holds an unrecognised effort.
    """
    env_var = _REASONING_ENV_VARS[role]
    effort = os.environ.get(env_var, "").strip().lower()
    if effort in _REASONING_OFF:
        return {}
    if effort not in _REASONING_EFFORTS:
        raise ValueError(
            f"Invalid {env_var}='{effort}'. Expected one of "
            f"{sorted(_REASONING_EFFORTS)}, or 'off' to disable."
        )
    return {"reasoning_effort": effort}


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(_EMBED_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts, retrying transient failures with backoff.

    Every embedding call in the system routes through here.  Without retry, a
    single dropped connection mid-ingestion aborts the file being processed and
    silently drops that whole document from the index — the ingestion loop
    catches per-file exceptions and moves on, so the run still reports success
    and the gap is invisible until a query mysteriously finds nothing.

    Retries cover connection drops, timeouts, rate limits, and 5xx responses.
    Authentication and bad-request errors are *not* retried: they fail
    identically every time, so retrying would only delay a clear error.

    Args:
        texts: Texts to embed in a single API call.
        model: Model or Azure deployment name.  Defaults to
            :func:`get_embedding_model`.

    Returns:
        One embedding vector per input text, in the same order.

    Raises:
        openai.OpenAIError: If every attempt fails, the last error is re-raised.
    """
    resolved_model = model or get_embedding_model()
    response = get_llm_client().embeddings.create(
        model=resolved_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def get_analysis_model() -> str:
    """Return the model/deployment identifier for the small analysis model.

    This is the model behind query classification, query expansion, the
    retrieval critic, and LLM-judge scoring in the evaluation harness.

    Returns:
        The Azure deployment name when the Azure backend is active, otherwise
        the public model name from ``ANALYSIS_MODEL``.
    """
    if is_azure_backend():
        return os.environ.get(
            "AZURE_OPENAI_ANALYSIS_DEPLOYMENT",
            os.environ.get("ANALYSIS_MODEL", _DEFAULT_ANALYSIS_MODEL),
        )
    return os.environ.get("ANALYSIS_MODEL", _DEFAULT_ANALYSIS_MODEL)


def get_embedding_model() -> str:
    """Return the model/deployment identifier for embeddings.

    Returns:
        The Azure deployment name when the Azure backend is active, otherwise
        the public model name from ``EMBEDDING_MODEL``.
    """
    if is_azure_backend():
        return os.environ.get(
            "AZURE_OPENAI_EMBED_DEPLOYMENT",
            os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        )
    return os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_public_client() -> OpenAI:
    """Construct a client for the public OpenAI API.

    Returns:
        An authenticated :class:`openai.OpenAI` instance.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. It is required when LLM_BACKEND=openai."
        )
    return OpenAI(api_key=api_key)


def _build_azure_client() -> AzureOpenAI:
    """Construct a client for an Azure OpenAI resource.

    Uses an API key when ``AZURE_OPENAI_API_KEY`` is set, and Managed Identity
    (via ``DefaultAzureCredential``) otherwise.  The key path exists for local
    development against a cloud resource; the identity path is the intended
    production configuration, since it removes the long-lived secret entirely.

    Returns:
        An authenticated :class:`openai.AzureOpenAI` instance.

    Raises:
        ValueError: If ``AZURE_OPENAI_ENDPOINT`` is not set.
        ImportError: If Managed Identity is selected but ``azure-identity`` is
            not installed.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT is not set. It is required when "
            "LLM_BACKEND=azure_openai (e.g. https://<name>.openai.azure.com/)."
        )
    api_version = os.environ.get(
        "AZURE_OPENAI_API_VERSION", _DEFAULT_AZURE_API_VERSION
    )

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=_build_token_provider(),
        api_version=api_version,
    )


def _build_token_provider():
    """Build an Azure AD bearer-token provider for Managed Identity auth.

    Imported lazily so that the ``azure-identity`` dependency is only required
    by deployments that actually use the Managed Identity path.

    Returns:
        A callable returning a fresh Azure AD access token for the Cognitive
        Services data plane.  The SDK invokes it per request and the underlying
        credential handles caching and refresh.

    Raises:
        ImportError: If ``azure-identity`` is not installed.
    """
    try:
        from azure.identity import (  # noqa: PLC0415
            DefaultAzureCredential,
            get_bearer_token_provider,
        )
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "azure-identity is required for Managed Identity authentication. "
            "Install it, or set AZURE_OPENAI_API_KEY to use key-based auth."
        ) from exc

    return get_bearer_token_provider(
        DefaultAzureCredential(), _COGNITIVE_SERVICES_SCOPE
    )
