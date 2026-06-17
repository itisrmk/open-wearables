"""Celery task for processing full-payload (webhook_stream) provider push events.

A single shared task handles all webhook_stream providers (Garmin, Suunto, …).
The provider-specific logic lives in each provider's WebhookHandler.process_payload;
this task is a thin async wrapper providing acks_late and retry guarantees.

Queue and retry policy are configured per-provider at the call site (send_task with queue= kwarg).
"""

from logging import getLogger
from typing import Any

from celery import Task, shared_task
from fastapi import HTTPException

from app.database import SessionLocal
from app.schemas.enums.provider import ProviderName
from app.services.providers.factory import ProviderFactory
from app.utils.structured_logging import log_structured

logger = getLogger(__name__)

# Upstream 4xx where retrying can't recover the object. 401 (token refresh) and
# 429 (rate limit) are excluded — those still benefit from a retry.
_NONRETRIABLE_UPSTREAM_STATUSES = frozenset({400, 403, 404, 410, 422})


def extract_payload_user_id(provider_name: str, payload: dict[str, Any]) -> str | None:
    """Best-effort extraction of the provider's user identifier from a webhook payload.

    The key differs per provider (Oura/Whoop/Polar use ``user_id``, Strava uses
    ``owner_id``, Suunto uses ``username``, Garmin nests ``userId`` per item).
    Returns ``None`` when no identifier is present.
    """
    try:
        provider = ProviderName(provider_name)
    except ValueError:
        return None

    match provider:
        case ProviderName.OURA | ProviderName.WHOOP | ProviderName.POLAR:
            uid = payload.get("user_id")
            return str(uid) if uid is not None else None
        case ProviderName.STRAVA:
            uid = payload.get("owner_id")
            return str(uid) if uid is not None else None
        case ProviderName.SUUNTO:
            return payload.get("username")
        case ProviderName.GARMIN:
            # Garmin batches items under data-type keys; userId lives per item.
            user_ids = {
                str(item["userId"])
                for items in payload.values()
                if isinstance(items, list)
                for item in items
                if isinstance(item, dict) and item.get("userId")
            }
            return ",".join(sorted(user_ids)) or None
        case _:
            return None


@shared_task(
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    default_retry_delay=30,
)
def process_webhook_push(
    self: Task, provider_name: str, payload: dict[str, Any], request_trace_id: str
) -> dict[str, Any]:
    """Process a full-payload webhook push event for any webhook_stream provider.

    Uses ProviderFactory to resolve the provider's WebhookHandler, then calls
    process_payload() with a fresh DB session. Retries up to 3 times on
    unexpected infrastructure errors.
    """
    try:
        factory = ProviderFactory()
        strategy = factory.get_provider(provider_name)
        if strategy.webhooks is None:
            raise ValueError(f"Provider '{provider_name}' has no webhook handler")
        with SessionLocal() as db:
            return strategy.webhooks.process_payload(db, payload, request_trace_id)
    except ValueError as exc:
        # Configuration error (unknown provider, missing handler) — retrying won't help.
        log_structured(
            logger,
            "error",
            "Webhook push task aborted — configuration error",
            provider=provider_name,
            trace_id=request_trace_id,
            provider_user_id=extract_payload_user_id(provider_name, payload),
            error=str(exc),
        )
        raise
    except HTTPException as exc:
        # Non-retriable upstream 4xx (deleted/unqueryable object): ack so the task
        # drops instead of retrying forever. 5xx and 401/429 fall through to retry.
        if exc.status_code in _NONRETRIABLE_UPSTREAM_STATUSES:
            log_structured(
                logger,
                "warning",
                "Webhook push task skipped — upstream non-retriable response",
                provider=provider_name,
                trace_id=request_trace_id,
                provider_user_id=extract_payload_user_id(provider_name, payload),
                upstream_status=exc.status_code,
                error=str(exc.detail),
            )
            return {
                "status": "skipped",
                "reason": "upstream_non_retriable",
                "upstream_status": exc.status_code,
            }
        log_structured(
            logger,
            "error",
            "Webhook push task failed, scheduling retry",
            provider=provider_name,
            trace_id=request_trace_id,
            provider_user_id=extract_payload_user_id(provider_name, payload),
            upstream_status=exc.status_code,
            error=str(exc.detail),
            attempt=self.request.retries,
            max_retries=self.max_retries,
        )
        raise self.retry(exc=exc)
    except Exception as exc:
        log_structured(
            logger,
            "error",
            "Webhook push task failed, scheduling retry",
            provider=provider_name,
            trace_id=request_trace_id,
            provider_user_id=extract_payload_user_id(provider_name, payload),
            error=str(exc),
            attempt=self.request.retries,
            max_retries=self.max_retries,
        )
        raise self.retry(exc=exc)
