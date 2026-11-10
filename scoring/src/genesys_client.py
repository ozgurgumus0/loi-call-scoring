"""Genesys Cloud client — stdlib only (no requests/SDK), so Phase 1 needs no pip installs.

Scope used: OAuth2 client-credentials -> Analytics conversation details + Recording read.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

from . import config


class GenesysError(RuntimeError):
    pass


class GenesysClient:
    def __init__(self) -> None:
        region = config.env("GENESYS_REGION", None, required=True)
        self._cid = config.env("GENESYS_CLIENT_ID", required=True)
        self._secret = config.env("GENESYS_CLIENT_SECRET", required=True)
        self._login = f"https://login.{region}"
        self._api = f"https://api.{region}"
        self._token: str | None = None
        self._token_exp = 0.0

    # --- auth ---
    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        basic = base64.b64encode(f"{self._cid}:{self._secret}".encode()).decode()
        body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
        req = urllib.request.Request(
            f"{self._login}/oauth/token", data=body, method="POST",
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            tok = json.load(r)
        self._token = tok["access_token"]
        self._token_exp = time.time() + float(tok.get("expires_in", 3600))
        return self._token

    # --- low-level request with 429/5xx backoff ---
    def _api_request(self, method: str, path: str,
                     body: dict | None = None, tries: int = 5) -> Any:
        url = f"{self._api}{path}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(tries):
            token = self._ensure_token()
            req = urllib.request.Request(
                url, data=data, method=method,
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504) and attempt < tries - 1:
                    retry_after = e.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else 2 ** attempt
                    time.sleep(min(wait, 30))
                    continue
                if e.code == 401 and attempt < tries - 1:
                    self._token = None
                    continue
                detail = e.read().decode(errors="replace")[:500]
                raise GenesysError(f"{method} {path} -> HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, ConnectionResetError, TimeoutError) as e:
                # Transient network issues (DNS hiccup, TLS reset, idle disconnect).
                # Backoff and retry; only surface after exhausting tries.
                if attempt < tries - 1:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise GenesysError(
                    f"{method} {path} -> network error: {type(e).__name__}: {e}"
                ) from None
        raise GenesysError(f"{method} {path} -> exhausted retries")

    # --- org sanity check ---
    def org_name(self) -> str:
        return self._api_request("GET", "/api/v2/organizations/me").get("name", "?")

    # --- analytics: iterate voice conversations in [start,end) (ISO8601 Z) ---
    def iter_voice_conversations(self, start_iso: str, end_iso: str,
                                 page_size: int = 100) -> Iterator[dict]:
        page = 1
        while True:
            body = {
                "interval": f"{start_iso}/{end_iso}",
                "order": "desc",
                "orderBy": "conversationStart",
                "paging": {"pageSize": page_size, "pageNumber": page},
                "segmentFilters": [{
                    "type": "and",
                    "predicates": [{"dimension": "mediaType", "value": "voice"}],
                }],
            }
            resp = self._api_request(
                "POST", "/api/v2/analytics/conversations/details/query", body)
            convs = (resp or {}).get("conversations") or []
            if not convs:
                return
            yield from convs
            if len(convs) < page_size:
                return
            page += 1

    # --- recordings ---
    def get_recordings(self, conversation_id: str) -> list[dict]:
        """Recordings for a conversation. HTTP 404 = no recording exists for
        this call (the common case in this org) -> return [] (not an error)."""
        path = f"/api/v2/conversations/{conversation_id}/recordings"
        try:
            resp = self._api_request("GET", path)
        except GenesysError as e:
            if "HTTP 404" in str(e):
                return []
            raise
        return resp if isinstance(resp, list) else []

    @staticmethod
    def audio_uri(recording: dict) -> tuple[str | None, int]:
        """Return (download_url, n_tracks). n_tracks>1 => multichannel (stereo-ish)."""
        uris = recording.get("mediaUris") or {}
        n = len(uris)
        if not n:
            return None, 0
        # track "0" = mixed/first channel; good enough for the MVP pull.
        first = uris.get("0") or next(iter(uris.values()))
        return (first or {}).get("mediaUri"), n

    @staticmethod
    def download(url: str, dest) -> int:
        """Stream a CloudFront-signed media URL to disk. Returns bytes written."""
        req = urllib.request.Request(url, method="GET")
        total = 0
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        return total
