"""Zoom client — stdlib only, matches the shape of GenesysClient.

Server-to-Server OAuth (account_credentials grant). Used for cloud-recording
ingest of Zoom Meetings (audio + Zoom's own VTT transcript + AI Companion
summaries). No PII in logs; UUIDs only.
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


class ZoomError(RuntimeError):
    pass


def safe_uuid(uuid: str) -> str:
    """Make a Zoom meeting UUID safe for a filename.

    Zoom UUIDs are base64 (may contain '+', '/', '='); Windows filenames
    can't hold any of those. URL-safe base64 alphabet, padding stripped.
    Reversible: '_' -> '/', '-' -> '+', re-pad to multiple of 4.
    """
    return uuid.replace("+", "-").replace("/", "_").rstrip("=")


class ZoomClient:
    def __init__(self) -> None:
        self._account = config.env("ZOOM_ACCOUNT_ID", required=True)
        self._cid = config.env("ZOOM_CLIENT_ID", required=True)
        self._secret = config.env("ZOOM_CLIENT_SECRET", required=True)
        self._login = "https://zoom.us"
        self._api = "https://api.zoom.us/v2"
        self._token: str | None = None
        self._token_exp = 0.0

    # --- auth ---
    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        basic = base64.b64encode(f"{self._cid}:{self._secret}".encode()).decode()
        body = urllib.parse.urlencode({
            "grant_type": "account_credentials",
            "account_id": self._account,
        }).encode()
        req = urllib.request.Request(
            f"{self._login}/oauth/token", data=body, method="POST",
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                tok = json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise ZoomError(f"auth HTTP {e.code}: {detail}") from None
        self._token = tok["access_token"]
        self._token_exp = time.time() + float(tok.get("expires_in", 3600))
        return self._token

    # --- low-level request with 429/5xx backoff ---
    def _api_request(self, method: str, path: str,
                     body: dict | None = None, tries: int = 5) -> Any:
        url = f"{self._api}{path}" if path.startswith("/") else path
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
                raise ZoomError(f"{method} {path} -> HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, ConnectionResetError, TimeoutError) as e:
                if attempt < tries - 1:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise ZoomError(
                    f"{method} {path} -> network error: {type(e).__name__}: {e}"
                ) from None
        raise ZoomError(f"{method} {path} -> exhausted retries")

    # --- listing ---
    def list_all_recordings(self, start_date: str, end_date: str,
                            page_size: int = 300) -> list[dict]:
        """Drain ALL pages of /v2/accounts/me/recordings into a list.

        Pagination must be drained quickly: Zoom's next_page_token has a
        short TTL (~15 min) and expires if downloads slow down the consumer.
        Listing alone is fast (one request per page), so collect everything
        upfront, then let the caller iterate / download at any speed.

        Zoom caps each query at 30 days; spans longer than that are split
        into 30-day chunks transparently. Date strings are ISO YYYY-MM-DD.
        """
        import datetime as dt
        s = dt.date.fromisoformat(start_date)
        e = dt.date.fromisoformat(end_date)
        out: list[dict] = []
        while s <= e:
            chunk_end = min(s + dt.timedelta(days=29), e)
            next_token: str | None = None
            while True:
                params = {"from": s.isoformat(), "to": chunk_end.isoformat(),
                          "page_size": page_size}
                if next_token:
                    params["next_page_token"] = next_token
                path = "/accounts/me/recordings?" + urllib.parse.urlencode(params)
                resp = self._api_request("GET", path) or {}
                out.extend(resp.get("meetings", []) or [])
                next_token = resp.get("next_page_token") or None
                if not next_token:
                    break
            s = chunk_end + dt.timedelta(days=1)
        return out

    def iter_account_recordings(self, start_date: str, end_date: str,
                                page_size: int = 300) -> Iterator[dict]:
        """Backwards-compatible iterator — drains pagination upfront."""
        yield from self.list_all_recordings(start_date, end_date, page_size)

    # --- file download ---
    def download(self, url: str, dest) -> int:
        """Stream a Zoom recording file URL to disk. Returns bytes written.

        Zoom accepts the bearer token via Authorization header. urllib drops
        custom headers on cross-host redirects, so include ?access_token=...
        in the query string too — Zoom honors either."""
        token = self._ensure_token()
        sep = "&" if "?" in url else "?"
        auth_url = f"{url}{sep}access_token={urllib.parse.quote(token)}"
        req = urllib.request.Request(auth_url, method="GET")
        total = 0
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        return total
