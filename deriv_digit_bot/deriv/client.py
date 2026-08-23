"""
Deriv WebSocket client.

Uses the Options API OTP bootstrap (REST -> OTP -> WS URL), falling
back to direct v3 connect for tokens not yet migrated. Message
payloads use the corrected schema learned the hard way earlier in
this project: `proposal` uses `underlying_symbol` (not `symbol`), and
`active_symbols` does not accept `product_type`.

This client is intentionally straightforward (connect, send, receive
by matching req_id) rather than a fully hardened production
supervisor - reconnect-on-drop is implemented, but this has not been
soak-tested against live market conditions. Treat DERIV_DEMO/LIVE use
as something to watch closely, not "fire and forget."
"""
from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any, Dict, Optional

import websockets

try:
    import httpx
except ImportError:
    httpx = None

REST_BASE = "https://api.derivws.com"
WS_BASE = "wss://ws.derivws.com/websockets/v3"


class DerivClient:
    def __init__(self, app_id: str, token: str, want_virtual: Optional[bool] = None,
                 forced_account_id: Optional[str] = None):
        """
        want_virtual: True to require a demo/virtual account, False to
        require a real-money account, None to accept whichever comes
        back first (only appropriate for HISTORICAL_SIMULATION, which
        never calls connect() at all). main.py passes this explicitly
        based on cfg.mode so DERIV_DEMO can never silently land on a
        real-money account or vice versa.
        forced_account_id: if set (DERIV_ACCOUNT_ID env var), skips
        auto-selection entirely and requires exactly this account id
        to be present - use this if you have multiple accounts of the
        same type and auto-selection's "first match" isn't the one
        you want.
        """
        self.app_id = app_id
        self.token = token
        self.want_virtual = want_virtual
        self.forced_account_id = forced_account_id or None
        self.resolved_account: Optional[Dict[str, Any]] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id_counter = itertools.count(1)
        self._pending: Dict[int, asyncio.Future] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._subscribers: Dict[str, list] = {}  # msg_type -> list of async callbacks

    async def connect(self) -> None:
        url = await self._resolve_ws_url()
        is_legacy = "authorize" not in url  # OTP URLs embed the auth; legacy doesn't
        try:
            self._ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        except websockets.InvalidStatus as e:
            status = getattr(e.response, "status_code", None)
            if is_legacy and status == 401:
                raise RuntimeError(
                    f"Legacy WebSocket connect was rejected with HTTP {status}. "
                    "This does NOT necessarily mean the account is migrated - it can equally "
                    "mean DERIV_APP_ID is invalid, unregistered, or still the default '1089' "
                    "(Deriv's public test ID, meant for unauthenticated calls only - not valid "
                    "for authorize/account-specific calls). Check DERIV_APP_ID is your own "
                    "registered app ID before assuming this is a migration issue. Also check "
                    "the WARNING printed just before this error (if any) for why the OTP "
                    "bootstrap fell back to legacy in the first place."
                ) from e
            raise
        self._recv_task = asyncio.create_task(self._recv_loop())
        if is_legacy:
            # Legacy fallback path still needs an explicit authorize call
            await self._send({"authorize": self.token})

    def _is_virtual(self, account: Dict[str, Any]) -> Optional[bool]:
        """Deriv account objects use is_virtual (bool/0-1) in most
        responses; some variants use account_type == 'demo'/'real'.
        Returns None if neither field is present (caller should treat
        that account as ambiguous, not assume it matches)."""
        if "is_virtual" in account:
            return bool(account["is_virtual"])
        if "account_type" in account:
            return str(account["account_type"]).lower() in ("demo", "virtual")
        return None

    def _select_account(self, accounts: list) -> Optional[Dict[str, Any]]:
        if not accounts:
            return None

        if self.forced_account_id:
            for a in accounts:
                if str(a.get("id")) == str(self.forced_account_id):
                    return a
            return None  # explicit id requested but not found - do not fall back

        if self.want_virtual is None:
            return accounts[0]

        matching = [a for a in accounts if self._is_virtual(a) == self.want_virtual]
        if matching:
            if len(matching) > 1:
                # multiple matching accounts (e.g. several real-money
                # currencies) - take the first but make this visible
                # rather than silently guessing which one matters.
                print(f"WARNING: {len(matching)} accounts match "
                      f"{'demo' if self.want_virtual else 'real-money'} - using "
                      f"{matching[0].get('id')} ({matching[0].get('currency', '?')}). "
                      f"Set DERIV_ACCOUNT_ID explicitly if this is wrong.")
            return matching[0]

        # Nothing matched the requested type - do NOT silently fall back
        # to a wrong-type account (that would mean DERIV_DEMO could pick
        # a real-money account, or vice versa).
        return None

    async def _resolve_ws_url(self) -> str:
        if not self.token:
            raise RuntimeError("DERIV_TOKEN is empty - cannot authenticate. Check the Railway variable is set.")

        if httpx is None:
            print("WARNING: httpx not installed - cannot use the Options API OTP bootstrap, "
                  "falling back to legacy direct connect. This WILL 401 if your account has "
                  "been migrated to the new Options API. Check requirements.txt was actually "
                  "installed (see Build Logs for a 'pip install' step that completed).")
            return f"{WS_BASE}?app_id={self.app_id}"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Deriv-App-ID": self.app_id,  # required by both endpoints per Deriv docs - was missing entirely
                }
                try:
                    accounts_resp = await client.get(f"{REST_BASE}/trading/v1/options/accounts", headers=headers)
                except httpx.RequestError as e:
                    # Could not even reach the REST host (DNS/network/timeout) -
                    # this is the one case where falling back to legacy is a
                    # reasonable guess rather than a hidden bug.
                    print(f"WARNING: could not reach {REST_BASE} ({e!r}) - "
                          f"falling back to legacy direct connect.")
                    return f"{WS_BASE}?app_id={self.app_id}"

                if accounts_resp.status_code == 404:
                    # Could genuinely mean "no Options accounts for this token"
                    # (real accounts, not enrolled in this product) - but could
                    # also mean a bad app_id/path. Print it either way instead
                    # of silently falling back, since that silence is exactly
                    # what makes this hard to diagnose from logs alone.
                    print(f"WARNING: GET {REST_BASE}/trading/v1/options/accounts returned 404 - "
                          f"falling back to legacy direct connect. If legacy also fails, check "
                          f"DERIV_APP_ID is a real registered app ID, not left as the default "
                          f"test value.")
                    return f"{WS_BASE}?app_id={self.app_id}"

                if accounts_resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"Deriv REST API rejected DERIV_TOKEN with HTTP {accounts_resp.status_code} "
                        f"when fetching accounts. This means the token is invalid, expired, or lacks "
                        f"the required scope - NOT that the account isn't migrated. Check DERIV_TOKEN "
                        f"in Railway's Variables tab."
                    )

                accounts_resp.raise_for_status()  # any other non-2xx is a real, surfaced error
                accounts_body = accounts_resp.json()
                # Defensive: Deriv's OTP endpoint wraps its payload in a
                # top-level "data" key, so the accounts list plausibly does
                # too even though the docs page doesn't show a response
                # example. Handle both shapes rather than assume either.
                if "data" in accounts_body and isinstance(accounts_body["data"], dict):
                    accounts = accounts_body["data"].get("accounts", [])
                else:
                    accounts = accounts_body.get("accounts", [])
                if not accounts:
                    return f"{WS_BASE}?app_id={self.app_id}"

                account = self._select_account(accounts)
                if account is None:
                    if self.forced_account_id:
                        raise RuntimeError(
                            f"DERIV_ACCOUNT_ID={self.forced_account_id} was not found among "
                            f"{len(accounts)} account(s) returned by Deriv for this token."
                        )
                    kind = "demo/virtual" if self.want_virtual else "real-money"
                    raise RuntimeError(
                        f"No {kind} account found for this token among {len(accounts)} "
                        f"account(s) returned by Deriv. Refusing to guess - check DERIV_TOKEN "
                        f"and your account setup on Deriv, or set DERIV_ACCOUNT_ID explicitly."
                    )
                self.resolved_account = account
                account_id = account["id"]
                print(f"Resolved Deriv account: id={account_id} "
                      f"currency={account.get('currency', '?')} "
                      f"virtual={self._is_virtual(account)}")

                otp_resp = await client.post(
                    f"{REST_BASE}/trading/v1/options/accounts/{account_id}/otp", headers=headers
                )
                if otp_resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"Deriv REST API rejected the OTP request with HTTP {otp_resp.status_code} "
                        f"for account {account_id}. Check DERIV_TOKEN has permission for this account."
                    )
                otp_resp.raise_for_status()
                otp_body = otp_resp.json()
                # Per Deriv docs the payload is nested: {"data": {"url": "wss://..."}}
                # - not a flat "websocket_url" key. Handle both defensively in
                # case the API returns either shape.
                if "data" in otp_body and isinstance(otp_body["data"], dict) and "url" in otp_body["data"]:
                    return otp_body["data"]["url"]
                if "websocket_url" in otp_body:
                    return otp_body["websocket_url"]
                raise RuntimeError(
                    f"OTP response didn't contain a recognizable URL field. "
                    f"Response keys: {list(otp_body.keys())}"
                )
        except RuntimeError:
            raise  # our own diagnosed errors - never swallow these into a silent fallback

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                req_id = msg.get("req_id")
                if req_id is not None and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)
                msg_type = msg.get("msg_type")
                for cb in self._subscribers.get(msg_type, []):
                    asyncio.create_task(cb(msg))
        except websockets.ConnectionClosed:
            pass

    async def _send(self, payload: Dict[str, Any], expect_response: bool = True) -> Optional[Dict[str, Any]]:
        req_id = next(self._req_id_counter)
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        if expect_response:
            self._pending[req_id] = fut
        await self._ws.send(json.dumps(payload))
        if expect_response:
            return await asyncio.wait_for(fut, timeout=15)
        return None

    def subscribe(self, msg_type: str, callback) -> None:
        self._subscribers.setdefault(msg_type, []).append(callback)

    async def subscribe_ticks(self, symbol: str) -> None:
        await self._send({"ticks": symbol, "subscribe": 1})

    async def get_active_symbols(self) -> Dict[str, Any]:
        return await self._send({"active_symbols": "brief"})

    async def get_proposal(self, contract_type: str, symbol: str, amount: float, duration: int,
                            duration_unit: str, currency: str = "USD") -> Dict[str, Any]:
        return await self._send({
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": currency,
            "duration": duration,
            "duration_unit": duration_unit,
            "underlying_symbol": symbol,
        })

    async def buy(self, proposal_id: str, price: float) -> Dict[str, Any]:
        return await self._send({"buy": proposal_id, "price": price})

    async def get_contract_status(self, contract_id: str) -> Dict[str, Any]:
        return await self._send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 0})

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
