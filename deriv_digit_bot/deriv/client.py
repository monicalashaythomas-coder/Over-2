"""
Deriv WebSocket client - Options API (REST OTP bootstrap + WebSocket).

This connection layer was ported from a working reference bot rather
than built from documentation alone, after several rounds of the
docs-only version failing in ways that turned out to rest on
unverified assumptions. Concretely, this version differs from the
previous one in ways that matter:

  - REST calls use urllib in a thread executor, not httpx. Removes a
    whole dependency and the failure class of "did httpx actually
    get installed in this container" that cost real debugging time.
  - NO legacy direct-connect fallback. The previous version fell back
    to the deprecated wss://ws.derivws.com/websockets/v3 path on
    various REST failures, which then failed itself with a confusing
    401 - two guesses stacked on top of each other. This version
    either gets a working OTP URL or raises clearly; there is nothing
    to silently fall back to.
  - Account selection matches Deriv's actual response shape: accounts
    carry a `type` (or `account_type`) field valued "real" or "demo" -
    not an `is_virtual` boolean, which was an assumption in the
    previous version that was never actually confirmed against a real
    response.
  - Supports a pre-set account_id (DERIV_ACCOUNT_ID) that skips the
    accounts-lookup call entirely and goes straight to the OTP
    endpoint - useful once you know which account you want.
  - Send-queue + recv-pump architecture with req_id-based routing via
    asyncio.Future, so a concurrent proposal fetch can't get
    misrouted into whatever's currently consuming the tick stream.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

REST_BASE = "https://api.derivws.com"


class DerivClient:
    def __init__(self, app_id: str, token: str, use_real_account: bool = False,
                 account_id: Optional[str] = None, ws_ping_interval: int = 30):
        self.app_id = app_id
        self.token = token
        self.use_real_account = use_real_account
        self.account_id = account_id or None
        self.ws_ping_interval = ws_ping_interval

        self.ws_url: Optional[str] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._send_queue: Optional[asyncio.Queue] = None
        self._inbox: Optional[asyncio.Queue] = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._req_id_counter = 1
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self.initial_balance: float = 0.0

    # ---- REST bootstrap (blocking; always run via run_in_executor) ----

    def _rest_request(self, path: str, method: str = "GET") -> dict:
        req = urllib.request.Request(
            f"{REST_BASE}{path}", method=method,
            headers={
                "Deriv-App-ID": self.app_id,
                "Authorization": f"Bearer {self.token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error calling {path}: {exc.reason}") from exc

    def _resolve_account_id(self) -> str:
        payload = self._rest_request("/trading/v1/options/accounts")
        accounts = payload.get("data") or payload.get("accounts") or []
        if not accounts:
            raise RuntimeError(
                "No accounts returned by Deriv for this token. Check DERIV_TOKEN "
                "is valid and was generated on developers.deriv.com (the current "
                "platform), not an older/legacy system."
            )
        wanted = "real" if self.use_real_account else "demo"
        for acc in accounts:
            t = str(acc.get("type") or acc.get("account_type") or "").lower()
            if t == wanted:
                return acc.get("account_id") or acc.get("id")
        first = accounts[0]
        first_id = first.get("account_id") or first.get("id")
        print(f"WARNING: no account with type='{wanted}' found among "
              f"{len(accounts)} account(s) - using the first one returned "
              f"({first_id}). Set DERIV_ACCOUNT_ID explicitly to control this.")
        return first_id

    def _fetch_ws_url(self) -> str:
        if not self.account_id:
            self.account_id = self._resolve_account_id()
        payload = self._rest_request(
            f"/trading/v1/options/accounts/{self.account_id}/otp", method="POST")
        url = (payload.get("data") or {}).get("url")
        if not url:
            raise RuntimeError(f"OTP response missing url field: {payload}")
        return url

    # ---- Connect ----

    async def connect(self) -> None:
        if not self.token:
            raise RuntimeError("DERIV_TOKEN is empty - cannot authenticate.")

        loop = asyncio.get_event_loop()
        self.ws_url = await loop.run_in_executor(None, self._fetch_ws_url)

        safe = self.ws_url.split("?")[0]
        print(f"Connecting -> {safe} (account {self.account_id})")

        self.ws = await websockets.connect(
            self.ws_url, ping_interval=self.ws_ping_interval,
            ping_timeout=20, close_timeout=10,
        )
        self._send_queue = asyncio.Queue()
        self._inbox = asyncio.Queue()
        self._start_io()

        # Confirm the connection actually works end-to-end before
        # returning - a balance check is cheap and catches auth
        # problems immediately rather than on the first real trade call.
        await self.send({"balance": 1})
        resp = await self.receive_type("balance", timeout=15)
        if resp is None or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            await self.close()
            raise RuntimeError(f"Post-connect balance check failed: {err}")
        bal = resp.get("balance", {})
        self.initial_balance = float(bal.get("balance", 0) or 0)
        print(f"Connected. account={self.account_id} "
              f"balance={self.initial_balance:.2f} {bal.get('currency', '')}")

    def _start_io(self) -> None:
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump(), name="send_pump")
        self._recv_task = asyncio.create_task(self._recv_pump(), name="recv_pump")
        self._req_id_counter = 1
        self._pending_requests = {}

    def _next_req_id(self) -> int:
        rid = self._req_id_counter
        self._req_id_counter += 1
        return rid

    async def _send_pump(self) -> None:
        while True:
            data, fut = await self._send_queue.get()
            try:
                await self.ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_queue.task_done()

    async def _recv_pump(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                rid = msg.get("req_id")
                if rid and rid in self._pending_requests:
                    fut = self._pending_requests.pop(rid)
                    if not fut.done():
                        fut.set_result(msg)
                else:
                    await self._inbox.put(msg)
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            for fut in self._pending_requests.values():
                if not fut.done():
                    fut.cancel()
            self._pending_requests.clear()
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            print(f"RECV error: {exc}")
            for fut in self._pending_requests.values():
                if not fut.done():
                    fut.cancel()
            self._pending_requests.clear()
            await self._inbox.put({"__disconnect__": True})

    # ---- Send / receive ----

    async def send(self, data: dict) -> None:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await self._send_queue.put((data, fut))
        await fut

    async def send_with_id(self, data: dict, timeout: float = 12) -> Optional[dict]:
        """Sends with a unique req_id and awaits the matching response via
        a Future - safe alongside a continuous tick stream, since the
        reply is routed by req_id instead of 'whatever's next in the
        inbox' (which a concurrent tick could easily beat it to)."""
        loop = asyncio.get_event_loop()
        rid = self._next_req_id()
        fut = loop.create_future()
        self._pending_requests[rid] = fut
        data = dict(data)
        data["req_id"] = rid
        await self.send(data)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(rid, None)
            if not fut.done():
                fut.cancel()
            return None
        except asyncio.CancelledError:
            self._pending_requests.pop(rid, None)
            return None

    async def receive(self, timeout: float = 10) -> dict:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def receive_type(self, msg_type: str, timeout: float = 10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    # ---- Trading calls used by main.py ----

    async def subscribe_ticks(self, symbol: str) -> bool:
        await self.send({"ticks": symbol, "subscribe": 1})
        resp = await self.receive_type("tick", timeout=10)
        if resp is None or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            print(f"Subscribe failed for {symbol}: {err}")
            return False
        return True

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self.send({"balance": 1})
            resp = await self.receive_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as exc:
            print(f"Balance fetch error: {exc}")
        return None

    async def get_active_symbols(self) -> Optional[dict]:
        return await self.send_with_id({"active_symbols": "brief"})

    async def get_proposal(self, contract_type: str, symbol: str, amount: float, duration: int,
                            duration_unit: str, currency: str = "USD") -> Optional[dict]:
        return await self.send_with_id({
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": currency,
            "duration": duration,
            "duration_unit": duration_unit,
            "underlying_symbol": symbol,
        })

    async def buy(self, proposal_id: str, price: float) -> Optional[dict]:
        return await self.send_with_id({"buy": proposal_id, "price": price})

    async def get_contract_status(self, contract_id: str) -> Optional[dict]:
        return await self.send_with_id({
            "proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 0
        })

    async def close(self) -> None:
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
