"""Framing between the host and the worker over one TCP connection.

A request is a header (ticket, kind, three lengths) followed by a metadata
part, a body and an optional raw payload; the ticket is assigned under the
send lock, so the stream carries requests in issue order and the worker,
which executes them one after another from one thread, applies that order.
Only some kinds get a response (ticket, status, payload); the host's reader
thread hands each to the thread waiting on its ticket. The worker's first
frame is a response with ticket 0 describing itself.

Fire-and-forget requests are held back and written together: a request
that waits for a response flushes what is pending in front of it, and so
does reaching ``_COALESCE`` pending bytes, so nothing waits on the host
for longer than a few dozen requests' worth of work.

Nothing here knows torch. It is a plain socket, not the LUPINE RPC core:
the hop is loopback for a subprocess worker and a LAN for an attached one.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from collections.abc import Iterator
from typing import Any

REQUEST = struct.Struct("<QBIIQ")
RESPONSE = struct.Struct("<QBQ")
# Payloads at least this large are sent on their own instead of copied into
# the pending bytes.
_SEPARATE_SEND = 64 * 1024
_COALESCE = 16384
_READ_BUFFER = 256 * 1024

Frame = tuple[int, int, bytes, bytes, bytearray | None]


class ConnectionClosed(RuntimeError):
    pass


def _parse_address(address: str) -> tuple[str, int]:
    host, _, port = address.rpartition(":")
    return host.strip("[]") or "127.0.0.1", int(port)


class _Connection:
    def __init__(self, sock: socket.socket):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock
        self.reader = sock.makefile("rb", buffering=_READ_BUFFER)
        self.write_lock = threading.Lock()
        self.pending = bytearray()
        self.closed = False
        self.messages = 0
        self.bytes = 0
        self.writes = 0

    def _write(self, head: bytes, *parts: Any) -> None:
        """Queues a frame under the write lock; a part too large to copy is
        sent at once, behind what is pending."""

        pending = self.pending
        pending += head
        for part in parts:
            if len(part) >= _SEPARATE_SEND:
                self._flush()
                self._send(part)
            else:
                pending += part

    def _flush(self) -> None:
        if self.pending:
            self._send(self.pending)
            del self.pending[:]

    def _send(self, data: Any) -> None:
        self.bytes += len(data)
        self.writes += 1
        try:
            self.sock.sendall(data)
        except OSError as exc:
            self.closed = True
            raise ConnectionClosed(f"lupine: the torch worker connection is closed ({exc})") from exc

    def _read_exact(self, size: int) -> bytes:
        data = self.reader.read(size)
        if len(data) != size:
            raise ConnectionClosed("lupine: the torch worker connection is closed")
        return data

    def _read_into(self, buffer: Any) -> None:
        view = memoryview(buffer).cast("B")
        while len(view):
            n = self.reader.readinto(view)
            if not n:
                raise ConnectionClosed("lupine: the torch worker connection is closed")
            view = view[n:]

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            # The peer may already be gone; there is nothing left to release.
            pass
        self.sock.close()


class _Waiter:
    __slots__ = ("done", "status", "payload", "into")

    def __init__(self, into: Any):
        self.done = threading.Event()
        self.status = -1
        self.payload = b""
        self.into = into


class Client(_Connection):
    """The host's end: send fire-and-forget requests, call for a response."""

    def __init__(self, sock: socket.socket):
        super().__init__(sock)
        self.next_ticket = 1
        self.waiters: dict[int, _Waiter] = {}
        self.error: ConnectionClosed | None = None
        ticket, status, length = RESPONSE.unpack(self._read_exact(RESPONSE.size))
        if ticket != 0 or status != 0:
            raise ConnectionClosed("lupine: the torch worker did not introduce itself")
        self.hello: dict[str, Any] = json.loads(self._read_exact(length))
        threading.Thread(target=self._read_loop, name="lupine-worker-reader", daemon=True).start()

    @classmethod
    def connect(cls, address: str, timeout: float = 30.0) -> Client:
        sock = socket.create_connection(_parse_address(address), timeout=timeout)
        sock.settimeout(None)
        return cls(sock)

    def send(self, kind: int, meta: bytes, body: bytes, extra: Any = b"") -> int:
        with self.write_lock:
            ticket = self.next_ticket
            self.next_ticket += 1
            self.messages += 1
            self._write(REQUEST.pack(ticket, kind, len(meta), len(body), len(extra)), meta, body, extra)
            if len(self.pending) >= _COALESCE:
                self._flush()
        return ticket

    def call(self, kind: int, meta: bytes, body: bytes, extra: Any = b"", into: Any = None) -> tuple[int, bytes]:
        """Sends and waits; returns the response status and payload (the
        payload lands in ``into`` instead when given)."""

        waiter = _Waiter(into)
        with self.write_lock:
            if self.error is not None:
                raise self.error
            ticket = self.next_ticket
            self.next_ticket += 1
            self.messages += 1
            self.waiters[ticket] = waiter
            self._write(REQUEST.pack(ticket, kind, len(meta), len(body), len(extra)), meta, body, extra)
            self._flush()
        waiter.done.wait()
        if waiter.status < 0:
            assert self.error is not None
            raise self.error
        return waiter.status, waiter.payload

    def _read_loop(self) -> None:
        try:
            while True:
                ticket, status, length = RESPONSE.unpack(self._read_exact(RESPONSE.size))
                waiter = self.waiters.pop(ticket, None)
                if waiter is not None and waiter.into is not None and status == 0:
                    if length != len(memoryview(waiter.into).cast("B")):
                        raise ConnectionClosed(
                            f"lupine: the worker sent {length} bytes for a {len(waiter.into)}-byte copy"
                        )
                    self._read_into(waiter.into)
                    payload = b""
                else:
                    payload = self._read_exact(length) if length else b""
                if waiter is not None:
                    waiter.status = status
                    waiter.payload = payload
                    waiter.done.set()
        except (ConnectionClosed, OSError, ValueError) as exc:
            error = exc if isinstance(exc, ConnectionClosed) else ConnectionClosed(f"lupine: the torch worker connection is closed ({exc})")
            with self.write_lock:
                self.error = error
                self.closed = True
                waiters = list(self.waiters.values())
                self.waiters.clear()
            for waiter in waiters:
                waiter.done.set()


class Server(_Connection):
    """The worker's end: introduce itself, then yield requests in order."""

    @classmethod
    def accept(cls, listener: socket.socket) -> Server:
        sock, _ = listener.accept()
        return cls(sock)

    def hello(self, info: dict[str, Any]) -> None:
        payload = json.dumps(info).encode()
        self._write(RESPONSE.pack(0, 0, len(payload)), payload)
        self._flush()

    def requests(self) -> Iterator[Frame]:
        while True:
            head = self.reader.read(REQUEST.size)
            if not head:
                return
            if len(head) != REQUEST.size:
                raise ConnectionClosed("lupine: truncated request")
            ticket, kind, meta_len, body_len, extra_len = REQUEST.unpack(head)
            meta = self._read_exact(meta_len) if meta_len else b""
            body = self._read_exact(body_len) if body_len else b""
            extra = None
            if extra_len:
                extra = bytearray(extra_len)
                self._read_into(extra)
            self.messages += 1
            yield ticket, kind, meta, body, extra

    def respond(self, ticket: int, status: int, payload: Any = b"") -> None:
        with self.write_lock:
            self._write(RESPONSE.pack(ticket, status, len(memoryview(payload).cast("B")) if payload else 0), payload)
            self._flush()
