"""Wire protocol shared with Assets/Scripts/RL/RLWire.cs.

Frame:   uint32 payload_length | payload
Payload: uint32 json_length | UTF-8 JSON header | binary blob

The header lists the arrays in the blob as {"name", "dtype", "shape"}; dtype is "f4" or "i4",
little-endian, C order. Every exchange is request/response, so the simulation only advances when
the trainer asks it to.
"""

from __future__ import annotations

import json
import socket
import struct
import time

import numpy as np

DTYPES = {"f4": np.dtype("<f4"), "i4": np.dtype("<i4")}
DTYPE_NAMES = {np.dtype("<f4"): "f4", np.dtype("<i4"): "i4"}


class UnityError(RuntimeError):
    """The environment rejected a message."""


class Connection:
    def __init__(self, host: str, port: int, connect_timeout: float = 120.0, io_timeout: float | None = 600.0,
                 alive=None):
        """alive: optional callable; when it returns False (the player process died) stop waiting."""
        deadline = time.time() + connect_timeout
        last_error = None
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError as e:  # Unity may still be booting
                last_error = e
                if alive is not None and not alive():
                    raise ConnectionError(f"the environment on port {port} exited before accepting a connection") from e
                if time.time() > deadline:
                    raise ConnectionError(f"could not reach the environment on {host}:{port}: {last_error}") from e
                time.sleep(0.5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(io_timeout)

    # ------------------------------------------------------------------ send

    def send(self, header: dict, arrays: list[tuple[str, np.ndarray]] | tuple = ()) -> None:
        header = dict(header)
        blobs = []
        infos = []
        for name, arr in arrays:
            arr = np.ascontiguousarray(arr)
            if arr.dtype.kind == "f":
                arr = arr.astype("<f4", copy=False)
            elif arr.dtype.kind in "iu":
                arr = arr.astype("<i4", copy=False)
            else:
                raise TypeError(f"array {name} has unsupported dtype {arr.dtype}")
            infos.append({"name": name, "dtype": DTYPE_NAMES[arr.dtype], "shape": list(arr.shape)})
            blobs.append(arr.tobytes())
        header["arrays"] = infos
        j = json.dumps(header).encode("utf-8")
        payload_len = 4 + len(j) + sum(len(b) for b in blobs)
        self.sock.sendall(b"".join([struct.pack("<II", payload_len, len(j)), j, *blobs]))

    # ------------------------------------------------------------------ receive

    def _read_exactly(self, n: int) -> bytes:
        chunks = []
        while n > 0:
            chunk = self.sock.recv(min(n, 1 << 20))
            if not chunk:
                raise ConnectionError("environment closed the connection")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def recv(self) -> tuple[dict, dict[str, np.ndarray]]:
        (payload_len,) = struct.unpack("<I", self._read_exactly(4))
        payload = self._read_exactly(payload_len)
        (json_len,) = struct.unpack_from("<I", payload, 0)
        header = json.loads(payload[4:4 + json_len].decode("utf-8"))
        if header.get("type") == "error":
            raise UnityError(header.get("message", "unknown error"))
        arrays = {}
        offset = 4 + json_len
        for info in header.get("arrays", []):
            dtype = DTYPES[info["dtype"]]
            shape = tuple(info["shape"])
            count = int(np.prod(shape)) if shape else 1
            arrays[info["name"]] = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape)
            offset += count * dtype.itemsize
        return header, arrays

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
