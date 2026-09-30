"""Unity environment workers.

Every simulation system in the game is a singleton, so one Unity process holds exactly one battle.
Parallel environments are parallel processes: the trainer launches N headless players on N ports
(or connects to ones you started yourself, e.g. the editor with GameBootstrap.rlTrainingServer on).

Stepping is split into send and receive so all workers simulate at the same time:
    for w in workers: w.step_async(actions[w])
    for w in workers: obs[w] = w.recv()
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

from .protocol import Connection
from .spec import Spec


@dataclass
class Obs:
    """One observation message. Arrays carry a leading team axis of 2 (0 = Player, 1 = Enemy)."""
    arrays: dict
    terminal: bool = False
    winner: int = -1            # 0 / 1, or -1 for a draw or an episode still running
    draw: bool = False
    reason: str = ""
    battle_time: float = 0.0
    episode: int = 0
    ships: tuple = (0, 0)       # hulls per team that hold an agent slot
    stats: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)       # the environment's own timing and memory report

    @classmethod
    def from_message(cls, header: dict, arrays: dict) -> "Obs":
        return cls(
            arrays=arrays,
            terminal=bool(header.get("terminal", False)),
            winner=int(header.get("winner", -1)),
            draw=bool(header.get("draw", False)),
            reason=header.get("reason", ""),
            battle_time=float(header.get("battle_time", 0.0)),
            episode=int(header.get("episode", 0)),
            ships=tuple(header.get("ships", (0, 0))),
            stats=header.get("stats", {}) or {},
            diag=header.get("diag", {}) or {},
        )


def init_message(max_team: int, max_allies: int, max_contacts: int, max_zones: int,
                 action_mode: str, decision_period: float, sim_dt: float, reflexes: bool,
                 max_obstacles: int = 8) -> dict:
    return {
        "type": "init", "max_team": max_team, "max_allies": max_allies, "max_contacts": max_contacts,
        "max_zones": max_zones, "max_obstacles": max_obstacles, "action_mode": action_mode,
        "decision_period": decision_period, "sim_dt": sim_dt, "reflexes": reflexes,
    }


class UnityWorker:
    """One Unity process: a player launched by make_workers, or one already listening on port."""

    def __init__(self, port: int, init: dict, process: subprocess.Popen | None = None, host: str = "127.0.0.1",
                 connect_timeout: float = 180.0):
        self.port = port
        self.process = process
        alive = (lambda: process.poll() is None) if process is not None else None
        try:
            self.conn = Connection(host, port, connect_timeout=connect_timeout, alive=alive)
        except ConnectionError as e:
            log = getattr(process, "log_path", None)
            raise ConnectionError(f"{e}" + (f" - see {log}" if log else "")) from e
        self.conn.send(init)
        header, _ = self.conn.recv()
        self.spec = Spec.from_json(header)

    def reset_async(self, config: dict) -> None:
        self.conn.send({"type": "reset", **config})

    def step_async(self, actions: np.ndarray) -> None:
        self.conn.send({"type": "step"}, [("actions", actions.astype(np.int32, copy=False))])

    def recv(self) -> Obs:
        header, arrays = self.conn.recv()
        return Obs.from_message(header, arrays)

    def render(self, path: str, width: int = 1280, height: int = 720, reveal: bool = True, margin: float = 250.0) -> dict:
        """Writes the current battle to a PNG (the player must have been launched with graphics)."""
        self.conn.send({"type": "render", "path": os.path.abspath(path), "width": width, "height": height,
                        "reveal": reveal, "margin": margin})
        header, _ = self.conn.recv()
        return header

    def close(self) -> None:
        try:
            self.conn.send({"type": "close"})
        except OSError:
            pass
        self.conn.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def launch_player(binary: str, port: int, log_dir: str | None, graphics: bool = False) -> subprocess.Popen:
    """A headless player. graphics=True keeps a GPU device (still no window) so it can render frames."""
    args = [binary, "-batchmode"] + ([] if graphics else ["-nographics"]) + ["-rlTrain", "-rlPort", str(port)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        args += ["-logFile", os.path.join(log_dir, f"unity_{port}.log")]
    env = os.environ.copy()
    if not graphics:
        # a headless player never draws, but with DISPLAY set it still attaches to the desktop's
        # display server and dies with it (logout, screen lock restart) - cut that tie
        for var in ("DISPLAY", "WAYLAND_DISPLAY"):
            env.pop(var, None)
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                            start_new_session=True)
    proc.log_path = os.path.join(log_dir, f"unity_{port}.log") if log_dir else None
    return proc


def make_workers(n: int, init: dict, binary: str | None, base_port: int, ports: list[int] | None = None,
                 log_dir: str | None = None, mock: bool = False, mock_kwargs: dict | None = None):
    """n launched players on consecutive ports, explicit ports to connect to, or mock workers."""
    if mock:
        from .mock_env import MockWorker
        return [MockWorker(init, seed=i, **(mock_kwargs or {})) for i in range(n)]
    if ports:
        return [UnityWorker(p, init) for p in ports]
    if not binary:
        raise ValueError("pass --unity-binary to launch players, --ports to connect to running ones, or --mock")
    # a port still held by an old player would accept our connection and then drop it
    for i in range(n):
        if not port_free(base_port + i):
            raise RuntimeError(f"port {base_port + i} is already in use - an earlier training player may still be "
                               f"shutting down; wait a moment or pass --set base_port=<another port>")
    # start every player first so they boot in parallel, then connect
    procs = [launch_player(binary, base_port + i, log_dir) for i in range(n)]
    try:
        return [UnityWorker(base_port + i, init, process=procs[i]) for i in range(n)]
    except Exception:
        for p in procs:
            p.kill()
        raise
