"""Callbacks: code that runs inside model.learn(), as in Stable-Baselines3.

    class MyCallback(BaseCallback):
        def _on_step(self) -> bool:
            ...                 # self.model, self.num_timesteps and self.n_calls are available
            return True         # False stops training

        def _on_rollout_end(self) -> bool:
            ...                 # after every update; a safe place to reset or evaluate the battles
            return True         # False stops training
"""

from __future__ import annotations

import os


class BaseCallback:
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.model = None
        self.n_calls = 0              # env.step() calls so far
        self.num_timesteps = 0        # decisions, summed over the parallel battles
        self.locals = {}

    def init_callback(self, model) -> None:
        self.model = model
        self.num_timesteps = model.num_timesteps
        self._on_training_start()

    def on_step(self, locals_: dict | None = None) -> bool:
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        self.locals = locals_ or {}
        return self._on_step()

    def on_rollout_end(self) -> bool:
        self.num_timesteps = self.model.num_timesteps
        return self._on_rollout_end()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        pass


class CallbackList(BaseCallback):
    def __init__(self, callbacks: list[BaseCallback]):
        super().__init__()
        self.callbacks = callbacks

    def init_callback(self, model) -> None:
        super().init_callback(model)
        for c in self.callbacks:
            c.init_callback(model)

    def on_step(self, locals_: dict | None = None) -> bool:
        return all([c.on_step(locals_) for c in self.callbacks])

    def on_rollout_end(self) -> bool:
        return all([c.on_rollout_end() for c in self.callbacks])

    def on_training_end(self) -> None:
        for c in self.callbacks:
            c.on_training_end()


class SaveOnIntervalCallback(BaseCallback):
    """Saves models/model_<timesteps>.pt every save_interval timesteps (the Week 9/10 callback).

    num_timesteps grows by n_envs per step, so it rarely lands exactly on a multiple of
    save_interval; this saves whenever it crosses one instead.
    """

    def __init__(self, save_interval: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
        self._saved = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        self._saved = self.num_timesteps // self.save_interval
        if self.num_timesteps == 0:
            self._save()                          # the untrained policy, for the "beginning" replay

    def _on_step(self) -> bool:
        if self.num_timesteps // self.save_interval > self._saved:
            self._saved = self.num_timesteps // self.save_interval
            self._save()
        return True

    def _save(self) -> None:
        path = self.model.save(os.path.join(self.save_path, f"model_{self.num_timesteps}"))
        if self.verbose > 0:
            print(f"Saving model to {path}")
