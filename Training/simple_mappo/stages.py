"""Training one curriculum stage at a time - what the Stage<k> notebooks use.

The five stage notebooks train ONE model, one stage after another. Each notebook continues the run's
latest checkpoint, which carries everything the training has learned so far:

    actor and critic weights      what the fleet knows
    optimiser state (Adam)        its momentum, so learning carries on smoothly
    value normaliser              the critic's scale for returns
    timesteps, updates, stage     where the run is
    logs/monitor.csv              every battle so far - also how the promotion window carries over

When a stage's promotion requirement is met, its notebook stops and saves; the model then carries
the next stage, and the next notebook continues from it.
"""

from __future__ import annotations

import os

from .mappo import MAPPO


def continue_training(env, stage: int, run_dir: str, log_dir: str, storage=None, force: bool = False,
                      **hyperparameters) -> MAPPO:
    """The model a stage notebook trains on `stage`.

    It is the run's latest checkpoint - from the team folder (storage) if there is one, else from
    run_dir on this machine - with all its training memory. Only stage 0 may start a new model.
    If the run is on another stage, this says which notebook to use; force=True trains this stage
    with the model anyway (to jump ahead, or to go back and train an earlier stage more).
    """
    latest = storage.pull(run_dir) if storage else None
    local = os.path.join(run_dir, "models", "model_latest.pt")
    if latest is None and os.path.exists(local):
        latest = local
    name = env.stages[stage]["name"]

    if latest is None:
        if stage != 0 and not force:
            raise RuntimeError(f"there is no checkpoint to continue yet - train '{env.stages[0]['name']}' first "
                               f"(the Stage0 notebook), or set FORCE_STAGE = True to start a new model on '{name}'")
        env.set_stage(stage)
        print(f"no checkpoint yet - starting a new model on '{name}'")
        return MAPPO(env, log_dir=log_dir, **hyperparameters)

    # everything the previous sessions learned; the hyperparameters are the run's own (same method)
    model = MAPPO.load(latest, env=env, log_dir=log_dir, verbose=1)
    at = env.stage
    if at == stage:
        print(f"continuing '{name}' at {model.num_timesteps:,} timesteps, update {model.n_updates}")
    elif not force:
        later = at > stage
        raise RuntimeError(f"the run is on '{env.stages[at]['name']}' ({model.num_timesteps:,} timesteps) - "
                           + (f"it already passed '{name}'. Open the Stage{at} notebook, or set FORCE_STAGE = True "
                              f"to train '{name}' again with this model" if later else
                              f"it has not reached '{name}' yet. Finish the Stage{at} notebook first, or set "
                              f"FORCE_STAGE = True to jump ahead with this model"))
    else:
        print(f"FORCE_STAGE: the model was on '{env.stages[at]['name']}' - now training it on '{name}' "
              f"({model.num_timesteps:,} timesteps so far)")
        env.set_stage(stage)
    return model
