"""A small, readable MAPPO for the naval simulation - the version to learn from.

    rewards.py    what each ship is paid for (edit this to change the fleet's goals)
    curriculum.py the stages, from a 1v1 duel to endless random battles, and when to move on
    env.py        NavalEnv: n_envs Unity battles behind a reset() / step() interface
    mappo.py      the actor, the centralised critic and the PPO update
    callbacks.py  BaseCallback and SaveOnIntervalCallback, as in Stable-Baselines3
    utils.py      Monitor, plots, evaluation and battle replays
    team.py       TeamStorage: one shared S3 folder per run, so teammates continue each other's training
    stages.py     continue_training: one model trained stage by stage (the Stage<k> notebooks)

The full research pipeline (transformer policy, behaviour cloning, league self-play) lives in
naval_rl/ and is unchanged; this package only borrows its TCP connection to Unity.
"""

from .callbacks import BaseCallback, CallbackList, SaveOnIntervalCallback
from .curriculum import DEFAULT_CURRICULUM, CurriculumCallback, load_curriculum
from .env import NavalEnv, find_unity_binary
from .mappo import MAPPO
from .rewards import REWARD_WEIGHTS, compute_rewards
from .stages import continue_training
from .team import TeamStorage, TeamStorageBusy, TeamSyncCallback, training_status
from .utils import (Monitor, VideoCallback, evaluate, load_results, plot_curriculum, plot_progress, plot_results,
                    plot_reward_terms, record_battle, record_unity_video, show_videos, ts2xy, view)

__all__ = ["BaseCallback", "CallbackList", "SaveOnIntervalCallback", "DEFAULT_CURRICULUM", "CurriculumCallback",
           "load_curriculum", "NavalEnv", "find_unity_binary", "MAPPO", "REWARD_WEIGHTS", "compute_rewards",
           "Monitor", "evaluate", "load_results", "plot_curriculum", "plot_progress", "plot_results",
           "plot_reward_terms", "record_battle", "record_unity_video", "VideoCallback", "show_videos", "ts2xy", "view", "TeamStorage",
           "TeamStorageBusy", "TeamSyncCallback", "continue_training"]
