"""A readable MAPPO with a fleet commander for the naval simulation - the version to learn from.

    rewards.py    what each ship is paid for (edit this to change the fleet's goals)
    curriculum.py the stages, from shooting at a target to random fleet battles, and when to move on
    env.py        NavalEnv: n_envs Unity battles behind a reset() / step() interface
    networks.py   the captain (every ship), the fleet commander and the centralised critic
    mappo.py      MAPPO: the PPO update for captains, commander and critic
    league.py     past copies of the fleet to practise against (self-play) and Elo ratings
    imitation.py  record the Elite rule AI and clone it - the warm start before RL
    callbacks.py  BaseCallback, SaveOnIntervalCallback, BestModelCallback, as in Stable-Baselines3
    utils.py      Monitor, plots, evaluation and battle replays
    team.py       TeamStorage: one shared S3 folder per run, so teammates continue each other's training
    stages.py     continue_training: one model trained stage by stage (the Stage<k> notebooks)

The first research pipeline (naval_rl/) is unchanged; this package only borrows its TCP connection
to Unity.
"""

from .callbacks import BaseCallback, BestModelCallback, CallbackList, SaveOnIntervalCallback
from .curriculum import DEFAULT_CURRICULUM, CurriculumCallback, load_curriculum
from .env import NavalEnv, find_unity_binary
from .imitation import (behaviour_cloning, collect_expert, load_dataset, merge_datasets, plot_imitation,
                        save_dataset, save_imitation)
from .league import OpponentPool
from .mappo import MAPPO
from .networks import Captain, CentralCritic, Commander, FleetPolicy
from .rewards import REWARD_GROUPS, REWARD_WEIGHTS, compute_rewards
from .stages import continue_training
from .team import TeamStorage, TeamStorageBusy, TeamSyncCallback, training_status
from .utils import (Monitor, VideoCallback, evaluate, load_results, plot_curriculum, plot_progress, plot_results,
                    plot_reward_terms, record_battle, record_unity_video, show_videos, ts2xy, view)

__all__ = ["BaseCallback", "BestModelCallback", "CallbackList", "SaveOnIntervalCallback", "DEFAULT_CURRICULUM",
           "CurriculumCallback", "load_curriculum", "NavalEnv", "find_unity_binary", "behaviour_cloning",
           "collect_expert", "load_dataset", "merge_datasets", "plot_imitation", "save_dataset", "save_imitation",
           "OpponentPool", "MAPPO", "Captain", "CentralCritic", "Commander", "FleetPolicy", "REWARD_GROUPS",
           "REWARD_WEIGHTS", "compute_rewards", "Monitor", "evaluate", "load_results", "plot_curriculum",
           "plot_progress", "plot_results", "plot_reward_terms", "record_battle", "record_unity_video",
           "VideoCallback", "show_videos", "ts2xy", "view", "TeamStorage", "TeamStorageBusy", "TeamSyncCallback",
           "training_status", "continue_training"]
