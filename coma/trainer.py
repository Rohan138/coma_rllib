import logging
from typing import List
from coma_smac.model import COMATorchModel
import numpy as np

from ray.rllib import SampleBatch
from ray.rllib.utils.exploration import EpsilonGreedy
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.trainer import with_common_config, COMMON_CONFIG
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.common import _get_shared_metrics, NUM_TARGET_UPDATES
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ConcatBatches, ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep

from coma.policy import COMATorchPolicy, EpsilonCOMA

logger = logging.getLogger(__name__)

LAST_ITER_UPDATE = "last_iter_update"
NUM_ITER_LOOP = "num_iter_loop"


# yapf: disable
# --__sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Settings for each individual policy ===
    # ID of the agent controlled by this policy
    "agent_id": None,
    # If false the algorithm changes to Independent Actor Critic.
    "use_coma": True,

    # Config
    "log_level": "DEBUG",
    "framework": "torch",
    'num_workers': 4,
    'num_envs_per_worker': 4,
    'num_gpus': 0,

    # Training
    "actor_lr": 5e-4,
    "critic_lr": 5e-4,
    "lambda": .9,
    "gamma": .95,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 100.0,
    'target_network_update_freq': 1,
    "rollout_fragment_length": 100,
    "train_batch_size": 1024,
    "timesteps_per_iteration": 0,
    "model": {
        "custom_model": COMATorchModel,
        "custom_model_config": {
            "actor_hiddens" : [64, 64],
            "actor_activation": "ReLU",
            "critic_hiddens": [128, 128],
            "critic_activation": "ReLU",
        }
    },
    "exploration_config": {
        "type": EpsilonCOMA,
        "initial_epsilon": 0.5,
        "final_epsilon": 0.01,
        "epsilon_timesteps": int(100000)
    },

    "training_intensity": None,
    # Force lockstep replay mode for MADDPG.
    "multiagent": merge_dicts(COMMON_CONFIG["multiagent"], {
        "replay_mode": "lockstep",
    }),

    "reward_range": None,
    "entropy_coeff": 0.0,
    "tau": 1,
    "n_step": 1,
    "evaluation_interval": None,
    "evaluation_num_episodes": 10,
    "evaluation_config": {
        "explore": False,
    },
})


class UpdateTargetNetwork:
    """Periodically call policy.update_target() on all trainable policies.

    This should be used with the .for_each() operator after training step
    has been taken.

    Examples:
        >>> train_op = UpdateTargetNetwork(...).for_each(TrainOneStep(...))
        >>> update_op = train_op.for_each(
        ...     UpdateTargetNetwork(workers, target_update_freq=500))
        >>> print(next(update_op))
        None

    Updates the LAST_ITER_UPDATE, NUM_ITER_LOOP and NUM_TARGET_UPDATES counters
    in the local iterator context. The value of the last update counter is used
    to track when we should update the target next.
    """

    def __init__(self,
                 workers,
                 target_update_freq,
                 policies=frozenset([])):
        self.workers = workers
        self.target_update_freq = target_update_freq
        self.policies = (policies or workers.local_worker().policies_to_train)
        self.metric = NUM_ITER_LOOP

    def __call__(self, _):
        metrics = _get_shared_metrics()
        metrics.counters[self.metric] += 1
        cur_ts = metrics.counters[self.metric]
        last_update = metrics.counters[LAST_ITER_UPDATE]
        if cur_ts - last_update >= self.target_update_freq:
            to_update = self.policies
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, p_id: p_id in to_update and p.update_target())
            metrics.counters[NUM_TARGET_UPDATES] += 1
            metrics.counters[LAST_ITER_UPDATE] = cur_ts


def execution_plan(workers, config):
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    train_op = rollouts \
        .combine(ConcatBatches(config["train_batch_size"], \
            count_steps_by=config["multiagent"]["count_steps_by"])) \
        .for_each(TrainOneStep(workers)) \
        .for_each(UpdateTargetNetwork(
        workers, config['target_network_update_freq']))

    return StandardMetricsReporting(train_op, workers, config)


COMATrainer = build_trainer(
    name="COMA",
    default_config=DEFAULT_CONFIG,
    default_policy=COMATorchPolicy,
    execution_plan=execution_plan)
