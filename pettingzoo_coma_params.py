import ray
from ray import tune
from ray.tune import schedulers
from ray.tune.registry import register_trainable, register_env
import supersuit as ss
import argparse
from importlib import import_module
from ray.tune import CLIReporter
import gym
from gym.spaces import Box, Discrete, MultiDiscrete, Dict
import os
import numpy as np
from coma_mpe.trainer import COMATrainer

# Optimization
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

def parse_args():
    # Environment
    parser = argparse.ArgumentParser("RLLib COMA with PettingZoo environments")

    parser.add_argument(
        "--env-type",
        choices=["mpe", "sisl", "atari", "butterfly", "classic", "magent"],
        default="mpe",
        help="The PettingZoo environment type"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="simple_spread_v2",
        help="The PettingZoo environment to use"
    )
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="torch",
        help="The DL framework specifier.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="ERROR",
        help="The log level for tune.run()")

    parser.add_argument("--max-episode-len", type=int, default=25,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000,
                        help="number of episodes")

    # Core training parameters
    parser.add_argument("--actor-lr", type=float, default=1e-4,
                        help="learning rate for actor Adam")
    parser.add_argument("--critic-lr", type=float, default=1e-2,
                        help="learning rate for critic Adam")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    parser.add_argument("--actor-fc", type=int, default=64,
                        help="number of units in the actor mlp")
    parser.add_argument("--critic-fc", type=int, default=128,
                        help="number of units in the critic mlp")
    parser.add_argument("--gru-fc", type=int, default=32,
                        help="number of units in the actor gru")
    parser.add_argument("--lam", type=float, default=0.95,
                        help="lambda for TD(lambda)")
    parser.add_argument("--rollout-fragment-length", type=int, default=1,
                        help="number of data points sampled /update /worker")
    parser.add_argument("--train-batch-size", type=int, default=256,
                        help="number of data points /update")
    parser.add_argument("--tau", type=float, default=0.95,
                        help="polyak factor")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")
    parser.add_argument("--update-freq", type=int, default=10,
                        help="Target network update frequency")

    # Checkpoint
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--local-dir", type=str, default="~/ray_results",
                        help="path to save checkpoints")
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="evaluate model every time this many iterations are completed")
    parser.add_argument("--eval-num-episodes", type=int, default=5,
                        help="Number of episodes to run for evaluation")
    parser.add_argument("--render", type=bool, default=False,
                        help="render environment for evaluation")
    parser.add_argument("--record", type=str, default=None,
                        help="path to store evaluation videos")
    return parser.parse_args()

class PettingZooSMACEnv:

    def __init__(self, _env, horizon):
        self._env = _env
        self.reset()
        self.horizon = horizon
        self.nbr_agents = len(self._env.possible_agents)
        self.agents = self._env.possible_agents
        obs_shape = self._env.observation_spaces[self.agents[0]].shape[0]
        act_shape = self._env.action_spaces[self.agents[0]].n
        self.observation_space = Dict({ \
            "obs": Box(-np.inf, np.inf, shape=(self.nbr_agents, obs_shape)),
            "state": self._env.state_space,
        })
        self.action_space = MultiDiscrete([act_shape] * self.nbr_agents)

    def _observe(self):
        state = self._env.state()
        return {"obs": self.obs, "state": state}
    
    def reset(self):
        obs = self._env.reset()
        self.obs = np.stack([o for o in obs.values()])
        return self._observe()
    
    def step(self, action_list):
        action_dict = {agent: action_list[i] for i, agent in enumerate(self.agents)}
        obs, reward, done, _ = self._env.step(action_dict)
        obs_list = [obs[agent] for agent in self.agents]
        self.obs = np.stack([o for o in obs_list])
        reward = sum(r for r in reward.values())
        done = any(d for d in done.values())
        return self._observe(), reward, done, {}

    def close(self):
        """Close the environment"""
        self._env.close()

    def __del__(self):
        self.close()


def main(args):
    ray.init()
    env_name = args.env_name
    env_str = "pettingzoo." + args.env_type + "." + env_name

    def env_creator(config):
        env = import_module(env_str)
        env = env.parallel_env(max_cycles=25)
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = PettingZooSMACEnv(env, 25)
        return env

    register_env(env_name, lambda config: env_creator(config))
    register_trainable('coma', COMATrainer)

    env = env_creator(args)
    obs_space = env.observation_space
    act_space = env.action_space
    print("observation spaces: ", obs_space)
    print("action spaces: ", act_space)

    config={
            # === Setup ===
            "framework": args.framework,
            "log_level": args.log_level,
            "env": env_name,
            "num_workers": args.num_workers,
            "num_gpus": args.num_gpus,
            "num_envs_per_worker": args.num_envs_per_worker,
            "horizon": args.max_episode_len,

            # === Policy Config ===
            # --- Model ---
            "grad_clip": 100,
            "gamma": args.gamma,
            "model": {
                "use_lstm": True,
                '_time_major': True,
                "custom_model_config": {
                    "gru_cell_size": args.gru_fc,
                    "fcnet_activation_stage1": "relu",
                    "fcnet_activation_stage2": "relu",
                    "fcnet_hiddens_stage1": [args.actor_fc,],
                    "fcnet_hiddens_stage2": [],
                    "fcnet_hiddens_critic": [args.critic_fc] * 2,
                    "fcnet_activation_critic": "relu",
                },
                "max_seq_len": args.max_episode_len,
            },

            # --- Optimization ---
            "actor_lr": tune.loguniform(1e-4, 1e-2),
            "critic_lr": tune.loguniform(1e-4, 1e-2),
            'target_network_update_freq': tune.qlograndint(1, 20, 4),
            "lambda": tune.quniform(0.90, 0.99, 0.01),
            "tau": tune.quniform(0.90, 0.99, 0.01),

            "rollout_fragment_length": args.rollout_fragment_length,
            "train_batch_size": args.train_batch_size,

    
            # === Evaluation and rendering ===
            "evaluation_interval": args.eval_freq,
            "evaluation_num_episodes": args.eval_num_episodes,
        }
    
    search_alg = TuneBOHB(metric='episode_reward_mean', mode='max')
    bohb = HyperBandForBOHB(metric='episode_reward_mean', mode='max', \
        time_attr='episodes_total', max_t=args.num_episodes)

    tune.run(
        COMATrainer,
        name="COMA",
        search_alg=search_alg,
        scheduler=bohb,
        num_samples=4,
        config=config,
        progress_reporter=CLIReporter(),
        stop={
            "episodes_total": args.num_episodes,
        },
        checkpoint_freq=args.checkpoint_freq,
        local_dir=os.path.join(args.local_dir, args.env_name),
        restore=args.restore,
        verbose = 1
    )

if __name__=='__main__':
    args = parse_args()
    main(args)