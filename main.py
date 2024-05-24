import os
import sys

from dreamer.envs.isaacgym import get_env

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.envs.envs import get_env_infos, make_atari_env, make_dmc_env
from dreamer.utils.utils import ConsoleOutputWrapper, get_base_directory, load_config


def main(args):
    config = load_config(args.config)

    if config.environment.benchmark == "atari":
        env = make_atari_env(
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            height=config.environment.height,
            width=config.environment.width,
            skip_frame=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    elif config.environment.benchmark == "dmc":
        env = make_dmc_env(
            domain_name=config.environment.domain_name,
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            visualize_reward=config.environment.visualize_reward,
            from_pixels=config.environment.from_pixels,
            height=config.environment.height,
            width=config.environment.width,
            frame_skip=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    elif config.environment.benchmark == "isaacgym":
        env, env_cfg = get_env(args)
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    log_dir = create_log_dir(config)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    device = config.operation.device

    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
        if args.do_log:
            agent.set_up_pipeline(log_dir, config, args)
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    agent.train(env)


def create_log_dir(config):
    exp_name = datetime.now().strftime("%b%d_%H_%M_%S")
    exp_name = (
        exp_name
        + "_"
        + ((config.operation.log_dir) if args.exp_name is None else args.exp_name)
    )
    log_root = os.path.join(get_base_directory() + "/runs/")

    log_dir = os.path.join(
        log_root,
        exp_name,
    )
    if not args.do_log:
        log_dir = None

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = ConsoleOutputWrapper(os.path.join(log_dir, "log.txt"))

    return log_dir


if __name__ == "__main__":

    from legged_gym.utils import get_args

    args = get_args()
    if "legged" in args.config:
        assert args.num_envs == 1, "Only one env is supported in openai gym for now"

    # dump args as yaml
    # import yaml
    # with open("args.yaml", "w") as f:
    #     yaml.dump(vars(args), f)
    # exit(0)

    main(args)
