
import sys
import os
# import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv, DummyVecEnv_ra

"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            if all_args.environment_name =="resource_allocation":
                import envs.resource_allocation as environment
                scenario = environment.load(all_args.scenario_name + ".py").Scenario()
                world = scenario.make_world()
            if all_args.environment_name =="mpe":
                import envs.mpe as environment
                scenario = environment.load(all_args.scenario_name + ".py").Scenario()
                world = scenario.make_world()
            # load scenario from script

            # scenario = Scenario()
            if all_args.environment_name == "resource_allocation":
                from envs.resource_allocation.environment import EnvCore
                env = EnvCore(world, scenario.reset_world, scenario.reward, scenario.observation)
            if all_args.environment_name == "mpe":
                from envs.mpe.environment import MultiAgentEnv
                env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
            if all_args.environment_name == "ReplenishmentEnv":
                from envs.ReplenishmentEnv.replenishment_env import ReplenishmentEnv
                if all_args.scenario_name == "4 warehouses":
                    config_path = '../envs/ReplenishmentEnv/config/demo3.yml'
                if all_args.scenario_name == "9 warehouses":
                    config_path = '../envs/ReplenishmentEnv/config/demo9.yml'
                env = ReplenishmentEnv(config_path)

                # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env
    if all_args.environment_name == "mpe" or all_args.environment_name == "ReplenishmentEnv":
        return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    elif all_args.environment_name == "resource_allocation":
        return DummyVecEnv_ra([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            if all_args.environment_name == "resource_allocation":
                import envs.resource_allocation as environment
            if all_args.environment_name == "mpe":
                import envs.mpe as environment
            # load scenario from script
            scenario = environment.load(all_args.scenario_name + ".py").Scenario()

            world = scenario.make_world()
            # scenario = Scenario()
            if all_args.environment_name == "resource_allocation":
                from envs.resource_allocation.environment import EnvCore
                env = EnvCore(world, scenario.reset_world, scenario.reward, scenario.observation)
            if all_args.environment_name == "mpe":
                from envs.mpe.environment import MultiAgentEnv
                env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    if all_args.scenario_name == "graph_based1":
        num_agents = 9
        parser.add_argument("--episode_length", type=int, default=10, help="Max length for any episode")
    if all_args.scenario_name == "graph_based100":
        num_agents = 20
        parser.add_argument("--episode_length", type=int, default=10, help="Max length for any episode")
    if all_args.scenario_name == "encircle":
        num_agents = 10
        parser.add_argument("--episode_length", type=int, default=50, help="Max length for any episode")
    if all_args.scenario_name == "encircle20":
        num_agents = 20
        parser.add_argument("--episode_length", type=int, default=50, help="Max length for any episode")
    if all_args.scenario_name == "4 warehouses":
        num_agents = 4
        parser.add_argument("--episode_length", type=int, default=100, help="Max length for any episode")
    if all_args.scenario_name == "9 warehouses":
        num_agents = 9
        parser.add_argument("--episode_length", type=int, default=100, help="Max length for any episode")
    learn_index = []
    for i in range(num_agents):
        if i == 0:
            learn_index.append([1, 2, num_agents - 2, num_agents - 1])
        elif i == (num_agents - 2):
            learn_index.append([0, num_agents - 4, num_agents - 3, num_agents - 1])
        elif i == (num_agents - 1):
            learn_index.append([0, num_agents - 2])
        elif i % 2 == 1:
            learn_index.append([i - 1, i + 1])
        else:
            learn_index.append([i - 2, i - 1, i + 1, i + 2])
    parser.add_argument(
        "--graph_based1_learn_index",
        type=list,
        default=[[1, 6, 7], [7], [3, 4], [2, 4], [2, 3], [1, 2, 3, 4, 7, 8], [0, 1, 7],
                 [1], [1, 2, 3, 4, 5, 7]],
        help="learn_index get from learning graph.",
    )
    parser.add_argument(
        "--graph_based100_learn_index",
        type=list,
        default=learn_index,
        help="learn_index get from learning graph.",
    )
    all_args = parser.parse_known_args(args)[0]
    if all_args.scenario_name == "graph_based1":
        all_args.learn_index = all_args.graph_based1_learn_index
    if all_args.scenario_name == "graph_based100" or all_args.scenario_name == "encircle" \
            or all_args.scenario_name == "encircle20" :
        all_args.learn_index = all_args.graph_based100_learn_index
    if all_args.scenario_name == "4 warehouses":
        all_args.learn_index = [[1], [], [6], [8], [1, 5, 6], [1, 6], [], [3, 6, 8], []]
        # all_args.learn_index_o = [[1, 2, 3, 4, 5, 6, 7, 8], [0, 2, 3, 4, 5, 6, 7, 8], [0, 1, 3, 4, 5, 6, 7, 8], [0, 1, 2, 4, 5, 6, 7, 8],
        #                           [0, 1, 2, 3, 5, 6, 7, 8], [0, 1, 2, 3, 4, 6, 7, 8], [0, 1, 2, 3, 4, 5, 7, 8], [0, 1, 2, 3, 4, 5, 6, 8], [0, 1, 2, 3, 4, 5, 6, 7]]
    if all_args.scenario_name == "4 warehouses":
        all_args.learn_index = [[1, 2, 3], [2, 3], [3], []]
        # all_args.learn_index_o = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    all_args.num_agents = num_agents
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    # else:
    #     raise NotImplementedError

    # assert (
    #     all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    # ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.algorithm_name == "mappo":
        from runner.mappo.env_runner import EnvRunner as Runner
    if all_args.algorithm_name == "gra":
        from runner.gra.env_runner import EnvRunner as Runner
    if all_args.algorithm_name == "maddpg":
        from runner.mlp.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
