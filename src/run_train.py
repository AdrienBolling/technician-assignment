from nn.agent.momappo_v1 import main
from env.TechnicianAssignment import TechnicianDispatchingBase
import argparse
from distutils.util import strtobool
import os

rewards = ["experience_hypervolume", "proportion_tickets_treated", "max_experience"]


def parse_args():
    """Argument parsing for the algorithm."""
    # fmt: off
    parser = argparse.ArgumentParser()

    # Env and experiment arguments
    parser.add_argument("--env-id", type=str, help="MOMAland id of the environment to run (check all_modules.py)", required=True)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True
    )
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="run in debug mode")
    parser.add_argument("--save-policies", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="save the trained policies")

    # Wandb
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="log metrics to wandb")
    parser.add_argument("--wandb-project", type=str, default="MOMAland", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="openrlbenchmark", help="the wandb's entity")
    parser.add_argument(
        "--auto-tag",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, the runs will be tagged with git tags, commit, and pull request number if possible",
    )

    # Algorithm specific arguments
    parser.add_argument("--num-weights", type=int, default=10, help="the number of different weights to train on")
    parser.add_argument("--weights-generation", type=str, default="OLS", help="The method to generate the weights - 'OLS' or 'uniform'")
    parser.add_argument("--num-steps-per-epoch", type=int, default=128, help="the number of steps per epoch (higher batch size should be better)")
    parser.add_argument("--timesteps-per-weight", type=int, default=2e3,
                        help="timesteps per weight vector")
    parser.add_argument("--update-epochs", type=int, default=2, help="the number epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, default=2, help="the number of minibatches (keep small in MARL)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--gae-lambda", type=float, default=0.99,
                        help="the lambda for the generalized advantage estimation")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="the epsilon for clipping in the policy objective")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="the coefficient for the entropy bonus")
    parser.add_argument("--vf-coef", type=float, default=0.8,
                        help="the coefficient for the value function loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--actor-net-arch", type=lambda x: list(map(int, x.split(','))), default=[256, 256],
                        help="actor network architecture excluding the output layer(size=action_space)")
    parser.add_argument("--critic-net-arch", type=lambda x: list(map(int, x.split(','))), default=[256, 256],
                        help="critic network architecture excluding the output layer (size=1)")
    parser.add_argument("--activation", type=str, default="tanh",
                        help="the activation function for the neural networks")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to anneal the learning rate linearly")

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":

    args = parse_args()

    env_args = {
        "render_mode": None,
        "num_technicians": 5,
        "technicians_history_horizon": 100,
        "num_experience_initial_seeds": 5,
        "ticket_generator": "random_embedded",
        "ticket_generator_config": None,
        "experience_propagation_var_scale": 0.05,
        "experience_decay_rate": 0.01,
        "grid_size": 100,
        "gini_index_horizon": 100,
        "max_timesteps": 1000,
        "log": False,
        "initial_random_seed": 42,
        "ticket_embedding_shape": 2,
        "featurizer_type": "identity",
        "transmission_factor": 0.1,
        "rewards": rewards,
        "proportion_reward_sigma": 0.1,
    }

    main(args, env_args=env_args)
