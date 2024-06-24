from copy import copy
from typing import Optional

import numpy as np
import wandb
from gymnasium import spaces
from typing_extensions import override
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper
from momaland.utils.conversions import mo_parallel_to_aec
from momaland.utils.env import MOParallelEnv

from src.env.ExperienceGrid import ExperienceGrid
from src.env.Technician import Technician
from src.env.TicketGenerator import TicketGeneratorFactory
from src.nn.feature_extractors.TicketFeaturizer import TicketFeaturizerFactory


# noinspection PyMethodMayBeStatic

def raw_env(*args, **kwargs):
    """Returns the environment in `Parallel` format.

    Args:
        **kwargs: keyword args to forward to create the `TechnicianDispatching` environment.

    Returns:
        A raw env.
    """
    return TechnicianDispatchingBase(*args, **kwargs)


def env(*args, **kwargs):
    """Returns the wrapped environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A wrapped AEC env.
    """
    env = raw_env(*args, **kwargs)
    env = mo_parallel_to_aec(env)
    env = AssertOutOfBoundsWrapper(env)
    return env


def parallel_env(*args, **kwargs):
    """Returns the wrapped env in `parallel` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A parallel env.
    """
    env = raw_env(*args, **kwargs)
    return env


class TechnicianDispatchingBase(MOParallelEnv):
    """
    Technician Dispatching Environment base class.

    The main API methods of this class are:
    - step
    - reset
    - render
    - close

    Further variations of the environment can be created by subclassing this class as long as the main API methods
    are untouched.
    """

    metadata = {
        'render_modes': ['human', 'cli'],
        'obsvervation_types': ['array', 'graph_files'],
        'is_parallelizable': True,
    }

    spec = {
        "id": "TechnicianDispatchingBase-v0",
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            num_technicians: Optional[int] = 5,
            technicians_history_horizon: Optional[int] = 100,
            num_experience_initial_seeds: Optional[int] = 5,
            ticket_generator="random_embedded",
            ticket_generator_config: Optional[dict] = None,
            experience_propagation_var_scale: Optional[float] = 0.05,
            experience_decay_rate: Optional[float] = 0.01,
            grid_size: Optional[int] = 100,
            gini_index_horizon: Optional[int] = 100,
            max_timesteps: Optional[int] = 1000,
            log: Optional[bool] = True,
            initial_random_seed: Optional[int] = 42,
            ticket_embedding_shape: Optional[int] = 2,
            featurizer_type: Optional[str] = "identity",
            transmission_factor: Optional[float] = 0.1,
            rewards: Optional[list] = None,
            proportion_reward_sigma: Optional[float] = 0.1,
    ):

        """
        Initialize the Technician Dispatching Environment.

        :param render_mode: (str) The render mode to use. Default: None.
        :param num_technicians: (int) The number of technicians to initialize. Default: 10. Only used if technicians is
            None.
        :param technicians_history_horizon: (int) The number of timesteps to consider for the technician history.
            Default: 100.
        :param num_experience_initial_seeds: (int) The number of initial experience seeds to generate (which will then
        be used to generate the initial experience of the technician)
        . Default: 5.
        :param ticket_generator: (str) The ticket generator to use. Default: "embedded_random".
        :param ticket_generator_config: (dict) The ticket generator configuration. Default: None. Only used if the
            ticket_generator is None.
        :param experience_propagation_var_scale: (float) The scale of the var (% of the range of the grid) for the
            experience propagation hyperbell curve. Default: 0.01.
        :param experience_decay_rate: (float) The technician experience decay rate. Default: 0.01.
        :param grid_size: (int) The number of points per axis in the grid. Default: 100.
        :param gini_index_horizon: (int) The number of timesteps to consider for the gini index. Default: 100.
        :param max_timesteps: (int) The maximum number of timesteps for an episode. Default: 1000.
        :param log: (bool) Whether to log the environment using wandb. Default: False.
        :param initial_random_seed: (int) The initial random seed for reproducibility. Default: 42.
        :param ticket_embedding_shape: (int) The shape of the ticket embedding. Default: 2.
        :param featurizer_type: (str) The ticket featurizer to use. Default: "identity".
        :param transmission_factor: (float) The transmission factor for the experience grid. Default: 0.1.
        :param rewards: (list) The rewards configuration. Default: None.
        :param proportion_reward_sigma: (float) The sigma for the proportion reward. Default: 0.1.
        :return: The Technician Dispatching Environment object.
        """
        # Save the randomness state prior to initialisation
        self.random_seed = np.random.get_state()
        # Set the random seed for reproducibility of initialisation, especially for the technicians
        self.initial_random_seed = initial_random_seed
        np.random.seed(self.initial_random_seed)

        # Ticket featurizer initialization
        self.ticket_embedding_shape = ticket_embedding_shape
        featurizer_factory = TicketFeaturizerFactory()
        self.ticket_featurizer = featurizer_factory.create_ticket_featurizer(featurizer_type=featurizer_type,
                                                                             ticket_embedding_shape=
                                                                             ticket_embedding_shape)

        # Ticket generator initialization
        original_config = ticket_generator_config if ticket_generator_config else {}
        ticket_generator_config = original_config | self.ticket_featurizer.get_featurizer_params()
        ticket_generator_factory = TicketGeneratorFactory()
        self.ticket_generator = ticket_generator_factory.create_ticket_generator(ticket_generator,
                                                                                 **ticket_generator_config)

        # Experience grid shared parameters initialization
        self.experience_propagation_var_scale = experience_propagation_var_scale
        self.experience_decay_rate = experience_decay_rate
        self.grid_size = grid_size
        ExperienceGrid.propagation_sigma = experience_propagation_var_scale * self.ticket_featurizer.value_range_len
        ExperienceGrid.transmission_factor = transmission_factor
        ExperienceGrid.ticket_featurizer = self.ticket_featurizer
        ExperienceGrid.ticket_embedding_shape = ticket_embedding_shape
        ExperienceGrid.grid_size = grid_size
        ExperienceGrid.grid_cell_volume = (self.ticket_featurizer.value_range_len / grid_size) * ticket_embedding_shape
        ExperienceGrid.propagation_threshold = float(np.log1p(0.001))  # Equals to a tenth of an experience point
        ExperienceGrid.feature_max = self.ticket_featurizer.max_value
        ExperienceGrid.feature_min = self.ticket_featurizer.min_value

        # Technicians initialization
        initial_experience_seeds = np.random.uniform(0, 1, (num_technicians, num_experience_initial_seeds,
                                                            self.ticket_embedding_shape))

        technicians_history_horizons = np.array([technicians_history_horizon] * num_technicians)
        self.technicians = self._init_technicians(
            num_technicians=num_technicians,
            technicians_history_horizons=technicians_history_horizons,
            initial_experience_seeds=initial_experience_seeds
        )
        self.agents_names = np.array(list(self.technicians.keys()))

        # Initialize the env state
        self.timestep = 0
        self.agents = []

        # Rewards initialization
        self._rewards = rewards
        self.proportion_reward_sigma = proportion_reward_sigma
        if self._rewards is None:
            self._rewards = ["experience_hypervolume", "proportion_tickets_treated", "max_experience"]

        # Spaces initialization
        self.action_spaces = dict(zip(self.agents_names, [self._action_space() for _ in self.agents_names]))
        self.observation_spaces = dict(zip(self.agents_names, [self._observation_space() for _ in self.agents_names]))
        self.reward_spaces = dict(zip(self.agents_names, [self._reward_space() for _ in self.agents_names]))

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self._init_human_render()
        elif self.render_mode == 'cli':
            self._init_cli_render()
        elif self.render_mode == 'wandb_info':
            self._init_wandb_info()
        elif not self.render_mode:
            print("No render mode selected")

        self.actions_occurrences = {action: 0 for action in range(3)}

        self.max_timesteps = max_timesteps
        self.gini_index_horizon = gini_index_horizon

        self.log = log
        if self.log:
            self.parameters_logged = False

        self.possible_agents = self.agents_names.tolist()

        self.previous_ticket = {agent: self.ticket_generator.blank_ticket() for agent in self.agents_names}

        self.current_tech_features_size = self.technicians[self.agents_names[0]].feature_size
        self.previous_ticket_features_size = self.ticket_generator.num_ticket_features
        self.other_tech_features_size = len(self.agents_names) * self.current_tech_features_size
        self.current_ticket_features_size = self.ticket_generator.num_ticket_features
        self.actor_features_sizes = {
            "current_tech_features_size": self.current_tech_features_size,
            "previous_ticket_features_size": self.previous_ticket_features_size,
            "other_tech_features_size": self.other_tech_features_size,
            "current_ticket_features_size": self.current_ticket_features_size
        }
        self.critic_features_sizes = {
            "current_tech_features_size": self.current_tech_features_size,
            "previous_ticket_features_size": self.previous_ticket_features_size,
            "other_features_size": self.current_ticket_features_size,
            "number_of_agents": len(self.agents_names)
        }

        # Restore the randomness state
        np.random.set_state(self.random_seed)
        self.supervisor = None
        self.supervised = None
        self.truncations = {agent: False for agent in self.agents_names}
        self.terminations = {agent: False for agent in self.agents_names}

    def _init_technicians(self, num_technicians, technicians_history_horizons, initial_experience_seeds):
        technicians = dict()
        for i in range(num_technicians):
            technicians[f"Technician_{i}"] = Technician(technician_history_horizon=technicians_history_horizons[i],
                                                        initial_experience_seeds=initial_experience_seeds[i])
        return technicians

    def _reset_technicians(self):
        for technician in self.technicians.values():
            technician.reset()

    def _action_space(self):
        """
        Define the action space for an agent

        In this case, each agent can choose to:
        - 0 : Don't take the ticket
        - 1 : Take the ticket (1 agent exactly)
        - 2 : Take the ticket as a supervisor (1 agent max)
        :return: The action space for an agent
        """

        return spaces.Discrete(3)

    def _observation_space(self):
        """
        Define the observation space for an agent
        In this case, the observation space is a vector built as follows:
        - The features of the technician corresponding to the agent
        - The features of the ticket being dispatched
        - The features of the other technicians
        :return: The observation space for an agent
        """

        num_features_technician = self.technicians[self.agents_names[0]].feature_size
        num_features_ticket = self.ticket_generator.num_ticket_features
        num_technicians = len(self.technicians)
        num_features = num_features_technician + 2 * num_features_ticket + (
                    num_technicians - 1) * num_features_technician
        # TODO: Define the boundaries of the observation space more precisely (sample won't be used anyway other than
        #  for test purposes)
        return spaces.Box(low=-1000, high=1000, shape=(num_features,), dtype=np.float32)

    def _reward_space(self):
        """
        Define the reward space for an agent
        In this case, the reward space is a vector built as follows:
        - 0 : Experience hypervolume aggregate
        - 1 : Gini index of the ticket distribution
        :return: The reward space for an agent
        """

        return spaces.Box(low=0, high=np.inf, shape=(len(self._rewards),), dtype=np.float32)

    def action_space(self, agent_name):
        """Returns the action space for the given agent."""
        return self.action_spaces[agent_name]

    def sample_legal_action_set(self):
        """
        Sample a legal action set for each agent
        :return: dict - a dictionary of legal actions for each agent
        """
        proportions = None
        # Biased proportion of agents taking the ticket
        # n_agents = len(self.agents_names)
        # proportions = np.array([5] + [1] * (n_agents - 1))
        # proportions = proportions / np.sum(proportions)
        # Choose a random agent to take the ticket
        agent = np.random.choice(self.agents_names, p=proportions if proportions is not None else None)
        # Choose if there's a supervisor
        with_super = np.random.choice([True, False])
        with_super = False
        if with_super:
            supervisor = np.random.choice([ag for ag in self.agents_names if ag != agent])
            actions = {agent: 1, supervisor: 2}
        else:
            actions = {agent: 1}
        # Fill the rest of the actions with 0
        for agent in self.agents_names:
            if agent not in actions:
                actions[agent] = 0

        return actions

    def observation_space(self, agent_name):
        """Returns the observation space for the given agent."""
        return self.observation_spaces[agent_name]

    def reward_space(self, agent_name):
        """Returns the reward space for the given agent."""
        return self.reward_spaces[agent_name]

    def _compute_agent_obs(self, agent_name):
        """
        Compute the observation for the given agent. This method can be overridden in subclasses to provide a
        different observation format.
        :param agent_name:
        :return: the observation for the given agent, here a numpy array formatted as follows:
        - The features of the technician corresponding to the agent
        - The features of the ticket being dispatched
        - The features of the other technicians
        """
        agent = self.technicians[agent_name]
        ticket = self.ticket_generator.current_ticket
        previous_ticket = self.previous_ticket[agent_name]
        other_agents = [self.technicians[other_agent] for other_agent in self.agents_names if other_agent != agent_name]

        obs = np.concatenate([agent.get_features(ticket), previous_ticket.get_features(), ticket.get_features(),
                              np.concatenate([other_agent.get_features(ticket) for other_agent in other_agents])])

        return obs

    @override
    def state(self):
        # Returns the global state of the environment
        # For each technician, returns the features of the technician and the features of the ticket they treated last
        # and at the end the features of the ticket currently being dispatched

        state = []
        for agent in self.agents_names:
            state.append(self.technicians[agent].get_features(self.ticket_generator.current_ticket))
            state.append(self.previous_ticket[agent].get_features())
        state.append(self.ticket_generator.current_ticket.get_features())

        return np.concatenate(state)

    def _compute_agent_reward(self, agent_name):
        """
        Compute the reward for the given agent. This method can be overridden in subclasses to provide a
        different reward format.
        :param agent_name:
        :return: the reward for the given agent, here a numpy array formatted as follows:
        - 0 : Experience hypervolume aggregate
        - 1 : Gini index of the ticket distribution
        """
        rewards = []
        for reward in self._rewards:
            if reward == "experience_hypervolume":
                rewards.append(self.technicians[agent_name].experience_grid.get_hypervolume())
            elif reward == "gini_index":
                rewards.append(1 - self.technicians[agent_name].get_gini_index())
            elif reward == "proportion_tickets_treated":
                propotion = (self.technicians[agent_name].get_proportion_of_tickets_treated())
                ideal_proportion = 1 / len(self.technicians)
                rewards.append(np.exp(-((propotion - ideal_proportion) ** 2) / (
                        2 * (self.proportion_reward_sigma ** 2))))
            elif reward == "max_experience":
                rewards.append(self.technicians[agent_name].experience_grid.get_max_experience())
            else:
                raise ValueError(f"Unknown reward type: {reward}")

        return np.array(rewards)

    def _compute_obs(self):
        obs = dict()

        for agent in self.agents_names:
            obs[agent] = self._compute_agent_obs(agent)

        return obs

    def _compute_reward(self):
        reward = dict()

        for agent in self.agents_names:
            reward[agent] = self._compute_agent_reward(agent)

        return reward

    def _compute_terminated(self):
        """
        For our environment, the only termination condition is a conflict of actions between agents
        (i.e. two agents taking the same ticket, or two agents trying to supervise the same ticket,
        or no agent taking the ticket)
        """

        if self.actions_occurrences[1] != 1:
            # self.agents = []
            return {agent: True for agent in self.agents_names}
        elif self.actions_occurrences[2] > 1:
            # self.agents = []
            return {agent: True for agent in self.agents_names}
        else:
            return {agent: False for agent in self.agents_names}

    def _compute_truncation(self):
        if self.timestep:
            # self.agents = []
            return {agent: self.timestep >= self.max_timesteps for agent in self.agents_names}
        else:
            return {agent: False for agent in self.agents_names}

    def _compute_infos(self):
        if self.actions_occurrences[1] == 0:
            return {agent: "No agents took the ticket" for agent in self.agents_names}
        elif self.actions_occurrences[1] > 1:
            return {agent: "Two agents took the ticket" for agent in self.agents_names}
        elif self.actions_occurrences[2] > 1:
            return {agent: "Two agents supervised the ticket" for agent in self.agents_names}
        else:
            return {agent: "No issue" for agent in self.agents_names}

    def _render_human(self):
        raise NotImplementedError("Human render not implemented")

    def _render_cli(self):
        raise NotImplementedError("CLI render not implemented")

    def _close_human(self):
        raise NotImplementedError("Human close not implemented")

    def _close_cli(self):
        raise NotImplementedError("CLI close not implemented")

    # PettingZoo API methods

    @override
    def reset(self, seed=None, return_info=True, options=None, hard=False):
        """
        Reset the environment. This method should be called at the beginning of a new episode.
        :return: The initial observation of the environment.
        """
        self.timestep = 0
        self.agents = copy(self.possible_agents)
        self.actions_occurrences = {action: 0 for action in range(3)}
        self.ticket_generator.reset()
        self.supervisor = None
        self.supervised = None
        if not hard:
            # Save random state
            saved_state = np.random.get_state()
            np.random.seed(self.initial_random_seed)
            self._reset_technicians()
            # Restore random state
            np.random.set_state(saved_state)
        else:
            self._reset_technicians()

        observation = self._compute_obs()
        infos = {agent: "Env reset" for agent in self.agents_names}

        self.terminations = {agent: False for agent in self.agents_names}
        self.truncations = {agent: False for agent in self.agents_names}

        if self.render_mode is not None:
            self.render()

        if self.log and not self.parameters_logged:
            self._init_logging()
            self.parameters_logged = True

        return observation, infos

    @override
    def step(self, actions):

        self.supervisor = None
        self.supervised = None
        self.actions_occurrences = {action: 0 for action in range(3)}
        for agent, action in actions.items():
            self._state_step_single(agent, action)

        if self.render_mode is not None:
            rendering = self.render()
        else:
            rendering = None

        observations = self._compute_obs()
        rewards = self._compute_reward()
        terminations = self._compute_terminated()
        self.terminations = terminations
        truncations = self._compute_truncation()
        self.truncations = truncations
        infos = self._compute_infos()

        if any(terminations.values()):
            # Each agent receives a reward of [-1 for each reward] if the episode is terminated
            rewards = {agent: [-1 for _ in range(len(self._rewards))] for agent in self.agents_names}

        if self.log:
            self._log_step(observations, rewards, terminations, truncations, infos, rendering)

        # Save the previous ticket
        for agent in self.agents_names:
            self.previous_ticket[agent] = self.ticket_generator.current_ticket if actions[agent] == 1 else (
                self.ticket_generator.blank_ticket())

        # if self.terminations[self.agents_names[0]]:
        #     self.agents = []

        return observations, rewards, terminations, truncations, infos

    @override
    def render(self):
        if self.render_mode:
            if self.render_mode == 'human':
                self._render_human()
            elif self.render_mode == 'cli':
                self._render_cli()
            else:
                raise NotImplementedError(f"Render mode {self.render_mode} not implemented")

    @override
    def close(self):
        if self.render_mode == 'human':
            self._close_human()
        elif self.render_mode == 'cli':
            self._close_cli()

    def _log_step(self, observations, rewards, terminations, truncations, infos, rendering):
        step_info = {
            "observations": observations,
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
            "infos": infos,
            "rendering": rendering
        }
        wandb.log({"step": step_info})
        print("Logged step")

    def _init_logging(self):
        parameters = {
            "num_technicians": len(self.technicians),
            "technicians_history_horizon": self.gini_index_horizon,
            "num_experience_initial_seeds": self.technicians[0].num_experience_initial_seeds,
            "ticket_generator": self.ticket_generator,
            "experience_propagation_var_scale": self.experience_propagation_var_scale,
            "experience_decay_rate": self.experience_decay_rate,
            "grid_size": self.grid_size,
            "gini_index_horizon": self.gini_index_horizon,
            "max_timesteps": self.max_timesteps,
            "log": self.log,
            "initial_random_seed": self.initial_random_seed,
            "ticket_embedding_shape": self.ticket_embedding_shape,
        }
        wandb.log({"env_parameters": parameters})
        print("Logged environment parameters")

    def _render_experience_grids(self, technician: Optional[Technician] = None):
        if technician:
            technician.experience_grid.render()
        else:
            for technician in self.technicians.values():
                technician.experience_grid.render()

    def _state_step(self, actions):
        """
        Update the environment state according to the actions taken by the agents.
        :param actions: dict - actions taken by the agents
        :return:
        """
        # Reset the actions occurrences
        self.actions_occurrences = {action: 0 for action in range(3)}
        # Update the actions occurrences
        for action in actions.values():
            action = int(action)
            self.actions_occurrences[action] += 1
        # Check if the actions are valid
        terminated = self._compute_terminated()
        if any(terminated.values()):
            return
        # Update the technicians experience grids
        # If there's a supervisor, the ticket is supervised
        if 2 in actions.values():
            supervisor = [agent for agent, action in actions.items() if action == 2][0]
            for technician, action in actions.items():
                if action == 1:
                    self.technicians[technician].add_ticket_experience(self.ticket_generator.current_ticket,
                                                                       supervisor=self.technicians[supervisor]
                                                                       if action == 2 else None)
                    self.technicians[technician].treated_last_ticket()
                else:
                    self.technicians[technician].did_not_treat_last_ticket()
        else:
            for technician, action in actions.items():
                if action == 1:
                    self.technicians[technician].add_ticket_experience(self.ticket_generator.current_ticket)
                    self.technicians[technician].treated_last_ticket()
                else:
                    self.technicians[technician].did_not_treat_last_ticket()

        self.ticket_generator.step()
        self.timestep += 1

        return

    def _state_step_single(self, agent, action):
        """
        Update the environment state according to the action taken by the agent.
        :param agent: str - the agent taking the action
        :param action: int - the action taken by the agent
        :return:
        """
        # Update the actions occurrences
        action = int(action)
        self.actions_occurrences[action] += 1
        # Update the technicians experience grids
        if action == 1:
            self.technicians[agent].add_ticket_experience(self.ticket_generator.current_ticket)
            self.technicians[agent].treated_last_ticket()

            if self.supervisor:
                self.technicians[agent].experience_grid.add_supervisor_experience(
                    ticket=self.ticket_generator.current_ticket, supervisor=self.technicians[self.supervisor])
            else:
                self.supervised = agent
        elif action == 2:
            if not self.supervisor:
                self.supervisor = agent
                if self.supervised:
                    self.technicians[self.supervised].experience_grid.add_supervisor_experience(
                        ticket=self.ticket_generator.current_ticket, supervisor=self.technicians[self.supervisor])

        self.ticket_generator.step()
        self.timestep += 1
