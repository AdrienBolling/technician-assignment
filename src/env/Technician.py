from collections import deque

import numpy as np
from src.env.ExperienceGrid import ExperienceGrid
from src.env.Ticket import Ticket


class Technician:
    """
    This class defines a Technician object, which represents a technician has the following attributes:
    - id: int - technician id
    - name: str - technician name
    - experience_grid: np.ndarray - grid representing the experience of the technician
    - learning_rate: float - learning rate of the technician
    - has_treated_last_hundred_tickets: np.ndarray - array of size 100 representing whether the technician has treated
    the last 100 tickets
    """

    _instance_counter = 0

    learning_rates = None

    def __init__(
            self,
            name: str = None,
            technician_history_horizon: int = 100,
            initial_experience_seeds: np.ndarray = None,
    ):
        """
        Constructor for Technician class
        :param name: str - technician name
        :param technician_history_horizon: int - history horizon of the technician
        :param initial_experience_seeds: np.ndarray - initial experience seeds for the technician
        """
        self.id = Technician._instance_counter
        Technician._instance_counter += 1
        self.name = name if name is not None else f"John {self.id}"
        self.learning_rate = np.random.uniform(0.5, 0.95)
        self.initial_experience_seeds = initial_experience_seeds
        self.experience_grid = ExperienceGrid(self.id, self.learning_rate, self.initial_experience_seeds)
        self.techician_history_horizon = technician_history_horizon
        self.has_treated_last_hundred_tickets = deque(maxlen=int(technician_history_horizon))

        self.feature_size = 4

    def reset(self):
        """
        Reset the technician
        :return:
        """
        self.experience_grid.reset()
        self.learning_rate = np.random.uniform(0.5, 0.95)
        self.has_treated_last_hundred_tickets = deque(maxlen=int(self.techician_history_horizon))

    def get_features(self, ticket: Ticket):
        """
        Get the features of the technician for the observation
        :return: np.ndarray - features of the technician

        """
        return np.array([
            self.learning_rate,
            np.mean(self.has_treated_last_hundred_tickets) if len(self.has_treated_last_hundred_tickets) > 0 else 0.0,
            self.experience_grid.get_hypervolume(),
            self.experience_grid.get_experience(ticket),
        ]
        )

    def treated_last_ticket(self):
        self.has_treated_last_hundred_tickets.append(1)

    def did_not_treat_last_ticket(self):
        self.has_treated_last_hundred_tickets.append(0)

    def add_ticket_experience(self, ticket: Ticket, supervisor: 'Technician' = None):
        """
        Add the ticket experience to the technician
        :param ticket: Ticket - ticket object
        :param supervisor: Technician - supervisor technician if any
        :return:
        """
        self.experience_grid.add_ticket_experience(ticket, supervisor=supervisor)

    def get_hypervolume(self):
        """
        Get the experience grid of the technician
        :return: np.ndarray - experience grid of the technician
        """
        return self.experience_grid.get_hypervolume()

    def get_proportion_of_tickets_treated(self):
        """
        Get the proportion of tickets treated by the technician
        :return: float - proportion of tickets treated by the technician
        """
        return np.mean(self.has_treated_last_hundred_tickets) if len(self.has_treated_last_hundred_tickets) > 0 else 0.0

    def get_gini_index(self):
        """
        Get the Gini index of the technician
        :return: float - Gini index of the technician
        """
        raise NotImplementedError

    def __str__(self):
        return f"Technician {self.id} - {self.name}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Technician):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not self.__eq__(other)