import numpy as np

from src.env.Ticket import Ticket


class TicketGenerator:
    """
    TicketGenerator interface to implement
    These classes are responsible for generating tickets for the environment.
    """
    current_ticket = None
    num_ticket_features = None

    def step(self) -> Ticket:
        """
        Generate a ticket
        :return: a Ticket object
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the ticket generator
        """
        raise NotImplementedError

    def blank_ticket(self) -> Ticket:
        """
        Generate a blank ticket
        :return:
        """
        raise NotImplementedError


class TicketGeneratorFactory:
    @staticmethod
    def create_ticket_generator(ticket_type: str, **kwargs) -> TicketGenerator:
        """
        Factory method to create a ticket generator object
        :param ticket_type: str - type of ticket generator
        :param kwargs: dict - additional arguments for the ticket generator
        :return: TicketGenerator object
        """
        if ticket_type == 'random_embedded':
            return RandomEmbeddedTicketGenerator(**kwargs)
        else:
            raise ValueError(f"Unknown ticket generator type: {ticket_type}")


class RandomEmbeddedTicketGenerator(TicketGenerator):
    """
    RandomEmbeddedTicketGenerator class to generate ticket objects with random features
    These tickets are already embedded, basically we only generate a random vector of features in a confined space
    """

    def __init__(
            self,
            ticket_embedding_shape: int,
            min_value: float,
            max_value: float,
    ):
        """
        Constructor for RandomEmbeddedTicketGenerator class
        :param ticket_embedding_shape: int - size of the ticket embedding
        :param min_value: float - minimum value for each feature
        :param max_value: float - maximum value for each feature
        """
        self.ticket_embedding_shape = ticket_embedding_shape
        self.min_value = min_value
        self.max_value = max_value

        self.num_ticket_features = ticket_embedding_shape

        self.current_ticket = self._generate_ticket()

    def _generate_ticket(self) -> Ticket:
        """
        Generate a random ticket with random features
        :return:
        """
        features = np.random.uniform(self.min_value, self.max_value, size=self.ticket_embedding_shape)
        ticket = Ticket(
            machine='dummy',
            model='dummy',
            category='dummy',
            problem='dummy',
            features=features
        )
        return ticket

    def blank_ticket(self) -> Ticket:
        """
        Generate a blank ticket
        :return:
        """
        features = np.zeros(self.ticket_embedding_shape)
        ticket = Ticket(
            machine='blank',
            model='blank',
            category='blank',
            problem='blank',
            features=features
        )
        return ticket

    def reset(self):
        pass

    def step(self) -> Ticket:
        """
        Generate a ticket
        :return: a Ticket object
        """
        self.current_ticket = self._generate_ticket()
