class Ticket:
    """
    Ticket class to represent a ticket object
    """

    _id_counter = 0

    def __init__(
            self,
            machine: str = 'dummy',
            model: str = 'dummy',
            category: str = 'dummy',
            problem: str = 'dummy',
            features=None,
    ):
        """
        Constructor for Ticket class
        :param machine: str - machine name
        :param model: str - machine model
        :param category: str - ticket category
        :param problem: str - ticket problem
        :param features: - ticket features
        """
        self.id = Ticket._id_counter
        Ticket._id_counter += 1

        self.machine = machine
        self.model = model
        self.category = category
        self.problem = problem

        self.features = features

    def get_features(self):
        """
        Get the features of the ticket
        :return: list - list of ticket features
        """
        return self.features

    def __str__(self):
        return (f"Ticket n:{self.id} - mac:{self.machine} - mod:{self.model} - cat:{self.category} - prb:{self.problem}"
                f"\n features: {self.features}")

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Ticket):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


# class TicketGenerator:
#     """
#     TicketGenerator class to generate ticket objects, also computes the features for the tickets before
#     giving them to the environment
#     """
#
#     def __init__(
#             self,
#             seed: Optional[int] = None,
#             possible_values: Optional[dict] = None,
#             feature_generator_params: Optional[dict] = None,
#     ):
#
#         """
#         Constructor for TicketGenerator class
#         :param seed: int - seed value for random number generator
#         :param possible_values: dict - dictionary of possible values for ticket fields
#         """
#
#         self.seed = seed
#
#         if possible_values:
#             self.possible_values = possible_values
#         else:
#             self.possible_values = {
#                 "machine": ["Haas", "Gemini", "Mazak"],
#                 "model": {
#                     "Haas": ["H-1", "H-2", "H-3"],
#                     "Gemini": ["G-1", "G-2", "G-3"],
#                     "Mazak": ["M-1", "M-2", "M-3"],
#                 },
#                 "category": ["Electrical", "Mechanical", "Software"],
#                 "problem": {
#                     "Electrical": ["Power Supply", "Motor", "Sensor"],
#                     "Mechanical": ["Bearing", "Shaft", "Gear"],
#                     "Software": ["Driver", "Firmware", "Application"],
#                 },
#             }
#
#         self.current_ticket_id = 0
#
#     def _generate_ticket(self, ticket_id: int):
#
#         """
#         Generate a ticket object with the given ticket id
#         :param ticket_id: int - ticket id
#         :return: a ticket object
#         """
#
#         machine = np.random.choice(self.possible_values["machine"])
#         model = np.random.choice(self.possible_values["model"][machine])
#         category = np.random.choice(self.possible_values["category"])
#         problem = np.random.choice(self.possible_values["problem"][category])
#
#         return Ticket(ticket_id, machine, model, category, problem)
#
#     def _setup_featurization(self, extractor='acm', **kwargs):
#
#         """
#         Sets up the featurization for the tickets
#         :param extractor: str - featurization method
#         :param kwargs: dict - additional arguments for the featurization method
#         :return:
#         """
#
#         if extractor == 'acm':
#             from src.env.feature_extraction import ACMFeatureExtractor
#             self.feature_extractor = ACMFeatureExtractor(**kwargs)
#         else:
#             raise ValueError(f"Unknown feature extractor: {extractor}, please use 'acm'")
