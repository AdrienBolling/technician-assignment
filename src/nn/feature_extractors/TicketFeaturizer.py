"""
This module contains the TicketFeaturizer class, which is responsible for extracting features from the raw ticket fields
"""
import jax
import jax.numpy as jnp
import pandas as pd
from jax.scipy.linalg import svd
from src.env.Ticket import Ticket


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class TicketFeaturizer:

    def __init__(self):
        """
        Constructor for TicketFeaturizer class
        """
        raise NotImplementedError

    def featurize(self, ticket: Ticket):
        """
        Featurize the ticket object, returns it as an array-like object (list, numpy array, jax array, tensor, etc.)
        """
        raise NotImplementedError

    def unfeaturize(self, features):
        """
        Unfeaturize the features back into a Ticket object
        note: this method is not required for the current implementation, but it is good to have it for completeness
        """
        raise NotImplementedError

    def get_featurizer_params(self):
        """
        Get the parameters of the featurizer
        """
        raise NotImplementedError


class TicketFeaturizerFactory():
    @staticmethod
    def create_ticket_featurizer(featurizer_type: str, **kwargs) -> TicketFeaturizer:
        """
        Factory method to create a ticket featurizer object
        :param featurizer_type: str - type of ticket featurizer
        :param kwargs: dict - additional arguments for the ticket featurizer
        :return: TicketFeaturizer object
        """
        if featurizer_type == 'identity':
            return IdentityTicketFeaturizer(**kwargs)
        elif featurizer_type == 'mca':
            return JaxMCATicketFeaturizer(**kwargs)
        else:
            raise ValueError(f"Unknown ticket featurizer type: {featurizer_type}")


@singleton
class IdentityTicketFeaturizer(TicketFeaturizer):
    """
    Implementation of the TicketFeaturizer class using an identity function
    """

    def __init__(self, ticket_embedding_shape, value_min=-1, value_max=1):
        """
        Constructor for IdentityTicketFeaturizer class
        """
        self.ticket_embedding_shape = ticket_embedding_shape
        self.min_value = value_min
        self.max_value = value_max
        self.value_range_len = value_max - value_min

    def featurize(self, ticket: Ticket) -> jnp.ndarray:
        """
        Featurize the ticket object, returns it as an array-like object (list, numpy array, jax array, tensor, etc.)
        """
        return jnp.array(ticket.get_features())

    def get_featurizer_params(self):
        return {
            'ticket_embedding_shape': self.ticket_embedding_shape,
            'min_value': self.min_value,
            'max_value': self.max_value,
        }


@singleton
class JaxMCATicketFeaturizer(TicketFeaturizer):
    """
    Implementation of the TicketFeaturizer class using an MCA (Multiple Correspondence Analysis) approach
    """

    def __init__(self, n_components: int):
        """
        Constructor for MCATicketFeaturizer class
        :param n_components: int - number of components to use for MCA
        """
        super().__init__()
        self.n_components = n_components

    @jax.jit
    def _perform_svd(self, data_matrix):
        # Center the matrix by subtracting the mean
        matrix_centered = data_matrix - jnp.mean(data_matrix, axis=0)
        # Perform SVD
        U, S, Vt = svd(matrix_centered, full_matrices=False)
        return U, S, Vt

    def mca_jax(self, df, n_components=2):
        # Convert DataFrame to one-hot encoded format
        df_encoded = pd.get_dummies(df)
        # Convert to JAX array
        matrix = jnp.array(df_encoded)
        # Perform SVD
        U, S, Vt = self._perform_svd(matrix)
        # Project data onto the first 'n_components' principal components
        projected_data = U[:, :n_components] * S[:n_components]
        return projected_data, U, S, Vt

    # def featurize(self, ticket: Ticket) -> jnp.ndarray:
    #     """
    #     Featurize the ticket object, returns it as an array-like object (list, numpy array, jax array, tensor, etc.)
    #     """
