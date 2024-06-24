import jax
import numpy as np

from src.env.Ticket import Ticket
import jax.numpy as jnp
import plotly.graph_objects as go

from jax.scipy.signal import convolve

from functools import partial


# Define all helpers function that will be jax-compiled
@partial(jax.jit, static_argnames=['grid_cell_volume'])
def compute_hypervolume(experience_grid, grid_cell_volume):
    """
    Get the hypervolume of the experience grid
    """
    return jnp.sum(experience_grid) * grid_cell_volume


@partial(jax.jit, static_argnames=['learning_rate', 'transmission_factor'])
def _compute_experience_with_supervisor(experience_grid, ticket_embedding, supervisor_experience: float,
                                        learning_rate: float,
                                        transmission_factor: float):
    """
    Compute the experience increase for a given ticket embedding
    """
    previous_experience = get_experience(experience_grid, ticket_embedding)
    experience_increase = jnp.log1p(jnp.exp(-previous_experience)) * learning_rate
    return (previous_experience + experience_increase) * (
            1 - transmission_factor) + supervisor_experience * transmission_factor


@partial(jax.jit, static_argnames=['learning_rate'])
def _compute_experience_without_supervisor(experience_grid, ticket_embedding: jnp.ndarray, learning_rate: float):
    """
    Compute the experience increase for a given ticket embedding
    :param ticket_embedding: jnp.ndarray - ticket embedding
    :return: float - new experience
    """
    previous_experience = get_experience(experience_grid, ticket_embedding)
    return previous_experience + jnp.log1p(jnp.exp(-previous_experience)) * learning_rate


@partial(jax.jit, static_argnames=['grid_size', 'ticket_embedding_shape'])
def _blank_grid(grid_size: int, ticket_embedding_shape: int):
    """
    Initialize the experience grid
    :return:
    """
    shape = tuple([grid_size for _ in range(ticket_embedding_shape)])
    return jnp.zeros(shape, dtype=jnp.float32)


def gaussian_kernel(size, sigma: float) -> jnp.ndarray:
    """Creates a Gaussian kernel."""
    low = -size // 2 + 1
    high = size // 2 + 1
    x = jnp.arange(low, high)
    gauss_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d /= gauss_1d.sum()  # Normalize
    gauss_2d = jnp.outer(gauss_1d, gauss_1d)
    return gauss_2d


@partial(jax.jit, static_argnames=['threshold'])
def _propagate_increase(grid: jnp.ndarray, coord, incr, threshold, gaussian) -> jnp.ndarray:
    """
    Increase the value at specific coordinates by the given increment and propagate it in a bell-curve manner.
    """
    delta_grid = jnp.zeros_like(grid)
    delta_grid = delta_grid.at[coord].add(incr)
    # Apply Gaussian filter using convolution
    propagated_grid = convolve(delta_grid, gaussian, mode='same')
    # Apply the threshold
    scaling_factor = incr / propagated_grid[coord]
    propagated_grid *= scaling_factor
    propagated_grid = jnp.where(propagated_grid < threshold, 0, propagated_grid)
    # Add the propagated values to the original grid
    updated_grid = grid + propagated_grid
    return updated_grid


@jax.jit
def get_experience(experience_grid, ticket_embedding: jnp.ndarray):
    """
    Get the experience for a given ticket embedding.
    """
    coord = tuple(ticket_embedding.astype(int))
    return experience_grid[coord]


class ExperienceGrid:
    """
    This class implements the ExperienceGrid object, which represents the experience of a technician as a grid of points
    in an n+1-dimensional space, n being the size of the ticket embedding, and the additional dimension representing the
    experience level of the technician

    The experience is calculated using the following formula:
    experience = log(number of tickets treated for this point in the grid + 1)
    thus an increment of experience is computed as:
    increment = log(exp(previous experience) + 1)

    it is possible for a technician to transfer a part of their experience to another technician, in this case, the
    increment is computed as:

    In addition to this, each increment will propagate to the neighboring points in the grid, following a normal
    distribution with mean at the current point and standard deviation of 'propagation_sigma'
    """

    @classmethod
    def check_or_default(cls, attribute, default):
        return getattr(cls, attribute) if getattr(cls, attribute) is not None else default

    @classmethod
    def propagation_sigma(cls):
        return cls.check_or_default('_propagation_sigma', default=1.0)

    @classmethod
    def transmission_factor(cls):
        return cls.check_or_default('_transmission_factor', default=0.5)

    @classmethod
    def ticket_featurizer(cls):
        return cls.check_or_default('_ticket_featurizer', default="DefaultFeaturizer")

    @classmethod
    def ticket_embedding_shape(cls):
        return cls.check_or_default('_ticket_embedding_shape', default=(10,))

    @classmethod
    def grid_size(cls):
        return cls.check_or_default('_grid_size', default=100)

    @classmethod
    def propagation_threshold(cls):
        return cls.check_or_default('_propagation_threshold', default=0.1)

    @classmethod
    def feature_max(cls):
        return cls.check_or_default('_feature_max', default=1.0)

    @classmethod
    def feature_min(cls):
        return cls.check_or_default('_feature_min', default=0.0)

    def __init__(
            self,
            grid_id: int,
            learning_rate: float,
            initial_experience_seeds: np.ndarray,
    ):
        self.grid_id = grid_id
        self.learning_rate = learning_rate
        self.experience_seeds = initial_experience_seeds
        self.experience_grid = _blank_grid(grid_size=self.grid_size, ticket_embedding_shape=self.ticket_embedding_shape)

        scaled_sigma = self.propagation_sigma * self.grid_size / (self.feature_max - self.feature_min)
        self.kernel_size = int(2 * jnp.ceil(3 * scaled_sigma) + 1)

        self.gaussian = gaussian_kernel(self.kernel_size, scaled_sigma)

        # Initialize the experience grid with the given seeds
        self.initialize_grid()

    def get_hypervolume(self):
        """
        Get the hypervolume of the experience grid
        :return: - hypervolume of the experience grid
        """
        return compute_hypervolume(self.experience_grid, self.grid_cell_volume)

    def add_ticket_experience(self, ticket: Ticket, supervisor=None):
        """
        Add experience to the experience grid
        :param ticket: Ticket - ticket object
        :param supervisor - supervisor technician object if any
        :return: nothing, modifies in place
        """
        ticket_embedding = self.ticket_featurizer.featurize(ticket)
        if supervisor:
            # Get the experience of the supervisor at the ticket embedding
            supervisor_experience = supervisor.experience_grid.get_experience(ticket_embedding)
            new_experience = _compute_experience_with_supervisor(
                experience_grid=self.experience_grid,
                ticket_embedding=ticket_embedding,
                supervisor_experience=supervisor_experience,
                learning_rate=self.learning_rate,
                transmission_factor=self.transmission_factor,
            )
        else:
            new_experience = _compute_experience_without_supervisor(
                experience_grid=self.experience_grid,
                ticket_embedding=ticket_embedding,
                learning_rate=self.learning_rate,
            )

        coord = self._get_coord_from_embedding(ticket_embedding)
        self.experience_grid = _propagate_increase(
            grid=self.experience_grid,
            coord=coord,
            incr=new_experience,
            threshold=float(jnp.log1p(0.0001)),
            gaussian=self.gaussian,
        )

    def add_supervisor_experience(self, ticket: Ticket, supervisor):
        """
        Add experience to the experience grid
        :param ticket: Ticket - ticket object
        :param supervisor - supervisor technician object
        :return: nothing, modifies in place
        """
        supervisor_experience = supervisor.experience_grid.get_experience(ticket)
        ticket_embedding = self.ticket_featurizer.featurize(ticket)
        current_experience = get_experience(self.experience_grid, ticket_embedding)

        # Compute the new experience with the transmission factor
        new_experience = (current_experience + supervisor_experience) * self.transmission_factor

        coord = self._get_coord_from_embedding(ticket_embedding)
        self.experience_grid.at[coord].set(new_experience)

    def _initialize_seed(self, seed_embedding: np.ndarray):
        """
        Initialize the random number generator with the given seed.
        :param seed_embedding: np.ndarray - seed embedding
        """
        new_experience = _compute_experience_without_supervisor(
            experience_grid=self.experience_grid,
            ticket_embedding=seed_embedding,
            learning_rate=self.learning_rate)
        coord = self._get_coord_from_embedding(seed_embedding)
        # Add the new experience to the grid
        self.experience_grid = _propagate_increase(
            grid=self.experience_grid,
            coord=coord,
            incr=new_experience,
            threshold=self.propagation_threshold,
            gaussian=self.gaussian
        )

    def _get_coord_from_embedding(self, embedding: np.ndarray):
        """
        Get the grid coordinates from the embedding
        :param embedding: np.ndarray - embedding
        :return: tuple - grid coordinates
        """
        # Map the embedding to the grid coordinates, taking into account the size of the grid
        coords_normalized = (embedding - self.feature_min) / (self.feature_max - self.feature_min)
        coords = tuple((coords_normalized * self.grid_size - 1).astype(int))
        return coords

    def initialize_grid(self):
        """
        Use the initialization seeds to initialize the experience grid
        :return:
        """
        for seed in self.experience_seeds:
            self._initialize_seed(seed)

    def reset(self):
        """
        Reset the experience grid
        :return:
        """
        self.experience_grid = _blank_grid(grid_size=self.grid_size, ticket_embedding_shape=self.ticket_embedding_shape)
        self.initialize_grid()

    def render(self, mode='2d', style='surface'):
        """
        Renders the experience grid
        :param mode: str - mode of rendering
        :return:
        """
        if mode == '2d':
            if style == 'surface':
                self._render2d_surface()
            elif style == 'scatter':
                self._render2d()
            else:
                raise ValueError(f"Unknown rendering style: {style}")
        else:
            raise ValueError(f"Unknown rendering mode: {mode}")

    def get_experience(self, ticket):
        """
        Get the experience for a given ticket
        :param ticket: Ticket - ticket object
        :return: float - experience value
        """
        ticket_embedding = self.ticket_featurizer.featurize(ticket)
        return get_experience(self.experience_grid, ticket_embedding)

    def get_max_experience(self):
        """
        Get the maximum experience value in the grid
        :return: float - maximum experience value
        """
        return jnp.max(self.experience_grid)

    def get_min_experience(self):
        """
        Get the minimum experience value in the grid
        :return: float - minimum experience value
        """
        return jnp.min(self.experience_grid)

    def _render2d(self):
        # Extract the grid from the ExperienceGrid object
        grid = self.experience_grid

        # Initialize x, y, and z lists to store the coordinates and experience values
        x, y, z = [], [], []

        # Calculate the step size for each coordinate based on grid size and value range
        x_step = (self.feature_max - self.feature_min) / self.grid_size
        y_step = (self.feature_max - self.feature_min) / self.grid_size

        # Iterate over the first two dimensions of the grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # Map grid indices to actual feature values
                x_coord = self.feature_min + i * x_step
                y_coord = self.feature_min + j * y_step

                x.append(x_coord)
                y.append(y_coord)
                z.append(float(grid[i, j]))  # Assuming the grid stores experience directly at these indices

        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                           mode='markers',
                                           marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8))])

        # Update plot layout
        fig.update_layout(title='2D Slice of Experience Grid in 3D Space',
                          scene=dict(
                              xaxis_title='Coord1',
                              yaxis_title='Coord2',
                              zaxis_title='Experience Value'),
                          margin=dict(l=0, r=0, b=0, t=0))

        # Show the plot
        fig.show()

    def _render2d_surface(self):
        # Extract the grid from the ExperienceGrid object
        grid = self.experience_grid

        # Initialize x, y, and z lists to store the coordinates and experience values
        x, y = [], []

        # Calculate the step size for each coordinate based on grid size and value range
        x_step = (self.feature_max - self.feature_min) / self.grid_size
        y_step = (self.feature_max - self.feature_min) / self.grid_size

        # Iterate over the first two dimensions of the grid to create the x and y coordinates
        for i in range(grid.shape[0]):
            x_coord = self.feature_min + i * x_step
            x.append(x_coord)

        for j in range(grid.shape[1]):
            y_coord = self.feature_min + j * y_step
            y.append(y_coord)

        # Convert the lists to a meshgrid for the surface plot
        x, y = np.meshgrid(x, y)

        # Create a surface plot
        fig = go.Figure(data=[go.Surface(z=grid, x=x, y=y, colorscale='Viridis')])

        # Update plot layout
        fig.update_layout(title='2D Slice of Experience Grid in 3D Space',
                          scene=dict(
                              xaxis_title='Coord1',
                              yaxis_title='Coord2',
                              zaxis_title='Experience Value'),
                          margin=dict(l=0, r=0, b=0, t=0))

        # Show the plot
        fig.show()
