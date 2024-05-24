import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamer.utils.utils import build_network, create_normal_dist, horizontal_forward
from torch.distributions import Normal


class RSSM(nn.Module):
    def __init__(
        self,
        action_size,
        stochastic_size,
        deterministic_size,
        device,
        recurrent_model_config,
        transition_model_config,
        representation_model_config,
    ):
        super().__init__()

        self.recurrent_model = RecurrentModel(
            action_size,
            device=device,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            hidden_size=recurrent_model_config["hidden_size"],
            activation=recurrent_model_config["activation"],
        )
        self.transition_model = TransitionModel(
            device=device,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            hidden_size=transition_model_config["hidden_size"],
            num_layers=transition_model_config["num_layers"],
            activation=transition_model_config["activation"],
            min_std=transition_model_config["min_std"],
        )
        self.representation_model = RepresentationModel(
            embedded_state_size=representation_model_config["embedded_state_size"],
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            hidden_size=representation_model_config["hidden_size"],
            num_layers=representation_model_config["num_layers"],
            activation=representation_model_config["activation"],
            min_std=representation_model_config["min_std"],
        )

    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(
            batch_size
        ), self.recurrent_model.input_init(batch_size)


class RecurrentModel(nn.Module):
    def __init__(
        self,
        action_size,
        device,
        stochastic_size,
        deterministic_size,
        hidden_size,
        activation,
    ):
        super().__init__()
        self.device = device
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.activation = getattr(nn, self.activation)()

        self.linear = nn.Linear(self.stochastic_size + action_size, self.hidden_size)
        self.recurrent = nn.GRUCell(self.hidden_size, self.deterministic_size)

    def forward(self, embedded_state, action, deterministic):
        x = torch.cat((embedded_state, action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.deterministic_size).to(self.device)


class TransitionModel(nn.Module):
    def __init__(
        self,
        device,
        stochastic_size,
        deterministic_size,
        hidden_size,
        num_layers,
        activation,
        min_std,
    ):
        super().__init__()
        self.device = device
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.min_std = min_std

        self.network = build_network(
            self.deterministic_size,
            self.hidden_size,
            self.num_layers,
            self.activation,
            self.stochastic_size * 2,
        )

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).to(self.device)


class RepresentationModel(nn.Module):
    def __init__(
        self,
        embedded_state_size,
        stochastic_size,
        deterministic_size,
        hidden_size,
        num_layers,
        activation,
        min_std,
    ):
        super().__init__()
        self.embedded_state_size = embedded_state_size
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.min_std = min_std

        self.network = build_network(
            self.embedded_state_size + self.deterministic_size,
            self.hidden_size,
            self.num_layers,
            self.activation,
            self.stochastic_size * 2,
        )

    def forward(self, embedded_observation, deterministic):
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=self.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class RewardModel(nn.Module):
    def __init__(
        self, stochastic_size, deterministic_size, hidden_size, num_layers, activation
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            hidden_size,
            num_layers,
            activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class ContinueModel(nn.Module):
    def __init__(
        self, stochastic_size, deterministic_size, hidden_size, num_layers, activation
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.hidden_size,
            self.num_layers,
            self.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = torch.distributions.Bernoulli(logits=x)
        return dist
