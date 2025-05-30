"""Module to implement the SIREN (Sinusoidal Representation Networks) architecture.

It includes the SineLayer class, which uses a sine activation function for periodic
representation learning, as described in the SIREN paper. Aswell as the SIREN MLP.
"""

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    """Sine Layer with a periodic activation function.

    It can be used as the first layer or subsequent hidden layers in a SIREN.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        is_first (bool): If True, initializes weights for the first layer according
                         to SIREN paper. Default is False.
        omega_0 (float): The frequency parameter for the sine activation.
                         Typically 30.0 for the first layer and subsequent layers.

    """

    def __init__(self, in_features: int, out_features: int, omega_0: float = 20.0, *, is_first: bool = False) -> None:
        """Initialize the SineLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            omega_0 (float): The frequency parameter for the sine activation
                                typically 30.0 for the first layer and subsequent layers.
            is_first (bool): If True, initializes weights for the first layer according to SIREN paper. Default is False.

        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.is_first: bool = is_first
        self.omega_0: float = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the linear layer."""
        with torch.no_grad():
            if self.is_first:
                # Initialization for the first layer (Sec 3.2 and Appendix 1.5)
                # Weights uniformly distributed in [-1/in_features, 1/in_features]
                bound: float = 1.0 / self.in_features
                self.linear.weight.uniform_(from_=-bound, to=bound)
            else:
                # Initialization for subsequent layers (Theorem 1.8, Appendix 1.3 & 1.5)
                # Weights uniformly distributed in [-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0]
                # This corresponds to W_hat when W = W_hat * omega_0
                bound: float = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

            # Bias initialization (often to zero or small uniform)
            # The SIREN paper is less explicit or varies in examples.
            # For hypernetwork target SIREN (Appendix 9.1), bias was U[-1/n, 1/n]
            # added omega_0 to follow initialization observations.
            bound: float = 1 / (self.in_features * self.omega_0)
            self.linear.bias.uniform_(-bound, bound)  # Or self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sine layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_features).
                              The input should be normalized to [-1, 1] for best results.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_features).
        The output is computed as:
            Output = sin(omega_0 * (Wx + b)), where W is the weight matrix and b is the bias.
            The sine activation introduces non-linearity to the output.
        Output = sin(omega_0 * (Wx + b)).

        """
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Sinusoidal Representation Networks.

    It is a neural network architecture that uses sine activation functions
    to learn periodic functions and representations.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        hidden_layers (int): Number of hidden layers.
        out_features (int): Number of output features.
        omega_0 (float): The frequency parameter for the sine activation.
                         Typically 30.0 for the first layer and subsequent layers.
        fixup_init (bool | None): Flag to apply fixup initialization.
        max_layers_l (int | None): Maximum number of residual layers.
        linear_layers_m (int | None): Number of linear layers in each residual branch.

    """

    def __init__(
        self,
        input_features: int,
        out_features: int,
        list_hidden_features: list[int],
        omega_0: float = 30.0,
        fixup_init: bool | None = None,
        is_for_bias_generation: bool | None = None,
        max_layers_l: int | None = None,
        linear_layers_m: int | None = None,
        final_bias_init_value: float | None = None,  # Or some other value to make sigmoid(3.0) ~ 0.95
    ) -> None:
        """Initialize the SIREN.

        Args:
            input_features (int): Number of input features.
            out_features (int): Number of output features.
            list_hidden_features (list[int]): List of integers representing the number
                                            of in/out features for each layer in the SIREN.
            omega_0 (float): The frequency parameter for the sine activation.
                            Typically 30.0 for the first layer and subsequent layers.
            fixup_init (bool | None): Flag to apply fixup initialization.
            is_for_bias_generation (bool | None): Flag to generate biases or not.
            max_layers_l (int | None): Maximum number of residual layers in the whole network.
            linear_layers_m (int | None): Number of linear layers in each residual branch.


        """
        super().__init__()
        # Param validation
        if input_features <= 0:
            msg = "Input features must be a positive integer."
            raise ValueError(msg)
        if out_features <= 0:
            msg = "Output features must be a positive integer."
            raise ValueError(msg)
        if list_hidden_features:
            msg = "List of hidden features cannot be empty (min 2 linear layers)."
            raise ValueError(msg)

        # Fixup parameter validation
        apply_fixup_scaling = False
        if fixup_init:
            if not isinstance(max_layers_l, int) or max_layers_l <= 0:
                msg = "If fixup_init is True, max_layers_l must be a positive integer."
                raise ValueError(msg)
            if not isinstance(linear_layers_m, int) or linear_layers_m <= 1:
                # m must be >= 2 (at least 2 linear like layers per branch)
                msg = f"If fixup_init is True, linear_layers_m must be an integer >= 2, got {linear_layers_m}."
                raise ValueError(msg)
            apply_fixup_scaling = True

        # Init SIREN
        list_num_features: list[int] = [input_features, *list_hidden_features]
        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(len(list_num_features) - 1):
            is_first: bool = i == 0
            layer: SineLayer = SineLayer(
                in_features=list_num_features[i],
                out_features=list_num_features[i + 1],
                omega_0=omega_0,
                is_first=is_first,
            )
            self.layers.append(layer)
        self.final_layer: nn.Linear = nn.Linear(
            in_features=list_num_features[-1],
            out_features=out_features,
            bias=True,
        )

        # Apply fixup init
        if apply_fixup_scaling and max_layers_l and linear_layers_m:
            if is_for_bias_generation:
                with torch.no_grad():
                    self.final_layer.weight.data.zero_()  # Make weights zero
                    self.final_layer.bias.data.zero_()  # Make bias zero
                    # This ensures the SIREN output is exactly 0.0
            else:
                scale_factor: float = max_layers_l ** (-1.0 / (2 * linear_layers_m - 2))
                with torch.no_grad():
                    self.final_layer.weight.data *= scale_factor
                    self.final_layer.bias.data.zero_()
        elif final_bias_init_value is not None:
            with torch.no_grad():
                self.final_layer.bias.data.fill_(final_bias_init_value)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SIREN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_features).
                              The input should be normalized to [-1, 1] for best results.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_features).

        """
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)
