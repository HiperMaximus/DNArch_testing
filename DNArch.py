"""Module for the DNArch Variational Autoencoder class.

DNArch: Differentiable Neural Architecture with Hypernetworks & Fourier Masking
-----------------------------------------------------------------------------

This module implements DNArch, a hypernetwork-driven Variational Autoencoder (VAE) that
learns both its weights and its own architecture in a fully differentiable manner. The key
idea is to use small sinusoidal coordinate networks (SIRENs) as hypernetworks to generate
and control every architectural component—from convolutional kernel weights and sizes,
feature-channel depths, and layer inclusion, to downsampling filters in Fourier space.

DNArch is organized around the following meta-components and step-by-step procedures:

1. Input Hyperparameters (Envelope of Search Space)
   -----------------------------------------------
   - L         : Maximum number of convolutional layers per network (encoder/decoder)
   - C_max     : Maximum number of channels for residual and identity branches
   - K_max     : Maximum spatial kernel size for convolutions
   - D         : Latent dimension size of the VAE
   - Resolution: Input image resolution (H, W, C)
   - FourierMaskSIREN: SIREN network used to generate frequency-domain masks

2. Continuous Mask Mechanisms
   ---------------------------
   DNArch uses differentiable masks to softly select architectural components:

   a. Spatial Kernel Mask (2D Gaussian):
      - Controls effective convolutional receptive field within a K_max × K_max kernel.
      - Defined as:
          M_spatial(x,y) = exp(-((x-μ_x)^2 + (y-μ_y)^2) / (2σ^2))
      - σ is clamped to σ_min to ensure a minimum kernel size (e.g., 3×3).

   b. Channel Depth Mask (1D Sigmoid):
      - Controls active input/output channels up to C_max.
      - Defined as:
          M_channels(c) = sigmoid((c - τ_c) / T_c)
      - τ_c is a learned cutoff, T_c ≥ T_c_min ensures smooth transitions.

   c. Layer Inclusion Mask (1D Sigmoid):
      - Controls pruning of each layer out of L possible layers.
      - Defined as:
          M_layers(ℓ) = sigmoid((ℓ - τ_ℓ) / T_ℓ)
      - Learned thresholds τ_ℓ and temperatures T_ℓ ≥ T_ℓ_min guarantee differentiability.

3. Hypernetwork Weight Generation via SIRENs
   -----------------------------------------
   For each convolutional layer ℓ = 1…L:
   - Instantiate a small SIREN that takes continuous coordinates (i, j, c_in, c_out) and outputs
     the raw kernel weight at that position.
   - Apply Spatial Kernel Mask to zero out locations outside the learned receptive field.
   - Apply Channel Depth Masks on the input/output channel dimensions.
   - Resulting masked weight tensor:
       W_ℓ_masked = M_channels_out ⋅ (W_ℓ ⊙ M_spatial) ⋅ M_channels_in

4. Downsampling via Fourier-Space Masking
   ---------------------------------------
   To smoothly reduce spatial resolution in residual streams:

   a. Compute centered 2D FFT:    X̂ = FFT2(X)
   b. Precompute integer squared-radius grid:
        r²[i,j] = (i - H/2)^2 + (j - W/2)^2  for 0 ≤ i,j < H
   c. Extract unique r² values and inverse indices once per resolution.
   d. Normalize radii: r = √r² / (√2 · (H/2))  → [0,1]
   e. For each layer ℓ, evaluate a small SIREN on inputs (ℓ_norm, r_unique)
      to produce mask values M_r_unique.
   f. Broadcast back via inverse indices to full H×W mask M_r.
   g. Multiply:   X̂_filtered = X̂ ⋅ M_r (broadcasted over channels)
   h. Inverse FFT2 for downsampled output:  X_down = IFFT2(X̂_filtered)

5. Encoder & Decoder Assembly
   ---------------------------
   - Loop ℓ = 1…L:
     • Compute M_layers(ℓ) to gate layer inclusion.
     • If included, generate masked convolution W_ℓ_masked and apply to activations.
     • Optionally apply Fourier-space downsampling in residual path.
     • Apply nonlinearity.
   - Encoder final layers predict μ(x), logσ(x) for latent z.
   - Decoder consumes z and mirrors the above process in reverse to reconstruct X.

6. Training Objective & Differentiability
   -------------------------------------
   - Optimize standard VAE loss:
       L = E_q(z|x)[-log p(x|z)] + KL(q(z|x)||p(z))
   - All masks and SIREN parameters are trained end-to-end via backprop.
   - Clamping or regularization maintains σ ≥ σ_min, T ≥ T_min, avoiding hard cutoffs.

"""

import torch
from torch import nn

from Masks import GaussianMask2D, SigmoidMask1D
from SIREN import SIREN


class ResidualBranch(nn.Module):
    """Residual branch class.

    A neural network module that implements a residual branch with identity and activation layers.

    Args:
        prelu_init (float): Initial value for the PReLU activation function. Default is 0.25.
        alpha (float): Scaling factor for the residual output. Default is 0.001.

    """

    def __init__(self, prelu_init: float = 0.25, alpha: float = 0.00001) -> None:
        """Initialize the ResidualBranch."""
        super().__init__()
        self.activation1: nn.Module = nn.PReLU(init=prelu_init)
        self.activation2: nn.Module = nn.PReLU(init=prelu_init)
        # Fix-up initialization like paremeters
        # Initial bias for the residual branch
        self.bias_before_pw = nn.Parameter(torch.tensor(0.0))
        # Bias after PReLU1 is effectively before DWConv
        self.bias_before_dw = nn.Parameter(torch.tensor(0.0))
        # Bias after PReLU2 is effectively before FFT
        self.bias_before_fft = nn.Parameter(torch.tensor(0.0))
        # Scaling factor for the residual output
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(
        self,
        x: torch.Tensor,
        # Pointwise and depthwise convolution parameters generated by SIREN
        pointwise_kernel: torch.Tensor,
        bias_pointwise: torch.Tensor,
        depthwise_kernel: torch.Tensor,
        bias_depthwise: torch.Tensor,
        # This mask is generated by SIREN based on centered radii for an RFFT spectrum
        # it is given on radial coordinates as to be isotropic in the frequency domain.
        fourier_mask_centered_rfft: torch.Tensor,  # Shape (H_feat, W_feat//2 + 1)
    ) -> torch.Tensor:
        """Forward pass through the ResidualBranch.

        Args:
            x (torch.Tensor): Input tensor, with shape (B, C, H, W).
            pointwise_kernel (torch.Tensor): Pointwise convolution kernel, with shape (C, C, 1, 1).
            bias_pointwise (torch.Tensor): Pointwise convolution bias, with shape (C,).
            depthwise_kernel (torch.Tensor): Depthwise convolution kernel, with shape (C, 1, K_h, K_w).
            bias_depthwise (torch.Tensor): Depthwise convolution bias, with shape (C,).
            fourier_mask_centered_rfft (torch.Tensor): Mask for Fourier space, with shape (H_feat, W_feat//2 + 1).

        Returns:
            torch.Tensor: Output tensor after processing through the residual branch, with shape
            (B, C, H, W).

        """
        height_feat, width_feat = x.shape[-2:]
        skip: torch.Tensor = x  # Save the input for the skip connection
        x = nn.functional.conv2d(input=x + self.bias_before_pw, weight=pointwise_kernel, bias=bias_pointwise, padding="same")
        x = self.activation1(x)
        x = nn.functional.conv2d(
            input=x + self.bias_before_dw,
            weight=depthwise_kernel,
            bias=bias_depthwise,
            padding="same",
            groups=x.shape[1],
        )
        x = self.activation2(x)
        x = torch.fft.rfft2(x + self.bias_before_fft, norm="ortho")  # (B,C,H,W//2+1)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        mask_to_apply = fourier_mask_centered_rfft.unsqueeze(0).unsqueeze(0)
        x = x * mask_to_apply  # Apply the mask
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.irfft2(x, s=(height_feat, width_feat), norm="ortho")
        return skip + self.alpha * x


class Encoder(nn.Module):
    """Encoder class.

    A neural network encoder that maps input images to a latent space.

    Args:
        image_size (int): Size of square image.
        latent_dim (int): Dimension of the latent space.
        max_layers (int): Maximum number of layers in the architecture.
        max_channels_residual (int): Maximum number of channels in the residual branch.
        max_channels_skip (int): Maximum number of channels in the identity branch.
        max_kernel_size (int): Maximum kernel size.

    """

    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        max_layers: int,
        max_channels_residual: int,
        max_channels_skip: int,
        max_kernel_size: int,
    ) -> None:
        """Initialize the Encoder.

        Args:
            image_size (int): Size of square image.
            latent_dim (int): Dimension/size of the latent space.
            max_layers (int): Maximum number of layers in the architecture.
            max_channels_residual (int): Maximum number of channels in the residual branch.
            max_channels_skip (int): Maximum number of channels in the identity branch.
            max_kernel_size (int): Maximum kernel size.

        """
        super().__init__()

        # Initialize encoder parameters
        self.image_size: int = image_size
        self.latent_dim: int = latent_dim
        self.max_layers: int = max_layers
        self.max_channels_residual: int = max_channels_residual
        self.max_channels_skip: int = max_channels_skip
        self.max_kernel_size: int = max_kernel_size

        # --- Learnable Architectural Networks (SIRENs) ---

        # Pointwise kernel generation
        # input_features := (layer,channel)
        self.pointwise_siren = SIREN(
            input_features=2,
            out_features=1,
            list_hidden_features=[128, 128, 128],
            omega_0=20.0,
            fixup_init=True,
            max_layers_l=self.max_layers,
            linear_layers_m=2,
        )
        # input_features := (layer,channel)
        self.pointwise_bias_siren = SIREN(
            input_features=2,
            out_features=1,
            list_hidden_features=[16, 16, 16],
            omega_0=20.0,
            fixup_init=True,
            is_for_bias_generation=True,
            max_layers_l=self.max_layers,
            linear_layers_m=2,
        )
        # Depthwise kernel generation
        # input_features := (layer,channel,k_x,k_y)
        self.depthwise_siren = SIREN(
            input_features=4,
            out_features=1,
            list_hidden_features=[128, 128, 128, 128],
            omega_0=20.0,
            fixup_init=True,
            max_layers_l=self.max_layers,
            linear_layers_m=2,
        )
        # input_features := (layer,channel)
        self.depthwise_bias_siren = SIREN(
            input_features=2,
            out_features=1,
            list_hidden_features=[16, 16, 16],
            omega_0=20.0,
            fixup_init=True,
            is_for_bias_generation=True,
            max_layers_l=self.max_layers,
            linear_layers_m=2,
        )
        # input_features := (layer,radius_coord)
        self.mask_siren = nn.Sequential(
            SIREN(input_features=2, out_features=1, list_hidden_features=[10, 8, 4], omega_0=10.0, final_bias_init_value=3.0),
            nn.Sigmoid(),
        )
        self.res_layers: nn.ModuleList = nn.ModuleList()
        for _ in range(self.max_layers):
            self.res_layers.append(ResidualBranch())


class Decoder(nn.Module):
    """Decoder class.

    A neural network decoder that maps latent vectors to reconstructed images.

    Args:
        image_size (int): Size of square image.
        latent_dim (int): Dimension of the latent space.
        max_layers (int): Maximum number of layers in the architecture.
        max_channels_residual (int): Maximum number of channels in the residual branch.
        max_channels_skip (int): Maximum number of channels in the identity branch.
        max_kernel_size (int): Maximum kernel size.

    """

    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        max_layers: int,
        max_channels_residual: int,
        max_channels_skip: int,
        max_kernel_size: int,
    ) -> None:
        """Initialize the Decoder.

        Args:
            image_size (int): Size of square image.
            latent_dim (int): Dimension of the latent space.
            max_layers (int): Maximum number of layers in the architecture.
            max_channels_residual (int): Maximum number of channels in the residual branch.
            max_channels_skip (int): Maximum number of channels in the identity branch.
            max_kernel_size (int): Maximum kernel size.

        """
        super().__init__()
        # Initialize decoder parameters
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.max_layers = max_layers
        self.max_channels_residual = max_channels_residual
        self.max_channels_skip = max_channels_skip
        self.max_kernel_size = max_kernel_size

        # Define layers and SIRENs here as needed


class DNArchVAE(nn.Module):
    """DNArch class.

    A hyper/meta neural network architecture to learn neural networks architectures.

    Args:
        image_size (int): Size of square image.
        latent_dim (int): Dimension of the latent space.
        max_layers (int): Maximum number of layers in the architecture.
        max_channels_residual (int): Maximum number of channels in the residual branch.
        max_channels_skip (int): Maximum number of channels in the identity branch.
        max_kernel_size (int): Maximum kernel size.

    """

    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        max_layers: int,
        max_channels_residual: int,
        max_channels_skip: int,
        max_kernel_size: int,
    ) -> None:
        """Initialize the DNArchVAE.

        Args:
            image_size (int): Size of square image.
            latent_dim (int): Dimension of the latent space.
            max_layers (int): Maximum number of layers in the architecture.
            max_channels_residual (int): Maximum number of channels in the residual branch.
            max_channels_skip (int): Maximum number of channels in the identity branch.
            max_kernel_size (int): Maximum kernel size.

        """
        super().__init__()
        # Validate input parameters
        if image_size <= 0:
            msg = "Image size must be a positive integer."
            raise ValueError(msg)
        if latent_dim <= 0:
            msg = "Latent dimension must be a positive integer."
            raise ValueError(msg)
        if max_layers <= 0:
            msg = "Maximum layers must be a positive integer."
            raise ValueError(msg)
        if max_channels_residual <= 0:
            msg = "Maximum channels in residual branch must be a positive integer."
            raise ValueError(msg)
        if max_channels_skip <= 0:
            msg = "Maximum channels in identity branch must be a positive integer."
            raise ValueError(msg)
        if max_kernel_size <= 0:
            msg = "Maximum kernel size must be a positive integer."
            raise ValueError(msg)
        if max_kernel_size % 2 == 0:
            msg = "Maximum kernel size must be an odd integer."
            raise ValueError(msg)

        # Initialize parameters
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.max_channels_residual = max_channels_residual
        self.max_channels_skip = max_channels_skip
        self.max_kernel_size = max_kernel_size
        self.layers_coords: torch.Tensor = torch.linspace(-1, 1, max_layers)
        self.channels_residual_coords: torch.Tensor = torch.linspace(
            -1,
            1,
            max_channels_residual,
        )
        self.channels_skip_coords: torch.Tensor = torch.linspace(
            -1,
            1,
            max_channels_skip,
        )
        self.kernel_size_coords: torch.Tensor = torch.linspace(
            -1,
            1,
            max_kernel_size,
        )
