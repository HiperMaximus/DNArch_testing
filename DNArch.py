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
from SIREN import SIREN


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
        self.siren_2d_depthwise: SIREN = SIREN(
            input_features=4, # (layer,c,k_x,k_y)
            out_features=1, # single weight per position
            list_hidden_features=[64, 64],
            omega_0=20.0,
        )
        self.siren_1d_pointwise: SIREN = SIREN(
            input_features=2, # (layer,c)
            out_features= 1, # single weight per channel
            list_hidden_features=[64, 64],
            omega_0=20.0,
        )

