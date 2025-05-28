import math

import torch
from torch import nn


class GaussianMask2D(nn.Module):
    """Generates a 2D Gaussian mask with learnable parameters.

    The mask is defined by its standard deviations in the x and y directions.
    The mask is applied to a grid of coordinates, and the output is a 2D tensor
    representing the Gaussian distribution.
    The mask is constrained to have a minimum variance, and the size of the mask
    is determined by the maximum kernel span in both x and y directions.
    The mask is generated using the formula:
        mask(x, y) = exp(-0.5 * ((x^2 / var_x) + (y^2 / var_y)))
    where var_x and var_y are the variances in the x and y directions, respectively.
    The mask is applied to the coordinates (x, y) and thresholded to create a binary mask.
    """

    def __init__(
        self,
        initial_sigma_x: float = 0.5,
        initial_sigma_y: float = 0.5,
        threshold: float = 0.1,
        max_kernel_span_x: int = 11,
        max_kernel_span_y: int = 11,
    ) -> None:
        """Initialize the GaussianMask2D class.

        Args:
            initial_sigma_x (float): Initial standard deviation for x.
            initial_sigma_y (float): Initial standard deviation for y.
            threshold (float): Threshold for the Gaussian mask.
            max_kernel_span_x (int): Maximum kernel span in x direction.
            max_kernel_span_y (int): Maximum kernel span in y direction.

        """
        super().__init__()
        # --- Input Validation ---
        if not (0.0 < threshold < 1.0):  # Threshold must be (0,1) for log(Tm)
            msg = f"Threshold must be in the range (0.0, 1.0), got {threshold}"
            raise ValueError(msg)
        if max_kernel_span_x <= 1 or max_kernel_span_y <= 1:
            msg = f"max_kernel_span_x ({max_kernel_span_x}) and max_kernel_span_y ({max_kernel_span_y}) must be greater than 1"
            raise ValueError(msg)
        if initial_sigma_x <= 0.0 or initial_sigma_y <= 0.0:
            msg = f"initial_sigma_x ({initial_sigma_x}) and initial_sigma_y ({initial_sigma_y}) must be positive"
            raise ValueError(msg)

        # --- Store Configuration Attributes ---
        self.threshold: float = threshold
        self.log_threshold: float = math.log(self.threshold)
        self.max_kernel_span_x: int = max_kernel_span_x
        self.max_kernel_span_y: int = max_kernel_span_y

        step_size_x: float = 2 / (max_kernel_span_x - 1)
        step_size_y: float = 2 / (max_kernel_span_y - 1)
        self.log_min_var_x: float = math.log((step_size_x**2) / (-2 * math.log(threshold)))
        self.log_min_var_y: float = math.log((step_size_y**2) / (-2 * math.log(threshold)))
        # --- Learnable log_var (log of variance) parameters ---
        initial_var_x: float = initial_sigma_x**2
        initial_var_y: float = initial_sigma_y**2
        self.log_var_x = nn.Parameter(torch.tensor(math.log(initial_var_x), dtype=torch.float32))
        self.log_var_y = nn.Parameter(torch.tensor(math.log(initial_var_y), dtype=torch.float32))

        # --- Initial size of the Gaussian mask ---
        self.initial_size_x: float = math.sqrt(-2 * initial_var_x * self.log_threshold)
        self.initial_size_y: float = math.sqrt(-2 * initial_var_y * self.log_threshold)

    def apply_var_constraints(self) -> None:
        """Apply the minimum variance constraint to log_var parameters."""
        with torch.no_grad():
            min_val_x_tensor = torch.tensor(self.log_min_var_x, device=self.log_var_x.device, dtype=self.log_var_x.dtype)
            min_val_y_tensor = torch.tensor(self.log_min_var_y, device=self.log_var_y.device, dtype=self.log_var_y.dtype)
            self.log_var_x.data.clamp_(min=min_val_x_tensor)
            self.log_var_y.data.clamp_(min=min_val_y_tensor)

    def get_mask_size(self) -> torch.Tensor:
        """Get the size of the Gaussian mask."""
        # half-widths in normalized units
        hw_x = torch.sqrt(-2 * torch.exp(self.log_var_x) * self.log_threshold)
        hw_y = torch.sqrt(-2 * torch.exp(self.log_var_y) * self.log_threshold)

        # scale to pixel-counts (full width)
        size_x: torch.Tensor = (hw_x / self.initial_size_x) * self.max_kernel_span_x
        size_y: torch.Tensor = (hw_y / self.initial_size_y) * self.max_kernel_span_y

        # stack as [height, width]
        return torch.stack((size_x, size_y), dim=0)

    def forward(self, y_coords: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        """Generate the 2D Gaussian mask, assuming mu=0."""
        # 1) clamp σ
        self.apply_var_constraints()

        # 2) compute variances
        var_x = torch.exp(self.log_var_x).clamp(min=1e-12)
        var_y = torch.exp(self.log_var_y).clamp(min=1e-12)

        # 3) build quarter‐mask on positive coords
        xg = x_coords.unsqueeze(0) # 1×(N//2+1)
        yg = y_coords.unsqueeze(1) # (N//2+1)×1
        q = torch.exp(-0.5*(yg**2/var_y + xg**2/var_x))
        q = torch.where(q >= self.threshold, q, torch.zeros_like(q))

        # 4) mirror to full mask:
        #    -(y,x) axes.   remove the duplicated zero‐line when concatenating.
        top    = torch.cat([q.flip(0)[1:], q], dim=0)      # full rows
        bottom = top.flip(1)                                # full columns
        full   = torch.cat([top, bottom[:,1:]], dim=1)      # shape N×N

        return full



class SigmoidMask1D(nn.Module):
    """1D Sigmoid mask with learnable offset μ' and 'temperature' τ (scale).

    Implements Eq. (5) & (10) of DNArch and guarantees ≥1 channel survives.
    """

    def __init__(
        self,
        max_span: int,
        initial_offset: float = 0.8,
        initial_temperature: float = 1.0,  # this is τ
        threshold: float = 0.1,
    ) -> None:
        """Initialize the SigmoidMask1D class.

        Args:
            max_span (int): Maximum span of the mask.
            initial_offset (float): Initial offset for the sigmoid function.
            initial_temperature (float): Initial temperature for the sigmoid function.
            threshold (float): Threshold for the sigmoid mask.

        """
        super().__init__()
        max_threshold = 0.5
        if not (0.0 < threshold < max_threshold):
            msg = f"threshold must be in (0,0.5), got {threshold}"
            raise ValueError(msg)
        if max_span < 1:
            msg = f"max_span must be ≥ 1, got {max_span}"
            raise ValueError(msg)
        if not (0.0 < initial_offset <= 1.0):
            msg = f"initial_offset must be in (0,1], got {initial_offset}"
            raise ValueError(msg)
        if initial_temperature <= 0:
            msg = f"initial_temperature must be > 0, got {initial_temperature}"
            raise ValueError(msg)

        self.max_span = max_span
        self.threshold = threshold
        # one channel = one step in [0,1]
        self.step = 1.0 / max_span

        # solve 1 - σ(τ(x_T - μ)) = T  ⇒  τ(x_T - μ) = -log(1/(1-T) - 1)
        self.logit_inv = math.log((1 / (1 - threshold)) - 1)

        # learnable params
        self.offset = nn.Parameter(torch.tensor(initial_offset, dtype=torch.float32))  # μ'
        self.temperature = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))  # τ

        # stability floors
        self.min_temperature = 1e-6
        self.max_temperature = 100.0

    def apply_var_constraints(self) -> None:
        """Clamp so that x_T ≥ 0 (ensuring channel 0 survives), and τ>0."""
        with torch.no_grad():
            self.offset.data.clamp_(min=self.step, max=1.0)  # μ' ∈ (0,1]
            # keep τ positive
            self.temperature.data.clamp_(min=self.min_temperature, max=self.max_temperature)

    def get_mask_size(self) -> torch.Tensor:
        """Get the size of the Sigmoid mask.

        The size is determined by the offset and temperature parameters.
        """
        self.apply_var_constraints()
        size_x = self.offset - (1.0 / self.temperature) * self.logit_inv
        size_x.clamp_(min=0.0, max=1.0)  # ensure size is in [0,1]
        return torch.tensor(self.max_span * size_x, dtype=torch.float32)

    def forward(self, x_coords: torch.Tensor) -> torch.Tensor:
        """Generate the 1D Sigmoid mask.

        The mask is defined by the sigmoid function:
            mask(x) = 1 - σ(τ(x - μ))

        Args:
            x_coords (torch.Tensor): 1D tensor of coordinates [0,1].
        returns:
            torch.Tensor: 1D tensor representing the mask.

        """
        mask_val = 1.0 - torch.sigmoid(self.temperature * (x_coords - self.offset))
        return torch.where(mask_val >= self.threshold, mask_val, torch.zeros_like(mask_val))

class RadialSigmoidMask2D(nn.Module):
    """Radial (isotropic) 2D sigmoid mask for Fourier downsampling.

    Receives fx, fy grids (half‐plane for rFFT) in forward, so you can
    precompute those once and reuse them to save memory.
    """

    def __init__(
        self,
        initial_offset: float = 0.8,      # μ in normalized radius space
        initial_temperature: float = 1.0, # τ (scale)
        threshold: float = 0.1,           # T ∈ (0,0.5)
        max_radius: float = math.sqrt(2), # max r you’ll ever feed in
        grid_size: int = 64,              # optional, just for computing radial_step
    ) -> None:
        super().__init__()
        # --- Validation ---
        if not (0.0 < threshold < 0.5):
            msg = f"threshold must be in (0,0.5), got {threshold}"
            raise ValueError(msg)
        if not (0.0 < initial_offset <= max_radius):
            msg = f"initial_offset must be in (0,√2], got {initial_offset}"
            raise ValueError(msg)
        if initial_temperature <= 0:
            msg = f"initial_temperature must be >0, got {initial_temperature}"
            raise ValueError(msg)
        if grid_size < 2:
            msg = f"grid_size must be ≥2, got {grid_size}"
            raise ValueError(msg)

        self.threshold = threshold
        self.logit_inv = math.log(1/(1-threshold) - 1)   # solves 1−σ(τ(x−μ))=T
        self.max_radius = max_radius

        # radial_step = smallest nonzero r on your fx/fy grid:
        # fx step = 1/(grid_size/2), fy step = 2/(grid_size-1)
        df_x = 1.0 / (grid_size // 2)
        df_y = 2.0 / (grid_size - 1)
        self.radial_step = min(df_x, df_y)

        # learnable
        self.offset = nn.Parameter(torch.tensor(initial_offset, dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))

        # stability clamps
        self.min_tau, self.max_tau = 1e-6, 1e3

    def apply_constraints(self) -> None:
        """Ensure μ ≥ radial_step (so r=0 survives) and τ stays positive."""
        with torch.no_grad():
            self.offset.data.clamp_(min=self.radial_step, max=self.max_radius)
            self.temperature.data.clamp_(min=self.min_tau, max=self.max_tau)

    def get_mask_size(self) -> torch.Tensor:
        """Continuous cutoff radius.

        x_T = μ - (1/τ)*logit_inv, clamped to [0, max_radius].
        """
        self.apply_constraints()
        x_T = self.offset - (1.0 / self.temperature) * self.logit_inv
        return torch.tensor(x_T.clamp(0.0, self.max_radius), dtype=torch.float32)

    def forward(self, fy: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        """Generate the radial sigmoid mask.

        Args:
          fy: Tensor of shape (..., R, 1) or (R, 1) with values in [-1,1]
          fx: Tensor of shape (..., 1, R//2+1) or (1, R//2+1) with values in [0,1].

        Returns:
          mask: same shape as broadcast(fy,fx), values in [0,1], zeroed below threshold.

        """
        # 1) enforce μ,τ constraints
        self.apply_constraints()

        # 2) build quarter‐plane coords
        C = self.R//2 + 1
        # fx: [0,1] len=C
        fx_q = torch.linspace(0.0, 1.0, steps=C, device=self.offset.device)
        # fy: [0,1] len=C  (only nonnegatives)
        fy_q = torch.linspace(0.0, 1.0, steps=C, device=self.offset.device)

        fx2d = fx_q.unsqueeze(0).expand(C, -1)   # (C, C)
        fy2d = fy_q.unsqueeze(1).expand(-1, C)   # (C, C)

        # 3) compute radial mask on quarter
        r_q = torch.sqrt(fx2d**2 + fy2d**2)
        q = 1.0 - torch.sigmoid(self.temperature * (r_q - self.offset))
        q = torch.where(q >= self.threshold, q, torch.zeros_like(q))

        # 4) mirror across fy=0 to get full half-plane (R×C)
        #   - drop the duplicated zero‐row at index 0 when mirroring
        top    = torch.cat([q.flip(0)[1:], q], dim=0)    # (R, C)

        return top
