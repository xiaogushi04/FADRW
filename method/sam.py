import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    Implements a two-step update to minimize sharpness of the loss landscape.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        # Ensure rho is non-negative
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # Default parameter dictionary containing rho and adaptive settings
        defaults = dict(rho=rho, adaptive=adaptive,** kwargs)
        # Call parent class constructor to initialize Optimizer
        super(SAM, self).__init__(params, defaults)

        # The base_optimizer is the actual optimizer performing parameter updates (e.g., SGD, Adam)
        # Pass param_groups to the base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # Synchronize param_groups to ensure consistency in first_step/second_step
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step of SAM: "Climb" to the local maximum in the gradient direction (w + e_w).
        """
        # Calculate L2 norm of all parameter gradients for scaling perturbations
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # Calculate scaling factor: rho / grad_norm
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save original parameters for later restoration
                self.state[p]["old_p"] = p.data.clone()
                # Calculate perturbation e_w
                # If adaptive=True, use parameter magnitude for adaptive scaling
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # Add perturbation to original parameters
                p.add_(e_w)

        # Whether to clear gradients
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step of SAM: First restore original parameters, then perform base_optimizer.step() update.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore parameters to original values (w)
                p.data = self.state[p]["old_p"]

        # Perform actual parameter update using base optimizer
        self.base_optimizer.step()

        # Whether to clear gradients
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Override Optimizer.step(), requiring closure to be provided for double forward-backward passes.
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # Closure must be called in a grad-enabled environment
        closure = torch.enable_grad()(closure)

        # First step with perturbation
        self.first_step(zero_grad=True)
        # Recalculate gradients
        closure()
        # Second step update
        self.second_step()

    def _grad_norm(self):
        """
        Calculate L2 norm of all parameter gradients for scaling first-step perturbations.
        In adaptive mode, calculate norm for p.grad * |p|.
        """
        # Find a shared device to ensure all tensors are on the same device
        shared_device = self.param_groups[0]["params"][0].device
        # Collect gradient norms for all parameters
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                .norm(p=2)
                .to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        # Override state_dict loading to ensure base_optimizer.param_groups syncs with self.param_groups
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
