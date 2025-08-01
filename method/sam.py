import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    Implements a two-step update to minimize sharpness of the loss landscape.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        # 确保 rho 非负
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # 默认参数字典，存放 rho 和 adaptive 等设置
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        # 调用父类构造函数初始化 Optimizer
        super(SAM, self).__init__(params, defaults)

        # base_optimizer 是实际执行参数更新的优化器（如 SGD、Adam）
        # 这里将 param_groups 传给 base_optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # 同步 param_groups，以便在 first_step/second_step 中使用相同参数组
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        SAM 的第一步：在梯度方向 "攀爬" 到局部最大点 (w + e_w)。
        """
        # 计算所有参数梯度的 L2 范数，用于缩放扰动
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # 计算缩放系数：rho / grad_norm
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 保存原始参数，以便后续还原
                self.state[p]["old_p"] = p.data.clone()
                # 计算扰动 e_w
                # 如果 adaptive=True，则使用 p 的幅度自适应缩放
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # 在原参数上加上扰动
                p.add_(e_w)

        # 是否清空梯度
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        SAM 的第二步：先还原到原始参数，再执行 base_optimizer.step() 更新。
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 将参数恢复到 w
                p.data = self.state[p]["old_p"]

        # 用 base_optimizer 执行实际的参数更新
        self.base_optimizer.step()

        # 是否清空梯度
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        重写 Optimizer.step()，强制要求提供 closure，以便二次前向后向。
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # closure 必须在开启 grad environment 下调用
        closure = torch.enable_grad()(closure)

        # 第一步扰动
        self.first_step(zero_grad=True)
        # 重新计算梯度
        closure()
        # 第二步更新
        self.second_step()

    def _grad_norm(self):
        """
        计算所有参数梯度的 L2 范数，用于缩放第一步扰动。
        如果 adaptive 模式，则对 p.grad * |p| 计算范数。
        """
        # 找一个共享设备，确保所有张量在同一 device
        shared_device = self.param_groups[0]["params"][0].device
        # 收集各参数的梯度范数
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
        # 重写加载 state_dict，以确保 base_optimizer.param_groups 与 self.param_groups 同步
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
