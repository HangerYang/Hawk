# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP32 master-weight Adam helpers for bf16 DDP training.

Matches the important DeepSpeed ZeRO behavior: model params stay bf16 for
compute/comms, while Adam moments / master weights stay float32.
"""

from collections import defaultdict
from typing import List

import torch


class FP32StateAdamW(torch.optim.Optimizer):
    """AdamW with fp32 master weights (FSDP-friendly).

    Maintains fp32 master copies of all parameters (in optimizer state).
    On each step:
      1. Cast bf16 gradients to fp32.
      2. Clip fp32 grad norm.
      3. Adam update on fp32 master weights.
      4. Copy fp32 master -> bf16 model params.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        super().__init__(params, defaults)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["master_param"] = p.data.detach().clone().to(torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_fp32_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if state["exp_avg"].dtype != torch.float32:
                    state["exp_avg"] = state["exp_avg"].to(torch.float32)
                if state["exp_avg_sq"].dtype != torch.float32:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.float32)
                if state["master_param"].dtype != torch.float32:
                    state["master_param"] = state["master_param"].to(torch.float32)

                fp32_grad = p.grad.detach().to(torch.float32)
                state["_fp32_grad"] = fp32_grad
                all_fp32_grads.append(fp32_grad)

        if self.max_grad_norm > 0 and all_fp32_grads:
            total_norm_sq = sum(g.norm().pow(2) for g in all_fp32_grads)
            total_norm = total_norm_sq.sqrt()
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            clip_coef_clamped = min(clip_coef.item(), 1.0)
            if clip_coef_clamped < 1.0:
                for g in all_fp32_grads:
                    g.mul_(clip_coef_clamped)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = state.pop("_fp32_grad")
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                master_param = state["master_param"]

                state["step"] += 1
                step_t = state["step"].item()

                if weight_decay != 0:
                    master_param.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                master_param.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(master_param.to(p.dtype))
                p.grad = None

        return loss


class FP32MasterWeightOptimizer(torch.optim.Optimizer):
    """Wrap any torch optimizer with fp32 master weights (DDP path).

    1. Clone bf16 model params -> fp32 masters.
    2. On step(): cast grads up, clip in fp32, step inner Adam, copy back.
    """

    def __init__(
        self,
        bf16_params: List[torch.Tensor],
        inner_optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
    ):
        # bf16_params is either a flat list (one group) or a list of lists,
        # one per inner param group -- used for an LR split (e.g. the row
        # compressor at its own LR alongside the drafter).
        if bf16_params and isinstance(bf16_params[0], (list, tuple)):
            groups = [list(g) for g in bf16_params]
        else:
            groups = [list(bf16_params)]
        if len(groups) != len(inner_optimizer.param_groups):
            raise ValueError(
                f"got {len(groups)} bf16 param groups but "
                f"{len(inner_optimizer.param_groups)} inner optimizer groups"
            )
        self._bf16_params = [p for group in groups for p in group]
        self._fp32_params: List[torch.Tensor] = [
            p.detach().clone().to(torch.float32).requires_grad_(True)
            for p in self._bf16_params
        ]
        offset = 0
        for group, inner_group in zip(groups, inner_optimizer.param_groups):
            inner_group["params"] = self._fp32_params[offset : offset + len(group)]
            offset += len(group)
        inner_optimizer.state = defaultdict(dict)

        self._inner = inner_optimizer
        self.max_grad_norm = max_grad_norm

        self._initializing = True
        super().__init__(self._fp32_params, inner_optimizer.defaults)
        self._initializing = False

        self.param_groups = self._inner.param_groups
        self.state = self._inner.state

    def step(self, closure=None):
        with torch.no_grad():
            for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_params):
                if bf16_p.grad is not None:
                    fp32_p.grad = bf16_p.grad.detach().to(torch.float32)
                    bf16_p.grad = None
                else:
                    fp32_p.grad = None

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._fp32_params, self.max_grad_norm)

        loss = self._inner.step(closure)

        with torch.no_grad():
            for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_params):
                bf16_p.data.copy_(fp32_p.data.to(bf16_p.dtype))

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for bf16_p in self._bf16_params:
            if set_to_none:
                bf16_p.grad = None
            elif bf16_p.grad is not None:
                bf16_p.grad.zero_()
        for fp32_p in self._fp32_params:
            if set_to_none:
                fp32_p.grad = None
            elif fp32_p.grad is not None:
                fp32_p.grad.zero_()

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        if getattr(self, "_initializing", True):
            return super().add_param_group(param_group)
        return self._inner.add_param_group(param_group)

    def __repr__(self):
        return f"FP32MasterWeightOptimizer({self._inner})"


# Back-compat aliases used by OnlineDFlashTrainer.
_FP32StateAdamW = FP32StateAdamW
_FP32MasterWeightOptimizer = FP32MasterWeightOptimizer
