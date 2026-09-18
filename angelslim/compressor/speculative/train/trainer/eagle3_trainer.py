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

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import Trainer

from angelslim.utils.lazy_imports import deepspeed

from ...utils import padding

logger = logging.getLogger(__name__)


class Eagle3Trainer(Trainer, ABC):
    """
    EAGLE3 Trainer for speculative decoding training.

    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(self, draft_model: nn.Module, length: int, **kwargs):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            **kwargs: Additional arguments passed to parent Trainer
        """
        draft_model_config = kwargs.pop("draft_model_config", None)
        self.skew_kl_loss_weight = float(
            getattr(draft_model_config, "skew_kl_loss_weight", 0.0) or 0.0
        )
        self.skew_kl_alpha = float(
            getattr(draft_model_config, "skew_kl_alpha", 0.1) or 0.1
        )
        self.skew_kl_direction = str(
            getattr(draft_model_config, "skew_kl_direction", "reverse") or "reverse"
        )
        self.skew_kl_loss_mode = str(
            getattr(draft_model_config, "skew_kl_loss_mode", "additive")
            or "additive"
        )
        self.skew_kl_stage_weights = list(
            getattr(draft_model_config, "skew_kl_stage_weights", []) or []
        )
        # Multi-depth CE: apply the shared lm_head to each layer's output.
        # Weights indexed [h0_weight, h1_weight, ..., h_last_weight].
        self.multi_depth_ce_weights: List[float] = list(
            getattr(draft_model_config, "multi_depth_ce_weights", []) or []
        )
        # Branch distillation. When the draft's top-1 is NOT the teacher's top-1
        # but sits inside the teacher's top-k, fork one extra draft step onto
        # that token and train it against what the teacher says *after taking
        # it* — which costs a second teacher forward on the substituted
        # sequence, since the stored target_logits only cover the real one.
        self.branch_distill_loss_weight: float = float(
            getattr(draft_model_config, "branch_distill_loss_weight", 0.0) or 0.0
        )
        self.branch_distill_target_top_k: int = int(
            getattr(draft_model_config, "branch_distill_target_top_k", 3) or 3
        )
        self.branch_distill_prob_ratio_threshold: float = float(
            getattr(draft_model_config, "branch_distill_prob_ratio_threshold", 0.0)
            or 0.0
        )
        self.branch_distill_objective: str = str(
            getattr(draft_model_config, "branch_distill_objective", "ce") or "ce"
        ).lower()
        # Curriculum: hold the branch term at 0 for the first warmup steps, then
        # ramp it linearly to the configured weight. Early on the branch loss
        # competes with main CE, which is still improving fast.
        self.branch_distill_warmup_steps: int = int(
            getattr(draft_model_config, "branch_distill_warmup_steps", 0) or 0
        )
        self.branch_distill_ramp_steps: int = int(
            getattr(draft_model_config, "branch_distill_ramp_steps", 0) or 0
        )
        # Synthetic counterfactuals: besides the natural "draft picked the
        # teacher's rank-2 token" branches, substitute the teacher's rank-2
        # token at a sampled set of other positions (draft wrong on some other
        # token, or draft correct), capped relative to the natural count.
        self.branch_distill_synthetic: bool = bool(
            getattr(draft_model_config, "branch_distill_synthetic", False)
        )
        # Synthetic branches get their OWN denominator (see _branch_loss), so
        # turning them on no longer halves the coefficient on the natural ones.
        # This is how much of the natural term's scale they are worth.
        self.branch_distill_synthetic_weight: float = float(
            getattr(draft_model_config, "branch_distill_synthetic_weight", 1.0)
        )
        self.branch_distill_synthetic_ratio: float = float(
            getattr(draft_model_config, "branch_distill_synthetic_ratio", 1.0)
        )
        # "change" objective only: weight each branch position by how much the
        # teacher's logits actually moved (RMS of the target delta), normalised
        # so the mean active weight stays ~1. Branches that barely change the
        # future then contribute proportionally less supervision.
        self.branch_distill_change_delta_weight: bool = bool(
            getattr(draft_model_config, "branch_distill_change_delta_weight", False)
        )
        if self.branch_distill_objective not in ("ce", "change"):
            raise ValueError(
                "branch_distill_objective must be 'ce' or 'change' "
                f"(got {self.branch_distill_objective!r})"
            )
        # Leading TTT steps that get a branch. Each one is a full extra teacher
        # forward, so the default is the first step only.
        self.branch_distill_steps: int = int(
            getattr(draft_model_config, "branch_distill_steps", 1) or 1
        )
        # Every branch position is substituted into one sequence and scored in a
        # single teacher forward, so a position's prefix can contain an earlier
        # position's substitution. Dense signal, slightly contaminated context.
        _bd_draft_k = getattr(draft_model_config, "branch_distill_top_k", 1)
        if self.branch_distill_loss_weight > 0.0 and _bd_draft_k not in (None, 1):
            raise ValueError(
                "branch_distill_top_k must be 1: the branch is taken on the "
                f"draft's top-1 token only (got {_bd_draft_k})"
            )
        super().__init__(model=draft_model, **kwargs)
        self.length = length
        self._train_start_time = None
        self._train_pending_log: dict = {}
        self._train_pending_log_count: int = 0
        self._eval_pending_log: dict = {}
        self._eval_pending_log_count: int = 0
        self._logged_aux_losses = False
        self._logged_branch_distill = False
        # Set by create_optimizer() for non-DeepSpeed bf16 DDP (ZeRO-equivalent).
        self._fp32_optimizer = None

    @staticmethod
    def _stage_weight(weights: List[float], idx: int, default: float = 1.0) -> float:
        if not weights:
            return default
        if idx < len(weights):
            return float(weights[idx])
        return float(weights[-1])

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=values.dtype, device=values.device)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _shift_left(tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Advance a sequence tensor n positions, zero-padding the tail."""
        for _ in range(n):
            tensor = padding(tensor, left=False)
        return tensor

    @staticmethod
    def _fork_cache(cache):
        """Copy the nested list structure of cache_hidden, sharing the tensors.

        The branch step must not append its keys/values to the rollout the main
        loop keeps using; the tensors themselves are never written in place.
        """
        if isinstance(cache, list):
            return [Eagle3Trainer._fork_cache(c) for c in cache]
        return cache

    def branch_teacher_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher logits for a token sequence carrying branch substitutions.

        Returns [B, S, V] aligned to `input_ids` (position p holds the
        distribution for token p+1). Substituting a token at position a changes
        every hidden state from a onwards, so no amount of stored teacher output
        answers this -- it takes a live target. The online trainers already hold
        one; the offline ones load one just for this forward (see
        `_OfflineBranchMixin` in offline_eagle3_trainer.py) and keep reading the
        real sequence's teacher output from disk.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot run branch distillation: it has no "
            "target model in memory to re-score the substituted sequence. Use "
            "an online trainer or set branch_distill_loss_weight to 0."
        )

    def _vistoken_param_split(self):
        """(compressor params, drafter params, compressor lr) or None.

        The row compressor is ~40k params learning routing from scratch; the
        drafter is a pretrained-ish 2H block being nudged. One LR cannot serve
        both, so they get their own groups.
        """
        compressor = getattr(self.draft_model, "vistoken", None)
        if compressor is None:
            return None
        names = {id(p) for p in compressor.parameters()}
        hot, rest = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (hot if id(p) in names else rest).append(p)
        if not hot:
            return None
        return hot, rest, float(compressor.cfg.lr)

    def create_optimizer(self, model=None):
        """Use FP32 master Adam under plain DDP to match DeepSpeed ZeRO moments.

        DeepSpeed ZeRO keeps Adam state in FP32 even when params are bf16.
        HF AdamW + bf16 DDP keeps moments in bf16 and plateaus. Mirror ZeRO
        with FP32MasterWeightOptimizer when DeepSpeed is off.
        """
        if self.is_deepspeed_enabled:
            split = self._vistoken_param_split()
            if split is None:
                return super().create_optimizer()
            # ZeRO keeps FP32 moments itself; all we need is the LR split, and
            # the ZeRO configs here leave the optimizer to HF/torch.
            from torch.optim import AdamW as _AdamW

            hot, rest, hot_lr = split
            args = self.args
            self.optimizer = _AdamW(
                [
                    {"params": rest, "lr": args.learning_rate},
                    {"params": hot, "lr": hot_lr, "weight_decay": 0.0},
                ],
                lr=args.learning_rate,
                betas=(
                    getattr(args, "adam_beta1", 0.9),
                    getattr(args, "adam_beta2", 0.999),
                ),
                eps=getattr(args, "adam_epsilon", 1e-8),
                weight_decay=args.weight_decay,
            )
            logger.info(
                "Eagle3 (DeepSpeed): LR split -- drafter %g, vistoken compressor %g.",
                args.learning_rate,
                hot_lr,
            )
            return self.optimizer

        from torch.optim import AdamW

        from .fp32_master_optimizer import FP32MasterWeightOptimizer, FP32StateAdamW

        split = self._vistoken_param_split()

        if self.is_fsdp_enabled:
            args = self.args
            if split is not None:
                hot, rest, hot_lr = split
                param_groups = [
                    {"params": rest},
                    {"params": hot, "lr": hot_lr, "weight_decay": 0.0},
                ]
            else:
                param_groups = [
                    {"params": [p for p in self.model.parameters() if p.requires_grad]}
                ]
            optimizer = FP32StateAdamW(
                param_groups,
                lr=args.learning_rate,
                betas=(
                    getattr(args, "adam_beta1", 0.9),
                    getattr(args, "adam_beta2", 0.999),
                ),
                eps=getattr(args, "adam_epsilon", 1e-8),
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
            )
            self.optimizer = optimizer
            logger.info(
                "Eagle3: using FP32StateAdamW (FSDP) so Adam moments match ZeRO FP32."
            )
            return self.optimizer

        bf16_params = [p for p in self.model.parameters() if p.requires_grad]
        if not bf16_params:
            return super().create_optimizer()

        args = self.args
        if split is not None:
            hot, rest, hot_lr = split
            bf16_params = [rest, hot]
            adam_params = [
                {"params": rest, "lr": args.learning_rate},
                {"params": hot, "lr": hot_lr, "weight_decay": 0.0},
            ]
            logger.info(
                "Eagle3: LR split -- drafter %g (%d tensors), vistoken "
                "compressor %g (%d tensors).",
                args.learning_rate,
                len(rest),
                hot_lr,
                len(hot),
            )
        else:
            adam_params = bf16_params
        inner_optimizer = AdamW(
            adam_params,
            lr=args.learning_rate,
            betas=(
                getattr(args, "adam_beta1", 0.9),
                getattr(args, "adam_beta2", 0.999),
            ),
            eps=getattr(args, "adam_epsilon", 1e-8),
            weight_decay=args.weight_decay,
        )
        fp32_opt = FP32MasterWeightOptimizer(
            bf16_params=bf16_params,
            inner_optimizer=inner_optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        self._fp32_optimizer = fp32_opt
        self.optimizer = fp32_opt
        logger.info(
            "Eagle3: using FP32MasterWeightOptimizer (DDP) so Adam moments match "
            "DeepSpeed ZeRO FP32 optimizer state."
        )
        return self.optimizer

    def _clip_grad_norm(self, *args, **kwargs):
        """Skip HF bf16 grad clip when FP32-master optimizers clip internally."""
        if self._fp32_optimizer is not None:
            return torch.tensor(0.0)

        from .fp32_master_optimizer import FP32StateAdamW

        optimizer = self.optimizer
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if isinstance(optimizer, FP32StateAdamW):
            return torch.tensor(0.0)
        return super()._clip_grad_norm(*args, **kwargs)

    def train(self, *args, **kwargs):
        """Override train method to record training start time for estimating remaining time."""
        self._train_start_time = time.time()
        return super().train(*args, **kwargs)

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Merge acc/ploss accumulators with the base Trainer's loss log.
        """
        if "loss" in logs and self._train_pending_log:
            train_count = max(self._train_pending_log_count, 1)
            acc_ploss = {k: v / train_count for k, v in self._train_pending_log.items()}
            merged = {}

            # step
            max_steps = 0
            if self.state is not None:
                global_step = self.state.global_step
                max_steps = self.state.max_steps
                merged["step"] = global_step

            # epoch
            if "epoch" in logs:
                merged["epoch"] = logs["epoch"]
            if "loss" in logs:
                merged["loss"] = logs["loss"]
            if "grad_norm" in logs:
                merged["grad_norm"] = logs["grad_norm"]

            if "learning_rate" in logs:
                merged["lr"] = logs["learning_rate"]

            # train acc/ploss
            merged.update(acc_ploss)

            # eval acc/ploss — merged when a training log fires
            if self._eval_pending_log:
                eval_count = max(self._eval_pending_log_count, 1)
                merged.update({k: v / eval_count for k, v in self._eval_pending_log.items()})
                self._eval_pending_log.clear()
                self._eval_pending_log_count = 0

            # remaining_time
            if (
                self.state is not None
                and self._train_start_time is not None
                and global_step > 0
                and max_steps > 0
            ):
                elapsed = time.time() - self._train_start_time
                time_per_step = elapsed / global_step
                remaining_seconds = int(time_per_step * (max_steps - global_step))
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                merged["remaining_time"] = f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"

            self._train_pending_log.clear()
            self._train_pending_log_count = 0
            super().log(merged, start_time)
        else:
            super().log(logs, start_time)

    @property
    def draft_model(self) -> nn.Module:
        """Underlying draft module (unwrap DDP / Accelerate / DeepSpeed)."""
        model = self.model
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            try:
                return accelerator.unwrap_model(model)
            except Exception:
                pass
        return model.module if hasattr(model, "module") else model

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
        return_outputs: bool = False,
    ) -> Tuple[List[torch.Tensor], List, List[float]]:
        """
        Compute the training loss for the model.

        Args:
            model: The model for which to compute the loss
            inputs: Input data dictionary with input_ids, attention_mask,
                loss_mask, position_ids
            num_items_in_batch: Number of items in batch (unused)
            return_outputs: Whether to return model outputs (unused)

        Returns:
            Tuple of (prediction_losses, value_losses, accuracies) for each step
        """
        data_for_draft_model = self.prepare_data_for_draft_model(inputs)

        attention_mask = data_for_draft_model["attention_mask"]  # Batch x Seq
        position_ids = data_for_draft_model["position_ids"]
        input_ids = data_for_draft_model["input_ids"]  # Batch x Seq
        target_logits = data_for_draft_model["target_logits"]  # Batch x Seq x Vocab
        loss_mask = data_for_draft_model["loss_mask"]  # Batch x Seq x 1
        hidden_states = data_for_draft_model["hidden_states"]  # Batch x Seq x Hidden
        hidden_states = self.down_project_hidden_states(hidden_states)
        attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
            hidden_states, attention_mask, position_ids
        )
        loss = self.draft_model_training_time_test(
            input_ids,
            hidden_states,
            attention_mask,
            position_ids,
            target_logits,
            loss_mask,
            log_prefix="train",
        )

        return loss

    @abstractmethod
    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training.
        """
        pass

    def down_project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Down project hidden states for draft model training.
        """
        # Step 4: Prepare hidden states with gradient tracking
        if not hidden_states.requires_grad:
            hidden_states.requires_grad = True
        if os.environ.get("EAGLE3_RMS_PROBE") == "1":
            with torch.no_grad():
                _H = self.draft_model.hidden_size
                self._probe_aux_rms = [
                    c.detach().float().pow(2).mean().sqrt().item()
                    for c in hidden_states.split(_H, dim=-1)
                ]
        hidden_states = self.draft_model.combine_hidden_states(hidden_states)
        return hidden_states

    def prepare_attention_mask_and_position_ids(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attention mask for draft model training.
        """
        # Step 5: Prepare attention mask and position IDs
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if position_ids.ndim == 3:
                # MRoPE format: (3, batch, seq_len), keep as-is
                position_ids = position_ids.long()
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)

        attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        return attention_mask, position_ids

    def branch_weight_now(self) -> float:
        """Branch loss weight at the current step, after the curriculum ramp."""
        w = self.branch_distill_loss_weight
        if w <= 0.0:
            return 0.0
        warmup = self.branch_distill_warmup_steps
        ramp = self.branch_distill_ramp_steps
        if warmup <= 0 and ramp <= 0:
            return w
        step = int(getattr(self.state, "global_step", 0) or 0)
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return w
        return w * min(1.0, (step - warmup) / float(ramp))

    def _branch_decide(self, idx, draft, logits, target_logits, loss_mask):
        """Pick branch positions and re-ask the teacher what follows them.

        At TTT step `idx`, index j predicts absolute position a = j + idx + 2:
        `input_ids` and `target_logits` were advanced once before the loop and
        once per step. A position branches when the draft's top-1 differs from
        the teacher's top-1 yet appears in the teacher's top-k. Those tokens are
        substituted at their absolute positions in ONE copy of the original
        sequence, which is scored by a single teacher forward.

        Returns the branch mask, the branch tokens, and the teacher's
        continuation logit change relative to the real path — all aligned to
        the *next* step's index frame, which is where the branch step is
        actually run.
        """
        ctx = getattr(self, "_branch_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "branch distillation needs the unshifted token sequence; "
                "prepare_data_for_draft_model must set self._branch_ctx"
            )
        with torch.no_grad():
            draft_top1_d = logits.argmax(-1)
            # d2t stores target_id - draft_id, so add it back to get target ids.
            draft_top1 = draft_top1_d + draft.d2t[draft_top1_d]
            teacher_top1 = target_logits.argmax(-1)
            k = max(int(self.branch_distill_target_top_k), 1)
            teacher_topk = target_logits.topk(k, dim=-1).indices
            scorable = loss_mask[..., 0].bool()
            candidate_mask = (
                (teacher_topk == draft_top1[..., None]).any(-1)
                & (draft_top1 != teacher_top1)
                & scorable
            )
            # The token each branch substitutes. Natural branches take the
            # draft's own (wrong) top-1; synthetic ones take the teacher's
            # rank-2 token, which is only defined when k >= 2.
            sub_token = draft_top1
            synthetic_mask = None
            rank2_token = (
                teacher_topk[..., 1] if teacher_topk.shape[-1] > 1 else None
            )
            teacher_top1_logit = target_logits.gather(
                -1, teacher_top1[..., None]
            ).squeeze(-1).float()
            draft_token_logit = target_logits.gather(
                -1, draft_top1[..., None].to(target_logits.device)
            ).squeeze(-1).float()
            prob_ratio = (draft_token_logit - teacher_top1_logit).exp()

            offset = idx + 2
            orig_ids = ctx["input_ids"]
            seq_len = orig_ids.shape[1]
            cols = torch.arange(candidate_mask.shape[1], device=orig_ids.device)
            # Drop branches whose token would land past the end of the sequence.
            candidate_mask = candidate_mask & (cols + offset < seq_len)[None, :]
            threshold = max(float(self.branch_distill_prob_ratio_threshold), 0.0)
            branch_mask = candidate_mask
            if threshold > 0.0:
                branch_mask = branch_mask & (prob_ratio > threshold)
            survivor_ratio = prob_ratio[branch_mask]
            n_synthetic = 0
            if self.branch_distill_synthetic:
                if rank2_token is None:
                    raise ValueError(
                        "branch_distill_synthetic needs branch_distill_target_"
                        "top_k >= 2 to have a rank-2 token to substitute"
                    )
                # Every other scorable in-range position is eligible: draft
                # wrong on a token that is not the teacher's rank 2, or draft
                # correct. Sample enough to roughly match the natural branches.
                eligible = scorable & (cols + offset < seq_len)[None, :] & ~branch_mask
                budget = int(
                    round(
                        float(branch_mask.sum().item())
                        * max(self.branch_distill_synthetic_ratio, 0.0)
                    )
                )
                n_eligible = int(eligible.sum().item())
                budget = min(budget, n_eligible)
                if budget > 0:
                    # Random scores on eligible positions; take the top `budget`.
                    scores = torch.rand(
                        eligible.shape, device=eligible.device
                    ).masked_fill(~eligible, -1.0)
                    pick = torch.topk(scores.flatten(), budget).indices
                    chosen = torch.zeros_like(eligible.flatten())
                    chosen[pick] = True
                    chosen = chosen.view_as(eligible)
                    sub_token = torch.where(chosen, rank2_token, sub_token)
                    branch_mask = branch_mask | chosen
                    synthetic_mask = chosen
                    n_synthetic = budget
            self._branch_last_stats = {
                "candidates": float(candidate_mask.sum().item()),
                "survivors": float(branch_mask.sum().item()),
                "synthetic": float(n_synthetic),
                "ratio_mean": (
                    float(survivor_ratio.mean().item())
                    if survivor_ratio.numel()
                    else 0.0
                ),
                "ratio_median": (
                    float(survivor_ratio.median().item())
                    if survivor_ratio.numel()
                    else 0.0
                ),
            }
            if not bool(branch_mask.any()):
                return None

            sub_ids = orig_ids.clone()
            rows, at = torch.nonzero(branch_mask, as_tuple=True)
            sub_ids[rows, at + offset] = sub_token[rows, at].to(sub_ids.dtype)

            branch_logits = self.branch_teacher_logits(sub_ids).detach()
            # branch_logits[:, a] is the distribution for a + 1; the branch step
            # predicts it at index j = a - offset.
            branch_logits = self._shift_left(branch_logits, offset).to(
                target_logits.device
            )
            real_next_logits = self._shift_left(target_logits, 1).to(
                branch_logits.device
            )
            branch_top1 = branch_logits.argmax(-1)
            branch_in_logits = branch_logits[..., draft.t2d].float()
            real_in_logits = real_next_logits[..., draft.t2d].float()
            target_p = nn.Softmax(dim=2)(branch_in_logits)
            branch_centered = branch_in_logits - branch_in_logits.mean(
                dim=-1, keepdim=True
            )
            real_centered = real_in_logits - real_in_logits.mean(
                dim=-1, keepdim=True
            )
            target_delta = branch_centered - real_centered
            entropy = -(target_p * target_p.clamp_min(1e-9).log()).sum(
                -1, keepdim=True
            )
            invocab_mass = (
                torch.logsumexp(branch_in_logits, dim=-1)
                - torch.logsumexp(branch_logits, dim=-1).float()
            ).exp()[..., None]
            return {
                "mask": branch_mask,
                "synthetic_mask": synthetic_mask,
                "tokens": sub_token,
                "target_p": target_p,
                "in_draft_vocab": draft.t2d[branch_top1][..., None].int(),
                "entropy": entropy,
                "invocab_mass": invocab_mass,
                "target_delta": target_delta,
            }

    def _branch_loss(
        self,
        pending,
        draft,
        input_ids,
        hidden_states,
        cache_hidden,
        attention_mask,
        position_ids,
        loss_mask,
        base_logits,
    ):
        """Run the forked draft step and score its logit change.

        Called at the top of the step that follows the decision, so the rollout
        state (hidden_states, cache, aux injects) is exactly what the normal
        step consumed — the branch only swaps the input token.
        """
        mask = pending["mask"]
        branch_ids = torch.where(mask, pending["tokens"].to(input_ids.dtype), input_ids)
        branch_embeds = draft.embed_input_ids(branch_ids)
        saved_layer_outs = getattr(draft, "_last_layer_outs", None)
        branch_hidden, _ = draft.encode_layers(
            inputs_embeds=branch_embeds,
            hidden_states=hidden_states,
            cache_hidden=self._fork_cache(cache_hidden),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        # The main loop owns the rollout's per-layer outs; give them back.
        draft._last_layer_outs = saved_layer_outs
        branch_logits = draft.compute_logits(branch_hidden)
        position_mask = mask[..., None].int() * loss_mask
        if self.branch_distill_objective == "change":
            denom = position_mask.sum().clamp_min(1).to(branch_logits.dtype)
            branch_centered = branch_logits.float() - branch_logits.float().mean(
                dim=-1, keepdim=True
            )
            base_centered = base_logits.float() - base_logits.float().mean(
                dim=-1, keepdim=True
            )
            draft_delta = branch_centered - base_centered
            target_delta = pending["target_delta"].to(
                device=draft_delta.device, dtype=draft_delta.dtype
            )
            delta_mse = (draft_delta - target_delta).pow(2).mean(dim=2, keepdim=True)
            active = position_mask.to(delta_mse.dtype)
            per_pos_sq = target_delta.pow(2).mean(dim=2, keepdim=True)
            synthetic = pending.get("synthetic_mask")
            if synthetic is not None and not self.branch_distill_change_delta_weight:
                # Separate denominators. Sharing one made every synthetic result
                # unreadable: adding synthetic positions halved the coefficient
                # on the NATURAL branches too, so "synthetic changed nothing"
                # could equally mean "synthetic exactly repaid the dilution it
                # caused". Averaged apart, the natural term is untouched and
                # synthetic is genuinely added on top -- one variable, not two.
                syn = (synthetic[..., None].to(active.dtype)) * active
                nat = active - syn
                nat_loss = (nat * delta_mse).sum() / nat.sum().clamp_min(1.0)
                syn_loss = (syn * delta_mse).sum() / syn.sum().clamp_min(1.0)
                loss = nat_loss + self.branch_distill_synthetic_weight * syn_loss
                with torch.no_grad():
                    target_delta_rms = (active * per_pos_sq).sum() / denom
                    draft_delta_rms = (
                        active * draft_delta.pow(2).mean(dim=2, keepdim=True)
                    ).sum() / denom
                return (
                    loss,
                    float(mask.sum().item()),
                    target_delta_rms.sqrt().detach(),
                    draft_delta_rms.sqrt().detach(),
                    float(nat.sum().item()),
                )
            if self.branch_distill_change_delta_weight:
                # alpha_i = RMS(delta_T^i) / mean_active(RMS(delta_T)), so the
                # loss stays a weighted *average* -- same scale as unweighted,
                # which keeps the configured branch weight comparable.
                s = per_pos_sq.detach().clamp_min(0).sqrt()
                s_mean = (active * s).sum() / denom
                alpha = active * (s / s_mean.clamp_min(1e-6))
                loss = (alpha * delta_mse).sum() / alpha.sum().clamp_min(1e-6)
            else:
                loss = (active * delta_mse).sum() / denom
            with torch.no_grad():
                target_delta_rms = (active * per_pos_sq).sum() / denom
                draft_delta_rms = (
                    active * draft_delta.pow(2).mean(dim=2, keepdim=True)
                ).sum() / denom
            return (
                loss,
                float(mask.sum().item()),
                target_delta_rms.sqrt().detach(),
                draft_delta_rms.sqrt().detach(),
                float(denom.item()),
            )

        branch_logp = nn.LogSoftmax(dim=2)(branch_logits)
        position_mask = pending["in_draft_vocab"] * position_mask
        denom = position_mask.sum().clamp_min(1).to(branch_logp.dtype)
        ce = -torch.sum(position_mask * pending["target_p"] * branch_logp, dim=2).sum()
        with torch.no_grad():
            branch_entropy = (position_mask * pending["entropy"]).sum() / denom
            branch_mass = (position_mask * pending["invocab_mass"]).sum() / denom
        return (
            ce / denom,
            float(mask.sum().item()),
            branch_entropy.detach(),
            branch_mass.detach(),
            float(denom.item()),
        )

    def draft_model_training_time_test(
        self,
        input_ids,
        hidden_states,
        attention_mask,
        position_ids,
        target_logits,
        loss_mask,
        log_prefix="",
    ):
        _, seq_length, _ = hidden_states.shape

        # Step 6: Initialize containers for losses, accuracies and cache
        plosses, acces = [], []
        skew_kl_losses = []
        draft = self.draft_model  # unwrapped — _aux_inject lives here
        branch_losses: List[torch.Tensor] = []
        branch_denoms: List[float] = []
        plosses_active: List[torch.Tensor] = []
        active_counts: List[float] = []
        total_counts: List[float] = []
        branch_aux_metric1: List[torch.Tensor] = []
        branch_aux_metric2: List[torch.Tensor] = []
        branch_hits = 0.0
        branch_candidates = 0.0
        branch_survivors = 0.0
        branch_ratio_weighted_sum = 0.0
        branch_ratio_medians: List[float] = []
        branch_synthetic = 0.0
        pending_branch = None
        branch_weight = self.branch_weight_now() if log_prefix == "train" else 0.0
        # Zero weight means skip the fork entirely -- it costs a teacher forward.
        use_branch_distill = branch_weight > 0.0
        n_draft_layers = len(getattr(draft, "layers", []) or [])
        target_aux_tape = None
        sl1_layer_sum = [0.0] * max(n_draft_layers, 0)
        sl1_layer_count = [0] * max(n_draft_layers, 0)
        # How many speculative tokens (1-indexed) to include in the metric.
        sl1_token_budget = min(3, self.length)
        if hasattr(draft, "init_cache_hidden"):
            cache_hidden = draft.init_cache_hidden()
        else:
            cache_hidden = [[], []]

        # Step 7: Iterative speculative decoding training loop
        _probe_on = os.environ.get("EAGLE3_RMS_PROBE") == "1"
        probe_step_rms: List[float] = []
        for idx in range(self.length):
            if _probe_on:
                with torch.no_grad():
                    probe_step_rms.append(
                        hidden_states.detach().float().pow(2).mean().sqrt().item()
                    )
            # Step 7.0b: Save the branch step decided one step ago. The fork
            # is scored after normal logits exist, but it must use the exact
            # pre-normal rollout state.
            branch_request = None
            if pending_branch is not None:
                branch_request = (
                    pending_branch,
                    input_ids,
                    hidden_states,
                    self._fork_cache(cache_hidden),
                    attention_mask,
                    position_ids,
                    loss_mask,
                )
                pending_branch = None

            # Step 7.1: Get input embeddings with gradient tracking
            inputs_embeds = draft.embed_input_ids(input_ids)
            if not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True

            # Step 7.2: Encode through draft model layers
            if getattr(draft, "gradient_checkpointing", False) and draft.training:
                hidden_states, cache_hidden = torch.utils.checkpoint.checkpoint(
                    draft.encode_layers,
                    inputs_embeds,
                    hidden_states,
                    cache_hidden,
                    attention_mask,
                    position_ids,
                    True,
                    use_reentrant=False,
                )
            else:
                hidden_states, cache_hidden = draft.encode_layers(
                    inputs_embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )

            # Step 7.2b: Smooth-L1(draft h_i, target aux_i) on tokens 1..3.
            if (
                target_aux_tape is not None
                and idx < sl1_token_budget
                and getattr(draft, "_last_layer_outs", None)
            ):
                with torch.no_grad():
                    outs = draft._last_layer_outs
                    for li, (h_draft, h_tgt) in enumerate(
                        zip(outs, target_aux_tape)
                    ):
                        if h_draft.shape != h_tgt.shape:
                            continue
                        sl1_layer_sum[li] += torch.nn.functional.smooth_l1_loss(
                            h_draft.detach().float(),
                            h_tgt.float(),
                        ).item()
                        sl1_layer_count[li] += 1

            # Step 7.3: Compute logits from hidden states
            logits = draft.compute_logits(hidden_states)

            if branch_request is not None:
                (
                    pending,
                    branch_input_ids,
                    branch_hidden_states,
                    branch_cache_hidden,
                    branch_attention_mask,
                    branch_position_ids,
                    branch_loss_mask,
                ) = branch_request
                (
                    branch_term,
                    branch_n,
                    branch_metric1,
                    branch_metric2,
                    branch_denom,
                ) = self._branch_loss(
                    pending,
                    draft,
                    branch_input_ids,
                    branch_hidden_states,
                    branch_cache_hidden,
                    branch_attention_mask,
                    branch_position_ids,
                    branch_loss_mask,
                    logits,
                )
                branch_losses.append(branch_term)
                branch_denoms.append(branch_denom)
                branch_aux_metric1.append(branch_metric1)
                branch_aux_metric2.append(branch_metric2)
                branch_hits += branch_n

            # Step 7.4: Compute target distribution and position mask
            with torch.no_grad():
                target_max_token = target_logits.argmax(-1)
                target_mask = draft.t2d[target_max_token][..., None].int()
                position_mask = target_mask * loss_mask

                target_head = target_logits[..., draft.t2d].float()
                target_p = nn.Softmax(dim=2)(target_head).detach()

            # Step 7.5: Compute loss (final layer CE, weighted by last depth weight)
            out_logp = nn.LogSoftmax(dim=2)(logits)
            loss = -torch.sum(position_mask * target_p * out_logp, dim=2).mean()
            # Diagnostic only -- does not touch `loss` or the optimization.
            # `loss` above divides by ALL B*S positions (masked ones contribute
            # 0); branch_loss divides by its own ACTIVE count. So the two are on
            # different scales and ploss_i vs branch_loss is not a valid
            # comparison. Renormalize the same base CE over active positions to
            # get a directly comparable number.
            with torch.no_grad():
                _act = position_mask.sum()
                _tot = position_mask.shape[0] * position_mask.shape[1]
                plosses_active.append(
                    (
                        -torch.sum(
                            position_mask * target_p * out_logp, dim=2
                        ).sum()
                        / _act.clamp_min(1)
                    ).detach()
                )
                active_counts.append(float(_act.item()))
                total_counts.append(float(_tot))
            # Multi-depth CE: shared lm_head applied to each intermediate layer out.
            if self.multi_depth_ce_weights:
                layer_outs_ce = getattr(draft, "_last_layer_outs", None)
                if layer_outs_ce:
                    loss = loss * self.multi_depth_ce_weights[-1]
                    n_shallow = min(
                        len(layer_outs_ce) - 1, len(self.multi_depth_ce_weights) - 1
                    )
                    for depth_i in range(n_shallow):
                        w_i = self.multi_depth_ce_weights[depth_i]
                        if w_i == 0.0:
                            continue
                        logits_i = draft.compute_logits(layer_outs_ce[depth_i])
                        out_logp_i = nn.LogSoftmax(dim=2)(logits_i)
                        loss = loss + w_i * (
                            -torch.sum(
                                position_mask * target_p * out_logp_i, dim=2
                            ).mean()
                        )

            if log_prefix == "train" and self.skew_kl_loss_weight > 0.0:
                stage_weight = self._stage_weight(
                    self.skew_kl_stage_weights, idx, default=1.0
                )
                if stage_weight != 0.0:
                    alpha = min(max(self.skew_kl_alpha, 0.0), 1.0)
                    out_logp_f = out_logp.float()
                    out_p = out_logp_f.exp()
                    eps = torch.finfo(torch.float32).tiny
                    direction = self.skew_kl_direction.lower()
                    terms = []
                    if direction in ("forward", "bidirectional", "both"):
                        mix_q = alpha * target_p + (1.0 - alpha) * out_p
                        terms.append(
                            target_p * (target_p.clamp_min(eps).log() - mix_q.clamp_min(eps).log())
                        )
                    if direction in ("reverse", "bidirectional", "both"):
                        mix_p = alpha * out_p + (1.0 - alpha) * target_p
                        terms.append(out_p * (out_logp_f - mix_p.clamp_min(eps).log()))
                    if not terms:
                        raise ValueError(
                            "skew_kl_direction must be forward, reverse, or bidirectional"
                        )
                    skew_kl = sum(torch.sum(term, dim=-1, keepdim=True) for term in terms)
                    skew_kl_step = self._masked_mean(skew_kl, position_mask)
                    if self.skew_kl_loss_mode.lower() == "replace":
                        beta = max(0.0, min(1.0, self.skew_kl_loss_weight * stage_weight))
                        loss = (1.0 - beta) * loss + beta * skew_kl_step
                        skew_kl_losses.append(skew_kl_step)
                    elif self.skew_kl_loss_mode.lower() == "additive":
                        skew_kl_losses.append(stage_weight * skew_kl_step)
                    else:
                        raise ValueError(
                            "skew_kl_loss_mode must be additive or replace"
                        )

            # Step 7.6: Compute accuracy
            with torch.no_grad():
                correct = (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
                accuracy = correct.sum().item() / (loss_mask.sum().item() + 1e-6)

            # Step 7.6b: Decide this step's branches and re-ask the teacher.
            # Consumed at the top of the next step; the last step has no next.
            if use_branch_distill and idx < min(
                self.branch_distill_steps, self.length - 1
            ):
                pending_branch = self._branch_decide(
                    idx, draft, logits, target_logits, loss_mask
                )
                branch_stats = getattr(self, "_branch_last_stats", None)
                if branch_stats is not None:
                    _survivors = branch_stats.get("survivors", 0.0)
                    branch_candidates += branch_stats.get("candidates", 0.0)
                    branch_survivors += _survivors
                    branch_synthetic += branch_stats.get("synthetic", 0.0)
                    branch_ratio_weighted_sum += (
                        branch_stats.get("ratio_mean", 0.0) * _survivors
                    )
                    if _survivors > 0.0:
                        branch_ratio_medians.append(
                            branch_stats.get("ratio_median", 0.0)
                        )

            # Step 7.7: Store loss and accuracy
            plosses.append(loss)
            acces.append(accuracy)

            # Step 7.8: Update inputs for next iteration (skip on last step)
            if idx < self.length - 1:
                input_ids = padding(input_ids, left=False)
                target_logits = padding(target_logits, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Keep target aux tape aligned with the shifted sequence so
                # tokens 2/3 still compare h_i vs the matching target HS.
                if target_aux_tape is not None and idx + 1 < sl1_token_budget:
                    target_aux_tape = tuple(
                        padding(t, left=False) for t in target_aux_tape
                    )
                # EAGLE 3.1 feeds post-norm HS into the next draft step.
                if hasattr(draft, "next_hidden_from_encode"):
                    hidden_states = draft.next_hidden_from_encode(hidden_states)
                if hasattr(draft, "shift_aux_inject"):
                    draft.shift_aux_inject(left=False)

        # Step 8: Compute weighted loss
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        skew_kl_loss = (
            sum(skew_kl_losses) / len(skew_kl_losses)
            if skew_kl_losses
            else ploss.new_zeros(())
        )
        skew_kl_add_weight = (
            self.skew_kl_loss_weight
            if self.skew_kl_loss_mode.lower() == "additive"
            else 0.0
        )
        ploss = ploss + skew_kl_add_weight * skew_kl_loss
        # Branch distillation: mean over branched steps, added to total.
        branch_loss = (
            torch.stack(branch_losses).mean()
            if branch_losses
            else ploss.new_zeros(())
        )
        if branch_losses:
            ploss = ploss + branch_weight * branch_loss

        if (
            log_prefix == "train"
            and not self._logged_aux_losses
            and (self.skew_kl_loss_weight > 0.0 or use_branch_distill)
        ):
            logger.info(
                "Eagle3 aux losses: skew_kl(w=%s,a=%s,dir=%s,mode=%s) multi_ce=%s",
                self.skew_kl_loss_weight,
                self.skew_kl_alpha,
                self.skew_kl_direction,
                self.skew_kl_loss_mode,
                self.multi_depth_ce_weights,
            )
            self._logged_aux_losses = True
        if use_branch_distill and not self._logged_branch_distill:
            logger.info(
                "Eagle3 branch distillation: objective=%s, w=%s, draft top-1 vs "
                "teacher top-%d, prob_ratio_threshold=%s, first %d step(s), "
                "one extra teacher forward per branched step.",
                self.branch_distill_objective,
                self.branch_distill_loss_weight,
                self.branch_distill_target_top_k,
                self.branch_distill_prob_ratio_threshold,
                min(self.branch_distill_steps, max(self.length - 1, 0)),
            )
            self._logged_branch_distill = True

        log = {f"{log_prefix}/acc_{i}": acces[i] for i in range(len(acces))}
        log.update({f"{log_prefix}/ploss_{i}": plosses[i].item() for i in range(len(plosses))})
        if log_prefix == "train" and skew_kl_losses:
            log[f"{log_prefix}/skew_kl_loss"] = skew_kl_loss.item()
        if log_prefix == "train" and plosses_active:
            # Main CE per ACTIVE position -- the like-for-like counterpart to
            # branch_loss. Compare these two, never ploss_i vs branch_loss.
            for i in range(len(plosses_active)):
                log[f"{log_prefix}/ploss_active_{i}"] = plosses_active[i].item()
            log[f"{log_prefix}/active_density"] = (
                active_counts[0] / max(total_counts[0], 1.0)
            )
        if use_branch_distill:
            log[f"{log_prefix}/branch_loss"] = branch_loss.item()
            log[f"{log_prefix}/branch_weight"] = branch_weight
            if branch_denoms and total_counts:
                # Per-position gradient scale. Main CE spreads 1/(B*S) over every
                # position; the branch term spreads w/denom over its active ones.
                # So the nominal w=0.1 is NOT the effective relative weight --
                # this ratio is. >1 means branch gradients dominate per position.
                log[f"{log_prefix}/branch_grad_ratio"] = (
                    branch_weight
                    * total_counts[0]
                    / max(branch_denoms[0], 1.0)
                )
            if branch_aux_metric1:
                if self.branch_distill_objective == "change":
                    log[f"{log_prefix}/branch_target_delta_rms"] = (
                        torch.stack(branch_aux_metric1).mean().item()
                    )
                    log[f"{log_prefix}/branch_draft_delta_rms"] = (
                        torch.stack(branch_aux_metric2).mean().item()
                    )
                else:
                    _ent = torch.stack(branch_aux_metric1).mean().item()
                    log[f"{log_prefix}/branch_target_entropy"] = _ent
                    log[f"{log_prefix}/branch_kl"] = branch_loss.item() - _ent
                    log[f"{log_prefix}/branch_invocab_mass"] = (
                        torch.stack(branch_aux_metric2).mean().item()
                    )
            if branch_survivors > 0.0 and self.branch_distill_synthetic:
                log[f"{log_prefix}/branch_synthetic_share"] = (
                    branch_synthetic / branch_survivors
                )
            if branch_candidates > 0.0:
                log[f"{log_prefix}/branch_ratio_survival"] = (
                    branch_survivors / branch_candidates
                )
            if branch_survivors > 0.0:
                log[f"{log_prefix}/branch_ratio_mean"] = (
                    branch_ratio_weighted_sum / branch_survivors
                )
            if branch_ratio_medians:
                log[f"{log_prefix}/branch_ratio_median"] = float(
                    torch.tensor(branch_ratio_medians).median().item()
                )
            # Branch positions per step, as a share of supervised tokens.
            log[f"{log_prefix}/branch_rate"] = branch_hits / (
                max(len(branch_losses), 1) * max(loss_mask.sum().item(), 1.0)
            )
        # Per-layer Smooth-L1(draft h_i, target aux_i), averaged over tokens 1..3.
        for li, (s, c) in enumerate(zip(sl1_layer_sum, sl1_layer_count)):
            if c > 0:
                log[f"{log_prefix}/aux_vs_draft_sl1_h{li}"] = s / float(c)
        if _probe_on:
            for i, v in enumerate(probe_step_rms):
                log[f"{log_prefix}/rms_step{i}"] = v
            for i, v in enumerate(getattr(self, "_probe_aux_rms", []) or []):
                log[f"{log_prefix}/rms_aux{i}"] = v
        # Route into the appropriate accumulator.
        if log_prefix == "eval":
            for k, v in log.items():
                self._eval_pending_log[k] = self._eval_pending_log.get(k, 0.0) + v
            self._eval_pending_log_count += 1
        else:
            for k, v in log.items():
                self._train_pending_log[k] = self._train_pending_log.get(k, 0.0) + v
            self._train_pending_log_count += 1
        # Step 9: Return loss
        return ploss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to handle DeepSpeed ZeRO-3 model saving.

        Args:
            output_dir: Directory to save the model. If None, uses self.args.output_dir
            _internal_call: Internal flag used by Trainer
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Check if using DeepSpeed ZeRO-3
        is_deepspeed_zero3 = (
            self.is_deepspeed_enabled
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        if is_deepspeed_zero3:
            # Handle ZeRO-3 model saving
            self._save_zero3_model(output_dir, _internal_call)
        else:
            # Fall back to parent class save_model
            super().save_model(output_dir, _internal_call)
            self._save_banded_aux_mix_weights(output_dir)

    def _save_banded_aux_mix_weights(self, output_dir: str) -> None:
        """Log and persist learned band weights alongside each checkpoint."""
        if not self.args.should_save or not self.accelerator.is_main_process:
            return
        getter = getattr(self.draft_model, "get_banded_aux_mix_weights", None)
        if getter is None:
            return
        weights = getter()
        if not weights:
            return
        logger.info("Progressive banded aux mix weights: %s", weights)
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, "banded_aux_mix_weights.json"), "w"
        ) as output_file:
            json.dump(weights, output_file, indent=2, sort_keys=True)
            output_file.write("\n")

    def _save_zero3_model(self, output_dir: str, _internal_call: bool = False):
        """
        Save model with DeepSpeed ZeRO-3 specific logic.

        Args:
            output_dir: Directory to save the model
            _internal_call: Internal flag used by Trainer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save with DeepSpeed's state_dict gathering
        # All processes must participate in parameter gathering to avoid deadlock
        with deepspeed.zero.GatheredParameters(self.model.parameters()):
            state_dict = self.model.state_dict()
            # The scalar logits may be ZeRO-partitioned, so inspect them while
            # all parameters are gathered.
            self._save_banded_aux_mix_weights(output_dir)

        # Only main process saves the model
        if self.args.should_save and self.accelerator.is_main_process:
            self.model.save_pretrained(
                output_dir,
                is_main_process=True,
                state_dict=state_dict,
                save_function=torch.save,
            )

            # Save training arguments
            from transformers.trainer import TRAINING_ARGS_NAME

            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Wait for all processes
        self.accelerator.wait_for_everyone()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        """
        data_for_draft_model = self.prepare_data_for_draft_model(inputs)

        attention_mask = data_for_draft_model["attention_mask"]
        # inputs_embeds = data_for_draft_model["inputs_embeds"]
        position_ids = data_for_draft_model.get("position_ids", None)
        input_ids = data_for_draft_model["input_ids"]
        target_logits = data_for_draft_model["target_logits"]
        loss_mask = data_for_draft_model["loss_mask"]
        hidden_states = data_for_draft_model["hidden_states"]

        with torch.no_grad():
            hidden_states = self.down_project_hidden_states(hidden_states)
            attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
                hidden_states, attention_mask, position_ids
            )
            loss = self.draft_model_training_time_test(
                input_ids,
                hidden_states,
                attention_mask,
                position_ids,
                target_logits,
                loss_mask,
                log_prefix="eval",
            )
        return (loss, None, None)
