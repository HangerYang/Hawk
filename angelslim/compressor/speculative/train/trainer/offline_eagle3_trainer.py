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

import os
from typing import Dict

import torch
from torch import nn

from ...utils import padding
from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory


# The environment variable that turns branch distillation on for an offline run.
# The dataset builder reads it too, to decide whether the .img sidecars a VLM
# branch forward needs have to be loaded.
BRANCH_TARGET_ENV = "BRANCH_DISTILL_TARGET"


class _OfflineBranchMixin:
    """The one teacher forward an offline run cannot read off disk.

    Everything else about the target is cached: `target_hiddens` in the .ckpt is
    the REAL sequence's forward, and `target_head` turns it back into logits.
    Branch distillation asks what the teacher predicts after ONE substituted
    token -- a sequence no generation pass ever produced -- and no stored hidden
    state answers that, because substituting at position a changes every hidden
    state from a onwards. So the two forwards are split: the real one stays
    cached, and only the branch one runs a live target. That is ONE target
    forward per step where the online trainer pays two.

    The target is built lazily, on the first batch that actually branches, so an
    offline run with `branch_distill_loss_weight` at 0 still never loads one and
    keeps offline's memory profile.
    """

    # Which backend the branch forward needs; the VLM trainer overrides it.
    _branch_modal_type = "LLM"

    def _branch_target(self):
        if getattr(self, "_branch_target_model", None) is None:
            path = os.environ.get(BRANCH_TARGET_ENV)
            if not path:
                raise ValueError(
                    "branch distillation on an offline trainer needs a live target to "
                    "re-score the substituted sequence; point "
                    f"{BRANCH_TARGET_ENV} at the SAME snapshot the .ckpt files were "
                    "generated from, or set branch_distill_loss_weight to 0"
                )
            from angelslim.utils import rank0_print

            from ..models.target import create_target_model

            rank0_print(f"[branch-distill] branch forward target: {path}")
            self._branch_target_model = create_target_model(
                backend="hf",
                model_path=path,
                torch_dtype=torch.bfloat16,
                target_model_type=getattr(
                    self.draft_model.config, "target_model_type", None
                ),
                modal_type=self._branch_modal_type,
            )
        return self._branch_target_model

    def _stash_branch_ctx(self, inputs, input_ids, attention_mask):
        """Keep the pre-shift sequence the branch forward will substitute into.

        `_branch_decide` indexes this by ABSOLUTE position, so it has to be the
        sequence as the target saw it -- before the left shift that aligns
        everything else to the draft's index frame.
        """
        if self.branch_distill_loss_weight <= 0.0:
            return
        self._branch_ctx = {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask,
            # Per sample, not batched: SmolVLM lays an image down in 9/13/17
            # tiles depending on aspect ratio, and padding the tile dimension
            # would hand the vision tower embeddings that no <image> token
            # claims.
            "images": inputs.get("branch_image"),
        }

    def branch_teacher_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher logits for a sequence carrying branch substitutions.

        One sample at a time, over its unpadded length, which is how the .ckpt
        files were generated -- batching with padding moves hidden states even
        under flash attention, and the branch loss scores a DIFFERENCE against
        those stored logits, so a shift common to both paths would not cancel.
        """
        ctx = getattr(self, "_branch_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "branch distillation ran before prepare_data_for_draft_model "
                "stashed the unshifted sequence"
            )
        target = self._branch_target()
        attention_mask = ctx["attention_mask"].to(input_ids.device)
        images = ctx["images"] or [None] * input_ids.shape[0]
        aux_ids = getattr(self.draft_model.config, "aux_hidden_states_layer_ids", None)

        rows = []
        with torch.no_grad():
            for b in range(input_ids.shape[0]):
                keep = attention_mask[b].bool()
                image_kwargs = {
                    k: v.to(input_ids.device) for k, v in (images[b] or {}).items()
                }
                out = target.get_hidden_states_and_logits(
                    input_ids=input_ids[b, keep][None],
                    attention_mask=attention_mask[b, keep][None],
                    aux_hidden_states_layer_ids=aux_ids,
                    **image_kwargs,
                )
                rows.append(out[1][0].detach())

        # Back to [B, S, V] on the padded frame the caller indexes. The collator
        # right-pads, so every kept position is a prefix and the tail stays zero.
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            rows[0].shape[-1],
            dtype=rows[0].dtype,
            device=input_ids.device,
        )
        for b, row in enumerate(rows):
            logits[b, : row.shape[0]] = row.to(input_ids.device)
        return logits


@Eagle3TrainerFactory.register("offline", "LLM")
class OfflineEagle3Trainer(_OfflineBranchMixin, Eagle3Trainer):
    """
    Offline EAGLE3 Trainer for speculative decoding training.

    Uses pre-computed hidden states and logits from offline processing,
    avoiding the need for online target model inference.
    """

    def __init__(self, draft_model: nn.Module, target_head: nn.Module, length: int, **kwargs):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(draft_model=draft_model, length=length, **kwargs)
        self.target_head = target_head
        # Built on the first batch that branches; see _OfflineBranchMixin.
        self._branch_target_model = None

    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training from offline-generated inputs.

        Args:
            inputs: Dictionary containing:
                - input_ids: Token IDs
                - target_hiddens: Pre-computed last hidden states from target model
                - hidden_states: Pre-computed aux hidden states from target model
                - attention_mask: Attention mask
                - loss_mask: Mask for loss computation
                - position_ids (optional): Position IDs

        Returns:
            Dictionary with prepared data for draft model training
        """
        #
        inputs_fields = [
            "input_ids",
            "target_hiddens",
            "hidden_states",
            "attention_mask",
            "loss_mask",
        ]
        output_fields = [
            "input_ids",
            "target_logits",
            "hidden_states",
            "attention_mask",
            "loss_mask",
            "position_ids",
        ]

        #
        target_logits = self.target_head(inputs["target_hiddens"])
        position_ids = inputs.get("position_ids", None)
        loss_mask = inputs["loss_mask"]
        input_ids = inputs["input_ids"]

        self._stash_branch_ctx(inputs, input_ids, inputs["attention_mask"])

        # Apply right padding and move tensors to correct device
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        outputs = {k: inputs[k] for k in inputs_fields if k in output_fields}
        outputs["target_logits"] = target_logits
        outputs["position_ids"] = position_ids
        outputs["loss_mask"] = loss_mask
        outputs["input_ids"] = input_ids

        return outputs


@Eagle3TrainerFactory.register("offline", "VLM")
class OfflineVLMEagle3Trainer(_OfflineBranchMixin, Eagle3Trainer):
    """
    Offline EAGLE3 Trainer for speculative decoding training.

    Uses pre-computed hidden states and logits from offline processing,
    avoiding the need for online target model inference.
    """

    _branch_modal_type = "VLM"

    def __init__(self, draft_model: nn.Module, target_head: nn.Module, length: int, **kwargs):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(draft_model=draft_model, length=length, **kwargs)
        self.target_head = target_head
        # Built on the first batch that branches; see _OfflineBranchMixin.
        self._branch_target_model = None

    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training from offline-generated inputs.

        Args:
            inputs: Dictionary containing:
                - input_ids: Token IDs
                - target_hiddens: Pre-computed last hidden states from target model
                - hidden_states: Pre-computed aux hidden states from target model
                - attention_mask: Attention mask
                - loss_mask: Mask for loss computation
                - position_ids (optional): Position IDs (3D for VLMs mrope)

        Returns:
            Dictionary with prepared data for draft model training
        """
        inputs_fields = [
            "input_ids",
            "target_hiddens",
            "hidden_states",
            "attention_mask",
            "loss_mask",
            "position_ids",
        ]
        output_fields = [
            "input_ids",
            "target_logits",
            "hidden_states",
            "attention_mask",
            "loss_mask",
            "position_ids",
        ]

        target_logits = self.target_head(
            inputs["target_hiddens"].to(self.target_head.lm_head.weight.dtype)
        )
        loss_mask = inputs["loss_mask"]
        input_ids = inputs["input_ids"]
        position_ids = inputs.get("position_ids", None)

        self._stash_branch_ctx(inputs, input_ids, inputs["attention_mask"])

        # Apply right padding and move tensors to correct device
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        outputs = {k: inputs[k] for k in inputs_fields if k in output_fields}
        outputs["target_logits"] = target_logits
        outputs["loss_mask"] = loss_mask
        outputs["input_ids"] = input_ids
        outputs["position_ids"] = position_ids

        return outputs

