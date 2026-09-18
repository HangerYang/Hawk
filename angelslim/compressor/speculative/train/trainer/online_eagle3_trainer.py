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
from typing import Any, Dict

import torch
from torch import nn

from ...utils import padding
from ..data.data_utils import paddingtensor, paddingtensor2D, reduce_vlm_image_rows
from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory


def _maybe_pool_draft_rows(
    input_ids,
    hidden_states,
    target_logits,
    loss_mask,
    attention_mask,
    position_ids,
):
    """Shrink draft image rows after the target has seen the full image."""
    how = os.environ.get("VISTOKEN_REDUCE", "none").lower()
    if how in ("", "none"):
        return (
            input_ids,
            hidden_states,
            target_logits,
            loss_mask,
            attention_mask,
            position_ids,
        )
    tok = os.environ.get("VISTOKEN_IMAGE_TOKEN_ID", "")
    if not tok:
        raise ValueError("VISTOKEN_REDUCE is set but VISTOKEN_IMAGE_TOKEN_ID is not")
    image_token_id = int(tok)
    factor = int(os.environ.get("VISTOKEN_FACTOR", "4"))
    keeppos = os.environ.get("VISTOKEN_KEEPPOS", "1") != "0"
    rows = []
    for b in range(input_ids.shape[0]):
        item = {
            "input_ids": input_ids[b : b + 1],
            "hidden_states": hidden_states[b : b + 1],
            "target_logits": target_logits[b : b + 1],
            "loss_mask": loss_mask[b : b + 1],
            "attention_mask": attention_mask[b : b + 1],
        }
        if position_ids is not None and position_ids.dim() == 2:
            item["position_ids"] = position_ids[b : b + 1]
        rows.append(
            reduce_vlm_image_rows(
                item, image_token_id, factor, how=how, keep_positions=keeppos
            )
        )
    max_len = max(row["input_ids"].shape[1] for row in rows)
    input_ids = torch.cat([paddingtensor2D(row["input_ids"], max_len) for row in rows])
    hidden_states = torch.cat(
        [paddingtensor(row["hidden_states"], max_len) for row in rows]
    )
    target_logits = torch.cat(
        [paddingtensor(row["target_logits"], max_len) for row in rows]
    )
    loss_mask = torch.cat([paddingtensor2D(row["loss_mask"], max_len) for row in rows])
    attention_mask = torch.cat(
        [paddingtensor2D(row["attention_mask"], max_len) for row in rows]
    )
    if any(row.get("position_ids") is not None for row in rows):
        position_ids = torch.cat(
            [
                paddingtensor2D(
                    row["position_ids"]
                    if row.get("position_ids") is not None
                    else torch.arange(
                        row["input_ids"].shape[1], device=input_ids.device
                    )[None],
                    max_len,
                )
                for row in rows
            ]
        )
    return (
        input_ids,
        hidden_states,
        target_logits,
        loss_mask,
        attention_mask,
        position_ids,
    )


@Eagle3TrainerFactory.register("online", "VLM")
class OnlineVLMEagle3Trainer(Eagle3Trainer):
    """Online EAGLE-3: the live target runs every step, then the drafter trains."""

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        length: int,
        draft_model_config: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            draft_model=draft_model,
            length=length,
            draft_model_config=draft_model_config,
            **kwargs,
        )
        self.target_model = target_model
        self._aux_hidden_states_layer_ids = getattr(
            draft_model_config, "aux_hidden_states_layer_ids", None
        )
        how = os.environ.get("VISTOKEN_REDUCE", "none").lower()
        if how not in ("", "none"):
            from angelslim.utils import rank0_print

            rank0_print(
                f"[vistoken] draft image rows /{os.environ.get('VISTOKEN_FACTOR', '4')}"
                f" ({how}) after target forward"
            )

    def prepare_data_for_draft_model(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]
        kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "attention_mask", "loss_mask", "gist_embeddings"]
        }
        hidden_states, target_logits, _, position_ids = (
            self.target_model.get_hidden_states_and_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
                **kwargs,
            )
        )
        (
            input_ids,
            hidden_states,
            target_logits,
            loss_mask,
            attention_mask,
            position_ids,
        ) = _maybe_pool_draft_rows(
            input_ids,
            hidden_states,
            target_logits,
            loss_mask,
            attention_mask,
            position_ids,
        )
        if self.branch_distill_loss_weight > 0.0:
            self._branch_ctx = {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask,
                "kwargs": kwargs,
            }
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)
        return {
            "hidden_states": hidden_states,
            "target_logits": target_logits,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def branch_teacher_logits(self, input_ids):
        ctx = getattr(self, "_branch_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "branch distillation ran before prepare_data_for_draft_model "
                "stashed the unshifted sequence"
            )
        with torch.no_grad():
            _, target_logits, _, _ = self.target_model.get_hidden_states_and_logits(
                input_ids=input_ids,
                attention_mask=ctx["attention_mask"],
                aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
                **ctx["kwargs"],
            )
        return target_logits.detach().to(input_ids.device)
