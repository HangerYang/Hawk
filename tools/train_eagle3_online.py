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

import argparse
import inspect
import os
from pathlib import Path

import torch
import transformers

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    Eagle3TrainerFactory,
    create_draft_model,
    create_target_model,
    infer_model_params,
)
from angelslim.utils import rank0_print


def filter_training_args(kwargs):
    supported = inspect.signature(transformers.TrainingArguments.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def parse_args():
    parser = argparse.ArgumentParser(description="Train EAGLE3 online model")
    parser.add_argument("--modal_type", type=str, default="VLM", choices=["LLM", "VLM"])
    parser.add_argument(
        "--training_mode", type=str, default="online", choices=["online", "offline"]
    )
    parser.add_argument("--target_model_name_or_path", type=str, required=True)
    parser.add_argument("--draft_model_config_path", type=str, required=True)
    parser.add_argument("--target_backend", type=str, default="hf", choices=["hf"])
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--embed_weight_key", type=str, default=None)
    parser.add_argument("--chat_template_type", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument(
        "--load_from_cache_file",
        type=lambda v: str(v).lower() in ("1", "true", "t", "yes", "y"),
        default=True,
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--training_time_test_length", type=int, default=7)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=float, default=5000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="no")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    return parser.parse_args()


def train():
    args = parse_args()
    if args.eval_data_path == "":
        args.eval_data_path = None

    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
    target_model_type = getattr(draft_model_config, "target_model_type", None)
    _, inferred_embed_weight_key, inferred_chat_template_type = infer_model_params(
        model_name_or_path=args.target_model_name_or_path,
        model_type=target_model_type,
    )
    if args.embed_weight_key is None:
        args.embed_weight_key = inferred_embed_weight_key
    if args.chat_template_type is None:
        args.chat_template_type = inferred_chat_template_type or "smolvlm"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    target_model = create_target_model(
        backend=args.target_backend,
        model_path=args.target_model_name_or_path,
        torch_dtype=dtype_map[args.torch_dtype],
        trust_remote_code=args.trust_remote_code,
        target_model_type=target_model_type,
        modal_type=args.modal_type,
    )
    draft_model = create_draft_model(draft_model_config)
    draft_model.load_embed_weights(args.target_model_name_or_path, args.embed_weight_key)
    draft_model.freeze_embed_weights()
    init_mix = getattr(draft_model_config, "eagle_aux_band_init_weights", None)
    if init_mix:
        freeze_mix = bool(getattr(draft_model_config, "eagle_aux_band_mix_frozen", False))
        draft_model.set_banded_aux_mix_weights(init_mix, freeze=freeze_mix)
    aux_ids = getattr(draft_model_config, "aux_hidden_states_layer_ids", None)
    eagle_aux_ids = getattr(draft_model_config, "eagle_aux_hidden_state_layer_ids", None)
    if aux_ids is not None:
        draft_model.config.aux_hidden_states_layer_ids = list(aux_ids)
        if eagle_aux_ids is None:
            eagle_aux_ids = [int(i) + 1 for i in aux_ids]
    if eagle_aux_ids is not None:
        draft_model.config.eagle_aux_hidden_state_layer_ids = list(eagle_aux_ids)

    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        display=args.display,
        target_model_type=target_model_type,
    )
    train_dataset, eval_dataset, data_collator = dataset_manager.create_online_datasets()
    if train_dataset is None:
        raise ValueError(f"no train data at {args.train_data_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    draft_model.build_vocab_mapping(
        dataset=train_dataset,
        cache_path=os.path.join(args.output_dir, "vocab_mapping_cache.pt"),
    )

    training_args = transformers.TrainingArguments(
        **filter_training_args(
            {
                "output_dir": args.output_dir,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "remove_unused_columns": False,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "optim": args.optim,
                "lr_scheduler_type": args.lr_scheduler_type,
                "fp16": args.fp16,
                "bf16": args.bf16,
                "eval_strategy": args.eval_strategy,
                "save_strategy": args.save_strategy,
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "logging_steps": args.logging_steps,
                "eval_steps": args.eval_steps,
                "report_to": args.report_to,
                "run_name": args.run_name,
                "deepspeed": args.deepspeed,
            }
        )
    )
    trainer = Eagle3TrainerFactory.create(
        training_mode=args.training_mode,
        modal_type=args.modal_type,
        draft_model=draft_model,
        target_model=target_model,
        length=args.training_time_test_length,
        draft_model_config=draft_model_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    resume = any(Path(args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    rank0_print("Training completed!")


if __name__ == "__main__":
    train()
