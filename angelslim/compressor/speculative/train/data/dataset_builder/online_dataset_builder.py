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
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Features, Value, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from transformers.image_utils import load_image

from angelslim.utils import rank0_print

from ..chat_templates import ChatTemplateType
from ..data_utils import (
    VLMSmolVLMDataCollatorWithPadding,
    stable_hf_map_cache_files,
)
from .base_dataset_builder import OnlineDatasetBuilder
from .dataset_builder_factory import DatasetBuilderFactory








@DatasetBuilderFactory.register("online", "VLM", "smolvlm")
@DatasetBuilderFactory.register("online", "VLM", "idefics3")
class OnlineSmolVLMDatasetBuilder(OnlineDatasetBuilder):
    """Online Eagle3 dataset builder for SmolVLM / Idefics3.

    Image tokens must be expanded by the processor together with images
    (unlike Qwen's pad-token expansion). Pixel values are recomputed in the
    collator from ``image_paths``.
    """

    def __init__(
        self,
        tokenizer: Union[AutoTokenizer, AutoProcessor],
        max_length: int = 2048,
        shuffle_seed: int = 42,
        chat_template_type: ChatTemplateType = ChatTemplateType.SMOLVLM,
        display: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            tokenizer,
            max_length,
            shuffle_seed,
            chat_template_type,
            display,
        )

    def build_dataset(
        self,
        datapath: str,
        num_proc: int = 8,
        shuffle: bool = True,
        sample_num: Optional[int] = None,
        min_loss_tokens: Optional[int] = None,
        load_from_cache_file: bool = False,
    ) -> Dataset:
        try:
            # Mixed text / VL target-gen jsonl: text parts are {type,text},
            # image parts are {type,image_url:{url}} (and/or {type,image}).
            # Leaving Features unset lets HF lock on the first chunk (text-only)
            # and then fail when image_url appears.
            features = Features(
                {
                    "id": Value("string"),
                    "conversations": [
                        {
                            "role": Value("string"),
                            "content": [
                                {
                                    "type": Value("string"),
                                    "text": Value("string"),
                                    "image": Value("string"),
                                    "image_url": {"url": Value("string")},
                                }
                            ],
                        }
                    ],
                }
            )
            ds = load_dataset("json", data_files=datapath, features=features)

            if shuffle:
                ds = ds["train"].shuffle(seed=self.shuffle_seed)
            else:
                ds = ds["train"]

            if sample_num is not None and 0 < sample_num < len(ds):
                ds = ds.select(range(sample_num))

            original_columns = ds.column_names
            # Pin cache under <data_dir>/.map_cache/ so restarts actually hit
            # (default HF fingerprints bound methods and miss every run).
            # Collator still rebuilds pixel_* each step from image_paths.
            cache_version = "v1"
            cache_files = stable_hf_map_cache_files(
                datapath,
                builder_tag=type(self).__name__,
                max_length=self.max_length,
                shuffle=shuffle,
                shuffle_seed=self.shuffle_seed,
                sample_num=sample_num,
                min_loss_tokens=min_loss_tokens,
                cache_version=cache_version,
            )
            if load_from_cache_file:
                rank0_print(
                    f"HF map cache (load_from_cache_file=true): {cache_files['root']}"
                )
            processed_ds = ds.map(
                self._preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_files["map"],
                desc="Processing SmolVLM conversations",
            )
            processed_ds = processed_ds.filter(
                lambda batch: [ids is not None for ids in batch["input_ids"]],
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_files["filter_empty"],
                desc="Filtering empty input_ids",
            )
            if min_loss_tokens is not None:
                processed_ds = processed_ds.filter(
                    lambda batch: [
                        sum(sum(x) if isinstance(x, list) else x for x in m) >= min_loss_tokens
                        for m in batch["loss_mask"]
                    ],
                    batched=True,
                    num_proc=num_proc,
                    load_from_cache_file=load_from_cache_file,
                    cache_file_name=cache_files["filter_loss"],
                    desc=f"Filtering sequences with loss tokens < {min_loss_tokens}",
                )

            torch_columns = [c for c in processed_ds.column_names if c != "image_paths"]
            processed_ds.set_format(type="torch", columns=torch_columns, output_all_columns=True)
            return processed_ds
        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def get_data_collator(self) -> Any:
        return VLMSmolVLMDataCollatorWithPadding(processor=self.tokenizer)

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "loss_mask": [],
            "image_paths": [],
        }
        ids = examples.get("id", list(range(len(examples["conversations"]))))
        for i in range(len(ids)):
            try:
                processed_example = self._process_single_conversation(
                    examples["conversations"][i]
                )
                if processed_example is not None:
                    for key in new_examples.keys():
                        if key not in processed_example:
                            new_examples[key].append(None)
                        else:
                            new_examples[key].append(processed_example[key])
                else:
                    for key in new_examples:
                        new_examples[key].append(None)
            except Exception as e:
                rank0_print(f"Error processing example: {e}")
                for key in new_examples:
                    new_examples[key].append(None)

        cleaned_new_examples = {}
        for key, value in new_examples.items():
            if any(v is not None for v in value):
                cleaned_new_examples[key] = value
        return cleaned_new_examples

    def _build_messages(self, source: List[Dict]) -> List[Dict]:
        # SmolVLM chat template does not use a system turn by default.
        if source and source[0].get("role") == "system":
            source = source[1:]

        expected_roles = ["user", "assistant"]
        if not source or source[0].get("role") != "user":
            source = source[1:] if source else source

        valid_turns = []
        for turn in source:
            if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                continue
            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                content = content.strip()
            if role and content:
                valid_turns.append({"role": role, "content": content})

        messages = []
        for i, turn in enumerate(valid_turns):
            expected_role = expected_roles[i % 2]
            if turn["role"] != expected_role:
                break
            messages.append(turn)
        return messages

    @staticmethod
    def _normalize_conversation(conversation_data: List[Dict]) -> List[Dict]:
        """Normalize openai_vl image_url items to ``{"type": "image", "image": url}``."""
        normalized = []
        for turn in conversation_data:
            content = turn.get("content")
            if not isinstance(content, list):
                normalized.append(turn)
                continue
            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    else:
                        url = image_url
                    if url:
                        new_content.append({"type": "image", "image": url})
                elif item_type == "image":
                    image = item.get("image")
                    if image:
                        new_content.append({"type": "image", "image": image})
                elif item_type == "text":
                    text = item.get("text")
                    if text is not None:
                        new_content.append({"type": "text", "text": text})
                else:
                    cleaned = {k: v for k, v in item.items() if v is not None}
                    if cleaned:
                        new_content.append(cleaned)
            normalized.append({"role": turn["role"], "content": new_content})
        return normalized

    def _extract_images(self, messages: List[Dict]) -> List:
        images = []
        image_paths = []
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "image" and item.get("image"):
                    src = item["image"]
                    image_paths.append(src)
                    if isinstance(src, Image.Image):
                        images.append(src)
                    else:
                        images.append(load_image(src))
        return images, image_paths

    @staticmethod
    def _sanitize_messages_for_processor(messages: List[Dict]) -> Optional[List[Dict]]:
        """
        Strip leftover ``<image>`` from text and keep only image/text items.

        Returns None if the conversation is malformed for the VLM processor
        (e.g. image placeholders in text with no image content items).
        """
        sanitized: List[Dict] = []
        n_image_items = 0
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                if "<image>" in content and n_image_items == 0:
                    # defer check until end using total counts
                    pass
                text = content.replace("<image>", "").strip()
                if not text and message.get("role") == "assistant":
                    return None
                sanitized.append({"role": message["role"], "content": text})
                continue
            if not isinstance(content, list):
                return None
            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    n_image_items += 1
                    new_content.append({"type": "image"})
                elif item.get("type") == "text":
                    text = str(item.get("text") or "").replace("<image>", "").strip()
                    new_content.append({"type": "text", "text": text})
            if not new_content:
                return None
            sanitized.append({"role": message["role"], "content": new_content})

        # Count leftover placeholders in any remaining string fields / text items
        leftover = 0
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                leftover += content.count("<image>")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        leftover += str(item.get("text") or "").count("<image>")

        # After sanitize, text no longer has placeholders; chat template will insert
        # one ``<image>`` per image content item. Reject if original text had
        # placeholders but we have no images to pass.
        if leftover > 0 and n_image_items == 0:
            return None
        if leftover > n_image_items > 0:
            # e.g. 2x <image> in text but only 1 image attached
            return None
        return sanitized

    def _process_single_conversation(self, conversation_data: List[Dict]) -> Optional[Dict]:
        if not conversation_data or not isinstance(conversation_data, list):
            return None

        try:
            conversation_data = self._normalize_conversation(conversation_data)
            messages = self._build_messages(conversation_data)
            if not messages:
                return None

            images, image_paths = self._extract_images(messages)
            sanitized = self._sanitize_messages_for_processor(messages)
            if sanitized is None:
                return None
            messages = sanitized

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            n_tokens_in_text = text.count("<image>")
            if n_tokens_in_text != len(images):
                # Processor requires a 1:1 match between placeholders and images.
                return None

            encode_kwargs = {
                "text": text,
                "return_tensors": "pt",
                "return_offsets_mapping": True,
                "max_length": self.max_length,
                "truncation": True,
                "padding": False,
            }
            if images:
                encoding = self.tokenizer(images=images, **encode_kwargs)
            else:
                encoding = self.tokenizer(**encode_kwargs)

            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]
            if offsets.ndim == 3:
                offsets = offsets[0]

            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            conversation = tok.decode(input_ids[0], skip_special_tokens=False)

            try:
                loss_mask = self._create_loss_mask_from_offsets(conversation, offsets)
            except Exception as e:
                rank0_print(f"Error creating loss mask: {e}")
                raise e

            attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids))

            if self.display and self.display_count == 0:
                try:
                    self._visualize_loss_mask(input_ids, loss_mask, conversation)
                except Exception as e:
                    rank0_print(f"Error visualizing loss mask: {e}")
                    raise e
                self.display_count += 1

            result = {
                "input_ids": input_ids.view(1, -1),
                "attention_mask": attention_mask.view(1, -1),
                "loss_mask": loss_mask.view(1, -1),
                "image_paths": json.dumps(image_paths),
            }
            return result
        except Exception as e:
            rank0_print(f"Error processing conversation: {e}")
            return None




