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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from angelslim.utils import rank0_print

from ..data_utils import VLMDataCollatorWithPadding
from .base_dataset_builder import DatasetBuilder
from .dataset_builder_factory import DatasetBuilderFactory


def _aux_layer_columns(data_dir: Path):
    """Blocks of the stored aux concat that this draft config wants.

    Generation may store MORE target layers than a draft consumes -- keeping a
    layer an ablation needs costs disk but saves regenerating the whole set --
    so the shards are not assumed to match the config. `aux_layers.json` beside
    them records the stored order; the draft config names what it wants.

    Returns None when the two agree, which is the common case and stays a
    no-op. Raises when a wanted layer was never stored, because the alternative
    is a shape error deep inside the draft's fusion FC.
    """
    import json

    meta = Path(data_dir) / "aux_layers.json"
    cfg_path = os.environ.get("DRAFT_MODEL_CONFIG_PATH")
    if not meta.exists() or not cfg_path or not os.path.exists(cfg_path):
        return None
    with open(meta) as f:
        stored = list(json.load(f)["aux_hidden_states_layer_ids"])
    with open(cfg_path) as f:
        wanted = json.load(f).get("aux_hidden_states_layer_ids")
    if not wanted or list(wanted) == stored:
        return None
    missing = [layer for layer in wanted if layer not in stored]
    if missing:
        raise ValueError(
            f"{meta} holds target layers {stored}, but {cfg_path} asks for "
            f"{list(wanted)}; {missing} was never generated"
        )
    rank0_print(f"[aux-layers] stored {stored} -> using {list(wanted)}")
    return [stored.index(layer) for layer in wanted]


class OfflineEagle3Dataset(Dataset):
    """
    Offline Dataset for EAGLE3 training.

    Loads pre-computed hidden states, logits, and other data from .ckpt files.
    Each .ckpt file contains a dictionary with keys: input_ids, target_logits,
    hidden_states, and loss_mask.
    """

    def __init__(self, data_dir: str, file_pattern: str = "*.ckpt", cache_in_memory: bool = False):
        """
        Initialize the OfflineEagle3Dataset.

        Args:
            data_dir: Directory containing .ckpt files
                (will search recursively in subdirectories)
            file_pattern: Pattern to match checkpoint files (default: "*.ckpt")
            cache_in_memory: Whether to cache all data in memory (default: False)
        """
        self.data_dir = Path(data_dir)
        self.cache_in_memory = cache_in_memory

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Recursively find all checkpoint files in subdirectories
        self.ckpt_files = sorted(list(self.data_dir.rglob(file_pattern)))

        if len(self.ckpt_files) == 0:
            raise ValueError(
                f"No checkpoint files found in {data_dir} "
                f"(including subdirectories) with pattern {file_pattern}"
            )

        rank0_print(
            f"Found {len(self.ckpt_files)} checkpoint files "
            f"in {data_dir} (including subdirectories)"
        )

        # Track valid indices (files that can be loaded successfully)
        self.valid_indices = list(range(len(self.ckpt_files)))

        # Cache data in memory if requested
        self.cached_data: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.cache_in_memory:
            rank0_print("Caching all data in memory...")
            self.cached_data = []
            failed_count = 0
            for i in range(len(self.ckpt_files)):
                data = self._load_ckpt(i)
                if data is not None:
                    self.cached_data.append(data)
                else:
                    failed_count += 1

            # Update valid indices based on successful loads
            self.valid_indices = list(range(len(self.cached_data)))

            if failed_count > 0:
                rank0_print(
                    f"Data caching completed. "
                    f"Successfully loaded {len(self.cached_data)} files, "
                    f"failed to load {failed_count} files"
                )
            else:
                rank0_print("Data caching completed")

    def _load_ckpt(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a checkpoint file.

        Args:
            idx: Index of the checkpoint file

        Returns:
            Dictionary containing input_ids, target_hiddens,
                hidden_states, and loss_mask, or None if loading fails
        """
        ckpt_path = self.ckpt_files[idx]

        try:
            data = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            warnings.warn(
                f"Failed to load checkpoint {ckpt_path}: {e}. Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate required keys
        required_keys = [
            "input_ids",  # B, N
            "target_hiddens",  # B, N, D
            "hidden_states",  # B, N, 3*D
            "loss_mask",  # B, N
        ]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            warnings.warn(
                f"Checkpoint {ckpt_path} is missing required keys: {missing_keys}. "
                f"Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate tensor types
        for key in required_keys:
            if not isinstance(data[key], torch.Tensor):
                warnings.warn(
                    f"Value for key '{key}' in {ckpt_path} is not a torch.Tensor. "
                    f"Skipping this file.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

        attention_mask = torch.ones_like(data["input_ids"])
        data["attention_mask"] = attention_mask  # B, N
        return data

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        if self.cached_data is not None:
            return len(self.cached_data)
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - input_ids: Token IDs (torch.Tensor)
                - target_logits: Pre-computed logits from target
                    model (torch.Tensor)
                - hidden_states: Pre-computed hidden states from
                    target model (torch.Tensor)
                - loss_mask: Mask for loss computation (torch.Tensor)
        """
        if self.cached_data is not None:
            return self.cached_data[idx]
        else:
            # Try to load the checkpoint, retry with next valid index if fails
            max_retries = len(self.valid_indices)
            for _attempt in range(max_retries):
                actual_idx = self.valid_indices[idx % len(self.valid_indices)]
                data = self._load_ckpt(actual_idx)
                if data is not None:
                    return data
                else:
                    # Remove failed index from valid_indices
                    self.valid_indices.remove(actual_idx)
                    if len(self.valid_indices) == 0:
                        raise RuntimeError(
                            "All checkpoint files failed to load. " "Cannot continue training."
                        )
                    # Try next index
                    idx += 1

            # If all retries failed, raise error
            raise RuntimeError(f"Failed to load any valid checkpoint after {max_retries} attempts")


class OfflineVLMEagle3Dataset(OfflineEagle3Dataset):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Branch distillation: the trainer re-scores a token-substituted copy of
        # this sequence with a live target, and on a VLM that forward needs the
        # picture. Preprocessing it here rather than in the trainer keeps it on
        # the dataloader workers, off the step's critical path.
        # Which blocks of the stored aux concat this draft wants (None = all).
        self._aux_cols = _aux_layer_columns(self.data_dir)
        self._branch_target = os.environ.get("BRANCH_DISTILL_TARGET")
        self._branch_src = None
        if self._branch_target:
            rank0_print(
                f"[branch-distill] re-score images from {self._branch_target}"
            )

    def _branch_source(self):
        """Built lazily: each dataloader worker needs its own."""
        if self._branch_src is None:
            from .branch_image_source import BranchImageSource

            self._branch_src = BranchImageSource(self._branch_target)
        return self._branch_src

    def _load_ckpt(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a checkpoint file.

        Args:
            idx: Index of the checkpoint file

        Returns:
            Dictionary containing input_ids, target_hiddens,
                hidden_states, and loss_mask, or None if loading fails
        """
        ckpt_path = self.ckpt_files[idx]

        try:
            data = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            warnings.warn(
                f"Failed to load checkpoint {ckpt_path}: {e}. Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate required keys
        required_keys = [
            "input_ids",  # B, N
            "target_hiddens",  # B, N, D
            "hidden_states",  # B, N, 3*D
            "loss_mask",  # B, N
        ]
        # position_ids and inputs_embeds are optional keys,
        # - HF Transformers backend saves position_ids (torch.Tensor)
        # - position_ids is stored as None unless the generator captured it
        optional_tensor_keys = [
            "position_ids",  # 3, B, N (optional; often stored as None)
            "inputs_embeds",  # B, N, D (optional)
        ]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            warnings.warn(
                f"Checkpoint {ckpt_path} is missing required keys: {missing_keys}. "
                f"Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate tensor types for required keys
        for key in required_keys:
            if not isinstance(data[key], torch.Tensor):
                warnings.warn(
                    f"Value for key '{key}' in {ckpt_path} is not a torch.Tensor. "
                    f"Skipping this file.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

        # Validate tensor types for optional keys
        for key in optional_tensor_keys:
            if key in data and not isinstance(data[key], torch.Tensor):
                data[key] = None

        if self._aux_cols is not None:
            hs = data["hidden_states"]
            block = data["target_hiddens"].shape[-1]
            data["hidden_states"] = torch.cat(
                [hs[..., i * block : (i + 1) * block] for i in self._aux_cols], dim=-1
            )

        attention_mask = torch.ones_like(data["input_ids"])
        data["attention_mask"] = attention_mask  # B, N
        if self._branch_target:
            images = self._branch_source().prepare(data, ckpt_path)
            if images is not None:
                data["branch_image"] = images
        return data




# SmolVLM/Idefics3 need nothing model-specific here: by the time an offline run
# reads these .ckpt files the target forward has already happened, so the
# builder only pads input_ids/loss_mask/hidden_states. The image-token
# expansion that makes SmolVLM special lives in the ONLINE builder, which is
# what tools/generate_hidden_for_draft_model.py uses to produce the .ckpt in
# the first place.
@DatasetBuilderFactory.register("offline", "VLM", "smolvlm")
@DatasetBuilderFactory.register("offline", "VLM", "idefics3")
class OfflineVLMDatasetBuilder(DatasetBuilder):
    def __init__(self, file_pattern: str = "*.ckpt", cache_in_memory: bool = False, **kwargs: Any):
        self.file_pattern = file_pattern
        self.cache_in_memory = cache_in_memory

    def build_dataset(self, datapath: str, **kwargs: Any) -> Dataset:
        """
        Create offline datasets from pre-computed .ckpt files.
        """
        return OfflineVLMEagle3Dataset(
            data_dir=datapath,
            file_pattern=self.file_pattern,
            cache_in_memory=self.cache_in_memory,
        )

    def get_data_collator(self) -> Any:
        return VLMDataCollatorWithPadding()


