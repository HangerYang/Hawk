import copy
import random

# typing 
from typing import List, Tuple
import time
import torch
from torch.nn.utils.rnn import pad_sequence

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def build_mask(input_ids: torch.Tensor):
    VISION_START = 151652
    IMAGE_TOKEN  = 151655
    # IMAGE_TOKEN  = 151656

    mask = torch.ones_like(input_ids, dtype=torch.bool)

    pos = (input_ids == VISION_START).nonzero(as_tuple=False).squeeze(-1)
    if len(pos) == 0:
        return mask

    start = pos.item()
    count = int((input_ids == IMAGE_TOKEN).sum().item())
    end = min(start + count + 1, input_ids.numel() - 1)

    mask[start:end+2] = False

    return mask


# LLaVA-family configs that predate an explicit image-token field. Every model
# wired up here does expose one, so this only ever backs a hand-built config.
IMAGE_TOKEN_ID_FALLBACK = 32000


def resolve_image_token_id(config, default=IMAGE_TOKEN_ID_FALLBACK):
    """The `<image>` placeholder id for a VLM, read from that model's own config.

    Families name it differently and the values are nowhere near each other
    (Qwen2.5-VL 151655, SmolVLM/Idefics3 49190, LLaVA 32000), so branching on
    the architecture and hardcoding a constant per branch fails silently the
    moment a model lands outside those branches: the image mask comes out
    all-False, the visual span is never compressed or pruned, and the draft
    still runs -- just as a plain EAGLE forward on the full sequence, with no
    error to notice.
    """
    for attr in ("image_token_id", "image_token_index"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    return default


def _target_image_token_id(model, default=32000):
    """The image placeholder id of whatever target this EaModel wraps.

    Upstream hardcodes 32000 (LLaVA) for every non-Qwen target. SmolVLM /
    Idefics3 uses 49190, so on that target nothing was pruned and a HiViS
    drafter -- whose whole premise is that it never sees the visual span --
    was handed ~800 image rows it was never trained on. Acceptance pinned at
    exactly tau=1.0 on every prompt.
    """
    cfg = getattr(getattr(model, "base_model", model), "config", None)
    for holder in (cfg, getattr(cfg, "text_config", None)):
        if holder is None:
            continue
        tok = resolve_image_token_id(holder, default=None)
        if tok is not None:
            return tok
    return default

def prune_image_tokens(
    input_ids: torch.LongTensor,
    embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    text_position_ids: torch.Tensor = None,
    image_token_id: int = 32000,
    pad_token_id: int = 0,
):
    batch_size = input_ids.shape[0]
    hidden_size = embeds.size(-1)
    device = hidden_states.device
    input_ids = input_ids.to(device)
    embeds = embeds.to(device)

    ids_list = []
    embeds_list = []
    states_list = []
    position_ids_list = []

    for batch_index in range(batch_size):
        if text_position_ids is None:
            keep_mask = input_ids[batch_index] != image_token_id
        else:
            keep_mask = build_mask(input_ids[batch_index])

        kept_ids = input_ids[batch_index][keep_mask]
        kept_embeds = embeds[batch_index][keep_mask]
        kept_states = hidden_states[batch_index][keep_mask]
        kept_position_ids = (
            text_position_ids[batch_index][keep_mask]
            if text_position_ids is not None
            else None
        )

        if kept_ids.numel() == 0:
            kept_ids = torch.tensor([pad_token_id], device=device, dtype=input_ids.dtype)
            kept_embeds = torch.zeros((1, hidden_size), device=device, dtype=embeds.dtype)
            kept_states = torch.zeros((1, hidden_size), device=device, dtype=hidden_states.dtype)
            if kept_position_ids is not None:
                kept_position_ids = torch.zeros(
                    1, device=device, dtype=text_position_ids.dtype
                )

        ids_list.append(kept_ids)
        embeds_list.append(kept_embeds)
        states_list.append(kept_states)
        if kept_position_ids is not None:
            position_ids_list.append(kept_position_ids)

    if batch_size == 1:
        pruned_input_ids = ids_list[0].unsqueeze(0)
        pruned_embeds = embeds_list[0].unsqueeze(0)
        pruned_hidden_states = states_list[0].unsqueeze(0)
    else:
        pruned_input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_token_id)
        pruned_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0.0)
        pruned_hidden_states = pad_sequence(states_list, batch_first=True, padding_value=0.0)

    if text_position_ids is None:
        return pruned_input_ids, pruned_embeds, pruned_hidden_states

    if batch_size == 1:
        pruned_position_ids = position_ids_list[0].unsqueeze(0)
    else:
        pruned_position_ids = pad_sequence(position_ids_list, batch_first=True, padding_value=0)
    return pruned_input_ids, pruned_embeds, pruned_hidden_states, pruned_position_ids

class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers

from .phase_timing import phase as _ph, commit as _ph_commit  # noqa: E402


def initialize_tree(
    input_ids,
    model,
    past_key_values,
    logits_processor,
    embed_model,
    pixel_values,
    image_sizes,
    image_grid_thw=None,
    pixel_values_videos=None,
    video_grid_thw=None,
    second_per_grid_ts=None,
):
    _r = getattr(model, "draft_image_reducer", None)
    if _r is not None and hasattr(_r, "prepare"):
        with _ph("prepare"):
            _r.prepare(input_ids)
    _e = _ph("embed").__enter__()
    if getattr(model, "is_smolvlm", False):
        # HiViS patches its llava/qwen target classes so that calling them
        # returns (inputs_embeds, position_ids). Idefics3/SmolVLM is loaded
        # straight from transformers, so build the merged embeddings with the
        # model's own helpers instead of a patched forward.
        input_embeds, position_ids = smolvlm_input_embeds(
            embed_model, input_ids, pixel_values, image_sizes
        )
    elif image_grid_thw is not None:
        input_embeds, position_ids = embed_model(input_ids, pixel_values=pixel_values, image_sizes=image_sizes, image_grid_thw=image_grid_thw)
    elif pixel_values_videos is not None:
        input_embeds, position_ids = embed_model(input_ids, pixel_values_videos=pixel_values_videos, second_per_grid_ts=second_per_grid_ts, video_grid_thw=video_grid_thw)
    else:
        input_embeds, position_ids = embed_model(input_ids, pixel_values=pixel_values, image_sizes=image_sizes)


    _e.__exit__()

    # input_embeds, index_mask = embed_model(input_ids, pixel_values=pixel_values, image_sizes=image_sizes, vision_feature_select_strategy="full")
    _p = _ph("prefill").__enter__()
    outputs, orig, hidden_states = model(
        input_ids=None if model.is_qwen_vl else input_ids,
        inputs_embeds=input_embeds,
        past_key_values=past_key_values, 
        position_ids=position_ids, 
        output_orig=True
    )
    _p.__exit__()

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]

    new_position_ids = None
    image_mask = None
    if model.draft_method == "vispec":
        # ViSpec keeps the visual span in the draft prompt. Its ImgAdaptor uses
        # this mask to compress that span inside the draft model.
        draft_input_ids = input_ids
        draft_embeds = input_embeds
        draft_hidden_states = hidden_states
        image_token_id = 151655 if model.is_qwen_vl else _target_image_token_id(model)
        image_mask = draft_input_ids == image_token_id
        draft_input_embeds = draft_embeds
    else:
        if model.draft_method in ("eagle", "angelslim_eagle3"):
            # EAGLE consumes the complete multimodal sequence without visual
            # pruning or an image-specific adaptor. The AngelSlim EAGLE-3 drafts
            # were trained the same way (image rows visible to the drafter), so
            # they share this branch; only `hidden_states` differs -- it is the
            # aux-layer concat rather than the last layer.
            draft_input_ids = input_ids
            draft_embeds = input_embeds
            draft_hidden_states = hidden_states
            # ...unless this checkpoint was trained on FEWER image rows than the
            # target sees. The target's prefill above is untouched and still ran
            # at full resolution; only what the drafter is handed changes here.
            reducer = getattr(model, "draft_image_reducer", None)
            if reducer is not None:
                with _ph("reduce"):
                    reduced = reducer(input_ids, hidden_states)
                if reduced is not None:
                    draft_input_ids, draft_hidden_states, new_position_ids = reduced
        elif model.is_qwen_vl:
            seq_length = input_ids.size(1)
            text_position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            (
                draft_input_ids,
                draft_embeds,
                draft_hidden_states,
                new_position_ids,
            ) = prune_image_tokens(
                input_ids=input_ids,
                embeds=input_embeds,
                hidden_states=hidden_states,
                text_position_ids=text_position_ids,
                image_token_id=151655,
            )
        else:
            draft_input_ids, draft_embeds, draft_hidden_states = (
                prune_image_tokens(
                    input_ids=input_ids,
                    embeds=input_embeds,
                    hidden_states=hidden_states,
                    image_token_id=_target_image_token_id(model),
                )
            )

        new_embeds = model.ea_layer.embed_tokens(token.to(draft_embeds.device))
        draft_input_embeds = torch.cat((draft_embeds, new_embeds), dim=1).to(
            input_embeds.device
        )

    draft_ids_with_token = torch.cat(
        (draft_input_ids, token.to(draft_input_ids.device)), dim=1
    )

    def generate_initial_tree(head):
        if model.draft_method == "angelslim_eagle3":
            # AngelSlim's drafter owns its lm_head, so it takes no `head`.
            return model.ea_layer.topK_genrate(
                draft_hidden_states,
                draft_ids_with_token,
                inputs_embeds=None,
                logits_processor=logits_processor,
                position_ids=new_position_ids,
            )
        if model.draft_method == "vispec":
            return model.ea_layer.topK_genrate(
                draft_hidden_states,
                draft_ids_with_token,
                head,
                logits_processor,
                input_embeds=draft_input_embeds,
                image_mask=image_mask,
            )
        if model.draft_method == "eagle":
            return model.ea_layer.topK_genrate(
                draft_hidden_states,
                draft_ids_with_token,
                head,
                logits_processor,
                input_embeds=draft_input_embeds,
            )
        return model.ea_layer.topK_genrate(
            draft_hidden_states,
            draft_ids_with_token,
            head,
            logits_processor,
            input_embeds=draft_input_embeds,
            position_ids=new_position_ids,
        )

    with _ph("first_tree"):
        try:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
                generate_initial_tree(model.base_model.lm_head)
            )
        except AttributeError:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
                generate_initial_tree(model.embed_model.lm_head)
            )
    return draft_input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token


def reset_tree_mode(
        model,
):
    try:
        model.base_model.model.tree_mask = None
        model.base_model.model.tree_mode = None
    except:
        model.base_model.tree_mask = None
        model.base_model.tree_mode = None



def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    # print(model.base_model.config.architectures[0])
    
    position_ids = tree_position_ids + input_ids.shape[1]
    if model.base_model.config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
        position_ids = (
            position_ids.unsqueeze(0) + model.embed_model.model.rope_deltas
        )
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    # print(input_ids.shape[1])
    # print("position_ids: ", position_ids)
    # print(model.embed_model.model.rope_deltas)
    # raise RuntimeError

    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs





def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
    input_ids,
    draft_input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token,
    past_key_values_data_list,
    current_length_data,
    model,
    hidden_state_new,
    sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    draft_input_ids = torch.cat(
        [draft_input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    if model.draft_method == "angelslim_eagle3":
        # accept_hidden_state_new is the aux-layer concat for the accepted
        # positions; the drafter re-runs its band mix + FC over it.
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((draft_input_ids, token.to(input_ids.device)), dim=1),
            inputs_embeds=None,
            logits_processor=logits_processor)
        new_token += accept_length + 1
        return (input_ids, draft_input_ids, draft_tokens, retrieve_indices,
                tree_mask, tree_position_ids, new_token, None, token)
    try:
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new, 
            input_ids=torch.cat((draft_input_ids, token.to(input_ids.device)), dim=1), 
            head=model.base_model.lm_head,
            logits_processor=logits_processor)
    except:
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new, 
            input_ids=torch.cat((draft_input_ids, token.to(input_ids.device)), dim=1), 
            head=model.embed_model.lm_head,
            logits_processor=logits_processor)


    new_token += accept_length + 1

    return input_ids, draft_input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)


def smolvlm_input_embeds(embed_model, input_ids, pixel_values, pixel_attention_mask=None):
    """Text embeddings with the vision tower's image rows scattered in.

    Mirrors what Idefics3Model.forward does before the text stack, using the
    model's own get_image_features / inputs_merger so the merge rule (and the
    image-token id it keys on) stays whatever transformers says it is.
    Positions are a plain arange: Idefics3 has no mrope.
    """
    inner = getattr(embed_model, "model", embed_model)
    inputs_embeds = inner.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        image_hidden_states = inner.get_image_features(
            pixel_values.to(dtype=embed_model.dtype),
            pixel_attention_mask,
        )
        inputs_embeds = inner.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_hidden_states.to(inputs_embeds.dtype),
        )
    position_ids = torch.arange(
        input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)
    return inputs_embeds, position_ids
