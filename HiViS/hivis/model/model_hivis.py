import copy
import json
import time

from safetensors import safe_open
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
# from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .modeling_qwen2_5_vl_kv import Qwen2_5_VLForConditionalGeneration


from .utils_hivis import *
from .kv_cache import initialize_past_key_values
from .cnets_hivis import Model as HiViSModel
from .cnets_vispec import Model as ViSpecModel
from .configs import EConfig


DRAFT_MODELS = {
    "hivis": HiViSModel,
    "vispec": ViSpecModel,
}

# AngelSlim EAGLE-3 drafts are built from their own checkpoint rather than
# EConfig + a DRAFT_MODELS class -- see hivis/model/angelslim_drafter.py.
ANGELSLIM_EAGLE3 = "angelslim_eagle3"





class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            draft_method="hivis",
    ):

        super().__init__()
        self.embed_model = base_model
        self.architecture = base_model.config.architectures[0]
        self.is_qwen_vl = self.architecture == "Qwen2_5_VLForConditionalGeneration"
        self.is_smolvlm = self.architecture == "Idefics3ForConditionalGeneration"
        # Resolved off the FULL VLM config (embed_model), not the text backbone
        # rebound below -- the text tower's config carries no image token.
        self.image_token_id = resolve_image_token_id(self.embed_model.config)
        if draft_method not in DRAFT_MODELS and draft_method != ANGELSLIM_EAGLE3:
            raise ValueError(
                f"Unsupported draft method: {draft_method}. "
                f"Choose from {sorted(list(DRAFT_MODELS) + [ANGELSLIM_EAGLE3])}"
            )
        self.draft_method = draft_method
        if self.is_smolvlm:
            # SmolVLM has no .language_model wrapper; text backbone is model.text_model.
            base_model = base_model
        elif hasattr(base_model, "language_model"):
            base_model = base_model.language_model
        else:
            base_model = base_model.model.language_model
        self.base_model = base_model
        self.config = base_model.config
        try:
            self.hidden_size = base_model.lm_head.weight.shape[-1]
            self.vocab_size = base_model.lm_head.weight.shape[0]
        except:
            self.hidden_size = self.embed_model.lm_head.weight.shape[-1]
            self.vocab_size = self.embed_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        self.processor = AutoProcessor.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        if not self.is_qwen_vl:
            config.max_position_embeddings = 32768
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        # Training (hivis/train/cnets_res.Model) defaults bias=True and never
        # reads config.json's "bias". SmolVLM's copied config says false, so
        # the saved tensors (fc.bias present) disagree with the json. Trust
        # the checkpoint.
        if "fc.weight" in ea_layer_state_dict:
            bias = "fc.bias" in ea_layer_state_dict
        # AngelSlim EAGLE-3: the drafter carries its own config and weights, and
        # needs the target's aux-layer hidden states rather than the last layer.
        # Cap for the preallocated static KV cache; None = max_position_embeddings.
        # See initialize_past_key_values -- a 128k default OOMs a 7B target.
        self.cache_max_len = None
        self.aux_layer_ids = None
        if draft_method == ANGELSLIM_EAGLE3:
            from .angelslim_drafter import build_drafter

            checkpoint_dir = os.path.dirname(ea_model_path)
            self.ea_layer, _raw, _dropped = build_drafter(
                checkpoint_dir,
                total_tokens=total_token,
                depth=depth,
                top_k=top_k,
                threshold=threshold,
                dtype=base_model.dtype,
            )
            self.aux_layer_ids = tuple(self.ea_layer.aux_hidden_states_layer_ids)

        draft_model_class = DRAFT_MODELS.get(draft_method)
        draft_model_kwargs = {
            "bias": bias,
            "total_tokens": total_token,
            "depth": depth,
            "top_k": top_k,
            "threshold": threshold,
        }
        if draft_method == "hivis":
            residual = ea_layer_state_dict.get("residual")
            draft_model_kwargs["residual_count"] = residual.shape[0]
        if draft_method in {"eagle", "vispec"}:
            draft_model_kwargs["path"] = base_model_name_or_path
        if draft_method != ANGELSLIM_EAGLE3:
            self.ea_layer = draft_model_class(config, **draft_model_kwargs)

        low_memory=False

        if self.is_smolvlm:
            device = base_model.model.text_model.layers[-1].self_attn.q_proj.weight.device
            lm_head_device = self.embed_model.lm_head.weight.device
        else:
            try:
                device = base_model.model.layers[-1].self_attn.q_proj.weight.device
                lm_head_device = self.base_model.lm_head.weight.device
            except Exception:
                device = base_model.layers[-1].self_attn.q_proj.weight.device
                lm_head_device = self.embed_model.lm_head.weight.device
        if device != lm_head_device:
            self.ea_layer.diff_device = True
            if not low_memory:
                lm_head_src = self.embed_model if self.is_smolvlm else self.base_model
                self.ea_layer.headweight = lm_head_src.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device
        else:
            self.ea_layer.diff_device = False

        if self.is_smolvlm:
            # SmolVLM's text stack is a STOCK transformers LlamaModel, which
            # neither accepts HiViS's legacy list KVCache nor honours
            # `tree_mask`. Both are required by the tree-verify loop, and both
            # already exist in HiViS's own patched Llama, so swap that in with
            # the target's weights. Same tensors, tree-capable attention.
            from .modeling_llama_kv import LlamaModel as KVLlamaModel

            _stock = self.embed_model.model.text_model
            # HiViS's fork predates the sdpa dispatch flags, so it must be
            # built as "eager"; its attention is a hand-written matmul+softmax
            # that consumes the 4D tree mask directly. This is the same
            # attention HiViS's own baselines run, so comparisons stay
            # backend-matched.
            _text_cfg = copy.deepcopy(self.embed_model.config.text_config)
            _text_cfg._attn_implementation = "eager"
            _text = KVLlamaModel(_text_cfg)
            _missing, _unexpected = _text.load_state_dict(
                _stock.state_dict(), strict=False
            )
            # rotary_emb.inv_freq is a derived buffer each module builds for
            # itself; modern transformers stopped shipping it in state dicts.
            _missing = [k for k in _missing if not k.endswith("rotary_emb.inv_freq")]
            if _missing or _unexpected:
                raise RuntimeError(
                    "tree-capable text backbone did not match the target: "
                    "missing=%s unexpected=%s" % (_missing[:6], _unexpected[:6])
                )
            _text = _text.to(dtype=_stock.dtype, device=_stock.embed_tokens.weight.device)
            _text.eval()
            self.text_backbone = _text
            self.embed_model.model.text_model = _text
        else:
            self.text_backbone = None

        print(base_model_name_or_path)
        if draft_method != ANGELSLIM_EAGLE3:
            self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        self.ea_layer.embed_model = self.embed_model

    def _target_input_embeds(self, inputs):
        """Prompt embeddings + positions, for targets HiViS did not patch."""
        if self.is_smolvlm:
            return smolvlm_input_embeds(
                self.embed_model,
                inputs["input_ids"],
                inputs.get("pixel_values"),
                inputs.get("pixel_attention_mask"),
            )
        return self.embed_model(**inputs)

    def _target_lm(self, input_ids=None, inputs_embeds=None, past_key_values=None,
                   position_ids=None):
        """Plain (non-tree) target forward exposing `.logits`.

        The autoregressive baseline has to run through the SAME backbone as the
        speculative path or the speedup ratio is meaningless -- so it goes
        through the swapped-in tree-capable decoder too, with the tree mask
        cleared.
        """
        if self.text_backbone is None:
            return self.base_model(
                input_ids, inputs_embeds=inputs_embeds, use_cache=True,
                past_key_values=past_key_values, position_ids=position_ids,
            )
        self.text_backbone.tree_mask = None
        out = self.text_backbone(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, position_ids=position_ids,
            use_cache=True, return_dict=True,
        )
        return CausalLMOutputWithPast(logits=self.base_model.lm_head(out.last_hidden_state))

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            draft_method="hivis",
            **kwargs,
    ):
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=="Qwen2_5_VLForConditionalGeneration":
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
               base_model_path, **kwargs
           )
        elif Type=="Qwen3VLForConditionalGeneration":
            from transformers import Qwen3VLForConditionalGeneration
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=="Idefics3ForConditionalGeneration":
            base_model = Idefics3ForConditionalGeneration.from_pretrained(
                base_model_path, **kwargs
            )

        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            if os.path.isdir(ea_model_path):
                # accelerate's per-epoch checkpoints (hivis/train/main_mix.py)
                # save tensors only, never a config.json. A hand-copied config
                # from another run silently mismatches the moment hidden_size,
                # intermediate_size or layer count differs -- so derive it from
                # this checkpoint's own weight shapes instead, plus the
                # attention hyperparams a hivis/eagle/vispec drafter always
                # shares with its target (it's a K-layer clone of it). Written
                # to disk so this only has to happen once per checkpoint.
                target_config = getattr(base_model.config, "text_config", base_model.config)
                n_layers = len({
                    k.split(".")[1] for k in ea_layer_state_dict if k.startswith("layers.")
                })
                cfg_dict = {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "vocab_size": ea_layer_state_dict["embed_tokens.weight"].shape[0],
                    "hidden_size": ea_layer_state_dict["embed_tokens.weight"].shape[1],
                    "intermediate_size": ea_layer_state_dict["layers.0.mlp.gate_proj.weight"].shape[0],
                    "num_hidden_layers": n_layers,
                    "num_attention_heads": target_config.num_attention_heads,
                    "num_key_value_heads": getattr(
                        target_config, "num_key_value_heads", target_config.num_attention_heads
                    ),
                    "hidden_act": getattr(target_config, "hidden_act", "silu"),
                    "rms_norm_eps": getattr(target_config, "rms_norm_eps", 1e-5),
                    "rope_theta": getattr(target_config, "rope_theta", 10000.0),
                    "max_position_embeddings": getattr(
                        target_config, "max_position_embeddings", 32768
                    ),
                    "bias": "fc.bias" in ea_layer_state_dict,
                }
                with open(configpath, "w") as f:
                    json.dump(cfg_dict, f, indent=2)
            else:
                configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            draft_method=draft_method,
        )



        if total_token==-1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]
            times=[]

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token=cans[times.index(min(times))]
            model.ea_layer.total_tokens=total_token-1




        return model

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            if self.is_qwen_vl:
                if self.aux_layer_ids is not None:
                    raise NotImplementedError(
                        "EAGLE-3 aux-layer capture is only wired for the non-Qwen "
                        "path; this branch would silently hand back last_hidden_state."
                    )
                outputs = self.base_model(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                hidden_states = outputs[0]
                if output_orig:
                    orig = self.embed_model.lm_head(hidden_states)
            else:
                # EAGLE-3 drafts consume several mid-stack layers, not just the
                # last one, so the aux layers are collected here and handed
                # downstream in place of last_hidden_state. HF returns
                # hidden_states[0] = embeddings and hidden_states[k+1] = the
                # output of layer k, which is the same indexing AngelSlim's own
                # vLLM hook uses (all_hidden_states[k] = output of layer k).
                want_aux = self.aux_layer_ids is not None
                target = self.base_model.model
                if self.text_backbone is not None:
                    # Call the text stack directly: the multimodal wrapper's
                    # forward would re-run the vision tower and pass kwargs the
                    # patched decoder does not take. Image rows are already
                    # merged into inputs_embeds by smolvlm_input_embeds.
                    target = self.text_backbone
                    # eagenerate sets the round's tree mask on base_model.model;
                    # it has to reach the module that builds the causal mask.
                    target.tree_mask = getattr(self.base_model.model, "tree_mask", None)
                outputs = target(
                    input_ids=None if inputs_embeds is not None else input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=want_aux,
                    output_attentions=False,
                )
                if output_orig:
                    orig = self.base_model.lm_head(outputs.last_hidden_state).float()
                if want_aux:
                    hs = outputs.hidden_states
                    hidden_states = torch.cat(
                        [hs[i + 1] for i in self.aux_layer_ids], dim=-1
                    )
                else:
                    hidden_states = outputs.last_hidden_state

        if output_orig:
            return outputs, orig, hidden_states
        return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            inputs,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=4096,
            log=False,
            is_llama3=False,

    ):
        input_ids = inputs.input_ids.to(self.base_model.device)
        pixel_values = inputs.pixel_values.to(self.base_model.device).to(torch.bfloat16) if hasattr(inputs, "pixel_values") else None
        image_sizes = inputs.image_sizes.to(self.base_model.device) if hasattr(inputs, "image_sizes") else None
        image_grid_thw = inputs.image_grid_thw.to(self.base_model.device) if hasattr(inputs, "image_grid_thw") else None
        pixel_values_videos = inputs.pixel_values_videos.to(self.base_model.device).to(torch.bfloat16) if hasattr(inputs, "pixel_values_videos") else None
        video_grid_thw = inputs.video_grid_thw.to(self.base_model.device) if hasattr(inputs, "video_grid_thw") else None
        second_per_grid_ts = inputs.second_per_grid_ts.to(self.base_model.device) if hasattr(inputs, "second_per_grid_ts") else None


        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.cache_max_len)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)


        from .phase_timing import phase as _ph, commit as _ph_commit
        with _ph("init_total"):
            draft_input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor, self.embed_model, pixel_values, image_sizes, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts
            )
        # print("initialized", self.embed_model.model.rope_deltas)

        
        t1 = t2 = 0
        new_token = 0
        accept_length_list = []

        for idx in range(max_length):
            #with Timer("all"):
            try:
                self.base_model.model.tree_mask = tree_mask
            except:
                self.base_model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            start = time.time()
            _v = _ph("verify").__enter__()
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            _v.__exit__()
            t1 += time.time() - start
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            try:
                accept_length_list.append(accept_length.item())
            except:
                accept_length_list.append(accept_length)

            start = time.time()
            _u = _ph("draft_update").__enter__()
            input_ids, draft_input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                draft_input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
            )
            _u.__exit__()
            t2 += time.time() - start
            # print(f"init time {t0}, verify time {t1}, draft time {t2}")
            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    # print(f"init time {t0}, verify time {t1}, draft time {t2}")
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                # print("eos_token_id")

                # print(f"init time {t0}, verify time {t1}, draft time {t2}")
                break
            if new_token > max_new_tokens:
                # print(f"init time {t0}, verify time {t1}, draft time {t2}")
                break
            if input_ids.shape[1] > max_length:
                # print(f"init time {t0}, verify time {t1}, draft time {t2}")
                break
        _ph_commit(rounds=idx + 1, new_token=int(new_token))
        if not log:
            # print(f"init time {t0}, verify time {t1}, draft time {t2}")
            return input_ids
        else:
            # print(f"init time {t0}, verify time {t1}, draft time {t2}")
            return input_ids, new_token, idx, accept_length_list


    @torch.no_grad()
    def naivegenerate(
            self,
            inputs,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=4096,
            log=False,
            is_llama3=False,

    ):
        inputs.to(self.base_model.device)
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values.to(self.base_model.device) if hasattr(inputs, "pixel_values") else None
        image_sizes = inputs.image_sizes.to(self.base_model.device) if hasattr(inputs, "image_sizes") else None
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        torch.cuda.synchronize()
        start_time = time.time()
        self.ea_layer.reset_kv()
        

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.cache_max_len)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        end_time = time.time()
        init_time = end_time - start_time

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        gen_time = 0.0
        torch.cuda.synchronize()
        start_time = time.time()
        input_embeds, position_ids = self._target_input_embeds(inputs)
        outputs = self._target_lm(inputs_embeds=input_embeds, past_key_values=past_key_values, position_ids=position_ids)
        torch.cuda.synchronize()
        end_time = time.time()
        gen_time += end_time - start_time

        new_token = 0

        if not hasattr(outputs, 'logits'):
            for idx in range(max_length):
                torch.cuda.synchronize()
                start_time = time.time()
                orig = self.embed_model.lm_head(outputs[0])
                torch.cuda.synchronize()
                end_time = time.time()
                gen_time += end_time - start_time
                if logits_processor is not None:
                    logits = orig[:, -1]
                    logits = logits_processor(None, logits)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                    input_id = torch.multinomial(probabilities, 1)
                else:
                    token = torch.argmax(orig[:, -1])
                    input_id = token[None, None]
                torch.cuda.synchronize()
                start_time = time.time()
                position_ids = torch.tensor([input_ids.shape[1]]).to(self.embed_model.model.rope_deltas.device)
                position_ids = (
                    position_ids.unsqueeze(0) + self.embed_model.model.rope_deltas
                )
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                outputs = self.base_model(input_id, position_ids=position_ids,use_cache=True, past_key_values=past_key_values)
                torch.cuda.synchronize()
                end_time = time.time()
                gen_time += end_time - start_time
                input_ids = torch.cat([input_ids, input_id], dim=-1)
                new_token+=1

                if is_llama3:
                    if stop_token_id in input_ids[0, input_len:].tolist():
                        break

                if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    break
                if new_token > max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        
        else:
            for idx in range(max_length):
                if logits_processor is not None:
                    logits = outputs.logits[:, -1]
                    logits = logits_processor(None, logits)
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    input_id = torch.multinomial(probabilities, 1)
                else:
                    input_id = outputs.logits[:, -1:].argmax(dim=-1)
                outputs = self._target_lm(input_id, past_key_values=past_key_values)
                input_ids = torch.cat([input_ids, input_id], dim=-1)
                new_token+=1

                if is_llama3:
                    if stop_token_id in input_ids[0, input_len:].tolist():
                        break

                if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    break
                if new_token > max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        if not log:
            return input_ids
        else:
            # return input_ids, new_token, idx
            return input_ids, new_token, idx, init_time
        

    @torch.no_grad()
    def ea_generate(
            self,
            inputs,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=4096,
            log=False,
            is_llama3=False,

    ):
        input_ids = inputs.input_ids.to(self.base_model.device)
        pixel_values = inputs.pixel_values.to(self.base_model.device).to(torch.bfloat16) if hasattr(inputs, "pixel_values") else None
        image_sizes = inputs.image_sizes.to(self.base_model.device) if hasattr(inputs, "image_sizes") else None

        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.cache_max_len)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        start = time.time()
        draft_input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor, self.embed_model, pixel_values, image_sizes
        )
        t0 = time.time() - start
        t1 = t2 = 0
        new_token = 0

        for idx in range(1, max_length+1):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            start = time.time()
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            t1 += time.time() - start
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            if True:
                for i, candidate in enumerate(candidates):
                    # Filter out padding tokens (-1)
                    valid_tokens = [t.item() for t in candidate if t.item() != -1]
                    # Convert tokens to text if you have access to tokenizer
                    # candidate_text = tokenizer.decode(valid_tokens)
                    text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    #print(f"Candidate {i}: [{text}]")
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            #with Timer("update_inference_inputs"):
            start = time.time()
            input_ids, draft_input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                draft_input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
            )
            t2+=time.time()-start

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    # print(f"init time {t0}, verify time {t1/idx}, draft time {t2/idx}")
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                # print(f"init time {t0}, verify time {t1/idx}, draft time {t2/idx}")
                break
            if new_token > max_new_tokens:
                # print(f"init time {t0}, verify time {t1/idx}, draft time {t2/idx}")
                break
            if input_ids.shape[1] > max_length:
                # print(f"init time {t0}, verify time {t1/idx}, draft time {t2/idx}")
                break


    @torch.no_grad()
    def naive_generate(
            self,
            inputs,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=4096,
            log=False,
            is_llama3=False,

    ):
        input_ids = inputs.input_ids.to(self.base_model.device)
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.cache_max_len)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        input_embeds, position_ids = self.embed_model(**inputs)
        outputs = self.base_model(inputs_embeds=input_embeds, past_key_values=past_key_values, position_ids=position_ids,use_cache=True)
        new_token = 0


        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids



            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
