import math
import torch
import torch.nn.functional as F

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache



class Qwen2AttentionD2O(Qwen2Attention):

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.layer_nums = config.num_hidden_layers
        self.hh_ratio = config.hh_ratio
        self.recent_ratio = config.recent_ratio
        self.uniform_hh_ratio = config.hh_ratio
        self.uniform_recent_ratio = config.recent_ratio
        self.layer_hh_ratio = 0.0
        self.layer_recent_ratio = 0.0
        self.hh_size = config.hh_size
        self.recent_size = config.recent_size
        self.alpha = config.alpha
        self.belta = config.belta

        self.hh_score = None
        self.threshold = None
        self.prefill_score = None
        self.prefill_size = 0
        self.cache_size = None

        self.multimodal_entropy = 0

    def _clean_cache(self):
        self.hh_score = None
        self.threshold = None
        self.prefill_score = None
        self.prefill_size = 0
        self.cache_size = None
        self.multimodal_entropy = 0
        self.layer_hh_ratio = 0.0
        self.layer_recent_ratio = 0.0

    def _attention_to_kv_scores(self, attn_weights):

        scores = attn_weights.detach()

        bsz, num_heads, q_len, kv_len = scores.shape

        if num_heads != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} attention heads, instead got {num_heads}"
            )

        scores = scores.reshape(
            bsz,
            self.num_key_value_heads,
            self.num_key_value_groups,
            q_len,
            kv_len,
        )

        scores = scores.sum(dim = 2)
        scores = scores.sum(dim = 0).sum(dim = 1)

        return scores

    def _store_prefill_scores(self, attn_weights):

        self.prefill_size = attn_weights.shape[-1]
        self.prefill_score = self._attention_to_kv_scores(attn_weights)

    def _build_first_decode_scores(self, attn_weights):

        new_scores = self._attention_to_kv_scores(attn_weights)

        if self.prefill_score is None:
            return new_scores

        prefix_score = (
            self.prefill_score + new_scores[:, :self.prefill_size]
        )

        new_token_scores = new_scores[:, self.prefill_size:]

        return torch.cat(
            [prefix_score, new_token_scores],
            dim = -1
        )

    def _select_keep_indices(self, scores):

        seq_len = scores.shape[-1]

        if self.cache_size is None:
            hh_ratio = (
                self.layer_hh_ratio
                if self.layer_hh_ratio > 0
                else self.hh_ratio
            )

            recent_ratio = (
                self.layer_recent_ratio
                if self.layer_recent_ratio > 0
                else self.recent_ratio
            )

            self.hh_size = int(seq_len * hh_ratio)
            self.recent_size = int(seq_len * recent_ratio)
            self.cache_size = self.hh_size + self.recent_size

        if seq_len <= self.cache_size:
            return None

        select_hh_scores = scores[:, :seq_len - self.recent_size].clone()

        protected = min(4, select_hh_scores.shape[-1])
        select_hh_scores[:, :protected] = float("inf")
        sort_indices = torch.argsort(
            select_hh_scores,
            dim = -1,
            descending = True,
        )

        keep_topk = sort_indices[:, :self.hh_size].sort(dim = -1).values

        keep_recent = torch.arange(
            seq_len - self.recent_size,
            seq_len,
            device = scores.device,
        ).unsqueeze(0).expand(scores.shape[0], -1)

        return torch.cat(
            [keep_topk, keep_recent],
            dim = -1,
        )

    def _split_cache(self, key_states, value_states, keep_idx):

        bsz, num_kv_heads, seq_len, head_dim = key_states.shape

        keep_mask = torch.zeros(
            num_kv_heads,
            seq_len,
            dtype = torch.bool,
            device = key_states.device,
        )

        keep_mask.scatter_(1, keep_idx, True)

        all_idx = torch.arange(
            seq_len,
            device = key_states.device,
        ).unsqueeze(0).expand(num_kv_heads, -1)

        pruned_idx = all_idx[~keep_mask].view(
            num_kv_heads,
            seq_len - keep_idx.shape[-1],
        )

        def gather_by_head(states, indices):

            expanded = indices.unsqueeze(0).unsqueeze(-1).expand(
                bsz,
                num_kv_heads,
                indices.shape[-1],
                head_dim,
            )

            return torch.gather(
                states,
                dim = -2,
                index = expanded,
            )

        kept_keys = gather_by_head(key_states, keep_idx)
        kept_values = gather_by_head(value_states, keep_idx)
        pruned_keys = gather_by_head(key_states, pruned_idx)
        pruned_values = gather_by_head(value_states, pruned_idx)

        return kept_keys, kept_values, pruned_keys, pruned_values

    def _merge_pruned_into_kept(
            self,
            kept_keys,
            kept_values,
            pruned_keys,
            pruned_values,
    ):

        if pruned_keys.shape[-2] == 0:
            return kept_keys, kept_values

        normalized_pruned = pruned_keys / torch.norm(
            pruned_keys,
            dim = -1,
            keepdim = True,
        ).clamp_min(1e-6)

        normalized_kept = kept_keys / torch.norm(
            kept_keys,
            dim = -1,
            keepdim = True,
        ).clamp_min(1e-6)

        similarity = normalized_pruned @ normalized_kept.transpose(-1, -2)

        max_values, max_indices = similarity.max(dim = -1)

        current_threshold = max_values.mean()

        if self.threshold is None:
            self.threshold = current_threshold.detach()
        else:
            self.threshold = (
                self.alpha * self.threshold + self.belta * current_threshold.detach()
            )

        filter_indices = (max_values.mean(dim = (0,1)) >= self.threshold)

        if not filter_indices.any():
            return kept_keys, kept_values

        head_dim = kept_keys.shape[-1]

        merged_indices = (
            max_indices[..., filter_indices].unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        merge_weights = (
            max_values[..., filter_indices].unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        keys_to_merge = pruned_keys[..., filter_indices, :]
        values_to_merge = pruned_values[..., filter_indices, :]

        merged_keys = torch.scatter_reduce(
            input = kept_keys,
            dim = 2,
            index = merged_indices,
            src = merge_weights * keys_to_merge,
            reduce = "mean",
            include_self = True,
        )

        merged_values = torch.scatter_reduce(
            input = kept_values,
            dim = 2,
            index = merged_indices,
            src = merge_weights * values_to_merge,
            reduce = "mean",
            include_self = True,
        )

        return merged_keys, merged_values

    def _update_hh_score(self, scores):

        scores = scores.clone()

        if self.hh_score is None:
            self.hh_score = scores
            return

        if scores.shape[-1] != self.hh_score.shape[-1] + 1:
            raise ValueError(
                "Expected new attention scores to contain exactly one additional generated token"
                f"Old: {self.hh_score.shape[-1]}"
                f"New: {scores.shape[-1]}"
            )

        scores[..., :-1] += self.hh_score
        self.hh_score = scores

    def _prune_hh_score(self, keep_idx):

        self.hh_score = torch.gather(
            self.hh_score,
            dim = 1,
            index = keep_idx,
        )

    def _compress_dynamic_cache(self, past_key_value, scores):

        if past_key_value is None:
            return past_key_value

        key_states = past_key_value.key_cache[self.layer_idx]
        value_states = past_key_value.value_cache[self.layer_idx]

        if key_states.shape[-2] != scores.shape[-1]:
            raise ValueError(
                "Lenghts of cache and attention-score dont match"
                f"Cache = {key_states.shape[-2]}, scores = {scores.shape[-1]}"
            )

        self._update_hh_score(scores)

        keep_idx = self._select_keep_indices(self.hh_score)
        if keep_idx is None:
            return past_key_value

        kept_k, kept_v, pruned_k, pruned_v = self._split_cache(key_states, value_states, keep_idx)

        merged_k, merged_v = self._merge_pruned_into_kept(kept_k, kept_v, pruned_k, pruned_v)

        past_key_value.key_cache[self.layer_idx] = merged_k
        past_key_value.value_cache[self.layer_idx] = merged_v

        self._prune_hh_score(keep_idx)

        return past_key_value


    def _handle_d2o_cache(
            self,
            attn_weights,
            past_key_value,
            q_len,
            use_cache,
    ):

        if not use_cache or past_key_value is None:
            return

        if q_len != 1:
            self._store_prefill_scores(attn_weights)

            attn_sum = attn_weights.detach().clone().sum(dim = -2)
            attn_variance = attn_sum.var(dim = -1).mean(dim = 0)
            attn_variance_min = attn_variance.min()
            attn_variance_max = attn_variance.max()

            normalized_variance = (
                attn_variance - attn_variance_min
            ) / ( attn_variance_max - attn_variance_min + 1e-6 )

            self.multimodal_entropy = normalized_variance.mean(dim = 0)
            return

        if self.layer_idx in (0, 1, self.layer_nums - 1):
            return

        if self.hh_score is None and self.prefill_score is not None:
            scores = self._build_first_decode_scores(attn_weights)

        else:
            scores = self._attention_to_kv_scores(attn_weights)


        self._compress_dynamic_cache(past_key_value, scores)

        
    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        **kwargs,
    ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1,2)

        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1,2)

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1,2)

        kv_seq_len = key_states.shape[-2]

        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len,
                self.layer_idx,
            )

        rotary_seq_len = kv_seq_len

        if position_ids is not None:
            rotary_seq_len = max(
                rotary_seq_len,
                int(position_ids.max().item()) + 1,
            )

        cos, sin = self.rotary_emb(value_states, seq_len = rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
            }

            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(
            key_states,
            self.num_key_value_groups,
        )

        value_states = repeat_kv(
            value_states,
            self.num_key_value_groups,
        )

        attn_weights = (
            torch.matmul(
                query_states,
                key_states.transpose(2,3),
            ) / math.sqrt(self.head_dim)
        )

        if attention_mask is not None:
            required_len = attn_weights.shape[-1]

            if attention_mask.shape[-1] > required_len:
                attention_mask = attention_mask[..., -required_len:]

            if attention_mask.shape[-1] != required_len:
                raise ValueError(
                    "Attention mask - cache length mismatch:"
                    f"Mask = {attention_mask.shape[-1]}, "
                    f"Cache = {required_len}"
                )

            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(
            attn_weights,
            dim = -1,
            dtype = torch.float32,
        ).to(query_states.dtype)

        attn_weights = F.dropout(
            attn_weights,
            p = self.attention_dropout,
            training = self.training,
        )

        self._handle_d2o_cache(
            attn_weights,
            past_key_value,
            q_len,
            use_cache,
        )

        attn_output = torch.matmul(
            attn_weights,
            value_states,
        )

        attn_output = (
            attn_output
            .transpose(1,2)
            .contiguous()
            .reshape(bsz, q_len, self.hidden_size)
        )

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Qwen2ForCausalLM_D2O(Qwen2ForCausalLM):

    def __init__(self, config: Qwen2Config):
        super().__init__(config)

        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = Qwen2AttentionD2O(
                config=config,
                layer_idx=layer_idx,
            )


    def _allocate_dynamic_layer_ratios(self):
    
            attention_layers = [
                layer.self_attn for layer in self.model.layers
            ]
    
            if attention_layers[0].prefill_score is None:
                return
    
            if attention_layers[0].layer_hh_ratio != 0:
                return
    
            rho = (attention_layers[0].uniform_hh_ratio + attention_layers[0].uniform_recent_ratio)
            if rho <= 0:
                return
    
            layer_entropies = torch.stack(
                [
                    attn.multimodal_entropy
                    for attn in attention_layers
                ]
            )
    
            weights = torch.softmax(-layer_entropies, dim = 0)
    
            dynamic_layer_ratios = (weights * len(attention_layers) * rho)
    
            dynamic_layer_ratios = torch.clamp(
                dynamic_layer_ratios,
                min = 0.01,
                max = 1.0,
            )
    
            hh_split_ratio = (attention_layers[0].uniform_hh_ratio / rho)
            recent_split_ratio = (attention_layers[0].uniform_recent_ratio / rho)
    
            for idx, attn in enumerate(attention_layers):
                layer_ratio = dynamic_layer_ratios[idx].item()
    
                attn.layer_hh_ratio = (layer_ratio * hh_split_ratio)
                attn.layer_recent_ratio = (layer_ratio * recent_split_ratio)

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            past_key_values = None,
            inputs_embeds = None,
            labels = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
    ):

        if input_ids is not None:
            current_q_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            current_q_len = inputs_embeds.shape[1]
        else:
            current_q_len = None

        if current_q_len == 1:
            self._allocate_dynamic_layer_ratios()

        
        if (
            isinstance(past_key_values, Cache) and attention_mask is not None and position_ids is not None
        ):
            if input_ids is not None:
                q_len = input_ids.shape[1]
            else:
                q_len = inputs_embeds.shape[1]

            physical_cache_len = past_key_values.get_seq_length()
            required_mask_len = physical_cache_len + q_len

            if attention_mask.shape[-1] > required_mask_len:
                attention_mask = attention_mask[:, -required_mask_len:]

        return super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )


    def _clean_cache(self):

        for layer in self.model.layers:
            layer.self_attn._clean_cache()
