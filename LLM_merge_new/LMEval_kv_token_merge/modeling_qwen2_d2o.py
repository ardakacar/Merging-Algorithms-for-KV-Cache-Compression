from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
)
from transformers.cache_utils import Cache
import torch


class Qwen2AttentionD2O(Qwen2Attention):

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.layer_nums = config.num_hidden_layers
        self.hh_ratio = config.hh_ratio
        self.recent_ratio = config.recent_ratio
        self.hh_ratio = config.hh_ratio
        self.recent_size = config.recent_size
        self.alpha = config.alpha
        self.belta = config.belta

        self.hh_score = None
        self.threshold = None
        self.prefill_score = None
        self.prefill_size = 0
        self.cache_size = None

    def _clean_cache(self):
        self.hh_score = None
        self.threshold = None
        self.prefill_score = None
        self.prefill_size = 0
        self.cache_size = None

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
            self.hh_size = int(seq_len * self.hh_ratio)
            self.recent_size = int(seq_len * self.recent_ratio)
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
            input = kept_keys,
            dim = 2,
            index = merged_indices,
            src = merge_weights * keys_to_merge,
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


class Qwen2ForCausalLM_D2O(Qwen2ForCausalLM):

    def __init__(self, config: Qwen2Config):
        super().__init__(config)

        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = Qwen2AttentionD2O(
                config=config,
                layer_idx=layer_idx,
            )

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
