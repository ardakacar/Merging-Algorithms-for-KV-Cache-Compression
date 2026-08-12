from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
)
from transformers.cache_utils import Cache


class Qwen2AttentionD2O(Qwen2Attention):

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.layer_nums = config.num_hidden_layers
        self.hh_ratio = config.hh_ratio
        self.recent_ratio = config.recent_ratio
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
