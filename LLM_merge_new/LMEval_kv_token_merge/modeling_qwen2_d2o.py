from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
)


class Qwen2AttentionD2O(Qwen2Attention):

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx


class Qwen2ForCausalLM_D2O(Qwen2ForCausalLM):

    def __init__(self, config: Qwen2Config):
        super().__init__(config)

        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = Qwen2AttentionD2O(
                config=config,
                layer_idx=layer_idx,
            )
