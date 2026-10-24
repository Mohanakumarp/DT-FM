"""Contiguous mT5 partitions, preserving shared embeddings and attention biases."""
import torch
from torch import nn
from torch.nn import functional as F


def pipeline_layout(config, world_size, pipeline_size=None):
    # Input embeddings, encoder blocks, decoder blocks, output head.
    units = config.num_layers + config.num_decoder_layers + 2
    if world_size < 1:
        raise ValueError("world-size must be positive")
    if pipeline_size is None:
        pipeline_size = max(size for size in range(1, min(units, world_size) + 1)
                            if world_size % size == 0)
    if pipeline_size < 1 or pipeline_size > units or world_size % pipeline_size:
        raise ValueError("pipeline-size must divide world-size and be between 1 and %d" % units)
    # Every stage owns at least one computation unit; no idle relay stages.
    boundaries = [units * rank // pipeline_size for rank in range(pipeline_size + 1)]
    return pipeline_size, world_size // pipeline_size, boundaries


def activation_shapes(config, last_unit, batch_size, source_length, target_length):
    encoder_end = config.num_layers
    decoder_end = encoder_end + config.num_decoder_layers
    shapes = {}
    if last_unit < decoder_end:
        shapes["encoded"] = (batch_size, source_length, config.d_model)
    shapes["decoded"] = (batch_size, target_length, config.d_model)
    if 0 < last_unit < encoder_end:
        shapes["encoder_bias"] = (batch_size, config.num_heads, source_length, source_length)
    if encoder_end < last_unit < decoder_end:
        shapes["decoder_bias"] = (batch_size, config.num_heads, target_length, target_length)
    return shapes


class MT5Partition(nn.Module):
    def __init__(self, model, pipeline_rank, boundaries):
        super().__init__()
        if model.config.tie_word_embeddings:
            raise ValueError("The mT5 pipeline requires untied output weights")
        self.config = model.config
        self.pipeline_rank = pipeline_rank
        self.boundaries = list(boundaries)
        self.start, self.end = boundaries[pipeline_rank:pipeline_rank + 2]
        encoder_end = self.config.num_layers
        decoder_end = encoder_end + self.config.num_decoder_layers
        self.shared = model.shared if self.start == 0 else None
        self.encoder_blocks = nn.ModuleDict({
            str(index): model.encoder.block[index]
            for index in range(self.config.num_layers)
            if self.start <= index + 1 < self.end
        })
        self.decoder_blocks = nn.ModuleDict({
            str(index): model.decoder.block[index]
            for index in range(self.config.num_decoder_layers)
            if self.start <= encoder_end + index + 1 < self.end
        })
        self.encoder_norm = model.encoder.final_layer_norm if self.start <= encoder_end < self.end else None
        self.decoder_norm = model.decoder.final_layer_norm if self.start <= decoder_end < self.end else None
        self.lm_head = model.lm_head if self.start <= decoder_end + 1 < self.end else None
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def forward(self, state, attention_mask, labels, input_ids=None):
        state = dict(state)
        source_length = attention_mask.shape[1]
        target_length = labels.shape[1]
        encoder_end = self.config.num_layers
        decoder_end = encoder_end + self.config.num_decoder_layers
        dtype = next(self.parameters()).dtype
        source_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        # Same causal mask as uncached, unmasked mT5 teacher-forced decoding.
        causal_mask = torch.full((target_length, target_length), torch.finfo(dtype).min,
                                 device=labels.device, dtype=dtype).triu(1)
        causal_mask = causal_mask[None, None].expand(len(labels), 1, -1, -1)
        for unit in range(self.start, self.end):
            if unit == 0:
                decoder_ids = labels.new_full(labels.shape, self.config.pad_token_id)
                decoder_ids[:, 0] = self.config.decoder_start_token_id
                decoder_ids[:, 1:] = labels[:, :-1]
                decoder_ids.masked_fill_(decoder_ids == -100, self.config.pad_token_id)
                state["encoded"] = self.dropout(self.shared(input_ids))
                state["decoded"] = self.shared(decoder_ids)
            elif unit <= encoder_end:
                outputs = self.encoder_blocks[str(unit - 1)](
                    state["encoded"], attention_mask=source_mask,
                    position_bias=state.get("encoder_bias"), use_cache=False,
                    cache_position=torch.arange(source_length, device=labels.device),
                )
                state["encoded"], state["encoder_bias"] = outputs[:2]
                if unit == encoder_end:
                    state["encoded"] = self.dropout(self.encoder_norm(state["encoded"]))
                    del state["encoder_bias"]
            elif unit <= decoder_end:
                index = unit - encoder_end - 1
                if index == 0:
                    state["decoded"] = self.dropout(state["decoded"])
                outputs = self.decoder_blocks[str(index)](
                    state["decoded"], attention_mask=causal_mask,
                    position_bias=state.get("decoder_bias"),
                    encoder_hidden_states=state["encoded"], encoder_attention_mask=source_mask,
                    # Cross-attention has no learned relative bias. Recomputing
                    # its constant padding bias avoids transmitting another tensor.
                    encoder_decoder_position_bias=None, use_cache=False,
                    cache_position=torch.arange(target_length, device=labels.device),
                )
                state["decoded"], state["decoder_bias"] = outputs[:2]
                if unit == decoder_end:
                    state["decoded"] = self.dropout(self.decoder_norm(state["decoded"]))
                    del state["decoder_bias"]
                    del state["encoded"]
            else:
                logits = self.lm_head(state["decoded"])
                return F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                       ignore_index=-100, reduction="sum")
        return state

    def pretrained_state_dict(self):
        state = {}
        if self.shared is not None:
            state.update({name: self.shared.weight for name in
                          ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight")})
        for prefix, blocks in (("encoder", self.encoder_blocks), ("decoder", self.decoder_blocks)):
            for index, block in blocks.items():
                state.update({"%s.block.%s.%s" % (prefix, index, key): value
                              for key, value in block.state_dict().items()})
        for prefix, norm in (("encoder", self.encoder_norm), ("decoder", self.decoder_norm)):
            if norm is not None:
                state.update({prefix + ".final_layer_norm." + key: value
                              for key, value in norm.state_dict().items()})
        if self.lm_head is not None:
            state.update({"lm_head." + key: value for key, value in self.lm_head.state_dict().items()})
        return state


class FullModelStage(nn.Module):
    """Architecture-independent full-model stage, replicated with data parallelism."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.start, self.end = 0, 1
        self.boundaries = [0, 1]

    def forward(self, state, attention_mask, labels, input_ids=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, use_cache=False)
        targets = labels if self.config.is_encoder_decoder else labels[:, 1:]
        return output.loss * (targets != -100).sum()

    def pretrained_state_dict(self):
        return self.model.state_dict()
