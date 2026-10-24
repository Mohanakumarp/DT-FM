"""Two/three-stage mT5 split, with one owner for its shared input embedding."""
from torch import nn
from torch.nn import functional as F


class MT5EncoderStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.encoder = model.encoder

    def forward(self, input_ids, attention_mask, labels):
        decoder_ids = labels.new_full(labels.shape, self.config.pad_token_id)
        decoder_ids[:, 0] = self.config.decoder_start_token_id
        decoder_ids[:, 1:] = labels[:, :-1]
        decoder_ids.masked_fill_(decoder_ids == -100, self.config.pad_token_id)
        encoded = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        # Both encoder and decoder embedding gradients accumulate on this rank.
        return encoded, self.encoder.embed_tokens(decoder_ids)

    def pretrained_state_dict(self):
        state = {"encoder." + k: v for k, v in self.encoder.state_dict().items()}
        state["shared.weight"] = state["encoder.embed_tokens.weight"]
        state["decoder.embed_tokens.weight"] = state["shared.weight"]
        return state


class MT5DecoderStage(nn.Module):
    def __init__(self, model, with_head=True):
        super().__init__()
        if model.config.tie_word_embeddings:
            raise ValueError("This split requires untied output weights, as in google/mt5-small")
        self.config = model.config
        self.decoder = model.decoder
        self.decoder.embed_tokens = None
        self.lm_head = model.lm_head if with_head else None

    def forward(self, encoded, decoder_embeds, attention_mask, labels):
        hidden = self.decoder(
            inputs_embeds=decoder_embeds,
            encoder_hidden_states=encoded,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        if self.lm_head is None:
            return hidden
        logits = self.lm_head(hidden)
        return F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )

    def pretrained_state_dict(self):
        state = {"decoder." + k: v for k, v in self.decoder.state_dict().items()}
        if self.lm_head is not None:
            state.update({"lm_head." + k: v for k, v in self.lm_head.state_dict().items()})
        return state


class MT5HeadStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.config.tie_word_embeddings:
            raise ValueError("This split requires untied output weights")
        self.config = model.config
        self.lm_head = model.lm_head

    def forward(self, hidden, labels):
        logits = self.lm_head(hidden)
        return F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )

    def pretrained_state_dict(self):
        return {"lm_head." + k: v for k, v in self.lm_head.state_dict().items()}
