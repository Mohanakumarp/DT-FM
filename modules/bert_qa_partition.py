"""Contiguous BERT question-answering stages with single parameter ownership."""
import torch
from torch import nn
from torch.nn import functional as F


def qa_layout(config, world_size, pipeline_size):
    units = config.num_hidden_layers + 2  # embeddings, encoder layers, QA head
    if not 1 <= pipeline_size <= units or world_size % pipeline_size:
        raise ValueError("pipeline-size must divide world-size and fit the BERT computation units")
    return [units * rank // pipeline_size for rank in range(pipeline_size + 1)]


class BertQAPartition(nn.Module):
    def __init__(self, model, pipeline_rank, boundaries):
        super().__init__()
        if model.config.model_type != "bert" or model.config.is_decoder or model.config.num_labels != 2:
            raise ValueError("QA pipeline supports encoder-only BERT with a two-logit span head")
        self.config = model.config
        self.boundaries = list(boundaries)
        self.start, self.end = boundaries[pipeline_rank:pipeline_rank + 2]
        self.embeddings = model.bert.embeddings if self.start == 0 else None
        self.layers = nn.ModuleDict({
            str(i): layer for i, layer in enumerate(model.bert.encoder.layer)
            if self.start <= i + 1 < self.end
        })
        self.qa_outputs = model.qa_outputs if self.end == self.config.num_hidden_layers + 2 else None

    def forward(self, hidden, attention_mask, token_type_ids, input_ids=None):
        dtype = next(self.parameters()).dtype
        mask = (1 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        for unit in range(self.start, self.end):
            if unit == 0:
                hidden = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
            elif unit <= self.config.num_hidden_layers:
                hidden = self.layers[str(unit - 1)](hidden, attention_mask=mask)[0]
            else:
                hidden = self.qa_outputs(hidden)
        return hidden

    def pretrained_state_dict(self):
        state = {}
        if self.embeddings is not None:
            state.update({"bert.embeddings." + key: value for key, value in self.embeddings.state_dict().items()})
        for index, layer in self.layers.items():
            state.update({"bert.encoder.layer.%s.%s" % (index, key): value
                          for key, value in layer.state_dict().items()})
        if self.qa_outputs is not None:
            state.update({"qa_outputs." + key: value for key, value in self.qa_outputs.state_dict().items()})
        return state

    def load_pretrained_state_dict(self, state):
        local = self.pretrained_state_dict()
        if local.keys() != state.keys():
            raise ValueError("Checkpoint parameters do not match this BERT partition")
        with torch.no_grad():
            for name, value in local.items():
                if value.shape != state[name].shape:
                    raise ValueError("Checkpoint parameter shape differs: " + name)
                value.copy_(state[name])


def span_loss(logits, starts, ends):
    # Mean of start/end cross-entropies, summed over examples. The caller
    # divides by the number of features across all data replicas exactly once.
    return (F.cross_entropy(logits[:, :, 0], starts, reduction="sum") +
            F.cross_entropy(logits[:, :, 1], ends, reduction="sum")) / 2
