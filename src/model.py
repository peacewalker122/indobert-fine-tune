import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .config import IGNORE_INDEX, INTENTS, MODEL_NAME, SLOTS


class IntentSlotModel(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, len(INTENTS))
        self.slot_head = nn.Linear(hidden, len(SLOTS))

    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        intent_logits = self.intent_head(self.dropout(hidden_states[:, 0]))
        slot_logits = self.slot_head(self.dropout(hidden_states))

        loss = None
        if intent_labels is not None and slot_labels is not None:
            loss = F.cross_entropy(intent_logits, intent_labels) + F.cross_entropy(
                slot_logits.transpose(1, 2), slot_labels, ignore_index=IGNORE_INDEX
            )
        return {"loss": loss, "intent_logits": intent_logits, "slot_logits": slot_logits}
