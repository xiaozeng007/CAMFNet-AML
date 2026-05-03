import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertModel, DistilBertTokenizer),
}

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased', freeze_layers=0):
        super().__init__()

        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune
        self.transformer_type = transformers
        self.freeze_layers = max(0, freeze_layers)
        if self.freeze_layers > 0:
            self._apply_freeze(self.freeze_layers)

    def _get_encoder_layers(self):
        encoder = getattr(self.model, 'encoder', None)
        if encoder is None:
            encoder = getattr(self.model, 'transformer', None)
        layers = None
        if encoder is not None:
            layers = getattr(encoder, 'layer', None)
            if layers is None:
                layers = getattr(encoder, 'layers', None)
        return layers

    def _apply_freeze(self, freeze_layers: int):
        self.set_freeze_layers(freeze_layers)

    def set_freeze_layers(self, freeze_layers: int):
        """
        Freeze the bottom `freeze_layers` transformer layers (0 = no freeze).
        Can be called dynamically for progressive unfreezing.
        """
        layers = self._get_encoder_layers()
        if layers is None:
            return
        freeze_layers = max(0, min(freeze_layers, len(layers)))
        for idx, layer in enumerate(layers):
            requires_grad = idx >= freeze_layers
            for param in layer.parameters():
                param.requires_grad = requires_grad
        self.freeze_layers = freeze_layers

    def unfreeze_additional_layers(self, num_layers: int):
        """
        Reduce the number of frozen layers by `num_layers`.
        """
        target = max(0, self.freeze_layers - max(0, num_layers))
        self.set_freeze_layers(target)

    def get_tokenizer(self):
        return self.tokenizer
    
    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            if self.transformer_type == 'distilbert':
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask)[0]
            else:
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]
        else:
            with torch.no_grad():
                if self.transformer_type == 'distilbert':
                    last_hidden_states = self.model(input_ids=input_ids,
                                                    attention_mask=input_mask)[0]
                else:
                    last_hidden_states = self.model(input_ids=input_ids,
                                                    attention_mask=input_mask,
                                                    token_type_ids=segment_ids)[0]
        return last_hidden_states
