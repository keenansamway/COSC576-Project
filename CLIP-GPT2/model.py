import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers import CLIPModel


class MultimodalFusionModel(nn.Module):
    def __init__(self, text_model, image_model, num_labels, embed_dim=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.text_model = text_model
        self.image_model = image_model
        
        # Pretrained transformers for encoding text and image
        self.text_encoder = GPT2LMHeadModel.from_pretrained(text_model)
        self.image_encoder = CLIPModel.from_pretrained(image_model)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(self.embed_dim, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self,
        captions,
        images,
        labels=None,
    ):
        encoded_text = self.text_encoder(captions, return_dict=True)
        
        encoded_images = self.image_encoder(images, return_dict=True)
        
        fused_output = self.fusion(
            torch.cat([encoded_text['logits']], [encoded_images['pooler_output']], dim=1)
        )
        
        logits = self.classifier(fused_output)
        
        out = {"logits": logits}
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out