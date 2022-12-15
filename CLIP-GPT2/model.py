import os
import warnings
import math
import pandas as pd
from prettytable import PrettyTable
from PIL import Image
from tqdm import tqdm
import datasets
from dataclasses import dataclass
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2Model, GPT2Tokenizer,
    
    CLIPVisionModel, CLIPProcessor,
    
    TrainingArguments, Trainer,
    
    logging, set_seed
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class MultimodalFusionModel(nn.Module):
    def __init__(self, text_model, image_model, vocab_size, embed_dim=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.text_model = text_model
        self.image_model = image_model
        
        # Pretrained transformers for encoding text and image
        self.text_encoder = GPT2Model.from_pretrained(text_model)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        self.image_encoder = CLIPVisionModel.from_pretrained(image_model)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        text_hidden_size = self.text_encoder.config.hidden_size
        #image_hidden_size = self.image_encoder.config.hidden_size
        
        #self.project_image = nn.Linear(image_hidden_size, text_hidden_size)
        
        self.positional_encoding = PositionalEncoding(text_hidden_size, dropout, max_len=1024)
        
        self.fusiontransformer = nn.Transformer(
            d_model=text_hidden_size,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            batch_first=True,
        )
        
        self.project_textencoder = nn.Sequential(
            nn.Linear(text_hidden_size, self.vocab_size, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Initialize this linear layer to GPT2's embedding layer
        self.project_textencoder[0].weight.data = self.text_encoder.wte.weight.data
        self.project_textencoder[0].weight.requires_grad = False
        
        self.project_fusionmodel = nn.Sequential(
            nn.Linear(text_hidden_size, self.vocab_size, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        
        self.criterion = nn.CrossEntropyLoss()
    
    
    def forward(self, 
                pixel_values : torch.FloatTensor,                       # (batch_size, 3, image_size, image_size)
                input_ids : Optional[torch.LongTensor]=None,            # (batch_size, sequence_length)
                attention_mask : Optional[torch.LongTensor]=None,       # (batch_size, sequence_length)
                labels : Optional[torch.LongTensor]=None):              # (batch_size, sequence_length)
        
        # encoded_text['last_hidden_state']: (batch_size, text_sequence_length, hidden_size=768)
        # encoded_images['last_hidden_state']: (batch_size, image_sequence_length=50, hidden_size=768)
        encoded_text = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)            
        encoded_images = self.image_encoder(pixel_values, return_dict=True)
        
        # if self.text_encoder.config.hidden_size != self.image_encoder.config.hidden_size:
        #     encoded_images["last_hidden_state"] = self.project_image(encoded_images["last_hidden_state"])
        
        tgt_mask = self.get_tgt_mask(encoded_text["last_hidden_state"].shape[1])
        tgt_key_padding_mask = (1 - attention_mask) > 0
        
        fusion_src = self.positional_encoding(encoded_images["last_hidden_state"])
        fusion_tgt = self.positional_encoding(encoded_text["last_hidden_state"])
        
        # fused_output: (batch_size, text_sequence_length, hidden_size=768)
        fused_output = self.fusiontransformer(
            fusion_src,
            fusion_tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        
        # projected_text: (batch_size, text_sequence_length, vocab_size)
        # projected_fusion: (batch_size, text_sequence_length, vocab_size)
        projected_text = self.project_textencoder(encoded_text["last_hidden_state"])
        projected_fusion = self.project_fusionmodel(fused_output)
        
        # logits: (batch_size, text_sequence_length, vocab_size)
        logits = projected_text + projected_fusion
        
        out = {"logits": logits}
        
        loss = None
        if labels is not None:                
            # Shift so that tokens < n predicts n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten tokens and calculate loss
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            out["loss"] = loss
        
        return out


    # https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask= mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)