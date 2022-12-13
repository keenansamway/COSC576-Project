# %%
import os, sys
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from ast import literal_eval
import datasets
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    GPT2Model, GPT2Tokenizer,
    
    CLIPVisionModel, CLIPProcessor,
    
    TrainingArguments, Trainer,
    
    logging, set_seed
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))


dataset = load_dataset(
    "csv",
    data_files={
        "train": "CLIP-GPT2/data/clip-gpt2/train_nolabel.csv",
        "test": "CLIP-GPT2/data/clip-gpt2/test_nolabel.csv"
    }
)
'''
dataset = dataset.map(
    lambda examples: {
        'label' : [
            literal_eval(l)
            for l in examples['label']
        ]
    }, batched=True
)
'''
#dataset

def showExample(id=None):
    data = dataset["test"]
    
    if id is None:
        id = torch.randint(len(data['image_id']), size=(1,)).item()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = Image.open(os.path.join("datasets/AVA/images", data[id]['image_id']))
    print("Caption:", data[id]['caption'])
    #print("Label:", data[id]['label'])
    #display(img)

#showExample()


@dataclass
class MyCollator:
    tokenizer: GPT2Tokenizer
    processor: CLIPProcessor
        
    def tokenize_text(self, texts):
        eos = self.tokenizer.eos_token
        texts = [eos + x + eos for x in texts]
        
        encoded_text = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text["input_ids"].squeeze(),
            "attention_mask": encoded_text["attention_mask"].squeeze(),
        }
        
    def tokenize_labels(self, labels):
        eos = self.tokenizer.eos_token
        labels = [eos + x + eos for x in labels]
        
        encoded_labels = self.tokenizer(
            labels,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        return {
            "labels": encoded_labels["input_ids"].squeeze(),
        }
        
    def process_image(self, images):
        processed_images = self.processor(
            images=[Image.open(os.path.join("..", "datasets/AVA/images", image_id)).convert('RGB') for image_id in images],
            return_tensors="pt",
            )
        return {
            "pixel_values": processed_images["pixel_values"].squeeze(),
        }
    
    def __call__(self, raw_batch_dict):        
        return {
            **self.tokenize_text(
                raw_batch_dict['caption']
                if isinstance(raw_batch_dict, dict) else
                [i['caption'] for i in raw_batch_dict]
            ),
            **self.process_image(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            **self.tokenize_labels(
                raw_batch_dict['caption']
                if isinstance(raw_batch_dict, dict) else
                [i['caption'] for i in raw_batch_dict]
            ),
        }


class MultimodalFusionModel(nn.Module):
    def __init__(self, text_model, image_model, vocab_size, embed_dim=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.text_model = text_model
        self.image_model = image_model
        
        # Pretrained transformers for encoding text and image
        with torch.no_grad():
            self.text_encoder = GPT2Model.from_pretrained(text_model)
            self.image_encoder = CLIPVisionModel.from_pretrained(image_model)
        
        num_features = self.text_encoder.config.hidden_size
        
        self.fusiontransformer = nn.Transformer(
            d_model=num_features,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            batch_first=True,
        )
        
        self.ensemble = nn.Sequential(
            nn.Linear(num_features + num_features, self.vocab_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, 
                pixel_values : torch.FloatTensor,           # (batch_size, 3, image_size, image_size)
                input_ids : torch.LongTensor=None,          # (batch_size, sequence_length)
                labels : torch.LongTensor=None,             # (batch_size, sequence_length)
                attention_mask : torch.LongTensor=None):
        
        if input_ids is None:
            # Set initial input to SOS token (50256 <==> "<|endoftext|>")
            input_ids = torch.tensor([[50256]])
            
        # encoded_text['last_hidden_state']: (batch_size, text_sequence_length, hidden_size=768)
        # encoded_images['last_hidden_state']: (batch_size, image_sequence_length=50, hidden_size=768)
        encoded_text = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)            
        encoded_images = self.image_encoder(pixel_values, return_dict=True)
        
        # fused_output: (batch_size, text_sequence_length, hidden_size=768)
        fused_output = self.fusiontransformer(
            encoded_images["last_hidden_state"],
            encoded_text["last_hidden_state"],
        )
        
        # logits: (batch_size, text_sequence_length, vocab_size)     
        logits = self.ensemble(
            torch.cat([encoded_text["last_hidden_state"], fused_output], dim=2)
            )
                
        out = {"logits": logits}
        
        if labels is not None:                
            # Shift so that tokens n-1 predicts n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten tokens and calculate loss
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            out["loss"] = loss
        
        return out

def createCollatorAndModel(text="gpt2", image="openai/clip-vit-base-patch32"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    processor = CLIPProcessor.from_pretrained(image)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    mycollator = MyCollator(tokenizer=tokenizer, processor=processor)
    model = MultimodalFusionModel(text, image, tokenizer.vocab_size)
    
    return mycollator, model.to(device)

text_model_type = 'CLIP-GPT2/models/gpt2-small-AVA/checkpoint-20500'
image_model_type = 'openai/clip-vit-base-patch32'

logging.set_verbosity_error()
mycollator, model = createCollatorAndModel(text=text_model_type, image=image_model_type)


args = TrainingArguments(
    output_dir="CLIP-GPT2/models/clip-gpt2/b32-small",
    seed=42,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    fp16=True,
    fp16_opt_level="O1",
    # warmup_ratio=0.01,
    # leraning_rate=5e-4,
    # weight_decay=1e-4,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    disable_tqdm=False,
    dataloader_pin_memory=True,
)
set_seed(42)

class MyTrainer(Trainer):
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        # Ignore _remove_unused_columns
        return dataset

trainer = MyTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=mycollator,
)


if __name__=="__main__":
    trainer.train()

