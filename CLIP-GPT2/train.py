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

from model import PositionalEncoding, MultimodalFusionModel
from mycollator import MyCollator


logging.set_verbosity_error()
set_seed(1234)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# if device.type == "cuda":
#     print(torch.cuda.get_device_name(0))


def showExample(id=None):
    data = dataset["test"]
    
    if id is None:
        id = torch.randint(len(data['image_id']), size=(1,)).item()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = Image.open(os.path.join("datasets/AVA/images", data[id]['image_id']))
    print("Caption:", data[id]['caption'])
    print("Image ID:", data[id]['image_id'])
    #display(img)

# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def createCollatorAndModel(text="gpt2", image="openai/clip-vit-base-patch32"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    processor = CLIPProcessor.from_pretrained(image)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    mycollator = MyCollator(tokenizer=tokenizer, processor=processor)
    model = MultimodalFusionModel(text, image, tokenizer.vocab_size)
    
    return mycollator, model.to(DEVICE)


class MyTrainer(Trainer):
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        # Ignore _remove_unused_columns
        return dataset



if __name__=="__main__":
    
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "CLIP-GPT2/data/clip-gpt2/train.csv",
            "validate": "CLIP-GPT2/data/clip-gpt2/validate.csv",
            "test": "CLIP-GPT2/data/clip-gpt2/test.csv",
        }
    )
    
    text_model_type = 'gpt2'
    #text_model_type = 'CLIP-GPT2/models/gpt2-small-AVA/checkpoint-20500'
    image_model_type = 'openai/clip-vit-base-patch16'

    mycollator, model = createCollatorAndModel(text=text_model_type, image=image_model_type)
    
    # Freeze / Unfreeze GPT2
    for param in model.text_encoder.parameters():
            param.requires_grad = False
    
    # Freeze / Unfreeze CLIP
    for param in model.image_encoder.parameters():
            param.requires_grad = True
            
    # Freeze / Unfreeze GPT2 Linear Projection
    model.project_textencoder[0].weight.requires_grad = False
    
    
    args = TrainingArguments(
        output_dir="CLIP-GPT2/models/clip-gpt2/b16-small",
        seed=42,
        evaluation_strategy="steps",
        eval_steps=5000,
        eval_accumulation_steps=20,
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=5,
        metric_for_best_model="eval_loss",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        fp16=True,
        fp16_opt_level="O1",
        # warmup_ratio=0.01,
        # leraning_rate=5e-4,
        # weight_decay=1e-4,
        # gradient_accumulation_steps=2,
        dataloader_num_workers=8,
        #load_best_model_at_end=True,
        disable_tqdm=False,
        dataloader_pin_memory=True,
        ignore_data_skip=False,
    )
    
    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validate'], 
        data_collator=mycollator,
    )
    
    trainer.train()
    #trainer.train(resume_from_checkpoint="CLIP-GPT2/models/clip-gpt2/b16-small/checkpoint-10000")
    
