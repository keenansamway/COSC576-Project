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
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text["input_ids"].squeeze(),
            "attention_mask": encoded_text["attention_mask"].squeeze(),
        }
        
    def tokenize_labels(self, labels):
        eos = self.tokenizer.eos_token
        # https://github.com/huggingface/transformers/issues/2001
        labels = [eos + x + eos for x in labels]
        
        encoded_labels = self.tokenizer(
            labels,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # https://github.com/huggingface/transformers/issues/7135#issuecomment-1172962080
        inv_bool_attention_mask = (1 - encoded_labels["attention_mask"]) > 0
        encoded_labels["input_ids"][inv_bool_attention_mask] = -100
        
        return {
            "labels": encoded_labels["input_ids"].squeeze(),
        }
        
    def process_image(self, images):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processed_images = self.processor(
                images=[Image.open(os.path.join("datasets/AVA/images", image_id)).convert('RGB') for image_id in images],
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