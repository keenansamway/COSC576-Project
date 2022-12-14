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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# if device.type == "cuda":
#     print(torch.cuda.get_device_name(0))


dataset = load_dataset(
    "csv",
    data_files={
        "train": "CLIP-GPT2/data/clip-gpt2/train.csv",
        "validate": "CLIP-GPT2/data/clip-gpt2/validate.csv",
        "test": "CLIP-GPT2/data/clip-gpt2/test.csv",
    }
)

#dataset

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
                
        # fused_output: (batch_size, text_sequence_length, hidden_size=768)
        fused_output = self.fusiontransformer(
            self.positional_encoding(encoded_images["last_hidden_state"]),
            self.positional_encoding(encoded_text["last_hidden_state"]),
            tgt_mask=tgt_mask,
            #tgt_key_padding_mask=tgt_key_padding_mask,
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
        return mask.to(device)

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
    tokenizer.padding_side = "left"
    
    mycollator = MyCollator(tokenizer=tokenizer, processor=processor)
    model = MultimodalFusionModel(text, image, tokenizer.vocab_size)
    
    return mycollator, model.to(device)

text_model_type = 'gpt2'
#text_model_type = 'CLIP-GPT2/models/gpt2-small-AVA/checkpoint-20500'
image_model_type = 'openai/clip-vit-base-patch32'

logging.set_verbosity_error()
mycollator, model = createCollatorAndModel(text=text_model_type, image=image_model_type)

args = TrainingArguments(
    output_dir="CLIP-GPT2/models/clip-gpt2/b32-small",
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
    num_train_epochs=4,
    fp16=True,
    fp16_opt_level="O1",
    # warmup_ratio=0.01,
    # leraning_rate=5e-4,
    # weight_decay=1e-4,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=12,
    load_best_model_at_end=True,
    disable_tqdm=False,
    dataloader_pin_memory=True,
    ignore_data_skip=True,
)
set_seed(42)

class MyTrainer(Trainer):
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        # Ignore _remove_unused_columns
        return dataset
    
    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    # ):
        
        

trainer = MyTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validate'],
    data_collator=mycollator,
)

def train():
    #trainer.train()
    trainer.train(resume_from_checkpoint="CLIP-GPT2/models/clip-gpt2/b32-small/checkpoint-120000")

# https://colab.research.google.com/github/sachinruk/blog/blob/master/_notebooks/2021-12-28-vit-to-gpt2-encoder-decoder-model.ipynb#scrollTo=806799ae
def top_k_top_p_filtering(
    next_token_logits: torch.FloatTensor,
    top_k: Optional[float]=None, 
    top_p: Optional[float]=None,
    device: Union[str, torch.device]="cpu",
    ) -> torch.FloatTensor:
    if top_k is None:
        top_k = next_token_logits.shape[-1]
    if top_p is None:
        top_p = 1.0
        
    p, largest_p_idx = F.softmax(next_token_logits, dim=-1).topk(top_k, dim=-1)
    cumulative_p = p.cumsum(dim=-1)
    threshold_repeated = top_p + torch.zeros((len(p),1)).to(device)
    idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k-1).squeeze()
    cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
    censored_p = (cumulative_p <= cutoffs[:, None]) * p
    renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)
    
    final_p = torch.zeros_like(next_token_logits)
    row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1,top_k).to(device)
    final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

    return final_p

TOP_K = 50
TOP_P = 0.95
def generate_sentence_from_image(model, pixel_values, tokenizer, max_text_length: int):
    generated_so_far = torch.LongTensor([[tokenizer.bos_token_id]] * len(pixel_values)).to(device)
    
    with torch.no_grad():
        for _ in tqdm(range(max_text_length)):
            attention_mask = torch.ones_like(generated_so_far)
            
            outputs = model(
                input_ids=generated_so_far,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
            )
            
            next_token_logits = outputs['logits'][:, -1, :]
            filtered_p = top_k_top_p_filtering(next_token_logits, top_k=TOP_K, top_p=TOP_P, device=device)
            next_token = torch.multinomial(filtered_p, num_samples=1)
            generated_so_far = torch.cat([generated_so_far, next_token], dim=1)
    
    return [tokenizer.decode(coded_sentence) for coded_sentence in generated_so_far]

def evaluate():
    text_model_type = 'CLIP-GPT2/models/gpt2-small-AVA/checkpoint-20500'
    image_model_type = 'openai/clip-vit-base-patch32'
    
    logging.set_verbosity_error()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = MultimodalFusionModel(text_model_type, image_model_type, tokenizer.vocab_size)
    model.load_state_dict(torch.load('CLIP-GPT2/models/clip-gpt2/b32-small/checkpoint-35000/pytorch_model.bin'))
    model.to(device)
    print("--loaded--")
    
    sample = mycollator(dataset['test'][2000:2005])
    pixel_values = sample['pixel_values'].to(device)
    
    model.eval()
    outputs = generate_sentence_from_image(model, pixel_values, tokenizer, max_text_length=20)
    
    for i in range(2000, 2005):
        print("-------------------------")
        print("Prediction:", outputs[i-2000])
        showExample(i)
        print("-------------------------")

if __name__=="__main__":
    
    train0_evaluate1 = 0
    
    if train0_evaluate1 == 0: train()
    elif train0_evaluate1 == 1: evaluate()
    
    
    
