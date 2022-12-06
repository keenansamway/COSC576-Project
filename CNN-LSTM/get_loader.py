import os, sys, io
import pandas as pd
import spacy
import re, string
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import transforms
from utils import clean_text

# https://spacy.io/usage/linguistic-features#tokenization
# https://spacy.io/api/tokenizer
spacy_eng = spacy.load("en_core_web_sm")

# class Vocabulary:
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        
        # Tokenize text
        tokenized_text = [tok.text for tok in spacy_eng.tokenizer(text)]
        return tokenized_text
    
    # Create dictionary of vocabulary and frequency
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    # add word to frequencies dictionary, set frequency to 1 (initial word)
                    frequencies[word] = 1
                else:
                    # word is in frequencies dictionary, increment frequency by 1
                    frequencies[word] += 1
                    
                if frequencies[word] == self.freq_threshold:
                    # word has reached frequency threshold in vocab dictionary
                    # add word to vocab dictionary (at most once)
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    # Convert tokens to numericalized representations           
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

# Flickr8k Dataset
class Flickr8k(Dataset):
    def __init__(self, imgs_dir, captions_file, test_file, transform=None, freq_threshold=5):
        self.imgs_dir = imgs_dir
        self.df = pd.read_feather(captions_file)
        self.test_file = test_file
        self.transform = transform
        
        self.imgs = self.df['image']
        self.captions = self.df['caption'].apply(clean_text)
        
        self.df['length'] = self.df['caption'].apply(lambda row: len(row.strip().split()))
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.imgs_dir, img_id)).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
        
# Flickr30k Dataset
class Flickr30k(Dataset):
    def __init__(self, imgs_dir, captions_file, test_file, transform=None, freq_threshold=5):
        self.imgs_dir = imgs_dir
        self.df = pd.read_csv(captions_file, delimiter="|")
        self.test_file = test_file
        self.transform = transform
        
        self.imgs = self.df['image_name']
        self.captions = self.df['caption_text'].apply(clean_text)
        
        self.df['length'] = self.df['caption_text'].apply(lambda row: len(row.strip().split()))
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.imgs_dir, img_id)).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
        
# PCCD Dataset
class PCCD(Dataset):
    def __init__(self, imgs_dir, captions_file, test_file, transform=None, freq_threshold=5):
        self.imgs_dir = imgs_dir
        self.df = pd.read_json(captions_file)
        self.test_file = test_file
        self.transform = transform
        
        self.df["general_impression"] = self.df["general_impression"].fillna("")
        
        # Get img, caption columns
        self.imgs = self.df["title"]
        self.captions = self.df["general_impression"].apply(clean_text)

        # Initialize and build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.imgs_dir, img_id)).convert("RGB")
                
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
    
# AVA-Captions Dataset
class AVA(Dataset):
    def __init__(self, imgs_dir, captions_file, test_file, transform=None, freq_threshold=5):
        self.imgs_dir = imgs_dir
        self.captions_file = captions_file
        self.test_file = test_file
        self.transform = transform
        self.freq_threshold = freq_threshold
        
        # Open dataframe
        self.df = pd.read_feather(captions_file)
        
        self.imgs = self.df['filename']
        self.captions = self.df['clean_sentence'].apply(clean_text)
        self.split = self.df['split']
        
        # Initialize and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.imgs_dir, img_id)).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
    
# custom collate
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        # Sort batch list by caption length in descending order (longest to shortest)
        #batch.sort(key=lambda x: len(x[1]), reverse=True)
        
        ## TEST IF SORTING IS NEEDED OR NOT
        
        imgs = [item[0] for item in batch]
        imgs = torch.stack(imgs)
        
        captions = [item[1] for item in batch]
        lengths = [len(cap) for cap in captions]

        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        # imgs:    (batch size, 3, 224, 224)
        # captions: (sequence length, batch size)
        # lengths: (batch size)
        return imgs, captions, lengths


# def get_loader()
def get_loader(dataset_to_use, imgs_folder, annotation_file, transform, test_file="", batch_size=32, num_workers=8, freq_threshold=5, shuffle=True, pin_memory=True):
    
    if dataset_to_use == "PCCD":
        dataset = PCCD(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "flickr8k":
        dataset = Flickr8k(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "flickr30k":
        dataset = Flickr30k(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "AVA":
        dataset = AVA(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    return loader, dataset


# if __name__ == "__main__":
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    imgs_dir = "datasets/flickr8k/images/"
    captions_file = "datasets/flickr8k/captions.txt"

    loader, dataset = get_loader("flickr8k", imgs_dir, captions_file, freq_threshold=5, transform=transform)
    
    # for i in range(50):
    #     print(dataset.vocab.itos[i])
    #     print(dataset.vocab.itos[len(dataset.vocab)-1-i])
    
    captions = dataset.captions.tolist()
    
    word_freq = {}
    for caption in captions:
        caption = caption.strip()
        for word in caption.split():
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    
    
    print(dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:30]))
    