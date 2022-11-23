import os, sys, io
import pandas as pd
import spacy
import re
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import transforms

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
        
        ## REMOVE USELESS WORDS AND CHARACTERS
        #cleaned_text = ''.join(e for e in text if e.isalnum())
        cleaned_text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        
        tokenized_text = [tok.text.lower() for tok in spacy_eng.tokenizer(cleaned_text)]
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
                    
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

# Flickr8k Dataset
class Flickr8k(Dataset):
    def __init__(self, imgs_dir, captions_file, test_file, transform=None, freq_threshold=5):
        self.imgs_dir = imgs_dir
        self.df = pd.read_csv(captions_file)
        self.test_file = test_file
        self.transform = transform
        
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        
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
        self.captions = self.df["general_impression"]

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

class AVA(Dataset):
    def __init__(self, imgs_dir, captions_file, transform=None, freq_threshold=5):
        self.imgs_dir = imgs_dir
        self.captions_file = captions_file
        self.transform = transform
        self.freq_threshold = freq_threshold
        
        # Open dataframe
        with io.open(self.captions_file, 'r', encoding='utf-8') as f:
            json_file = json.load(f)
        self.df = pd.DataFrame(json_file['images'])
        
        #Get img, caption columns
        self.imgs = []
        self.captions = []
        for i, img in enumerate(self.df['images']):
            for j, caption in enumerate(self.df['sentences'][i]):
                self.imgs.append(img)
                self.captions.append(caption['raw'])
        
        # Initialize and build vocab
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
        numericalized_caption.append(self.vocab.stoi("<EOS>"))
        
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
        targets = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)
        
        lengths = [len(cap) for cap in captions]

        # imgs:    (batch size, 3, 224, 224)
        # targets: (sequence length, batch size)
        # lengths: (batch size)
        return imgs, targets, lengths


# def get_loader()
def get_loader(dataset_to_use, imgs_folder, annotation_file, transform, test_file="", batch_size=32, num_workers=8, freq_threshold=5, shuffle=True, pin_memory=True):
    
    if dataset_to_use == "PCCD":
        dataset = PCCD(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "flickr8k":
        dataset = Flickr8k(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "AVA":
        dataset = AVA(imgs_folder, annotation_file, transform=transform, freq_threshold=freq_threshold)
    
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
    
    imgs_dir = "datasets/PCCD/images/full"
    captions_file = "datasets/PCCD/raw.json"

    loader, dataset = get_loader(imgs_dir, captions_file, freq_threshold=5, transform=transform)
    
    # for i in range(50):
    #     print(dataset.vocab.itos[i])
    #     print(dataset.vocab.itos[len(dataset.vocab)-1-i])
    
    print(dataset.imgs)
    
    '''
    for idx, (imgs, captions, lengths) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
        
        if idx >5:
            sys.exit()
    '''