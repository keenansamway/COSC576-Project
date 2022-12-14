import os, sys
import torch
import torch.nn as nn

import torch.optim as optim
import torchvision.transforms as transforms
from utils import print_examples, save_checkpoint, load_checkpoint
from get_loader import get_loader, Flickr8k, Flickr30k, PCCD, AVA
from model import CNNtoLSTM, DecoderLSTM

 
def test(path):
    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 1e-3
    dropout = 0.2
    
    #dataset_to_use = "PCCD"
    dataset_to_use = "flickr8k"
    #dataset_to_use = "flickr30k"
    #dataset_to_use = "AVA"
    
    if dataset_to_use == "PCCD":
        imgs_folder = "datasets/PCCD/images/full"
        annotation_file = "datasets/PCCD/raw.json"
        test_file = "datasets/PCCD/images/PCCD_test.txt"
        
    elif dataset_to_use == "flickr8k":
        imgs_folder = "datasets/flickr8k/test_examples"
        annotation_file = "datasets/flickr8k/captions_train.feather"
        test_file = "datasets/flickr8k/flickr8k_test.txt"
    
    elif dataset_to_use == "flickr30k":
        imgs_folder = "datasets/flickr30k/test_examples"
        annotation_file = "datasets/flickr30k/captions.txt"
        test_file = "datasets/flickr30k/flickr30k_test.txt"
    
    elif dataset_to_use == "AVA":
        imgs_folder = "datasets/AVA/images"
        annotation_file = "datasets/AVA/AVA_sample_10percent.feather"
        test_file = "datasets/AVA/AVA_test.txt"
    
    transform = transforms.Compose(
        [
            transforms.Resize((232,232)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ]
    )

    freq_threshold = 5
    if dataset_to_use == "PCCD":
        dataset = PCCD(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "flickr8k":
        dataset = Flickr8k(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "flickr30k":
        dataset = Flickr30k(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "AVA":
        dataset = AVA(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    
    vocab_size = len(dataset.vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Nvidia CUDA Acceleration or Apple M1 Metal Acceleration or CPU 

    # initialize model, loss, etc
    model = CNNtoLSTM(
        embed_size=embed_size, 
        hidden_size=hidden_size, 
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout,
        ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    load_checkpoint(torch.load(path), model, optimizer)
    
    ## Generate text from images
    print_examples(model, device, dataset)
    
    ## Generate text from random initialization
    #start_token = torch.tensor(dataset.vocab.stoi["<SOS>"]).to(device)
    #hiddens = torch.randn(1, embed_size).to(device)
    #outputs = model.decoder.generate_text(start_token, hiddens, dataset.vocab)
    #print(outputs)

if __name__ == "__main__":
    path = f"CNN-LSTM/runs/checkpoint{10}.pth.tar"

    test(path)
