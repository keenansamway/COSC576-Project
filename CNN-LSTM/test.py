import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import print_examples, save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import CNNtoLSTM


def test():
    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-4
    batch_size = 32
    num_workers = 2
    dropout = 0.5
        
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )
    
    path = "CNN-LSTM/runs/checkpoint.pth.tar"

    
    _, dataset = get_loader(
        imgs_folder="datasets/PCCD/images/full",
        annotation_file="datasets/PCCD/raw.json",
        test_file="datasets/PCCD/images/PCCD_test.txt",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    vocab_size = len(dataset.vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           ## Nvidia CUDA Acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    ## Apple M1 Metal Acceleration

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
    
    print_examples(model, device, dataset)

if __name__ == "__main__":
    test()