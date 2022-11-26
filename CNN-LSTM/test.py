import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import print_examples, save_checkpoint, load_checkpoint
from get_loader import get_loader, Flickr8k, PCCD, AVA
from model import CNNtoLSTM, DecoderLSTM


def test():
    # Hyperparameters
    embed_size = 512
    hidden_size = 512
    num_layers = 1
    learning_rate = 1e-3
    dropout = 0.4
    
    
    #dataset_to_use = "PCCD"
    #dataset_to_use = "flickr8k"
    dataset_to_use = "AVA"
    
    if dataset_to_use == "PCCD":
        imgs_folder = "datasets/PCCD/images/full"
        annotation_file = "datasets/PCCD/raw.json"
        test_file = "datasets/PCCD/images/PCCD_test.txt"
        
    elif dataset_to_use == "flickr8k":
        imgs_folder = "datasets/flickr8k/test_examples"
        annotation_file = "datasets/flickr8k/captions.txt"
        test_file = "datasets/flickr8k/flickr8k_test.txt"
    
    elif dataset_to_use == "AVA":
        imgs_folder = "datasets/AVA/images"
        annotation_file = "datasets/AVA/CLEAN_AVA_SAMPLE_COMMENTS.feather"
        test_file = "datasets/AVA/AVA_test.txt"
    
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )
    
    path = "CNN-LSTM/runs/checkpoint.pth.tar"

    freq_threshold = 5
    if dataset_to_use == "PCCD":
        dataset = PCCD(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "flickr8k":
        dataset = Flickr8k(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    elif dataset_to_use == "AVA":
        dataset = AVA(imgs_folder, annotation_file, test_file, transform=transform, freq_threshold=freq_threshold)
    
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
    
    ## Generate text from images
    print_examples(model, device, dataset)
    
    ## Generate text from random initialization
    #start_token = torch.tensor(dataset.vocab.stoi["<SOS>"]).to(device)
    #hiddens = torch.rand(1, embed_size).to(device)
    #outputs = model.decoder.generate_text(start_token, hiddens, dataset.vocab)
    #print(outputs)

if __name__ == "__main__":
    test()