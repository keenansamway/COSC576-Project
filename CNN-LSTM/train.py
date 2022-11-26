import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import print_examples, save_checkpoint, load_checkpoint
from get_loader import get_loader
from torch.nn.utils.rnn import pad_sequence
from model import CNNtoLSTM

"""
Used code from:
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
https://github.com/rammyram/image_captioning/blob/master/Image_Captioning.ipynb

PyTorch has an issue with the backwards pass in LSTM when using batch first on MPS (Apple M1) device
"""

def train():
    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-4
    batch_size = 64
    num_workers = 0
    dropout = 0.4
    
    num_epochs = 20
    
    #dataset_to_use = "PCCD"
    dataset_to_use = "flickr8k"
    #dataset_to_use = "AVA"
    
    if dataset_to_use == "PCCD":
        imgs_folder = "datasets/PCCD/images/full"
        annotation_file = "datasets/PCCD/raw.json"
        
    elif dataset_to_use == "flickr8k":
        imgs_folder = "datasets/flickr8k/images"
        annotation_file = "datasets/flickr8k/captions.txt"
    
    elif dataset_to_use == "AVA":
        imgs_folder = "datasets/AVA/images"
        annotation_file = "datasets/AVA/CLEAN_AVA_FULL_COMMENTS.feather"
    
    load_model = True
    save_model = True
    train_CNN = False
    # True False
    
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )
    
    train_loader, dataset = get_loader(
        dataset_to_use=dataset_to_use,
        imgs_folder=imgs_folder,
        annotation_file=annotation_file,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        freq_threshold=5,
    )
    vocab_size = len(dataset.vocab)
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           ## Nvidia CUDA Acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    ## Apple M1 Metal Acceleration
    
    # initialize model, loss, etc
    model = CNNtoLSTM(
        embed_size=embed_size, 
        hidden_size=hidden_size, 
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout,
        train_CNN=train_CNN,
        ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            
    # for tensorboard
    if save_model:
        writer = SummaryWriter(os.path.join("CNN-LSTM/runs/", dataset_to_use))
    step = 0
    
    if load_model:
        step = load_checkpoint(torch.load("CNN-LSTM/runs/checkpoint.pth.tar"), model, optimizer)
        
    model.train()
    #15e
    for epoch in range(num_epochs):    
        for idx, (imgs, captions, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
            captions = captions.to(device)
            targets = targets.to(device)
            
            #targets = captions[1:]
            #captions = captions[:-1]
            
            outputs = model(imgs, captions)
            
            targets = targets.view(-1)
            outputs = outputs.view(-1, vocab_size)
            
            loss = criterion(outputs, targets)
            
            if save_model:
                writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
            
            loss.backward()
            optimizer.step()
            
            
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, filename="CNN-LSTM/runs/checkpoint.pth.tar")
        print("Epoch [{}/[{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

if __name__ == "__main__":
    train()