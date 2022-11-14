import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import print_examples, save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import CNNtoLSTM

def train():
    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-2
    batch_size = 128
    num_workers = 2
    num_epochs = 10
    
    load_model = False
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
        imgs_folder="datasets/PCCD/images/full",
        annotation_file="datasets/PCCD/raw.json",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
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
        ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    for name, param in model.encoder.resnet.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN
            
    # for tensorboard
    writer = SummaryWriter("CNN-LSTM/runs/PCCD")
    step = 0
    
    
    if load_model:
        step = load_checkpoint(torch.load("CNN-LSTM/runs/checkpoint.pth.tar"), model, optimizer)
        
    model.train()
    
    # for tensorboard
    
    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)
        
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
            
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

    print("Epoch [{}/[{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

if __name__ == "__main__":
    train()