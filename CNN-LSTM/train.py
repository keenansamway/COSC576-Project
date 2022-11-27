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
from model import CNNtoLSTM, EncoderCNN, DecoderLSTM

"""
Used code from:
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
https://github.com/rammyram/image_captioning/blob/master/Image_Captioning.ipynb
https://github.com/RoyalSkye/Image-Caption
https://www.kaggle.com/code/giangtran2408/image-captioning-with-pytorch/notebook

PyTorch has an issue with the backwards pass in LSTM when using batch first on MPS (Apple M1) device
"""

def train():
    # Hyperparameters
    embed_size = 128
    hidden_size = 128
    num_layers = 1
    learning_rate = 3e-4
    batch_size = 64
    num_workers = 2
    dropout = 0.4
    
    num_epochs = 5
    
    #dataset_to_use = "PCCD"
    dataset_to_use = "flickr8k"
    #dataset_to_use = "flickr30k"
    #dataset_to_use = "AVA"
    
    if dataset_to_use == "PCCD":
        imgs_folder = "datasets/PCCD/images/full"
        train_file = "datasets/PCCD/raw.json"
        
    elif dataset_to_use == "flickr8k":
        imgs_folder = "datasets/flickr8k/images"
        train_file = "datasets/flickr8k/captions_train.feather"
        validate_file = "datasets/flickr8k/captions_validate.feather"
        
    elif dataset_to_use == "flickr30k":
        imgs_folder = "datasets/flickr30k/images"
        train_file = "datasets/flickr30k/captions.txt"
    
    elif dataset_to_use == "AVA":
        imgs_folder = "datasets/AVA/images"
        train_file = "datasets/AVA/CLEAN_AVA_SAMPLE_COMMENTS.feather"
    
    load_model = False
    save_model = True
    train_CNN = False
    # True False
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           ## Nvidia CUDA Acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    ## Apple M1 Metal Acceleration
    
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )
    
    train_loader, train_dataset = get_loader(
        dataset_to_use=dataset_to_use,
        imgs_folder=imgs_folder,
        annotation_file=train_file,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        freq_threshold=5,
    )
    
    validate_loader, _ = get_loader(
        dataset_to_use=dataset_to_use,
        imgs_folder=imgs_folder,
        annotation_file=validate_file,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        freq_threshold=5,
    )
    
    vocab_size = len(train_dataset.vocab)
    
    # initialize model, loss, etc
    model = CNNtoLSTM(
        embed_size=embed_size, 
        hidden_size=hidden_size, 
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout,
        train_CNN=train_CNN,
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            
    # for tensorboard
    if save_model:
        writer = SummaryWriter(os.path.join("CNN-LSTM/runs/", dataset_to_use))
    step = 0
    
    if load_model:
        step = load_checkpoint(torch.load("CNN-LSTM/runs/checkpoint.pth.tar"), model, optimizer)
        
    model.train()

    for epoch in range(num_epochs):    
        train_loss = 0
        pbar = tqdm(train_loader, desc="Epoch: {}".format(epoch+1), total=len(train_loader), leave=True)
        for idx, (imgs, raw_captions, lengths) in enumerate(pbar):
            # imgs: (batch size, 3, 224, 224)
            # captions: (caption length, batch size)
            
            imgs = imgs.to(device)
            raw_captions = raw_captions.to(device)
            
            captions = raw_captions[:-1]
            targets = raw_captions[1:]

            # outputs: (caption length, batch size, vocab size)
            outputs = model(imgs, captions)
            outputs = outputs.permute(1, 2, 0)
            targets = targets.permute(1, 0)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(desc="Epoch [{}/[{}] - Train Loss: {:.5f}".format(epoch+1, num_epochs, loss.item()))
            
            train_loss = loss.item()
            
            if save_model:
                writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
        
        # Try to get validation loss working
        """
        validate_loss = 0
        num = 0
        for idx, (imgs, raw_captions, lengths) in enumerate(validate_loader):
            imgs = imgs.to(device)
            raw_captions = raw_captions.to(device)
            
            captions = raw_captions[:-1]
            targets = raw_captions[1:]
            
            model.eval()
            
            outputs = model(imgs, captions)
            outputs = outputs.permute(1, 2, 0)
            targets = targets.permute(1, 0)
            
            loss = criterion(outputs, targets)
            validate_loss += loss.item()
            num += 1
            
            model.train()
        avg_validate_loss = validate_loss / num
        
        pbar.set_description(desc="Epoch [{}/[{}] - Train Loss: {:.5f} - Validate Loss: {:.5f}".format(epoch+1, num_epochs, loss.item(), avg_validate_loss))
        """
        
        if save_model:
            save_checkpoint(model, optimizer, step, filename="CNN-LSTM/runs/checkpoint.pth.tar")
    
           
if __name__ == "__main__":
    train()