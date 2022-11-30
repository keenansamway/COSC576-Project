import os, sys
import math
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

With attention and/or transformers
https://github.com/RoyalSkye/Image-Caption
https://www.kaggle.com/code/giangtran2408/image-captioning-with-pytorch/notebook

PyTorch has an issue with the backwards pass in LSTM when using batch first on MPS (Apple M1) device
"""


def train(path):
    # Hyperparameters
    embed_size = 512
    hidden_size = 512
    num_layers = 1
    learning_rate = 1e-4
    batch_size = 64
    num_workers = 2
    dropout = 0.2
    
    start_epochs = 0
    num_epochs = 10
    save_every_x_epochs = 10
    
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
        #validate_file = "datasets/flickr8k/captions_validate.feather"
        
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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Nvidia CUDA Acceleration or Apple M1 Metal Acceleration or CPU  
      
    transform = transforms.Compose(
        [
            transforms.Resize((232,232)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
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
    """
    validate_loader, validate_dataset = get_loader(
        dataset_to_use=dataset_to_use,
        imgs_folder=imgs_folder,
        annotation_file=validate_file,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        freq_threshold=5,
    )
    """
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
    
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            
    # for tensorboard
    if save_model:
        writer = SummaryWriter(os.path.join("CNN-LSTM/runs/", dataset_to_use))
    step = 0
    
    if load_model:
        step = load_checkpoint(torch.load(path), model, optimizer)
        
    model.train()

    for epoch in range(start_epochs, num_epochs):    
        #train_loss = 0
        #train_n = 0
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
            
            pbar.set_description(desc="Epoch [{}]/[{}] - Train Loss: {:.5f}".format(epoch+1, num_epochs, loss.item()))
            
            #train_loss += loss.item()
            #train_n += 1
            
            if save_model:
                writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
        
        #avg_train_loss = train_loss / train_n
        
        # Calculate validation loss # NEEDS UPDATING
        """
        validate_loss = 0
        validate_n = 0
        model.eval()
        for idx, (imgs, raw_captions, lengths) in enumerate(validate_loader):
            imgs = imgs.to(device)
            raw_captions = raw_captions.to(device)
            
            captions = raw_captions[:-1]
            targets = raw_captions[1:]
            
            outputs = model(imgs, captions)
            outputs = outputs.permute(1, 2, 0)
            targets = targets.permute(1, 0)
            
            loss = criterion(outputs, targets)
            
            validate_loss += loss.item()
            validate_n += 1   
        model.train()
        
        avg_validate_loss = validate_loss / validate_n
        
        if save_model:
            writer.add_scalar("Validation loss", avg_validate_loss, global_step=step)
        
        print("Avg Train Loss: {:.5f} - Validate Loss: {:.5f}".format(avg_train_loss, avg_validate_loss))
        """
        
        if save_model:
            save_checkpoint(model, optimizer, step, filename=f"CNN-LSTM/runs/checkpoint{math.ceil((epoch+1)/save_every_x_epochs)*save_every_x_epochs}.pth.tar")
    
           
if __name__ == "__main__":
    path = f"CNN-LSTM/runs/checkpoint{100}.pth.tar"
    train(path)