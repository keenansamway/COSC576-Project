import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm


# def train():
def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )
    
    

#if __name__ == "__main__":
    