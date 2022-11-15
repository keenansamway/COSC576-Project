import os, sys
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

def save_checkpoint(state, filename="CNN-LSTM/runs/checkpoint.pth.tar"):
    print("-- Saving Checkpoint --")
    torch.save(state, filename)
    
    
def load_checkpoint(checkpoint, model, optimizer):
    print("-- Loading Checkpoint --")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]
    )
    
    filename_loc = "datasets/PCCD/images/PCCD_test.txt"
    images_loc = "datasets/PCCD/images/full"
    
    filename_list = pd.read_csv(filename_loc, header=None)
    filename_list = filename_list.values.reshape(-1).tolist()
    
    
    model.eval()
    
    for i, dir in enumerate(filename_list):
        path = os.path.join(images_loc, dir)
        test_img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        print(f"Example {i}) OUTPUT: " + " ".join(model.caption_image(test_img.to(device), dataset.vocab)))
        if i > 2:
            break
        
    model.train()
    
if __name__ == "__main__":
    filename_loc = "datasets/PCCD/images/PCCD_test.txt"
    images_loc = "datasets/PCCD/images/full"
    
    filename_list = pd.read_csv(filename_loc, header=None)
    filename_list = filename_list.values.reshape(-1).tolist()
    
    print(filename_list)