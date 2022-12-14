import os, sys
import re, string
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

def save_checkpoint(checkpoint, filename):
    torch.save(checkpoint, filename)
    print("-- Saved Checkpoint --")
    
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    print("-- Loaded Checkpoint --")
    return step

regex = re.compile('[%s]' % re.escape(string.punctuation))
def clean_text(row):
        # Lower case & remove punctuation
        row = str(row).strip()
        row = row.lower()
        return regex.sub("", row)

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ]
    )
    
    filename_loc = dataset.test_file
    images_loc = dataset.imgs_dir
    
    filename_list = pd.read_csv(filename_loc, header=None)
    filename_list = filename_list.values.reshape(-1).tolist()
    
    model.eval()
    
    for i, dir in enumerate(filename_list):
        path = os.path.join(images_loc, dir)
        start_token = torch.tensor(dataset.vocab.stoi["<SOS>"]).to(device)
        test_img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        print(f"Example {i+1}) OUTPUT: " + " ".join(model.caption_image(start_token, test_img, dataset.vocab)))
        # if i > 5:
        #     break
        
    model.train()
    
if __name__ == "__main__":
    filename_loc = "datasets/PCCD/images/PCCD_test.txt"
    images_loc = "datasets/PCCD/images/full"
    
    filename_list = pd.read_csv(filename_loc, header=None)
    filename_list = filename_list.values.reshape(-1).tolist()
    
    print(filename_list)