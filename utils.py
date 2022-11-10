import torch
from PIL import Image

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("-- Saving Checkpoint --")
    torch.save(state, filename)
    
    
def load_checkpoint(checkpoint, model, optimizer):
    print("-- Loading Checkpoint --")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

