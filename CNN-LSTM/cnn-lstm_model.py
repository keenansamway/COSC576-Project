import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# class EncoderCNN(nn.Module):
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_model=False):
        super(EncoderCNN, self).__init__()
        self.train_model = train_model
        
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        features = self.model(images)
        features = self.relu(features)
        features = self.dropout(features)
        return features

# class DecoderRNN(nn.Module):
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        
        hiddens = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# class CNNtoRNN(nn.Module):
class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoLSTM, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
        