import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# class EncoderCNN(nn.Module):
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_model=False):
        super(EncoderCNN, self).__init__()
        self.train_model = train_model
        
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        features = self.resnet(images)
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
        packed = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        
        hiddens, _ = self.lstm(packed)
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
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            inputs = self.encoder(image).unsqueeze(0)
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(inputs, states)     # hiddens: ()
                output = self.decoder.linear(hiddens.squeeze(0))        # outputs: ()
                predicted = torch.argmax(output, dim=1)                 # predicted: ()
                
                result_caption.append(predicted.item())
                inputs = self.decoder.embed(predicted).unsqueeze(0)     # input: ()
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        return [vocabulary.itos[idx] for idx in result_caption]