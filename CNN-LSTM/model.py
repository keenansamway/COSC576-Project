import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# class EncoderCNN(nn.Module):
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout, train_model=False):
        super(EncoderCNN, self).__init__()
        self.train_model = train_model
        
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.linear = nn.Linear(self.in_features, embed_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        # images: (batch_size, 3, 224, 224)
        
        features = self.resnet(images)                      # features: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)      # features: (batch_size, 2048)
        features = self.linear(features)                    # features: (batch_size, embed_size)
        features = self.relu(features)                      
        # features = self.bn(features)
        features = self.dropout(features)                   # features: (batch_size, embed_size)
        return features

# class DecoderRNN(nn.Module):
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super(DecoderLSTM, self).__init__()
        if num_layers < 2: dropout=0.0
        
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (caption_length, batch_size)
        
        captions = captions[:-1]                                            # captions: (caption_length-1, batch_size)
        embeddings = self.word_embedding(captions)                          # embeddings: (caption_length-1, batch_size, embed_size)
        packed = torch.cat((features.unsqueeze(0), embeddings), dim=0)      # packed: (caption_length, batch_size, embed_size)
        lstm_out, _ = self.lstm(packed)                                     # lstm_out: (caption_length, batch_size, hidden_size)
        outputs = self.linear(lstm_out)                                     # outputs: (caption_length, batch_size, vocab_size)
        
        return outputs

# class CNNtoRNN(nn.Module):
class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super(CNNtoLSTM, self).__init__()
        self.encoder = EncoderCNN(embed_size, dropout)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers, dropout)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            # image: (3, 224, 224)

            inputs = self.encoder(image)                                    # inputs: (batch_size=1, embed_size)
            inputs = inputs.unsqueeze(0)                                    # inputs: (1, batch_size=1, embed_size)
            hidden = None

            for _ in range(max_length):
                lstm_out, hidden = self.decoder.lstm(inputs, hidden)        # lstm_out: (1, batch_size=1, hidden_size)
                lstm_out = lstm_out.squeeze(0)                              # lstm_out: (batch_size=1, hidden_size)
                output = self.decoder.linear(lstm_out)                      # output: (batch_size=1, vocab_size)
                
                predicted = torch.argmax(output, dim=1)                     # predicted: (batch_size=1)
                result_caption.append(predicted.item())
                
                inputs = self.decoder.word_embedding(predicted)             # input: (batch_size=1, embed_size)
                inputs = inputs.unsqueeze(0)                                # input: (1, batch_size=1, embed_size)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
        return [vocabulary.itos[idx] for idx in result_caption]