import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


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
        #features = self.relu(features)                      
        # features = self.bn(features)
        features = self.dropout(features)                   # features: (batch_size, embed_size)
        return features

# Language Model
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout, bidirectional=False):
        super(DecoderLSTM, self).__init__()
        if num_layers < 2: dropout=0.0
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word Embeddings - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # Input:  (sequence length, batch size)
        # Output: (sequence length, batch size, embed size)
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_size,
            )
        
        # LSTM - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # Input:  (sequence length, batch size, embed size)
        # Output: (sequence length, batch size, 2 if bidirectional else 1 * hidden size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=False,
            bidirectional=bidirectional,
            )
        
        # Fully Connected
        # Input:  (sequence length, batch size, hidden size)
        # Output: (sequence length, batch size, vocab size)
        self.linear = nn.Linear(
            in_features=hidden_size, 
            out_features=vocab_size,
            )
    
    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (caption_length, batch_size)
        
        captions1 = captions[:-1]                                      # captions: (caption_length-1, batch_size)
        embeddings = self.embed(captions1)                             # embeddings: (caption_length-1, batch_size, embed_size)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)  # packed: (caption_length, batch_size, embed_size)
        
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=False, enforce_sorted=True)    
        
        lstm_out, _ = self.lstm(embeddings)                              # lstm_out[0]: (caption_length, batch_size, hidden_size)
        linear_outputs = self.linear(lstm_out)                           # outputs: (caption_length, batch_size, vocab_size)
        
        #outputs = linear_outputs.reshape(-1, self.vocab_size)
        
        return linear_outputs
    
    def generate_text(self, inputs, vocabulary, max_length=50):
        result_text = []
        
        with torch.no_grad():
            features = inputs                                    # inputs: (batch_size=1, embed_size)
            lstm_in = features.unsqueeze(0)                                    # inputs: (1, batch_size=1, embed_size)
            hidden = None

            for _ in range(max_length):
                lstm_out, hidden = self.lstm(lstm_in, hidden)        # lstm_out: (1, batch_size=1, hidden_size)
                linear_in = lstm_out.squeeze(0)                              # lstm_out: (batch_size=1, hidden_size)
                linera_out = self.linear(linear_in)                      # output: (batch_size=1, vocab_size)
                
                predicted = torch.argmax(linera_out, dim=1)                     # predicted: (batch_size=1)
                result_text.append(predicted.item())
                
                lstm_in = self.embed(predicted)                      # input: (batch_size=1, embed_size)
                lstm_in = lstm_in.unsqueeze(0)                                # input: (1, batch_size=1, embed_size)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
        return [vocabulary.itos[idx] for idx in result_text]

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
    
    def caption_image(self, image, vocabulary, max_length=20):
        result_caption = []
        
        with torch.no_grad():
            # image: (3, 224, 224)

            features = self.encoder(image)                                    # inputs: (batch_size=1, embed_size)
            lstm_in = features.unsqueeze(0)                                    # inputs: (1, batch_size=1, embed_size)
            hidden = None

            for _ in range(max_length):
                lstm_out, hidden = self.decoder.lstm(lstm_in, hidden)        # lstm_out: (1, batch_size=1, hidden_size)
                linear_in = lstm_out.squeeze(0)                              # lstm_out: (batch_size=1, hidden_size)
                linera_out = self.decoder.linear(linear_in)                      # output: (batch_size=1, vocab_size)
                
                predicted = torch.argmax(linera_out, dim=1)                     # predicted: (batch_size=1)
                result_caption.append(predicted.item())
                
                lstm_in = self.decoder.embed(predicted)                      # input: (batch_size=1, embed_size)
                lstm_in = lstm_in.unsqueeze(0)                                # input: (1, batch_size=1, embed_size)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
        return [vocabulary.itos[idx] for idx in result_caption]