import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


# class EncoderCNN(nn.Module):
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_model):
        super(EncoderCNN, self).__init__()
        self.train_model = train_model
        
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.in_features = self.resnet.fc.in_features
        
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        for param in self.resnet.parameters():
            param.requires_grad = train_model
        
        self.linear = nn.Linear(self.in_features, embed_size)
        """ 
        self.fc = nn.Sequential(
            nn.Linear(self.in_features, embed_size//2),
            nn.ReLU(),
            nn.Linear(embed_size//2, embed_size),
            nn.ReLU(),
        )
         """
        #self.relu = nn.ReLU()
                
    def forward(self, images):
        # images: (batch_size, 3, 224, 224)
        
        features = self.resnet(images)                      # features: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)      # features: (batch_size, 2048)
        #resnet (minus final layer) output
        
        features = self.linear(features)                    # features: (batch_size, embed_size)
        #features = self.fc(features)
        
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
            padding_idx=0,
            )
        
        # LSTM - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # Input:  (sequence length, batch size, embed size)
        # States: (2 if bidirectional else 1 * num layers, batch size, hidden size)
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
        '''
        self.fc = nn.Linear(
            in_features=hidden_size, 
            out_features=vocab_size,
            )
        '''
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.ReLU(),
            nn.Linear(hidden_size*4, hidden_size*8),
            nn.ReLU(),
            nn.Linear(hidden_size*8, vocab_size)
        )
    
    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (caption_length, batch_size)
        
        embeddings = self.embed(captions)
        states = self.init_hidden(features)
        
        lstm_out, _ = self.lstm(embeddings, states)
        fc_out = self.fc(lstm_out)
                
        return fc_out
    
    def init_hidden(self, features):
        state = torch.stack([features]*(self.num_layers), dim=0)
        return (state, state)
        
    ## REVIEW
    def generate_text(self, start_token, hiddens, vocabulary, max_length=50):
        result_caption = [start_token.item()]
        start_token = start_token.unsqueeze(0)
        
        with torch.no_grad():
            lstm_in = self.embed(start_token)
            
            features = hiddens.squeeze(0)
            state = torch.stack([features]*(self.num_layers), dim=0)
            states = (state, state)

            for _ in range(max_length):
                lstm_out, states = self.lstm(lstm_in, states)
                fc_out = self.fc(lstm_out)
                
                predicted = torch.argmax(fc_out, dim=1)
                result_caption.append(predicted.item())
                
                lstm_in = self.embed(predicted)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
        return [vocabulary.itos[idx] for idx in result_caption]

# class CNNtoRNN(nn.Module):
class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout, train_CNN=False):
        super(CNNtoLSTM, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_CNN)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers, dropout)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    ## REVIEW
    def caption_image(self, start_token, image, vocabulary, max_length=50):
        result_caption = [start_token.item()]
        start_token = start_token.unsqueeze(0)
        
        with torch.no_grad():
            # start_token: (1)
            # image: (3, 224, 224)
            
            lstm_in = self.decoder.embed(start_token)
            
            features = self.encoder(image).squeeze(0)
            states = self.decoder.init_hidden(features)

            for _ in range(max_length):
                lstm_out, states = self.decoder.lstm(lstm_in, states)
                fc_out = self.decoder.fc(lstm_out)
                
                predicted = torch.argmax(fc_out, dim=1)
                result_caption.append(predicted.item())
                
                lstm_in = self.decoder.embed(predicted)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
        return [vocabulary.itos[idx] for idx in result_caption]