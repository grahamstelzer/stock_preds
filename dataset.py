# we will try sliding window approach:

# in our training, we want to do:
# for i, (inputs, targets) in enumerate(train_loader):
#     # Move data to device
#     inputs, targets = inputs.to(device), targets.to(device)

# which means train_loader should be:
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# so dataset should have in __getitem__():
# TODO: figure out what dataset should return!!
#       "return input_features, target_value(s)" i think?


# TODO: build dataset from saved files from preproc
# TODO: make sure the input word embeddings are simply the tokenized representations

# TODO: double check tbe batching, I dont know if the returned "combined"
#       sequence is the len of one headline or multiple
#           - try a test headline


import torch
import csv
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class SlidingWindowDataset(torch.utils.data.Dataset):

    def __init__(self, headlines, stock_vectors, d_model, window_size):

        # save values
        self.d_model = d_model
        self.window_size = window_size

        # setup linear layer for the stocks_vector transformation
        self.linear = nn.Linear(6, d_model) # NOTE: change this on adding volume or not

        # NOTE: apparently it is not really standard practice to setup the embeddings here
        #       in this case, they are just part of the data format, so it SHOULD be fine
        #       - TODO: research better techniques


        word_embeddings = self.generate_word_embeddings(headlines) # should be 2650, 10, 512
        stock_embeddings = self.generate_stock_embeddings(stock_vectors) # should be 2650, 1, 512

        # Concatenate word embeddings with stock embeddings
        self.sequence = []
        for word_emb, stock_emb in zip(word_embeddings, stock_embeddings):

            stock_emb = stock_emb.detach()
            word_emb = word_emb.detach()

            combined = torch.cat((word_emb, stock_emb), dim=0)  # Concatenate along the sequence length dimension
            
            self.sequence.append(combined)

        self.sequence = torch.stack(self.sequence)  # Shape: (2650, seq_len, 512)

        print(self.sequence.shape)
        print(self.sequence[0])

    def __len__(self):
        return len(self.sequence) - self.window_size

    def __getitem__(self, idx):
        # Input: Current window
        window = self.sequence[idx:idx + self.window_size]  # (window_size, d_model)
        # print(window.shape)
        # Target: Next date
        target = self.sequence[idx + self.window_size]  # (1, d_model)
        # print(target.shape)
        return window, target

    def generate_word_embeddings(self, sentences):
        
        # tokenize
        tokenized = [sentence.lower().split() for sentence in sentences]
        
        vocab = {word: idx for idx, word in enumerate({word for sentence in tokenized for word in sentence})}
        vocab_size = len(vocab)

        token_indices = [[vocab[word] for word in sentence] for sentence in tokenized]

        seq_len = max(len(seq) for seq in tokenized)  # Define the maximum sequence length
        padded_indices = [torch.tensor(seq + [0] * (seq_len - len(seq)), dtype=torch.long, requires_grad=False)[:seq_len] for seq in token_indices]

        # TODO: move this out to class init
        d_model = 512  # Dimension of embedding
        embedding_layer = nn.Embedding(vocab_size, d_model)

        padded_tensor = torch.stack(padded_indices)

        word_embeddings = embedding_layer(padded_tensor)  # Shape: (batch_size, seq_len, d_model)

        print(word_embeddings.shape)  # Output: (3, 10, 512)

        return word_embeddings
    
    def generate_stock_embeddings(self, stock_vectors):

        stock_tensors = [torch.tensor(stock_vector, dtype=torch.float32) for stock_vector in stock_vectors]

        normalized_stocks = [(stock_tensor - stock_tensor.mean()) / stock_tensor.std() for stock_tensor in stock_tensors]

        transformed_stocks = [self.linear(stock_tensor).unsqueeze(0) for stock_tensor in normalized_stocks]

        stock_embeddings = torch.cat(transformed_stocks, dim=0).unsqueeze(1)  # Full sequence (total_seq_len, 1, d_model)

        print(stock_embeddings.shape)

        return stock_embeddings







# get items for dataset input
# NOTE: must make sure the stockdata lens are what the dataset expects
#       currently 6 and 6, considers Open...Volume categories from training.csv

def get_headlines():
    headlines = []
    with open('./data/train/training.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            headlines.append(row[8])  # headliens at column index 8
    return headlines

def get_stockdata():
    all_data = []
    with open('./data/train/training.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            sublist = [float(item) for item in row[2:8]]
            all_data.append(sublist)
        return all_data





# transformer model
# TODO: move this to model.py

import matplotlib.pyplot as plt

class Transformer(nn.Module):

    def __init__(self, width=512, layers=4): # TODO: other params?
        super().__init__()

        self.layers = layers

        # TODO: tweak these
        layer = nn.TransformerEncoderLayer(
            d_model = width,
            nhead=width // 64,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

        # TODO: should match target from dataset??
        #       I think embed out is supposed to be a final transformation before 
        #       we return so that it matches the target when calculating loss value
        self.embed_out = nn.Linear(width, width) 

    def forward(self, x):
        # TODO: double check the dimensions on this
        # x should be (batch, window_size (as seq_len), d_model)??
        x = self.transformer(x)

        # then return as (batch, window, d_model)?
        return self.embed_out(x)




d_model = 512 
window_size = 9
sentences = get_headlines()
small_floats = get_stockdata()

dataset = SlidingWindowDataset(sentences, small_floats, d_model, window_size)
encoder_model = Transformer()


# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=32)

# model = train_transformer(
#     model=transformer_model,
#     train_dataloader=train_dataloader,
#     val_dataloader=val_dataloader,
#     num_epochs=10,
#     learning_rate=1e-4,
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# )

from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns

def train_transformer(model, train_loader, optimizer, num_epochs, device='cuda'):
    """
    Training loop for transformer model with sliding window prediction
    focusing on the last row of each target matrix for loss calculation.
    """
    model.train()
    criterion = nn.MSELoss()  # or your preferred loss function
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, (windows, targets) in enumerate(progress_bar):
            # Move data to device
            windows = windows.to(device)  # shape: [batch_size, 9, 27, 512]
            targets = targets.to(device)  # shape: [batch_size, 27, 512]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            # Reshape windows if needed for your transformer
            batch_size = windows.shape[0]
            windows_reshaped = windows.view(batch_size, 243, 512)  # shape: [batch_size, 9*27, 512]
            
            # Get model predictions
            outputs = model(windows_reshaped)  # shape: [batch_size, 27, 512]
            

            # Extract only the last row (27th row) from both predictions and targets
            predicted_last_row = outputs[:, -1, :]  # shape: [batch_size, 512]
            target_last_row = targets[:, -1, :]    # shape: [batch_size, 512]
            # Graph the first value in each of these rows over time using matplotlib
            if batch_idx == len(train_loader) - 1:  # Only plot at the end of each epoch
                for i in range(3):  # Only plot the first 3 values
                    pred_values = predicted_last_row[:, i].detach().cpu().numpy()
                    target_values = target_last_row[:, i].detach().cpu().numpy()
                    
                    plt.plot(pred_values, label=f'Predicted Value {i}')
                    plt.plot(target_values, linestyle='dotted', label=f'Target Value {i}')
                
                plt.xlabel('Time')
                plt.ylabel('Values')
                plt.title(f'First 3 Values in Last Row Over Time - Epoch {epoch + 1}')
                plt.legend()
                plt.show()
            # Calculate loss only on the last row
            loss = criterion(predicted_last_row, target_last_row)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.6f}'
            })
        
        epoch_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} - Average Loss: {epoch_loss:.6f}')
        
    return model

# Example usage:
def main():
    # Assuming your dataset and model are already defined
    batch_size = 32
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model, optimizer
    model = encoder_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    trained_model = train_transformer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )


main()