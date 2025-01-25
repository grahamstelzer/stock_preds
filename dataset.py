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

            # TODO: research detach() function
            stock_emb = stock_emb.detach()
            word_emb = word_emb.detach()

            combined = torch.cat((word_emb, stock_emb), dim=0)  # Concatenate along the sequence length dimension
            
            self.sequence.append(combined)

        self.sequence = torch.stack(self.sequence)  # Shape: (2650, seq_len, 512)

        # print(self.sequence.shape)
        # print(self.sequence[0])

    def __len__(self):
        return len(self.sequence) - self.window_size

    def __getitem__(self, idx):
        # Input: Current window
        window = self.sequence[idx:idx + self.window_size]  # (window_size, seq_len, d_model)
        # Target: Next date
        target = self.sequence[idx + self.window_size]  # (seq_len, d_model)

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

        # print(word_embeddings.shape)  # Output: (3, 10, 512)

        return word_embeddings
    
    def generate_stock_embeddings(self, stock_vectors):

        stock_tensors = [torch.tensor(stock_vector, dtype=torch.float32) for stock_vector in stock_vectors]

        normalized_stocks = [(stock_tensor - stock_tensor.mean()) / stock_tensor.std() for stock_tensor in stock_tensors]

        transformed_stocks = [self.linear(stock_tensor).unsqueeze(0) for stock_tensor in normalized_stocks]

        stock_embeddings = torch.cat(transformed_stocks, dim=0).unsqueeze(1)  # Full sequence (total_seq_len, 1, d_model)

        # print(stock_embeddings.shape)

        return stock_embeddings

    def reverse_transform(self, tensor):
        """
        Reverse the transformation of a 1 by 512 tensor back to a 1 by 6 tensor using the same linear layer.
        """
        # Use the inverse of the linear transformation
        inverse_linear = nn.Linear(self.d_model, 6).to(tensor.device)
        
        # Copy the weights and biases from the original linear layer
        inverse_linear.weight.data = self.linear.weight.data.t().to(tensor.device)
        inverse_linear.bias.data = -torch.matmul(self.linear.weight.data.t().to(tensor.device), self.linear.bias.data.to(tensor.device))
        
        original_tensor = inverse_linear(tensor)

        print(f"rev_trans: {original_tensor}")
        
        return original_tensor






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
        # self.embed_out = nn.Linear(width, width) 

    def forward(self, x):
        # TODO: double check the dimensions on this
        # x should be (batch, window_size (as seq_len), d_model)??
        x = self.transformer(x)

        # then return as (batch, window, d_model)?
        return x




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

    NOTE:
        rework the training loop, need to make sure the sliding window is looking at the right values
        currently loss values are super low, but the tensors being looked at are wrong
        remember we only care about the last row of each 27*512 matrix for calculating loss, since that should be stock values
        i think we may need to try and predict the full matrix, but send less attentiveness to the first 26 rows if possible
    """
    model.train()
    criterion = nn.MSELoss()  # or your preferred loss function
    tar_pred_pairs = []


    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, (windows, targets) in enumerate(progress_bar):

            # print(windows.shape) # (32, 9, 27, 512)     batch_size, window_size, seq_len, dmodel
            # print(targets.shape) # (32, 27, 512)        batch_size, seq_len, dmodel

            # input into transformer should be (32, 243, 512)
            #   or 32 instances of 243 by 512 matrices
            #   where each 243 by 512 is technically the seq_len * window_size (27 * 9)  

            #   batches should start with matrices 0-8 (window_size=9 as indices) and predict the 9th as a single 27 by 512 matrix
            #   then we will do matrices 1-9 (window_size=9 as indices) and try to predict the 10th as a 27 by 512
            
            # Reshape the windows to (batch_size, window_size * seq_len, d_model)
            windows = windows.view(windows.size(0), -1, windows.size(-1))  # (32, 243, 512)
            
            # Move data to device
            windows, targets = windows.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(windows)  # (batch_size, window_size * seq_len, d_model)
            print(outputs.shape)
            
            # Extract the last row of each target matrix
            target_last_row = targets[-1:, -1, :]  # (batch_size, d_model)
            predicted_last_row = outputs[-1:, -1, :]  # (batch_size, d_model)


            # 32, 512??
            # print(target_last_row.shape)
            # print(predicted_last_row.shape)

            # print(target_last_row)
            # print(predicted_last_row)

            # Calculate loss only on the last row
            loss = criterion(predicted_last_row, target_last_row)


            if batch_idx % 10 == 0:
                un_embedded_target = dataset.reverse_transform(target_last_row)
                un_embedded_predicted = dataset.reverse_transform(predicted_last_row)

                # print(un_embedded_target.shape)
                # print(un_embedded_predicted.shape)

                tar_pred_pairs.append([un_embedded_target, un_embedded_predicted]) # should be ( ((6),(6)), ((6),(6)) ... )

                print(f"len of tpp: {len(tar_pred_pairs)}")

            
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
        
    # plot the first values of all the sublists in 
    print(tar_pred_pairs)

    targets = [pair[0][0].tolist() for pair in tar_pred_pairs]
    predictions = [pair[1][0].tolist() for pair in tar_pred_pairs]

    plt.figure(figsize=(10, 5))
    plt.plot(targets, 'r--', label='Target')
    plt.plot(predictions, 'b-', label='Prediction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Target vs Prediction')
    plt.legend()
    plt.show()
    
    return model

def main():
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