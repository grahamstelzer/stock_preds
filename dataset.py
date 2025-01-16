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

# TODO: somewhere we need to extract all the "headlines" and "stocks_vector"s from the combined CSV
#       so we can send to dataset constructor

import torch
import torch.nn as nn

class SlidingWindowDataset(torch.utils.data.Dataset):
    # take in data like such:
    # Index,Date,Open,High,Low,Close,Adj Close,Volume,Headline
    # 1767,2011-08-24,12.93,13.2,12.88,13.03,12.99,144318923.0,Senna to replace Heidfeld in Belgium
    def __init__(self, headlines, stock_vectors, embedding_model, d_model, window_size):

        # save values
        self.embedding_model = embedding_model
        self.d_model = d_model
        self.window_size = window_size
        
        # setup linear layer for the stocks_vector transformation
        self.linear = nn.Linear(6, d_model) # NOTE: change this on adding volume or not

        # NOTE: apparently it is not really standard practice to setup the embeddings here
        #       in this case, they are just part of the data format, so it SHOULD be fine
        #       - TODO: research better techniques

        self.embeddings = [embedding_model(headline) for headline in headlines]
        self.stock_vectors = [torch.tensor(stock_vector) for stock_vector in stock_vectors]
        self.transformed_stock_vectors = [self.linear(stock_vector).unsqueeze(0) for stock_vector in self.stock_vectors]

        print("embeddings:")
        print(self.embeddings)
        print("stock vectors")
        print(self.stock_vectors)
        print("transformed stock vectors:")
        print(self.transformed_stock_vectors)

        self.combined = [torch.cat([emb, tsf], dim=0) for emb, tsf in zip(self.embeddings, self.transformed_stock_vectors)]
        self.sequence = torch.cat(self.combined, dim=0)  # Full sequence (total_seq_len, d_model)

    def __len__(self):
        return len(self.sequence) - self.window_size

    def __getitem__(self, idx):
        # Input: Current window
        window = self.sequence[idx:idx + self.window_size]  # (window_size, d_model)
        # Target: Next date
        target = self.sequence[idx + self.window_size]  # (1, d_model)
        return window, target
    

    


# Define a mock embedding model
class MockEmbeddingModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, sentence):
        # Mock embedding: Convert each character in the sentence to a random vector
        seq_len = len(sentence)
        return torch.randn(seq_len, self.d_model)

# Parameters
d_model = 16  # Dimensionality of the embeddings
window_size = 5  # Size of the sliding window
sentences = ["hello", "world", "torch", "dataset", "example"]  # Sample sentences
small_floats = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] for _ in sentences]  # Dummy small float vectors

# Initialize the mock embedding model
embedding_model = MockEmbeddingModel(d_model)

# Create the SlidingWindowDataset
dataset = SlidingWindowDataset(sentences, small_floats, embedding_model, d_model, window_size)

# Check dataset properties
print(f"Dataset length: {len(dataset)}")

# Access a sample
for i in range(min(3, len(dataset))):  # Show the first 3 samples
    window, target = dataset[i]
    print(f"Sample {i}:")
    print(f"  Window shape: {window.shape}")  # Should be (window_size, d_model)
    print(f"  Target shape: {target.shape}")  # Should be (d_model,)

# Create a DataLoader for batching
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through the DataLoader
print("\nDataloader output:")
for batch_idx, (windows, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Windows shape: {windows.shape}")  # Should be (batch_size, window_size, d_model)
    print(f"  Targets shape: {targets.shape}")  # Should be (batch_size, d_model)



# Example data
sentences = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4"]
numerical_data = [[2, 2, 2, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]]

# Create the dataset and DataLoader
# dataset = SequentialDataset(sentences, numerical_data, window_size=1)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
