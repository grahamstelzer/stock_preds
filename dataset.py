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



import torch

class SequentialDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, numerical_data, window_size=1):
        self.sentences = sentences
        self.numerical_data = numerical_data
        self.window_size = window_size

    def __len__(self):
        return len(self.numerical_data) - self.window_size

    def __getitem__(self, idx):
        x_text = self.sentences[idx]
        x_num = self.numerical_data[idx]
        y_true = self.numerical_data[idx + self.window_size]  # Next numerical values
        return x_text, x_num, y_true

# Example data
sentences = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4"]
numerical_data = [[2, 2, 2, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]]

# Create the dataset and DataLoader
dataset = SequentialDataset(sentences, numerical_data, window_size=1)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

# Iterate over training data
for x_text, x_num, y_true in train_loader:
    print("Input Sentence:", x_text)
    print("Input Numerical Values:", x_num)
    print("True Output:", y_true)
