# NOTE: possible skeleton for how training will work
# TODO: fill this in better

from config inport num_epochs, model, loss_fun

def train(num_epochs):
    for epoch in range(num_epochs):
        for x_text, x_num, y_true in data_loader:
            # Preprocess inputs (e.g., tokenize sentences, normalize numbers)
            x_text_embedded = text_embedding(x_text)
            x_num_embedded = num_embedding(x_num)
            
            # Forward pass
            y_pred = model(x_text_embedded, x_num_embedded)
            
            # Compute loss
            loss = loss_fn(y_pred, y_true)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
