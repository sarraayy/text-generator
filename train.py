# train.py
import pickle, torch, torch.nn as nn, torch.optim as optim
from model import Transformer
import csv, os

SEQ_LEN = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocab
with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
stoi, itos = vocab['stoi'], vocab['itos']
vocab_size = len(stoi)

# Load dataset
text = open('input.txt').read()
data = [stoi[c] for c in text]
split = int(0.9*len(data))
train_data, val_data = data[:split], data[split:]

def get_batch(data):
    import random
    idx = random.randint(0, len(data)-SEQ_LEN-1)
    x = torch.tensor(data[idx:idx+SEQ_LEN], dtype=torch.long)
    y = torch.tensor(data[idx+1:idx+SEQ_LEN+1], dtype=torch.long)
    return x, y

# Model, loss, optimizer
model = Transformer(vocab_size=vocab_size)
model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Make sure out/ folder exists
os.makedirs("out", exist_ok=True)

# Open CSV log file
with open("training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])  # header

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for _ in range(len(train_data)//BATCH_SIZE):
            x_batch, y_batch = [], []
            for _ in range(BATCH_SIZE):
                x, y = get_batch(train_data)
                x_batch.append(x)
                y_batch.append(y)
            x = torch.stack(x_batch).to(DEVICE)
            y = torch.stack(y_batch).to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(train_data)//BATCH_SIZE)

        # Validation
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_data)
            x_val = x_val.unsqueeze(0).to(DEVICE)
            y_val = y_val.unsqueeze(0).to(DEVICE)
            val_logits = model(x_val)
            val_loss = criterion(val_logits.view(-1, vocab_size), y_val.view(-1))

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss.item():.4f}")

        # Save to CSV
        writer.writerow([epoch+1, avg_loss, val_loss.item()])

# Save model
torch.save(model.state_dict(), 'out/model.pt')
print("Training complete. Model saved to out/model.pt")
