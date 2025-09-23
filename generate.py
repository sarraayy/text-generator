# generate.py
import os, pickle, torch
from model import Transformer

SEQ_LEN = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'out/model.pt'
VOCAB_PATH = 'vocab.pkl'

# Load vocab
with open(VOCAB_PATH,'rb') as f:
    vocab = pickle.load(f)
stoi, itos = vocab['stoi'], vocab['itos']
vocab_size = len(stoi)

# Load model
model = Transformer(vocab_size=vocab_size).to(DEVICE)
if not os.path.exists(MODEL_PATH):
    print("Checkpoint not found. Train first.")
    exit(1)

try:
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
except Exception as e:
    print(f"Checkpoint mismatch: {e}")
    exit(1)

model.eval()

def generate(prompt, max_len=200):
    tokens = [stoi.get(c, 0) for c in prompt]  # map prompt chars to IDs
    for _ in range(max_len):
        x = torch.tensor(tokens[-SEQ_LEN:], dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)[0, -1]
            next_id = torch.argmax(logits).item()  # greedy decoding
        tokens.append(next_id)
    return ''.join([itos[i] for i in tokens])

# Generate samples
for p in ["ROMEO:", "JULIET:"]:
    text = generate(p, max_len=300)
    out_file = f"generated_{p[:-1]}.txt"
    with open(out_file, 'w') as f:
        f.write(text)
    print(f"Saved {out_file}")
