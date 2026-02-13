# train_torch_onehot.py
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import load_poems, save_run_log, simple_tokenize, build_vocab, tokens_to_ids, make_sequences

DEVICE = "mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu"


class SeqDatasetOneHot(Dataset):
    def __init__(self, X, Y, vocab_size):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.V = vocab_size

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        x_ids = self.X[idx]                    # [T]
        y_ids = self.Y[idx]                    # [T]
        # one-hot: [T, V]
        x_oh = torch.zeros(x_ids.size(0), self.V, dtype=torch.float32)
        x_oh.scatter_(1, x_ids.unsqueeze(1), 1.0)
        return x_oh, y_ids


class OneHotRNNLM(nn.Module):
    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        self.rnn = nn.RNN(input_size=vocab_size,
                          hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x_oh, h0=None):
        out, hn = self.rnn(x_oh, h0)        # out: [B,T,H]
        logits = self.fc(out)               # [B,T,V]
        return logits, hn


@torch.no_grad()
def generate(model, stoi, itos, seed_text="<bos>", max_new=40, temperature=1.0):
    model.eval()
    tokens = seed_text.split()
    ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    V = len(stoi)

    h = None
    for _ in range(max_new):
        x = torch.tensor(ids[-1:], dtype=torch.long,
                         device=DEVICE)  # last token
        x_oh = torch.zeros(1, 1, V, device=DEVICE)
        x_oh.scatter_(2, x.view(1, 1, 1), 1.0)

        logits, h = model(x_oh, h)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=0)
        nxt = torch.multinomial(probs, 1).item()
        ids.append(nxt)

    words = [itos[i] for i in ids]
    return " ".join(words)


def rnn_onehot_main():
    text = load_poems("poems.txt")
    tokens = ["<bos>"] + simple_tokenize(text) + ["<eos>"]
    stoi, itos = build_vocab(tokens, min_freq=1)
    ids = tokens_to_ids(tokens, stoi)

    seq_len = 25
    X, Y = make_sequences(ids, seq_len)
    ds = SeqDatasetOneHot(X, Y, vocab_size=len(stoi))
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    model = OneHotRNNLM(vocab_size=len(stoi), hidden=256).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Training One-Hot RNN on", DEVICE)

    epoch_losses = []
    epoch_times = []
    samples = []

    for epoch in range(50):
        model.train()
        t0 = time.perf_counter()
        total = 0.0
        steps = 0
        for x_oh, y in dl:
            x_oh = x_oh.to(DEVICE)      # [B,T,V]
            y = y.to(DEVICE)            # [B,T]

            logits, _ = model(x_oh)     # [B,T,V]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()
            steps += 1

        print(f"Epoch {epoch+1} | loss: {total/steps:.4f}")
        epoch_loss = float(total / max(steps, 1))
        epoch_losses.append(epoch_loss)

        sample = generate(model, stoi, itos, seed_text="<bos>",
                          max_new=40, temperature=0.9)
        samples.append(sample)

        t1 = time.perf_counter()
        epoch_times.append(float(t1 - t0))

        print("Sample:", sample)
        print(f"Epoch time: {epoch_times[-1]:.3f}s")

    out_path = save_run_log(
        out_path="runs/run_onehot.json",
        run_name="torch_onehot",
        epoch_losses=epoch_losses,
        epoch_times=epoch_times,
        samples=samples,
        config={
            "seq_len": seq_len,
            "batch_size": 64,
            "hidden": 256,
            "lr": 1e-3,
            "vocab_size": len(stoi),
            "epochs": 50,
            "device": DEVICE,
        },
        notes="PyTorch RNN with one-hot inputs",
        sample_for_quality="last",
    )
    print(f"Saved run log to: {out_path}")
    print(f"Total training time (one-hot): {sum(epoch_times):.2f}s")

    return out_path


if __name__ == "__main__":
    rnn_onehot_main()
