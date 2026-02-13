# scratch_rnn_numpy.py
import time
import numpy as np
from utils import load_poems, save_run_log, simple_tokenize, build_vocab, tokens_to_ids


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def one_hot(idx, V):
    v = np.zeros((V, 1))
    v[idx] = 1.0
    return v


class ScratchRNN:
    def __init__(self, vocab_size, hidden_size=64, lr=1e-2, seed=42):
        rng = np.random.default_rng(seed)
        self.V = vocab_size
        self.H = hidden_size
        self.lr = lr

        # weights
        self.Wxh = rng.normal(0, 0.01, (self.H, self.V))
        self.Whh = rng.normal(0, 0.01, (self.H, self.H))
        self.Why = rng.normal(0, 0.01, (self.V, self.H))
        self.bh = np.zeros((self.H, 1))
        self.by = np.zeros((self.V, 1))

    def forward(self, inputs, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = hprev

        for t, idx in enumerate(inputs):
            xs[t] = one_hot(idx, self.V)                         # [V,1]
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @
                            hs[t-1] + self.bh)  # [H,1]
            ys[t] = self.Why @ hs[t] + self.by                  # [V,1]
            ps[t] = softmax(ys[t].ravel()).reshape(-1, 1)       # [V,1]
        return xs, hs, ps

    def loss_and_grads(self, inputs, targets, hprev):
        xs, hs, ps = self.forward(inputs, hprev)

        loss = 0.0
        for t in range(len(inputs)):
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        # grads init
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dhnext = np.zeros((self.H, 1))

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            # softmax CE gradient
            dy[targets[t]] -= 1.0
            dWhy += dy @ hs[t].T
            dby += dy

            dh = self.Why.T @ dy + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh                    # tanh'
            dbh += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t-1].T
            dhnext = self.Whh.T @ dhraw

        # clip
        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -5, 5, out=d)

        hlast = hs[len(inputs)-1]
        return loss, (dWxh, dWhh, dWhy, dbh, dby), hlast

    def step(self, grads):
        dWxh, dWhh, dWhy, dbh, dby = grads
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

    def sample(self, start_idx, itos, length=30, temperature=1.0):
        h = np.zeros((self.H, 1))
        x = one_hot(start_idx, self.V)
        out = []

        for _ in range(length):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = softmax((y.ravel() / max(temperature, 1e-6)))
            idx = np.random.choice(range(self.V), p=p)
            out.append(itos[idx])
            x = one_hot(idx, self.V)
        return " ".join(out)


def rnn_main():
    text = load_poems("poems.txt")
    tokens = ["<bos>"] + simple_tokenize(text) + ["<eos>"]
    stoi, itos = build_vocab(tokens, min_freq=1)
    ids = tokens_to_ids(tokens, stoi)

    rnn = ScratchRNN(vocab_size=len(stoi), hidden_size=128, lr=0.05)
    seq_len = 25
    h = np.zeros((rnn.H, 1))

    epoch_losses = []
    epoch_times = []
    samples = []

    for epoch in range(50):
        t0 = time.perf_counter()
        total_loss = 0.0
        n = 0
        for i in range(0, len(ids) - seq_len - 1, seq_len):
            inp = ids[i:i+seq_len]
            tgt = ids[i+1:i+seq_len+1]
            loss, grads, h = rnn.loss_and_grads(inp, tgt, h)
            rnn.step(grads)
            total_loss += loss
            n += 1

        avg = total_loss / max(n, 1)
        epoch_losses.append(float(avg))

        sample = rnn.sample(stoi["<bos>"], itos, length=30, temperature=0.9)
        samples.append(sample)

        t1 = time.perf_counter()
        epoch_times.append(float(t1 - t0))

        print(f"Epoch {epoch+1} | avg loss: {avg:.4f}")
        print("Sample:", sample)
        print(f"Epoch time: {epoch_times[-1]:.3f}s")

    out_path = save_run_log(
        out_path="runs/run_scratch_numpy.json",
        run_name="scratch_numpy",
        epoch_losses=epoch_losses,
        epoch_times=epoch_times,
        samples=samples,
        config={
            "seq_len": seq_len,
            "hidden": rnn.H,
            "lr": rnn.lr,
            "vocab_size": rnn.V,
            "epochs": 50,
        },
        notes="Scratch RNN (NumPy) run log",
        sample_for_quality="last",
    )
    print(f"Saved run log to: {out_path}")

    return out_path


if __name__ == "__main__":
    rnn_main()
