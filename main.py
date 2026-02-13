from rnn import rnn_main
from rnn_onehot import rnn_onehot_main
from rnn_embedding import rnn_embedding_main
from utils import compare_runs

import os

if __name__ == "__main__":
    new = input("Run all experiments? (y/n): ").strip().lower()
    if new == "y" or not os.path.exists("runs/run_scratch_numpy.json"):
        print("Running rnn from scratch (NumPy)...")
        result_1 = rnn_main()
    else:
        result_1 = "runs/run_scratch_numpy.json"
    if new == "y" or not os.path.exists("runs/run_onehot.json"):
        print("\nRunning PyTorch RNN with one-hot inputs...")
        result_2 = rnn_onehot_main()
    else:
        result_2 = "runs/run_onehot.json"
    if new == "y" or not os.path.exists("runs/run_embedding.json"):
        print("\nRunning PyTorch RNN with trainable embeddings...")
        result_3 = rnn_embedding_main()
    else:
        result_3 = "runs/run_embedding.json"

    compare_runs([result_1, result_2, result_3])
