import tensorflow as tf
import numpy as np
import yaml
import json
import vae as vae
import os

def main():

    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)

    os.makedirs("metrics/evaluation", exist_ok=True)

    with open('data/tokenized/char_to_int.json', 'r') as f:
        char_to_int = json.load(f)
    num_classes = len(char_to_int)

    x_test =np.load("data/tokenized/x_test.npy")

    model = vae.VAE(params["latent_dim"], num_classes,
        params["max_len_smiles"], params["num_samples"])

    x_test_batched = vae.DataBatch(x_test,
        num_classes, batch_size = params["batch_size"])

    model.load_weights("model")
    scores = model.evaluate(x_test_batched)
    with open("metrics/scores.json", 'w') as f:
        json.dump({'test_loss': scores[1], 'test_accuracy': scores[2]}, f)

if __name__ == "__main__":
    main()