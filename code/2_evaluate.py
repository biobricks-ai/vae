import tensorflow as tf
import numpy as np
import yaml
import json
import vae as vae
import os

def main():

    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)

    os.makedirs("metrics/test", exist_ok=True)

    with open('data/tokenized/char_to_int.json', 'r') as f:
        char_to_int = json.load(f)
    num_classes = len(char_to_int)

    x_test =np.load("data/tokenized/x_test.npy")
    x_test_batched = vae.DataBatch(x_test,
        num_classes, batch_size = params["batch_size"])

    model = vae.VAE(params["latent_dim"], num_classes,
        params["max_len_smiles"], params["num_samples"])
    model.compile(loss = model.loss_function, 
        metrics = [model.accuracy])
    latest = tf.train.latest_checkpoint("model")
    model.load_weights(latest)
    scores = model.evaluate(x_test_batched)
    with open("metrics/test/scores.json", 'w') as f:
        json.dump({'test_loss': scores[0], 
        'test_accuracy': scores[1]}, f)

if __name__ == "__main__":
    main()