import tensorflow as tf
# import tensorboard as tb
from dvclive.keras import DVCLiveCallback
import vae as vae
import numpy as np
import json
import os
import yaml
import datetime

def main():
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.makedirs("model", exist_ok=True)
    os.makedirs("metrics/train", exist_ok=True)
    os.makedirs("logs/train", exist_ok=True)

    with open("params.yaml", 'r') as stream:
        params = yaml.safe_load(stream)

    x_train = np.load("data/tokenized/x_train.npy")
    x_train = x_train[1:100,:,:]
    x_validation =np.load("data/tokenized/x_validation.npy")

    # x_train = tf.data.TFRecordDataset("data/x_train.tfdata")
    # x_validation = tf.data.TFRecordDataset("data/x_validation.tfdata")
    # x_test = tf.data.TFRecordDataset("data/x_test.tfdata")

    with open('data/tokenized/char_to_int.json', 'r') as f:
        char_to_int = json.load(f)
    num_classes = len(char_to_int)

    x_train_batched = vae.DataBatch(x_train,
        num_classes, batch_size = params["batch_size"])
    x_validation_batched = vae.DataBatch(x_validation,
        num_classes, batch_size = params["batch_size"])

    model = vae.VAE(params["latent_dim"], num_classes,
        params["max_len_smiles"], params["num_samples"])
    optimizer = tf.optimizers.Adam(
        learning_rate = params["learning_rate"], clipvalue = 0.1)
    model.compile(optimizer = optimizer, 
        loss = model.loss_function, metrics = [model.accuracy])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "model/" + 'weights-{epoch:02d}-{val_loss:.4f}.ckpt',
        monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode = 'auto', save_weights_only = True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1,
        min_lr = 0.0)
    save_metrics = vae.SaveMetrics(
        "metrics/training/loss.csv", "metrics/training/accuracy.csv")
    log_dir = "logs/train/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    history = model.fit( x = x_train_batched,
        validation_data = x_validation_batched,
        epochs = params["num_epochs"], use_multiprocessing = True,
        workers = params["num_workers"],
        callbacks = [checkpoint, reduce_lr,
            DVCLiveCallback(save_dvc_exp = True)])
    # history = model.fit( x = x_train_batched,
    #     validation_data = x_validation_batched,
    #     epochs = params["num_epochs"], use_multiprocessing = True,
    #     workers = params["num_workers"],
    #     callbacks = [checkpoint, reduce_lr, save_metrics,
    #         tensorboard_cb])
    # print(history.history)
    with open("metrics/training/metrics.json", "w") as file:
        json.dump(history.history, file)

if __name__ == "__main__":
    main()

# def compute_loss(reconstruction, original, mean, logvar):
#     reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(original, reconstruction))
#     kl_loss = -0.5 * tf.reduce_mean(tf.exp(logvar) + tf.square(mean) - 1 - logvar)
#     return reconstruction_loss + kl_loss

# @tf.function
# def train_step(inputs):
#     with tf.GradientTape() as tape:
#         reconstruction, mean, logvar = model(inputs)
#         loss = compute_loss(reconstruction, inputs, mean, logvar)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

# Training loop
# for epoch in range(num_epochs):
#     for inputs in dataset:
#         loss = train_step(inputs)
#     print("Epoch {}, Loss: {}".format(epoch, loss))