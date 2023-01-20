import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import csv

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim, num_samples = 16, name = 'encoder'):
        super(Encoder, self).__init__(name = name)
        self.latent_dim = latent_dim
        self.num_samples = num_samples

        self.conv1 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, strides=3)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=7, strides=3)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.conv3 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=7, strides=3)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation('relu')

        self.dense1 = tf.keras.layers.Dense(512)
        # self.dense_norm1 = tf.keras.layers.BatchNormalization()
        # self.dense_act1 = tf.keras.layers.Activation('relu')

        self.z_mean_dense = tf.keras.layers.Dense(self.latent_dim)
        self.z_log_var_dense = tf.keras.layers.Dense(self.latent_dim)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = tf.keras.layers.Flatten()(x)

        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)

        self.dist = tfp.distributions.Normal(
            loc = z_mean, scale = tf.exp(0.5*z_log_var))
        sampled = self.dist.sample(self.num_samples)
        z = tf.transpose(sampled, [1, 0, 2])
        return z, z_mean, z_log_var

class Decoder(tf.keras.Model):

    def __init__(self, charset_length, max_length, name = 'decoder'):
        super(Decoder, self).__init__(name = name)
        self.charset_length = charset_length
        self.max_length = max_length
        self.dense1 = tf.keras.layers.Dense(512)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.repeat = tf.keras.layers.RepeatVector(self.max_length)
        self.gru1 = tf.keras.layers.GRU(512, return_sequences=True)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.gru2 = tf.keras.layers.GRU(512, return_sequences=True)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.gru3 = tf.keras.layers.GRU(512, return_sequences=True)
        self.out_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.charset_length))
        self.out_act = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.repeat(x)
        x = self.gru1(x)
        x = self.norm2(x)
        x = self.gru2(x)
        x = self.norm3(x)
        x = self.gru3(x)
        outputs_logits = self.out_dense(x)
        outputs = self.out_act(outputs_logits)
        return outputs, outputs_logits

class VAE(tf.keras.Model):

    def __init__(self, latent_dim, charset_length, max_length, 
        num_samples = 16, name = 'vae'):
        super(VAE, self).__init__(name = name)
        self.latent_dim = latent_dim
        self.charset_length = charset_length
        self.max_length = max_length
        self.num_samples = num_samples

        self.encoder = Encoder(
            self.latent_dim, num_samples = self.num_samples)
        self.decoder = Decoder(charset_length, max_length)

    def call(self, inputs):
        z, self.z_mean, self.z_log_var = self.encoder(inputs)
        z_reshaped = tf.reshape(z, (-1, self.encoder.latent_dim))
        outputs, self.outputs_logits = self.decoder(z_reshaped)
        return outputs

    def loss_function(self, y_true, y_pred):
        latent_loss = -0.5*tf.reduce_sum(1.0 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), 1)
        y_true_r = tf.reshape(y_true, [-1, 1, self.max_length])
        y_true_c = tf.cast(y_true_r, tf.int64)
        tiled = tf.tile(y_true_c, (1, self.num_samples, 1))
        y_true_rep = tf.reshape(tiled, (-1, self.max_length))
        # recon_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     y_true_rep, self.outputs_logits,
        #     reduction = tf.keras.losses.Reduction.SUM)
        # recon_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     y_true_rep, self.outputs_logits)
        recon_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(y_true_rep, self.outputs_logits, reduction=tf.compat.v1.losses.Reduction.SUM)
        recon_loss = recon_loss/tf.cast(self.num_samples, tf.float32)
        vae_loss = latent_loss + recon_loss
        return vae_loss

    def accuracy(self, y_true, y_pred):
        y_true_r = tf.reshape(y_true, [-1, 1, self.max_length])
        y_true_c = tf.cast(y_true_r, tf.int64)
        tiled = tf.tile(y_true_c, (1, self.num_samples, 1))
        y_true_rep = tf.reshape(tiled, (-1, self.max_length))
        y_pred_class = tf.argmax(y_pred, axis = -1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            y_true_rep, y_pred_class), tf.float32))
        return accuracy

## Auxiliary classes

class DataBatch(tf.keras.utils.Sequence):

    def __init__(self, x, num_classes, batch_size = 32):
        self.x = x
        self.num_classes = num_classes
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_one_hot = tf.keras.utils.to_categorical(batch_x, num_classes=self.num_classes)
        return (batch_x_one_hot, batch_x)

class SaveMetrics(tf.keras.callbacks.Callback):
    def __init__(self, metrics_file):
        self.metrics_file = metrics_file
        with open(self.metrics_file, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 
                'val_accuracy'])
    
    def on_epoch_end(self, epoch, logs={}):
        with open(self.metrics_file, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch, logs.get('loss'),
                logs.get('accuracy'), logs.get('val_loss'), 
                logs.get('val_accuracy')])




# class VAE(tf.keras.Model):
#     def __init__(self, latent_dim):
#         super(VAE, self).__init__()
#         self.latent_dim = latent_dim
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.InputLayer(input_shape=(784,)),
#             tf.keras.layers.Dense(512, activation='relu'),
#             tf.keras.layers.Dense(256, activation='relu'),
#             tf.keras.layers.Dense(latent_dim + latent_dim),
#         ])
#         self.decoder = tf.keras.Sequential([
#             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#             tf.keras.layers.Dense(256, activation='relu'),
#             tf.keras.layers.Dense(512, activation='relu'),
#             tf.keras.layers.Dense(784, activation='sigmoid'),
#         ])

#     def call(self, inputs):
#         x = self.encoder(inputs)
#         mean, logvar = tf.split(x, num_or_size_splits=2, axis=-1)
#         stddev = tf.exp(0.5*logvar)
#         epsilon = tf.random.normal(shape=stddev.shape)
#         z = mean + stddev * epsilon
#         reconstruction = self.decoder(z)
#         return reconstruction, mean, logvar
