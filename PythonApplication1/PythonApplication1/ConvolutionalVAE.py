import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


epochs = 20
origin_dim = 28 * 28
batch_size = 128
#intermediate_dim = 64
latent_dim = 2 

# Make a sampling layer, this maps the MNIST digit to latent-space triplet (z_mean, z_log_var, z), this is how the bottleneck is displayed. 
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


## Make the encoder, input > cov2D > flatten > dense (pretty sure this is doing nothing right now)
# outputs, as this is variational you have two outputs, the mean and the sigma of the latent dimension, so it takes a sample from this distribtion to run through back propagation. As you cant back propagation from a sample distribution epsilon is added to z to allow it to be run through the decoder. This is what the sampling funciton does. (WHY RUN THROUGH LAMBDA)

latent_dim = 2


encoder_inputs = keras.Input(shape=(28, 28, 1))

# This is where working thing starts! (below)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #14 x 14 x 32 output
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_sigma = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_sigma])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_sigma, z], name="encoder")
encoder.summary()

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
## Working decoder below
x = layers.Dense(7 * 7 * 32, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 32))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

outputs = decoder(encoder(encoder_inputs)[2])
print(outputs)

vae = keras.Model(encoder_inputs, outputs, name='vae')

reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
reconstruction_loss *= origin_dim
reconstruction_loss = K.mean(reconstruction_loss)


#reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(encoder_inputs, outputs), axis= (1,2)))
kl_loss = -0.5 * (1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis = 1))
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer=keras.optimizers.Adam())

# train VAE on MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# reshaping the data (google why this way)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# then changing the data type and making pixel values between 1 and 0
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print(x_train.shape, x_test.shape)                 
#vae = VAE(encoder, decoder)
#vae_output = decoder(encoder(encoder_inputs)[2])
#vae  = keras.Model(encoder_inputs, vae_output, name='vae')
#vae.summary()
#vae.compile(optimizer=keras.optimizers.Adam())
# Tensorboard
from keras.callbacks import TensorBoard
import datetime
log_dir = "C:/Users/Mischa/sophie/logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)
# Adding early stopping
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)


history = vae.fit(x_train, x_train, batch_size= batch_size, epochs=epochs, verbose = 2, callbacks=[tensorboard_callback, es_callback], validation_data=(x_test,x_test))
# .fit function returns history object with loss metrics
print(history.history.keys())


# plot latent space
def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

#plot_latent_space(vae)

# loss plot
def lossplot(history):
    loss_values = history.history['loss']
    reconstruction_loss = history.history['reconstruction_loss']
    kl_loss = history.history['kl_loss']
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, reconstruction_loss, label='Reconstruction Loss')
    #plt.plot(epochs, kl_loss, label='KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#lossplot(history)

#loss_values = history.history['accuracy']
#epochs = range(1, len(loss_values)+1)
#plt.plot(epochs, loss_values, label='Training Accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.show()

# Display how the latent space clusters the digit classes

def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.show()


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(encoder, x_train, y_train)


# Plotting reconstruction vs actual
def reconstruction_plot(x_test, encoder, decoder, n=10):

    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        z = encoder.predict(x_test)[2]
        predictions = decoder.predict(z)
        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(predictions[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
   # plt.show()


#from CVAEplots import reconstruction_plot
reconstruction_plot(x_test, encoder, decoder)

plt.show()
