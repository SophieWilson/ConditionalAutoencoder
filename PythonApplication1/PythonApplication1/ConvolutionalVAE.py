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


epochs = 5
#origin_dim = 28 * 28
batch_size = 128
#intermediate_dim = 64
latent_dim = 2 

# Make a sampling layer, this maps the MNIST digit to latent-space triplet (z_mean, z_log_var, z), this is how the bottleneck is displayed. 

class sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_sigma = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean = 0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon


## Make the encoder, input > cov2D > flatten > dense (pretty sure this is doing nothing right now)
# outputs, as this is variational you have two outputs, the mean and the sigma of the latent dimension, so it takes a sample from this distribtion to run through back propagation. As you cant back propagation from a sample distribution epsilon is added to z to allow it to be run through the decoder. This is what the sampling funciton does. (WHY RUN THROUGH LAMBDA)

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
#x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
#x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
#x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', name = 'encoded')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_sigma = layers.Dense(latent_dim, name="z_log_sigma")(x)
z = sampling()([z_mean, z_log_sigma])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_sigma, z], name="encoder")
encoder.summary()


decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
decoder.summary()
#decoder_inputs = keras.Input(shape=(16,))
#self.dense1 = layers.Dense(7 * 7 * 64, activation="relu")
#self.reshape1 = layers.Reshape((7, 7, 64))
#self.cov1T = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
#self.cov2T = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
##x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
#self.outdec = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same') # 28 x 28 x 1

# Make the VAE class

#class VAE(keras.Model):
#    ''' use a custom loss function, this includes a KL divergence regularisation term which ensures that z is close to normal (0 mean, 1 sd) '''
#    def __init__(self, encoder, decoder, **kwargs):
#        super(VAE, self).__init__(**kwargs)
#        self.encoder = encoder
#        self.decoder = decoder

#        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
#        self.reconstruction_loss_tracker = keras.metrics.Mean(
#            name="reconstruction_loss"
#        )
#        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

#    @property
#    def metrics(self):
#        return [
#            self.total_loss_tracker,
#            self.reconstruction_loss_tracker,
#            self.kl_loss_tracker,
#        ]

    #def train_step(self, x_train):
    #    with tf.GradientTape() as tape:
    #        z_mean, z_log_var, z = self.encoder(x_train)
    #        reconstruction = self.decoder(z)
    #        reconstruction_loss = tf.reduce_mean(
    #            tf.reduce_sum(
    #                keras.losses.binary_crossentropy(x_train, reconstruction), axis=(1, 2)
    #            )
    #        )
    #        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #        total_loss = reconstruction_loss + kl_loss
    #    grads = tape.gradient(total_loss, self.trainable_weights)
    #    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #    self.total_loss_tracker.update_state(total_loss)
    #    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    #    self.kl_loss_tracker.update_state(kl_loss)
    #    return {
    #        "loss": self.total_loss_tracker.result(),
    #        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
    #        "kl_loss": self.kl_loss_tracker.result(),
    #    }



# train VAE on MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, outputs, name='vae')
vae.compile(optimizer=keras.optimizers.Adam())
history = vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)
# .fit function returns history object with loss metrics
print(history.history.keys())

# fit the data
vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data = (x_test, x_test))

#encoder.compile(optimizer=keras.optimizers.Adam())
#encoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data = (x_test, x_test))
#encoder.predict(x_train)

# fit the data





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
    #plt.show()

lossplot(history)
#loss_values = history.history['accuracy']
#epochs = range(1, len(loss_values)+1)
#plt.plot(epochs, loss_values, label='Training Accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.show()

# Display how the latent space clusters the digit classes

def plot_clusters(encoder, data, labels, batch_size):
    x_test_encoded, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.show()

plot_clusters(encoder, x_test, y_test, batch_size)

# Display a 2D grid of the digits
def digit_grid(decoder, n=30, figsize=15):
    n = 15  # figure with 15x15 digits
    scale = 1.0
    digit_size = 28 # size of digits
    figure = np.zeros((digit_size * n, digit_size * n))
    # We will sample n points within [-15, 15] standard deviations
    #linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_x): 
        for j, xi in enumerate(grid_y): # cycling through grid spots
            z_sample = np.array([[xi, yi]]) # sampling from space
            x_decoded = decoder.predict(z_sample) # taking prediction from that latent space
            digit = x_decoded[0].reshape(digit_size, digit_size) # reshaping to plot
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    
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
    #plt.show()

digit_grid(decoder)

# Plotting reconstruction vs actual
def reconstruction_plot(x_test, vae, n=10):
    prediction = vae.predict(x_test)
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(prediction[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
    #plt.show()

reconstruction_plot(x_test, vae)
plt.show()
