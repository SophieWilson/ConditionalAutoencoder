import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


from keras.utils import to_categorical
from keras.layers.merge import concatenate
# load in MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # 10 * 784 * 60,000
# convert y to onehot
plot_labels_test = y_test
plot_labels_train = y_train
y_train = to_categorical(y_train) # tuple 10,000 * 10
y_test = to_categorical(y_test) # tuple 10,000 * 10
epochs = 101
origin_dim = 28 * 28 # 78
batch_size = 128
intermediate_dim = 64
latent_dim = 2
n_y = y_train.shape[1] # 10
n_x = x_train.shape[1] # 784
print(n_y, n_x)
n_z = 50

# Make a sampling layer, this maps the MNIST digit to latent-space triplet (z_mean, z_log_var, z), this is how the bottleneck is displayed. 
from keras import backend as K
def sampling(args):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean = 0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

# Set inputs
x = keras.Input(shape=(784, ))
label = keras.Input(shape=(n_y, ))
encoder_inputs = concatenate([x, label])
#encoder_inputs = x
## Make the encoder
# outputs, as this is variational you have two outputs, the mean and the sigma of the latent dimension, so it takes a sample from this distribtion to run through back propagation. As you cant back propagation from a sample distribution epsilon is added to z to allow it to be run through the decoder. This is what the sampling funciton does.
h = layers.Dense(512, activation='relu')(encoder_inputs)
h = layers.Dense(128, activation='relu')(h)
#h = layers.Dense(64, activation='relu')(h)
h = layers.Dense(intermediate_dim, activation='relu')(h)
z_mean = layers.Dense(latent_dim, name="z_mean")(h)
z_log_sigma = layers.Dense(latent_dim, name="z_log_sigma")(h)

z = layers.Lambda(sampling)([z_mean, z_log_sigma])
z_label = concatenate([z, label])
# initiating the encoder, it ouputs the latent dim dimensions
encoder = keras.Model([x, label], [z_mean, z_log_sigma, z_label], name="encoder")
encoder.summary()

### Make the decoder, takes the latent input to output the image
# the only input to decoder is z_label
latent_inputs = keras.Input(shape=(12), name = 'z_sampling')
dec_x = layers.Dense(512, activation='relu')(latent_inputs)
dec_x = layers.Dense(128, activation='relu')(dec_x)
#dec_x = layers.Dense(64, activation='relu')(dec_x)
dec_x = layers.Dense(intermediate_dim, activation='relu')(dec_x)
decoder_outputs =  layers.Dense(origin_dim, activation='sigmoid')(dec_x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# instantiate the VAE model (simplified)
#print(encoder(encoder_inputs))
#decoder takes only z input from encoder
outputs = decoder(encoder([x, label])[2])
cvae = keras.Model([x, label], outputs, name='vae')


# use a custom loss function, this includes a KL divergence regularisation term which ensures that z is close to normal (0 mean, 1 sd)

reconstruction_loss = keras.losses.binary_crossentropy(x, outputs)
reconstruction_loss *= origin_dim

kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
# mean was worse
vae_loss = reconstruction_loss + kl_loss
# does this just show up as loss?
cvae.add_loss(vae_loss)
cvae.compile(optimizer='adam',)
# can add more metrics below with this (i'll try it out later)
#model.metrics_tensors.append(kl_loss)
#model.metrics_names.append("kl_loss")
#model.metrics_tensors.append(reconstruction_loss)
#model.metrics_names.append("mse_loss")

# Tensorboard
from keras.callbacks import TensorBoard
import datetime
log_dir = "C:/Users/Mischa/sophie/logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Adding early stopping
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

# fit the data to MNIST
history = cvae.fit([x_train, y_train], x_train, epochs=epochs, batch_size=batch_size, validation_data = ([x_test, y_test], x_test), verbose = 2, callbacks=[tensorboard_callback, es_callback])
print(history.history.keys())
## Plots
import matplotlib.pyplot as plt
# Display how the latent space clusters the digit classesimport matplotlib.pyplot as plt
def plot_clusters(encoder, data, labels, batch_size):
    x_test_encoded, _, _ = encoder.predict([x_test, y_test], batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

plot_clusters(encoder, [x_test, y_test], plot_labels_test, batch_size)

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

#digit_grid(decoder)

# Plotting reconstruction vs actual
def reconstruction_plot(data, vae, n=10):
    prediction = vae.predict(data)
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
    plt.show()

reconstruction_plot([x_test, y_test], cvae)

# loss plot
def lossplot(history):
    loss_values = history.history['loss']
    val_loss = history.history['val_loss']
    #reconstruction_loss = history.history['reconstruction_loss']
    #kl_loss = history.history['kl_loss']
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss, label = 'Val_loss')
    #plt.plot(epochs, kl_loss, label='KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

lossplot(history)

