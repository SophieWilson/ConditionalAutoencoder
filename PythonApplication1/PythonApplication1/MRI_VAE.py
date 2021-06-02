# THIS IS CONVOLUTIONAL
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

import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import math
import glob
from matplotlib import pyplot as plt
import pandas as pd

# Import and preprocess data

filepath_df = pd.read_csv('Z:/PRONIA_data/Tables/pronia_full_niftis.csv')
mri_types =['wp0', # whole brain
            'wp1', # gm
            'wp2', # wm
            'mwp2', # modulated? wm
            'rp1', # gm mask
            'rp2'] # wm mask

num_subjs = 100 # max 698
niis = []

import nibabel as nib #reading MR images
for i in range(num_subjs):
    row = filepath_df.iloc[i]
    nii_path = row['wp0']
    nii = nib.load(nii_path)
    nii = nii.get_fdata()
    nii = nii[:, 78:80, :]
    for j in range(nii.shape[1]):
        niis.append((nii[:,j,:]))


## Set autoencoder variables
epochs = 200
origin_dim = 28 * 28
#intermediate_dim = 64
latent_dim = 200 # compressing image to something
batch_size = 16
inchannel = 1
X, y = 124, 124
depth = len(nii[2]) # the number of slices set above

## Preprocessing

images = np.asarray(niis) # shape num_subjs*121*121
# reshape to matrix in able to feed into network
images = images.reshape(-1, depth, 121, 121) # num_subjs,depth,121,121,
### min-max normalisation to rescale between 1 and 0 to improve accuracy
m = np.max(images)
mi = np.min(images) 
images = (images - mi) / (m - mi)
## Pad images with zeros at boundaries so the dimenson is even and easier to downsample images by two while passing through model. Add in three rows and columns to make dim 176*176
temp = np.zeros([num_subjs, depth,124,124])
temp[:,:,3:,3:,] = images
print(images.shape)
images = temp # dim now 5*2*124*124 # could replace temp with images 

# test train split (no labels for vae)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(images, images, test_size=0.2, random_state=13)


# Make a sampling layer, this maps the MNIST digit to latent-space triplet (z_mean, z_log_var, z), this is how the bottleneck is displayed. 
from keras import backend as K
def sampling(args):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean = 0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

## Make the encoder
# outputs, as this is variational you have two outputs, the mean and the sigma of the latent dimension, so it takes a sample from this distribtion to run through back propagation. As you cant back propagation from a sample distribution epsilon is added to z to allow it to be run through the decoder. This is what the sampling function does. (WHY RUN THROUGH LAMBDA)
from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Conv3DTranspose

encoder_inputs = keras.Input(shape=(depth, X, y, inchannel)) # it will add a None layer as batch size
print(encoder_inputs.shape)
x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x) 
x = layers.Conv3D(64, (3, 3, 3), activation="relu",  padding="same")(x)
#x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x) 
x = layers.Conv3D(32, (3, 3, 3), activation="relu",  padding="same")(x)
#x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x) 
x = layers.Conv3D(16, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.Flatten()(x)
#x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_sigma = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling)([z_mean, z_log_sigma])
## initiating the encoder, it ouputs the latent dim dimensions
encoder = keras.Model(encoder_inputs, [z_mean, z_log_sigma, z], name="encoder")
encoder.summary()

#### Make the decoder, takes the latent keras
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(latent_dim, activation='relu')(latent_inputs)
x =  layers.Dense(62*62*16, activation='relu')(x)
x = layers.Reshape((1, 62, 62, 16))(x)
x = layers.Conv3DTranspose(16, (3, 3, 3), activation="relu", strides=2, padding="same")(x)
x = layers.Conv3DTranspose(32, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.Conv3DTranspose(64, (3, 3, 3), activation="relu",  padding="same")(x)
decoder_outputs = layers.Conv3DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# Initiate decoder
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# Instantiate and fit VAE model 
outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, outputs, name='vae')

# Custom loss function, this includes a KL divergence regularisation term which ensures that z is close to normal (0 mean, 1 sd)
reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
reconstruction_loss *= origin_dim
reconstruction_loss = K.mean(reconstruction_loss) # mean to avoid incompatible shape error
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=keras.optimizers.Adam())

# Tensorboard
from keras.callbacks import TensorBoard
import datetime
log_dir = "C:/Users/Mischa/sophie/MRI_VAE" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)
# Early stopping
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# fit the data
history = vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data = (x_test, x_test), verbose=2, callbacks=[tensorboard_callback, es_callback])

## PLOT

# Display how the latent space clusters the digit classes
import matplotlib.pyplot as plt
#def plot_clusters(encoder, data, labels, batch_size):
#    x_test_encoded, _, _ = encoder.predict(x_test, batch_size=batch_size)
#    plt.figure(figsize=(6, 6))
#    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#    plt.colorbar()
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    #plt.show()

#plot_clusters(encoder, x_test, y_test, batch_size)

# Display a 2D grid of the digits
#def digit_grid(decoder, n=30, figsize=15):
#    n = 15  # figure with 15x15 digits
#    scale = 1.0
#    digit_size = 28 # size of digits
#    figure = np.zeros((digit_size * n, digit_size * n))
#    # We will sample n points within [-15, 15] standard deviations
#    #linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
#    grid_x = np.linspace(-scale, scale, n)
#    grid_y = np.linspace(-scale, scale, n)[::-1]

#    for i, yi in enumerate(grid_x): 
#        for j, xi in enumerate(grid_y): # cycling through grid spots
#            z_sample = np.array([[xi, yi]]) # sampling from space
#            x_decoded = decoder.predict(z_sample) # taking prediction from that latent space
#            digit = x_decoded[0].reshape(digit_size, digit_size) # reshaping to plot
#            figure[i * digit_size: (i + 1) * digit_size,
#                   j * digit_size: (j + 1) * digit_size] = digit
    
#    plt.figure(figsize=(figsize, figsize))
#    start_range = digit_size // 2
#    end_range = n * digit_size + start_range
#    pixel_range = np.arange(start_range, end_range, digit_size)
#    sample_range_x = np.round(grid_x, 1)
#    sample_range_y = np.round(grid_y, 1)
#    plt.xticks(pixel_range, sample_range_x)
#    plt.yticks(pixel_range, sample_range_y)
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.imshow(figure, cmap="Greys_r")
#    #plt.show()

#digit_grid(decoder)

# Plotting reconstruction vs actual
def reconstruction_plot(x_test, vae, n=10):
    prediction = vae.predict(x_test)
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i][0]) # x_test[1][1] is shape 124,124
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(prediction[i][0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
    #plt.show()

reconstruction_plot(x_test, vae)
plt.show()


def lossplot(history):
    ''' Plotting loss as a line graph, history is the variable saved in model.fit() '''
    loss_values = history.history['loss']
    val_loss = history.history['val_loss']
    #reconstruction_loss = history.history['reconstruction_loss']
    #kl_loss = history.history['kl_loss']
    epochs = range(1, len(loss_values)+1)
    fig = plt.figure()
    fig.suptitle('Training loss plot', fontsize=10)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss, label = 'Val_loss')
    #plt.plot(epochs, kl_loss, label='KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#lossplot(history)

