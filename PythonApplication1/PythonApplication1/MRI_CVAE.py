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
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
import nibabel as nib #reading MR images
from sklearn.model_selection import train_test_split
# load in MNIST
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # 10 * 784 * 60,000
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # 10 * 784 * 10,000
# convert y to onehot
#plot_labels_test = y_test
#plot_labels_train = y_train
#y_train = to_categorical(y_train) # tuple 10,000 * 10
#y_test = to_categorical(y_test) # tuple 10,000 * 10

filepath_df = pd.read_csv('Z:/PRONIA_data/Tables/pronia_full_niftis.csv')

num_subjs = 698 # max 698
mri_types =['wp0', # whole brain
            'wp1', # gm
            'wp2', # wm
            'mwp2', # modulated? wm
            'rp1', # gm mask
            'rp2'] # wm mask

niis = []
#'wp0', # whole brain?
#'wp1', # gm
#'wp2', # wm
#'mwp2', # modulated? wm
#'rp1', # gm mask
#'rp2', # wm mask
labels = []
for i in range(100):
    row = filepath_df.iloc[i]
    nii_path = row['wp0']
    nii = nib.load(nii_path)
    nii = nii.get_fdata()
    nii = nii[:, 78:79, :] # was 78:129
    labels.append(row['SEX_T0'])
    niis.append(nii)
    #for j in range(nii.shape[1]):  # To get only one layer remove this loop and take only one slice
    #    niis.append((nii[:,j,:]))
    #    labels.append(row['STUDYGROUP'])




print(nii.shape) # 121, 51, 121
#nii[:,0,:].shape

# Loading in the labels
metadata = pd.read_csv('Z:/PRONIA_data/Tables/pronia_full_niftis.csv')
print(len(labels))

# Preprocessing
images = np.asarray(niis)
print(images.shape)

#from CVAEplots import plot_data
#plot_data(images)

images = images.reshape(-1, 121, 121, 1)
#print(images.shape) # 510,121,121,1


x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size=0.2, random_state=13) # x_train 408, 121, 121
x_train = x_train.astype('float32') / 255 # 408, 121, 121
x_test = x_test.astype('float32') / 255 # 102, 121, 121

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # 408, 14641
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # 102, 14641

y_train = to_categorical(y_train) # tuple 10,000 * 10
y_test = to_categorical(y_test) # tuple 10,000 * 10
# shape 530*121*121
print(y_test.shape, x_test.shape)

## plotting data
#from CVAEplots import plot_data
#plot_data(x_test)

## reshape to matrix in able to feed into network
#images = images.reshape(-1, 121, 121, 1) # 510, 121, 121, 1
## min-max normalisation to rescale (why?)
#m = np.max(images)
#mi = np.min(images)
#print(m ,mi) # 2.99, 0 
#images = (images - mi) / (m - mi) # 0, 1
## Pad images iwth zeros at boundaries so the dimenson is even and easier to downsample images by two while passing through model. Add in three rows and columns to make dim 176*176
#temp = np.zeros([510,124,124,1])
#temp[:,3:,3:,:] = images
#images = temp # dim now 510 *124*124*1
## test train split (no labels for standard autoencoder)


## Data exploration ##
# training shape
epochs = 50
origin_dim = 121 * 121
batch_size = 128
intermediate_dim = 64
latent_dim = 2
n_y = y_train.shape[1] # 10
n_x = x_train.shape[1] # 784
print(n_y, n_x)
n_z = 2

# Make a sampling layer, this maps the MNIST digit to latent-space triplet (z_mean, z_log_var, z), this is how the bottleneck is displayed. 
from keras import backend as K
def sampling(args):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean = 0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

# Set inputs
x = keras.Input(shape=(origin_dim, ))
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
latent_inputs = keras.Input(shape=(5), name = 'z_sampling')
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
#from CVAEplots import plot_clusters
#plot_clusters(encoder, x_test, y_test, plot_labels_test, batch_size)

from CVAEplots import reconstruction_plot
reconstruction_plot(x_test, y_test, cvae)

#from CVAEplots import lossplot
#lossplot(history)

#from CVAEplots import plot_latent_space
#plot_latent_space(1, 1.5, 8, decoder)

#from CVAEplots import latent_space_traversal
#latent_space_traversal(10, 1.5, decoder)

#from CVAEplots import plot_y_axis_change, plot_x_axis_change
#plot_y_axis_change(1, 10, 1.5, decoder)
#plot_x_axis_change(1, 10, 1.5, decoder)

# Plotting digits as wrong labels 
# setting fake label
label = np.repeat(2, len(y_test))
label_fake = to_categorical(label, num_classes=2)
reconstruction_plot(x_test, label_fake, cvae)
###############################################################






#for i in range(n_z+n_y):
#	tmp = np.zeros((1,n_z+n_y))
#	tmp[0,i] = 1
#	generated = decoder.predict(tmp)
#	file_name = './img' + str(i) + '.jpg'
#	print(generated)
#	imsave(file_name, generated.reshape((28,28)))
#	sleep(0.5)

# this loop prints a transition through the number line

#pic_num = 0
#variations = 30 # rate of change; higher is slower
#for j in range(n_z, n_z + n_y - 1):
#	for k in range(variations):
#		v = np.zeros((1, n_z+n_y))
#		v[0, j] = 1 - (k/variations)
#		v[0, j+1] = (k/variations)
#		generated = decoder.predict(v)
#		pic_idx = j - n_z + (k/variations)
#		file_name = './transition_50/img{0:.3f}.jpg'.format(pic_idx)
#		imsave(file_name, generated.reshape((28,28)))
#		pic_num += 1
		
