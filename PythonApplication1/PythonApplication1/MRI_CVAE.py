
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

filepath_df = pd.read_csv('Z:/PRONIA_data/Tables/pronia_full_niftis.csv')

mri_types =['wp0', # whole brain
            'wp1', # gm
            'wp2', # wm
            'mwp2', # modulated? wm
            'rp1', # gm mask
            'rp2'] # wm mask

niis = []
num_subjs = 500 # max 698
labels = []

for i in range(num_subjs):
    row = filepath_df.iloc[i]
    nii_path = row['wp0']
    nii = nib.load(nii_path)
    nii = nii.get_fdata()
    nii = nii[:, 78:88, :]
    labels.append(row['STUDYGROUP'])
    for j in range(nii.shape[1]):
        niis.append((nii[:,j,:]))
        

depth = len(nii[2])

images = np.asarray(niis) # shape num_subjs*121*121
# reshape to matrix in able to feed into network
print('55', images.shape)
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
x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size=0.2, random_state=13, stratify = labels)
y_train = to_categorical(y_train) # tuple num_patients * num_labels
y_test = to_categorical(y_test) # tuple num_patients * num_labels
print(y_test.shape, y_train.shape)

#y_test = np.expand_dims(y_test, 2)
 # Autoencoder variables
epochs = 500
batch_size = 8
intermediate_dim = 124
latent_dim = 100
n_y = y_train.shape[1] # 2
n_x = x_train.shape[1] # 784
n_z = 2
X, y = 124, 124 
inchannel = 1
origin_dim = 28*28

# Make a sampling layer, this maps the MNIST digit to latent-space triplet (z_mean, z_log_var, z), this is how the bottleneck is displayed. 
from keras import backend as K
def sampling(args):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean = 0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

# Set inputs
from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Conv3DTranspose

label = keras.Input(shape=(n_y, ))
encoder_inputs = keras.Input(shape=(depth, X, y, inchannel)) # it will add a None layer as batch size

x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x) 
x = layers.Conv3D(64, (3, 3, 3), activation="relu",  padding="same")(x)
#x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x) 
x = layers.Conv3D(32, (3, 3, 3), activation="relu",  padding="same")(x)
#x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x) 
x = layers.Conv3D(16, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.Conv3D(8, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.Flatten()(x)
#x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_sigma = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling)([z_mean, z_log_sigma])
z_label = concatenate([z, label])
## initiating the encoder, it ouputs the latent dim dimensions
encoder = keras.Model([encoder_inputs, label], [z_mean, z_log_sigma, z_label], name="encoder")
encoder.summary()

#### Make the decoder, takes the latent keras
latent_inputs = keras.Input(shape=(105,))
#x = layers.Dense(4, activation='relu')(latent_inputs)
x =  layers.Dense(5*62*62*8, activation='relu')(latent_inputs)
x = layers.Reshape((5, 62, 62, 8))(x)
x = layers.Conv3DTranspose(8, (3, 3, 3), activation="relu", strides=2, padding="same")(x)
x = layers.Conv3DTranspose(16, (3, 3, 3), activation="relu", padding="same")(x)
x = layers.Conv3DTranspose(32, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.Conv3DTranspose(64, (3, 3, 3), activation="relu",  padding="same")(x)
decoder_outputs = layers.Conv3DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# Initiate decoder
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# Instantiate and fit VAE model 
outputs = decoder(encoder([encoder_inputs, label])[2])
cvae = keras.Model([encoder_inputs, label], outputs, name='cvae')

# use a custom loss function, this includes a KL divergence regularisation term which ensures that z is close to normal (0 mean, 1 sd)

reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
reconstruction_loss *= origin_dim
reconstruction_loss = K.mean(reconstruction_loss) # mean to avoid incompatible shape error
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
# mean was worse
cvae_loss = reconstruction_loss + kl_loss
# does this just show up as loss?
cvae.add_loss(cvae_loss)
cvae.compile(optimizer='adam',)

# Tensorboard
from keras.callbacks import TensorBoard
import datetime
log_dir = "C:/Users/Mischa/sophie/MRI_CVAE" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Adding early stopping
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

# fit the data 
history = cvae.fit([x_train, y_train], x_train, epochs=epochs, batch_size=batch_size, validation_data = ([x_test, y_test], x_test), verbose = 2, callbacks=[tensorboard_callback, es_callback])

## Plots
#from CVAEplots import plot_clusters
#plot_clusters(encoder, x_test, y_test, plot_labels_test, batch_size)

from CVAE_3Dplots import reconstruction_plot
reconstruction_plot(x_test, y_test, cvae, slice=2)

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
label_fake = to_categorical(label, num_classes=3)
reconstruction_plot(x_test, label_fake, cvae, slice= 10)
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
		
