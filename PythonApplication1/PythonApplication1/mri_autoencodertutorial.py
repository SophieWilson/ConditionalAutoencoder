  
#import os
#import tensorflow as tf
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
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
import nibabel as nib #reading MR images
from sklearn.model_selection import train_test_split
import math
import glob
from matplotlib import pyplot as plt
import pandas as pd



filepath_df = pd.read_csv('Z:/PRONIA_data/Tables/pronia_full_niftis.csv')

num_subjs = 200 # max 698
mri_types =['wp0', # whole brain
            'wp1', # gm
            'wp2', # wm
            'mwp2', # modulated? wm
            'rp1', # gm mask
            'rp2'] # wm mask


#def get_niis(num_images=num_subjs, mri_type='wp0', verbose=True):
#    import nibabel as nib
#    niis = []
#    for i in range(num_images):
#        if verbose:
#            print(f'Loading image data from {i+1}/{num_images}: ')
#        row = filepath_df.iloc[i]
#        nii_path = row[mri_type]

#        #if not on_bb:
#            #nii_path = nii_path.replace('/rds/', rds_path)

#    niis = np.stack(niis)
#    return niis

#niis = get_niis(10)


niis = []

for i in range(num_subjs):
    row = filepath_df.iloc[i]
    nii_path = row['wp0']
    nii = nib.load(nii_path)
    nii = nii.get_fdata()
    nii = nii[:, 78:90, :]
    for j in range(nii.shape[1]):
        niis.append((nii[:,j,:]))

#print(nii.shape) # 121, 51, 121

nii[:,0,:].shape

# Preprocessing

images = np.asarray(niis)
# shape 530*121*121
#print(images.shape) # 510, 121, 121

# reshape to matrix in able to feed into network
images = images.reshape(-1, 121, 121, 1)
#print(images.shape) # 510,121,121,1

# min-max normalisation to rescale (why?)
m = np.max(images)
mi = np.min(images)
print(m ,mi) # 2.99, 0 
images = (images - mi) / (m - mi)
#print(np.min(images), np.max(images)) # 0, 1

# Pad images iwth zeros at boundaries so the dimenson is even and easier to downsample images by two while passing through model. Add in three rows and columns to make dim 176*176
temp = np.zeros([2400,124,124,1])
temp[:,3:,3:,:] = images
images = temp # dim now 510 *124*124*1

# test train split (no labels for standard autoencoder)
train_X,valid_X,train_ground,valid_ground = train_test_split(images, images, test_size=0.2, random_state=13)

## Data exploration ##
# training shape
print("Dataset (images) shape: {shape}".format(shape=images.shape))

# plot
#plt.figure(figsize=[5,5])
## Display the first image in training data
#plt.subplot(121)
#curr_img = np.reshape(train_X[0], (124,124))
#plt.imshow(curr_img, cmap='gray')
## Display the first image in testing data
#plt.subplot(122)
#curr_img = np.reshape(valid_X[0], (124,124))
#plt.imshow(curr_img, cmap='gray')

# Convolutional autoencoder
batch_size = 64
epochs = 500
inChannel = 1
x, y = 124, 124
input_img = Input(shape = (x, y, inChannel))


def autoencoder(input_img):
    #encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32 output
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32 output
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64 output
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64 output
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same', name="encoded")(pool2) #7 x 7 x 128 (small and thick) output (bottleneck)

    #decoder
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

# Compile
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()

# Train
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(valid_X, valid_ground))

# Plot loss
#history = autoencoder_train

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
    #plt.show()

lossplot(autoencoder_train)

# Predict data
pred = autoencoder.predict(valid_X)


## Plotting reconstruction vs actual
def reconstruction_plot(valid_ground, pred, vae, n=5):
    #prediction = vae.predict(data)
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(valid_ground[i, ..., 0])
        plt.gray()
        ax.set_title('Real')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(pred[i, ..., 0])
        plt.gray()
        ax.set_title('Reconstruction ')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
    plt.show()

reconstruction_plot(valid_ground, pred, autoencoder_train)