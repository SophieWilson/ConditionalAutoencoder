# Importing all of the bits
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import pandas as pd
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
import tensorflow as tf
import seaborn as sns
# configuring tensorflow (not sure what this does but it makes it work),
# configuring the GPU
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.test.is_built_with_cuda()


# loading the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Plotting some images from the dataset
#plt.figure(figsize=[5,5])
# Display the first image in training data
#plt.subplot(121)
curr_img = np.reshape(train_images[0], (28,28))
#curr_lbl = train_labels[0]
#plt.imshow(curr_img, cmap='gray')
#plt.title("(Label: " + str(curr_lbl) + ")")
# Display the first image in testing data
#plt.subplot(122)
curr_img = np.reshape(test_images[0], (28,28))
#curr_lbl = test_labels[0]
#plt.imshow(curr_img, cmap='gray')
#plt.title("(Label: " + str(curr_lbl) + ")")

# Data preprocessing
# reshaping the data (google why this way)
train_images = np.reshape(train_images, (len(train_images), 28, 28, 1))
test_images = np.reshape(test_images, (len(test_images), 28, 28, 1))
# then changing the data type and making pixel values between 1 and 0
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
                           

# splitting the training set into test and train to properly validate the
# autoencoder, the test set is 20% of the imported training set.  For this we
# dont want the labels as its an autoencoder so the ground truths are the
# images.
#from sklearn.model_selection import train_test_split
#train_X,valid_X,train_ground,valid_ground = train_test_split(train_images,
#train_images, test_size=0.2, random_state = 13)

############ Now making the Convolutional autoencoder ######################
#  python -m tensorboard.main --logdir=./change/me/to/a/location

# the input shape is 28,28, 1 (1 as only one colour channel)
input_img = Input(shape = (28, 28, 1))

# Making the encoder in a function
from tensorflow.keras import layers

# Making the encoder in a function
def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32 output
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32 output
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64 output
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64 output
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same', name="encoded")(pool2) #7 x 7 x 128 (small and thick) output (bottleneck)

    #decoder
    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

# now we compile
autoencoder = Model(input_img, autoencoder(input_img))
print(autoencoder.summary())
autoencoder.compile(loss='binary_crossentropy', optimizer = 'adam')


# Fitting the model to the training data
from keras.callbacks import TensorBoard
autoencoder_train = autoencoder.fit(train_images, train_images, batch_size=128,epochs=5,verbose=1,validation_data=(test_images, test_images), callbacks=[TensorBoard(log_dir='C:/Users/Mischa/sophie')])
# making the predicitons
prediction = autoencoder.predict(test_images)

# This will extract all layers outputs from the model
extractor = keras.Model(inputs = autoencoder.inputs, outputs=[layer.output for
layer in autoencoder.layers])
extractor = keras.Model(train_images)


# This will get one named layer from the model
layer = 'encoded'
intermediate_layer_model = keras.Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer('encoded').output)
intermediate_output = intermediate_layer_model.predict(train_images) # intermediate output is 1,7,7,128
len(intermediate_output)

# plotting predictions vs actual
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(prediction[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)   


dr_model = keras.Model(inputs=autoencoder.get_layer('input_1').input, 
                                        outputs=autoencoder.get_layer('encoded').output)
#dr_model.summary()



#x = []
#y = []
#z = []
#print(test_images.shape)
#for i in range(10):
#    z.append(test_labels[i])
#    op = dr_model.predict(np.array([test_images[i]]))
#    x.append(op[0][0])
#    y.append(op[0][1])
 
#df = pd.DataFrame()
#print(df.dtypes)
#df['x'] = x
#print(df.head())
#print(x[1])
#df['y'] = y
#df['z'] = ["digit-" + str(k) for k in z]
#print(df.dtypes)

#plt.figure(figsize=(8, 6))
##sns.scatterplot(x='x', y='y', data=df)
#plt.show()
#print('FIN')

for i in range(1, n+1):
    # display reconstruction learning
    ax = plt.subplot(1, n, i)
    plt.imshow(intermediate_output[i].reshape((7, 7*8)).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

