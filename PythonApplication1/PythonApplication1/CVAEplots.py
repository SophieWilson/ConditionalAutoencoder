import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
## Plots
## All of the plotting functions for CVAE and VAE models in one place as they're clogging up the actual script, example of how to call below each function ##
# Functions in this script:
#   plot_clusters, digit_grid, reconstruction_plot, lossplot, construct_numvec, plot_latent_space, latent_space_traversal, plot_x_axis_change, plot_y_axis_change

def plot_clusters(encoder, x_test, y_test, labels, batch_size):
    ''' Display how the latent space clusters the digit classes '''
    x_test_encoded, _, _ = encoder.predict([x_test, y_test], batch_size=batch_size)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=labels)
    plt.colorbar()
    fig.suptitle('Input images plotted in latent space', fontsize=10)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
#plot_clusters(encoder, [x_test, y_test], plot_labels_test, batch_size)

def digit_grid(decoder, n=30, figsize=15):
    ''' Display a 2D grid of the digits in latent space, currently only works for VAE models '''
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
    
    fig = plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    fig.suptitle('2D grid of digits in latent space (VAE)', fontsize=10)
    plt.xlabel("z[0]")
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
#digit_grid(decoder)

# Plotting reconstruction vs actual
def reconstruction_plot(x_test, y_test, model, n=9):
    ''' Reconstruct model outputs vs actual digits
        n is number of digits, data is test (or train) image inputs, model is model'''
    prediction = model.predict([x_test, y_test])
    fig = plt.figure(figsize=(20, 4))
    fig.suptitle('Reconstructions vs input digits', fontsize=10)
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        quard = int(math.sqrt(x_test.shape[1])) # sqrt of x dimension
        plt.imshow(x_test[i].reshape(quard, quard))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        ax.set_title(y_test[i])
        plt.imshow(prediction[i].reshape(quard,quard))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

reconstruction_plot([x_test, y_test], cvae)

def plot_data(x_test, n=9):
    fig = plt.figure(figsize=(20, 2))
    fig.suptitle('Input digits', fontsize=15)
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        quard = int(math.sqrt(x_test.shape[1])) # sqrt of x dimension
        plt.imshow(x_test[i].reshape(quard, quard))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# loss plot
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

# Reconstructing specific digits
def construct_numvec(digit, z = None, n_z = 2, n_y = 10):
    ''' make number vector, its called in plot_latent_space, must change n_z and n_y values  here if you want to fix a plot '''
    out = np.zeros((1, n_z + n_y))
    out[:, digit + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)
    
# Plotting one label
#sample_3 = construct_numvec(3)
#plt.figure(figsize=(3, 3))
#plt.imshow(decoder.predict(sample_3).reshape(28,28), cmap = plt.cm.gray)
#plt.show()


def plot_latent_space(dig, max_z, sides, decoder):
    ''' Plotting latent space with respect to specific numbers 
        dig = 1
        sides = 8
        max_z = 1.5 '''
    img_it = 0 
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Latent space of specific digit', fontsize=10)
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        for j in range(0, sides):
            z2 = (((j / (sides-1)) * max_z)*2) - max_z
            z_ = [z1, z2]
            vec = construct_numvec(dig, z_)
            decoded = decoder.predict(vec)
            ax = plt.subplot(sides, sides, 1 + img_it)
            img_it +=1
            plt.imshow(decoded.reshape(28, 28), cmap = plt.cm.gray)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
    plt.show()


# Plotting one axis change
def latent_space_traversal(sides, max_z, decoder):
    ''' Plotting latent space axis change vs labels on the other axis. 
        Plotting y axis change '''
    img_it = 0
    fig = plt.figure(figsize = (20, 20))
    fig.suptitle('Latent space traversal', fontsize=10)
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, 0]
        for j in range(0, sides):
            vec = construct_numvec(j, z_)
            decoded = decoder.predict(vec)
            ax = plt.subplot(sides, sides, 1 + img_it)
            img_it +=1
            plt.imshow(decoded.reshape(28, 28), cmap = plt.cm.gray)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
    plt.show()
#latent_space_traversal(10, 1.5, decoder)

def plot_x_axis_change(dig, sides, max_z, decoder):
    ''' Plotting x axis change
        have been using 1, 10, 1.5 for inputs '''
    img_it = 0
    fig = plt.figure(figsize = (2, 20))
    fig.suptitle('Varying X', fontsize=10)
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        z_ = [0, z1] # This is where the x axis changes
        vec = construct_numvec(dig, z_)
        decoded = decoder.predict(vec)
        ax = plt.subplot(10, 1, 1 + img_it)
        img_it +=1
        plt.imshow(decoded.reshape(28, 28), cmap = plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
    plt.show()
#plot_x_axis_change(1, 10, 1.5, decoder)

def plot_y_axis_change(dig, sides, max_z, decoder):
    ''' Plotting y axis change 
        have been using 1, 10, 1.5 for inputs '''
    img_it = 0
    fig = plt.figure(figsize = (2, 20))
    fig.suptitle('Varying y', fontsize=10)
    vec = construct_numvec(dig, z_)
    decoded = decoder.predict(vec)
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, 0] # This is where the y axis changes
        ax = plt.subplot(10, 1, 1 + img_it)
        img_it +=1
        plt.imshow(decoded.reshape(28, 28), cmap = plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
    plt.show()
#plot_y_axis_change(1, 10, 1.5, decoder)

# Plotting digits as wrong labels 
# setting fake label
#label = np.repeat(7, 10000)
#label_fake = to_categorical(label, num_classes=10)
