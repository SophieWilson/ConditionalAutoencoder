
import tensorflow as tf
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

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
import nibabel as nib #reading MR images

filepath_df = pd.read_csv('Z:/PRONIA_data/Tables/pronia_full_niftis.csv')

mri_types =['wp0', # whole brain
            'wp1', # gm
            'wp2', # wm
            'mwp2', # modulated? wm
            'rp1', # gm mask
            'rp2'] # wm mask

niis = []
labels = []
num_subjs =200 # max 698
# Read in MRI image stacks
for i in range(num_subjs):
    row = filepath_df.iloc[i]
    nii_path = row['wp0']
    nii = nib.load(nii_path)
    nii = nii.get_fdata() 
    nii = nii[:, 35:53, :] #gives slices 36-53, the ones with the most vai
    labels.append(row['STUDYGROUP'])
    for j in range(nii.shape[1]):
        niis.append((nii[:,j,:]))
        
depth = len(nii[2])

# Prepare to crop images
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
# crop images in niis list
images = []
for i in range(len(niis)):
    img = niis[i]
    img = crop_center(img, 96, 96)
    images.append(img)
    
images = np.asarray(images) # shape num_subjs*121*121
# reshape to matrix in able to feed into network
images = images.reshape(-1, depth, 96, 96) # num_subjs,depth,121,121,

### min-max normalisation to rescale between 1 and 0 to improve accuracy
m = np.max(images)
mi = np.min(images) 
images = (images - mi) / (m - mi)



def structural_sim_data(data):
    ''' Returns a list of length depth, containing samples*samples-samples variances '''
    from skimage.metrics import structural_similarity as ssim
    print(len(data[0]))
    results = []   
    temp = []
    count = 0
    for k in range(int(len(data[0]))): # looping through slice depth
        for i in range(len(data)): # looping through patients in x 60
            for j in range(len(data)): # looping again to compare 60
                if (i == j):
                    continue
                else:
                    (score, diff) = ssim(data[i][k], data[j][k], full=True)
                    temp.append(score)
                    #print(len(temp))
        results.append(temp)
        temp = [] 
        print(count)
        count+=1
    return results

image_var = structural_sim_data(images)
import statistics
slice_var = []
counter = 0
for i in range(len(images[0])):
    counter += 1
    slice_var.append([i, statistics.mean(image_var[i])])
    if (counter%10 == 0):
        print(counter)

df = pd.DataFrame(slice_var)
df.to_csv('C:/Users/Mischa/Documents/Uni Masters/Diss project/Practise/MRI_cvae/50_101.csv')

#### Latent space variation analysis

def latent_space_var(label, num_recon, max_z, decoder, latent_dim):
    ''' this is messy and rubbish sorry, but it does work '''
    decoded_list = [[] for x in range(latent_dim)]
    for i in range(latent_dim): # looping through dimensions
        z_ = [0] * i
        for j in range(0, num_recon): # looping through number of images
            z1 = (((j / (num_recon-1)) * max_z)*2) - max_z
            z_.append(z1)
            vec = construct_numvec(label, z_)
            decoded = decoder.predict(vec) # 1, 16, 96, 96, 1
            decoded = decoded[0,:,:,:,0] # slice depth, 96, 96
            decoded_list[i].append(decoded)
            z_.pop()
    return decoded_list

lat_var = latent_space_var(0, 5, 4, decoder, 5)
# lat_var is of shape 30, 10, (16, 96, 96), 30 dimensions, 10 recons 

def structural_sim_latent(data):
    ''' Returns a list of length depth, containing samples*samples-samples variances, dont think i have to seperate the slices as it seems to cope with multiple'''
    from skimage.metrics import structural_similarity as ssim
    print(len(data[0]))
    results = []   
    temp = []
    count = 0
    for k in range(int(len(data))): # looping through latent dim 
        for i in range(len(data[0])): # looping through recons
            for j in range(len(data[0])): # looping through recon again 
                if (i == j):
                    continue
                else:
                    (score, diff) = ssim(data[k][i], data[k][j], full=True)
                    temp.append(score)
                    #print(len(temp))
        results.append(temp)
        temp = [] 
        print(count)
        count+=1
    return results

latent_ssim = structural_sim_latent(lat_var)

latent_slice_var = []
counter = 0
for i in range(len(latent_ssim)):
    counter += 1
    latent_slice_var.append([i, statistics.mean(latent_ssim[i])])
    #if (counter%10 == 0):
        #print(counter)

df = pd.DataFrame(slice_var)