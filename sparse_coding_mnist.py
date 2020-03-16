import bisect
import math
import struct as st

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from random import sample
from numpy.random import RandomState

from sklearn import decomposition


img_side_len = 28 # side length of each image, in pixels

rng = RandomState(1289) # random state seed

pct_var_explained = 98 # target percent variance explained by PC space

filename = {'train_images' : 'train-images-idx3-ubyte', 'test_images' : 't10k-images-idx3-ubyte'}

def plot_components(dictionary, cmap):
    ''' Plots dictionary components (row vectors of data) as 2d images
    
    Note: assumes size of each component equals # of pixels per image in the original MNIST data
    
    Parameters
    ----------
    data: (numpy array, [# of representations] x [# of pixels per image])
    
    '''
    
    max_rows = 12;
    
    n_cols = 20;
    n_rows = min(max_rows, math.ceil(np.size(dictionary,0)/n_cols))
    
    plt.figure()
    for i, img_rep in enumerate(dictionary[:n_cols*n_rows]):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img_rep.reshape((img_side_len,img_side_len)), cmap = cmap)
        plt.gca().set_axis_off()

    plt.show()

# -----------------------------------------------------------------------------
# Convert MNIST Dataset to Numpy arrays

def get_image_array(dataset_name):
    ''' Convert MNIST binary data to Numpy array of pixel data for each image
    
    Parameters
    ----------
    dataset_name: (str) name of MNIST dataset ('train_images' or 'test_images')
    
    Returns
    -------
    images_array: (numpy array, [# of images] x [# of pixels per image]) array of vectorized pixel data for each image
    
    '''
    
    with open('../'+filename[dataset_name],'rb') as images_file:
        images_file.seek(0)
        magic = st.unpack('>4B',images_file.read(4)) # magic number (0, 0, 8 = data type, 3 = # of dimensions)

        n_imgs = st.unpack('>I',images_file.read(4))[0] # number of images
        n_rows = st.unpack('>I',images_file.read(4))[0] # number of rows
        n_cols = st.unpack('>I',images_file.read(4))[0] # number of columns

        n_bytes_total = n_imgs*n_rows*n_cols
        
        images_array = np.zeros((n_imgs,n_rows,n_cols))
        images_array = 255 - \
            np.asarray(st.unpack('>'+'B'*n_bytes_total, images_file.read(n_bytes_total))).reshape((n_imgs,n_rows*n_cols))
    
    return images_array

# Convert MNIST training and test image data from binary files to numpy arrays
train_data = get_image_array('train_images')
test_data = get_image_array('test_images')

# -----------------------------------------------------------------------------
# Perform PCA on MNIST data

pca_model = decomposition.PCA(whiten = True, random_state = rng) # Initialize PCA object
train_data_zeroavg = train_data - np.mean(train_data, 0) # Zero-average training data

# Perform PCA with no restrictions on number of components, then evaluate variance explained by first n components
pca_model.fit(train_data_zeroavg)

# Plot variance explained as a function of number of PCs
cum_var_explained = np.cumsum(pca_model.explained_variance_ratio_)
n_pcs = bisect.bisect(cum_var_explained, pct_var_explained/100.)+1 # number of PCs (first n) to select

plt.figure()
plt.bar(np.array(range(len(cum_var_explained)))+1, cum_var_explained)
plt.axhline(y = pct_var_explained/100, color = 'r', alpha = .5)

plt.title('%u%% Variance Explained by First %u PCs' % (pct_var_explained, n_pcs))
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative Fraction of Variance Explained')

plt.show()

# Redo PCA with just enough PCs to hit the target percent variance explained
pca_model = decomposition.PCA(n_components = n_pcs, whiten = True, random_state = rng)
pca_model.fit(train_data_zeroavg)
train_data_pc = pca_model.transform(train_data_zeroavg) # Compressed PC-space representation of training data

# -----------------------------------------------------------------------------
# Train sparse coding model on MNIST data

dictionary_size =  n_pcs*2 # number of components in dictionary
alpha = .5 # sparseness parameter
n_iter = 500 # number of iterations

# Initialize MNIST sparse coding model
sparse_model = decomposition.MiniBatchDictionaryLearning(n_components = dictionary_size, alpha = alpha, fit_algorithm = 'cd', 
	n_iter = n_iter, random_state = rng)

# Fit model
sparse_model.fit(train_data_pc)

components = pca_model.inverse_transform(sparse_model.components_) # get components in pixel space

k_components = 50
inds = sample(range(np.size(components,0)),k_components)
components_subset = components[inds,:]

plot_components(components_subset, 'coolwarm')