########################################  Importing necessary libraries ########################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import utils


######################################## Defining needed functions #############################################

def center_data(Y):
    """
    Centering  the  original data
    Args:
         Y (ndarray): input data. Shape (n_observations x n_pixels)
    Outputs:
        X (ndarray): centered data
    """
    mean_vector = np.mean(Y, axis=0)
    mean_matrix = np.repeat(mean_vector,Y.shape[0])
    mean_matrix = np.reshape(mean_matrix,(Y.shape[0],Y.shape[1]),order='F')

    X = Y-mean_matrix
    return X


def get_cov_matrix(X):
    """ Calculate covariance matrix from centered data X
    Args:
        X (np.ndarray): centered data matrix
    Outputs:
        cov_matrix (np.ndarray): covariance matrix
    """

    cov_matrix = np.dot(np.transpose(X),X)
    cov_matrix = cov_matrix/(X.shape[0]-1)

    return cov_matrix


def perform_PCA(X, eigenvecs, k):
    """
    Perform dimensionality reduction with PCA
    Inputs:
        X (ndarray): original data matrix. Has dimensions (n_observations)x(n_variables)
        eigenvecs (ndarray): matrix of eigenvectors. Each column is one eigenvector. The k-th eigenvector
        is associated to the k-th eigenvalue
        k (int): number of principal components to use
    Returns:
        Xred
    """
    V = eigenvecs[:,0:k]
    Xred = np.dot(X,V)
    return Xred

def reconstruct_image(Xred, eigenvecs):
    X_reconstructed = Xred.dot(eigenvecs[:,:Xred.shape[1]].T)

    return X_reconstructed

######################################## Main Body #####################################################

# Loading the data
imgs = utils.load_images('./data/')

# Checking the data
height, width = imgs[0].shape
print(f'\nYour dataset has {len(imgs)} images of size {height}x{width} pixels\n')
plt.imshow(imgs[0], cmap='gray')

# Flatting images
imgs_flatten = np.array([im.reshape(-1) for im in imgs])
print(f'imgs_flatten shape: {imgs_flatten.shape}')

# Centering the data in order to get the covariance matrix
X = center_data(imgs_flatten)
plt.imshow(X[0].reshape(64,64), cmap='gray')

# Getting the  covariance matrix
cov_matrix = get_cov_matrix(X)
print(f'Covariance matrix shape: {cov_matrix.shape}')

# Computing the eigenvalues and eigenvectors (For computational efficiency,  only  computing the first biggest 55 eigenvalues and their corresponding eigenvectors)
scipy.random.seed(7) # The random seed is fixed  to  ensure the same eigenvectors are calculated each time
eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(cov_matrix, k=55)
print(f'Ten largest eigenvalues: \n{eigenvals[-10:]}')

# Performing perform_PCA
Xred2 = perform_PCA(X, eigenvecs,2)
print(f'Xred2 shape: {Xred2.shape}')

# Performing and reconstructing  images for different number of  principal components

Xred1 = perform_PCA(X, eigenvecs,1) # reduce dimensions to 1 component
Xred5 = perform_PCA(X, eigenvecs, 5) # reduce dimensions to 5 components
Xred10 = perform_PCA(X, eigenvecs, 10) # reduce dimensions to 10 components
Xred20 = perform_PCA(X, eigenvecs, 20) # reduce dimensions to 20 components
Xred30 = perform_PCA(X, eigenvecs, 30) # reduce dimensions to 30 components
Xrec1 = reconstruct_image(Xred1, eigenvecs) # reconstruct image from 1 component
Xrec5 = reconstruct_image(Xred5, eigenvecs) # reconstruct image from 5 components
Xrec10 = reconstruct_image(Xred10, eigenvecs) # reconstruct image from 10 components
Xrec20 = reconstruct_image(Xred20, eigenvecs) # reconstruct image from 20 components
Xrec30 = reconstruct_image(Xred30, eigenvecs) # reconstruct image from 30 components

fig, ax = plt.subplots(2,3, figsize=(22,15))
ax[0,0].imshow(imgs[21], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(Xrec1[21].reshape(height,width), cmap='gray')
ax[0,1].set_title('reconstructed from 1 components', size=20)
ax[0,2].imshow(Xrec5[21].reshape(height,width), cmap='gray')
ax[0,2].set_title('reconstructed from 5 components', size=20)
ax[1,0].imshow(Xrec10[21].reshape(height,width), cmap='gray')
ax[1,0].set_title('reconstructed from 10 components', size=20)
ax[1,1].imshow(Xrec20[21].reshape(height,width), cmap='gray')
ax[1,1].set_title('reconstructed from 20 components', size=20)
ax[1,2].imshow(Xrec30[21].reshape(height,width), cmap='gray')
ax[1,2].set_title('reconstructed from 30 components', size=20)
