# PCA_image_compression
One of the useful applications of eigenvalues and eigenvectors is the dimensionality reduction algorithm called Principal Component Analyisis, or PCA.
Here we apply PCA on an image dataset to perform image compression.
We use a  portion of the Cat and dog face dataset from Kaggle. In particular, cat images.
To apply PCA we  begin by defining the covariance matrix. After that we compute the eigenvalues and eigenvectors of the covariance matrix. Each of these eigenvectors will be a principal component. For performing the dimensionality reduction, we take the 
principal components associated to the biggest eigenvalues, and transform the original data by projecting it onto the direction of these principal components (eigenvectors).
