# fix_directions version of the model

This version is friendly to longer sequences and insensitive to the sequence length. For instance, you can train on RRIs with a length of 32, test on RRIs with a length of 59, and achieve even better performance than if you were to train and test directly on RRIs with a length of 59. 
The main idea has changed from projecting the sample vector onto a sphere followed by a shift, to mapping the radius onto a range between 0 and 1. By separating the radial and angular components as features, we recognize that the radial component's informational value does not increase in higher dimensions. Therefore, we only require the average radius in two dimensions to quantify the degree of variability. Subsequently, we compute the dot product between a learnable fixed direction and all sample vectors on the Poincaré plot, assigning weights to the Gegenbauer coefficients for each frequency. This process corresponds to high-dimensional zonal spherical convolution. For those interested in delving into how the weighting of the Gegenbauer coefficients is associated with spherical convolution in three dimension case, please refer to:

- **Figure 4** of "Learning SO(3) Equivariant Representations with Spherical CNNs" 2018: 52-68.
- [Learning SO(3) Equivariant Representations with Spherical CNNs (PDF)](https://arxiv.org/pdf/1711.06721.pdf)
