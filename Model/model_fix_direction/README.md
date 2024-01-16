 fix_directions version of the model

This version is friendly to longer sequences and insensitive to the sequence length. For instance, you can train on RRIs with a length of 32, test on RRIs with a length of 59, and achieve better performance than if you were to train and test directly on RRIs with a length of 59. The main idea has shifted from projecting the sample vector onto the sphere through shifting and projection to mapping the radius to a weight between 0 and 1 for integration into the calculations. Afterwards, it is necessary to weight the gegenbauer coefficients for each frequency, which is relevant to high-dimensional zonal spherical convolution. For those interested in the three-dimensional case, please refer to:

- **Figure 4** of "Learning SO(3) Equivariant Representations with Spherical CNNs" 2018: 52-68.
- [Learning SO(3) Equivariant Representations with Spherical CNNs (PDF)](https://arxiv.org/pdf/1711.06721.pdf)
