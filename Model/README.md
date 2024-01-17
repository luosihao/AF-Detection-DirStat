- `My_e2e.py`: The main model.
- `Beran.py`: A faster version of `S_Pool.py` when `degree=10`.
- `S_pool.py`: Sobolev test statistics calculation.

  
- `model_fix_direction/`: This directory contains modified versions of the model with fixed directions. This model has a larger number of parameters, approximately 1.5 times more than the original model, but similar performance. Despite the increased parameter count, these models offer significantly improved speed, making them well-suited for processing longer RRI sequences of length n. With a complexity of O(n), these models can efficiently handle large datasets and longer RRI sequences. 


Please note that the training weights for these models are not provided. However, if you set the batch_size to 20000 or more and utilize a 2080Ti GPU, you can expect to train an epoch on the complete UVAF dataset in <1 minute. 
