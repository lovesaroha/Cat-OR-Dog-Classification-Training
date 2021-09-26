# Cat-OR-Dog-Classification-Training
Training a keras model to recognize real images of cats and dogs in order to classify an incoming image as one or the other. Download images from [mledu-datasets/cats_and_dogs_filtered.zip](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip). Download inception model weights for transfer learning from [mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5](https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5)

## Requirements
- Python 3 or higher.

## Packages

### Numpy
```bash
py -m pip install numpy
```

### Tensorflow
```bash
py -m pip install tensorflow
```

## Start Training
```bash
py convolutional_neural_network.py
```
```bash
py transfer_learning.py
```







